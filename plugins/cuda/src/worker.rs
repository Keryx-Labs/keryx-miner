use crate::{Error, NonceGenEnum};
use cust::context::CurrentContext;
use cust::device::DeviceAttribute;
use cust::function::Function;
use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::CudaApiVersion;
use keryx_miner::xoshiro256starstar::Xoshiro256StarStar;
use keryx_miner::Worker;
use log::{error, info};
use rand::{Fill, RngCore};
use std::ffi::CString;
use std::sync::{Arc, Weak};

static BPS: f32 = 1.;

static PTX_90: &str = include_str!("../resources/keryx-cuda-sm90.ptx");
static PTX_89: &str = include_str!("../resources/keryx-cuda-sm89.ptx");
static PTX_86: &str = include_str!("../resources/keryx-cuda-sm86.ptx");
static PTX_80: &str = include_str!("../resources/keryx-cuda-sm80.ptx");
static PTX_75: &str = include_str!("../resources/keryx-cuda-sm75.ptx");
static PTX_70: &str = include_str!("../resources/keryx-cuda-sm70.ptx");
static PTX_61: &str = include_str!("../resources/keryx-cuda-sm61.ptx");
// sm_30 (Kepler) and sm_20 (Fermi) dropped: CUDA 12+ no longer compiles for
// these architectures, and they predate practical GPU mining anyway.

fn ptx_resource(major: i32, minor: i32) -> Option<(&'static str, &'static str)> {
    if major >= 9 {
        // CUDA PTX is forward-compatible: sm_90 PTX is the pinned CUDA 12.2 fallback for newer GPUs.
        Some((PTX_90, "sm_90"))
    } else if major == 8 && minor >= 9 {
        Some((PTX_89, "sm_89"))
    } else if major == 8 && minor >= 6 {
        Some((PTX_86, "sm_86"))
    } else if major == 8 {
        Some((PTX_80, "sm_80"))
    } else if major > 7 || (major == 7 && minor >= 5) {
        Some((PTX_75, "sm_75"))
    } else if major == 7 {
        Some((PTX_70, "sm_70"))
    } else if major > 6 || (major == 6 && minor >= 1) {
        Some((PTX_61, "sm_61"))
    } else {
        None
    }
}

fn ptx_has_winner_abi(ptx: &str) -> bool {
    let Some(entry) = ptx.split(".entry heavy_hash(").nth(1) else {
        return false;
    };
    let Some(parameters) = entry.split(')').next() else {
        return false;
    };
    let types = parameters
        .split(',')
        .filter_map(|parameter| {
            let tokens = parameter.split_whitespace().collect::<Vec<_>>();
            let param = tokens.iter().position(|token| *token == ".param")?;
            tokens[param + 1..].iter().copied().find(|token| {
                token.starts_with(".u") || token.starts_with(".s") || token.starts_with(".b") || token.starts_with(".f")
            })
        })
        .collect::<Vec<_>>();
    types == [".u64", ".u64", ".u64", ".u8", ".u64", ".u64", ".u64"]
}

pub struct Kernel<'kernel> {
    func: Arc<Function<'kernel>>,
    block_size: u32,
    grid_size: u32,
}

impl<'kernel> Kernel<'kernel> {
    pub fn new(module: Weak<Module>, name: &'kernel str) -> Result<Kernel<'kernel>, Error> {
        let func = Arc::new(unsafe {
            module.as_ptr().as_ref().unwrap().get_function(name).map_err(|e| {
                error!("Error loading function: {}", e);
                e
            })?
        });
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

        let device = CurrentContext::get_device()?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let grid_size = sm_count * func.max_active_blocks_per_multiprocessor(block_size.into(), 0)?;

        Ok(Self { func, block_size, grid_size })
    }

    pub fn get_workload(&self) -> u32 {
        self.block_size * self.grid_size
    }

    pub fn set_workload(&mut self, workload: u32) {
        self.grid_size = (workload + self.block_size - 1) / self.block_size
    }
}

pub struct CudaGPUWorker<'gpu> {
    // NOTE: The order is important! context must be closed last
    heavy_hash_kernel: Kernel<'gpu>,
    stream: Stream,
    start_event: Event,
    stop_event: Event,
    _module: Arc<Module>,

    rand_state: DeviceBuffer<u64>,
    final_nonce_buff: DeviceBuffer<u64>,
    winner_found_buff: DeviceBuffer<u32>,

    device_id: u32,
    pub workload: usize,
    _context: Context,

    random: NonceGenEnum,
}

impl<'gpu> Worker for CudaGPUWorker<'gpu> {
    fn id(&self) -> String {
        let device = CurrentContext::get_device().unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn load_block_constants(
        &mut self,
        hash_header: &[u8; 72],
        matrix: &[[u16; 64]; 64],
        target: &[u64; 4],
    ) -> Result<(), Error> {
        let u8matrix: Arc<[[u8; 64]; 64]> = Arc::new(matrix.map(|row| row.map(|v| v as u8)));
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap())?;
        hash_header_gpu.copy_from(hash_header)?;

        let mut matrix_gpu = self._module.get_global::<[[u8; 64]; 64]>(&CString::new("matrix").unwrap())?;
        matrix_gpu.copy_from(&u8matrix)?;

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap())?;
        target_gpu.copy_from(target)?;
        Ok(())
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) -> Result<(), Error> {
        let func = &self.heavy_hash_kernel.func;
        let stream = &self.stream;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state.copy_from(&[rand::thread_rng().next_u64()])?;
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        self.final_nonce_buff.copy_from(&[0])?;
        self.winner_found_buff.copy_from(&[0])?;
        self.start_event.record(stream)?;
        unsafe {
            launch!(
                func<<<
                    self.heavy_hash_kernel.grid_size, self.heavy_hash_kernel.block_size,
                    0, stream
                >>>(
                    nonce_mask, nonce_fixed,
                    self.workload,
                    random,
                    self.rand_state.as_device_ptr(),
                    self.final_nonce_buff.as_device_ptr(),
                    self.winner_found_buff.as_device_ptr()
                )
            )?;
        }
        if let Err(error) = self.stop_event.record(stream) {
            // The kernel may already be using the shared winner buffers. Drain the stream before
            // returning so recovery cannot reset them underneath an in-flight launch.
            stream.synchronize()?;
            return Err(Box::new(error));
        }
        Ok(())
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        //self.stream.synchronize()?;
        self.stop_event.synchronize()?;
        if self.stop_event.elapsed_time_f32(&self.start_event)? > 1000. / BPS {
            return Err("Cuda takes longer then block rate. Please reduce your workload.".into());
        }
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    #[inline(always)]
    fn read_winner(&mut self) -> Result<Option<u64>, Error> {
        let mut found = [0u32; 1];
        self.winner_found_buff.copy_to(&mut found)?;
        if found[0] == 0 {
            return Ok(None);
        }
        let mut nonce = [0u64; 1];
        self.final_nonce_buff.copy_to(&mut nonce)?;
        Ok(Some(nonce[0]))
    }
}

impl<'gpu> CudaGPUWorker<'gpu> {
    pub fn new(
        device_id: u32,
        workload: f32,
        is_absolute: bool,
        blocking_sync: bool,
        random: NonceGenEnum,
    ) -> Result<Self, Error> {
        info!("Starting a CUDA worker");
        let sync_flag = match blocking_sync {
            true => ContextFlags::SCHED_BLOCKING_SYNC,
            false => ContextFlags::SCHED_AUTO,
        };
        let device = Device::get_device(device_id).unwrap();
        let _context = Context::new(device)?;
        _context.set_flags(sync_flag)?;

        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);

        let driver_api = CudaApiVersion::get().ok();
        if let Some(ver) = driver_api {
            info!("GPU #{} CUDA driver API version {}.{}", device_id, ver.major(), ver.minor());
        }

        let (ptx, selected_module) = ptx_resource(major, minor).ok_or_else(|| {
            format!("CUDA compute {}.{} not supported. Keryx requires sm_61 (GTX 10xx) or newer.", major, minor)
        })?;
        if !ptx_has_winner_abi(ptx) {
            return Err(format!(
                "embedded {selected_module} PTX uses the obsolete heavy_hash ABI; regenerate CUDA resources with CUDA 12.2"
            )
            .into());
        }

        let _module = Arc::new(Module::from_ptx(ptx, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
            error!("Failed to load {} PTX (driver too old?): {}", selected_module, e);
            e
        })?);

        let (driver_major, driver_minor) = match driver_api {
            Some(v) => (v.major().to_string(), v.minor().to_string()),
            None => ("unknown".to_string(), "unknown".to_string()),
        };
        info!(
            "GPU #{} selection summary | cc={}.{} | driver_api={}.{} | module=ptx:{}",
            device_id, major, minor, driver_major, driver_minor, selected_module
        );

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut heavy_hash_kernel = Kernel::new(Arc::downgrade(&_module), "heavy_hash")?;

        let mut chosen_workload = 0u32;
        if is_absolute {
            chosen_workload = 1;
        } else {
            let cur_workload = heavy_hash_kernel.get_workload();
            if chosen_workload == 0 || chosen_workload < cur_workload {
                chosen_workload = cur_workload;
            }
        }
        chosen_workload = (chosen_workload as f32 * workload) as u32;
        info!("GPU #{} Chosen workload: {}", device_id, chosen_workload);
        heavy_hash_kernel.set_workload(chosen_workload);

        let final_nonce_buff = vec![0u64; 1].as_slice().as_dbuf()?;
        let winner_found_buff = vec![0u32; 1].as_slice().as_dbuf()?;

        let rand_state: DeviceBuffer<u64> = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(4 * (chosen_workload as usize)).unwrap();
                info!("GPU #{} is generating initial seed. This may take some time.", device_id);
                let mut seed = [1u64; 4];
                seed.try_fill(&mut rand::thread_rng())?;
                buffer.copy_from(
                    Xoshiro256StarStar::new(&seed)
                        .iter_jump_state()
                        .take(chosen_workload as usize)
                        .flatten()
                        .collect::<Vec<u64>>()
                        .as_slice(),
                )?;
                info!("GPU #{} initialized", device_id);
                buffer
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(1).unwrap();
                let seed = rand::thread_rng().next_u64();
                buffer.copy_from(&[seed])?;
                buffer
            }
        };
        Ok(Self {
            device_id,
            _context,
            _module,
            start_event: Event::new(EventFlags::DEFAULT)?,
            stop_event: Event::new(EventFlags::DEFAULT)?,
            workload: chosen_workload as usize,
            stream,
            rand_state,
            final_nonce_buff,
            winner_found_buff,
            heavy_hash_kernel,
            random,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_cuda_12_ptx_resources_by_compute_capability() {
        assert_eq!(ptx_resource(6, 0), None);
        assert_eq!(ptx_resource(6, 1).map(|(_, label)| label), Some("sm_61"));
        assert_eq!(ptx_resource(7, 0).map(|(_, label)| label), Some("sm_70"));
        assert_eq!(ptx_resource(7, 5).map(|(_, label)| label), Some("sm_75"));
        assert_eq!(ptx_resource(8, 0).map(|(_, label)| label), Some("sm_80"));
        assert_eq!(ptx_resource(8, 6).map(|(_, label)| label), Some("sm_86"));
        assert_eq!(ptx_resource(8, 9).map(|(_, label)| label), Some("sm_89"));
        assert_eq!(ptx_resource(9, 0).map(|(_, label)| label), Some("sm_90"));
    }

    #[test]
    fn newer_devices_use_forward_compatible_sm90_ptx() {
        assert_eq!(ptx_resource(10, 0).map(|(_, label)| label), Some("sm_90"));
        assert_eq!(ptx_resource(12, 0).map(|(_, label)| label), Some("sm_90"));
    }

    #[test]
    fn rejects_obsolete_heavy_hash_ptx_abi() {
        let old = ".entry heavy_hash(\n.param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u8 p3,\n.param .u64 p4,\n.param .u64 p5\n)";
        let current = ".entry heavy_hash(\n.param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u8 p3,\n.param .u64 p4,\n.param .u64 p5,\n.param .u64 p6\n)";

        assert!(!ptx_has_winner_abi(old));
        assert!(ptx_has_winner_abi(current));
    }

    #[test]
    fn rejects_reordered_or_wrong_width_heavy_hash_parameters() {
        let reordered = ".entry heavy_hash(\n.param .u64 p0,\n.param .u64 p1,\n.param .u8 p2,\n.param .u64 p3,\n.param .u64 p4,\n.param .u64 p5,\n.param .u64 p6\n)";
        let wrong_winner = ".entry heavy_hash(\n.param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u8 p3,\n.param .u64 p4,\n.param .u64 p5,\n.param .u32 p6\n)";

        assert!(!ptx_has_winner_abi(reordered));
        assert!(!ptx_has_winner_abi(wrong_winner));
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn publishes_nonce_zero_from_multiple_blocks() {
        cust::init(cust::CudaFlags::empty()).expect("initialize CUDA");
        let mut worker = CudaGPUWorker::new(0, 4096.0, true, false, NonceGenEnum::Lean).expect("create CUDA worker");
        worker.load_block_constants(&[0; 72], &[[0; 64]; 64], &[u64::MAX; 4]).expect("load CUDA constants");

        for _ in 0..100 {
            worker.calculate_hash(None, 0, 0).expect("launch CUDA kernel");
            worker.sync().expect("synchronize CUDA kernel");
            assert_eq!(worker.read_winner().unwrap(), Some(0));
        }
    }
}
