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

static PTX_120: &str = include_str!("../resources/keryx-cuda-sm120.ptx");
static PTX_100: &str = include_str!("../resources/keryx-cuda-sm100.ptx");
static PTX_90: &str = include_str!("../resources/keryx-cuda-sm90.ptx");
static PTX_89: &str = include_str!("../resources/keryx-cuda-sm89.ptx");
static PTX_86: &str = include_str!("../resources/keryx-cuda-sm86.ptx");
static PTX_80: &str = include_str!("../resources/keryx-cuda-sm80.ptx");
static PTX_75: &str = include_str!("../resources/keryx-cuda-sm75.ptx");
static PTX_70: &str = include_str!("../resources/keryx-cuda-sm70.ptx");
static PTX_61: &str = include_str!("../resources/keryx-cuda-sm61.ptx");
static FATBIN_LEGACY: &[u8] = include_bytes!("../resources/keryx-legacy.fatbin");
static FATBIN_NEXTGEN: &[u8] = include_bytes!("../resources/keryx-nextgen.fatbin");
// sm_30 (Kepler) and sm_20 (Fermi) dropped: CUDA 12+ no longer compiles for
// these architectures, and they predate practical GPU mining anyway.

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

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        let u8matrix: Arc<[[u8; 64]; 64]> = Arc::new(matrix.map(|row| row.map(|v| v as u8)));
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap()).unwrap();
        hash_header_gpu.copy_from(hash_header).map_err(|e| e.to_string()).unwrap();

        let mut matrix_gpu = self._module.get_global::<[[u8; 64]; 64]>(&CString::new("matrix").unwrap()).unwrap();
        matrix_gpu.copy_from(&u8matrix).map_err(|e| e.to_string()).unwrap();

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap()).unwrap();
        target_gpu.copy_from(target).map_err(|e| e.to_string()).unwrap();
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        let func = &self.heavy_hash_kernel.func;
        let stream = &self.stream;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state.copy_from(&[rand::thread_rng().next_u64()]).unwrap();
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        // Clear the winner slot host-side: the kernel must not clear it while its own blocks
        // are already publishing into it.
        self.final_nonce_buff.copy_from(&[0]).unwrap();
        self.start_event.record(stream).unwrap();
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
                    self.final_nonce_buff.as_device_ptr()
                )
            )
            .unwrap(); // We see errors in sync
        }
        self.stop_event.record(stream).unwrap();
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
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        self.final_nonce_buff.copy_to(nonces)?;
        Ok(())
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
        let mut _module: Option<Arc<Module>> = None;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);

        let driver_api = CudaApiVersion::get().ok();
        if let Some(ver) = driver_api {
            info!(
                "GPU #{} CUDA driver API version {}.{}",
                device_id,
                ver.major(),
                ver.minor()
            );
        }

        let mut selected_module = String::new();
        let mut selection_path = String::new();

        let load_ptx = |ptx, label: &str| {
            Module::from_ptx(ptx, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Failed to load {} PTX: {}", label, e);
                e
            })
        };

        let load_fatbin = |fatbin: &[u8], label: &str| {
            Module::from_fatbin(fatbin, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Failed to load {} fatbin: {}", label, e);
                e
            })
        };

        let is_nextgen_cc = major > 8 || (major == 8 && minor >= 9);
        let fatbin_loaded = if is_nextgen_cc {
            match load_fatbin(FATBIN_NEXTGEN, "nextgen") {
                Ok(m) => {
                    info!("GPU #{} using nextgen fatbin", device_id);
                    _module = Some(Arc::new(m));
                    selected_module = "fatbin:nextgen".to_string();
                    selection_path = "nextgen-fatbin".to_string();
                    true
                }
                Err(_) => match load_fatbin(FATBIN_LEGACY, "legacy (fallback)") {
                    Ok(m) => {
                        info!("GPU #{} using legacy fatbin fallback", device_id);
                        _module = Some(Arc::new(m));
                        selected_module = "fatbin:legacy".to_string();
                        selection_path = "nextgen-fatbin->legacy-fatbin".to_string();
                        true
                    }
                    Err(_) => false,
                },
            }
        } else {
            match load_fatbin(FATBIN_LEGACY, "legacy") {
                Ok(m) => {
                    info!("GPU #{} using legacy fatbin", device_id);
                    _module = Some(Arc::new(m));
                    selected_module = "fatbin:legacy".to_string();
                    selection_path = "legacy-fatbin".to_string();
                    true
                }
                Err(_) => false,
            }
        };

        if !fatbin_loaded {
        selection_path = if is_nextgen_cc {
            "fatbin-miss(nextgen+legacy)->ptx".to_string()
        } else {
            "fatbin-miss(legacy)->ptx".to_string()
        };
        // For sm_89 (Ada/RTX 40) and sm_100 (Blackwell/RTX 50), the PTX was compiled with
        // CUDA 13.2 (PTX ISA 9.2) which requires driver >= 570. If the driver is older, we
        // fall back to sm_86 (CUDA 12.0 / PTX 8.0, driver >= 520) which runs on all these
        // architectures via NVIDIA's backward-compatible PTX JIT.
        if major >= 12 {
            // sm_120 (consumer Blackwell — GeForce RTX 5090 etc.). NVIDIA splits
            // sm_100 (datacenter Blackwell — H100/B100) and sm_120 (consumer) as
            // separate compute architectures, so the sm_100 PTX errors out with
            // `unknown error` on a 5090 and the card used to land on JIT'd sm_86
            // at roughly half the native throughput. The sm_120 PTX is ISA 9.0
            // (driver >= 580); on older drivers we fall through to sm_86.
            _module = Some(Arc::new(match load_ptx(PTX_120, "sm_120") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_120 PTX", device_id);
                    selected_module = "ptx:sm_120".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} sm_120 PTX failed; trying sm_100 then sm_86 fallback (update driver to 580+)", device_id);
                    match load_ptx(PTX_100, "sm_100") {
                        Ok(m) => {
                            selected_module = "ptx:sm_100".to_string();
                            selection_path.push_str("->sm_120-fail->sm_100");
                            m
                        }
                        Err(_) => {
                            selected_module = "ptx:sm_86".to_string();
                            selection_path.push_str("->sm_120-fail->sm_100-fail->sm_86");
                            load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                        }
                    }
                }
            }));
        } else if major >= 10 {
            // sm_100+ (datacenter Blackwell — H100 / B100 / GH100)
            _module = Some(Arc::new(match load_ptx(PTX_100, "sm_100") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_100 PTX", device_id);
                    selected_module = "ptx:sm_100".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Blackwell optimisation)", device_id);
                    selected_module = "ptx:sm_86".to_string();
                    selection_path.push_str("->sm_100-fail->sm_86");
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            }));
        } else if major == 9 {
            // sm_90 (Hopper — H100 / H200 / GH200). Routed here before the Ada branch so it
            // loads the native sm_90 PTX instead of JIT'ing the sm_89 build. Falls back to
            // sm_89 then sm_86 if the driver is too old for the sm_90 PTX ISA.
            _module = Some(Arc::new(match load_ptx(PTX_90, "sm_90") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_90 PTX", device_id);
                    selected_module = "ptx:sm_90".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} sm_90 PTX failed; trying sm_89 then sm_86 fallback", device_id);
                    match load_ptx(PTX_89, "sm_89") {
                        Ok(m) => {
                            selected_module = "ptx:sm_89".to_string();
                            selection_path.push_str("->sm_90-fail->sm_89");
                            m
                        }
                        Err(_) => {
                            selected_module = "ptx:sm_86".to_string();
                            selection_path.push_str("->sm_90-fail->sm_89-fail->sm_86");
                            load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                        }
                    }
                }
            }));
        } else if major == 8 && minor >= 9 {
            // sm_89 (RTX 40 / Ada Lovelace)
            _module = Some(Arc::new(match load_ptx(PTX_89, "sm_89") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_89 PTX", device_id);
                    selected_module = "ptx:sm_89".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Ada Lovelace optimisation)", device_id);
                    selected_module = "ptx:sm_86".to_string();
                    selection_path.push_str("->sm_89-fail->sm_86");
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            }));
        } else if major == 8 && minor >= 6 {
            // sm_86 (RTX 30 / Ampere)
            selected_module = "ptx:sm_86".to_string();
            _module = Some(Arc::new(load_ptx(PTX_86, "sm_86")?));
        } else if major == 8 {
            // sm_80 (A100 / CMP 170HX, data-center Ampere). Reaching here means minor < 6
            // (sm_86+ and sm_89+ are caught above). The sm_86 PTX would NOT load on sm_80
            // because a PTX .target is a *minimum* compute capability, so we ship a native
            // sm_80 PTX. If the driver is too old for its PTX ISA, fall back to sm_75, which
            // runs on sm_80 and up via the backward-compatible PTX JIT.
            _module = Some(Arc::new(match load_ptx(PTX_80, "sm_80") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_80 PTX", device_id);
                    selected_module = "ptx:sm_80".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_75 PTX (update driver for native sm_80)", device_id);
                    selected_module = "ptx:sm_75".to_string();
                    selection_path.push_str("->sm_80-fail->sm_75");
                    load_ptx(PTX_75, "sm_75 (fallback)").map_err(|_| e)?
                }
            }));
        } else if major > 7 || (major == 7 && minor >= 5) {
            // sm_75 (RTX 20 / Turing)
            selected_module = "ptx:sm_75".to_string();
            _module = Some(Arc::new(Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?));
        } else if major == 7 {
            // sm_70/sm_72 (Volta)
            _module = Some(Arc::new(match load_ptx(PTX_70, "sm_70") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_70 PTX", device_id);
                    selected_module = "ptx:sm_70".to_string();
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_61 PTX (update driver for native sm_70)", device_id);
                    selected_module = "ptx:sm_61".to_string();
                    selection_path.push_str("->sm_70-fail->sm_61");
                    load_ptx(PTX_61, "sm_61 (fallback)").map_err(|_| e)?
                }
            }));
        } else if major > 6 || (major == 6 && minor >= 1) {
            // sm_61 (GTX 10 / Pascal)
            selected_module = "ptx:sm_61".to_string();
            _module = Some(Arc::new(Module::from_ptx(PTX_61, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?));
        } else {
            return Err(format!(
                "CUDA compute {}.{} not supported. Keryx requires sm_61 (GTX 10xx) or newer.",
                major, minor
            )
            .into());
        }
        }

        let (driver_major, driver_minor) = match driver_api {
            Some(v) => (v.major().to_string(), v.minor().to_string()),
            None => ("unknown".to_string(), "unknown".to_string()),
        };
        info!(
            "GPU #{} selection summary | cc={}.{} | driver_api={}.{} | module={} | path={}",
            device_id,
            major,
            minor,
            driver_major,
            driver_minor,
            selected_module,
            selection_path
        );

        let _module = _module.expect("CUDA module must be selected before use");

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
            heavy_hash_kernel,
            random,
        })
    }
}
