use crate::cli::NonceGenEnum;
use crate::Error;
use include_dir::{include_dir, Dir};
use keryx_miner::xoshiro256starstar::Xoshiro256StarStar;
use keryx_miner::Worker;
use log::{info, warn};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, ClMem, CL_MAP_WRITE, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::platform::Platform;
use opencl3::program::{Program, CL_FINITE_MATH_ONLY, CL_MAD_ENABLE, CL_STD_2_0};
use opencl3::types::{cl_uchar, cl_uint, cl_ulong, CL_BLOCKING};
use rand::{thread_rng, Fill, RngCore};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

static BINARY_DIR: Dir = include_dir!("./plugins/opencl/resources/bin/");
static PROGRAM_SOURCE: &str = include_str!("../resources/keryx-opencl.cl");
const HEAVY_HASH_ARGUMENTS: u32 = 11;

fn opencl_error(error: impl std::fmt::Display) -> Error {
    std::io::Error::new(std::io::ErrorKind::Other, error.to_string()).into()
}

/// Capability-driven default for the `--opencl-workload` ratio when the user
/// didn't set it. The ratio is multiplied by (max_work_group_size * compute_units)
/// downstream, so a bigger card needs a *smaller* ratio for the same nonce count.
/// Tuned on the live pool (0 rejects): the old flat 512 under-saturated both the
/// 60-CU gfx906 MI50 (best at 2048 → +24-33%) and the 16-CU gfx1102 7600 XT
/// (best at 4096 → +10%). Grouped by family for the archs not directly measured;
/// unknown archs keep the conservative legacy 512.
fn default_workload_scale(arch: &str) -> f32 {
    match arch {
        // CDNA / GCN5 MI-series — high CU count, wave64; saturates at a lower ratio.
        "gfx906" | "gfx908" | "gfx90a" | "gfx940" | "gfx941" | "gfx942" => 2048.,
        // RDNA 3/4 — fewer CUs; wants a higher ratio to saturate.
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1200" | "gfx1201" => 4096.,
        // Untested / older archs: keep the conservative legacy default.
        _ => 512.,
    }
}

pub struct OpenCLGPUWorker {
    context: Arc<Context>,
    random: NonceGenEnum,
    local_size: usize,
    workload: usize,

    heavy_hash: Kernel,

    queue: CommandQueue,

    random_state: Buffer<cl_ulong>,
    final_nonce: Buffer<cl_ulong>,
    winner_found: Buffer<cl_uint>,
    final_hash: Buffer<[cl_ulong; 4]>,

    hash_header: Buffer<cl_uchar>,
    matrix: Buffer<cl_uchar>,
    target: Buffer<cl_ulong>,

    experimental_amd: bool,
}

impl Worker for OpenCLGPUWorker {
    fn id(&self) -> String {
        let device = Device::new(self.context.default_device());
        device.name().unwrap()
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) -> Result<(), Error> {
        let cl_uchar_matrix = match self.experimental_amd {
            true => matrix
                .iter()
                .flat_map(|row| row.chunks(2).map(|v| ((v[0] << 4) | v[1]) as cl_uchar))
                .collect::<Vec<cl_uchar>>(),
            false => matrix.iter().flat_map(|row| row.map(|v| v as cl_uchar)).collect::<Vec<cl_uchar>>(),
        };
        self.queue
            .enqueue_write_buffer(&mut self.hash_header, CL_BLOCKING, 0, hash_header, &[])
            .map_err(opencl_error)?
            .wait()
            .map_err(opencl_error)?;
        self.queue
            .enqueue_write_buffer(&mut self.matrix, CL_BLOCKING, 0, cl_uchar_matrix.as_slice(), &[])
            .map_err(opencl_error)?
            .wait()
            .map_err(opencl_error)?;
        self.queue
            .enqueue_write_buffer(&mut self.target, CL_BLOCKING, 0, target, &[])
            .map_err(opencl_error)?
            .wait()
            .map_err(opencl_error)?;
        Ok(())
    }

    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) -> Result<(), Error> {
        self.queue
            .enqueue_write_buffer(&mut self.winner_found, CL_BLOCKING, 0, &[0], &[])
            .map_err(opencl_error)?
            .wait()
            .map_err(opencl_error)?;
        if self.random == NonceGenEnum::Lean {
            self.queue
                .enqueue_write_buffer(&mut self.random_state, CL_BLOCKING, 0, &[thread_rng().next_u64()], &[])
                .map_err(opencl_error)?
                .wait()
                .map_err(opencl_error)?;
        }
        let random_type: cl_uchar = match self.random {
            NonceGenEnum::Lean => 0,
            NonceGenEnum::Xoshiro => 1,
        };
        let kernel_event = ExecuteKernel::new(&self.heavy_hash)
            .set_arg(&(self.local_size as u64))
            .set_arg(&nonce_mask)
            .set_arg(&nonce_fixed)
            .set_arg(&self.hash_header)
            .set_arg(&self.matrix)
            .set_arg(&self.target)
            .set_arg(&random_type)
            .set_arg(&self.random_state)
            .set_arg(&self.winner_found)
            .set_arg(&self.final_nonce)
            .set_arg(&self.final_hash)
            .set_global_work_size(self.workload)
            .enqueue_nd_range(&self.queue)
            .map_err(opencl_error)?;

        kernel_event.wait().map_err(opencl_error)?;

        /*let mut nonces = [0u64; 1];
        let mut hash = [[0u64; 4]];
        self.queue.enqueue_read_buffer(&self.final_nonce, CL_BLOCKING, 0, &mut nonces, &[]).map_err(|e| e.to_string()).unwrap();
        self.queue.enqueue_read_buffer(&self.final_hash, CL_BLOCKING, 0, &mut hash, &[]).map_err(|e| e.to_string()).unwrap();
        log::info!("Hash from kernel: {:?}", hash);*/
        /*for event in &self.events{
            release_event(*event).unwrap();
        }
        let event = kernel_event.get();
        self.events = vec!(event);
        retain_event(event);*/
        Ok(())
    }

    fn sync(&self) -> Result<(), Error> {
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload as usize
    }

    fn read_winner(&mut self) -> Result<Option<u64>, Error> {
        let mut found = [0u32; 1];
        self.queue
            .enqueue_read_buffer(&self.winner_found, CL_BLOCKING, 0, &mut found, &[])
            .map_err(|e| e.to_string())?;
        if found[0] == 0 {
            return Ok(None);
        }
        let mut nonce = [0u64; 1];
        self.queue
            .enqueue_read_buffer(&self.final_nonce, CL_BLOCKING, 0, &mut nonce, &[])
            .map_err(|e| e.to_string())?;
        Ok(Some(nonce[0]))
    }
}

impl OpenCLGPUWorker {
    pub fn new(
        device: Device,
        workload: f32,
        is_absolute: bool,
        experimental_amd: bool,
        use_binary: bool,
        random: &NonceGenEnum,
    ) -> Result<Self, Error> {
        let name =
            device.board_name_amd().unwrap_or_else(|_| device.name().unwrap_or_else(|_| "Unknown Device".into()));
        info!("{}: Using OpenCL", name);
        let version = device.version().unwrap_or_else(|_| "unkown version".into());
        info!(
            "{}: Device supports {} with extensions: {}",
            name,
            version,
            device.extensions().unwrap_or_else(|_| "NA".into())
        );

        // Normalised arch string ("gfx906", "gfx1102", …) — ROCm reports it as
        // e.g. "gfx906:sramecc+:xnack-", so strip the feature suffix. Used both for
        // the capability-driven workload default and the v_dot8 gate below.
        let arch = device
            .name()
            .unwrap_or_else(|_| "Unknown".into())
            .split(':')
            .next()
            .unwrap_or("")
            .to_lowercase();

        let local_size = device.max_work_group_size().map_err(|e| e.to_string())?;
        let chosen_workload = match is_absolute {
            true => workload as usize,
            false => {
                let max_work_group_size =
                    (local_size * (device.max_compute_units().map_err(|e| e.to_string())? as usize)) as f32;
                // workload <= 0 is the AUTO sentinel (no --opencl-workload given):
                // pick a per-arch ratio that saturates the card. The ratio is scaled
                // by CU count here, so larger cards need a smaller ratio. Measured on
                // the live pool, 0 rejects: gfx906 MI50 316→~393 MH/s @2048, gfx1102
                // 7600 XT 301→~332 MH/s @4096 vs the old flat 512. An explicit
                // --opencl-workload always overrides this.
                let ratio = if workload > 0. { workload } else { default_workload_scale(&arch) };
                (ratio * max_work_group_size) as usize
            }
        };
        info!("{}: Chosen workload is {}", name, chosen_workload);
        let context =
            Arc::new(Context::from_device(&device).unwrap_or_else(|_| panic!("{}::Context::from_device failed", name)));
        let context_ref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };

        // v_dot8_u32_u4 (packed 4-bit dot product, 8 MACs/instr) is the fast path for
        // the 64x64 matrix multiply — half the dot instructions of v_dot4_u32_u8.
        // Every arch implementing the DLOPS dot instructions should use it, so make
        // it capability-driven (the default; no flag needed). The legacy
        // `--experimental-amd` flag still forces it on for capable archs not in the
        // default list below. (`arch` is computed above for the workload default.)
        // DLOPS-capable archs that ship WITHOUT a precompiled v_dot4 binary, so the JIT
        // path can emit the packed v_dot8 kernel with no .bin/host-layout mismatch:
        // CDNA / GCN5 MI-series (gfx9xx) and RDNA 3/4 (gfx11xx/12xx). The RDNA 1.5/2
        // parts (gfx1011/1012/1030-1034) also support v_dot8 but ship a v_dot4 .bin,
        // so they stay on v_dot4 until those binaries are regenerated.
        let vdot8_default = matches!(
            arch.as_str(),
            "gfx906" | "gfx908" | "gfx90a" | "gfx940" | "gfx941" | "gfx942"
                | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1200" | "gfx1201"
        );
        // Also v_dot8-capable, but only opt-in (they have a shipped v_dot4 .bin).
        let vdot8_capable = vdot8_default
            || matches!(arch.as_str(), "gfx1011" | "gfx1012" | "gfx1030" | "gfx1031" | "gfx1032" | "gfx1034");
        let experimental_amd_use = vdot8_default || (experimental_amd && vdot8_capable);

        let options = match experimental_amd_use {
            // true => "-D __FORCE_AMD_V_DOT4_U32_U8__=1 ",
            true => "-D __FORCE_AMD_V_DOT8_U32_U4__=1 ",
            false => "",
        };

        let mut program = match use_binary {
            true => {
                let mut device_name = device.name().unwrap_or_else(|_| "Unknown".into()).to_lowercase();
                if device_name.contains(':') {
                    device_name = device_name.split_once(':').expect("We checked for `:`").0.to_string();
                }
                info!("{}: Looking for binary for {}", name, device_name);
                match BINARY_DIR.get_file(format!("{}_keryx-opencl.bin", device_name)) {
                    Some(binary) => {
                        Program::create_and_build_from_binary(&context, &[binary.contents()], "").unwrap_or_else(|e|{
                        //Program::create_and_build_from_binary(&context, &[include_bytes!("../resources/keryx-opencl-linked.bc")], "").unwrap_or_else(|e|{
                            warn!("{}::Program::create_and_build_from_source failed: {}. Reverting to compiling from source", name, e);
                            from_source(&context, &device, options).unwrap_or_else(|e| panic!("{}::Program::create_and_build_from_binary failed: {}", name, e))
                        })
                    }
                    None => {
                        warn!("Binary file not found for {}. Reverting to compiling from source.", device_name);
                        from_source(&context, &device, options)
                            .unwrap_or_else(|e| panic!("{}::Program::create_and_build_from_binary failed: {}", name, e))
                    }
                }
            }
            false => from_source(&context, &device, options)
                .unwrap_or_else(|e| panic!("{}::Program::create_and_build_from_binary failed: {}", name, e)),
        };
        info!("Kernels: {:?}", program.kernel_names());
        let mut heavy_hash =
            Kernel::create(&program, "heavy_hash").unwrap_or_else(|_| panic!("{}::Kernel::create failed", name));
        if heavy_hash.num_args().map_err(|e| e.to_string())? != HEAVY_HASH_ARGUMENTS {
            warn!("{}: Embedded OpenCL binary uses an older kernel ABI; compiling current source", name);
            program = from_source(&context, &device, options)
                .unwrap_or_else(|e| panic!("{}::Program::create_and_build_from_source failed: {}", name, e));
            heavy_hash =
                Kernel::create(&program, "heavy_hash").unwrap_or_else(|_| panic!("{}::Kernel::create failed", name));
        }

        let queue =
            CommandQueue::create_with_properties(&context, device.id(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0)
                .unwrap_or_else(|_| {
                    warn!("{}: Out-of-order queue unsupported, falling back to in-order", name);
                    CommandQueue::create_with_properties(&context, device.id(), 0, 0)
                        .unwrap_or_else(|_| panic!("{}::CommandQueue::create_with_properties failed", name))
                });

        let final_nonce = Buffer::<cl_ulong>::create(context_ref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .expect("Buffer allocation failed");
        let winner_found = Buffer::<cl_uint>::create(context_ref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .expect("Buffer allocation failed");
        let final_hash = Buffer::<[cl_ulong; 4]>::create(context_ref, CL_MEM_WRITE_ONLY, 1, ptr::null_mut())
            .expect("Buffer allocation failed");

        let hash_header = Buffer::<cl_uchar>::create(context_ref, CL_MEM_READ_ONLY, 72, ptr::null_mut())
            .expect("Buffer allocation failed");
        let matrix = Buffer::<cl_uchar>::create(context_ref, CL_MEM_READ_ONLY, 64 * 64, ptr::null_mut())
            .expect("Buffer allocation failed");
        let target = Buffer::<cl_ulong>::create(context_ref, CL_MEM_READ_ONLY, 4, ptr::null_mut())
            .expect("Buffer allocation failed");

        let mut seed = [1u64; 4];
        seed.try_fill(&mut rand::thread_rng())?;

        let random_state = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation");
                let random_state =
                    Buffer::<cl_ulong>::create(context_ref, CL_MEM_READ_WRITE, 4 * chosen_workload, ptr::null_mut())
                        .expect("Buffer allocation failed");
                let rand_state =
                    Xoshiro256StarStar::new(&seed).iter_jump_state().take(chosen_workload).collect::<Vec<[u64; 4]>>();
                let mut random_state_local: *mut c_void = std::ptr::null_mut::<c_void>();
                info!("{}: Generating initial seed. This may take some time.", name);

                queue
                    .enqueue_map_buffer(
                        &random_state,
                        CL_BLOCKING,
                        CL_MAP_WRITE,
                        0,
                        32 * chosen_workload,
                        &mut random_state_local,
                        &[],
                    )
                    .map_err(|e| e.to_string())?
                    .wait()
                    .unwrap();
                if random_state_local.is_null() {
                    return Err(format!("{}::could not load random state vector to memory. Consider changing random or lowering workload", name).into());
                }
                unsafe {
                    random_state_local.copy_from(rand_state.as_ptr() as *mut c_void, 32 * chosen_workload);
                }
                // queue.enqueue_svm_unmap(&random_state,&[]).map_err(|e| e.to_string())?;
                queue
                    .enqueue_unmap_mem_object(random_state.get(), random_state_local, &[])
                    .map_err(|e| e.to_string())
                    .unwrap()
                    .wait()
                    .unwrap();
                info!("{}: Done generating initial seed", name);
                random_state
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation");
                let mut random_state = Buffer::<cl_ulong>::create(context_ref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
                    .expect("Buffer allocation failed");
                queue
                    .enqueue_write_buffer(&mut random_state, CL_BLOCKING, 0, &[thread_rng().next_u64()], &[])
                    .map_err(|e| e.to_string())
                    .unwrap()
                    .wait()
                    .unwrap();
                random_state
            }
        };
        Ok(Self {
            context,
            local_size,
            workload: chosen_workload,
            random: *random,
            heavy_hash,
            random_state,
            queue,
            final_nonce,
            winner_found,
            final_hash,
            hash_header,
            matrix,
            target,
            // Must equal the kernel's v_dot8 decision: this flag drives the packed
            // (2-nibbles/byte) matrix upload in load_block_constants, which the
            // v_dot8_u32_u4 kernel requires. `experimental_amd_use` already folds in
            // both the capability default and the legacy --experimental-amd flag.
            experimental_amd: experimental_amd_use,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_kernel_source_has_expected_abi() {
        let Ok(platforms) = opencl3::platform::get_platforms() else {
            return;
        };
        let Some(device_id) = platforms
            .iter()
            .filter_map(|platform| platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok())
            .flatten()
            .next()
        else {
            return;
        };
        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("create OpenCL test context");
        let program = from_source(&context, &device, "").expect("build current OpenCL source");
        let kernel = Kernel::create(&program, "heavy_hash").expect("create heavy_hash kernel");

        assert_eq!(kernel.num_args().unwrap(), HEAVY_HASH_ARGUMENTS);
    }

    #[test]
    #[ignore = "requires an OpenCL GPU"]
    fn publishes_nonce_zero_from_multiple_workgroups() {
        let device_id = opencl3::platform::get_platforms()
            .expect("enumerate OpenCL platforms")
            .iter()
            .filter_map(|platform| platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok())
            .flatten()
            .next()
            .expect("find OpenCL GPU");
        let mut worker = OpenCLGPUWorker::new(
            Device::new(device_id),
            4096.0,
            true,
            false,
            false,
            &NonceGenEnum::Lean,
        )
        .expect("create OpenCL worker");
        worker.load_block_constants(&[0; 72], &[[0; 64]; 64], &[u64::MAX; 4]).expect("load OpenCL constants");

        for _ in 0..100 {
            worker.calculate_hash(None, 0, 0).expect("launch OpenCL kernel");
            worker.sync().expect("synchronize OpenCL kernel");
            assert_eq!(worker.read_winner().unwrap(), Some(0));
        }
    }
}

fn from_source(context: &Context, device: &Device, options: &str) -> Result<Program, String> {
    let version = device.version()?;
    let v = version.split(' ').nth(1).unwrap();
    let mut compile_options = options.to_string();
    compile_options += CL_MAD_ENABLE;
    compile_options += CL_FINITE_MATH_ONLY;
    if v == "2.0" || v == "2.1" || v == "3.0" {
        info!("Compiling with OpenCl 2");
        compile_options += CL_STD_2_0;
    }
    compile_options += &match Platform::new(device.platform().unwrap()).name() {
        Ok(name) => format!(
            "-D{} ",
            name.chars()
                .map(|c| match c.is_ascii_alphanumeric() {
                    true => c,
                    false => '_',
                })
                .collect::<String>()
                .to_uppercase()
        ),
        Err(_) => String::new(),
    };
    compile_options += &match device.compute_capability_major_nv() {
        Ok(major) => format!("-D __COMPUTE_MAJOR__={} ", major),
        Err(_) => String::new(),
    };
    compile_options += &match device.compute_capability_minor_nv() {
        Ok(minor) => format!("-D __COMPUTE_MINOR__={} ", minor),
        Err(_) => String::new(),
    };

    // Hack to recreate the AMD flags
    compile_options += &match device.pcie_id_amd() {
        Ok(_) => {
            // ROCm reports gfx906/gfx90a as e.g. "gfx906:sramecc+:xnack-". The ':'
            // and '+'/'-' are not valid in a preprocessor macro name, so
            // `-D __<name>__` would define a garbage symbol and `__gfx906__` would
            // never be set — silently dropping the JIT path onto the scalar
            // fallback. Strip the target-feature suffix (the binary-lookup path
            // above already does this) so `__gfx906__` is defined cleanly.
            let mut device_name = device.name().unwrap_or_else(|_| "Unknown".into()).to_lowercase();
            if let Some((base, _)) = device_name.split_once(':') {
                device_name = base.to_string();
            }
            format!("-D OPENCL_PLATFORM_AMD -D __{}__ ", device_name)
        }
        Err(_) => String::new(),
    };

    info!("Build OpenCL with {}", compile_options);

    Program::create_and_build_from_source(context, PROGRAM_SOURCE, compile_options.as_str())
}
