//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in a raw CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Two walk sources, both gathering the canonical name-sorted GGUF layout:
//! - `load_llama`: zero-dup over the in-process llama.cpp engine's resident tensors (the
//!   inference GPU — one VRAM copy serves inference + walk).
//! - `load_raw`: a standalone VRAM upload of the GGUF's raw quantized bytes (mining-only GPUs
//!   on a multi-GPU rig, or when llama's resident layout is not byte-compatible).
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

use anyhow::{anyhow, Result};
use log::{info, warn};

use cudarc::driver::{result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};

const PTX_SM90: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm90.ptx"));
const PTX_SM89: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm89.ptx"));
const PTX_SM86: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm86.ptx"));
const PTX_SM80: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm80.ptx"));
const PTX_SM75: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm75.ptx"));
const PTX_SM70: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm70.ptx"));
const PTX_SM61: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm61.ptx"));
const FATBIN_LEGACY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_mine_legacy.fatbin"));
const FATBIN_NEXTGEN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_mine_nextgen.fatbin"));
const CHUNK_BYTES: usize = 32;
const POM_KERNEL_NAME: &str = "pom_mine";
const POM_V3_KERNEL_NAME: &str = "pom_mine_v3";
const POM_V3_DUMP_KERNEL_NAME: &str = "pom_mine_v3_dump";
const POM_V4_KERNEL_NAME: &str = "pom_mine_v4";
const POM_V4_SHARED_BYTES: u32 = 2048;
const POM_V4_CHASE_KERNEL_NAME: &str = "pom_mine_v4_chase";
const POM_V4_TC_KERNEL_NAME: &str = "pom_mine_v4_tc";
/// Must match V4_TC_WARPS / V4_TC_PIPE in cuda/pom_mine.cu.
const V4_TC_WARPS: u64 = 4;
const V4_TC_PIPE: u32 = 3;
/// Per-warp dynamic shared: V4_TC_PIPE tile buffers + the state, 256 u32 each.
const POM_V4_TC_SHARED_PER_WARP: u32 = 256 * (V4_TC_PIPE + 1) * 4;
/// v3 dynamic shared bytes (the 64 KB tile) — needs the opt-in attribute; cc >= 7.0 only.
const POM_V3_SHARED_BYTES: u32 = crate::pom_v3::POM_V3_TILE_BYTES as u32;

const POM_PTX_CANDIDATES: [(&str, &str, &str); 7] = [
    ("pom_mine_mod_sm90", "sm_90", PTX_SM90),
    ("pom_mine_mod_sm89", "sm_89", PTX_SM89),
    ("pom_mine_mod_sm86", "sm_86", PTX_SM86),
    ("pom_mine_mod_sm80", "sm_80", PTX_SM80),
    ("pom_mine_mod_sm75", "sm_75", PTX_SM75),
    ("pom_mine_mod_sm70", "sm_70", PTX_SM70),
    ("pom_mine_mod_sm61", "sm_61", PTX_SM61),
];

#[derive(Clone, Debug)]
pub struct GpuKernelInfo {
    pub device_id: u32,
    pub cc_major: Option<i32>,
    pub cc_minor: Option<i32>,
    pub image: String,
    pub load_path: String,
}

fn gpu_kernel_info() -> &'static Mutex<HashMap<u32, GpuKernelInfo>> {
    static GPU_KERNEL_INFO: OnceLock<Mutex<HashMap<u32, GpuKernelInfo>>> = OnceLock::new();
    GPU_KERNEL_INFO.get_or_init(|| Mutex::new(HashMap::new()))
}

fn set_gpu_kernel_info(
    device_id: usize,
    cc: Option<(i32, i32)>,
    image: &str,
    load_path: &str,
) {
    let entry = GpuKernelInfo {
        device_id: device_id as u32,
        cc_major: cc.map(|x| x.0),
        cc_minor: cc.map(|x| x.1),
        image: image.to_string(),
        load_path: load_path.to_string(),
    };
    if let Ok(mut g) = gpu_kernel_info().lock() {
        g.insert(device_id as u32, entry);
    }
}

pub fn list_gpu_kernel_info() -> Vec<GpuKernelInfo> {
    let mut out = gpu_kernel_info()
        .lock()
        .map(|g| g.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    out.sort_by_key(|e| e.device_id);
    out
}

#[derive(Debug)]
struct LoadedPomKernel {
    module: sys::CUmodule,
    function: sys::CUfunction,
    /// v3 (H6) entries — `None` when the loaded image predates the v3 kernel (stale fatbin)
    /// or the card cannot take the 64 KB opt-in shared attribute. Legacy mining is unaffected.
    function_v3: Option<sys::CUfunction>,
    function_v3_dump: Option<sys::CUfunction>,
    function_v4: Option<sys::CUfunction>,
    /// Tensor-core v4 solver entries. `tc_enabled` is armed per device after the compute
    /// capability is known: below sm_80 `pom_mine_v4_tc` is a stub that finds nothing.
    function_v4_chase: Option<sys::CUfunction>,
    function_v4_tc: Option<sys::CUfunction>,
    tc_enabled: bool,
}

impl Drop for LoadedPomKernel {
    fn drop(&mut self) {
        let module = self.module;
        if !module.is_null() {
            // Best-effort cleanup; a drop failure here would only leak the module.
            let _ = unsafe { result::module::unload(module) };
        }
    }
}

unsafe impl Send for LoadedPomKernel {}
unsafe impl Sync for LoadedPomKernel {}

impl LoadedPomKernel {
    /// The caller must have the target device's context bound to the current thread
    /// (`CudaContext::bind_to_thread`) — raw module loading works on the current context.
    fn from_fatbin(label: &'static str, fatbin: &'static [u8]) -> Result<Self> {
        if fatbin.is_empty() {
            return Err(anyhow!("PoM GPU: {} fatbin is empty", label));
        }
        let module = unsafe { result::module::load_data(fatbin.as_ptr() as *const c_void) }?;
        let function = unsafe { result::module::get_function(module, CString::new(POM_KERNEL_NAME).unwrap()) }?;
        let (function_v3, function_v3_dump) = load_v3_functions(module);
        let function_v4 = unsafe { result::module::get_function(module, CString::new(POM_V4_KERNEL_NAME).unwrap()) }.ok();
        let function_v4_chase =
            unsafe { result::module::get_function(module, CString::new(POM_V4_CHASE_KERNEL_NAME).unwrap()) }.ok();
        let function_v4_tc =
            unsafe { result::module::get_function(module, CString::new(POM_V4_TC_KERNEL_NAME).unwrap()) }.ok();
        Ok(Self { module, function, function_v3, function_v3_dump, function_v4, function_v4_chase, function_v4_tc, tc_enabled: false })
    }

    fn from_ptx(_label: &'static str, ptx: &'static str) -> Result<Self> {
        let c_src = CString::new(ptx)?;
        let module = unsafe { result::module::load_data(c_src.as_ptr() as *const c_void) }?;
        let function = unsafe { result::module::get_function(module, CString::new(POM_KERNEL_NAME).unwrap()) }?;
        let (function_v3, function_v3_dump) = load_v3_functions(module);
        let function_v4 = unsafe { result::module::get_function(module, CString::new(POM_V4_KERNEL_NAME).unwrap()) }.ok();
        let function_v4_chase =
            unsafe { result::module::get_function(module, CString::new(POM_V4_CHASE_KERNEL_NAME).unwrap()) }.ok();
        let function_v4_tc =
            unsafe { result::module::get_function(module, CString::new(POM_V4_TC_KERNEL_NAME).unwrap()) }.ok();
        Ok(Self { module, function, function_v3, function_v3_dump, function_v4, function_v4_chase, function_v4_tc, tc_enabled: false })
    }

    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        bases_dev: &CudaSlice<u64>,
        prefix_dev: &CudaSlice<u64>,
        t_count: u32,
        n_total_chunks: u64,
        p_words: &[u64; 4],
        s_words: &[u64; 4],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
        walk_v2: u32,
    ) -> Result<Option<u64>> {
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = stream.clone_htod(&[u64::MAX])?;
        // The kernel grinds 2 nonces per thread (ILP x2 — see cuda/pom_mine.cu), so the grid
        // covers ceil(batch/2) threads. Nonce coverage of [start, start+batch) is unchanged.
        let threads = (batch + 1) / 2;
        let grid = ((threads + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        let (bases_ptr, _bases_guard) = bases_dev.device_ptr(stream);
        let (prefix_ptr, _prefix_guard) = prefix_dev.device_ptr(stream);
        let (winner_ptr, _winner_guard) = winner.device_ptr(stream);

        let mut params: [*mut c_void; 22] = [
            (&bases_ptr as *const _ as *mut c_void),
            (&prefix_ptr as *const _ as *mut c_void),
            (&t_count as *const _ as *mut c_void),
            (&n_total_chunks as *const _ as *mut c_void),
            (&k as *const _ as *mut c_void),
            (&p_words[0] as *const _ as *mut c_void),
            (&p_words[1] as *const _ as *mut c_void),
            (&p_words[2] as *const _ as *mut c_void),
            (&p_words[3] as *const _ as *mut c_void),
            (&s_words[0] as *const _ as *mut c_void),
            (&s_words[1] as *const _ as *mut c_void),
            (&s_words[2] as *const _ as *mut c_void),
            (&s_words[3] as *const _ as *mut c_void),
            (&timestamp as *const _ as *mut c_void),
            (&t[0] as *const _ as *mut c_void),
            (&t[1] as *const _ as *mut c_void),
            (&t[2] as *const _ as *mut c_void),
            (&t[3] as *const _ as *mut c_void),
            (&start as *const _ as *mut c_void),
            (&batch as *const _ as *mut c_void),
            (&winner_ptr as *const _ as *mut c_void),
            (&walk_v2 as *const _ as *mut c_void),
        ];

        unsafe { result::launch_kernel(self.function, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, stream.cu_stream(), &mut params) }?;
        stream.synchronize()?;

        let w = stream.clone_dtoh(&winner)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }

    /// v3 (H6) grind: one CUDA block per nonce over `[start, start + batch)`.
    #[allow(clippy::too_many_arguments)]
    fn launch_v3(
        &self,
        stream: &Arc<CudaStream>,
        bases_dev: &CudaSlice<u64>,
        prefix_dev: &CudaSlice<u64>,
        t_count: u32,
        n_tiles: u64,
        p_words: &[u64; 4],
        s_words: &[u64; 4],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Result<Option<u64>> {
        let function = self.function_v3.ok_or_else(|| anyhow!("PoM GPU: loaded kernel image has no v3 entry"))?;
        let t = words4(target_le);
        let k = crate::pom_v3::POM_V3_K as u32;
        let winner = stream.clone_htod(&[u64::MAX])?;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (crate::pom_v3::POM_V3_D as u32, 1, 1),
            shared_mem_bytes: POM_V3_SHARED_BYTES,
        };

        let (bases_ptr, _bases_guard) = bases_dev.device_ptr(stream);
        let (prefix_ptr, _prefix_guard) = prefix_dev.device_ptr(stream);
        let (winner_ptr, _winner_guard) = winner.device_ptr(stream);

        let mut params: [*mut c_void; 21] = [
            (&bases_ptr as *const _ as *mut c_void),
            (&prefix_ptr as *const _ as *mut c_void),
            (&t_count as *const _ as *mut c_void),
            (&n_tiles as *const _ as *mut c_void),
            (&k as *const _ as *mut c_void),
            (&p_words[0] as *const _ as *mut c_void),
            (&p_words[1] as *const _ as *mut c_void),
            (&p_words[2] as *const _ as *mut c_void),
            (&p_words[3] as *const _ as *mut c_void),
            (&s_words[0] as *const _ as *mut c_void),
            (&s_words[1] as *const _ as *mut c_void),
            (&s_words[2] as *const _ as *mut c_void),
            (&s_words[3] as *const _ as *mut c_void),
            (&timestamp as *const _ as *mut c_void),
            (&t[0] as *const _ as *mut c_void),
            (&t[1] as *const _ as *mut c_void),
            (&t[2] as *const _ as *mut c_void),
            (&t[3] as *const _ as *mut c_void),
            (&start as *const _ as *mut c_void),
            (&batch as *const _ as *mut c_void),
            (&winner_ptr as *const _ as *mut c_void),
        ];

        unsafe { result::launch_kernel(function, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, stream.cu_stream(), &mut params) }?;
        stream.synchronize()?;

        let w = stream.clone_dtoh(&winner)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }

    /// Arms the tensor-core v4 solver for this device. Needs sm_80+ (real int8 mma SASS),
    /// both entries present, and `KERYX_POM_V4_TC` not set to 0.
    fn arm_tc(&mut self, device_id: usize, cc: Option<(i32, i32)>) {
        if std::env::var("KERYX_POM_V4_TC").ok().as_deref() == Some("0") {
            return;
        }
        let sm80 = matches!(cc, Some((major, _)) if major >= 8);
        self.tc_enabled = sm80 && self.function_v4_chase.is_some() && self.function_v4_tc.is_some();
        if self.tc_enabled {
            info!("PoM[gpu{}]: v4 tensor-core solver armed", device_id);
        }
    }

    /// v4 (re-walk) grind: one block of 32 threads per nonce over `[start, start + batch)`.
    #[allow(clippy::too_many_arguments)]
    fn launch_v4(&self, stream: &Arc<CudaStream>, bases_dev: &CudaSlice<u64>, prefix_dev: &CudaSlice<u64>, t_count: u32, n_tiles: u64, p_words: &[u64; 4], s_words: &[u64; 4], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Result<Option<u64>> {
        let t = words4(target_le);
        let k = crate::pom_v4::POM_V4_K as u32;
        let winner = stream.clone_htod(&[u64::MAX])?;
        let (winner_ptr, _wg) = winner.device_ptr(stream);
        let (bases_ptr, _bg) = bases_dev.device_ptr(stream);
        let (prefix_ptr, _pg) = prefix_dev.device_ptr(stream);

        if self.tc_enabled {
            let chase = self.function_v4_chase.ok_or_else(|| anyhow!("PoM GPU: no pom_mine_v4_chase entry"))?;
            let walk = self.function_v4_tc.ok_or_else(|| anyhow!("PoM GPU: no pom_mine_v4_tc entry"))?;
            // The chase overwrites every word, so the buffer needs no zeroing.
            let offsets = stream.alloc_zeros::<u32>(batch as usize * k as usize)?;
            let (offsets_ptr, _og) = offsets.device_ptr(stream);

            let chase_cfg = LaunchConfig {
                grid_dim: (((batch + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut chase_params: [*mut c_void; 13] = [
                (&bases_ptr as *const _ as *mut c_void), (&prefix_ptr as *const _ as *mut c_void),
                (&t_count as *const _ as *mut c_void), (&n_tiles as *const _ as *mut c_void), (&k as *const _ as *mut c_void),
                (&s_words[0] as *const _ as *mut c_void), (&s_words[1] as *const _ as *mut c_void),
                (&s_words[2] as *const _ as *mut c_void), (&s_words[3] as *const _ as *mut c_void),
                (&timestamp as *const _ as *mut c_void),
                (&start as *const _ as *mut c_void), (&batch as *const _ as *mut c_void),
                (&offsets_ptr as *const _ as *mut c_void),
            ];
            unsafe { result::launch_kernel(chase, chase_cfg.grid_dim, chase_cfg.block_dim, chase_cfg.shared_mem_bytes, stream.cu_stream(), &mut chase_params) }?;

            let walk_cfg = LaunchConfig {
                grid_dim: (((batch + V4_TC_WARPS - 1) / V4_TC_WARPS) as u32, 1, 1),
                block_dim: ((V4_TC_WARPS * 32) as u32, 1, 1),
                shared_mem_bytes: (V4_TC_WARPS as u32) * POM_V4_TC_SHARED_PER_WARP,
            };
            let mut walk_params: [*mut c_void; 21] = [
                (&bases_ptr as *const _ as *mut c_void), (&prefix_ptr as *const _ as *mut c_void),
                (&t_count as *const _ as *mut c_void), (&k as *const _ as *mut c_void),
                (&p_words[0] as *const _ as *mut c_void), (&p_words[1] as *const _ as *mut c_void),
                (&p_words[2] as *const _ as *mut c_void), (&p_words[3] as *const _ as *mut c_void),
                (&s_words[0] as *const _ as *mut c_void), (&s_words[1] as *const _ as *mut c_void),
                (&s_words[2] as *const _ as *mut c_void), (&s_words[3] as *const _ as *mut c_void),
                (&timestamp as *const _ as *mut c_void),
                (&t[0] as *const _ as *mut c_void), (&t[1] as *const _ as *mut c_void),
                (&t[2] as *const _ as *mut c_void), (&t[3] as *const _ as *mut c_void),
                (&start as *const _ as *mut c_void), (&batch as *const _ as *mut c_void),
                (&offsets_ptr as *const _ as *mut c_void), (&winner_ptr as *const _ as *mut c_void),
            ];
            unsafe { result::launch_kernel(walk, walk_cfg.grid_dim, walk_cfg.block_dim, walk_cfg.shared_mem_bytes, stream.cu_stream(), &mut walk_params) }?;
            stream.synchronize()?;
            let w = stream.clone_dtoh(&winner)?[0];
            return Ok(if w == u64::MAX { None } else { Some(w) });
        }

        let function = self.function_v4.ok_or_else(|| anyhow!("PoM GPU: loaded kernel image has no pom_mine_v4 entry"))?;
        let cfg = LaunchConfig { grid_dim: (batch as u32, 1, 1), block_dim: (crate::pom_v4::POM_V4_D as u32, 1, 1), shared_mem_bytes: POM_V4_SHARED_BYTES };
        let mut params: [*mut c_void; 21] = [
            (&bases_ptr as *const _ as *mut c_void), (&prefix_ptr as *const _ as *mut c_void),
            (&t_count as *const _ as *mut c_void), (&n_tiles as *const _ as *mut c_void), (&k as *const _ as *mut c_void),
            (&p_words[0] as *const _ as *mut c_void), (&p_words[1] as *const _ as *mut c_void), (&p_words[2] as *const _ as *mut c_void), (&p_words[3] as *const _ as *mut c_void),
            (&s_words[0] as *const _ as *mut c_void), (&s_words[1] as *const _ as *mut c_void), (&s_words[2] as *const _ as *mut c_void), (&s_words[3] as *const _ as *mut c_void),
            (&timestamp as *const _ as *mut c_void),
            (&t[0] as *const _ as *mut c_void), (&t[1] as *const _ as *mut c_void), (&t[2] as *const _ as *mut c_void), (&t[3] as *const _ as *mut c_void),
            (&start as *const _ as *mut c_void), (&batch as *const _ as *mut c_void), (&winner_ptr as *const _ as *mut c_void),
        ];
        unsafe { result::launch_kernel(function, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, stream.cu_stream(), &mut params) }?;
        stream.synchronize()?;
        let w = stream.clone_dtoh(&winner)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }

    /// v3 (H6) dump: re-walk ONE (winning) nonce and return (states S_0..=S_K concatenated,
    /// snippets, fold64(root_K)) for the host proof-build.
    #[allow(clippy::too_many_arguments)]
    fn launch_v3_dump(
        &self,
        stream: &Arc<CudaStream>,
        bases_dev: &CudaSlice<u64>,
        prefix_dev: &CudaSlice<u64>,
        t_count: u32,
        n_tiles: u64,
        s_words: &[u64; 4],
        timestamp: u64,
        nonce: u64,
    ) -> Result<(Vec<u8>, Vec<u8>, u64)> {
        let function =
            self.function_v3_dump.ok_or_else(|| anyhow!("PoM GPU: loaded kernel image has no v3 dump entry"))?;
        let k = crate::pom_v3::POM_V3_K;
        let d = crate::pom_v3::POM_V3_D;
        let states = stream.clone_htod(vec![0u8; (k + 1) * d * d].as_slice())?;
        let snippets = stream.clone_htod(vec![0u8; k * crate::pom_v3::POM_V3_SNIPPET_BYTES].as_slice())?;
        let final_state = stream.clone_htod(&[0u64])?;
        let k32 = k as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (d as u32, 1, 1),
            shared_mem_bytes: POM_V3_SHARED_BYTES,
        };

        let (bases_ptr, _bases_guard) = bases_dev.device_ptr(stream);
        let (prefix_ptr, _prefix_guard) = prefix_dev.device_ptr(stream);
        let (states_ptr, _states_guard) = states.device_ptr(stream);
        let (snippets_ptr, _snippets_guard) = snippets.device_ptr(stream);
        let (final_ptr, _final_guard) = final_state.device_ptr(stream);

        let mut params: [*mut c_void; 14] = [
            (&bases_ptr as *const _ as *mut c_void),
            (&prefix_ptr as *const _ as *mut c_void),
            (&t_count as *const _ as *mut c_void),
            (&n_tiles as *const _ as *mut c_void),
            (&k32 as *const _ as *mut c_void),
            (&s_words[0] as *const _ as *mut c_void),
            (&s_words[1] as *const _ as *mut c_void),
            (&s_words[2] as *const _ as *mut c_void),
            (&s_words[3] as *const _ as *mut c_void),
            (&timestamp as *const _ as *mut c_void),
            (&nonce as *const _ as *mut c_void),
            (&states_ptr as *const _ as *mut c_void),
            (&snippets_ptr as *const _ as *mut c_void),
            (&final_ptr as *const _ as *mut c_void),
        ];

        unsafe { result::launch_kernel(function, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, stream.cu_stream(), &mut params) }?;
        stream.synchronize()?;

        Ok((stream.clone_dtoh(&states)?, stream.clone_dtoh(&snippets)?, stream.clone_dtoh(&final_state)?[0]))
    }
}

/// Best-effort v3 entry lookup + opt-in shared attribute. `None` entries mean the image
/// predates the v3 kernel or the card cannot honor 64 KB of dynamic shared.
fn load_v3_functions(module: sys::CUmodule) -> (Option<sys::CUfunction>, Option<sys::CUfunction>) {
    let get = |name: &str| unsafe { result::module::get_function(module, CString::new(name).unwrap()) }.ok();
    let arm = |f: sys::CUfunction| {
        unsafe {
            result::function::set_function_attribute(
                f,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                POM_V3_SHARED_BYTES as i32,
            )
        }
        .is_ok()
        .then_some(f)
    };
    (get(POM_V3_KERNEL_NAME).and_then(arm), get(POM_V3_DUMP_KERNEL_NAME).and_then(arm))
}

fn is_nextgen_device(device_id: usize) -> bool {
    let Ok(dev) = result::device::get(device_id as i32) else {
        return false;
    };
    let major = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(0);
    let minor = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    major > 8 || (major == 8 && minor >= 9)
}

fn gpu_compute_capability(device_id: usize) -> Option<(i32, i32)> {
    let dev = result::device::get(device_id as i32).ok()?;
    let major = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .ok()?;
    let minor = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .ok()?;
    Some((major, minor))
}

/// The caller must have `device_id`'s context bound to the current thread (module loads target
/// the current CUDA context).
fn select_pom_kernel(device_id: usize) -> Result<LoadedPomKernel> {
    static FATBIN_STATUS_LOGGED: Once = Once::new();
    FATBIN_STATUS_LOGGED.call_once(|| {
        let legacy = FATBIN_LEGACY.len();
        let nextgen = FATBIN_NEXTGEN.len();
        if legacy > 0 || nextgen > 0 {
            info!(
                "PoM: prebuilt fatbins detected (legacy={} bytes, nextgen={} bytes); PTX fallback ladder currently active",
                legacy,
                nextgen
            );
        } else {
            info!("PoM: no prebuilt fatbins detected; using PTX fallback ladder");
        }
    });

    let is_nextgen_cc = is_nextgen_device(device_id);

    let fatbin_candidates: [(&str, &str, &[u8]); 2] = if is_nextgen_cc {
        [
            ("pom_mine_mod_nextgen", "nextgen fatbin", FATBIN_NEXTGEN),
            ("pom_mine_mod_legacy", "legacy fatbin", FATBIN_LEGACY),
        ]
    } else {
        [
            ("pom_mine_mod_legacy", "legacy fatbin", FATBIN_LEGACY),
            ("pom_mine_mod_nextgen", "nextgen fatbin", FATBIN_NEXTGEN),
        ]
    };

    for (module_name, label, fatbin) in fatbin_candidates {
        match LoadedPomKernel::from_fatbin(label, fatbin) {
            Ok(mut kernel) => {
                let cc = gpu_compute_capability(device_id);
                if let Some((major, minor)) = cc {
                    info!(
                        "PoM[gpu{} cc{}.{}]: startup loaded {} via {}",
                        device_id,
                        major,
                        minor,
                        label,
                        module_name,
                    );
                } else {
                    info!("PoM[gpu{}]: startup loaded {} via {}", device_id, label, module_name);
                }
                set_gpu_kernel_info(device_id, cc, label, module_name);
                kernel.arm_tc(device_id, cc);
                return Ok(kernel);
            }
            Err(e) => {
                warn!("PoM[gpu{}]: {} load failed: {}", device_id, label, e);
            }
        }
    }

    for (module_name, label, ptx) in POM_PTX_CANDIDATES {
        match LoadedPomKernel::from_ptx(label, ptx) {
            Ok(mut kernel) => {
                let cc = gpu_compute_capability(device_id);
                if let Some((major, minor)) = cc {
                    info!(
                        "PoM[gpu{} cc{}.{}]: startup loaded {} PTX fallback via {}",
                        device_id,
                        major,
                        minor,
                        label,
                        module_name,
                    );
                } else {
                    info!("PoM[gpu{}]: startup loaded {} PTX fallback via {}", device_id, label, module_name);
                }
                set_gpu_kernel_info(
                    device_id,
                    cc,
                    &format!("{} PTX fallback", label),
                    module_name,
                );
                kernel.arm_tc(device_id, cc);
                return Ok(kernel);
            }
            Err(e) => {
                warn!("PoM[gpu{}]: {} PTX load failed: {}", device_id, label, e);
            }
        }
    }

    Err(anyhow!("PoM GPU: no compatible PTX image for this device/driver"))
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Total VRAM (MB) of every CUDA device, in **CUDA device order** — the same ordering
/// `CudaContext::new(id)` uses — so an entry `(id, mb)` is the VRAM of the device the miner would
/// mine/serve on for that `id`. Sourced from the CUDA driver, NOT nvidia-smi: nvidia-smi orders by
/// PCI position, which disagrees with CUDA's default `FASTEST_FIRST` ordering on a mixed rig, so a
/// line-order mapping would read the wrong card's VRAM. Returns an empty vec when no CUDA driver is
/// present (CPU-only / AMD hosts). Never panics — a driver-load failure inside cudarc is caught and
/// treated as "no devices".
pub fn query_all_gpus_vram() -> Vec<(usize, u64)> {
    std::panic::catch_unwind(|| {
        if result::init().is_err() {
            return Vec::new();
        }
        let count = result::device::get_count().unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for ordinal in 0..count {
            let Ok(dev) = result::device::get(ordinal) else {
                continue;
            };
            // SAFETY: `dev` is a valid device handle just returned by `device::get(ordinal)`.
            if let Ok(bytes) = unsafe { result::device::total_mem(dev) } {
                out.push((ordinal as usize, (bytes / (1024 * 1024)) as u64));
            }
        }
        out
    })
    .unwrap_or_default()
}

pub struct PomGpuMiner {
    /// Kept for context lifetime + `bind_to_thread` on launches from worker threads.
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: LoadedPomKernel,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _uploads: Vec<CudaSlice<u8>>, // tensors we uploaded ourselves, kept alive for the gather
}

impl PomGpuMiner {
    /// Standalone walk source: upload the mining model's raw GGUF bytes to a specific CUDA
    /// device (canonical name-sorted tensor order) and build the gather index over our own
    /// copies. Used on mining-only GPUs that don't host the in-process llama engine — the
    /// uploaded bytes ARE the canonical on-disk bytes, so no byte-gate is needed here (the
    /// N-guard in `ensure_installed_inner` still cross-checks against the host index).
    pub fn load_raw(gguf_path: &str, device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();

        let mut file = std::fs::File::open(gguf_path)?;
        let meta = crate::gguf::GgufMeta::read(&mut file)?;
        let names = meta.sorted_names(); // canonical order — matches pom-rt-builder / the node R_T

        let mut uploads: Vec<CudaSlice<u8>> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut host_buf: Vec<u8> = Vec::new();
        for name in &names {
            let t = &meta.tensors[name];
            let chunks = t.nbytes / CHUNK_BYTES as u64;
            if chunks == 0 {
                continue;
            }
            host_buf.resize(t.nbytes as usize, 0);
            crate::pom::read_exact_at(&file, &mut host_buf, meta.tensor_data_offset + t.offset)?;
            let dev = stream.clone_htod(host_buf.as_slice())?;
            bases.push(dev.device_ptr(&stream).0 as u64);
            uploads.push(dev);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(anyhow!("PoM GPU: model produced 0 chunks"));
        }

        let bases_dev = stream.clone_htod(bases.as_slice())?;
        let prefix_dev = stream.clone_htod(prefix.as_slice())?;
        // Load the best prebuilt module for this card and keep the raw CUfunction cached.
        let kernel = select_pom_kernel(device_id)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _uploads: uploads,
        })
    }

    /// Gather over the IN-PROCESS llama.cpp engine in canonical GGUF (name-sorted) order.
    /// A tensor whose resident copy matches the GGUF unambiguously (unique name, exact size)
    /// is walked zero-dup in place; host-resident tensors get a small device upload of our
    /// own; anything llama repacked, duplicated or dropped is uploaded from the possession
    /// index instead, so the walked blob is always the canonical one R_T pins.
    /// `model_id` selects the host possession index for the uploads and the consensus byte-gate.
    pub fn load_llama(gguf: &str, device_id: usize, model_id: &[u8; 32]) -> Result<Self> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let ts = crate::llama_engine::tensors()
            .ok_or_else(|| anyhow!("PoM GPU: llama engine tensors unavailable"))?;
        let canonical = canonical_tensor_list(gguf)
            .ok_or_else(|| anyhow!("PoM GPU: canonical GGUF tensor list unreadable"))?;
        let plan = plan_canonical_gather(&canonical, &ts);
        if plan.exceeds_upload_budget() {
            return Err(anyhow!(
                "PoM GPU: llama-resident layout too foreign — {} of {} bytes need a canonical upload",
                plan.index_bytes, plan.total_bytes
            ));
        }
        let idx = crate::pom::active_index_for_model(model_id);
        if plan.index_bytes > 0 && idx.is_none() {
            return Err(anyhow!("PoM GPU: canonical fallback needs the host possession index"));
        }
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        for (nbytes, source) in &plan.entries {
            let chunks = (nbytes / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let base = match source {
                GatherSource::DevicePtr(p) => *p,
                GatherSource::HostPtr(p) => {
                    // Host-resident in ggml (CPU buffer): the walk needs device memory — upload
                    // our own copy of the raw bytes (identical to the GGUF bytes, same as the
                    // pointer).
                    let host: &[u8] = unsafe { std::slice::from_raw_parts(*p as *const u8, *nbytes) };
                    let dev = stream.clone_htod(host)?;
                    let p = dev.device_ptr(&stream).0 as u64;
                    uploads.push(dev);
                    p
                }
                GatherSource::FromIndex => {
                    let idx = idx.as_ref().unwrap();
                    let start = *prefix.last().unwrap();
                    let mut buf = vec![0u8; chunks as usize * CHUNK_BYTES];
                    for i in 0..chunks as usize {
                        buf[i * CHUNK_BYTES..][..CHUNK_BYTES]
                            .copy_from_slice(&idx.read_chunk_bytes(start + i as u64));
                    }
                    let dev = stream.clone_htod(buf.as_slice())?;
                    let p = dev.device_ptr(&stream).0 as u64;
                    uploads.push(dev);
                    p
                }
            };
            bases.push(base);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(anyhow!("PoM GPU: llama engine produced 0 chunks"));
        }
        info!(
            "PoM llama canonical gather: {} tensors ({} zero-dup, {} host uploads, {} index uploads{}), N={} chunks",
            bases.len(),
            plan.zero_dup,
            plan.host_uploads,
            plan.index_uploads.len(),
            if plan.index_uploads.is_empty() {
                String::new()
            } else {
                format!(": {}", plan.index_uploads.iter().take(4).cloned().collect::<Vec<_>>().join(", "))
            },
            n_total_chunks
        );
        // BYTE GATE (consensus safety): the pool does not deep-verify every share, so a wrong
        // gather would mine garbage silently. Read back evenly-spaced chunks from the llama-owned
        // device memory and compare them byte-for-byte against the host index (GGUF pread) — any
        // mismatch refuses to mine. Full-model byte-identity for this llama build was proven once
        // by `tools/llama_zerodup_spike`; this guards every startup against regressions.
        if let Some(idx) = crate::pom::active_index_for_model(model_id) {
            if idx.n_chunks == n_total_chunks {
                let samples = 128u64;
                for kk in 0..=samples {
                    let off = if kk == samples { n_total_chunks - 1 } else { kk * (n_total_chunks / (samples + 1)) };
                    let j = prefix.partition_point(|&p| p <= off) - 1;
                    let dev_addr = bases[j] + (off - prefix[j]) * CHUNK_BYTES as u64;
                    let mut got = [0u8; CHUNK_BYTES];
                    unsafe { result::memcpy_dtoh_sync(&mut got, dev_addr)? };
                    let want = idx.read_chunk_bytes(off);
                    if got != want {
                        return Err(anyhow!(
                            "PoM llama byte gate FAILED at chunk {off} — llama-resident bytes differ from the GGUF; refusing to mine"
                        ));
                    }
                }
                info!("PoM llama byte gate: {} sampled chunks match the host index byte-for-byte.", samples + 1);
            }
        }

        let bases_dev = stream.clone_htod(bases.as_slice())?;
        let prefix_dev = stream.clone_htod(prefix.as_slice())?;
        let kernel = select_pom_kernel(device_id)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _uploads: uploads,
        })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    /// `h3` salts the pph words host-side (POM_H3_PPH_SALT); `h5_1` swaps the SEED words to the
    /// v2 salt (POM_H5_1_PPH_SALT) while the pow words stay H3 — the kernel is era-agnostic,
    /// it folds whatever word sets it receives.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool, walk_v2: bool, h5_1: bool, h5_2: bool, v3: bool, v4: bool) -> Result<Option<u64>> {
        // Worker threads rotate; make sure this device's context is current before raw launches.
        self.ctx.bind_to_thread()?;
        if v4 {
            let s_words = crate::pom::pph_words_v4(pre_pow_hash);
            let p_words = crate::pom::pph_words_for_era(pre_pow_hash, true);
            let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
            if n_tiles == 0 {
                return Err(anyhow!("PoM GPU: blob too small for the v4 walk"));
            }
            return self.kernel.launch_v4(&self.stream, &self.bases_dev, &self.prefix_dev, self.t_count, n_tiles, &p_words, &s_words, timestamp, target_le, start, batch);
        }
        let p_words = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        let s_words = crate::pom::seed_pph_words_for_era(pre_pow_hash, h3, h5_1, h5_2);
        if v3 {
            let n_tiles = self.n_total_chunks / crate::pom_v3::POM_V3_TILE_CHUNKS;
            if n_tiles == 0 {
                return Err(anyhow!("PoM GPU: blob too small for the v3 walk"));
            }
            return self.kernel.launch_v3(
                &self.stream,
                &self.bases_dev,
                &self.prefix_dev,
                self.t_count,
                n_tiles,
                &p_words,
                &s_words,
                timestamp,
                target_le,
                start,
                batch,
            );
        }
        self.kernel.launch(
            &self.stream,
            &self.bases_dev,
            &self.prefix_dev,
            self.t_count,
            self.n_total_chunks,
            &p_words,
            &s_words,
            timestamp,
            target_le,
            start,
            batch,
            walk_v2 as u32,
        )
    }

    /// v3 dump for the winning nonce: (states S_0..=S_K, snippets, fold64(root_K)).
    pub fn dump_v3(&self, pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool, h5_1: bool, h5_2: bool) -> Result<(Vec<u8>, Vec<u8>, u64)> {
        self.ctx.bind_to_thread()?;
        let s_words = crate::pom::seed_pph_words_for_era(pre_pow_hash, h3, h5_1, h5_2);
        let n_tiles = self.n_total_chunks / crate::pom_v3::POM_V3_TILE_CHUNKS;
        if n_tiles == 0 {
            return Err(anyhow!("PoM GPU: blob too small for the v3 walk"));
        }
        self.kernel.launch_v3_dump(&self.stream, &self.bases_dev, &self.prefix_dev, self.t_count, n_tiles, &s_words, timestamp, nonce)
    }

}

// Per-GPU PoM miners. Host-side WeightIndex remains shared; only the CUDA-resident worker state
// is duplicated per device. This avoids all workers contending over a single GPU0-bound miner.
fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// Guards the one-time shared host index build. All workers may race into PoM activation, but the
// heavy GGUF -> WeightIndex build must happen exactly once for the process.
fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

/// Install the GPU miner for a specific CUDA device.
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

/// Removes only `device_id`'s entry from a `device -> miner` map, leaving every other device's
/// entry untouched. Pulled out as a tiny generic helper (over the map's value type) purely so
/// this scoping behavior is unit-testable without a real, CUDA-backed `PomGpuMiner` — production
/// always calls it through `uninstall` against `HashMap<u32, Arc<PomGpuMiner>>`.
fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) -> Option<T> {
    map.remove(&device_id)
}

/// Block until `item` is the only remaining handle, or the deadline passes. Returns whether the
/// wait succeeded.
fn wait_for_sole_owner<T>(item: &Arc<T>, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while Arc::strong_count(item) > 1 {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    true
}

/// Drop the GPU miner for `device_id` only, releasing its hold on that device's mining-model VRAM
/// (gather + uploads) so the inference engine can load another model there. Mining on that
/// device is paused during inference anyway.
///
/// Scoped to a single device on purpose: only the device colocated with inference (the llama
/// engine's GPU — see `slm::load_and_run_inference`) ever shares VRAM with the inference engine
/// via `load_llama`'s zero-dup gather, or otherwise needs to make room for an inference model
/// swap. Other devices in a multi-GPU rig run fully standalone `PomGpuMiner`s
/// (`PomGpuMiner::load_raw`) that never touch the inference engine's VRAM. A previous version of
/// this function called `g.clear()`, dropping every device's resident miner on every inference
/// model swap — needlessly forcing GPU1+ rigs to fully reload their GGUF from disk and rebuild
/// the gather index (`ensure_installed_inner`'s own doc comment calls this reload "Heavy") even
/// though nothing about them changed.
pub fn uninstall(device_id: u32) {
    let removed = match miners().lock() {
        Ok(mut g) => remove_device_entry(&mut g, device_id),
        Err(_) => None,
    };
    // BARRIER before the caller frees any VRAM this miner walks over: a mining thread clones the
    // handle and launches outside the map lock, so removing the entry does not stop an in-flight
    // walk. Its launch synchronizes before it drops its handle, so waiting for the last handle is
    // enough. Freeing under a live walk raises a sticky CUDA_ERROR_ILLEGAL_ADDRESS that poisons
    // the device's context for every user of it, inference included.
    if let Some(miner) = removed {
        if !wait_for_sole_owner(&miner, std::time::Duration::from_secs(30)) {
            log::error!("PoM[gpu{}]: a walk still holds the miner after 30s — releasing anyway", device_id);
        }
    }
}

/// Whether the GPU miner is currently installed for `device_id`.
pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

/// Raised before an OPoI inference is spawned, lowered once no inference is in flight. While
/// raised, no PoM operation may start or reload a model — including a worker that acquired the
/// lifecycle lock before the pause.
static INFERENCE_PAUSED: AtomicBool = AtomicBool::new(false);

pub fn set_inference_paused(paused: bool) {
    INFERENCE_PAUSED.store(paused, Ordering::Release);
}

pub fn inference_paused() -> bool {
    INFERENCE_PAUSED.load(Ordering::Acquire)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Convenience: search a nonce batch via the installed miner for a specific device.
#[allow(clippy::too_many_arguments)]
pub fn mine(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool, walk_v2: bool, h5_1: bool, h5_2: bool, v3: bool, v4: bool) -> Option<u64> {
    if inference_paused() {
        return None;
    }
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch, h3, walk_v2, h5_1, h5_2, v3, v4).ok().flatten()
}

/// Convenience: v3 dump for the winning nonce via the installed miner for a specific device.
pub fn dump_v3(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool, h5_1: bool, h5_2: bool) -> Option<(Vec<u8>, Vec<u8>, u64)> {
    if inference_paused() {
        return None;
    }
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.dump_v3(pre_pow_hash, timestamp, nonce, h3, h5_1, h5_2).ok()
}

/// Per-GPU mining-tier identity for rebuilds: `device_id -> (model_id, gguf_path)`. A heterogeneous
/// rig mines a different tier per GPU (the highest its VRAM holds), so this is keyed by device rather
/// than a single process-wide tier.
static MINING_TIERS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();

fn mining_tiers() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    MINING_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a GPU's mining tier so its miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut g) = mining_tiers().lock() {
        g.insert(device_id, (model_id, gguf_path));
    }
}

/// Per-GPU **hardware** tier (VRAM-derived, DAA-independent). Distinct from `mining_tiers` (the
/// per-GPU *model*, which the H5 crossing swaps): a device keeps its hardware tier for life; only
/// the model that tier mines changes at the era boundary.
static DEVICE_TIERS: OnceLock<Mutex<HashMap<u32, crate::models::Tier>>> = OnceLock::new();

fn device_tiers() -> &'static Mutex<HashMap<u32, crate::models::Tier>> {
    DEVICE_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a GPU's fixed hardware tier so the era crossing can look up which model that tier must
/// mine at the new DAA (`pom_model_for_tier`).
pub fn set_device_tier(device_id: u32, tier: crate::models::Tier) {
    if let Ok(mut g) = device_tiers().lock() {
        g.insert(device_id, tier);
    }
}

/// Hot-swap the resident mining model at an era crossing: when `daa` reaches a model's gate, the
/// affected GPUs switch to the era-correct model in place, no restart. No-op each block until a
/// device's era-correct model actually changes — and inert entirely with the current fixed post-H5
/// lineup. Called each tick from the loop, so a miner upgraded before a gate crosses over on its own.
pub fn advance_mining_tier_if_due(daa: u64) {
    if inference_paused() {
        return;
    }
    let devices: Vec<(u32, crate::models::Tier)> = match device_tiers().lock() {
        Ok(g) => g.iter().map(|(d, t)| (*d, *t)).collect(),
        Err(_) => return,
    };
    let mut swapped = false;
    for &(dev, tier) in &devices {
        // No model for this tier in the era being entered: nothing to swap to, the device simply
        // has nothing valid to mine until its own gate.
        let Some(spec) = crate::models::pom_model_for_tier(daa, tier) else { continue };
        let current = mining_tiers().lock().ok().and_then(|g| g.get(&dev).map(|(id, _)| *id));
        if current == Some(spec.model_id) {
            continue;
        }
        swapped = true;
        let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        info!("PoM[gpu{}]: era crossing at DAA {} — mining model → {}.", dev, daa, spec.name);
        set_mining_tier(dev, spec.model_id, gguf.clone());
        // Free the retired model's possession index (indices are keyed by MODEL, so the new
        // model's index simply builds under its own key at the next ensure_installed).
        if let Some(old_id) = current {
            crate::pom::clear_index(&old_id);
        }
        // Same staleness for the in-process llama engine: `ensure_loaded` is load-once, so after the
        // crossing it would keep hosting the previous era's model. Unload it when it lives on this
        // GPU with a different GGUF so the next `ensure_installed` brings up the new model.
        // Drain this device's walk BEFORE freeing the tensors it may be gathering over.
        uninstall(dev); // force a resident reload of the new model on the next ensure_installed
        if crate::llama_engine::active_gpu() == Some(dev as usize) && !crate::llama_engine::active_for(&gguf, dev as usize) {
            crate::llama_engine::unload();
        }
    }
    // The served lineup (`SUPPORTED_SPECS`) drives the coinbase `ai:cap` announcement + inference
    // routing — refresh it as the union of era-correct models so the miner stops announcing the
    // previous era's model_ids after the crossing.
    if swapped {
        let mut union: Vec<&'static crate::models::ModelSpec> = Vec::new();
        for &(_, tier) in &devices {
            let Some(spec) = crate::models::pom_model_for_tier(daa, tier) else { continue };
            if !union.iter().any(|s| s.model_id == spec.model_id) {
                union.push(spec);
            }
        }
        if !union.is_empty() {
            // Leaked to satisfy the &'static lineup API — at most once per era crossing.
            crate::slm::init_supported(Box::leak(union.into_boxed_slice()));
        }
    }
}

/// Per-device lifecycle lock: held for a whole miner (re)build, and by the engine eviction while
/// it frees the hosted tensors. A build reads llama's resident pointers before any miner is
/// installed, so the uninstall barrier alone cannot see it — without this lock an inference swap
/// can free those tensors mid-build and poison the device's primary context.
fn device_lifecycle(device_id: u32) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<u32, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut g = LOCKS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|p| p.into_inner());
    g.entry(device_id).or_default().clone()
}

static LLAMA_MODEL_SWAP: Mutex<()> = Mutex::new(());

fn with_swap_lifecycle_locks<T>(host: Option<u32>, target_dev: u32, swap: impl FnOnce() -> T) -> T {
    let mut devices = vec![target_dev];
    if let Some(host) = host.filter(|host| *host != target_dev) {
        devices.push(host);
        devices.sort_unstable();
    }
    let locks: Vec<_> = devices.into_iter().map(device_lifecycle).collect();
    let _guards: Vec<_> = locks.iter().map(|lock| lock.lock().unwrap_or_else(|p| p.into_inner())).collect();
    swap()
}

/// Replace the llama engine's resident model while the old and target GPU miners are unable to
/// rebuild. Keeping both lifecycle locks through the new load closes the gap where the old miner
/// could reload its model first and make `ensure_loaded` report a cross-GPU busy error.
/// Where the walk reads one canonical tensor's bytes from.
enum GatherSource {
    /// llama's resident device copy, walked in place (zero-dup).
    DevicePtr(u64),
    /// llama's resident host copy — raw GGUF bytes, uploaded to the device.
    HostPtr(u64),
    /// No usable resident copy — bytes come from the host possession index.
    FromIndex,
}

struct GatherPlan {
    /// (nbytes, source) per canonical tensor, name-sorted order.
    entries: Vec<(usize, GatherSource)>,
    total_bytes: u64,
    index_bytes: u64,
    zero_dup: usize,
    host_uploads: usize,
    index_uploads: Vec<String>,
}

impl GatherPlan {
    /// Past this share of the blob, walking a raw canonical copy costs less VRAM than
    /// keeping llama resident plus the uploads.
    fn exceeds_upload_budget(&self) -> bool {
        self.index_bytes * 4 > self.total_bytes
    }
}

/// Match llama's resident tensors against the canonical GGUF list. A canonical tensor is
/// walked from llama only on an unambiguous match — unique name AND exact byte size; a
/// duplicated name, a resized copy or a missing tensor falls back to the possession index,
/// so runtime repacking never changes what the walk reads.
fn plan_canonical_gather(canonical: &[(String, usize)], resident: &[(String, u64, usize, bool)]) -> GatherPlan {
    let mut by_name: HashMap<&str, Vec<&(String, u64, usize, bool)>> = HashMap::with_capacity(resident.len());
    for t in resident {
        by_name.entry(t.0.as_str()).or_default().push(t);
    }
    let mut plan = GatherPlan {
        entries: Vec::with_capacity(canonical.len()),
        total_bytes: 0,
        index_bytes: 0,
        zero_dup: 0,
        host_uploads: 0,
        index_uploads: Vec::new(),
    };
    for (name, nbytes) in canonical {
        plan.total_bytes += *nbytes as u64;
        let source = match by_name.get(name.as_str()).map(Vec::as_slice) {
            Some([t]) if t.2 == *nbytes && t.3 => {
                plan.zero_dup += 1;
                GatherSource::DevicePtr(t.1)
            }
            Some([t]) if t.2 == *nbytes => {
                plan.host_uploads += 1;
                GatherSource::HostPtr(t.1)
            }
            _ => {
                plan.index_bytes += *nbytes as u64;
                plan.index_uploads.push(name.clone());
                GatherSource::FromIndex
            }
        };
        plan.entries.push((*nbytes, source));
    }
    plan
}

/// Canonical (name, nbytes) list of a GGUF in name-sorted order — the layout the possession
/// index chunks and R_T commits.
fn canonical_tensor_list(gguf: &str) -> Option<Vec<(String, usize)>> {
    let mut file = std::fs::File::open(gguf).ok()?;
    let meta = crate::gguf::GgufMeta::read(&mut file).ok()?;
    meta.sorted_names()
        .into_iter()
        .map(|name| {
            let nbytes = usize::try_from(meta.tensors[&name].nbytes).ok()?;
            Some((name, nbytes))
        })
        .collect()
}

/// None when the llama-resident layout can back the canonical walk (with per-tensor index
/// fallback), Some(reason) when the raw canonical copy must be walked instead.
fn llama_gather_blocker(gguf: &str, model_id: &[u8; 32]) -> Option<String> {
    let resident = match crate::llama_engine::tensors() {
        Some(ts) => ts,
        None => return Some("llama engine tensors unavailable".into()),
    };
    let canonical = match canonical_tensor_list(gguf) {
        Some(c) => c,
        None => return Some("canonical GGUF tensor list unreadable".into()),
    };
    let plan = plan_canonical_gather(&canonical, &resident);
    if plan.exceeds_upload_budget() {
        return Some(format!(
            "llama-resident layout too foreign ({} of {} bytes would need a canonical upload)",
            plan.index_bytes, plan.total_bytes
        ));
    }
    if plan.index_bytes > 0 && crate::pom::active_index_for_model(model_id).is_none() {
        return Some("canonical fallback needs the host possession index".into());
    }
    None
}

pub fn load_llama_for_inference(gguf: &str, target_dev: u32) -> Result<u64, crate::llama_engine::LoadError> {
    let _swap_guard = LLAMA_MODEL_SWAP.lock().unwrap_or_else(|p| p.into_inner());
    let host = crate::llama_engine::active_gpu().map(|g| g as u32);
    with_swap_lifecycle_locks(host, target_dev, || {
        if let Some(host) = host {
            uninstall(host);
        }
        if host != Some(target_dev) {
            uninstall(target_dev);
        }
        crate::llama_engine::replace_loaded(gguf, target_dev as usize)
    })
}

/// Ensure the GPU miner is installed; if an inference evicted the mining model, reload it
/// (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when needed —
/// inference has priority, so mining reloads its model when it next gets the GPU. Returns true if
/// the miner is ready to mine.
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if inference_paused() {
        return false;
    }
    if is_installed(device_id) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.fetch_add(1, Ordering::Relaxed);
    let lock = device_lifecycle(device_id);
    let guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    let ok = ensure_installed_inner(device_id, daa);
    drop(guard);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

/// PoM tier index of the mining model at a given block DAA. Recomputed per block (not frozen
/// at index-build time): below the H4 gate it is None, so the miner never claims a tier for a
/// block outside the lineup's era.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// The model a CUDA device currently mines, if assigned.
pub fn mining_model_id(device_id: u32) -> Option<[u8; 32]> {
    mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)
}

/// The CUDA device that mines `model_id` (from the per-GPU tier assignment), if any. Inference for a
/// model is routed to the device that already holds it, so only that GPU pauses mining and the walk
/// can share the resident weights (zero-dup). Returns the lowest matching `device_id` when several
/// GPUs mine the same tier; `None` when no GPU is assigned this model.
pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = mining_tiers().lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// UI helper: current mining-model label by CUDA device id.
/// Returns entries sorted by device id.
pub fn list_mining_model_labels() -> Vec<(u32, String)> {
    let snapshot: Vec<(u32, [u8; 32])> = match mining_tiers().lock() {
        Ok(g) => g.iter().map(|(dev, (id, _))| (*dev, *id)).collect(),
        Err(_) => return Vec::new(),
    };

    let mut out: Vec<(u32, String)> = snapshot
        .into_iter()
        .map(|(dev, model_id)| {
            let label = crate::models::REGISTRY
                .iter()
                .copied()
                .find(|m| m.model_id == model_id)
                .map(|m| m.dir_name.to_string())
                .unwrap_or_else(|| hex::encode(model_id)[..8].to_string());
            (dev, label)
        })
        .collect();
    out.sort_by_key(|(dev, _)| *dev);
    out
}

/// Models that OOM'd when loading on a given GPU: `(device_id, model_id)`. Once banlisted, that GPU
/// never retries that model (avoids a hot-spin reloading a model that doesn't fit); the OOM handler
/// downgrades the GPU to a smaller downloaded tier instead.
static OOM_BANLIST: OnceLock<Mutex<HashSet<(u32, [u8; 32])>>> = OnceLock::new();

fn oom_banlist() -> &'static Mutex<HashSet<(u32, [u8; 32])>> {
    OOM_BANLIST.get_or_init(|| Mutex::new(HashSet::new()))
}

fn is_oom_banlisted(device_id: u32, model_id: &[u8; 32]) -> bool {
    oom_banlist().lock().map(|g| g.contains(&(device_id, *model_id))).unwrap_or(false)
}

fn oom_banlist_add(device_id: u32, model_id: [u8; 32]) {
    if let Ok(mut g) = oom_banlist().lock() {
        g.insert((device_id, model_id));
    }
}

/// After a GPU fails to load its assigned tier (OOM), reassign it to the largest **already-downloaded**
/// PoM model strictly smaller than the failed one that hasn't itself been banlisted on this GPU — so a
/// card whose VRAM estimate was optimistic (driver overhead + KV cache + fragmentation) mines a
/// smaller tier instead of idling. Returns true if a downgrade was applied. No extra prefetch is
/// needed: the candidate set is the served union (a mixed rig already downloaded the smaller tiers).
fn downgrade_after_oom(device_id: u32, failed_model: &[u8; 32], daa: u64) -> bool {
    let Some(failed_tier) = crate::models::pom_tier_index(failed_model, daa) else {
        return false;
    };
    let pick = crate::slm::served_pom_specs()
        .into_iter()
        .filter_map(|s| crate::models::pom_tier_index(&s.model_id, daa).map(|t| (t, s)))
        .filter(|(t, s)| *t < failed_tier && !is_oom_banlisted(device_id, &s.model_id))
        .max_by_key(|(t, _)| *t);
    match pick {
        Some((tier, spec)) => {
            let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            info!("PoM[gpu{}]: OOM on tier {} — downgrading to tier {} ({}).", device_id, failed_tier, tier, spec.name);
            set_mining_tier(device_id, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!("PoM[gpu{}]: OOM and no smaller downloaded tier available — this GPU will not mine PoM (lower the tier flag or add VRAM).", device_id);
            false
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum MinerLoadFailureKind {
    PtxIncompatible,
    OomLikely,
    Other,
}

fn classify_miner_load_error(err: &str) -> MinerLoadFailureKind {
    let s = err.to_ascii_lowercase();
    if s.contains("invalid_ptx")
        || s.contains("invalid ptx")
        || s.contains("ptx") && (s.contains("compatible") || s.contains("no kernel image"))
    {
        return MinerLoadFailureKind::PtxIncompatible;
    }
    if s.contains("out of memory")
        || s.contains("cuda_error_out_of_memory")
        || s.contains("memory allocation")
        || s.contains("alloc") && s.contains("failed")
    {
        return MinerLoadFailureKind::OomLikely;
    }
    MinerLoadFailureKind::Other
}

fn is_transient_gpu_runtime_fault(err: &str) -> bool {
    let s = err.to_ascii_lowercase();
    s.contains("illegal address")
        || s.contains("illegal memory")
        || s.contains("cuda_error_illegal_address")
        || s.contains("invalid device pointer")
        || s.contains("misaligned address")
}

fn reset_stale_gpu_state(device_id: u32, use_llama: bool) {
    // Order matters: the miner walks llama's resident tensors, so it must be released — and any
    // in-flight walk drained — before those tensors are freed.
    uninstall(device_id);
    if use_llama {
        crate::llama_engine::unload_for_gpu(device_id as usize);
    }
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    // Re-check under the per-device lifecycle lock: a worker that queued on the lock before the
    // pause was raised must not reload the mining model while inference is still generating.
    if inference_paused() {
        return false;
    }
    let (model_id, gguf) = match mining_tiers().lock().ok().and_then(|g| g.get(&device_id).cloned()) {
        Some(x) => x,
        None => return false,
    };
    // This GPU's tier at the current block DAA (recomputed per block, H2-gated).
    let tier = match crate::models::pom_tier_index(&model_id, daa) {
        Some(t) => t,
        None => return false,
    };
    if is_oom_banlisted(device_id, &model_id) {
        return false; // this model OOM'd on this GPU before — don't retry (avoids a hot reload spin).
    }
    // Build THIS model's possession index once (host, heavy) — deferred from boot so the pre-PoM
    // legacy phase starts immediately, and keyed by model so a mixed rig builds one index per
    // distinct model it mines (shared across every GPU on it).
    if crate::pom::active_index_for_model(&model_id).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index_for_model(&model_id).is_none() {
            info!("PoM: building host weight index for tier {} (gpu{}) - this can take a while...", tier, device_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf, model_id) {
                Ok(mut idx) => {
                    // Opt-in: hold the full Merkle tree in RAM for lookup-time proof build.
                    if std::env::var("KERYX_RESIDENT_TREE").is_ok_and(|v| v == "1") {
                        let need = idx.n_chunks.saturating_mul(64);
                        let need_gb = need / 1_000_000_000;
                        match crate::pom::available_ram_bytes() {
                            Some(avail) if avail < need + need / 4 => log::warn!(
                                "PoM: KERYX_RESIDENT_TREE set but only ~{} GB RAM available for a ~{} GB tree — keeping frugal path for tier {}",
                                avail / 1_000_000_000, need_gb, tier
                            ),
                            _ => {
                                info!("PoM: building resident tree for tier {} (~{} GB RAM)...", tier, need_gb);
                                idx.build_dense();
                            }
                        }
                    }
                    info!("PoM: tier {} host index ready — N={} chunks", tier, idx.n_chunks);
                    crate::pom::set_index(model_id, idx);
                }
                Err(e) => {
                    log::error!("PoM: host index build failed for tier {} on gpu{}: {}", tier, device_id, e);
                    return false;
                }
            }
        }
    }
    // One CUDA-resident PoM worker per GPU. This avoids all workers contending for a single
    // GPU0-bound miner object while still sharing the host-side index across the process.
    //
    // The in-process llama.cpp engine hosts the model on the inference GPU (a process-global
    // singleton — only that GPU brings it up): there the walk gathers over ITS resident tensors,
    // one VRAM copy serving inference + walk. Every other mining GPU uploads its own standalone
    // copy of the canonical GGUF bytes (`load_raw`). The N-guard below validates the gather
    // against the host index on every path, so a mismatch refuses to mine rather than producing
    // bad proofs. A load OOM surfaces as an Err or, in cudarc, a panic; catch both so the OOM
    // handler can banlist + downgrade instead of crashing the mining thread or hot-spinning on a
    // model that doesn't fit this GPU.
    let inference_gpu = device_for_model(&model_id).unwrap_or(0);
    let mut use_llama = false;
    if device_id == inference_gpu {
        // Only this GPU can serve the model: no engine here means no inference anywhere.
        use_llama = match crate::llama_engine::ensure_loaded(&gguf, device_id as usize) {
            Ok(_) => {
                crate::slm::mark_model_available(&model_id, "llama_engine_loaded");
                true
            }
            // A busy engine hosts another model and is swapped on demand, so the model stays
            // announced: withdrawing here would silence every model but the first on a mixed rig.
            Err(e) if e.is_busy() => false,
            Err(e) => {
                warn!("PoM[gpu{}]: llama engine unavailable — {}", device_id, e);
                let reason = if e.is_oom() { "llama_engine_oom" } else { "llama_engine_load_failed" };
                crate::slm::mark_model_unavailable(&model_id, reason);
                false
            }
        };
    }
    // BYTE-COMPAT GATE: llama.cpp repacks some architectures on load (e.g. tied embeddings
    // materialise the embedding a second time), so its resident layout can differ from the
    // canonical GGUF the walk MUST gather and R_T pins. The gather reconciles that per tensor
    // (affected tensors are uploaded from the possession index); only a layout too foreign to
    // reconcile within the upload budget falls back to a raw canonical copy without llama.
    // OWNERSHIP GATE: the walk dereferences llama's tensor pointers on THIS device. If llama
    // placed them on another card, the launch hits unmapped memory and raises a sticky
    // CUDA_ERROR_ILLEGAL_ADDRESS that poisons the primary context for every user of the device,
    // llama included — the card then loops on rebuilds until the process restarts.
    if use_llama {
        if let Some((name, owner)) = crate::llama_engine::foreign_device_tensor(device_id as usize) {
            warn!(
                "PoM[gpu{}]: llama placed '{}' on device {} — walking a raw canonical copy; inference for this model is unavailable.",
                device_id, name, owner
            );
            crate::llama_engine::unload();
            use_llama = false;
            crate::slm::mark_model_unavailable(&model_id, "llama_wrong_device");
        }
    }
    if use_llama {
        if let Some(reason) = llama_gather_blocker(&gguf, &model_id) {
            warn!(
                "PoM[gpu{}]: {} — walking a raw canonical copy; inference for this model is unavailable.",
                device_id, reason
            );
            crate::llama_engine::unload();
            use_llama = false;
            crate::slm::mark_model_unavailable(&model_id, "llama_layout_incompatible");
        }
    }
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if use_llama {
            info!("PoM[gpu{}]: zero-dup — walking the llama.cpp engine's resident weights", device_id);
            PomGpuMiner::load_llama(&gguf, device_id as usize, &model_id)
        } else {
            PomGpuMiner::load_raw(&gguf, device_id as usize)
        }
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            let e_msg = e.to_string();
            if is_transient_gpu_runtime_fault(&e_msg) {
                log::warn!(
                    "PoM[gpu{}]: transient GPU runtime fault while loading miner ({}); dropping stale miner state and forcing a rebuild on the next cycle.",
                    device_id,
                    e_msg
                );
                reset_stale_gpu_state(device_id, use_llama);
                return false;
            }
            match classify_miner_load_error(&e_msg) {
                MinerLoadFailureKind::PtxIncompatible => {
                    log::error!(
                        "PoM[gpu{}]: PTX incompatibility while loading miner (not OOM): {}. \
                         Check driver/PTX compatibility; skipping OOM downgrade.",
                        device_id,
                        e_msg
                    );
                }
                MinerLoadFailureKind::OomLikely => {
                    log::error!(
                        "PoM[gpu{}]: device miner build failed (OOM likely): {} — banlisting this model and downgrading.",
                        device_id,
                        e_msg
                    );
                    oom_banlist_add(device_id, model_id);
                    downgrade_after_oom(device_id, &model_id, daa);
                }
                MinerLoadFailureKind::Other => {
                    log::error!(
                        "PoM[gpu{}]: device miner build failed (non-OOM): {} — not applying OOM downgrade.",
                        device_id,
                        e_msg
                    );
                }
            }
            return false;
        }
        Err(_) => {
            log::error!("PoM[gpu{}]: device miner load panicked (likely OOM) — banlisting this model and downgrading.", device_id);
            oom_banlist_add(device_id, model_id);
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
    };
    let n = gm.n_chunks();
    // N-guard: the gather must match the host index, else blocks would be rejected.
    if let Some(idx) = crate::pom::active_index_for_model(&model_id) {
        if n != idx.n_chunks {
            log::error!("PoM[gpu{}]: gather N={} != tier {} index N={} — refusing to mine", device_id, n, tier, idx.n_chunks);
            return false;
        }
    }
    install(device_id, gm);
    info!("PoM[gpu{}]: GPU miner ready — N={} chunks resident (matches shared index)", device_id, n);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, ptr: u64, nbytes: usize, dev: bool) -> (String, u64, usize, bool) {
        (name.into(), ptr, nbytes, dev)
    }

    #[test]
    fn canonical_gather_is_zero_dup_on_an_exact_cover() {
        let canonical = vec![("a.weight".into(), 64), ("b.weight".into(), 96)];
        let resident = vec![t("b.weight", 20, 96, true), t("a.weight", 10, 64, true)];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.zero_dup, 2);
        assert_eq!(plan.index_bytes, 0);
        assert!(!plan.exceeds_upload_budget());
        assert!(matches!(plan.entries[0], (64, GatherSource::DevicePtr(10))));
        assert!(matches!(plan.entries[1], (96, GatherSource::DevicePtr(20))));
    }

    #[test]
    fn a_runtime_only_extra_tensor_is_ignored() {
        let canonical = vec![("token_embd.weight".into(), 64)];
        let resident = vec![t("token_embd.weight", 10, 64, true), t("output.weight", 30, 64, true)];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.zero_dup, 1);
        assert_eq!(plan.index_bytes, 0);
    }

    #[test]
    fn a_duplicated_name_falls_back_to_the_index() {
        let canonical = vec![("blk.0.w".into(), 960), ("token_embd.weight".into(), 64)];
        let resident = vec![
            t("token_embd.weight", 10, 64, false),
            t("token_embd.weight", 30, 64, true),
            t("blk.0.w", 20, 960, true),
        ];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.zero_dup, 1);
        assert_eq!(plan.index_uploads, vec!["token_embd.weight".to_string()]);
        assert_eq!(plan.index_bytes, 64);
        assert!(matches!(plan.entries[1], (64, GatherSource::FromIndex)));
        assert!(!plan.exceeds_upload_budget());
    }

    #[test]
    fn a_resized_or_missing_tensor_falls_back_to_the_index() {
        let canonical = vec![("a.w".into(), 640), ("b.w".into(), 64), ("c.w".into(), 64)];
        let resident = vec![t("a.w", 10, 640, true), t("b.w", 20, 96, true)];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.zero_dup, 1);
        assert_eq!(plan.index_uploads, vec!["b.w".to_string(), "c.w".to_string()]);
        assert_eq!(plan.index_bytes, 128);
    }

    #[test]
    fn a_host_resident_tensor_uploads_from_its_pointer() {
        let canonical = vec![("token_embd.weight".into(), 64)];
        let resident = vec![t("token_embd.weight", 10, 64, false)];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.host_uploads, 1);
        assert!(matches!(plan.entries[0], (64, GatherSource::HostPtr(10))));
    }

    #[test]
    fn a_layout_needing_too_many_uploads_exceeds_the_budget() {
        // 2 of 3 equal-size tensors missing → 2/3 of the blob > 1/4 budget.
        let canonical = vec![("a.w".into(), 64), ("b.w".into(), 64), ("c.w".into(), 64)];
        let resident = vec![t("a.w", 10, 64, true)];
        let plan = plan_canonical_gather(&canonical, &resident);
        assert_eq!(plan.index_bytes, 128);
        assert!(plan.exceeds_upload_budget());
    }

    // These exercise `remove_device_entry` directly with a dummy value type, rather than going
    // through `install`/`uninstall`, because `PomGpuMiner` can only be constructed via `load_raw`/
    // `load_llama`, both of which require real CUDA hardware unavailable in
    // CI/unit-test environments. `remove_device_entry` holds the entire scoping logic that
    // `uninstall` delegates to, so this still covers the behavior that matters: only the targeted
    // device's entry is removed, every other device's entry survives untouched.

    #[test]
    fn barrier_waits_for_the_last_walk_to_release_the_miner() {
        use std::sync::mpsc;
        use std::time::Duration;

        let miner = Arc::new("gpu0-miner");
        let held = Arc::clone(&miner);
        let (tx, rx) = mpsc::channel();
        let walker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            drop(held);
            let _ = tx.send(());
        });

        assert!(wait_for_sole_owner(&miner, Duration::from_secs(5)), "must wait, not give up");
        assert_eq!(Arc::strong_count(&miner), 1);
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        walker.join().unwrap();
    }

    #[test]
    fn barrier_gives_up_after_the_deadline_rather_than_hanging() {
        use std::time::Duration;

        let miner = Arc::new("gpu0-miner");
        let _stuck = Arc::clone(&miner);

        assert!(!wait_for_sole_owner(&miner, Duration::from_millis(50)));
    }

    #[test]
    fn remove_device_entry_hands_back_the_removed_miner() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");

        assert_eq!(remove_device_entry(&mut map, 0), Some("gpu0-miner"));
        assert_eq!(remove_device_entry(&mut map, 0), None);
    }

    #[test]
    fn remove_device_entry_only_clears_target_device() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");
        map.insert(1, "gpu1-miner");
        map.insert(2, "gpu2-miner");

        remove_device_entry(&mut map, 0);

        assert!(!map.contains_key(&0));
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
        assert_eq!(map.get(&2), Some(&"gpu2-miner"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn remove_device_entry_on_missing_device_is_a_no_op() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(1, "gpu1-miner");

        remove_device_entry(&mut map, 0);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
    }

    #[test]
    fn detects_transient_illegal_address_faults() {
        assert!(is_transient_gpu_runtime_fault("CUDA_ERROR_ILLEGAL_ADDRESS"));
        assert!(is_transient_gpu_runtime_fault("illegal memory access was encountered"));
        assert!(!is_transient_gpu_runtime_fault("out of memory"));
    }

    #[test]
    fn model_swap_holds_both_gpu_lifecycles_until_replacement_load_finishes() {
        use std::sync::mpsc;
        use std::time::Duration;

        const HOST: u32 = 10_000;
        const TARGET: u32 = 10_001;
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let swap = std::thread::spawn(move || {
            with_swap_lifecycle_locks(Some(HOST), TARGET, || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let mut waiters = Vec::new();
        for device in [HOST, TARGET] {
            let (acquired_tx, acquired_rx) = mpsc::channel();
            let waiter = std::thread::spawn(move || {
                let lock = device_lifecycle(device);
                let _guard = lock.lock().unwrap_or_else(|p| p.into_inner());
                acquired_tx.send(()).unwrap();
            });
            assert!(
                acquired_rx.recv_timeout(Duration::from_millis(50)).is_err(),
                "gpu{device} lifecycle escaped before the replacement model finished loading"
            );
            waiters.push((acquired_rx, waiter));
        }

        release_tx.send(()).unwrap();
        swap.join().unwrap();
        for (acquired_rx, waiter) in waiters {
            acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            waiter.join().unwrap();
        }
    }
}

#[cfg(test)]
impl PomGpuMiner {
    /// Test-only walk source: upload arbitrary chunk-aligned segments (no GGUF, no llama).
    pub(crate) fn load_test_segments(device_id: usize, segments: Vec<Vec<u8>>) -> Result<Self> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for seg in &segments {
            let chunks = (seg.len() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let dev = stream.clone_htod(seg.as_slice())?;
            bases.push(dev.device_ptr(&stream).0 as u64);
            uploads.push(dev);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        let bases_dev = stream.clone_htod(bases.as_slice())?;
        let prefix_dev = stream.clone_htod(prefix.as_slice())?;
        let kernel = select_pom_kernel(device_id)?;
        Ok(Self {
            ctx,
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _uploads: uploads,
        })
    }
}

/// GPU lockstep tests — need a CUDA card: `cargo test --release -- --ignored v3_kernel`.
#[cfg(test)]
mod v3_kernel_tests {
    use super::*;
    use crate::pom_v3;

    const PPH: [u8; 32] = [7u8; 32];
    const TIMESTAMP: u64 = 0x11_2233_4455;

    /// Chunk-aligned but NOT tile-aligned segment cuts — tiles straddle segment boundaries,
    /// exercising the per-chunk gather.
    fn split_blob(blob: &[u8]) -> Vec<Vec<u8>> {
        let cuts = [999 * 32, 5000 * 32, blob.len()];
        let mut segs = Vec::new();
        let mut start = 0;
        for &c in &cuts {
            segs.push(blob[start..c].to_vec());
            start = c;
        }
        segs
    }

    #[test]
    #[ignore]
    fn v3_kernel_matches_host_reference() {
        let blob = pom_v3::lockstep_blob();
        let miner = PomGpuMiner::load_test_segments(0, split_blob(&blob)).unwrap();
        let nonce = 42u64;
        let (states, snippets, final_state) = miner.dump_v3(&PPH, TIMESTAMP, nonce, true, true, true).unwrap();

        let seed = crate::pom::pom_block_seed(&PPH, TIMESTAMP, nonce, true, true, true);
        let (ref_states, ref_snippets, _) = pom_v3::ref_walk(seed, &blob);
        assert_eq!(snippets, ref_snippets, "GPU snippets differ from the host reference");
        assert_eq!(states, ref_states, "GPU states differ from the host reference");

        let d2 = pom_v3::POM_V3_D * pom_v3::POM_V3_D;
        let root = pom_v3::v3_state_root(&ref_states[pom_v3::POM_V3_K * d2..]);
        assert_eq!(final_state, pom_v3::fold64(&root), "GPU blake3 tree differs from the host");
    }

    #[test]
    #[ignore]
    fn v3_grind_end_to_end() {
        let blob = pom_v3::lockstep_blob();
        let miner = PomGpuMiner::load_test_segments(0, split_blob(&blob)).unwrap();
        // Trivial target: every nonce wins, atomicMin returns the batch base.
        let target = [0xFFu8; 32];
        let found = miner.mine(&PPH, TIMESTAMP, &target, 1000, 8, true, true, true, true, true, false).unwrap().unwrap();
        assert_eq!(found, 1000);

        let (states, snippets, final_state) = miner.dump_v3(&PPH, TIMESTAMP, found, true, true, true).unwrap();
        let seed = crate::pom::pom_block_seed(&PPH, TIMESTAMP, found, true, true, true);
        let index = crate::pom::index_from_ram(blob);
        let proof = pom_v3::build_proof_v3(0, &PPH, found, seed, &states, &snippets, &index).unwrap();
        assert_eq!(pom_v3::fold64(&proof.roots[pom_v3::POM_V3_K]), final_state);
        assert!(pom_v3::verify_proof_v3(&PPH, found, seed, &proof, &index.r_t, index.n_chunks));
    }
}
