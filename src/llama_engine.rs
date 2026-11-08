//! In-process llama.cpp engine via a dlopen'd `libkeryx-llama.so`.
//!
//! The .so sits next to the miner binary (or `KERYX_LLAMA_SO` points at it) — `cargo build`
//! produces it there. It is THE inference engine: llama.cpp owns the single resident VRAM copy
//! of each served model on that model's inference GPU (one engine per GPU), the PoM walk
//! gathers straight over its tensor pointers (zero-dup — byte-identity proven by
//! tools/llama_zerodup_spike), and inference text generation
//! runs in-process. Absent .so = no inference (responses are dropped); mining still works via
//! the standalone raw-upload walk (`pom_gpu::load_raw`).
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes and WHO GENERATES the
//! user-facing inference text. The walk kernel, the host possession index, proofs and `tag_fixed` are
//! untouched; `ensure_installed_inner`'s N-guard cross-checks the gather against the host index.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type AbiFn = unsafe extern "C" fn() -> c_int;
type ErrorFn = unsafe extern "C" fn() -> *const c_char;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type ContextInfoFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type CountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type InfoFn = unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type CtxAcquireFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type TensorDeviceFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
type CanonBlobFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> usize;
type WantCanonFn = unsafe extern "C" fn(c_int);
type ProbeDeviceFn = unsafe extern "C" fn(c_int) -> c_int;
type GenTracedFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const i32,
    c_int,
    c_int,
    *const c_char,
    c_int,
    *mut c_char,
    c_int,
    *mut i32,
    c_int,
    *mut c_int,
    *mut c_int,
    c_int,
    c_int,
) -> c_int;
type TraceConfigureFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type TraceLeafCountFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type TraceReadFn = unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut u32, usize, *mut u32) -> c_int;
type ReplayFn = unsafe extern "C" fn(
    *mut c_void,
    *const i32,
    c_int,
    c_int,
    u32,
    u32,
    u32,
    *mut u8,
    usize,
    *mut u8,
    usize,
    *mut u32,
    *mut u32,
) -> c_int;
type TraceTokensFn = unsafe extern "C" fn(*mut c_void, *const i32, c_int) -> c_int;
type ReplayMultiFn = unsafe extern "C" fn(
    *mut c_void,
    *const i32,
    c_int,
    c_int,
    *const u32,
    *const u32,
    u32,
    u32,
    *mut u8,
    usize,
    *mut u32,
) -> c_int;

type SeqConfigureFn = unsafe extern "C" fn(*mut c_void, *const u32, u32, u32, u32, u32, u32) -> c_int;
type SeqRootsFn = unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut u32) -> c_int;
type SeqReplayFn = unsafe extern "C" fn(
    *mut c_void,
    *const i32,
    c_int,
    c_int,
    *const u32,
    *const u64,
    *const u64,
    u32,
    u32,
    *mut u8,
    usize,
    *mut u64,
    *mut u8,
    usize,
    *mut u32,
) -> c_int;

pub(crate) const ABI: c_int = 9;
/// Context kept for the block task while the context pad is resident.
const TASK_CONTEXT_TOKENS: c_int = 64;
/// Largest captured leaf the engine hands back.
const CAPTURE_CAP: usize = 4 << 20;

/// Why a load attempt failed. `stage` says how far it got; the detail carries the engine's own
/// message, including the VRAM figures when CUDA reported them.
#[derive(Debug, Clone)]
pub struct LoadError {
    attempt: u64,
    stage: &'static str,
    detail: String,
    cuda_touched: bool,
}

impl LoadError {
    fn new(attempt: u64, stage: &'static str, detail: impl Into<String>, cuda_touched: bool) -> Self {
        Self { attempt, stage, detail: detail.into(), cuda_touched }
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    /// The card could not fit the model — retrying the same tier on this GPU is pointless.
    pub fn is_oom(&self) -> bool {
        let detail = self.detail.to_ascii_lowercase();
        detail.contains("out of memory") || detail.contains("oom") || detail.contains("cudaerrormemoryallocation")
    }

    /// The failure happened after CUDA was touched, so the device context may be unusable and
    /// the caller should reset its GPU state rather than retry in place.
    pub fn cuda_context_may_be_invalid(&self) -> bool {
        self.cuda_touched && self.detail.to_ascii_lowercase().contains("cuda")
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.detail)
    }
}

impl std::error::Error for LoadError {}

fn next_attempt() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

struct Engine {
    model: *mut c_void,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    generate_traced: GenTracedFn,
    trace_configure: TraceConfigureFn,
    trace_leaf_count: TraceLeafCountFn,
    trace_read: TraceReadFn,
    replay: ReplayFn,
    replay_multi: ReplayMultiFn,
    trace_tokens: TraceTokensFn,
    seq_configure: SeqConfigureFn,
    seq_roots: SeqRootsFn,
    seq_replay: SeqReplayFn,
    ctx_acquire: CtxAcquireFn,
    free: FreeFn,
    tensor_device: Option<TensorDeviceFn>,
    canonical_blob: Option<CanonBlobFn>,
    last_error: Option<ErrorFn>,
    gguf: String,
    attempt: u64,
    /// Context window the engine allocated, in tokens.
    n_ctx: c_int,
    /// Largest full context that last fitted with room for compute buffers.
    ctx_fit: c_int,
    /// Context currently allocated, in tokens (0: none).
    ctx_now: c_int,
    /// The current context is the load-size one (or its floor fallback).
    ctx_full: bool,
    /// Leaves per traced position, 0 until the trace is configured.
    trace_leaves: usize,
    /// BLAKE3 of the configured trace manifest.
    trace_manifest: [u8; 32],
    /// Sequence leaves per response, 0 while sequence leaves are off.
    seq_leaves: usize,
    /// BLAKE3 of the configured sequence layout.
    seq_layout: [u8; 32],
}

/// A traced response: its text and everything needed to commit to and replay it.
#[derive(Debug, Clone, Default)]
pub struct TracedGeneration {
    pub text: String,
    /// Every token fed to the model: the prompt, then each decoded token.
    pub tokens: Vec<i32>,
    pub n_prompt: u32,
    /// Traced positions (`tokens.len()` when the trace kept up).
    pub n_pos: u32,
    pub ubatch: Vec<u32>,
    pub leaves: Vec<[u8; 32]>,
    /// Sequence leaf hashes, when sequence leaves are on.
    pub seq: Option<Vec<[u8; 32]>>,
}

/// Why a generation produced no text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenError {
    /// No engine hosts this GGUF on this GPU.
    Unavailable,
    /// The prompt does not fit the engine's context window.
    PromptTooLong(String),
    /// The prompt head does not tokenize to the expected prefix offset.
    PromptHead(String),
    /// Any other native failure, with the engine's detail when it gave one.
    Failed(String),
}
// The wrapper serializes generation internally; tensor info is read-only after load.
unsafe impl Send for Engine {}

type Slot = Arc<Mutex<Option<Engine>>>;

/// Serializes every call into the native library (load, generate, free). Take it after a slot.
static NATIVE: Mutex<()> = Mutex::new(());

fn native() -> std::sync::MutexGuard<'static, ()> {
    NATIVE.lock().unwrap_or_else(|p| p.into_inner())
}

/// One engine slot per CUDA ordinal.
fn slot(gpu: usize) -> Slot {
    static SLOTS: OnceLock<Mutex<HashMap<usize, Slot>>> = OnceLock::new();
    let mut g = SLOTS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|p| p.into_inner());
    g.entry(gpu).or_default().clone()
}

/// `KERYX_LLAMA_SO=<path>` wins; else the platform-native shared library next to our own
/// executable (`libkeryx-llama.dylib` on macOS, `libkeryx-llama.so` elsewhere).
fn so_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("KERYX_LLAMA_SO") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
        log::warn!("llama engine: KERYX_LLAMA_SO points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    // macOS ships a .dylib (Mach-O). Every other unix (Linux/BSD) ships a .so (ELF). Probe the
    // native name first, and on macOS also fall back to .so — some HiveOS-adjacent tooling may
    // repackage the Linux .so alongside the macOS binary during cross-arch testing.
    #[cfg(target_os = "macos")]
    let candidates: [&str; 2] = ["libkeryx-llama.dylib", "libkeryx-llama.so"];
    #[cfg(target_os = "windows")]
    let candidates: [&str; 1] = ["keryx-llama.dll"];
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let candidates: [&str; 1] = ["libkeryx-llama.so"];
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Option<T> {
    // Symbol<T> derefs to &T; copy the fn pointer out. Sound because the Library is
    // intentionally leaked below (the engine keeps raw fn pointers for its lifetime).
    lib.get::<T>(name.as_bytes()).ok().map(|s| *s)
}

/// Startup probe: is the inference engine library actually usable?
///
/// The engine is only ever dlopened lazily, on the first inference request. A deleted, renamed or
/// stale library therefore leaves PoW/PoM fully working — the possession walk uploads the
/// canonical GGUF itself and never needs this library — while every inference response is silently
/// dropped hours into a session. Resolve the library up front, load it, and check the ABI and
/// every symbol the engine calls. Returns the resolved path, or a human-readable reason.
///
/// Assumes the CUDA runtime probe already passed: this library links cuBLAS/cudart, so a missing
/// CUDA runtime would surface here as a load failure and be misattributed to the engine.
///
/// The probe handle is dropped on return; `ensure_loaded` reloads the library for real later.
pub fn probe_library() -> Result<std::path::PathBuf, String> {
    let Some(so) = so_path() else {
        #[cfg(target_os = "macos")]
        let name = "libkeryx-llama.dylib";
        #[cfg(target_os = "windows")]
        let name = "keryx-llama.dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let name = "libkeryx-llama.so";
        return Err(format!("{} not found next to the miner binary", name));
    };
    let lib = unsafe { libloading::Library::new(&so) }
        .map_err(|e| format!("{} failed to load: {}", so.display(), e))?;
    unsafe {
        let (Some(abi), Some(_load), Some(_count), Some(_info), Some(_gen), Some(_free), Some(_err)) = (
            sym::<AbiFn>(&lib, "keryx_llama_abi"),
            sym::<LoadFn>(&lib, "keryx_llama_load"),
            sym::<CountFn>(&lib, "keryx_llama_tensor_count"),
            sym::<InfoFn>(&lib, "keryx_llama_tensor_info"),
            sym::<GenFn>(&lib, "keryx_llama_generate"),
            sym::<FreeFn>(&lib, "keryx_llama_free"),
            sym::<ErrorFn>(&lib, "keryx_llama_last_error"),
        ) else {
            return Err(format!("{} is missing engine symbols", so.display()));
        };
        let got = abi();
        if got != ABI {
            return Err(format!("{} has ABI {}, this miner expects {}", so.display(), got, ABI));
        }
    }
    Ok(so)
}

/// Child-process body of the per-GPU engine check: loads the library and runs one tiny graph on
/// `gpu`. Exit codes: 0 ok, 40 library without the probe symbol, 41 library load failure, 42 probe
/// failed; a library without kernels for the device aborts the process instead.
pub fn run_device_probe_child(gpu: usize) -> i32 {
    let Some(so) = so_path() else {
        eprintln!("engine probe: library not found next to the miner binary");
        return 41;
    };
    let lib = match unsafe { libloading::Library::new(&so) } {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("engine probe: {} failed to load: {}", so.display(), e);
            return 41;
        }
    };
    let (probe, last_error) = unsafe {
        (sym::<ProbeDeviceFn>(&lib, "keryx_llama_probe_device"), sym::<ErrorFn>(&lib, "keryx_llama_last_error"))
    };
    let Some(probe) = probe else {
        std::mem::forget(lib);
        return 40;
    };
    let rc = unsafe { probe(gpu as c_int) };
    let code = if rc == 0 {
        0
    } else {
        let why = last_error
            .map(|f| unsafe { CStr::from_ptr(f()) }.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("engine probe: gpu {} failed (code {}): {}", gpu, rc, why);
        42
    };
    std::mem::forget(lib);
    code
}

pub enum DeviceProbe {
    Ok,
    Unsupported(String),
    Unknown(String),
}

/// Checks that the engine library can run a kernel on `gpu`, in a child process: ggml aborts the
/// whole process when the library has no kernel image for the device, which must not take the
/// miner down.
pub fn probe_device(gpu: usize) -> DeviceProbe {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => return DeviceProbe::Unknown(format!("current_exe: {}", e)),
    };
    let mut child = match Command::new(&exe)
        .arg("--probe-engine-device")
        .arg(gpu.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return DeviceProbe::Unknown(format!("spawn: {}", e)),
    };
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut s) = stderr {
            let _ = s.read_to_string(&mut text);
        }
        text
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if start.elapsed() > Duration::from_secs(120) => {
                let _ = child.kill();
                let _ = child.wait();
                return DeviceProbe::Unknown("timed out after 120 s".into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return DeviceProbe::Unknown(format!("wait: {}", e)),
        }
    };
    let text = reader.join().unwrap_or_default();
    let tail = {
        let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let keep = lines.len().saturating_sub(3);
        lines.drain(..keep);
        lines.join(" | ")
    };
    match status.code() {
        Some(0) => DeviceProbe::Ok,
        Some(40) => DeviceProbe::Unknown("the library predates the device check".into()),
        Some(41) => DeviceProbe::Unknown(tail),
        Some(code) => DeviceProbe::Unsupported(format!("exit {}: {}", code, tail)),
        None => DeviceProbe::Unsupported(if tail.is_empty() {
            "aborted: no CUDA kernel for this GPU in the library".to_string()
        } else {
            format!("aborted: {}", tail)
        }),
    }
}

/// Load the .so + the model on `gpu` (idempotent, blocking — a model load takes seconds). A
/// different model already on `gpu` is freed first: the caller must have drained that GPU's
/// walk. Returns the attempt id of the resident load, or why it failed.
pub fn ensure_loaded(gguf: &str, gpu: usize) -> Result<u64, LoadError> {
    ensure_loaded_with(gguf, gpu, true)
}

/// [`ensure_loaded`]; `canonical` false keeps llama's own weight layout for a fresh load.
pub fn ensure_loaded_with(gguf: &str, gpu: usize, canonical: bool) -> Result<u64, LoadError> {
    let attempt = next_attempt();
    let failed = |stage: &'static str, detail: String, cuda_touched: bool| {
        LoadError::new(attempt, stage, detail, cuda_touched)
    };
    let slot = slot(gpu);
    let mut g = slot.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(e) = g.as_ref() {
        if e.gguf == gguf {
            return Ok(e.attempt);
        }
    }
    let _native = native();
    if let Some(e) = g.take() {
        unsafe { (e.free)(e.model) };
    }
    let Some(so) = so_path() else {
        return Err(failed("library", "keryx-llama shared library not found".to_string(), false));
    };
    // Never unloaded (the old dlopen path never dlclosed either): the Engine keeps raw fn
    // pointers into the library for the life of the process, so leak it deliberately.
    let lib: &'static libloading::Library = match unsafe { libloading::Library::new(&so) } {
        Ok(l) => Box::leak(Box::new(l)),
        Err(e) => {
            return Err(failed("library", format!("load({}) failed: {}", so.display(), e), false));
        }
    };
    unsafe {
        let (Some(abi), Some(load), Some(count), Some(info), Some(gen), Some(free)) = (
            sym::<AbiFn>(lib, "keryx_llama_abi"),
            sym::<LoadFn>(lib, "keryx_llama_load"),
            sym::<CountFn>(lib, "keryx_llama_tensor_count"),
            sym::<InfoFn>(lib, "keryx_llama_tensor_info"),
            sym::<GenFn>(lib, "keryx_llama_generate"),
            sym::<FreeFn>(lib, "keryx_llama_free"),
        ) else {
            return Err(failed("symbols", format!("{} is missing engine symbols", so.display()), false));
        };
        let got = abi();
        if got != ABI {
            return Err(failed("abi", format!("{} has ABI {}, this miner expects {}", so.display(), got, ABI), false));
        }
        let (Some(generate_traced), Some(trace_configure), Some(trace_leaf_count), Some(trace_read), Some(replay), Some(replay_multi), Some(trace_tokens)) = (
            sym::<GenTracedFn>(lib, "keryx_llama_generate_traced"),
            sym::<TraceConfigureFn>(lib, "keryx_llama_trace_configure"),
            sym::<TraceLeafCountFn>(lib, "keryx_llama_trace_leaf_count"),
            sym::<TraceReadFn>(lib, "keryx_llama_trace_read"),
            sym::<ReplayFn>(lib, "keryx_llama_replay"),
            sym::<ReplayMultiFn>(lib, "keryx_llama_replay_multi"),
            sym::<TraceTokensFn>(lib, "keryx_llama_trace_tokens"),
        ) else {
            return Err(failed("symbols", format!("{} is missing trace symbols", so.display()), false));
        };
        let (Some(seq_configure), Some(seq_roots), Some(seq_replay)) = (
            sym::<SeqConfigureFn>(lib, "keryx_llama_seq_configure"),
            sym::<SeqRootsFn>(lib, "keryx_llama_seq_roots"),
            sym::<SeqReplayFn>(lib, "keryx_llama_seq_replay"),
        ) else {
            return Err(failed("symbols", format!("{} is missing sequence symbols", so.display()), false));
        };
        let Some(ctx_acquire) = sym::<CtxAcquireFn>(lib, "keryx_llama_context_acquire") else {
            return Err(failed("symbols", format!("{} is missing context symbols", so.display()), false));
        };
        let last_error = sym::<ErrorFn>(lib, "keryx_llama_last_error");
        let tensor_device = sym::<TensorDeviceFn>(lib, "keryx_llama_tensor_device");
        let canonical_blob = sym::<CanonBlobFn>(lib, "keryx_llama_canonical_blob");
        if let Some(want) = sym::<WantCanonFn>(lib, "keryx_llama_want_canonical") {
            want(canonical as c_int);
        }
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return Err(failed("path", "GGUF path contains a NUL byte".to_string(), false)),
        };
        log::info!("llama engine: loading {} on GPU {} via {} (in-process, zero-dup)…", gguf, gpu, so.display());
        let (floor, cap) = crate::models::spec_for_gguf(gguf).map_or((4096, 4096), |m| (m.ctx_floor as c_int, m.ctx_cap as c_int));
        let max_ctx = keryx_verifiable_inference::shape::MAX_TRACE_POSITIONS.min(c_int::MAX as u32) as c_int;
        let configured_ctx: Option<c_int> =
            std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse::<c_int>().ok()).map(|c| c.clamp(1, max_ctx));
        let ladder = configured_ctx.map_or_else(|| context_ladder(cap, floor), |c| vec![c]);
        let mut model = std::ptr::null_mut();
        let mut n_ctx: c_int = 0;
        let mut detail = String::new();
        for &candidate in &ladder {
            model = load(cg.as_ptr(), gpu as c_int, candidate);
            if !model.is_null() {
                n_ctx = candidate;
                break;
            }
            detail = last_error.map_or_else(
                || "model load failed (VRAM? arch?)".to_string(),
                |f| {
                    let msg = CStr::from_ptr(f()).to_string_lossy().into_owned();
                    if msg.is_empty() { "model load failed (VRAM? arch?)".to_string() } else { msg }
                },
            );
            if !context_allocation_failed(&detail) {
                break;
            }
            log::warn!("llama engine: {}-token context did not fit ({})", candidate, detail);
        }
        if model.is_null() {
            return Err(failed("native_load", detail, true));
        }
        if n_ctx < floor {
            if configured_ctx.is_none() {
                free(model);
                return Err(failed(
                    "context",
                    format!(
                        "context {} tokens is below the {}-token floor of this model (free VRAM, or set KERYX_LLAMA_CTX to run below the floor knowingly)",
                        n_ctx, floor
                    ),
                    true,
                ));
            }
            log::warn!("llama engine: context {} tokens is below the {}-token floor clients assume for this model", n_ctx, floor);
        }
        let context_info = sym::<ContextInfoFn>(lib, "keryx_llama_context_info")
            .map(|f| CStr::from_ptr(f(model)).to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("n_ctx={}", n_ctx));
        *g = Some(Engine {
            model,
            count,
            info,
            generate: gen,
            generate_traced,
            trace_configure,
            trace_leaf_count,
            trace_read,
            replay,
            replay_multi,
            trace_tokens,
            seq_configure,
            seq_roots,
            seq_replay,
            ctx_acquire,
            free,
            tensor_device,
            canonical_blob,
            last_error,
            gguf: gguf.to_string(),
            attempt,
            n_ctx,
            ctx_fit: n_ctx,
            ctx_now: n_ctx,
            ctx_full: true,
            trace_leaves: 0,
            trace_manifest: [0u8; 32],
            seq_leaves: 0,
            seq_layout: [0u8; 32],
        });
        log::info!("llama engine: ✓ active — llama.cpp hosts the model + serves inference ({}).", context_info);
        Ok(attempt)
    }
}

/// Engine on `gpu` hosts exactly `gguf`?
pub fn active_for(gguf: &str, gpu: usize) -> bool {
    slot(gpu).lock().is_ok_and(|g| g.as_ref().is_some_and(|e| e.gguf == gguf))
}

/// Free the model resident on `gpu`, if any. The caller must have drained that GPU's walk.
pub fn unload(gpu: usize) {
    let slot = slot(gpu);
    let mut g = slot.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(e) = g.take() {
        let _native = native();
        unsafe { (e.free)(e.model) };
    }
}

/// Tensors resident on `gpu` in CANONICAL (name-sorted) order: (name, data_ptr, nbytes, is_device).
pub fn tensors(gpu: usize) -> Option<Vec<(String, u64, usize, bool)>> {
    let slot = slot(gpu);
    let g = slot.lock().ok()?;
    let e = g.as_ref()?;
    let n = unsafe { (e.count)(e.model) };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut name: *const c_char = std::ptr::null();
        let mut data: *mut c_void = std::ptr::null_mut();
        let mut nbytes: usize = 0;
        let mut is_dev: c_int = 0;
        let ok = unsafe { (e.info)(e.model, i, &mut name, &mut data, &mut nbytes, &mut is_dev) };
        if !ok || name.is_null() || data.is_null() {
            return None;
        }
        let nm = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
        out.push((nm, data as u64, nbytes, is_dev != 0));
    }
    Some(out)
}

/// Device base and byte length of the canonical weight stream of the engine on `gpu`, when its
/// weights were loaded in the walk's canonical layout.
pub fn canonical_blob_for(gpu: usize) -> Option<(u64, u64)> {
    let slot = slot(gpu);
    let g = slot.lock().ok()?;
    let e = g.as_ref()?;
    let f = e.canonical_blob?;
    let mut base: *mut c_void = std::ptr::null_mut();
    let n = unsafe { f(e.model, &mut base) };
    (n > 0 && !base.is_null()).then_some((base as u64, n as u64))
}

/// First resident tensor whose bytes do NOT live on `expected_gpu`, as (name, owning ordinal).
///
/// The possession walk gathers straight over these pointers: launching the kernel on a device
/// that does not own them dereferences unmapped memory, which raises a sticky
/// CUDA_ERROR_ILLEGAL_ADDRESS and poisons the whole primary context — inference included.
pub fn foreign_device_tensor(expected_gpu: usize) -> Option<(String, i32)> {
    let slot = slot(expected_gpu);
    let g = slot.lock().ok()?;
    let e = g.as_ref()?;
    let tensor_device = e.tensor_device?;
    let n = unsafe { (e.count)(e.model) };
    for i in 0..n {
        let mut name: *const c_char = std::ptr::null();
        let mut data: *mut c_void = std::ptr::null_mut();
        let mut nbytes: usize = 0;
        let mut is_dev: c_int = 0;
        let ok = unsafe { (e.info)(e.model, i, &mut name, &mut data, &mut nbytes, &mut is_dev) };
        if !ok || name.is_null() || data.is_null() || is_dev == 0 {
            continue;
        }
        let owner = unsafe { tensor_device(e.model, i) };
        if owner >= 0 && owner as usize != expected_gpu {
            let nm = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
            return Some((nm, owner));
        }
    }
    None
}

/// Generate inference text with the engine on `gpu`, which must host `gguf`.
/// Brings back the load-size context, releasing the context pad of `gpu` first.
fn ensure_full_context(e: &mut Engine, gpu: usize, need: usize) -> Result<(), String> {
    if e.ctx_full && e.ctx_now > 0 && e.ctx_now as usize >= need {
        return Ok(());
    }
    crate::pom_gpu::release_pad(gpu as u32);
    let _native = native();
    let floor = crate::models::spec_for_gguf(&e.gguf).map_or(e.n_ctx, |m| (m.ctx_floor as c_int).min(e.n_ctx));
    let min_n = floor.max(need.min(c_int::MAX as usize) as c_int).min(e.n_ctx);
    let start = if min_n > e.ctx_fit { e.n_ctx } else { e.ctx_fit.max(min_n) };
    // Largest size that fits and leaves room for compute buffers; the smallest one only has to fit.
    let mut got = -1;
    let mut fitted: Option<c_int> = None;
    for n in context_ladder(start, min_n) {
        got = unsafe { (e.ctx_acquire)(e.model, n) };
        if got < 0 {
            continue;
        }
        if n == min_n || crate::pom_gpu::free_vram_mib(gpu).is_none_or(|f| f >= COMPUTE_HEADROOM_MIB) {
            break;
        }
        fitted.get_or_insert(n);
    }
    if got < 0 {
        if let Some(n) = fitted {
            got = unsafe { (e.ctx_acquire)(e.model, n) };
        }
    }
    e.ctx_now = got.max(0);
    e.ctx_full = got >= 0;
    if got < 0 {
        return Err(engine_error(e, "context allocation failed"));
    }
    if got < e.n_ctx && got < e.ctx_fit {
        log::warn!("llama engine: {}-token context too large for GPU {} — using {} tokens", e.n_ctx, gpu, got);
    }
    e.ctx_fit = got;
    if (got as usize) < need {
        return Err(format!("llama engine: {}-token context cannot hold {} positions on GPU {}", got, need, gpu));
    }
    log::info!("llama engine: {}-token context restored on GPU {}", got, gpu);
    Ok(())
}

/// A context covering `n_pos` positions: the block-task one when it is enough, else the full one.
fn ensure_context_for(e: &mut Engine, gpu: usize, n_pos: usize) -> Result<(), String> {
    if e.ctx_now > 0 && e.ctx_now as usize >= n_pos {
        return Ok(());
    }
    if n_pos <= TASK_CONTEXT_TOKENS as usize {
        let _native = native();
        let got = unsafe { (e.ctx_acquire)(e.model, TASK_CONTEXT_TOKENS) };
        e.ctx_now = got.max(0);
        e.ctx_full = false;
        if got >= 0 && got as usize >= n_pos {
            return Ok(());
        }
    }
    ensure_full_context(e, gpu, n_pos)
}

/// Shrinks the engine context of `gpu` to the block-task size; a no-op when it already is.
pub fn use_task_context(gpu: usize) {
    let slot = slot(gpu);
    let Ok(mut g) = slot.lock() else { return };
    let Some(e) = g.as_mut() else { return };
    if !e.ctx_full && e.ctx_now > 0 {
        return;
    }
    let _native = native();
    let got = unsafe { (e.ctx_acquire)(e.model, TASK_CONTEXT_TOKENS) };
    e.ctx_now = got.max(0);
    e.ctx_full = false;
    if got < 0 {
        log::warn!("llama engine: {}", engine_error(e, "block-task context allocation failed"));
    } else {
        log::info!("llama engine: {}-token mining context on GPU {}", got, gpu);
    }
}

pub fn generate(gguf: &str, gpu: usize, prompt: &str, max_tokens: usize) -> Result<String, GenError> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| GenError::Unavailable)?;
    let e = g.as_mut().filter(|e| e.gguf == gguf).ok_or(GenError::Unavailable)?;
    ensure_full_context(e, gpu, 0).map_err(GenError::Failed)?;
    let e = &*e;
    let cp = CString::new(prompt).map_err(|_| GenError::Failed("prompt contains a NUL byte".to_string()))?;
    let mut buf = vec![0u8; 64 * 1024];
    let _native = native();
    let n = unsafe { (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
    if n < 0 {
        let detail = e
            .last_error
            .map(|f| unsafe { CStr::from_ptr(f()) }.to_string_lossy().into_owned())
            .unwrap_or_default();
        return Err(if n == -2 { GenError::PromptTooLong(detail) } else { GenError::Failed(detail) });
    }
    buf.truncate(n as usize);
    Ok(generated_text(buf))
}

/// The generated bytes as text, never longer than them: a multi-byte character cut by the token
/// budget is dropped, any other invalid sequence becomes one `?`.
fn generated_text(buf: Vec<u8>) -> String {
    match String::from_utf8(buf) {
        Ok(text) => text,
        Err(e) => {
            let bytes = e.into_bytes();
            let mut text = String::with_capacity(bytes.len());
            let mut chunks = bytes.utf8_chunks().peekable();
            while let Some(chunk) = chunks.next() {
                text.push_str(chunk.valid());
                let bad = chunk.invalid();
                let cut_at_end = chunks.peek().is_none() && is_cut_char(bad);
                if !bad.is_empty() && !cut_at_end {
                    text.push('?');
                }
            }
            text
        }
    }
}

/// Whether `tail` is the start of a multi-byte character cut short.
fn is_cut_char(tail: &[u8]) -> bool {
    let need = match tail.first() {
        Some(b) if b & 0xE0 == 0xC0 => 2,
        Some(b) if b & 0xF0 == 0xE0 => 3,
        Some(b) if b & 0xF8 == 0xF0 => 4,
        _ => return false,
    };
    tail.len() < need && tail[1..].iter().all(|b| b & 0xC0 == 0x80)
}

fn engine_error(e: &Engine, fallback: &str) -> String {
    e.last_error
        .map(|f| unsafe { CStr::from_ptr(f()) }.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

/// Enables the canonical trace on the engine hosting `gguf` on `gpu`; returns the leaves per position.
pub fn trace_configure(gguf: &str, gpu: usize, manifest: &[u8]) -> Result<usize, String> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf).ok_or_else(|| "no engine hosts this model".to_string())?;
    ensure_full_context(e, gpu, 0)?;
    let _native = native();
    let rc = unsafe { (e.trace_configure)(e.model, manifest.as_ptr() as *const c_void, manifest.len(), 0) };
    if rc != 0 {
        return Err(engine_error(e, &format!("trace configuration failed ({rc})")));
    }
    let nl = unsafe { (e.trace_leaf_count)(e.model) };
    if nl <= 0 {
        return Err("trace configured without leaves".to_string());
    }
    e.trace_leaves = nl as usize;
    e.trace_manifest = *blake3::hash(manifest).as_bytes();
    Ok(e.trace_leaves)
}

/// Whether the engine hosting `gguf` on `gpu` traces its responses under `manifest`.
pub fn traced_with(gguf: &str, gpu: usize, manifest: &[u8]) -> bool {
    let h = *blake3::hash(manifest).as_bytes();
    slot(gpu).lock().is_ok_and(|g| g.as_ref().is_some_and(|e| e.gguf == gguf && e.trace_leaves > 0 && e.trace_manifest == h))
}

/// The request-bound prefix of a traced prompt: its token ids, the token count of the prompt
/// head they follow, and the rest of the prompt after them.
pub struct PromptPrefix<'a> {
    pub tokens: &'a [i32],
    pub head_tokens: u32,
    pub rest: &'a str,
}

/// Traced generation: the text, the fed tokens and the leaves of every position. The prompt is
/// padded to `min_prompt` positions and the answer to `answer bytes / answer_bytes_per_position`
/// generated positions (0 = no answer padding). With `prefix`, `prompt` is the head alone.
pub fn generate_traced(
    gguf: &str,
    gpu: usize,
    prompt: &str,
    prefix: Option<PromptPrefix<'_>>,
    max_tokens: usize,
    min_prompt: u32,
    answer_bytes_per_position: u32,
) -> Result<TracedGeneration, GenError> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| GenError::Unavailable)?;
    let e = g.as_mut().filter(|e| e.gguf == gguf && e.trace_leaves > 0).ok_or(GenError::Unavailable)?;
    ensure_full_context(e, gpu, 0).map_err(GenError::Failed)?;
    let e = &*e;
    let cp = CString::new(prompt).map_err(|_| GenError::Failed("prompt contains a NUL byte".to_string()))?;
    let rest = CString::new(prefix.as_ref().map_or("", |p| p.rest)).map_err(|_| GenError::Failed("prompt contains a NUL byte".to_string()))?;
    let (prefix_ptr, n_prefix, head_tokens) = match &prefix {
        Some(p) => (p.tokens.as_ptr(), p.tokens.len() as c_int, p.head_tokens.min(c_int::MAX as u32) as c_int),
        None => (std::ptr::null(), 0, 0),
    };
    let mut buf = vec![0u8; 64 * 1024];
    let mut tokens = vec![0i32; e.n_ctx.max(1) as usize];
    let (mut n_prompt, mut n_fed): (c_int, c_int) = (0, 0);
    let _native = native();
    let n = unsafe {
        (e.generate_traced)(
            e.model,
            cp.as_ptr(),
            prefix_ptr,
            n_prefix,
            head_tokens,
            rest.as_ptr(),
            max_tokens as c_int,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
            tokens.as_mut_ptr(),
            tokens.len() as c_int,
            &mut n_prompt,
            &mut n_fed,
            min_prompt.min(c_int::MAX as u32) as c_int,
            answer_bytes_per_position.min(c_int::MAX as u32) as c_int,
        )
    };
    if n < 0 {
        let detail = engine_error(e, "");
        return Err(match n {
            -2 => GenError::PromptTooLong(detail),
            -8 => GenError::PromptHead(detail),
            _ => GenError::Failed(detail),
        });
    }
    buf.truncate(n as usize);
    let text = generated_text(buf);
    tokens.truncate(n_fed.max(0) as usize);
    let mut n_pos: u32 = 0;
    let mut leaves = vec![[0u8; 32]; tokens.len() * e.trace_leaves];
    let mut ubatch = vec![0u32; tokens.len()];
    let rc = unsafe {
        (e.trace_read)(
            e.model,
            leaves.as_mut_ptr() as *mut u8,
            leaves.len() * 32,
            ubatch.as_mut_ptr(),
            ubatch.len(),
            &mut n_pos,
        )
    };
    if rc != 0 {
        return Err(GenError::Failed(format!("trace read failed ({rc}), {n_pos} positions traced for {} fed", tokens.len())));
    }
    leaves.truncate(n_pos as usize * e.trace_leaves);
    ubatch.truncate(n_pos as usize);
    let seq = if e.seq_leaves > 0 { Some(read_seq_roots(e).map_err(GenError::Failed)?) } else { None };
    Ok(TracedGeneration { text, tokens, n_prompt: n_prompt.max(0) as u32, n_pos, ubatch, leaves, seq })
}

fn read_seq_roots(e: &Engine) -> Result<Vec<[u8; 32]>, String> {
    let mut roots = vec![[0u8; 32]; e.seq_leaves];
    let mut n: u32 = 0;
    let rc = unsafe { (e.seq_roots)(e.model, roots.as_mut_ptr() as *mut u8, roots.len() * 32, &mut n) };
    if rc != 0 || n as usize != e.seq_leaves {
        return Err(engine_error(e, &format!("sequence roots unavailable ({rc})")));
    }
    Ok(roots)
}

/// One layer of the sequence layout: (layer, kind, slot, heads, head_dim).
pub type SeqLayer = [u32; 5];

/// Enables the sequence leaves on the engine hosting `gguf` on `gpu` (`layers` empty: disables
/// them); returns the sequence leaves per response.
pub fn seq_configure(gguf: &str, gpu: usize, layers: &[SeqLayer], n_attn: u32, n_recur: u32, seq_heads: u32, interval: u32) -> Result<usize, String> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf).ok_or_else(|| "no engine hosts this model".to_string())?;
    let mut key = Vec::with_capacity(layers.len() * 20 + 16);
    for v in layers.iter().flatten().chain([n_attn, n_recur, seq_heads, interval].iter()) {
        key.extend_from_slice(&v.to_le_bytes());
    }
    let key = *blake3::hash(&key).as_bytes();
    let n_leaves = if layers.is_empty() { 0 } else { 2 * seq_heads as usize * (n_attn + n_recur) as usize + seq_heads as usize * n_attn as usize };
    if e.seq_layout == key && e.seq_leaves == n_leaves {
        return Ok(n_leaves);
    }
    ensure_full_context(e, gpu, 0)?;
    let _native = native();
    let rc = unsafe { (e.seq_configure)(e.model, layers.as_ptr() as *const u32, layers.len() as u32, n_attn, n_recur, seq_heads, interval) };
    if rc != 0 {
        e.seq_leaves = 0;
        return Err(engine_error(e, &format!("sequence leaf configuration failed ({rc})")));
    }
    e.seq_leaves = n_leaves;
    e.seq_layout = key;
    Ok(n_leaves)
}

/// One captured span of a sequence leaf: its bytes and the chaining value of every chunk of the leaf.
pub struct SeqCapture {
    pub bytes: Vec<u8>,
    pub cvs: Vec<[u8; 32]>,
}

/// Replays a traced response to its end and returns bytes `[lo, hi)` of each `(leaf, lo, hi)` span,
/// with the chunk chaining values of its leaf (at most `max_chunks`).
pub fn seq_replay(gguf: &str, gpu: usize, tokens: &[i32], n_prompt: u32, spans: &[(u32, u64, u64)], max_chunks: u32) -> Result<Vec<SeqCapture>, String> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf && e.trace_leaves > 0 && e.seq_leaves > 0).ok_or_else(|| "no sequenced engine hosts this model".to_string())?;
    ensure_full_context(e, gpu, tokens.len())?;
    let _native = native();
    let stream: Vec<u32> = spans.iter().map(|s| s.0).collect();
    let lo: Vec<u64> = spans.iter().map(|s| s.1).collect();
    let hi: Vec<u64> = spans.iter().map(|s| s.2).collect();
    let total: usize = spans.iter().map(|s| (s.2 - s.1) as usize).sum();
    let mut bytes = vec![0u8; total];
    let mut lens = vec![0u64; spans.len()];
    let mut cvs = vec![[0u8; 32]; spans.len() * max_chunks as usize];
    let mut n_cvs = vec![0u32; spans.len()];
    let rc = unsafe {
        (e.seq_replay)(
            e.model,
            tokens.as_ptr(),
            tokens.len() as c_int,
            n_prompt as c_int,
            stream.as_ptr(),
            lo.as_ptr(),
            hi.as_ptr(),
            spans.len() as u32,
            max_chunks,
            bytes.as_mut_ptr(),
            bytes.len(),
            lens.as_mut_ptr(),
            cvs.as_mut_ptr() as *mut u8,
            cvs.len() * 32,
            n_cvs.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(engine_error(e, &format!("sequence replay failed ({rc})")));
    }
    let (mut bo, mut co) = (0usize, 0usize);
    let mut out = Vec::with_capacity(spans.len());
    for k in 0..spans.len() {
        let (len, n) = (lens[k] as usize, n_cvs[k] as usize);
        out.push(SeqCapture { bytes: bytes[bo..bo + len].to_vec(), cvs: cvs[co..co + n].to_vec() });
        bo += len;
        co += n;
    }
    Ok(out)
}

/// Replays a traced response up to `pos` and returns the bytes of leaves `leaf_x` and `leaf_y` there.
pub fn replay(gguf: &str, gpu: usize, tokens: &[i32], n_prompt: u32, pos: u32, leaf_x: u32, leaf_y: u32) -> Result<(Vec<u8>, Vec<u8>), String> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf && e.trace_leaves > 0).ok_or_else(|| "no traced engine hosts this model".to_string())?;
    ensure_full_context(e, gpu, tokens.len())?;
    let e = &*e;
    let mut x = vec![0u8; CAPTURE_CAP];
    let mut y = vec![0u8; CAPTURE_CAP];
    let (mut x_len, mut y_len): (u32, u32) = (0, 0);
    let _native = native();
    let rc = unsafe {
        (e.replay)(
            e.model,
            tokens.as_ptr(),
            tokens.len() as c_int,
            n_prompt as c_int,
            pos,
            leaf_x,
            leaf_y,
            x.as_mut_ptr(),
            x.len(),
            y.as_mut_ptr(),
            y.len(),
            &mut x_len,
            &mut y_len,
        )
    };
    if rc != 0 {
        return Err(engine_error(e, &format!("replay failed ({rc})")));
    }
    x.truncate(x_len as usize);
    y.truncate(y_len as usize);
    Ok((x, y))
}

/// A traced evaluation of raw tokens: the batch size and leaves of every position.
pub struct TracedTokens {
    pub ubatch: Vec<u32>,
    pub leaves: Vec<[u8; 32]>,
}

/// Traces `tokens` fed as a prompt, with nothing generated.
pub fn trace_tokens(gguf: &str, gpu: usize, tokens: &[i32]) -> Result<TracedTokens, String> {
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf && e.trace_leaves > 0).ok_or_else(|| "no traced engine hosts this model".to_string())?;
    ensure_context_for(e, gpu, tokens.len())?;
    let e = &*e;
    let _native = native();
    let rc = unsafe { (e.trace_tokens)(e.model, tokens.as_ptr(), tokens.len() as c_int) };
    if rc != 0 {
        return Err(engine_error(e, &format!("traced evaluation failed ({rc})")));
    }
    let mut n_pos: u32 = 0;
    let mut leaves = vec![[0u8; 32]; tokens.len() * e.trace_leaves];
    let mut ubatch = vec![0u32; tokens.len()];
    let rc = unsafe { (e.trace_read)(e.model, leaves.as_mut_ptr() as *mut u8, leaves.len() * 32, ubatch.as_mut_ptr(), ubatch.len(), &mut n_pos) };
    if rc != 0 || n_pos as usize != tokens.len() {
        return Err(format!("trace read failed ({rc}), {n_pos} positions traced for {} fed", tokens.len()));
    }
    Ok(TracedTokens { ubatch, leaves })
}

/// Most captures one replay takes.
pub const REPLAY_MAX_CAPTURES: usize = 32;

/// Replays a traced response up to the last requested position and returns the bytes of each
/// `(position, leaf)` capture, in order. `slot_bytes` bounds every captured leaf.
pub fn replay_multi(gguf: &str, gpu: usize, tokens: &[i32], n_prompt: u32, caps: &[(u32, u32)], slot_bytes: usize) -> Result<Vec<Vec<u8>>, String> {
    if caps.is_empty() || caps.len() > REPLAY_MAX_CAPTURES || slot_bytes == 0 || slot_bytes > CAPTURE_CAP || !slot_bytes.is_multiple_of(4) {
        return Err("invalid replay request".to_string());
    }
    let slot = slot(gpu);
    let mut g = slot.lock().map_err(|_| "engine slot poisoned".to_string())?;
    let e = g.as_mut().filter(|e| e.gguf == gguf && e.trace_leaves > 0).ok_or_else(|| "no traced engine hosts this model".to_string())?;
    ensure_full_context(e, gpu, tokens.len())?;
    let e = &*e;
    let pos: Vec<u32> = caps.iter().map(|c| c.0).collect();
    let leaf: Vec<u32> = caps.iter().map(|c| c.1).collect();
    let mut out = vec![0u8; caps.len() * slot_bytes];
    let mut lens = vec![0u32; caps.len()];
    let _native = native();
    let rc = unsafe {
        (e.replay_multi)(
            e.model,
            tokens.as_ptr(),
            tokens.len() as c_int,
            n_prompt as c_int,
            pos.as_ptr(),
            leaf.as_ptr(),
            caps.len() as u32,
            slot_bytes as u32,
            out.as_mut_ptr(),
            out.len(),
            lens.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(engine_error(e, &format!("replay failed ({rc})")));
    }
    Ok(out.chunks(slot_bytes).zip(&lens).map(|(c, &l)| c[..(l as usize).min(slot_bytes)].to_vec()).collect())
}

/// Free VRAM a restored context must leave for compute buffers.
const COMPUTE_HEADROOM_MIB: u64 = 1024;

/// Context sizes to try, largest first: the cap, then halves of it down to the floor.
fn context_ladder(cap: c_int, floor: c_int) -> Vec<c_int> {
    let mut out = Vec::new();
    let mut n = cap.max(floor);
    while n > floor {
        out.push(n);
        n /= 2;
    }
    out.push(floor);
    out
}

/// The load failed while allocating the context, not while loading the weights.
fn context_allocation_failed(detail: &str) -> bool {
    detail.contains("context: llama_init_from_model failed")
}

/// Context window (tokens) of the engine resident on `gpu`, if any.
pub fn context_tokens(gpu: usize) -> Option<u32> {
    slot(gpu).lock().ok().and_then(|g| g.as_ref().map(|e| e.n_ctx as u32))
}

#[cfg(test)]
mod tests {
    #[test]
    fn generated_text_survives_a_cut_or_invalid_byte() {
        assert_eq!(generated_text("héllo".as_bytes().to_vec()), "héllo");
        let mut cut = "ok é".as_bytes().to_vec();
        cut.pop();
        assert_eq!(generated_text(cut), "ok ");
        let mut emoji = "a😀".as_bytes().to_vec();
        emoji.truncate(3);
        assert_eq!(generated_text(emoji), "a");
        assert_eq!(generated_text(vec![b'a', 0xFF, b'b']), "a?b");
        assert_eq!(generated_text(vec![b'a', 0xFF]), "a?");
        for bad in [vec![0x80u8, 0x80, 0x80, 0x80], vec![0xC3, b'x', 0xFF, 0xE2, 0x82]] {
            assert!(generated_text(bad.clone()).len() <= bad.len());
        }
    }

    use super::*;

    #[test]
    fn load_error_distinguishes_oom_from_a_corrupt_context() {
        let oom = LoadError::new(1, "native_load", "cudaMalloc failed: out of memory [vram: 412 MiB free]", true);
        assert!(oom.is_oom());
        assert_eq!(oom.attempt(), 1);

        let broken = LoadError::new(2, "native_load", "model: [cuda: unspecified launch failure]", true);
        assert!(!broken.is_oom());
        assert!(broken.cuda_context_may_be_invalid());

        // Nothing touched the device yet: the context cannot be blamed.
        let missing = LoadError::new(3, "library", "keryx-llama shared library not found", false);
        assert!(!missing.is_oom());
        assert!(!missing.cuda_context_may_be_invalid());
        assert_eq!(missing.to_string(), "library: keryx-llama shared library not found");
    }

    #[test]
    fn context_ladder_runs_from_the_cap_down_to_the_floor() {
        assert_eq!(context_ladder(131_072, 32_768), vec![131_072, 65_536, 32_768]);
        assert_eq!(context_ladder(32_768, 32_768), vec![32_768]);
        assert_eq!(context_ladder(4_096, 8_192), vec![8_192]);
        assert!(context_allocation_failed("context: llama_init_from_model failed [vram: 0 MiB free / 16302 MiB total]"));
        assert!(!context_allocation_failed("model: unsupported architecture"));
    }

    #[test]
    #[ignore = "requires two CUDA GPUs, libkeryx-llama, and KERYX_TEST_MODEL_GPU0/GPU1 GGUF paths"]
    fn two_gpus_keep_their_models_resident_side_by_side() {
        let gpu0 = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let gpu1 = std::env::var("KERYX_TEST_MODEL_GPU1").expect("set KERYX_TEST_MODEL_GPU1");

        let first1 = ensure_loaded(&gpu1, 1).unwrap();
        let first0 = ensure_loaded(&gpu0, 0).unwrap();
        assert!(active_for(&gpu1, 1));
        assert!(active_for(&gpu0, 0));
        assert!(generate(&gpu1, 1, "Reply with only OK.", 16).is_ok());
        assert!(generate(&gpu0, 0, "Reply with only OK.", 16).is_ok());
        assert!(generate(&gpu1, 0, "Reply with only OK.", 16).is_err());
        assert_eq!(ensure_loaded(&gpu1, 1).unwrap(), first1);
        assert_eq!(ensure_loaded(&gpu0, 0).unwrap(), first0);
        unload(0);
        assert!(!active_for(&gpu0, 0));
        assert!(active_for(&gpu1, 1));
        unload(1);
    }

    #[test]
    #[ignore = "requires one CUDA GPU, libkeryx-llama, and KERYX_TEST_MODEL_GPU0 GGUF path"]
    fn single_gpu_load_and_generate() {
        let model = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");

        ensure_loaded(&model, 0).unwrap();
        assert!(active_for(&model, 0));
        assert!(generate(&model, 0, "Reply with only OK.", 16).is_ok());
        unload(0);
    }
}
