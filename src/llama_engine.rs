//! In-process llama.cpp engine via a dlopen'd `libkeryx-llama.so`.
//!
//! The .so sits next to the miner binary (or `KERYX_LLAMA_SO` points at it) — `cargo build`
//! produces it there. It is THE inference engine: llama.cpp owns the single resident VRAM copy
//! of the model on the inference GPU, the PoM walk gathers straight over its tensor pointers
//! (zero-dup — byte-identity proven by tools/llama_zerodup_spike), and OPoI text generation
//! runs in-process. Absent .so = no inference (responses are dropped); mining still works via
//! the standalone raw-upload walk (`pom_gpu::load_raw`).
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes and WHO GENERATES the
//! user-facing OPoI text. The walk kernel, the host possession index, proofs and `tag_fixed` are
//! untouched; `ensure_installed_inner`'s N-guard cross-checks the gather against the host index.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

type AbiFn = unsafe extern "C" fn() -> c_int;
type ErrorFn = unsafe extern "C" fn() -> *const c_char;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type CountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type InfoFn =
    unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);

const ABI: c_int = 3;

#[derive(Clone, Debug)]
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

    pub fn is_oom(&self) -> bool {
        let detail = self.detail.to_ascii_lowercase();
        detail.contains("out of memory")
            || detail.contains("cuda_error_out_of_memory")
            || detail.contains("memory allocation")
    }

    pub fn cuda_context_may_be_invalid(&self) -> bool {
        if !self.cuda_touched {
            return false;
        }
        let detail = self.detail.to_ascii_lowercase();
        detail.contains("illegal address")
            || detail.contains("illegal memory")
            || detail.contains("cuda_error_illegal_address")
            || detail.contains("misaligned address")
            || detail.contains("launch failed")
            || detail.contains("cuda_error_launch_failed")
            || detail.contains("launch timeout")
            || detail.contains("cuda_error_launch_timeout")
            || detail.contains("context is destroyed")
            || detail.contains("cuda_error_context_is_destroyed")
            || detail.contains("device-side assert")
            || detail.contains("cuda_error_assert")
            || detail.contains("hardware stack error")
            || detail.contains("cuda_error_hardware_stack_error")
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "attempt={} stage={}: {}", self.attempt, self.stage, self.detail)
    }
}

impl std::error::Error for LoadError {}

#[derive(Copy, Clone)]
struct Api {
    last_error: ErrorFn,
    load: LoadFn,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
}

struct Engine {
    model: *mut c_void,
    last_error: ErrorFn,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
    attempt: u64,
    gpu: usize,
    gguf: String,
}
// The wrapper serializes generation internally; tensor info is read-only after load.
unsafe impl Send for Engine {}

fn engine() -> &'static Mutex<Option<Engine>> {
    static E: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(None))
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

unsafe fn required_symbol<T: Copy>(
    lib: &libloading::Library,
    path: &std::path::Path,
    name: &'static str,
) -> Result<T, (&'static str, String)> {
    sym::<T>(lib, name).ok_or_else(|| ("symbols", format!("{} is missing required symbol {}", path.display(), name)))
}

unsafe fn resolve_api(lib: &libloading::Library, path: &std::path::Path) -> Result<Api, (&'static str, String)> {
    let abi = required_symbol::<AbiFn>(lib, path, "keryx_llama_abi")?;
    let last_error = required_symbol::<ErrorFn>(lib, path, "keryx_llama_last_error")?;
    let load = required_symbol::<LoadFn>(lib, path, "keryx_llama_load")?;
    let count = required_symbol::<CountFn>(lib, path, "keryx_llama_tensor_count")?;
    let info = required_symbol::<InfoFn>(lib, path, "keryx_llama_tensor_info")?;
    let generate = required_symbol::<GenFn>(lib, path, "keryx_llama_generate")?;
    let free = required_symbol::<FreeFn>(lib, path, "keryx_llama_free")?;
    let got = abi();
    if got != ABI {
        return Err(("abi", format!("{} has ABI {}, this miner expects {}", path.display(), got, ABI)));
    }
    Ok(Api { last_error, load, count, info, generate, free })
}

fn next_attempt() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn model_label(gguf: &str) -> String {
    let path = std::path::Path::new(gguf);
    let name = match path.file_name().and_then(|name| name.to_str()) {
        Some(name) if name.eq_ignore_ascii_case("model.gguf") => path.parent().and_then(|parent| parent.file_name()),
        _ => path.file_stem().or_else(|| path.file_name()),
    };
    name.map(|name| name.to_string_lossy().into_owned()).unwrap_or_else(|| "unknown".to_string())
}

unsafe fn native_error(error: ErrorFn) -> String {
    let ptr = error();
    if ptr.is_null() {
        "native operation failed without details".to_string()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Startup probe: is the inference engine library actually usable?
///
/// The engine is only ever dlopened lazily, on the first inference request. A deleted, renamed or
/// stale library therefore leaves PoW/PoM fully working — the possession walk uploads the
/// canonical GGUF itself and never needs this library — while every OPoI response is silently
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
    let lib =
        unsafe { libloading::Library::new(&so) }.map_err(|e| format!("{} failed to load: {}", so.display(), e))?;
    unsafe { resolve_api(&lib, &so).map_err(|(_, detail)| detail)? };
    Ok(so)
}

/// Capability checks derive from the same complete resolver used by startup and real loads.
pub fn library_available() -> bool {
    probe_library().is_ok()
}

/// Load the .so + the model once (idempotent, blocking — a model load takes seconds). Returns
/// whether the engine is active for `gguf` on `gpu`. Safe to call from multiple threads.
pub fn ensure_loaded(gguf: &str, gpu: usize) -> Result<u64, LoadError> {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.as_ref() {
        if e.gguf == gguf && e.gpu == gpu {
            return Ok(e.attempt);
        }
        // Only a SAME-GPU model swap may free-and-reload: the caller reaches here from
        // `ensure_installed_inner` with its own walk uninstalled. A different GPU must not
        // steal the engine — the hosting GPU's zero-dup walk still gathers over these
        // resident tensors, so freeing them here would be a device use-after-free (and the
        // two GPUs would thrash full model loads stealing the singleton back and forth).
        if e.gpu != gpu {
            let attempt = next_attempt();
            let error = LoadError::new(
                attempt,
                "active_gpu",
                format!("GPU {} already hosts model {}", e.gpu, model_label(&e.gguf)),
                false,
            );
            log::error!(
                "event=llama_load_failed attempt={} gpu={} model={} stage=active_gpu cuda_touched=false detail=\"{}\"",
                attempt,
                gpu,
                model_label(gguf),
                error.detail
            );
            return Err(error);
        }
        if let Some(e) = g.take() {
            log::info!(
                "event=llama_unload_start attempt={} gpu={} model={} reason=\"same_gpu_model_swap\"",
                e.attempt,
                e.gpu,
                model_label(&e.gguf)
            );
            unsafe { (e.free)(e.model) };
            log::info!("event=llama_unload_success attempt={} gpu={}", e.attempt, e.gpu);
        }
    }
    let attempt = next_attempt();
    let model_name = model_label(gguf);
    let failed = |stage, detail, cuda_touched| {
        let error = LoadError::new(attempt, stage, detail, cuda_touched);
        log::error!(
            "event=llama_load_failed attempt={} gpu={} model={} stage={} cuda_touched={} detail=\"{}\"",
            attempt,
            gpu,
            model_name,
            error.stage,
            cuda_touched,
            error.detail
        );
        error
    };
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
        let api = resolve_api(lib, &so).map_err(|(stage, detail)| failed(stage, detail, false))?;
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return Err(failed("path", "GGUF path contains a NUL byte".to_string(), false)),
        };
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        log::info!(
            "event=llama_load_start attempt={} gpu={} model={} abi={} context={} dll={}",
            attempt,
            gpu,
            model_name,
            ABI,
            n_ctx,
            so.display()
        );
        let model = (api.load)(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() {
            return Err(failed("native_load", native_error(api.last_error), true));
        }
        *g = Some(Engine {
            model,
            last_error: api.last_error,
            count: api.count,
            info: api.info,
            generate: api.generate,
            free: api.free,
            attempt,
            gpu,
            gguf: gguf.to_string(),
        });
        log::info!("event=llama_load_success attempt={} gpu={} model={}", attempt, gpu, model_name);
        Ok(attempt)
    }
}

/// Engine active for exactly this (gguf, gpu)?
pub fn active_for(gguf: &str, gpu: usize) -> bool {
    match engine().lock() {
        Ok(g) => g.as_ref().map_or(false, |e| e.gguf == gguf && e.gpu == gpu),
        Err(_) => false,
    }
}

/// The CUDA ordinal hosting the engine's resident model, if the engine is active.
pub fn active_gpu() -> Option<usize> {
    engine().lock().ok()?.as_ref().map(|e| e.gpu)
}

pub fn active_attempt() -> Option<u64> {
    engine().lock().ok()?.as_ref().map(|e| e.attempt)
}

pub fn available() -> bool {
    match engine().lock() {
        Ok(g) => g.is_some(),
        Err(_) => false,
    }
}

/// Free the resident model and disable the engine (available() -> false). Used when swapping
/// the engine to another model (inference request / era crossing), and when llama's resident
/// layout is NOT byte-compatible with the canonical possession index (e.g. repacked tied
/// embeddings) — the walk must gather the canonical GGUF bytes, so we free llama's VRAM and
/// the caller walks a raw canonical upload instead.
pub fn unload(reason: &str) {
    if let Ok(mut g) = engine().lock() {
        if let Some(e) = g.take() {
            log::info!(
                "event=llama_unload_start attempt={} gpu={} model={} reason=\"{}\"",
                e.attempt,
                e.gpu,
                model_label(&e.gguf),
                reason
            );
            unsafe { (e.free)(e.model) };
            log::info!("event=llama_unload_success attempt={} gpu={}", e.attempt, e.gpu);
        }
    }
}

/// Free the resident model and disable the engine only if the given GPU currently hosts it.
/// This is used for stale-GPU recovery after a transient fault on that specific device.
pub fn unload_for_gpu(gpu: usize, reason: &str) {
    if let Ok(mut g) = engine().lock() {
        if g.as_ref().is_some_and(|e| e.gpu != gpu) {
            return;
        }
        if let Some(e) = g.take() {
            log::info!(
                "event=llama_unload_start attempt={} gpu={} model={} reason=\"{}\"",
                e.attempt,
                e.gpu,
                model_label(&e.gguf),
                reason
            );
            unsafe { (e.free)(e.model) };
            log::info!("event=llama_unload_success attempt={} gpu={}", e.attempt, e.gpu);
        }
    }
}

/// Resident tensors in CANONICAL (name-sorted) order: (name, data_ptr, nbytes, is_device).
pub fn tensors() -> Option<Vec<(String, u64, usize, bool)>> {
    let g = engine().lock().ok()?;
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
            log::error!(
                "event=llama_tensor_failed attempt={} gpu={} index={} detail=\"{}\"",
                e.attempt,
                e.gpu,
                i,
                unsafe { native_error(e.last_error) }
            );
            return None;
        }
        let nm = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
        out.push((nm, data as u64, nbytes, is_dev != 0));
    }
    Some(out)
}

/// Generate OPoI text via the in-process engine. None on any failure (caller falls back).
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let cp = CString::new(prompt).ok()?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = unsafe {
        (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int)
    };
    if n <= 0 {
        log::error!("event=llama_generate_failed attempt={} gpu={} detail=\"{}\"", e.attempt, e.gpu, unsafe {
            native_error(e.last_error)
        });
        return None;
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_error_distinguishes_oom_from_corrupt_context() {
        let oom = LoadError::new(7, "native_load", "model_load: CUDA_ERROR_OUT_OF_MEMORY", true);
        let corrupt = LoadError::new(8, "native_load", "decode: CUDA_ERROR_ILLEGAL_ADDRESS", true);

        assert_eq!(oom.to_string(), "attempt=7 stage=native_load: model_load: CUDA_ERROR_OUT_OF_MEMORY");
        assert_eq!(oom.attempt(), 7);
        assert!(oom.is_oom());
        assert!(!oom.cuda_context_may_be_invalid());
        assert!(!corrupt.is_oom());
        assert!(corrupt.cuda_context_may_be_invalid());
    }

    #[test]
    fn model_label_identifies_canonical_and_standalone_ggufs() {
        let canonical = std::path::Path::new("models").join("Qwen3-8B").join("model.gguf");
        let standalone = std::path::Path::new("tmp").join("qwen3-smoke.gguf");

        assert_eq!(model_label(canonical.to_str().unwrap()), "Qwen3-8B");
        assert_eq!(model_label(standalone.to_str().unwrap()), "qwen3-smoke");
    }
}
