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
type InfoFn = unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);

const ABI: c_int = 3;

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

/// The attempt id of the currently resident load, if any. Lets a caller tell "still the load I
/// asked for" from "someone swapped the engine underneath me".
pub fn active_attempt() -> Option<u64> {
    engine().lock().ok()?.as_ref().map(|e| e.attempt)
}

struct Engine {
    model: *mut c_void,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
    gpu: usize,
    gguf: String,
    attempt: u64,
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

/// Load the .so + the model once (idempotent, blocking — a model load takes seconds). Returns the
/// attempt id of the resident load, or why it failed. Safe to call from multiple threads.
pub fn ensure_loaded(gguf: &str, gpu: usize) -> Result<u64, LoadError> {
    let attempt = next_attempt();
    let failed = |stage: &'static str, detail: String, cuda_touched: bool| {
        LoadError::new(attempt, stage, detail, cuda_touched)
    };
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
            return Err(failed("busy", format!("engine hosts a model on GPU {} — not stealing it", e.gpu), false));
        }
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
        }
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
        let last_error = sym::<ErrorFn>(lib, "keryx_llama_last_error");
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return Err(failed("path", "GGUF path contains a NUL byte".to_string(), false)),
        };
        log::info!("llama engine: loading {} on GPU {} via {} (in-process, zero-dup)…", gguf, gpu, so.display());
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let model = load(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() {
            let detail = last_error.map_or_else(
                || "model load failed (VRAM? arch?)".to_string(),
                |f| {
                    let msg = CStr::from_ptr(f()).to_string_lossy().into_owned();
                    if msg.is_empty() { "model load failed (VRAM? arch?)".to_string() } else { msg }
                },
            );
            return Err(failed("native_load", detail, true));
        }
        *g = Some(Engine { model, count, info, generate: gen, free, gpu, gguf: gguf.to_string(), attempt });
        log::info!("llama engine: ✓ active — llama.cpp hosts the model + serves OPoI inference.");
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
pub fn unload() {
    if let Ok(mut g) = engine().lock() {
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
        }
    }
}

/// Free the resident model and disable the engine only if the given GPU currently hosts it.
/// This is used for stale-GPU recovery after a transient fault on that specific device.
pub fn unload_for_gpu(gpu: usize) {
    if let Ok(mut g) = engine().lock() {
        if g.as_ref().is_some_and(|e| e.gpu != gpu) {
            return;
        }
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
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
    let n = unsafe { (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
    if n <= 0 {
        return None;
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).ok()
}

#[cfg(test)]
mod tests {
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
}
