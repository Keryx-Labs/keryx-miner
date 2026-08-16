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

use std::collections::HashMap;
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
type TensorDeviceFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
type TensorInfo = (String, u64, usize, bool);

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

    /// The engine is simply hosting another model right now — it is swapped on demand when an
    /// inference request arrives, so this is not an inability to serve.
    pub fn is_busy(&self) -> bool {
        self.stage == "busy"
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
    tensor_device: Option<TensorDeviceFn>,
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
    load(gguf, gpu, false)
}

/// Atomically replace the resident model, including across GPUs. The caller must first drain the
/// hosting GPU's tensor readers. The engine mutex prevents another GPU from claiming the singleton
/// between freeing the old model and loading the replacement.
pub fn replace_loaded(gguf: &str, gpu: usize) -> Result<u64, LoadError> {
    load(gguf, gpu, true)
}

fn load(gguf: &str, gpu: usize, allow_gpu_change: bool) -> Result<u64, LoadError> {
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
        if e.gpu != gpu && !allow_gpu_change {
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
        let tensor_device = sym::<TensorDeviceFn>(lib, "keryx_llama_tensor_device");
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return Err(failed("path", "GGUF path contains a NUL byte".to_string(), false)),
        };
        log::info!("llama engine: loading {} on GPU {} via {} (in-process, zero-dup)…", gguf, gpu, so.display());
        let configured_ctx = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok());
        let n_ctx: c_int = configured_ctx.unwrap_or(4096);
        let mut model = load(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() {
            let mut detail = last_error.map_or_else(
                || "model load failed (VRAM? arch?)".to_string(),
                |f| {
                    let msg = CStr::from_ptr(f()).to_string_lossy().into_owned();
                    if msg.is_empty() { "model load failed (VRAM? arch?)".to_string() } else { msg }
                },
            );
            if context_retry_size(n_ctx, configured_ctx.is_some(), &detail).is_some() {
                log::warn!("llama engine: 4096-token context did not fit; retrying with 1024 tokens");
                model = load(cg.as_ptr(), gpu as c_int, 1024);
                if model.is_null() {
                    detail = last_error.map_or_else(
                        || "model load failed (VRAM? arch?)".to_string(),
                        |f| CStr::from_ptr(f()).to_string_lossy().into_owned(),
                    );
                }
            }
            if model.is_null() {
                return Err(failed("native_load", detail, true));
            }
        }
        *g = Some(Engine { model, count, info, generate: gen, free, tensor_device, gpu, gguf: gguf.to_string(), attempt });
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

/// Restrict llama.cpp's runtime tensor list to tensors that actually exist in the source GGUF,
/// preserving the canonical GGUF name order used by PoM. Runtime-only aliases (for example a
/// materialised `output.weight` for tied embeddings) are valid for inference, but they are not
/// part of the on-disk model committed by R_T and must never enter the possession walk.
fn canonical_tensor_view(resident: &[TensorInfo], canonical: &[(String, usize)]) -> Option<(Vec<TensorInfo>, usize)> {
    let mut by_name: HashMap<&str, &TensorInfo> = HashMap::with_capacity(resident.len());
    for tensor in resident {
        if by_name.insert(tensor.0.as_str(), tensor).is_some() {
            return None;
        }
    }

    let mut out = Vec::with_capacity(canonical.len());
    for (name, expected_nbytes) in canonical {
        let tensor = *by_name.get(name.as_str())?;
        if tensor.2 != *expected_nbytes {
            return None;
        }
        out.push(tensor.clone());
    }

    let ignored = resident.len().saturating_sub(out.len());
    Some((out, ignored))
}

/// Resident tensors in CANONICAL GGUF (name-sorted) order: (name, data_ptr, nbytes, is_device).
///
/// llama.cpp can expose additional runtime-materialised tensors that are not present in the GGUF.
/// We only filter those extras after proving that every canonical GGUF tensor is present with the
/// exact expected byte length. If that proof fails, return the raw runtime list unchanged so the
/// existing PoM N/byte safety gates fail closed rather than silently accepting a changed layout.
pub fn tensors() -> Option<Vec<TensorInfo>> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let n = unsafe { (e.count)(e.model) };
    let mut resident = Vec::with_capacity(n);
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
        resident.push((nm, data as u64, nbytes, is_dev != 0));
    }

    let canonical = (|| -> Option<Vec<(String, usize)>> {
        let mut file = std::fs::File::open(&e.gguf).ok()?;
        let meta = crate::gguf::GgufMeta::read(&mut file).ok()?;
        meta.sorted_names()
            .into_iter()
            .map(|name| {
                let nbytes = usize::try_from(meta.tensors[&name].nbytes).ok()?;
                Some((name, nbytes))
            })
            .collect()
    })();

    let Some(canonical) = canonical else {
        log::warn!(
            "llama engine: could not reconstruct the canonical GGUF tensor list; leaving the runtime tensor view unfiltered for PoM safety gates"
        );
        return Some(resident);
    };

    match canonical_tensor_view(&resident, &canonical) {
        Some((view, ignored)) => {
            if ignored > 0 {
                log::info!(
                    "llama engine: PoM canonical tensor view ignored {} runtime-only tensor(s) materialised by llama.cpp",
                    ignored
                );
            }
            Some(view)
        }
        None => {
            log::warn!(
                "llama engine: runtime tensor view does not exactly cover the GGUF canonical tensors; leaving it unfiltered for PoM safety gates"
            );
            Some(resident)
        }
    }
}

/// First resident tensor whose bytes do NOT live on `expected_gpu`, as (name, owning ordinal).
///
/// The possession walk gathers straight over these pointers: launching the kernel on a device
/// that does not own them dereferences unmapped memory, which raises a sticky
/// CUDA_ERROR_ILLEGAL_ADDRESS and poisons the whole primary context — inference included.
pub fn foreign_device_tensor(expected_gpu: usize) -> Option<(String, i32)> {
    let g = engine().lock().ok()?;
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

        let busy = LoadError::new(4, "busy", "engine hosts a model on GPU 0 — not stealing it", false);
        assert!(busy.is_busy());
        assert!(!busy.is_oom());
        assert!(!LoadError::new(5, "native_load", "out of memory", true).is_busy());

        // Nothing touched the device yet: the context cannot be blamed.
        let missing = LoadError::new(3, "library", "keryx-llama shared library not found", false);
        assert!(!missing.is_oom());
        assert!(!missing.cuda_context_may_be_invalid());
        assert_eq!(missing.to_string(), "library: keryx-llama shared library not found");
    }

    #[test]
    fn retries_default_context_only_after_context_allocation_failure() {
        let context_oom = "context: llama_init_from_model failed [vram: 0 MiB free / 16302 MiB total]";
        assert_eq!(context_retry_size(4096, false, context_oom), Some(1024));
        assert_eq!(context_retry_size(4096, true, context_oom), None);
        assert_eq!(context_retry_size(1024, false, context_oom), None);
        assert_eq!(context_retry_size(4096, false, "model: unsupported architecture"), None);
        assert_eq!(
            context_retry_size(
                4096,
                false,
                "context: llama_init_from_model failed [vram: unavailable (cudaMemGetInfo failed: CUDA_ERROR_INVALID_CONTEXT)]",
            ),
            None,
        );
    }

    #[test]
    fn canonical_tensor_view_ignores_runtime_only_tied_embedding_alias() {
        let resident = vec![
            ("output.weight".to_string(), 30, 32, true),
            ("token_embd.weight".to_string(), 10, 64, true),
            ("blk.0.attn_q.weight".to_string(), 20, 96, true),
        ];
        let canonical = vec![
            ("blk.0.attn_q.weight".to_string(), 96),
            ("token_embd.weight".to_string(), 64),
        ];

        let (view, ignored) = canonical_tensor_view(&resident, &canonical).expect("canonical view");
        assert_eq!(ignored, 1);
        assert_eq!(view.len(), 2);
        assert_eq!(view[0].0, "blk.0.attn_q.weight");
        assert_eq!(view[1].0, "token_embd.weight");
    }

    #[test]
    fn canonical_tensor_view_rejects_missing_canonical_tensor() {
        let resident = vec![("token_embd.weight".to_string(), 10, 64, true)];
        let canonical = vec![
            ("blk.0.attn_q.weight".to_string(), 96),
            ("token_embd.weight".to_string(), 64),
        ];
        assert!(canonical_tensor_view(&resident, &canonical).is_none());
    }

    #[test]
    fn canonical_tensor_view_rejects_size_mismatch() {
        let resident = vec![("token_embd.weight".to_string(), 10, 96, true)];
        let canonical = vec![("token_embd.weight".to_string(), 64)];
        assert!(canonical_tensor_view(&resident, &canonical).is_none());
    }

    #[test]
    fn gemma4_observed_runtime_layout_reduces_to_canonical_pom_chunks() {
        const CHUNK_BYTES: usize = 32;
        const CANONICAL_N: usize = 305_318_656;
        const RESIDENT_N: usize = 331_123_456;

        let canonical_bytes = CANONICAL_N * CHUNK_BYTES;
        let runtime_only_bytes = (RESIDENT_N - CANONICAL_N) * CHUNK_BYTES;
        let resident = vec![
            ("token_embd.weight".to_string(), 0x1000, canonical_bytes, true),
            ("output.weight".to_string(), 0x2000, runtime_only_bytes, true),
        ];
        let canonical = vec![("token_embd.weight".to_string(), canonical_bytes)];

        let resident_n = resident.iter().map(|(_, _, nbytes, _)| nbytes / CHUNK_BYTES).sum::<usize>();
        assert_eq!(resident_n, RESIDENT_N, "fixture must reproduce the Gemma-4 llama-resident N");

        let (view, ignored) = canonical_tensor_view(&resident, &canonical).expect("Gemma-4 canonical view");
        let canonical_n = view.iter().map(|(_, _, nbytes, _)| nbytes / CHUNK_BYTES).sum::<usize>();
        assert_eq!(ignored, 1);
        assert_eq!(canonical_n, CANONICAL_N, "runtime-only Gemma tensor must not alter PoM N");
    }

    #[test]
    #[ignore = "requires one CUDA GPU, keryx-llama shared library, and KERYX_TEST_GEMMA4 GGUF path"]
    fn gemma4_live_layout_and_inference() {
        const CHUNK_BYTES: usize = 32;
        let model = std::env::var("KERYX_TEST_GEMMA4").expect("set KERYX_TEST_GEMMA4 to Gemma-4-12B-abliterated/model.gguf");

        ensure_loaded(&model, 0).expect("Gemma-4 must load in llama.cpp");
        assert!(active_for(&model, 0));

        let pom_view = tensors().expect("Gemma-4 resident tensors");
        let pom_n = pom_view.iter().map(|(_, _, nbytes, _)| nbytes / CHUNK_BYTES).sum::<usize>();

        let mut file = std::fs::File::open(&model).expect("open Gemma-4 GGUF");
        let meta = crate::gguf::GgufMeta::read(&mut file).expect("parse Gemma-4 GGUF");
        let canonical_n = meta
            .sorted_names()
            .into_iter()
            .map(|name| usize::try_from(meta.tensors[&name].nbytes).expect("tensor size") / CHUNK_BYTES)
            .sum::<usize>();

        assert_eq!(pom_n, canonical_n, "Gemma-4 PoM view must match the canonical GGUF layout");
        let text = generate("Reply with only OK.", 16).expect("Gemma-4 OPoI generation");
        assert!(!text.trim().is_empty(), "Gemma-4 generated an empty OPoI response");
        unload();
    }

    #[test]
    #[ignore = "requires two CUDA GPUs, libkeryx-llama, and KERYX_TEST_MODEL_GPU0/GPU1 GGUF paths"]
    fn cross_gpu_replace_moves_the_singleton_without_busy() {
        let gpu0 = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let gpu1 = std::env::var("KERYX_TEST_MODEL_GPU1").expect("set KERYX_TEST_MODEL_GPU1");

        ensure_loaded(&gpu1, 1).unwrap();
        assert!(active_for(&gpu1, 1));
        assert!(generate("Reply with only OK.", 16).is_some());
        replace_loaded(&gpu0, 0).unwrap();
        assert!(active_for(&gpu0, 0));
        assert!(generate("Reply with only OK.", 16).is_some());
        replace_loaded(&gpu1, 1).unwrap();
        assert!(active_for(&gpu1, 1));
        assert!(generate("Reply with only OK.", 16).is_some());
        unload();
    }

    #[test]
    #[ignore = "requires one CUDA GPU, libkeryx-llama, and KERYX_TEST_MODEL_GPU0 GGUF path"]
    fn single_gpu_load_and_generate() {
        let model = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");

        ensure_loaded(&model, 0).unwrap();
        assert!(active_for(&model, 0));
        assert!(generate("Reply with only OK.", 16).is_some());
        unload();
    }
}

fn context_retry_size(n_ctx: c_int, explicitly_configured: bool, detail: &str) -> Option<c_int> {
    (!explicitly_configured
        && n_ctx > 1024
        && detail.contains("context: llama_init_from_model failed")
        && detail.contains(" MiB free / "))
    .then_some(1024)
}
