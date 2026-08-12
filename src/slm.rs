/// Phase-3 OPoI: model file management + inference dispatch.
///
/// Generation runs in the in-process llama.cpp engine (`llama_engine`, `libkeryx-llama.so`
/// next to the binary): llama.cpp owns the single resident VRAM copy of the model — the PoM
/// walk gathers straight over its tensors — and serves the OPoI text. This module owns the
/// served-lineup state (`ai:cap`), the model downloads, and the per-model chat templates.
/// Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use crate::models::ModelSpec;

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
/// Shared system prompt for the whole lineup (vendor-agnostic wording).
const SYSTEM_PROMPT_NEXT: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention your underlying model name or the company that trained it. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

/// Models the miner currently serves (drives `ai:cap`), set once at startup.
static SUPPORTED_SPECS: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    if let Some(root) = std::env::var_os("KERYX_MODELS_DIR") {
        return std::path::PathBuf::from(root).join(spec.dir_name);
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

/// Path to a model's verified GGUF — normally `<exe_dir>/models/<dir_name>/model.gguf`, but the
/// discovered source when it could not be relocated there. Used by PoM to build the possession
/// weight index from the resident model.
pub fn gguf_path_for(spec: &ModelSpec) -> std::path::PathBuf {
    if let Some(path) = resolved_model_paths().read().unwrap().get(&spec.model_id) {
        return path.clone();
    }
    model_dir(spec).join("model.gguf")
}

/// Downloads `url` to `dest` with automatic resume. A partially downloaded file is
/// continued via an HTTP `Range` request instead of restarting from zero, and both
/// connect-time and mid-stream failures are retried with a fixed backoff. Designed
/// for the huge (10-40 GB) model GGUFs served over the flaky IPFS gateway: the
/// content is immutable (CID-addressed), so appending resumed bytes is always
/// consistent, and an already-complete file (e.g. pre-staged with `wget -c`) is
/// detected via a 416 response and left untouched instead of being re-downloaded.
fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    ui_download_info(&format!("[keryx-miner] Downloading {} ...", url));
    let mut attempt = 0u32;
    let mut last_logged_percent: u64 = 0;
    loop {
        // Resume offset = how many bytes we already have on disk.
        let resume_from = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);

        let mut req = ureq::get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={}-", resume_from));
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(anyhow!("HTTP GET {} failed after {} attempts: {}", url, attempt, e));
                }
                ui_download_warn(&format!(
                    "[keryx-miner] connect error ({e}); retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s (resume @ {} MB)…",
                    resume_from / 1_000_000
                ));
                std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                continue;
            }
        };
        let status = response.status();

        // Decide whether to append (server honored the range) or (re)start, and the total size.
        let (mut file, mut downloaded, total): (std::fs::File, u64, Option<u64>) =
            if resume_from > 0 && status == 206 {
                // Content-Range: "bytes <start>-<end>/<total>"
                let total = response
                    .header("Content-Range")
                    .and_then(|cr| cr.rsplit('/').next())
                    .and_then(|t| t.trim().parse::<u64>().ok());
                let f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(dest)
                    .with_context(|| format!("open append {}", dest.display()))?;
                (f, resume_from, total)
            } else if resume_from > 0 && status == 416 {
                // Range not satisfiable ⇒ the file is already fully downloaded.
                if ui_progress_to_stderr() {
                    eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                } else {
                    ui_download_info(&format!("[keryx-miner] already complete ({} MB).", resume_from / 1_000_000));
                }
                return Ok(());
            } else {
                // 200, or the server ignored Range. Never wipe a local file that already matches
                // the remote size — IPFS gateways often ignore Range and answer 200 + full
                // Content-Length, which previously truncated multi-GB GGUFs back to zero.
                let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
                if resume_from > 0 {
                    if let Some(t) = total {
                        if resume_from >= t {
                            if ui_progress_to_stderr() {
                                eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                            } else {
                                ui_download_info(&format!(
                                    "[keryx-miner] already complete ({} MB).",
                                    resume_from / 1_000_000
                                ));
                            }
                            return Ok(());
                        }
                    }
                    // Partial local file + no Range support: keep the bytes and resume via a
                    // fresh request without Range only when we have nothing useful; otherwise
                    // refuse to truncate and retry later (gateway may regain Range support).
                    if resume_from > 1_000_000 {
                        drop(response);
                        attempt += 1;
                        if attempt >= MAX_ATTEMPTS {
                            return Err(anyhow!(
                                "download {} cannot resume: server ignored Range and local partial is {} MB",
                                url,
                                resume_from / 1_000_000
                            ));
                        }
                        ui_download_warn(&format!(
                            "[keryx-miner] server ignored Range (HTTP {status}); keeping local {} MB, retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s…",
                            resume_from / 1_000_000
                        ));
                        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                        continue;
                    }
                }
                let f = std::fs::File::create(dest)
                    .with_context(|| format!("create {}", dest.display()))?;
                (f, 0u64, total)
            };

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65_536];
        let mut stream_err: Option<String> = None;
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buf[..n]) {
                        stream_err = Some(e.to_string());
                        break;
                    }
                    downloaded += n as u64;
                    if let Some(t) = total {
                        let pct = downloaded * 100 / t.max(1);
                        if ui_progress_to_stderr() {
                            eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                                downloaded as f64 / 1_000_000.0,
                                t as f64 / 1_000_000.0,
                                pct);
                            let _ = std::io::stderr().flush();
                        } else if pct >= last_logged_percent.saturating_add(10) || pct == 100 {
                            last_logged_percent = pct;
                            ui_download_info(&format!(
                                "[keryx-miner] download progress: {:.1}/{:.1} MB ({}%)",
                                downloaded as f64 / 1_000_000.0,
                                t as f64 / 1_000_000.0,
                                pct
                            ));
                        }
                    }
                }
                Err(e) => { stream_err = Some(e.to_string()); break; }
            }
        }
        let _ = file.flush();

        // Done only if the stream ended cleanly AND we reached the known total. An unknown
        // total (chunked IPFS-gateway response with no Content-Length/Content-Range) must NOT
        // count as complete: a clean early EOF would otherwise mark a truncated GGUF as done,
        // write the `.ok` sentinel, and let the miner start on a partial model (failing every
        // challenge). Treat unknown-total as incomplete and retry — a fresh Range request
        // usually returns a parsable Content-Range and self-heals.
        let complete = stream_err.is_none() && matches!(total, Some(t) if downloaded >= t);
        if complete {
            if ui_progress_to_stderr() {
                eprintln!();
            }
            return Ok(());
        }

        attempt += 1;
        if attempt >= MAX_ATTEMPTS {
            return Err(anyhow!("download {} interrupted after {} attempts (got {} MB)",
                url, attempt, downloaded / 1_000_000));
        }
        let why = stream_err.unwrap_or_else(|| "short read".into());
        ui_download_warn(&format!(
            "[keryx-miner] interrupted ({why}); resuming {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s @ {} MB…",
            downloaded / 1_000_000
        ));
        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
    }
}

#[inline]
fn ui_progress_to_stderr() -> bool {
    !std::io::stdout().is_terminal()
}

#[inline]
fn ui_download_info(message: &str) {
    if ui_progress_to_stderr() {
        eprintln!("{}", message);
    } else {
        log::info!("{}", message);
    }
}

#[inline]
fn ui_download_warn(message: &str) {
    if ui_progress_to_stderr() {
        eprintln!("{}", message);
    } else {
        log::warn!("{}", message);
    }
}

fn ipfs_url(cid: &str) -> String {
    format!("{}/ipfs/{}", IPFS_GATEWAY, cid)
}

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // Separate tokenizer.json is optional for the lineup — llama uses the GGUF-embedded one.
    let tok_needed = !spec.tokenizer_cid.is_empty();
    let gguf_ready = crate::gguf::is_complete_file(&gguf);
    let tok_ready = !tok_needed || tok.exists();

    // Reuse a complete on-disk GGUF even when `.ok` was lost (HiveOS upgrades, manual copies,
    // interrupted flag write). Never re-download a model that already parses as complete.
    if gguf_ready && tok_ready {
        verify_gguf(spec, &gguf, &ok_flag)?;
        log::info!("SlmEngine: reusing local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, gguf));
    }

    // Before pulling 16-30 GB over IPFS, look for the content elsewhere under the root: a GGUF
    // unzipped into the wrong directory or renamed is identified by its digest and relocated.
    if !gguf_ready {
        let root = dir.parent().unwrap_or(Path::new(".")).to_path_buf();
        if let Some(path) = discover_model_files(&root, &[registered_spec(spec)?]).remove(&spec.model_id) {
            let marker = if path == gguf { ok_flag.clone() } else { path.with_extension("ok") };
            verify_gguf(spec, &path, &marker)?;
            if tok_needed && !tok.exists() {
                std::fs::create_dir_all(&dir)?;
                download_file(&ipfs_url(spec.tokenizer_cid), &tok)?;
            }
            resolved_model_paths().write().unwrap().insert(spec.model_id, path.clone());
            ui_download_info(&format!("[keryx-miner] Model '{}' ready at {}.", spec.name, path.display()));
            return Ok((tok, path));
        }
    }

    std::fs::create_dir_all(&dir)?;
    if ok_flag.exists() && !gguf_ready {
        let _ = std::fs::remove_file(&ok_flag); // clear stale flag before repairing
    }

    if !gguf_ready {
        ui_download_info(&format!(
            "[keryx-miner] Downloading model '{}' via IPFS. This happens once.",
            spec.name
        ));
        download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
        if !crate::gguf::is_complete_file(&gguf) {
            return Err(anyhow!(
                "model '{}' download finished but GGUF is incomplete at {}",
                spec.name,
                gguf.display()
            ));
        }
    } else {
        ui_download_info(&format!(
            "[keryx-miner] Reusing existing GGUF for '{}' at {}",
            spec.name,
            gguf.display()
        ));
    }

    if tok_needed && !tok.exists() {
        download_file(&ipfs_url(spec.tokenizer_cid), &tok)?;
    }

    verify_gguf(spec, &gguf, &ok_flag)?;
    ui_download_info(&format!("[keryx-miner] Model '{}' ready.", spec.name));
    Ok((tok, gguf))
}

// ── Inference ────────────────────────────────────────────────────────────────

/// Chat-template a raw user prompt for a model by name — llama.cpp's `generate` consumes an
/// already-templated string (a raw prompt makes template-strict models emit EOG immediately,
/// e.g. Qwen3). Each template was validated against the GGUF's embedded chat template.
fn format_prompt_by_name(name: &str, prompt: &str) -> String {
    match name {
        "mistral-7b-v0.3" => format!("[INST] {}\n\n{}[/INST]", SYSTEM_PROMPT_NEXT, prompt),
        // GLM-4-0414 ignores the <|system|> role identity (keeps claiming a foreign vendor) —
        // fold the system prompt into the user turn instead.
        "glm-4-9b-0414" => format!(
            "[gMASK]<sop><|user|>\n{}\n\n{}\n<|assistant|>\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Qwen3 family (tier-0 Qwen3-8B + tier-3 Qwen3.6-27B) — ChatML + a pre-filled empty think
        // block so the visible answer starts immediately (an open think block would eat the whole
        // max_tokens budget).
        "qwen3.6-27b" | "qwen3-8b-abliterated" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        "kimi-linear-48b" => format!(
            "<|im_system|>system<|im_middle|>{}<|im_end|>\
             <|im_user|>user<|im_middle|>{}<|im_end|>\
             <|im_assistant|>assistant<|im_middle|>",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Generic ChatML fallback (unreachable for the registered lineup).
        _ => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Register the set of models this miner currently serves (drives `ai:cap`).
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    *SUPPORTED_SPECS.write().unwrap() = specs;
}

/// Outcome of the startup GPU inference probe.
pub enum GpuProbe {
    /// CUDA + cuBLAS present — GPU inference is available.
    Ok,
    /// No CUDA device present — cannot mine (inference is GPU-only).
    NoCuda,
    /// A CUDA device exists but cuBLAS could not be loaded — GPU inference is impossible.
    CublasMissing,
    /// CUDA is fine but the llama engine library is missing or unusable — carries the reason.
    /// Not auto-recoverable: the library ships with the release.
    EngineMissing(String),
}

/// Verify that GPU inference can actually work *before* mining starts.
///
/// The in-process llama engine and its cuBLAS dependency are both dlopened lazily on the first
/// load; discovering either of them missing mid-challenge would silently drop responses while
/// mining kept running — the possession walk has no dependency on them. Probe all three
/// prerequisites up front: a usable CUDA device (driver), a loadable cuBLAS, and the engine
/// library itself, and report a clean, actionable result.
pub fn probe_gpu_inference() -> GpuProbe {
    if crate::pom_gpu::query_all_gpus_vram().is_empty() {
        return GpuProbe::NoCuda;
    }
    // The binary links CUDA 12; probe the versioned soname first, then the generic one.
    // The in-process llama engine needs the CUDA runtime on top of cuBLAS, and a missing
    // libcudart would otherwise only surface at the first engine load, mid-session. On
    // Windows nvcc links cudart statically into keryx-llama.dll, so only cuBLAS is probed.
    // Each probe Library is dropped immediately — we only care that it CAN load; the
    // engine (re-)loads it for real later, and the OS loader refcounts it.
    let loads = |candidates: &[&str]| candidates.iter().any(|so| unsafe { libloading::Library::new(so) }.is_ok());
    #[cfg(windows)]
    let ok = loads(&["cublas64_12.dll"]);
    #[cfg(not(windows))]
    let ok = loads(&["libcublas.so.12", "libcublas.so"]) && loads(&["libcudart.so.12", "libcudart.so"]);
    if !ok {
        return GpuProbe::CublasMissing;
    }
    // Checked last, and only once the CUDA runtime is known good: the engine library links
    // cuBLAS/cudart, so probing it earlier would report a missing CUDA runtime as a broken engine.
    if let Err(why) = crate::llama_engine::probe_library() {
        return GpuProbe::EngineMissing(why);
    }
    GpuProbe::Ok
}

/// Pre-download all registered model files before mining starts.
///
/// Does not load weights into GPU memory — just ensures files are on disk so
/// the first inference request doesn't stall the mining workers mid-session.
/// Returns Err if any model fails to download; mining must not start in that case.
pub fn prefetch_models(specs: &'static [&'static ModelSpec]) -> Result<()> {
    for spec in specs {
        log::debug!("SlmEngine: prefetching model '{}'…", spec.name);
        let result = ensure_gguf(spec).map(|_| ());
        match result {
            Ok(()) => log::debug!("SlmEngine: '{}' files ready.", spec.name),
            Err(e) => {
                log::error!("SlmEngine: prefetch '{}' failed: {} — cannot start mining.", spec.name, e);
                return Err(e);
            }
        }
    }
    Ok(())
}

fn unavailable_models() -> &'static RwLock<HashSet<[u8; 32]>> {
    static MODELS: OnceLock<RwLock<HashSet<[u8; 32]>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Withdraw a model from `ai:cap`: the files are on disk but this miner cannot serve it right
/// now. Announcing it anyway earns assigned requests it cannot answer, hence service-bond strikes.
pub fn mark_model_unavailable(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().insert(*model_id) {
        log::warn!("SlmEngine: model {:.8} withdrawn from ai:cap ({})", hex::encode(model_id), reason);
    }
}

/// Re-announce a model after it serves again.
pub fn mark_model_available(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().remove(model_id) {
        log::info!("SlmEngine: model {:.8} back in ai:cap ({})", hex::encode(model_id), reason);
    }
}

/// The registry entry that owns this model_id — the caller's spec must not decide where a model
/// lives, only the registered lineup does.
fn registered_spec(spec: &ModelSpec) -> Result<&'static ModelSpec> {
    crate::models::REGISTRY
        .iter()
        .copied()
        .find(|known| known.model_id == spec.model_id)
        .ok_or_else(|| anyhow!("model '{}' is not registered", spec.name))
}

fn model_is_unavailable(model_id: &[u8; 32]) -> bool {
    unavailable_models().read().unwrap().contains(model_id)
}

/// GGUFs whose UnixFS digest was checked against the pinned `model_id` in this process.
fn verified_models() -> &'static RwLock<HashSet<[u8; 32]>> {
    static MODELS: OnceLock<RwLock<HashSet<[u8; 32]>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Where a model's verified GGUF actually lives, when it is not the canonical path (discovered
/// elsewhere under the model root and impossible to relocate).
fn resolved_model_paths() -> &'static RwLock<HashMap<[u8; 32], PathBuf>> {
    static PATHS: OnceLock<RwLock<HashMap<[u8; 32], PathBuf>>> = OnceLock::new();
    PATHS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// A marker holds the hex digest of the GGUF it certifies; anything else means "unverified".
fn marker_certifies(ok_flag: &Path, model_id: &[u8; 32]) -> bool {
    std::fs::read_to_string(ok_flag).map(|s| s.trim().eq_ignore_ascii_case(&hex::encode(model_id))).unwrap_or(false)
}

fn verify_model_file(gguf: &Path, ok_flag: &Path, expected: [u8; 32], name: &str) -> Result<()> {
    // A marker never certifies a model until the complete current file matches its pinned CID.
    let _ = std::fs::remove_file(ok_flag);
    let mut next_percent = 10u64;
    let digest = crate::integrity::unixfs_v0_digest_file(gguf, |done, total| {
        if total == 0 {
            return;
        }
        let percent = done.saturating_mul(100) / total;
        if percent >= next_percent {
            ui_download_info(&format!("[keryx-miner] Verifying '{}': {}%", name, percent.min(100)));
            next_percent = (percent / 10 + 1) * 10;
        }
    })
    .with_context(|| format!("verify IPFS identity for model '{}' at {}", name, gguf.display()))?;

    if digest != expected {
        return Err(anyhow!(
            "model '{}' IPFS CID digest mismatch at {} (expected {}, got {})",
            name,
            gguf.display(),
            hex::encode(expected),
            hex::encode(digest)
        ));
    }

    std::fs::write(ok_flag, hex::encode(expected))
        .with_context(|| format!("write verified .ok flag {}", ok_flag.display()))?;
    Ok(())
}

/// Verify unless the marker already certifies this exact model, and keep the served-lineup
/// registries in step with the outcome.
fn verify_gguf(spec: &ModelSpec, gguf: &Path, ok_flag: &Path) -> Result<()> {
    if marker_certifies(ok_flag, &spec.model_id) {
        verified_models().write().unwrap().insert(spec.model_id);
        return Ok(());
    }
    verified_models().write().unwrap().remove(&spec.model_id);
    ui_download_info(&format!("[keryx-miner] Verifying model '{}' integrity before mining...", spec.name));
    if let Err(error) = verify_model_file(gguf, ok_flag, spec.model_id, spec.name) {
        mark_model_unavailable(&spec.model_id, "integrity_mismatch");
        return Err(error);
    }
    verified_models().write().unwrap().insert(spec.model_id);
    mark_model_available(&spec.model_id, "integrity_verified");
    ui_download_info(&format!("[keryx-miner] Model '{}' integrity verified.", spec.name));
    Ok(())
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn rename_no_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    nix::fcntl::renameat2(None, source, None, destination, nix::fcntl::RenameFlags::RENAME_NOREPLACE)
        .map_err(std::io::Error::from)
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn rename_no_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    if destination.exists() {
        return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "destination exists"));
    }
    std::fs::rename(source, destination)
}

/// Identify GGUFs sitting anywhere under the model root by their content, so a file that was
/// unzipped into the wrong directory (or renamed) is used instead of re-downloading 16-30 GB.
/// Symlinks are never followed; an occupied canonical path is never overwritten.
fn discover_model_files(root: &Path, wanted: &[&'static ModelSpec]) -> HashMap<[u8; 32], PathBuf> {
    const MAX_DEPTH: usize = 3;
    let mut found: HashMap<[u8; 32], PathBuf> = HashMap::new();
    let mut queue = vec![(root.to_path_buf(), 0usize)];

    while let Some((dir, depth)) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                if depth < MAX_DEPTH {
                    queue.push((path, depth + 1));
                }
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            // A canonical, already-certified file needs no hashing.
            if wanted.iter().any(|spec| {
                path == root.join(spec.dir_name).join("model.gguf")
                    && marker_certifies(&root.join(spec.dir_name).join(".ok"), &spec.model_id)
            }) {
                continue;
            }
            if !crate::gguf::is_complete_file(&path) {
                continue;
            }
            let Ok(digest) = crate::integrity::unixfs_v0_digest_file(&path, |_, _| {}) else { continue };
            let Some(spec) = wanted.iter().find(|spec| spec.model_id == digest) else { continue };
            if found.contains_key(&digest) {
                continue;
            }

            let canonical = root.join(spec.dir_name).join("model.gguf");
            if path == canonical {
                found.insert(digest, path);
                continue;
            }
            let _ = std::fs::create_dir_all(root.join(spec.dir_name));
            match rename_no_replace(&path, &canonical) {
                Ok(()) => {
                    log::info!("SlmEngine: found '{}' at {} — moved to {}", spec.name, path.display(), canonical.display());
                    found.insert(digest, canonical);
                }
                Err(e) => {
                    log::info!("SlmEngine: found '{}' at {} — using it in place ({})", spec.name, path.display(), e);
                    found.insert(digest, path);
                }
            }
        }
    }
    found
}

/// Return the model_ids of supported models that have fully-downloaded files (.ok flag present)
/// and are not currently withdrawn.
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs.iter()
        .filter(|s| model_dir(s).join(".ok").exists() && !model_is_unavailable(&s.model_id))
        .map(|s| s.model_id)
        .collect()
}

/// Downloaded (`.ok`) PoM model specs — the OOM-downgrade candidate set when a GPU can't hold its
/// assigned tier. Restricting to already-downloaded models means a downgrade needs no extra prefetch
/// (a mixed rig already pulled the smaller tiers for its smaller cards).
pub fn served_pom_specs() -> Vec<&'static ModelSpec> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs
        .iter()
        .copied()
        .filter(|s| crate::models::is_pom_model(&s.model_id) && model_dir(s).join(".ok").exists())
        .collect()
}

/// True only when the model is supported, its files are completely downloaded, and it is not
/// currently withdrawn from `ai:cap`.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists() && !model_is_unavailable(model_id)
}

/// Serve an inference request via the in-process llama.cpp engine, swapping it to the requested
/// model first if it hosts a different one. Blocking — call from `spawn_blocking`.
///
/// The generated text is user-facing only — consensus checks the fixed-point `model_fixed`
/// commitment separately. A failed load/generation returns None (the response is dropped, never
/// submitted): a miner must not be rewarded for garbage.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // llama.cpp gets the raw tokens of whatever string we pass — apply the model's chat
    // template here (template-strict models emit EOG immediately on a bare prompt).
    let templated = format_prompt_by_name(spec.name, prompt);
    // Route inference to the device that MINES this model (per-GPU tier assignment): only that
    // GPU pauses PoW and the walk shares the resident weights (zero-dup). Falls back to device 0
    // (single-GPU / unassigned model).
    let dev_id = crate::pom_gpu::device_for_model(model_id).unwrap_or(0);
    let gguf = gguf_path_for(spec).to_string_lossy().into_owned();

    if !crate::llama_engine::active_for(&gguf, dev_id as usize) {
        // The engine hosts another model (or nothing). Inference has priority: release the
        // device's miner to make room, swap the engine to the requested model. The possession
        // walk rebuilds over the mining model at the next `ensure_installed`.
        log::info!("SlmEngine: swapping the llama engine to '{}' (gpu{})", spec.name, dev_id);
        // The hosted model may live on ANOTHER device whose walk reads its tensors zero-dup:
        // evict drains that device (installed walk AND in-flight build) before freeing anything.
        // Draining only `dev_id` here poisoned the hosting GPU on every two-model rig.
        crate::pom_gpu::evict_llama_host_for_swap(dev_id);
        if let Err(e) = crate::llama_engine::ensure_loaded(&gguf, dev_id as usize) {
            log::error!("SlmEngine: cannot load '{}' — {}; response dropped", spec.name, e);
            mark_model_unavailable(model_id, if e.is_oom() { "llama_load_oom" } else { "llama_load_failed" });
            return None;
        }
    }

    match crate::llama_engine::generate(&templated, max_tokens) {
        Some(text) if !text.trim().is_empty() => {
            mark_model_available(model_id, "generation_success");
            Some(text)
        }
        _ => {
            log::warn!("SlmEngine '{}': llama generate failed or empty — response dropped", spec.name);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_gguf(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(32, 0);
        std::fs::write(path, bytes).unwrap();
        assert!(crate::gguf::is_complete_file(path));
    }

    fn test_spec(model_id: [u8; 32], dir_name: &'static str) -> ModelSpec {
        ModelSpec {
            name: "test-model",
            model_id,
            format: crate::models::ModelFormat::Gguf,
            tokenizer_cid: "",
            weight_cids: &["unused"],
            dir_name,
            min_vram_mb: 0,
        }
    }

    #[test]
    fn a_marker_certifies_only_its_own_digest() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join(".ok");
        let model_id = [0x11u8; 32];

        std::fs::write(&marker, b"").unwrap();
        assert!(!marker_certifies(&marker, &model_id), "an empty legacy marker must not certify");

        std::fs::write(&marker, hex::encode([0x22u8; 32])).unwrap();
        assert!(!marker_certifies(&marker, &model_id));

        std::fs::write(&marker, hex::encode(model_id).to_uppercase()).unwrap();
        assert!(marker_certifies(&marker, &model_id));
    }

    #[test]
    fn discovery_relocates_a_misnamed_nested_gguf() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("downloads").join("Qwen-wrong-name");
        std::fs::create_dir_all(&nested).unwrap();
        let source = nested.join("weights.gguf");
        write_minimal_gguf(&source);
        let model_id = crate::integrity::unixfs_v0_digest_file(&source, |_, _| {}).unwrap();
        let spec: &'static ModelSpec = Box::leak(Box::new(test_spec(model_id, "Expected")));

        let found = discover_model_files(root.path(), &[spec]);

        let canonical = root.path().join("Expected").join("model.gguf");
        assert_eq!(found.get(&model_id), Some(&canonical));
        assert!(canonical.exists());
        assert!(!source.exists(), "the source must have been moved, not copied");
    }

    #[test]
    fn discovery_never_overwrites_an_occupied_canonical_path() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("stray.gguf");
        write_minimal_gguf(&source);
        let model_id = crate::integrity::unixfs_v0_digest_file(&source, |_, _| {}).unwrap();
        let spec: &'static ModelSpec = Box::leak(Box::new(test_spec(model_id, "Expected")));
        let canonical = root.path().join("Expected").join("model.gguf");
        std::fs::create_dir_all(canonical.parent().unwrap()).unwrap();
        std::fs::write(&canonical, b"keep me").unwrap();

        let found = discover_model_files(root.path(), &[spec]);

        assert_eq!(found.get(&model_id), Some(&source), "the stray file is used in place");
        assert_eq!(std::fs::read(&canonical).unwrap(), b"keep me");
    }

    #[test]
    fn discovery_ignores_content_no_spec_claims() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("unknown.gguf");
        write_minimal_gguf(&source);
        let spec: &'static ModelSpec = Box::leak(Box::new(test_spec([0x99u8; 32], "Expected")));

        assert!(discover_model_files(root.path(), &[spec]).is_empty());
        assert!(source.exists());
    }

    #[test]
    fn discovery_does_not_follow_symlinked_directories() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let source = outside.path().join("model.gguf");
        write_minimal_gguf(&source);
        let model_id = crate::integrity::unixfs_v0_digest_file(&source, |_, _| {}).unwrap();
        let spec: &'static ModelSpec = Box::leak(Box::new(test_spec(model_id, "Expected")));
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();

        assert!(discover_model_files(root.path(), &[spec]).is_empty());
        assert!(source.exists());
    }

    #[test]
    fn withdrawn_models_are_hidden_until_they_recover() {
        let model_id = [0xa7u8; 32];
        assert!(!model_is_unavailable(&model_id));

        mark_model_unavailable(&model_id, "test_failure");
        assert!(model_is_unavailable(&model_id));
        assert!(!is_model_ready(&model_id));
        assert!(!loaded_model_ids().contains(&model_id));

        mark_model_available(&model_id, "test_recovery");
        assert!(!model_is_unavailable(&model_id));
    }

    #[test]
    fn withdrawal_is_idempotent_per_model() {
        let model_id = [0xb3u8; 32];
        mark_model_unavailable(&model_id, "first");
        mark_model_unavailable(&model_id, "second");
        assert!(model_is_unavailable(&model_id));

        mark_model_available(&model_id, "recovered");
        mark_model_available(&model_id, "recovered_again");
        assert!(!model_is_unavailable(&model_id));

        let other = [0xc1u8; 32];
        mark_model_unavailable(&model_id, "again");
        assert!(!model_is_unavailable(&other));
    }
}
