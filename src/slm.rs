/// Phase-3 OPoI: model file management + inference dispatch.
///
/// Generation runs in the in-process llama.cpp engine (`llama_engine`, `libkeryx-llama.so`
/// next to the binary): llama.cpp owns the single resident VRAM copy of the model — the PoM
/// walk gathers straight over its tensors — and serves the OPoI text. This module owns the
/// served-lineup state (`ai:cap`), the model downloads, and the per-model chat templates.
/// Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
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

fn unavailable_models() -> &'static RwLock<HashSet<[u8; 32]>> {
    static MODELS: OnceLock<RwLock<HashSet<[u8; 32]>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(HashSet::new()))
}

fn verified_models() -> &'static RwLock<HashSet<[u8; 32]>> {
    static MODELS: OnceLock<RwLock<HashSet<[u8; 32]>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(HashSet::new()))
}

fn model_is_verified(model_id: &[u8; 32]) -> bool {
    verified_models().read().unwrap().contains(model_id)
}

pub fn mark_model_unavailable(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().insert(*model_id) {
        log::warn!(
            "event=ai_capability_withdrawn model={} reason=\"{}\"",
            hex::encode(&model_id[..4]),
            reason
        );
    }
}

pub fn mark_model_available(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().remove(model_id) {
        log::info!(
            "event=ai_capability_restored model={} reason=\"{}\"",
            hex::encode(&model_id[..4]),
            reason
        );
    }
}

fn model_is_unavailable(model_id: &[u8; 32]) -> bool {
    unavailable_models().read().unwrap().contains(model_id)
}

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

/// Path to a model's GGUF file (`<exe_dir>/models/<dir_name>/model.gguf`). Used by PoM to
/// build the possession weight index from the resident model.
pub fn gguf_path_for(spec: &ModelSpec) -> std::path::PathBuf {
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

fn verify_model_file(
    gguf: &std::path::Path,
    ok_flag: &std::path::Path,
    expected: [u8; 32],
    name: &str,
) -> Result<()> {
    // A marker never represents a model until the complete current file matches its pinned CID.
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
        let _ = std::fs::remove_file(ok_flag);
        return Err(anyhow!(
            "model '{}' IPFS CID digest mismatch at {} (expected {}, got {}); replace the GGUF before mining",
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

fn verify_gguf(spec: &ModelSpec, gguf: &std::path::Path, ok_flag: &std::path::Path) -> Result<()> {
    verified_models().write().unwrap().remove(&spec.model_id);
    ui_download_info(&format!(
        "[keryx-miner] Verifying model '{}' integrity before mining...",
        spec.name
    ));
    if let Err(error) = verify_model_file(gguf, ok_flag, spec.model_id, spec.name) {
        mark_model_unavailable(&spec.model_id, "integrity_mismatch");
        return Err(error);
    }
    verified_models().write().unwrap().insert(spec.model_id);
    ui_download_info(&format!("[keryx-miner] Model '{}' integrity verified.", spec.name));
    Ok(())
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
}

/// Verify that GPU inference can actually work *before* mining starts.
///
/// The in-process llama engine dlopens cuBLAS lazily on the first load; discovering a missing
/// `libcublas` mid-challenge would silently drop responses. Probe both prerequisites up front:
/// a usable CUDA device (driver) and a loadable cuBLAS, and report a clean, actionable result.
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
    if ok {
        return GpuProbe::Ok;
    }
    GpuProbe::CublasMissing
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

/// Models advertised to the node must have complete files and a compatible inference engine.
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    if !crate::llama_engine::library_available() {
        return Vec::new();
    }
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs.iter()
        .filter(|s| model_is_verified(&s.model_id) && !model_is_unavailable(&s.model_id))
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
        .filter(|s| crate::models::is_pom_model(&s.model_id) && model_is_verified(&s.model_id))
        .collect()
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    if !crate::llama_engine::library_available() || model_is_unavailable(model_id) {
        return false;
    }
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_is_verified(&spec.model_id)
}

/// Serve an inference request via the in-process llama.cpp engine, swapping it to the requested
/// model first if it hosts a different one. Blocking — call from `spawn_blocking`.
///
/// The generated text is user-facing only — consensus checks the fixed-point `model_fixed`
/// commitment separately. A failed load/generation returns None (the response is dropped, never
/// submitted): a miner must not be rewarded for garbage.
pub fn load_and_run_inference(
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    correlation: &str,
) -> Option<String> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else {
        log::error!("event=ai_inference_failed correlation={} stage=model_lookup reason=unsupported_model", correlation);
        return None;
    };

    // llama.cpp gets the raw tokens of whatever string we pass — apply the model's chat
    // template here (template-strict models emit EOG immediately on a bare prompt).
    let templated = format_prompt_by_name(spec.name, prompt);
    // Route inference to the device that MINES this model (per-GPU tier assignment): only that
    // GPU pauses PoW and the walk shares the resident weights (zero-dup). Falls back to device 0
    // (single-GPU / unassigned model).
    let dev_id = crate::pom_gpu::device_for_model(model_id).unwrap_or(0);
    let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
    log::info!(
        "event=ai_inference_start correlation={} gpu={} model={} max_tokens={}",
        correlation,
        dev_id,
        spec.name,
        max_tokens
    );

    if !crate::llama_engine::active_for(&gguf, dev_id as usize) {
        // The engine hosts another model (or nothing). Inference has priority: release the
        // device's miner to make room, swap the engine to the requested model. The possession
        // walk rebuilds over the mining model at the next `ensure_installed`.
        log::info!(
            "event=llama_swap_start correlation={} gpu={} model={} previous_gpu={:?}",
            correlation,
            dev_id,
            spec.name,
            crate::llama_engine::active_gpu()
        );
        log::info!("event=pom_release_start correlation={} gpu={}", correlation, dev_id);
        crate::pom_gpu::uninstall(dev_id);
        log::info!("event=pom_release_success correlation={} gpu={}", correlation, dev_id);
        crate::llama_engine::unload(&format!("inference_swap correlation={}", correlation));
        match crate::llama_engine::ensure_loaded(&gguf, dev_id as usize) {
            Ok(attempt) => {
                mark_model_available(&spec.model_id, "inference_load_success");
                log::info!(
                    "event=llama_swap_success correlation={} attempt={} gpu={} model={}",
                    correlation,
                    attempt,
                    dev_id,
                    spec.name
                );
            }
            Err(e) => {
                mark_model_unavailable(&spec.model_id, "inference_load_failed");
                log::error!(
                    "event=ai_inference_failed correlation={} attempt={} gpu={} model={} stage=load detail=\"{}\"",
                    correlation,
                    e.attempt(),
                    dev_id,
                    spec.name,
                    e
                );
                return None;
            }
        }
    }

    match crate::llama_engine::generate(&templated, max_tokens) {
        Some(text) if !text.trim().is_empty() => {
            log::info!(
                "event=ai_inference_success correlation={} attempt={} gpu={} model={} output_bytes={}",
                correlation,
                crate::llama_engine::active_attempt().unwrap_or(0),
                dev_id,
                spec.name,
                text.len()
            );
            Some(text)
        }
        _ => {
            mark_model_unavailable(&spec.model_id, "generation_failed");
            log::warn!(
                "event=ai_inference_failed correlation={} attempt={} gpu={} model={} stage=generate reason=empty_or_failed",
                correlation,
                crate::llama_engine::active_attempt().unwrap_or(0),
                dev_id,
                spec.name
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn failed_models_are_withdrawn_until_recovered() {
        let model_id = [0xa5; 32];
        mark_model_available(&model_id, "test_reset");

        mark_model_unavailable(&model_id, "test_failure");
        assert!(model_is_unavailable(&model_id));

        mark_model_available(&model_id, "test_recovery");
        assert!(!model_is_unavailable(&model_id));
    }

    #[test]
    fn model_verification_rejects_same_size_corruption_and_removes_marker() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        let marker = dir.path().join(".ok");
        std::fs::write(&gguf, b"verified model bytes").unwrap();
        let expected = crate::integrity::unixfs_v0_digest_file(&gguf, |_, _| {}).unwrap();

        verify_model_file(&gguf, &marker, expected, "test-model").unwrap();
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), hex::encode(expected));

        let mut bytes = std::fs::read(&gguf).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&gguf, bytes).unwrap();
        assert!(verify_model_file(&gguf, &marker, expected, "test-model").is_err());
        assert!(!marker.exists(), "stale verification marker must be removed");
    }

    #[test]
    fn interrupted_model_never_creates_verified_marker() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        let marker = dir.path().join(".ok");
        let mut file = std::fs::File::create(&gguf).unwrap();
        file.write_all(b"partial").unwrap();

        assert!(verify_model_file(&gguf, &marker, [0x5a; 32], "test-model").is_err());
        assert!(!marker.exists());
    }

    #[test]
    fn unreadable_model_removes_stale_marker() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("missing.gguf");
        let marker = dir.path().join(".ok");
        std::fs::write(&marker, "stale").unwrap();

        assert!(verify_model_file(&gguf, &marker, [0x5a; 32], "test-model").is_err());
        assert!(!marker.exists());
    }
}
