/// Phase-3 Inference: model file management + inference dispatch.
///
/// Generation runs in the in-process llama.cpp engine (`llama_engine`, `libkeryx-llama.so`
/// next to the binary): llama.cpp owns the single resident VRAM copy of the model — the PoM
/// walk gathers straight over its tensors — and serves the inference text. This module owns the
/// served-lineup state (`ai:cap`), the model downloads, and the per-model chat templates.
/// Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::models::ModelSpec;

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
/// Shared system prompt for the whole lineup (vendor-agnostic wording).
const SYSTEM_PROMPT_NEXT: &str =
    "You are a Keryx Network AI, a decentralized assistant served by the GPU miners of the Keryx network. \
     Answer in the language of the user's message. \
     Do not introduce yourself or describe Keryx unless the user asks about it. If asked: Keryx is a proof-of-work BlockDAG \
     derived from Kaspa (about 10 blocks per second) where each mining GPU proves on every block that it holds a model in VRAM \
     (Proof-of-Model), so mining and inference are the same job; requests are on-chain transactions paid in KRX. \
     Web search results, facts or earlier messages included in the request are your sources for recent or specific information: \
     use them. Without them, answer from your training knowledge and say when it may be out of date; \
     never claim to have searched the web or cite sources that are not included in the request. \
     Never mention your underlying model name or the company that trained it: if asked, you are a Keryx Network AI. \
     Be thorough but concise.";

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
        log::info!("LlmEngine: reusing local model '{}' at {}", spec.name, dir.display());
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
        // GLM-4-0414 ignores the <|system|> role identity (keeps claiming a foreign vendor) —
        // fold the system prompt into the user turn instead.
        "glm-4-9b-0414" => format!(
            "[gMASK]<sop><|user|>\n{}\n\n{}\n<|assistant|>\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Qwen3 family — ChatML + a pre-filled empty think block so the visible answer starts
        // immediately (an open think block would eat the whole max_tokens budget). This is the
        // `enable_thinking = false` branch of their embedded template, verbatim.
        "qwen3.8-27b" | "qwen3.6-27b" | "qwen3.5-9b-abliterated" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Gemma 4 is NOT the classic <start_of_turn> Gemma: turns are `<|turn>role … <turn|>`,
        // and its generation prompt carries an empty thought channel when thinking is off — the
        // same role as Qwen's empty think block. Without it the model opens its own and the
        // channel markers leak into the answer. BOS is omitted on purpose: this GGUF sets
        // add_bos_token, so the tokenizer prepends it.
        "gemma-4-12b-abliterated" => format!(
            "<|turn>system\n{}<turn|>\n\
             <|turn>user\n{}<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>",
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

/// The chat template of an audited model split around the request-bound prefix: the head the
/// prefix tokens follow, and the rest after them. `None` for a model without a prefix slot.
pub(crate) fn format_prompt_parts(name: &str, prompt: &str) -> Option<(String, String)> {
    Some(match name {
        "glm-4-9b-0414" => (
            "[gMASK]<sop><|user|>\nRequest id:".to_string(),
            format!("\n{}\n\n{}\n<|assistant|>\n", SYSTEM_PROMPT_NEXT, prompt),
        ),
        "qwen3.8-27b" | "qwen3.6-27b" | "qwen3.5-9b-abliterated" => (
            "<|im_start|>system\nRequest id:".to_string(),
            format!(
                "\n{}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "gemma-4-12b-abliterated" => (
            "<|turn>system\nRequest id:".to_string(),
            format!(
                "\n{}<turn|>\n\
                 <|turn>user\n{}<turn|>\n\
                 <|turn>model\n<|channel>thought\n<channel|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "kimi-linear-48b" => (
            "<|im_system|>system<|im_middle|>Request id:".to_string(),
            format!(
                "\n{}<|im_end|>\
                 <|im_user|>user<|im_middle|>{}<|im_end|>\
                 <|im_assistant|>assistant<|im_middle|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        _ => return None,
    })
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
        log::debug!("LlmEngine: prefetching model '{}'…", spec.name);
        let result = ensure_gguf(spec).map(|_| ());
        match result {
            Ok(()) => log::debug!("LlmEngine: '{}' files ready.", spec.name),
            Err(e) => {
                log::error!("LlmEngine: prefetch '{}' failed: {} — cannot start mining.", spec.name, e);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// A model withdrawn from `ai:cap`, with the schedule of its next serve probe (`None` = stays
/// withdrawn until the miner restarts).
struct Withdrawal {
    reason: String,
    attempts: u32,
    next_probe: Option<Instant>,
}

fn unavailable_models() -> &'static RwLock<HashMap<[u8; 32], Withdrawal>> {
    static MODELS: OnceLock<RwLock<HashMap<[u8; 32], Withdrawal>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Probe recoveries per model since its last served response.
fn probe_recoveries() -> &'static RwLock<HashMap<[u8; 32], u32>> {
    static RECOVERIES: OnceLock<RwLock<HashMap<[u8; 32], u32>>> = OnceLock::new();
    RECOVERIES.get_or_init(|| RwLock::new(HashMap::new()))
}

const PROBE_BACKOFF_INITIAL: Duration = Duration::from_secs(60);
const PROBE_BACKOFF_MAX: Duration = Duration::from_secs(30 * 60);
/// Probe recoveries without a served response in between after which a model stays withdrawn.
const MAX_PROBE_RECOVERIES: u32 = 2;
const PROBE_TICK: Duration = Duration::from_secs(15);
const PROBE_PROMPT: &str = "Reply with one word: OK";
const PROBE_MAX_TOKENS: usize = 8;

static PROBE_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
/// Generations in flight (requests, challenges and probes).
static SERVING: AtomicUsize = AtomicUsize::new(0);

struct ServingGuard;

impl ServingGuard {
    fn new() -> Self {
        SERVING.fetch_add(1, Ordering::AcqRel);
        ServingGuard
    }
}

impl Drop for ServingGuard {
    fn drop(&mut self) {
        SERVING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Withdrawal reasons a later probe can clear: the engine may load the model next time.
fn probeable(reason: &str) -> bool {
    matches!(reason, "llama_load_failed" | "llama_load_oom" | "llama_engine_load_failed" | "llama_engine_oom")
}

fn probe_backoff(attempts: u32) -> Duration {
    PROBE_BACKOFF_INITIAL.saturating_mul(1u32 << attempts.min(16)).min(PROBE_BACKOFF_MAX)
}

/// Withdraw a model from `ai:cap`: the files are on disk but this miner cannot serve it right
/// now. The GPUs mining its tier park until it serves again.
pub fn mark_model_unavailable(model_id: &[u8; 32], reason: &str) {
    let recoveries = probe_recoveries().read().unwrap().get(model_id).copied().unwrap_or(0);
    let mut models = unavailable_models().write().unwrap();
    if models.contains_key(model_id) {
        return;
    }
    let retry = probeable(reason) && recoveries < MAX_PROBE_RECOVERIES;
    let next_probe = retry.then(|| Instant::now() + PROBE_BACKOFF_INITIAL);
    models.insert(*model_id, Withdrawal { reason: reason.to_string(), attempts: 0, next_probe });
    drop(models);
    let id = hex::encode(model_id);
    if retry {
        log::warn!(
            "LlmEngine: model {:.8} withdrawn from ai:cap ({}) — mining on its tier parked until it serves again, first probe in {}s",
            id, reason, PROBE_BACKOFF_INITIAL.as_secs()
        );
    } else if probeable(reason) {
        log::error!(
            "LlmEngine: model {:.8} withdrawn from ai:cap ({}) after {} probe recoveries — mining on its tier stays parked; restart the miner or change its tier assignment",
            id, reason, recoveries
        );
    } else {
        log::warn!("LlmEngine: model {:.8} withdrawn from ai:cap ({}) — mining on its tier parked", id, reason);
    }
}

/// Re-announce a model after it serves again.
pub fn mark_model_available(model_id: &[u8; 32], reason: &str) {
    if reason == "generation_success" {
        note_recovery(model_id, PROBE_IN_FLIGHT.load(Ordering::Acquire));
    }
    if unavailable_models().write().unwrap().remove(model_id).is_some() {
        log::info!("LlmEngine: model {:.8} back in ai:cap ({}) — mining on its tier resumes", hex::encode(model_id), reason);
    }
}

/// A served response resets the model's probe-recovery count; a probe recovery raises it.
fn note_recovery(model_id: &[u8; 32], from_probe: bool) {
    let mut recoveries = probe_recoveries().write().unwrap();
    if from_probe {
        *recoveries.entry(*model_id).or_insert(0) += 1;
    } else {
        recoveries.remove(model_id);
    }
}

fn schedule_next_probe(model_id: &[u8; 32]) {
    let mut models = unavailable_models().write().unwrap();
    let Some(w) = models.get_mut(model_id) else { return };
    if w.next_probe.is_none() {
        return;
    }
    w.attempts += 1;
    let delay = probe_backoff(w.attempts);
    w.next_probe = Some(Instant::now() + delay);
    log::warn!(
        "LlmEngine: model {:.8} still cannot serve ({}) — probe {} failed, next probe in {}s",
        hex::encode(model_id), w.reason, w.attempts, delay.as_secs()
    );
}

/// How often the probe task looks for a withdrawn model due for a serve probe.
pub fn probe_tick() -> Duration {
    PROBE_TICK
}

pub fn probe_in_flight() -> bool {
    PROBE_IN_FLIGHT.load(Ordering::Acquire)
}

/// The withdrawn model whose probe is due, when nothing is generating.
pub fn withdrawn_model_due_for_probe() -> Option<[u8; 32]> {
    if publishing_blocked() || probe_in_flight() || SERVING.load(Ordering::Acquire) > 0 {
        return None;
    }
    let now = Instant::now();
    let models = unavailable_models().read().unwrap();
    models.iter().filter(|(_, w)| w.next_probe.is_some_and(|t| t <= now)).map(|(id, _)| *id).min()
}

/// Try to serve a withdrawn model again; it is re-announced on success. Blocking.
pub fn probe_withdrawn_model(model_id: &[u8; 32]) -> bool {
    if PROBE_IN_FLIGHT.swap(true, Ordering::AcqRel) {
        return false;
    }
    log::info!("LlmEngine: probing withdrawn model {:.8}", hex::encode(model_id));
    let served = load_and_run_inference(model_id, PROBE_PROMPT, PROBE_MAX_TOKENS).is_some();
    if !served {
        schedule_next_probe(model_id);
    }
    PROBE_IN_FLIGHT.store(false, Ordering::Release);
    served
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

pub fn model_is_unavailable(model_id: &[u8; 32]) -> bool {
    unavailable_models().read().unwrap().contains_key(model_id)
}

#[cfg(test)]
fn withdrawal_state(model_id: &[u8; 32]) -> Option<(u32, bool)> {
    unavailable_models().read().unwrap().get(model_id).map(|w| (w.attempts, w.next_probe.is_some()))
}

/// Raised while the public gateways cannot fetch from this miner's IPFS node: every model is
/// withdrawn from `ai:cap` until the node is reachable again.
static PUBLISHING_BLOCKED: AtomicBool = AtomicBool::new(false);

pub fn set_publishing_blocked(blocked: bool) {
    if PUBLISHING_BLOCKED.swap(blocked, Ordering::AcqRel) != blocked {
        if blocked {
            log::warn!("LlmEngine: IPFS node unreachable from the public gateways — all models withdrawn from ai:cap");
        } else {
            log::info!("LlmEngine: IPFS node reachable again — models back in ai:cap");
        }
    }
}

pub fn publishing_blocked() -> bool {
    PUBLISHING_BLOCKED.load(Ordering::Acquire) && !inline_answers()
}

/// Raised once the chain reaches the private-inference gate: answers travel inline and this
/// miner no longer publishes anything to IPFS.
static INLINE_ANSWERS: AtomicBool = AtomicBool::new(false);

/// Record a chain DAA score seen from the node.
pub fn note_chain_daa(daa: u64) {
    if daa >= crate::pom::private_inference_activation_daa() && !INLINE_ANSWERS.swap(true, Ordering::AcqRel) {
        log::info!("LlmEngine: private-inference gate reached — answers travel inline, IPFS no longer required");
    }
}

/// True once answers travel inline (chain at or past the private-inference gate).
pub fn inline_answers() -> bool {
    INLINE_ANSWERS.load(Ordering::Acquire)
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
                    log::info!("LlmEngine: found '{}' at {} — moved to {}", spec.name, path.display(), canonical.display());
                    found.insert(digest, canonical);
                }
                Err(e) => {
                    log::info!("LlmEngine: found '{}' at {} — using it in place ({})", spec.name, path.display(), e);
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
    if publishing_blocked() {
        return Vec::new();
    }
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
    if publishing_blocked() {
        return false;
    }
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists() && !model_is_unavailable(model_id)
}

/// Serve an inference request via the in-process llama.cpp engine on the GPU that mines the
/// model, loading it there first if needed. Blocking — call from `spawn_blocking`.
///
/// The generated text is user-facing only — consensus checks the fixed-point `model_fixed`
/// commitment separately. A failed load/generation returns None (the response is dropped, never
/// submitted): a miner must not be rewarded for garbage.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    run_inference(model_id, prompt, max_tokens, None).map(|g| g.text)
}

/// What a traced inference commits: tier, plaintext prompt length, request id and stage.
#[derive(Clone, Copy)]
struct Traced {
    tier: u8,
    prompt_len: u32,
    request_hash: [u8; 32],
    chain: bool,
    /// Sequence leaves and the attention rule's manifest (with `chain`).
    attention: bool,
}

/// A generated answer and, when the tier is audited, the trace an opening needs.
pub struct Generated {
    pub text: String,
    pub trace: Option<crate::trace::ResponseTrace>,
}

/// `load_and_run_inference` with the canonical trace of `tier` captured.
/// Traced inference for an audited tier; `prompt_len` is the request's plaintext prompt length,
/// `chain` selects the stage-2 trace, `attention` its sequence leaves.
#[allow(clippy::too_many_arguments)]
pub fn load_and_run_traced_inference(
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    tier: u8,
    prompt_len: u32,
    request_hash: &[u8; 32],
    chain: bool,
    attention: bool,
) -> Option<Generated> {
    run_inference(model_id, prompt, max_tokens, Some(Traced { tier, prompt_len, request_hash: *request_hash, chain, attention: chain && attention }))
}

/// A traced tier that cannot be served as the consensus requires stops the miner.
fn fatal(detail: &str) -> ! {
    crate::exit_visibly(3, &format!("LlmEngine: {} — stopping, a traced tier cannot be served", detail))
}

/// Cuts `text` to the longest answer the request's token budget allows, on a char boundary.
fn truncate_answer(mut text: String, max_tokens: usize) -> String {
    let cap = keryx_verifiable_inference::shape::max_answer_len(max_tokens.min(u32::MAX as usize) as u32).min(usize::MAX as u64) as usize;
    if text.len() > cap {
        let mut end = cap;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// The engine hosting `spec`, resident on its mining GPU: `(gguf path, device)`.
fn resident_engine(spec: &ModelSpec) -> Option<(String, u32)> {
    // Route inference to the device that MINES this model (per-GPU tier assignment): only that
    // GPU pauses PoW and the walk shares the resident weights (zero-dup). Falls back to device 0
    // (single-GPU / unassigned model).
    let dev_id = crate::pom_gpu::device_for_model(&spec.model_id).unwrap_or(0);
    let gguf = gguf_path_for(spec).to_string_lossy().into_owned();

    if !crate::llama_engine::active_for(&gguf, dev_id as usize) {
        // Not resident on its GPU yet (or displaced). Inference has priority: release the
        // device's miner to make room and load the model. The possession walk rebuilds at the
        // next `ensure_installed`.
        log::info!("LlmEngine: loading the llama engine for '{}' (gpu{})", spec.name, dev_id);
        if let Err(e) = crate::pom_gpu::load_llama_for_inference(&gguf, dev_id) {
            log::error!("LlmEngine: cannot load '{}' — {}; response dropped", spec.name, e);
            mark_model_unavailable(&spec.model_id, if e.is_oom() { "llama_load_oom" } else { "llama_load_failed" });
            return None;
        }
    }
    Some((gguf, dev_id))
}

/// The possession index of `spec`, the resident one or one built from the GGUF.
fn weight_index(spec: &ModelSpec, gguf: &str) -> Option<std::sync::Arc<crate::pom::WeightIndex>> {
    match crate::pom::active_index_for_model(&spec.model_id) {
        Some(index) => Some(index),
        None => match crate::pom::WeightIndex::build_from_gguf(gguf, spec.model_id) {
            Ok(index) => Some(std::sync::Arc::new(index)),
            Err(e) => {
                log::warn!("LlmEngine '{}': no possession index for the opening ({})", spec.name, e);
                None
            }
        },
    }
}

/// Engines whose self-check passed, by `(gguf, device, stage)`.
static PREFIX_CHECKED: std::sync::Mutex<std::collections::BTreeSet<(String, u32, u8)>> = std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Traces a fixed prompt carrying a known request id on the engine hosting `spec`, replays two
/// prefix positions and verifies their points as the node would.
fn prefix_self_check(spec: &ModelSpec, gguf: &str, dev_id: u32, tier: u8) -> Result<(), String> {
    use crate::trace::{build_prefix_point, draw_prefix, prefix_ctx, prefix_tokens, TierManifest, TraceRecord};
    use keryx_verifiable_inference::prefix::verify_prefix_point;
    use keryx_verifiable_inference::verify::{ResponseCtx, TierCtx};
    let Some(ctx) = prefix_ctx(tier) else { return Ok(()) };
    if PREFIX_CHECKED.lock().unwrap().contains(&(gguf.to_string(), dev_id, crate::trace::STAGE_1A)) {
        return Ok(());
    }
    let m = TierManifest::for_tier(tier).ok_or("no manifest")?;
    let (head, rest) = format_prompt_parts(spec.name, "Reply with one word.").ok_or("no prefix template")?;
    let request_hash: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
    let tokens = prefix_tokens(ctx, &request_hash);
    let prefix = crate::llama_engine::PromptPrefix { tokens: &tokens, head_tokens: ctx.offset, rest: &rest };
    let g = crate::llama_engine::generate_traced(gguf, dev_id as usize, &head, Some(prefix), 1, 0, 0).map_err(|e| format!("{e:?}"))?;
    let (lo, hi) = (ctx.offset as usize, ctx.offset as usize + tokens.len());
    if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
        return Err("the fed prompt does not carry the request prefix".to_string());
    }
    let rec = TraceRecord::new(&m, g.n_prompt, g.ubatch, g.leaves).ok_or("incomplete trace")?;
    let index = weight_index(spec, gguf).ok_or("no possession index")?;
    let tier_ctx = TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: m.m_t, n_entries: m.entries.len() as u64, n_products: m.n_products as u64 };
    let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
    for seed in [[0x11u8; 32], [0x77u8; 32]] {
        let t = draw_prefix(ctx, &seed);
        let (x, _) = crate::llama_engine::replay(gguf, dev_id as usize, &g.tokens, g.n_prompt, t as u32, 0, 1)?;
        if *blake3::hash(&x).as_bytes() != rec.leaf(t, 0) {
            return Err(format!("the replay of position {t} does not reproduce the trace"));
        }
        let point = build_prefix_point(&m, &rec, ctx, &request_hash, t, x, &index).ok_or("no prefix point")?;
        verify_prefix_point(&tier_ctx, ctx, &resp, &seed, &request_hash, &point).map_err(|e| format!("position {t}: {e:?}"))?;
    }
    PREFIX_CHECKED.lock().unwrap().insert((gguf.to_string(), dev_id, crate::trace::STAGE_1A));
    log::info!("LlmEngine '{}': prefix self-check passed (gpu{})", spec.name, dev_id);
    Ok(())
}

/// Replays a response once for the `wanted` `(position, leaf)` captures, checked against its trace.
fn replay_captures(gguf: &str, dev_id: u32, m: &crate::trace::TierManifest, trace: &crate::trace::ResponseTrace, wanted: &[(u32, u32)]) -> Result<crate::trace::Captures, String> {
    use crate::trace::{captures_match, leaf_cap, Captures};
    let rec = &trace.record;
    let slot = wanted.iter().map(|&(_, l)| leaf_cap(m, l)).max().unwrap_or(0).div_ceil(4) * 4;
    let bytes = crate::llama_engine::replay_multi(gguf, dev_id as usize, &trace.tokens, rec.n_prompt, wanted, slot)
        .map_err(|e| format!("{e} for captures {wanted:?}"))?;
    let caps: Captures = wanted.iter().copied().zip(bytes).collect();
    if !captures_match(rec, &caps) {
        return Err("the replay does not reproduce the committed trace".to_string());
    }
    Ok(caps)
}

/// Replays a response for the sequence spans of `plans`.
fn replay_spans(gguf: &str, dev_id: u32, trace: &crate::trace::ResponseTrace, plans: &[crate::trace_seq::SpanPlan]) -> Result<Vec<crate::llama_engine::SeqCapture>, String> {
    let spans: Vec<(u32, u64, u64)> = plans.iter().map(|p| p.capture()).collect();
    crate::llama_engine::seq_replay(gguf, dev_id as usize, &trace.tokens, trace.record.n_prompt, &spans, crate::trace_seq::max_chunks(plans))
}

/// The attention opening of the draw of `seed` at the phase of `path`, `prev` the partials of the
/// previous phase, checked as the node checks it.
#[allow(clippy::too_many_arguments)]
fn open_attention(
    gguf: &str,
    dev_id: u32,
    m: &crate::trace::TierManifest,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
    path: &[u64],
    prev: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    use crate::trace_seq::{attn_opening, draw_attn, seq_response_ctx};
    use keryx_verifiable_inference::attn_blocks::{encode_partials, partials_root, pick_partials};
    use keryx_verifiable_inference::chain::{attn_query, verify_attn_phase};
    let rec = &trace.record;
    let seq = trace.seq.as_ref().ok_or("no sequence leaves")?;
    let d = draw_attn(m, rec, seed).ok_or("no attention draw")?;
    let caps = replay_captures(gguf, dev_id, m, trace, &d.captures(path).ok_or("no attention captures")?)?;
    let plans = d.spans(m, rec.n_pos as u64, path).ok_or("no attention spans")?;
    let spans = replay_spans(gguf, dev_id, trace, &plans)?;
    let full = attn_opening(m, rec, seq, index, &d, &caps, &spans, response_txid, path, prev).map_err(|e| format!("{e} (slot {}, head {}, t {}, path {path:?})", d.slot, d.head, d.t))?;
    let ctx = m.chain.ok_or("no stage-2 context")?;
    let (tier, resp) = (m.tier_ctx(index), seq_response_ctx(rec, seq));
    let parent_root = prev.and_then(|b| partials_root(b, d.dv));
    let verify = |op: &keryx_verifiable_inference::opening::AttnOpening| verify_attn_phase(&tier, &ctx, &resp, seed, path, parent_root.as_ref(), op);
    let op = if d.leaf(path) {
        full
    } else {
        let (q, scale) = attn_query(&tier, &ctx, &resp, seed, &full).map_err(|e| format!("attention query {e:?}"))?;
        let (keys, values) = d.rows(&full, path).ok_or("no cache rows")?;
        let p0 = d.tree.node(path).ok_or("no node")?.0;
        let partials = pick_partials(&q, scale, &keys, &values, p0, &d.tree, path, |p| verify(&d.phase_fields(&full, path, encode_partials(p))).is_ok())
            .ok_or_else(|| format!("no partials verify at layer {}, position {}, path {path:?}", d.layer, d.t))?;
        d.phase_fields(&full, path, partials)
    };
    verify(&op).map_err(|e| format!("attention phase {e:?} at layer {}, position {}, path {path:?}", d.layer, d.t))?;
    let body = op.encode();
    if body.len() + SEEDED_OVERHEAD > keryx_verifiable_inference::opening::MAX_BODY_LEN {
        return Err(format!("attention opening of {} B over the cap", body.len()));
    }
    Ok(body)
}

/// Bytes the seeded envelope adds to an opening body.
const SEEDED_OVERHEAD: usize = 33;

/// The recurrent opening of the draw of `seed`, checked as the node checks it.
#[allow(clippy::too_many_arguments)]
fn open_recurrent(
    gguf: &str,
    dev_id: u32,
    m: &crate::trace::TierManifest,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
) -> Result<Vec<u8>, String> {
    use crate::trace_seq::{draw_recur, recur_opening, seq_response_ctx};
    let rec = &trace.record;
    let seq = trace.seq.as_ref().ok_or("no sequence leaves")?;
    let d = draw_recur(m, rec, seed).ok_or("no recurrent draw")?;
    let caps = replay_captures(gguf, dev_id, m, trace, &d.captures())?;
    let plans = d.spans(m, rec.n_pos as u64).ok_or("no recurrent spans")?;
    let spans = replay_spans(gguf, dev_id, trace, &plans)?;
    let op = recur_opening(m, rec, seq, index, &d, &caps, &spans, response_txid)?;
    let ctx = m.chain.ok_or("no stage-2 context")?;
    keryx_verifiable_inference::recur::verify_recur_point(&m.tier_ctx(index), &ctx, &seq_response_ctx(rec, seq), seed, &op)
        .map_err(|e| format!("recurrent point {e:?} at layer {}, segment {}", d.layer, d.segment))?;
    let body = op.encode();
    if body.len() + SEEDED_OVERHEAD > keryx_verifiable_inference::opening::MAX_BODY_LEN {
        return Err(format!("recurrent opening of {} B over the cap", body.len()));
    }
    Ok(body)
}

/// Every opening of a stage-2 response: the trace opening, the attention and recurrent ones under
/// the attention rule when the tier has such layers, then the residual one.
#[allow(clippy::too_many_arguments)]
fn open_pieces(
    gguf: &str,
    dev_id: u32,
    m: &crate::trace::TierManifest,
    pctx: &keryx_verifiable_inference::prefix::PrefixCtx,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    request_hash: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
) -> Result<Vec<Vec<u8>>, String> {
    let (body, resid) = open_chain(gguf, dev_id, m, pctx, trace, seed, request_hash, response_txid, index)?;
    let mut bodies = vec![body];
    if trace.seq.is_some() {
        bodies.push(open_attention(gguf, dev_id, m, trace, seed, response_txid, index, &[], None)?);
        if m.chain.is_some_and(|c| c.n_recur > 0) {
            bodies.push(open_recurrent(gguf, dev_id, m, trace, seed, response_txid, index)?);
        }
    }
    bodies.push(resid);
    Ok(bodies)
}

/// A trace opening body carrying the residual point of its residual piece, as the block task
/// commits it.
fn inline_resid(body: &[u8], resid: &[u8]) -> Result<Vec<u8>, String> {
    use keryx_verifiable_inference::opening::{Opening, ResidOpening, MAX_BODY_LEN};
    let mut o = Opening::decode(body).ok_or("undecodable stage-2 opening")?;
    o.resid = Some(ResidOpening::decode(resid).ok_or("undecodable residual opening")?.resid);
    let b = o.encode();
    if b.len() + SEEDED_OVERHEAD > MAX_BODY_LEN {
        return Err(format!("block task opening of {} B over the cap", b.len()));
    }
    Ok(b)
}

/// Replays a stage-2 response once for every leaf the draw of `seed` opens and builds its trace
/// and residual openings.
#[allow(clippy::too_many_arguments)]
fn open_chain(
    gguf: &str,
    dev_id: u32,
    m: &crate::trace::TierManifest,
    pctx: &keryx_verifiable_inference::prefix::PrefixCtx,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    request_hash: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    use crate::trace::{chain_opening, draw_chain};
    use keryx_verifiable_inference::opening::MAX_BODY_LEN;
    let rec = &trace.record;
    let d = draw_chain(m, pctx, rec, seed).ok_or("no stage-2 draw")?;
    let caps = replay_captures(gguf, dev_id, m, trace, &d.captures(m.n_products as u64))?;
    let (mut o, mut r) = chain_opening(m, pctx, rec, &trace.tokens, seed, request_hash, response_txid, &d, &caps, index).ok_or_else(|| "no stage-2 opening".to_string())?;
    o.roots = trace.seq.as_ref().map(|seq| (rec.trace_root, seq.root));
    r.roots = o.roots;
    let resid = r.encode();
    let body = o.encode();
    for (name, b) in [("stage-2", &body), ("residual", &resid)] {
        if b.len() + SEEDED_OVERHEAD > MAX_BODY_LEN {
            return Err(format!("{name} opening of {} B over the cap", b.len()));
        }
    }
    let ctx = m.chain.ok_or("no stage-2 context")?;
    crate::trace::verify_chain_opening(&m.tier_ctx(index), &ctx, pctx, rec, seed, request_hash, &body, &resid).map_err(|e| format!("stage-2 opening fails its own check: {e}"))?;
    Ok((body, resid))
}

/// Traces a fixed prompt under the stage-2 manifest, then opens and verifies it under two seeds
/// as the node would.
fn chain_self_check(spec: &ModelSpec, gguf: &str, dev_id: u32, tier: u8, attention: bool) -> Result<(), String> {
    use crate::trace::{chain_ctx, prefix_tokens, verify_chain_opening, ResponseTrace, SeqTrace, TierManifest, TraceRecord, STAGE_CHAIN};
    let key = if attention { STAGE_CHAIN | 0x80 } else { STAGE_CHAIN };
    if PREFIX_CHECKED.lock().unwrap().contains(&(gguf.to_string(), dev_id, key)) {
        return Ok(());
    }
    let m = if attention { TierManifest::for_attn(tier) } else { TierManifest::for_chain(tier) }.ok_or("no stage-2 manifest")?;
    let (_, pctx) = chain_ctx(tier).ok_or("no stage-2 context")?;
    let ctx = m.chain.ok_or("no stage-2 context")?;
    let (head, rest) = format_prompt_parts(spec.name, "Reply with one word.").ok_or("no prefix template")?;
    let request_hash: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
    let tokens = prefix_tokens(&pctx, &request_hash);
    let prefix = crate::llama_engine::PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
    let g = crate::llama_engine::generate_traced(gguf, dev_id as usize, &head, Some(prefix), 4, 0, 0).map_err(|e| format!("{e:?}"))?;
    let (lo, hi) = (pctx.offset as usize, pctx.offset as usize + tokens.len());
    if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
        return Err("the fed prompt does not carry the request prefix".to_string());
    }
    let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens).ok_or("incomplete stage-2 trace")?;
    let seq = match (attention, g.seq) {
        (true, Some(leaves)) => Some(SeqTrace::new(leaves).ok_or("empty sequence leaves")?),
        (true, None) => return Err("no sequence leaves".to_string()),
        (false, _) => None,
    };
    let trace = ResponseTrace { record, tokens: g.tokens, seq };
    let index = weight_index(spec, gguf).ok_or("no possession index")?;
    let tier_ctx = m.tier_ctx(&index);
    for seed in [[0x11u8; 32], [0x77u8; 32]] {
        let bodies = open_pieces(gguf, dev_id, &m, &pctx, &trace, &seed, &request_hash, [0u8; 32], &index)?;
        verify_chain_opening(&tier_ctx, &ctx, &pctx, &trace.record, &seed, &request_hash, &bodies[0], bodies.last().ok_or("no residual piece")?)?;
    }
    PREFIX_CHECKED.lock().unwrap().insert((gguf.to_string(), dev_id, key));
    log::info!("LlmEngine '{}': stage-2 self-check passed (gpu{})", spec.name, dev_id);
    Ok(())
}

/// Arms the canonical trace of `tier` on the engine hosting `spec`, under the stage-2 manifest
/// when `chain`; `false` when the tier has no manifest or the engine cannot trace. A tier that
/// cannot pass its self-check stops the miner.
fn arm_trace(spec: &ModelSpec, gguf: &str, dev_id: u32, tier: u8, chain: bool, attention: bool) -> bool {
    use crate::trace::TierManifest;
    let manifest = if attention {
        TierManifest::raw_attn(tier)
    } else if chain {
        TierManifest::raw_chain(tier)
    } else {
        TierManifest::raw_for_tier(tier)
    };
    let Some(manifest) = manifest else { return false };
    if !crate::llama_engine::traced_with(gguf, dev_id as usize, manifest) {
        if let Err(e) = crate::llama_engine::trace_configure(gguf, dev_id as usize, manifest) {
            log::warn!("LlmEngine '{}': trace unavailable ({}) — answering untraced", spec.name, e);
            return false;
        }
    }
    let seq = match (attention, TierManifest::for_attn(tier)) {
        (true, Some(m)) => {
            let Some(c) = m.chain else { return false };
            crate::llama_engine::seq_configure(gguf, dev_id as usize, &m.seq_layers(), c.n_attn, c.n_recur, c.seq_heads, m.seq_interval())
        }
        (true, None) => return false,
        (false, _) => crate::llama_engine::seq_configure(gguf, dev_id as usize, &[], 0, 0, 0, 0),
    };
    if let Err(e) = seq {
        log::warn!("LlmEngine '{}': sequence leaves unavailable ({}) — answering untraced", spec.name, e);
        return false;
    }
    let checked = if chain { chain_self_check(spec, gguf, dev_id, tier, attention) } else { prefix_self_check(spec, gguf, dev_id, tier) };
    if let Err(e) = checked {
        fatal(&format!("'{}' fails the {} self-check on gpu{}: {}", spec.name, if chain { "stage-2" } else { "prefix" }, dev_id, e));
    }
    true
}

fn run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize, traced: Option<Traced>) -> Option<Generated> {
    let _serving = ServingGuard::new();
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // llama.cpp gets the raw tokens of whatever string we pass — apply the model's chat
    // template here (template-strict models emit EOG immediately on a bare prompt).
    let templated = format_prompt_by_name(spec.name, prompt);
    let (gguf, dev_id) = resident_engine(spec)?;
    let traced = traced.filter(|t| arm_trace(spec, &gguf, dev_id, t.tier, t.chain, t.attention));

    let outcome = if let Some(Traced { tier, prompt_len, request_hash, chain, attention }) = traced {
        let m = if attention {
            crate::trace::TierManifest::for_attn(tier)?
        } else if chain {
            crate::trace::TierManifest::for_chain(tier)?
        } else {
            crate::trace::TierManifest::for_tier(tier)?
        };
        let min_prompt = keryx_verifiable_inference::shape::min_prompt_positions(prompt_len);
        let per_position = keryx_verifiable_inference::shape::ANSWER_BYTES_PER_POSITION;
        let owned_ctx = if chain { crate::trace::chain_ctx(tier).map(|c| c.1) } else { crate::trace::prefix_ctx(tier).cloned() };
        let ctx = owned_ctx.as_ref();
        let parts = ctx.and_then(|_| format_prompt_parts(spec.name, prompt));
        if ctx.is_some() && parts.is_none() {
            fatal(&format!("'{}' has no prefix template", spec.name));
        }
        let tokens = ctx.map(|c| crate::trace::prefix_tokens(c, &request_hash));
        let (head, prefix) = match (&parts, &tokens, ctx) {
            (Some((head, rest)), Some(tokens), Some(c)) => {
                (head.as_str(), Some(crate::llama_engine::PromptPrefix { tokens, head_tokens: c.offset, rest }))
            }
            _ => (templated.as_str(), None),
        };
        let generated = match crate::llama_engine::generate_traced(&gguf, dev_id as usize, head, prefix, max_tokens, min_prompt, per_position) {
            Err(crate::llama_engine::GenError::PromptHead(detail)) => fatal(&format!("'{}': {}", spec.name, detail)),
            other => other,
        };
        generated.and_then(|g| {
            if let (Some(c), Some(tokens)) = (ctx, &tokens) {
                let (lo, hi) = (c.offset as usize, c.offset as usize + tokens.len());
                if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
                    return Err(crate::llama_engine::GenError::Failed("the fed prompt does not carry the request prefix".to_string()));
                }
            }
            let record = if chain {
                crate::trace::TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens)
            } else {
                crate::trace::TraceRecord::new(&m, g.n_prompt, g.ubatch, g.leaves)
            };
            let seq = if attention { g.seq.and_then(crate::trace::SeqTrace::new) } else { None };
            let record = record.filter(|_| !attention || seq.is_some());
            if record.is_none() {
                log::warn!("LlmEngine '{}': incomplete trace — answering without a commitment", spec.name);
            }
            Ok((g.text, record.map(|record| crate::trace::ResponseTrace { record, tokens: g.tokens, seq })))
        })
    } else {
        crate::llama_engine::generate(&gguf, dev_id as usize, &templated, max_tokens).map(|text| (text, None))
    };

    match outcome {
        Ok((text, trace)) if !text.trim().is_empty() => {
            mark_model_available(model_id, "generation_success");
            Some(Generated { text: truncate_answer(text, max_tokens), trace })
        }
        Ok(_) => {
            log::warn!("LlmEngine '{}': llama generate returned an empty answer — response dropped", spec.name);
            None
        }
        Err(crate::llama_engine::GenError::PromptTooLong(detail)) => {
            log::warn!("LlmEngine '{}': prompt too long ({}) — response dropped", spec.name, detail);
            mark_model_available(model_id, "generation_success");
            None
        }
        Err(e) => {
            log::warn!("LlmEngine '{}': llama generate failed ({:?}) — response dropped", spec.name, e);
            None
        }
    }
}

/// The block task trace of `tier` for the window seeded by `window_seed`, on the engine hosting
/// `model_id`. Blocking.
pub fn run_block_task(model_id: &[u8; 32], tier: u8, window_seed: &[u8; 32]) -> Option<crate::trace::ResponseTrace> {
    use keryx_verifiable_inference::block_task::{task_tokens, TASK_LEN};
    let _serving = ServingGuard::new();
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;
    let (gguf, dev_id) = resident_engine(spec)?;
    if !arm_trace(spec, &gguf, dev_id, tier, true, false) {
        return None;
    }
    let m = crate::trace::TierManifest::for_chain(tier)?;
    let (_, pctx) = crate::trace::chain_ctx(tier)?;
    let tokens: Vec<i32> = task_tokens(&pctx.table, window_seed).into_iter().map(|t| t as i32).collect();
    let traced = match crate::llama_engine::trace_tokens(&gguf, dev_id as usize, &tokens) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("LlmEngine '{}': block task not traced ({})", spec.name, e);
            return None;
        }
    };
    let record = crate::trace::TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &tokens);
    if record.is_none() {
        log::warn!("LlmEngine '{}': incomplete block task trace", spec.name);
    }
    Some(crate::trace::ResponseTrace { record: record?, tokens, seq: None })
}

/// The task opening of the block `block_hash` from the task trace of its window, checked as the
/// node will check it. Blocking.
pub fn open_block_task(model_id: &[u8; 32], tier: u8, trace: &crate::trace::ResponseTrace, window_seed: &[u8; 32], block_hash: &[u8; 32]) -> Option<Vec<u8>> {
    use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, verify_block_task};
    let _serving = ServingGuard::new();
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;
    let (gguf, dev_id) = resident_engine(spec)?;
    if !arm_trace(spec, &gguf, dev_id, tier, true, false) {
        return None;
    }
    let index = weight_index(spec, &gguf)?;
    let m = crate::trace::TierManifest::for_chain(tier)?;
    let (ctx, pctx) = crate::trace::chain_ctx(tier)?;
    let body = match open_chain(&gguf, dev_id, &m, &task_prefix(&pctx), trace, &draw_seed(block_hash), &task_bytes(window_seed), [0u8; 32], &index)
        .and_then(|(b, r)| inline_resid(&b, &r))
    {
        Ok(b) => b,
        Err(e) => {
            log::warn!("LlmEngine '{}': block task opening failed ({})", spec.name, e);
            return None;
        }
    };
    let opening = keryx_verifiable_inference::opening::Opening::decode(&body)?;
    if let Err(e) = verify_block_task(&m.tier_ctx(&index), &ctx, &pctx, &trace.record.trace_root, window_seed, block_hash, &opening) {
        log::error!("LlmEngine '{}': block task opening fails its self-check ({:?})", spec.name, e);
        return None;
    }
    Some(body)
}

/// Builds the opening body of an audited response on the GPU that served it: replays the
/// response to each drawn position and proves the captured bytes against the possession index.
/// The opening of an audited response: the drawn points and, for a response accepted under the
/// prefix rule, its prefix point.
/// The attention phase of `path` of an audited response, `prev` the partials of its previous
/// phase; `None` when it cannot be built.
pub fn open_attention_phase(
    model_id: &[u8; 32],
    tier: u8,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    response_txid: [u8; 32],
    path: &[u64],
    prev: &[u8],
) -> Option<Vec<u8>> {
    let _serving = ServingGuard::new();
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;
    let (gguf, dev_id) = resident_engine(spec)?;
    if trace.seq.is_none() || !arm_trace(spec, &gguf, dev_id, tier, true, true) {
        return None;
    }
    let index = weight_index(spec, &gguf)?;
    let m = crate::trace::TierManifest::for_attn(tier)?;
    match open_attention(&gguf, dev_id, &m, trace, seed, response_txid, &index, path, Some(prev)) {
        Ok(body) => Some(body),
        Err(e) => {
            log::warn!("LlmEngine '{}': attention phase {:?} failed ({})", spec.name, path, e);
            None
        }
    }
}

pub fn open_trace(
    model_id: &[u8; 32],
    tier: u8,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    response_txid: [u8; 32],
    request_hash: &[u8; 32],
    accepted_daa: u64,
) -> Option<Vec<u8>> {
    use crate::trace::{build_point, build_prefix_point, draw_point, draw_prefix, opening_body, prefix_active, prefix_ctx, TierManifest, POINTS};
    let _serving = ServingGuard::new();
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;
    let chain = trace.record.stage == crate::trace::STAGE_CHAIN;
    let attention = chain && trace.seq.is_some();
    let (gguf, dev_id) = resident_engine(spec)?;
    if !arm_trace(spec, &gguf, dev_id, tier, chain, attention) {
        return None;
    }
    let index = weight_index(spec, &gguf)?;
    if chain {
        let m = if attention { TierManifest::for_attn(tier)? } else { TierManifest::for_chain(tier)? };
        let (_, pctx) = crate::trace::chain_ctx(tier)?;
        return match open_pieces(&gguf, dev_id, &m, &pctx, trace, seed, request_hash, response_txid, &index) {
            Ok(bodies) => Some(crate::trace_seq::bundle(&bodies)),
            Err(e) => {
                log::warn!("LlmEngine '{}': stage-2 opening failed ({})", spec.name, e);
                None
            }
        };
    }
    let m = TierManifest::for_tier_at(tier, accepted_daa)?;
    let rec = &trace.record;
    let mut points = Vec::with_capacity(POINTS as usize);
    for i in 0..POINTS {
        let p = draw_point(&m, rec.n_prompt, rec.n_pos, seed, i)?;
        let (x, y) = match crate::llama_engine::replay(&gguf, dev_id as usize, &trace.tokens, rec.n_prompt, p.t as u32, p.leaf_x(), p.leaf_y()) {
            Ok(xy) => xy,
            Err(e) => {
                log::warn!("LlmEngine '{}': replay failed at position {} ({})", spec.name, p.t, e);
                return None;
            }
        };
        if *blake3::hash(&x).as_bytes() != rec.leaf(p.t, p.leaf_x()) || *blake3::hash(&y).as_bytes() != rec.leaf(p.t, p.leaf_y()) {
            log::error!("LlmEngine '{}': replay of position {} does not reproduce the committed trace", spec.name, p.t);
            return None;
        }
        points.push(build_point(&m, rec, &p, x, &y, &index)?);
    }
    let mut prefix = None;
    if let Some(ctx) = prefix_ctx(tier).filter(|_| prefix_active(accepted_daa)) {
        let t = draw_prefix(ctx, seed);
        let (x, _) = match crate::llama_engine::replay(&gguf, dev_id as usize, &trace.tokens, rec.n_prompt, t as u32, 0, 1) {
            Ok(xy) => xy,
            Err(e) => {
                log::warn!("LlmEngine '{}': replay failed at prefix position {} ({})", spec.name, t, e);
                return None;
            }
        };
        if *blake3::hash(&x).as_bytes() != rec.leaf(t, 0) {
            log::error!("LlmEngine '{}': replay of prefix position {} does not reproduce the committed trace", spec.name, t);
            return None;
        }
        prefix = Some(build_prefix_point(&m, rec, ctx, request_hash, t, x, &index)?);
    }
    opening_body(response_txid, points, prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with no arm of its own falls through to generic ChatML, which template-strict
    /// models answer with leaked control tokens or an unclosed thinking block. The generic tail
    /// is the tell: every real arm ends on something else.
    #[test]
    fn every_registered_model_has_its_own_chat_template() {
        // Positive control: an unknown model does fall back, so the tail below really is the
        // fallback's signature and the loop is not asserting a tautology.
        assert!(format_prompt_by_name("not-a-real-model", "hi").ends_with("<|im_start|>assistant\n"));
        for spec in crate::models::REGISTRY {
            let templated = format_prompt_by_name(spec.name, "hi");
            assert!(
                !templated.ends_with("<|im_start|>assistant\n"),
                "{} has no chat template of its own — it falls back to generic ChatML",
                spec.name
            );
        }
    }

    /// The split template is the plain one with the request-id line inserted after the first
    /// role line, for every audited model.
    #[test]
    fn prefix_parts_rebuild_the_plain_template() {
        for name in ["qwen3.5-9b-abliterated", "glm-4-9b-0414", "gemma-4-12b-abliterated", "qwen3.8-27b", "kimi-linear-48b"] {
            let (head, rest) = format_prompt_parts(name, "hi").unwrap();
            assert!(head.ends_with("Request id:"), "{name}: {head:?}");
            assert!(rest.starts_with('\n'), "{name}: {rest:?}");
            assert_eq!(format!("{head}{rest}").replace("Request id:\n", ""), format_prompt_by_name(name, "hi"), "{name}");
        }
        assert!(format_prompt_parts("not-a-real-model", "hi").is_none());
    }

    /// The attention phases after `first`, each child drawn from a stand-in for the chain
    /// randomness, every phase checked as the node checks it.
    #[allow(clippy::too_many_arguments)]
    fn attention_phases(
        gguf: &str,
        m: &crate::trace::TierManifest,
        trace: &crate::trace::ResponseTrace,
        seed: &[u8; 32],
        index: &crate::pom::WeightIndex,
        first: &[u8],
    ) -> Result<Vec<Vec<u8>>, String> {
        use keryx_verifiable_inference::attn_blocks::{child_weights, decode_partials, draw_weighted_child};
        use keryx_verifiable_inference::opening::{inner_body, AttnOpening};
        let dv = crate::trace_seq::draw_attn(m, &trace.record, seed).ok_or("no attention draw")?.dv;
        let decode = |b: &[u8]| inner_body(b).as_deref().and_then(AttnOpening::decode).or_else(|| AttnOpening::decode(b)).ok_or("undecodable attention body");
        let (mut path, mut prev, mut bodies) = (Vec::new(), decode(first)?.partials, Vec::new());
        while !prev.is_empty() {
            let rand = *blake3::hash(&[seed.as_slice(), &[path.len() as u8]].concat()).as_bytes();
            path.push(draw_weighted_child(&rand, &child_weights(&decode_partials(&prev, dv).ok_or("undecodable partials")?).ok_or("no draw weights")?));
            let body = open_attention(gguf, 0, m, trace, seed, [3u8; 32], index, &path, Some(&prev))?;
            prev = decode(&body)?.partials;
            bodies.push(body);
        }
        Ok(bodies)
    }

    /// Traces under the attention rule commit sequence leaves, and their trace, attention and
    /// recurrent openings verify as the node would.
    /// `KERYX_TEST_MODEL_GPU0=<gguf> cargo test --release -- --ignored attention_openings_verify --nocapture`
    #[test]
    #[ignore]
    fn attention_openings_verify() {
        use crate::llama_engine;
        std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1");
        std::env::set_var("LLAMA_ATTN_ROT_DISABLE", "1");
        use crate::trace::{chain_ctx, prefix_tokens, verify_chain_opening, ResponseTrace, SeqTrace, TierManifest, TraceRecord};
        let gguf = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let spec = crate::models::spec_for_gguf(&gguf).expect("registry model");
        let tier = crate::models::pom_tier_index(&spec.model_id, u64::MAX).expect("tiered model");
        let m = TierManifest::for_attn(tier).expect("attention manifest");
        let ctx = m.chain.unwrap();
        let (_, pctx) = chain_ctx(tier).expect("stage-2 context");
        let index = crate::pom::WeightIndex::build_from_gguf(&gguf, spec.model_id).unwrap();
        let tier_ctx = m.tier_ctx(&index);
        llama_engine::ensure_loaded(&gguf, 0).unwrap();
        let nl = llama_engine::trace_configure(&gguf, 0, TierManifest::raw_attn(tier).unwrap()).unwrap();
        assert_eq!(nl, m.n_leaves());
        let n_seq = llama_engine::seq_configure(&gguf, 0, &m.seq_layers(), ctx.n_attn, ctx.n_recur, ctx.seq_heads, m.seq_interval()).unwrap();
        eprintln!("{}: {} position leaves, {} sequence leaves ({} attention, {} recurrent layers)", spec.name, nl, n_seq, ctx.n_attn, ctx.n_recur);
        let rh: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let toks = prefix_tokens(&pctx, &rh);
        let notes: String = (0..300).map(|i| format!("Note {i}: the quick brown fox jumps over the lazy dog near river bank number {i}.\n")).collect();
        let very_long = format!("Summarize the following notes in one sentence.\n{notes}");
        let many: String = (0..600).map(|i| format!("Note {i}: the quick brown fox jumps over the lazy dog near river bank number {i}.\n")).collect();
        let huge = format!("Summarize the following notes in one sentence.\n{many}");
        let prompts = [
            ("short", "Explain in two sentences why the sky is blue.", 24),
            ("long", &"Summarize: the quick brown fox jumps over the lazy dog. ".repeat(12), 16),
            ("very long", &very_long, 8),
            ("huge", &huge, 8),
        ];
        for (label, question, max_tokens) in prompts {
            let (head, rest) = format_prompt_parts(spec.name, question).expect("prefix template");
            let prefix = llama_engine::PromptPrefix { tokens: &toks, head_tokens: pctx.offset, rest: &rest };
            let t0 = std::time::Instant::now();
            let g = llama_engine::generate_traced(&gguf, 0, &head, Some(prefix), max_tokens, 0, 0).unwrap();
            eprintln!("{label}: {} positions ({} prompt) in {:.2?}: {:?}", g.n_pos, g.n_prompt, t0.elapsed(), g.text);
            let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone(), &g.tokens).expect("complete stage-2 trace");
            let seq = SeqTrace::new(g.seq.clone().expect("sequence leaves")).expect("sequence leaves");
            assert_eq!(seq.leaves.len(), n_seq);
            let trace = ResponseTrace { record, tokens: g.tokens.clone(), seq: Some(seq) };
            let depth = |c: u64| crate::trace_seq::draw_attn(&m, &trace.record, blake3::hash(&c.to_le_bytes()).as_bytes()).map_or(0, |d| d.tree.depth());
            let counters: Vec<u64> = if label == "very long" || label == "huge" {
                let (n_deep, n_any) = if label == "huge" { (10, 4) } else { (30, 10) };
                let deep = (0..8_000u64).filter(|&c| depth(c) == 2).take(n_deep);
                deep.chain((0..8_000u64).filter(|&c| depth(c) < 2).take(n_any)).collect()
            } else {
                (0..40).collect()
            };
            eprintln!("{label}: {} seeds, {} of them with three phases", counters.len(), counters.iter().filter(|&&c| depth(c) == 2).count());
            for counter in counters {
                let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
                let t0 = std::time::Instant::now();
                let bodies = open_pieces(&gguf, 0, &m, &pctx, &trace, &seed, &rh, [3u8; 32], &index).unwrap_or_else(|e| panic!("{label} seed {counter}: {e}"));
                let took = t0.elapsed();
                verify_chain_opening(&tier_ctx, &ctx, &pctx, &trace.record, &seed, &rh, &bodies[0], bodies.last().unwrap()).unwrap_or_else(|e| panic!("{label} seed {counter}: {e}"));
                let sizes: Vec<usize> = bodies.iter().map(Vec::len).collect();
                let t1 = std::time::Instant::now();
                let phases = attention_phases(&gguf, &m, &trace, &seed, &index, &bodies[1]).unwrap_or_else(|e| panic!("{label} seed {counter}: {e}"));
                let phase_sizes: Vec<usize> = phases.iter().map(Vec::len).collect();
                eprintln!("{label} seed {counter}: {} pieces verified, sizes {:?} B, replays+build {:.2?}; later attention phases {:?} B in {:.2?}", bodies.len(), sizes, took, phase_sizes, t1.elapsed());
            }
        }
    }

    /// Stage-2 traces of a real model open and verify as the node would, over many seeds.
    /// `KERYX_TEST_MODEL_GPU0=<gguf> cargo test --release -- --ignored stage2_openings_verify --nocapture`
    #[test]
    #[ignore]
    fn stage2_openings_verify() {
        use crate::llama_engine;
        use crate::trace::{chain_ctx, draw_chain, prefix_tokens, verify_chain_opening, ResponseTrace, TierManifest, TraceRecord};
        let gguf = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let spec = crate::models::spec_for_gguf(&gguf).expect("registry model");
        let tier = crate::models::pom_tier_index(&spec.model_id, u64::MAX).expect("tiered model");
        let m = TierManifest::for_chain(tier).expect("stage-2 manifest");
        let (ctx, pctx) = chain_ctx(tier).expect("stage-2 context");
        let index = crate::pom::WeightIndex::build_from_gguf(&gguf, spec.model_id).unwrap();
        let tier_ctx = m.tier_ctx(&index);
        llama_engine::ensure_loaded(&gguf, 0).unwrap();
        let nl = llama_engine::trace_configure(&gguf, 0, TierManifest::raw_chain(tier).unwrap()).unwrap();
        assert_eq!(nl, m.n_leaves());
        let rh: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let toks = prefix_tokens(&pctx, &rh);
        for (label, question, max_tokens) in [("short", "Explain in two sentences why the sky is blue.", 24), ("long", &"Summarize: the quick brown fox jumps over the lazy dog. ".repeat(40), 12)] {
            let (head, rest) = format_prompt_parts(spec.name, question).expect("prefix template");
            let prefix = llama_engine::PromptPrefix { tokens: &toks, head_tokens: pctx.offset, rest: &rest };
            let g = llama_engine::generate_traced(&gguf, 0, &head, Some(prefix), max_tokens, 0, 0).unwrap();
            eprintln!("{label}: {} positions ({} prompt): {:?}", g.n_pos, g.n_prompt, g.text);
            let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone(), &g.tokens).expect("complete stage-2 trace");
            let trace = ResponseTrace { record, tokens: g.tokens.clone(), seq: None };
            let (mut sizes, mut kinds) = (Vec::new(), std::collections::BTreeMap::<String, u32>::new());
            for counter in 0..32u64 {
                let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
                let d = draw_chain(&m, &pctx, &trace.record, &seed).unwrap();
                for (p, _) in &d.points {
                    if let keryx_verifiable_inference::manifest::Chain::Product(pl) = m.entry(p.g_rank).chain {
                        *kinds.entry(format!("link{}", pl.link)).or_default() += 1;
                    }
                }
                *kinds.entry(format!("resid{:?}{}", d.resid.kind, if d.resid.layer + 1 == ctx.n_layers as u64 { "-last" } else { "" })).or_default() += 1;
                let t0 = std::time::Instant::now();
                let (body, resid) = open_chain(&gguf, 0, &m, &pctx, &trace, &seed, &rh, [3u8; 32], &index).unwrap_or_else(|e| panic!("{label} seed {counter}: {e}"));
                let took = t0.elapsed();
                verify_chain_opening(&tier_ctx, &ctx, &pctx, &trace.record, &seed, &rh, &body, &resid).unwrap_or_else(|e| panic!("{label} seed {counter}: {e}"));
                eprintln!("{label} seed {counter}: verified, body {} B, residual {} B, replay+build {:.2?}", body.len(), resid.len(), took);
                sizes.push(body.len().max(resid.len()));
            }
            eprintln!("{label}: max body {} B, coverage {:?}", sizes.iter().max().unwrap(), kinds);
        }

        // Block task: the window trace opens and verifies for several block hashes.
        use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, task_tokens, verify_block_task, TASK_LEN};
        let window_seed = [0x5au8; 32];
        let tokens: Vec<i32> = task_tokens(&pctx.table, &window_seed).into_iter().map(|t| t as i32).collect();
        let t0 = std::time::Instant::now();
        let traced = llama_engine::trace_tokens(&gguf, 0, &tokens).unwrap();
        eprintln!("block task traced in {:.2?}", t0.elapsed());
        let record = TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &tokens).expect("complete task trace");
        let task = ResponseTrace { record, tokens, seq: None };
        let tp = task_prefix(&pctx);
        for counter in 0..16u64 {
            let block_hash = *blake3::hash(&counter.to_le_bytes()).as_bytes();
            let t0 = std::time::Instant::now();
            let body = open_chain(&gguf, 0, &m, &tp, &task, &draw_seed(&block_hash), &task_bytes(&window_seed), [0u8; 32], &index)
                .and_then(|(b, r)| inline_resid(&b, &r))
                .unwrap_or_else(|e| panic!("task {counter}: {e}"));
            let took = t0.elapsed();
            let opening = keryx_verifiable_inference::opening::Opening::decode(&body).unwrap();
            verify_block_task(&tier_ctx, &ctx, &pctx, &task.record.trace_root, &window_seed, &block_hash, &opening)
                .unwrap_or_else(|e| panic!("task {counter}: {e:?}"));
            eprintln!("task {counter}: verified, body {} B, replay+build {:.2?}", body.len(), took);
        }
        llama_engine::unload(0);
    }

    /// The two H6 models close the reasoning block in the generation prompt, each in its own
    /// dialect — otherwise the model opens one and spends the whole token budget inside it.
    #[test]
    fn h6_models_prefill_a_closed_reasoning_block() {
        let qwen = format_prompt_by_name("qwen3.5-9b-abliterated", "hi");
        assert!(qwen.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{qwen}");

        let gemma = format_prompt_by_name("gemma-4-12b-abliterated", "hi");
        assert!(gemma.ends_with("<|turn>model\n<|channel>thought\n<channel|>"), "{gemma}");
        assert!(gemma.contains("<|turn>user\nhi<turn|>"), "{gemma}");
        // The tokenizer prepends BOS (add_bos_token), so the template must not carry one.
        assert!(!gemma.contains("<bos>"), "{gemma}");
    }

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
            ctx_floor: 4_096,
            ctx_cap: 4_096,
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

    #[test]
    fn a_load_failure_schedules_a_probe_and_backs_off() {
        let model_id = [0xd4u8; 32];
        mark_model_unavailable(&model_id, "llama_load_failed");
        assert_eq!(withdrawal_state(&model_id), Some((0, true)));

        schedule_next_probe(&model_id);
        schedule_next_probe(&model_id);
        assert_eq!(withdrawal_state(&model_id), Some((2, true)));
        assert_eq!(probe_backoff(0), PROBE_BACKOFF_INITIAL);
        assert_eq!(probe_backoff(1), PROBE_BACKOFF_INITIAL * 2);
        assert_eq!(probe_backoff(20), PROBE_BACKOFF_MAX);

        mark_model_available(&model_id, "generation_success");
        assert_eq!(withdrawal_state(&model_id), None);
    }

    #[test]
    fn an_integrity_withdrawal_is_never_probed() {
        let model_id = [0xe5u8; 32];
        mark_model_unavailable(&model_id, "integrity_mismatch");
        assert_eq!(withdrawal_state(&model_id), Some((0, false)));
        schedule_next_probe(&model_id);
        assert_eq!(withdrawal_state(&model_id), Some((0, false)));
        mark_model_available(&model_id, "integrity_verified");
    }

    #[test]
    fn repeated_probe_recoveries_make_the_withdrawal_permanent() {
        let model_id = [0xf6u8; 32];
        for _ in 0..MAX_PROBE_RECOVERIES {
            mark_model_unavailable(&model_id, "llama_load_oom");
            assert_eq!(withdrawal_state(&model_id), Some((0, true)));
            note_recovery(&model_id, true);
            mark_model_available(&model_id, "probe");
        }
        mark_model_unavailable(&model_id, "llama_load_oom");
        assert_eq!(withdrawal_state(&model_id), Some((0, false)));

        // A served response resets the count: the next withdrawal probes again.
        note_recovery(&model_id, false);
        mark_model_available(&model_id, "served");
        mark_model_unavailable(&model_id, "llama_load_oom");
        assert_eq!(withdrawal_state(&model_id), Some((0, true)));
        mark_model_available(&model_id, "cleanup");
    }

    #[test]
    fn a_probe_is_only_due_when_nothing_generates() {
        let model_id = [0x1au8; 32];
        unavailable_models().write().unwrap().insert(
            model_id,
            Withdrawal { reason: "llama_load_failed".into(), attempts: 0, next_probe: Some(Instant::now()) },
        );
        let guard = ServingGuard::new();
        assert_eq!(withdrawn_model_due_for_probe(), None);
        drop(guard);
        assert!(withdrawn_model_due_for_probe().is_some());
        unavailable_models().write().unwrap().remove(&model_id);
    }
}
