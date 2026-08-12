#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::IsTerminal;
#[cfg(unix)]
use std::os::fd::AsRawFd;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::logging::init_logging;
use crate::miner::MinerManager;
use crate::stats::{spawn_stats_server, MinerStats};
use crate::target::Uint256;
use crate::ui::{spawn_ui, UiState};

mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod logging;
mod miner;
mod pow;
mod stats;
mod target;
mod ui;
mod watch;

// PoM mining is CUDA-only (the walk kernel is CUDA). The OpenCL/AMD plugin did legacy
// kHeavyHash only — it cannot produce a possession proof, so an OpenCL worker's blocks are
// rejected post-PoM. It is no longer loaded (dropping its dead --opencl-*/--experimental-amd
// flags with it). AMD PoM lives in Muskwak's Vulkan fork.
const WHITELIST: [&str; 2] = ["libkeryxcuda", "keryxcuda"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

/// Attempt to install the CUDA runtime libraries inference needs, on a Debian/Ubuntu host (HiveOS).
///
/// OPoI GPU inference (the in-process llama engine) needs cuBLAS/cuBLASLt; cuRAND is kept for
/// compatibility. These ship with the CUDA toolkit but not with the bare NVIDIA
/// driver that mining rigs usually have. Rather than forcing miners to run apt by hand, we add
/// the NVIDIA CUDA repo and install `libcublas-12-2` (cuBLAS + cuBLASLt) and `libcurand-12-2`
/// ourselves, then register their directory with ldconfig. Runs as root on HiveOS, so no sudo.
///
/// Version 12-2 (not 12-6) is deliberate: the shipped kernels and the .so are compiled with the
/// CUDA 12.2 toolkit so they JIT on driver >= 535 (typical HiveOS), and the cuBLAS runtime must
/// match that minimum. Installing 12-6 here would pull a runtime needing driver >= 560.
/// Returns true on success.
#[cfg(target_os = "linux")]
fn install_cuda_libs() -> bool {
    use std::process::Command;
    // Only meaningful where apt-get exists (Debian/Ubuntu, incl. HiveOS).
    let has_apt = Command::new("sh").args(["-c", "command -v apt-get"]).status().map(|s| s.success()).unwrap_or(false);
    if !has_apt {
        error!("CUDA lib auto-install needs apt-get (Debian/Ubuntu) — not found on this system.");
        return false;
    }
    // The CUDA libs install into /usr/local/cuda-*/targets/x86_64-linux/lib, which is NOT in
    // the default loader search path. Installing alone is not enough: we must register that
    // directory with ldconfig so dlopen("libcublas.so.12" / "libcurand.so.10") resolves it.
    let script = r#"set -e
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq libcublas-12-2 libcurand-12-2 cuda-cudart-12-2
CUBLAS_PATH=$(find /usr/local /usr/lib -name 'libcublas.so.12' 2>/dev/null | head -1)
if [ -z "$CUBLAS_PATH" ]; then echo "libcublas.so.12 not found after install"; exit 1; fi
echo "$(dirname "$CUBLAS_PATH")" > /etc/ld.so.conf.d/keryx-cuda.conf
ldconfig
ldconfig -p | grep -q libcublas.so.12 || { echo "libcublas still not in loader cache"; exit 1; }
ldconfig -p | grep -q libcurand.so   || { echo "libcurand still not in loader cache"; exit 1; }
ldconfig -p | grep -q libcudart.so.12 || { echo "libcudart still not in loader cache"; exit 1; }
rm -f cuda-keyring.deb"#;
    let output = match Command::new("bash")
        .args(["-c", script])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            error!("CUDA lib auto-install failed to launch: {e}");
            return false;
        }
    };

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if !line.trim().is_empty() {
            info!("CUDA install: {line}");
        }
    }
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        if !line.trim().is_empty() {
            warn!("CUDA install: {line}");
        }
    }

    if output.status.success() {
        true
    } else {
        error!("CUDA lib auto-install failed with status {}", output.status);
        false
    }
}

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    fn api_entry() -> ApiEscrowEntry {
        ApiEscrowEntry {
            coinbase_txid: "AB".repeat(32),
            block_hash: "CD".repeat(32),
            confirm_daa: 10,
            amount_sompi: 20,
            output_index: 1,
        }
    }

    #[test]
    fn recovery_validates_and_normalizes_entries() {
        let (state, total) = recovered_escrow_state(vec![api_entry()]).unwrap();
        assert_eq!(total, 20);
        assert_eq!(state.entries[0].coinbase_txid, "ab".repeat(32));
        assert_eq!(state.entries[0].block_hash, "cd".repeat(32));
    }

    #[test]
    fn recovery_rejects_invalid_network_values() {
        let mut entry = api_entry();
        entry.confirm_daa = -1;
        assert!(recovered_escrow_state(vec![entry]).is_err());

        let mut entry = api_entry();
        entry.output_index = i64::from(u32::MAX) + 1;
        assert!(recovered_escrow_state(vec![entry]).is_err());

        let mut entry = api_entry();
        entry.coinbase_txid = "not-a-hash".into();
        assert!(recovered_escrow_state(vec![entry]).is_err());

        assert!(recovered_escrow_state(vec![api_entry(), api_entry()]).is_err());
    }
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

#[cfg(unix)]
fn redirect_stderr_for_tui(path: &str) -> Result<(), String> {
    use nix::libc::STDERR_FILENO;
    use nix::unistd::dup2;

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open stderr log '{}': {}", path, e))?;

    dup2(file.as_raw_fd(), STDERR_FILENO).map_err(|e| format!("dup2(stderr -> '{}') failed: {}", path, e))?;
    Ok(())
}

#[cfg(not(unix))]
fn redirect_stderr_for_tui(_path: &str) -> Result<(), String> {
    Ok(())
}

extern "C" fn plugin_log_sink(level: u8, msg_ptr: *const u8, msg_len: usize) {
    if msg_ptr.is_null() || msg_len == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(msg_ptr, msg_len) };
    let msg = String::from_utf8_lossy(bytes);
    let lvl = match level {
        keryx_miner::PLUGIN_LOG_ERROR => log::Level::Error,
        keryx_miner::PLUGIN_LOG_WARN => log::Level::Warn,
        keryx_miner::PLUGIN_LOG_INFO => log::Level::Info,
        keryx_miner::PLUGIN_LOG_DEBUG => log::Level::Debug,
        keryx_miner::PLUGIN_LOG_TRACE => log::Level::Trace,
        _ => log::Level::Info,
    };
    log::log!(lvl, "{}", msg);
}

/// Query GPU stats via nvidia-smi and warn on power/VRAM issues for the selected model tier.
///
/// VRAM requirements (GGUF weights only, not counting CUDA workspace):
///   Qwen3-8B-abliterated →  ~4.6 GB
///   Mistral-7B-v0.3 →  ~5.9 GB
///   GLM-4-9B        →  ~8.3 GB
///   Qwen3.6-27B     → ~16.5 GB  (requires ≥24 GB card)
///   Kimi-Linear-48B → ~29.7 GB  (requires ≥32 GB card)
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.limit,power.max_limit,memory.total", "--format=csv,noheader,nounits"])
        .output();

    // nvidia-smi prints one line per GPU; the power + VRAM check applies to GPU 0
    // (the device the miner mines/serves on).
    let (current_w, vram_mb) = match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut cur = 0u32;
            let mut vram = 0u64;
            for (i, line) in s.trim().lines().take(1).enumerate() {
                let mut parts = line.split(',');
                let line_cur: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
                let _max: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
                let line_vram: u64 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
                if i == 0 {
                    cur = line_cur as u32;
                }
                vram += line_vram;
            }
            (cur, vram)
        }
        _ => return,
    };

    // VRAM sufficiency for the selected tier (Q4_K_M weights + KV cache + CUDA workspace).
    // Insufficient VRAM means GPU inference for this tier will OOM. This is non-fatal — a
    // host/CPU path can still serve it — so warn rather than error, and do NOT then claim the
    // model is "ready" on the same GPU (the contradictory ERROR-then-ready pair).
    let (model_label, min_vram_mb): (&str, u64) = if needs_very_high {
        ("Kimi-Linear-48B (--very-high)", 30_000)
    } else if needs_high {
        ("Qwen3.6-27B (--high)", 20_000)
    } else {
        ("GLM-4-9B (default)", 10_000)
    };

    if vram_mb < min_vram_mb {
        log::warn!(
            "⚠  {} needs ≥{} GB VRAM but only {} GB on this GPU — GPU inference for this tier \
             will OOM. Use a smaller tier (--high Qwen3.6-27B / --light Mistral-7B-v0.3) or let \
             the per-GPU assignment downgrade it.",
            model_label,
            min_vram_mb / 1024,
            vram_mb / 1024,
        );
    } else {
        log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
    }
}

/// Per-tier VRAM floor (MB) for **auto-assignment** — the practical minimum to load that tier's
/// model (weights + KV cache + CUDA workspace). Distinct from `ModelSpec.min_vram_mb`, which is 0
/// for the smallest tiers (never gated out of `ai:cap`) and so can't rank tier 0 vs 1 by VRAM.
/// Largest tier first, so a device picks the biggest tier it can hold. Era-aware: staging targets
/// the latest scheduled lineup (`spec_for_tier`), so the floors MUST rank against that lineup's
/// models — with H6 staged the tier-0 model is 8 GB-class and tier 2 is 16 GB-class.
fn pom_tier_ladder() -> &'static [(keryx_miner::models::Tier, u64)] {
    if keryx_miner::models::h6_staged() {
        &[
            (keryx_miner::models::Tier::VeryHigh, 28_000),
            (keryx_miner::models::Tier::High, 22_000),
            (keryx_miner::models::Tier::Default, 15_000),
            (keryx_miner::models::Tier::Light, 11_000),
            (keryx_miner::models::Tier::VeryLight, 7_000),
        ]
    } else {
        &[
            (keryx_miner::models::Tier::VeryHigh, 28_000),
            (keryx_miner::models::Tier::High, 22_000),
            (keryx_miner::models::Tier::Default, 11_000),
            (keryx_miner::models::Tier::Light, 7_000),
            // Qwen3-8B tier 0 loads in ~5,409 MiB @ ctx 4096; floor sits ~300 MiB below its 6 GB
            // min_vram so a 6 GB card reporting slightly under 6000 (total_mem) keeps tier 0, and
            // ~300 MiB above the load need for OOM margin.
            (keryx_miner::models::Tier::VeryLight, 5_700),
        ]
    }
}

/// Ordinal rank of a tier (VeryLight=0 … VeryHigh=4), for the "≤ ceiling" comparison.
fn tier_rank(t: keryx_miner::models::Tier) -> u8 {
    use keryx_miner::models::Tier::*;
    match t {
        VeryLight => 0,
        Light => 1,
        Default => 2,
        High => 3,
        VeryHigh => 4,
    }
}

/// Parse a `--force-model` tier name. None on an unrecognised token.
fn parse_tier_name(s: &str) -> Option<keryx_miner::models::Tier> {
    use keryx_miner::models::Tier;
    match s.trim().to_ascii_lowercase().as_str() {
        "very-light" | "verylight" | "very_light" => Some(Tier::VeryLight),
        "light" => Some(Tier::Light),
        "default" => Some(Tier::Default),
        "high" => Some(Tier::High),
        "very-high" | "veryhigh" | "very_high" => Some(Tier::VeryHigh),
        _ => None,
    }
}

/// Assign each CUDA device the highest PoM tier that (a) is ≤ the `ceiling` flag and (b) fits its
/// VRAM — so a heterogeneous rig mines a different tier per GPU instead of the lowest common
/// denominator, small cards downgrade instead of failing, and big cards are not pushed past the
/// user's ceiling. VRAM is CUDA-driver-sourced (`query_all_gpus_vram`), so `device_id`s match the
/// devices the walk loads onto. Empty when PoM is disabled on this network; a single device-0 entry
/// (highest tier ≤ ceiling) when no CUDA device is enumerated, so the fallback walk still has a tier.
/// `forced` (from `--force-model`, indexed by device id) wins per-card over both the ceiling and
/// the VRAM floor — the tier is still proven by the walk on the real weights, so forcing is only
/// ever self-penalising (OOM / partial-possession slowdown / bigger burn on a smaller tier).
/// Returns `(device_id, hardware_tier, spec)` per GPU: the stable VRAM-picked hardware tier and
/// the single model that tier mines and serves.
fn assign_pom_tiers(
    ceiling: keryx_miner::models::Tier,
    forced: &[Option<keryx_miner::models::Tier>],
) -> Vec<(u32, keryx_miner::models::Tier, &'static keryx_miner::models::ModelSpec)> {
    if keryx_miner::pom::pom_activation_daa() == u64::MAX {
        return Vec::new(); // PoM disabled on this network — serve only, don't mine possession.
    }
    let ceiling_rank = tier_rank(ceiling);
    // Assignment floor + tier + model for each tier ≤ ceiling, largest first.
    let candidates: Vec<(u64, keryx_miner::models::Tier, &'static keryx_miner::models::ModelSpec)> = pom_tier_ladder()
        .iter()
        .filter(|(t, _)| tier_rank(*t) <= ceiling_rank)
        .map(|(t, floor)| (*floor, *t, keryx_miner::models::spec_for_tier(*t)))
        .collect();

    let pick = |vram_mb: u64| -> Option<(keryx_miner::models::Tier, &'static keryx_miner::models::ModelSpec)> {
        candidates.iter().copied().find(|(floor, _, _)| *floor <= vram_mb).map(|(_, t, s)| (t, s))
    };

    let devices = keryx_miner::pom_gpu::query_all_gpus_vram();
    if devices.is_empty() {
        // No enumeration: a forced first entry still wins for the fallback device-0 tier.
        if let Some(t) = forced.first().copied().flatten() {
            log::warn!(
                "No CUDA device enumerated for PoM tier assignment — assigning the forced tier to device 0 (fallback)."
            );
            return vec![(0u32, t, keryx_miner::models::spec_for_tier(t))];
        }
        log::warn!(
            "No CUDA device enumerated for PoM tier assignment — assigning the ceiling tier to device 0 (fallback)."
        );
        return candidates.first().map(|(_, t, s)| vec![(0u32, *t, *s)]).unwrap_or_default();
    }
    let mut out = Vec::with_capacity(devices.len());
    for (id, vram_mb) in devices {
        match forced.get(id as usize).copied() {
            Some(Some(t)) => {
                let spec = keryx_miner::models::spec_for_tier(t);
                log::info!(
                    "PoM: GPU {} → {} (--force-model; VRAM floor bypassed — an undersized card will OOM loading it).",
                    id, spec.dir_name
                );
                out.push((id as u32, t, spec));
                continue;
            }
            Some(None) => log::warn!(
                "PoM: GPU {}: --force-model entry unrecognised — using auto (names: very-light|light|default|high|very-high).",
                id
            ),
            None => {}
        }
        match pick(vram_mb) {
            Some((t, spec)) => out.push((id as u32, t, spec)),
            None => {
                log::warn!("PoM: GPU {} ({} MB VRAM) fits no tier ≤ the ceiling — it will not mine PoM.", id, vram_mb)
            }
        }
    }
    out
}

/// The served lineup (drives `ai:cap` + prefetch) = the distinct models across all GPU assignments.
/// Falls back to the `ceiling` tier's model when nothing was assigned (PoM disabled, or every GPU too
/// small), so `ai:cap`/inference still have a lineup.
fn lineup_from_assignments(
    assignments: &[(u32, keryx_miner::models::Tier, &'static keryx_miner::models::ModelSpec)],
    ceiling: keryx_miner::models::Tier,
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    let mut union: Vec<&'static keryx_miner::models::ModelSpec> = Vec::new();
    for (_, _, spec) in assignments {
        if !union.iter().any(|s| s.model_id == spec.model_id) {
            union.push(*spec);
        }
    }
    if union.is_empty() {
        return Box::leak(vec![keryx_miner::models::spec_for_tier(ceiling)].into_boxed_slice());
    }
    // Leaked once at startup to keep the &'static API of init_supported / prefetch.
    Box::leak(union.into_boxed_slice())
}

/// The prefetch lineup = every model each assigned tier may mine across the currently-scheduled
/// eras (`pom_models_all_eras`), so the H5 crossing hot-swaps without a mid-run download stall.
/// While H5 is unscheduled this equals `lineup_from_assignments` (only the current-era model).
fn prefetch_lineup_from_assignments(
    assignments: &[(u32, keryx_miner::models::Tier, &'static keryx_miner::models::ModelSpec)],
    ceiling: keryx_miner::models::Tier,
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    let mut union: Vec<&'static keryx_miner::models::ModelSpec> = Vec::new();
    for (_, gpu_tier, _) in assignments {
        for spec in keryx_miner::models::pom_models_all_eras(*gpu_tier) {
            if !union.iter().any(|s| s.model_id == spec.model_id) {
                union.push(spec);
            }
        }
    }
    if union.is_empty() {
        for spec in keryx_miner::models::pom_models_all_eras(ceiling) {
            if !union.iter().any(|s| s.model_id == spec.model_id) {
                union.push(spec);
            }
        }
    }
    Box::leak(union.into_boxed_slice())
}

async fn get_client(
    keryxd_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    escrow_privkey: Option<String>,
    escrow_state_file: String,
    ipfs_url: String,
) -> Result<Box<dyn Client + 'static>, Error> {
    if keryxd_address.starts_with("stratum+tcp://") {
        let (_schema, address) = keryxd_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            ipfs_url.clone(),
        )
        .await?)
    } else if keryxd_address.starts_with("grpc://") {
        Ok(KeryxdHandler::connect(
            keryxd_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            escrow_privkey,
            escrow_state_file,
            ipfs_url,
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

#[derive(Debug)]
struct EscrowFlushError(String);

impl std::fmt::Display for EscrowFlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EscrowFlushError {}

#[derive(serde::Deserialize)]
struct ApiEscrowEntry {
    coinbase_txid: String,
    block_hash: String,
    confirm_daa: i64,
    amount_sompi: i64,
    output_index: i64,
}

fn recovered_escrow_state(api_entries: Vec<ApiEscrowEntry>) -> Result<(escrow::EscrowState, u64), String> {
    let mut entries = Vec::with_capacity(api_entries.len());
    let mut outpoints = std::collections::HashSet::with_capacity(api_entries.len());
    let mut total_sompi = 0u64;
    for api_entry in api_entries {
        if api_entry.coinbase_txid.len() != 64 || !api_entry.coinbase_txid.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("Recovery API returned an invalid coinbase transaction ID".into());
        }
        if api_entry.block_hash.len() != 64 || !api_entry.block_hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("Recovery API returned an invalid block hash".into());
        }
        let confirm_daa = u64::try_from(api_entry.confirm_daa)
            .map_err(|_| "Recovery API returned a negative confirmation DAA score")?;
        let amount_sompi =
            u64::try_from(api_entry.amount_sompi).map_err(|_| "Recovery API returned a negative escrow amount")?;
        if amount_sompi == 0 {
            return Err("Recovery API returned a zero escrow amount".into());
        }
        let output_index =
            u32::try_from(api_entry.output_index).map_err(|_| "Recovery API returned an invalid output index")?;
        let coinbase_txid = api_entry.coinbase_txid.to_ascii_lowercase();
        if !outpoints.insert((coinbase_txid.clone(), output_index)) {
            return Err("Recovery API returned a duplicate escrow outpoint".into());
        }
        total_sompi = total_sompi.checked_add(amount_sompi).ok_or("Recovered escrow amount overflow")?;
        entries.push(escrow::EscrowEntry {
            coinbase_txid,
            block_hash: api_entry.block_hash.to_ascii_lowercase(),
            confirm_daa,
            amount_sompi,
            output_index,
            claimed: false,
            slashed: false,
            orphan_slashed: false,
            orphan_retries: 0,
            orphan_retry_after_daa: None,
            submit_retries: 0,
            batch_cap: 0,
            cap_set_daa: 0,
            is_inference: false,
            csv_window: escrow::csv_window_for_daa(confirm_daa),
        });
    }
    Ok((escrow::EscrowState { entries }, total_sompi))
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
    stats: Arc<MinerStats>,
    shutdown_requested: Arc<AtomicBool>,
) -> Result<(), Error> {
    let ipfs_url = opt.ipfs_url.clone();
    tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url)).await.ok();

    let mut client = get_client(
        opt.keryxd_address.clone(),
        opt.mining_address.clone().unwrap_or_default(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        escrow_privkey,
        opt.escrow_state_file.clone(),
        opt.ipfs_url.clone(),
    )
    .await?;

    if opt.devfund_percent > 0 {
        client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
    }
    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager, stats);
    let listen_result = tokio::select! {
        listen_res = client.listen(&mut miner_manager) => {
            listen_res
        }
        _ = wait_for_shutdown(shutdown_requested) => {
            info!("Shutdown requested, stopping client listen loop");
            Ok(())
        }
    };
    // Flush funds-critical client state before potentially blocking on worker shutdown.
    let mut flush_error = None;
    for attempt in 1..=3 {
        match client.flush_escrow_state() {
            Ok(()) => {
                flush_error = None;
                break;
            }
            Err(e) => {
                flush_error = Some(e.to_string());
                warn!("Escrow final flush attempt {}/3 failed: {}", attempt, e);
                if attempt < 3 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
    drop(client);
    drop(miner_manager);
    if let Some(error) = flush_error {
        return Err(Box::new(EscrowFlushError(error)));
    }
    listen_result
}

async fn wait_for_shutdown(shutdown_requested: Arc<AtomicBool>) {
    while !shutdown_requested.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Tokio async worker count. The miner's async workload is tiny (one gRPC/stratum connection +
/// a few tasks and timers), so we cap workers instead of spawning one per logical CPU — dozens of
/// idle executor threads on a many-core rig are pure scheduler overhead. Override with
/// KERYX_ASYNC_WORKERS.
fn tokio_worker_threads() -> usize {
    std::env::var("KERYX_ASYNC_WORKERS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2).clamp(1, 8)
}

/// Optional cap for the `spawn_blocking` pool (SLM inference, IPFS upload, model prefetch). Only
/// applied when KERYX_BLOCKING_THREADS is set: the blocking pool spawns lazily and idles out, so
/// tokio's default costs nothing at rest and capping it low would bottleneck parallel multi-model
/// prefetch on multi-GPU rigs.
fn tokio_blocking_threads() -> Option<usize> {
    std::env::var("KERYX_BLOCKING_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).map(|n| n.clamp(2, 64))
}

fn main() -> Result<(), Error> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(tokio_worker_threads()).enable_all();
    if let Some(n) = tokio_blocking_threads() {
        builder.max_blocking_threads(n);
    }
    let rt = builder.build()?;
    rt.block_on(run())
}

async fn run() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    let (app, mut plugin_manager): (App, PluginManager) = keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.get_matches();

    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;

    if opt.resident_tree {
        std::env::set_var("KERYX_RESIDENT_TREE", "1");
    }

    // Model storage root is configurable: explicit --models-dir wins, otherwise
    // --hiveos defaults to a stable shared HiveOS path.
    if let Some(dir) = opt.models_dir.as_ref() {
        std::env::set_var("KERYX_MODELS_DIR", dir);
    } else if opt.hiveos && std::env::var_os("KERYX_MODELS_DIR").is_none() {
        std::env::set_var("KERYX_MODELS_DIR", "/hive/miners/custom/models");
    }

    let is_tty = std::io::stdout().is_terminal();
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    {
        let shutdown_requested = Arc::clone(&shutdown_requested);
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
                loop {
                    match terminate.as_mut() {
                        Ok(terminate) => {
                            tokio::select! {
                                _ = tokio::signal::ctrl_c() => {}
                                _ = terminate.recv() => {}
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to register SIGTERM handler: {}", e);
                            if tokio::signal::ctrl_c().await.is_err() {
                                return;
                            }
                        }
                    }
                    if shutdown_requested.swap(true, Ordering::AcqRel) {
                        eprintln!("Second shutdown signal received; forcing exit");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(unix))]
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    return;
                }
                if shutdown_requested.swap(true, Ordering::AcqRel) {
                    eprintln!("Second shutdown signal received; forcing exit");
                    std::process::exit(1);
                }
            }
        });
    }
    let ui_state = if is_tty { Some(Arc::new(UiState::new())) } else { None };
    let plain_log_path = opt.plain_log_file.clone().or_else(|| std::env::var("KERYX_PLAIN_LOG_FILE").ok());
    let plain_log_file =
        plain_log_path.and_then(|path| match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(file),
            Err(e) => {
                eprintln!("Failed to open plain log file '{}': {}", path, e);
                None
            }
        });
    init_logging(opt.log_level(), ui_state.clone(), !is_tty, plain_log_file)?;
    plugin_manager.set_log_sink(Some(plugin_log_sink));
    for warning in plugin_manager.drain_startup_warnings() {
        warn!("{}", warning);
    }

    if let Ok(dir) = std::env::var("KERYX_MODELS_DIR") {
        info!("Models directory: {}", dir);
    }

    if is_tty {
        let stderr_path = std::env::var("KERYX_STDERR_LOG_FILE").ok().unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{}/.keryx/stderr.log", home)
        });
        if let Some(parent) = std::path::Path::new(&stderr_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match redirect_stderr_for_tui(&stderr_path) {
            Ok(()) => info!("TUI: stderr redirected to {}", stderr_path),
            Err(e) => warn!("TUI: failed to redirect stderr ({})", e),
        }
    }

    let worker_count = plugin_manager.process_options(&matches)?;
    for warning in plugin_manager.drain_startup_warnings() {
        warn!("{}", warning);
    }

    let stats = Arc::new(MinerStats::new(opt.hiveos));
    stats.set_mining_address(opt.mining_address.clone());
    stats.set_api_port(opt.stats_port);
    let _ui_guard =
        ui_state.as_ref().map(|ui| spawn_ui(Arc::clone(&stats), Arc::clone(ui), Arc::clone(&shutdown_requested)));

    match spawn_stats_server(Arc::clone(&stats), opt.stats_bind.clone(), opt.stats_port) {
        Ok(_handle) => {
            info!("Stats API listening on {}:{}", opt.stats_bind, opt.stats_port);
        }
        Err(e) => {
            warn!("Failed to start stats API on {}:{} ({})", opt.stats_bind, opt.stats_port, e);
        }
    }

    info!("=================================================================================");
    info!("                 Keryx-Miner GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address.as_deref().unwrap_or("(recovery mode)"));
    info!("=================================================================================");

    // Recovery mode: rebuild escrow_state.json from the Keryx public API, then exit.
    // Must run before escrow key loading to avoid creating a new random key on disk.
    // Uses escrow.key to derive the pubkey — only claimable UTXOs are returned.
    if opt.recover_escrow {
        let escrow_privkey = match escrow::load_key(&opt.escrow_key_file) {
            Ok(k) => k,
            Err(e) => {
                error!("{}", e);
                return Err(e.into());
            }
        };
        let pubkey_hex = match escrow::pubkey_hex_from_privkey(&escrow_privkey) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to derive pubkey from escrow key: {}", e);
                return Err(e.into());
            }
        };
        let url = format!("{}/api/v1/escrow/{}", opt.recover_escrow_api.trim_end_matches('/'), pubkey_hex);
        info!("Querying escrow UTXOs from {}", url);

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone).call().map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let (state, total_sompi) = recovered_escrow_state(api_entries)?;
        let count = state.entries.len();
        if shutdown_requested.load(Ordering::Acquire) {
            return Err("Shutdown requested before recovered escrow state was saved".into());
        }
        escrow::save_state_atomic(std::path::Path::new(&opt.escrow_state_file), &state)?;

        info!("Recovered {} escrow entries — claimable: {:.4} KRX", count, total_sompi as f64 / 1e8);
        info!("State saved to '{}'.", opt.escrow_state_file);
        return Ok(());
    }

    // Resolve OPoI escrow private key (once, before the reconnect loop).
    let escrow_privkey: Option<String> = match escrow::load_or_generate_key(&opt.escrow_key_file) {
        Ok(k) => {
            info!("OPoI: escrow key loaded from '{}'.", opt.escrow_key_file);
            Some(k)
        }
        Err(e) => {
            error!("Failed to load/generate OPoI escrow key: {}", e);
            return Err(e.into());
        }
    };

    // Phase-3 OPoI / PoM: load inference models before mining starts. Under PoM each tier
    // mines AND serves exactly ONE model (1 GPU = 1 tier); multi-tier coverage is a network
    // property, not a per-GPU one.
    //   --very-light → Qwen3-8B-abliterated
    //   --light      → Mistral-7B-v0.3
    //   (no flag)    → GLM-4-9B      [default]
    //   --high       → Qwen3.6-27B
    //   --very-high  → Kimi-Linear-48B

    // Per-card tier overrides (--force-model, CUDA-driver order). Parsed once here — the power
    // warning below and the per-GPU assignment both need them.
    let forced_tiers: Vec<Option<keryx_miner::models::Tier>> =
        opt.force_model.as_deref().map(|s| s.split(',').map(parse_tier_name).collect()).unwrap_or_default();
    if let Some(raw) = opt.force_model.as_deref() {
        info!(
            "--force-model: {} — per-card tier override (VRAM floor bypassed; unlisted/extra cards use auto).",
            raw.trim()
        );
    }
    let forced_max_rank = forced_tiers.iter().flatten().map(|t| tier_rank(*t)).max();

    // Warn if GPU power limit is below safe threshold for the selected model tier.
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads.
    check_gpu_power_limit(
        opt.high || opt.very_high || forced_max_rank.is_some_and(|r| r >= 3),
        opt.very_high || forced_max_rank.is_some_and(|r| r >= 4),
    );

    let tier = if opt.very_high {
        info!("--very-high mode: top tier — mines Kimi-Linear-48B under PoM.");
        keryx_miner::models::Tier::VeryHigh
    } else if opt.high {
        info!("--high mode: high tier — mines Qwen3.6-27B under PoM.");
        keryx_miner::models::Tier::High
    } else if opt.light {
        info!("--light mode: light tier — mines Mistral-7B-v0.3 under PoM.");
        keryx_miner::models::Tier::Light
    } else if opt.very_light {
        info!("--very-light mode: smallest tier — mines Qwen3-8B-abliterated under PoM.");
        keryx_miner::models::Tier::VeryLight
    } else {
        info!("default mode: mines GLM-4-9B under PoM.");
        keryx_miner::models::Tier::Default
    };
    // Per-GPU PoM assignment: each CUDA device mines the highest tier ≤ the flag ceiling that its
    // VRAM holds (small cards downgrade instead of failing; big cards are not pushed past the
    // ceiling). --force-model entries win per-card over both. VRAM is CUDA-driver-sourced so
    // device_ids match the devices the walk loads onto.
    let pom_assignments = assign_pom_tiers(tier, &forced_tiers);
    // The served/announced lineup (ai:cap) = the current-era models across all GPUs.
    let specs = lineup_from_assignments(&pom_assignments, tier);
    keryx_miner::slm::init_supported(specs);
    log::debug!("OPoI Phase-3 active — {} model(s) staged.", specs.len());
    // Prefetch BOTH scheduled eras so the H5 crossing hot-swaps without a mid-run download stall
    // (equals `specs` while H5 is unscheduled). Block until every such model is downloaded before
    // mining: never start hashing while a model is still downloading.
    let prefetch_specs = prefetch_lineup_from_assignments(&pom_assignments, tier);
    match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(prefetch_specs)).await {
        Ok(Ok(())) => info!("Model files ready ({}) — starting mining.", prefetch_specs.len()),
        Ok(Err(e)) => {
            error!("Model prefetch failed — refusing to mine without the lineup: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            error!("Model prefetch task panicked: {}", e);
            return Err(e.into());
        }
    }
    // PoM possession setup is fully LAZY: nothing GPU- or host-heavy happens at boot. The
    // possession index AND the GPU walk are built by the mining loop the first time PoM is
    // active for the block being mined. Here we only record cheap config.
    if !pom_assignments.is_empty() {
        // The tier *index* is computed per block from the block DAA (None below the H4 gate —
        // this binary refuses to mine pre-H4 blocks), so only the model is recorded here. The
        // fixed hardware tier is recorded too, so the H5 era crossing can hot-swap tier 0's model.
        for (device_id, gpu_tier, spec) in &pom_assignments {
            let gpath = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            keryx_miner::pom_gpu::set_mining_tier(*device_id, spec.model_id, gpath);
            keryx_miner::pom_gpu::set_device_tier(*device_id, *gpu_tier);
            info!(
                "PoM: GPU {} → {} (index + GPU walk load lazily once the lineup is active).",
                device_id, spec.dir_name
            );
        }
    }

    // Verify GPU inference works before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    info!("Probing GPU inference (cuBLAS + llama engine) before mining…");
    match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
        Ok(keryx_miner::slm::GpuProbe::Ok) => {
            info!("GPU inference verified — cuBLAS and the llama engine loaded successfully.")
        }
        Ok(keryx_miner::slm::GpuProbe::NoCuda) => {
            error!("No CUDA device detected — OPoI inference is GPU-only and is mandatory, cannot mine.");
            return Err("No CUDA device — cannot start OPoI mining".into());
        }
        Ok(keryx_miner::slm::GpuProbe::EngineMissing(why)) => {
            error!("Inference engine unavailable: {}", why);
            error!("OPoI inference is mandatory: mining without it would answer no request at all.");
            error!("Restore the library shipped with this release next to the miner binary, then restart.");
            return Err("llama inference engine unavailable — cannot start OPoI mining".into());
        }
        Ok(keryx_miner::slm::GpuProbe::CublasMissing) => {
            warn!("CUDA GPU detected but a CUDA runtime lib is missing — installing them automatically (one-time)…");
            #[cfg(target_os = "linux")]
            {
                let installed = tokio::task::spawn_blocking(install_cuda_libs).await.unwrap_or(false);
                if !installed {
                    error!("Automatic CUDA lib install failed — install them manually then restart:");
                    error!("  apt-get install -y libcublas-12-2 libcurand-12-2 cuda-cudart-12-2");
                    return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
                }
                // Re-probe in-process. The dynamic loader may still hold a stale cache, so if
                // the freshly-installed libs aren't picked up here, exit cleanly and let the
                // supervisor (HiveOS/PM2) relaunch us with a fresh loader cache.
                match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
                    Ok(keryx_miner::slm::GpuProbe::Ok) => {
                        info!("CUDA libs installed — GPU inference verified, starting mining.");
                    }
                    // Restarting cannot conjure a library that is not on disk; fail here rather
                    // than hand the supervisor a restart loop.
                    Ok(keryx_miner::slm::GpuProbe::EngineMissing(why)) => {
                        error!("CUDA libs installed, but the inference engine is unavailable: {}", why);
                        error!("Restore the library shipped with this release next to the miner binary, then restart.");
                        return Err("llama inference engine unavailable — cannot start OPoI mining".into());
                    }
                    _ => {
                        info!("CUDA libs installed successfully — restarting miner to activate them.");
                        std::process::exit(0);
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                error!("CUDA GPU detected but a CUDA runtime lib failed to load — install the CUDA 12.6 toolkit and restart.");
                return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
            }
        }
        Err(e) => {
            error!("GPU probe task panicked: {}", e);
            return Err(e.into());
        }
    }
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16));
    if opt.devfund_percent > 0 {
        info!(
            "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
            opt.devfund_percent / 100,
            opt.devfund_percent % 100,
            opt.devfund_address
        );
    }
    loop {
        if shutdown_requested.load(Ordering::Acquire) {
            info!("Shutdown requested, exiting miner main loop");
            break;
        }
        match client_main(
            &opt,
            block_template_ctr.clone(),
            &plugin_manager,
            escrow_privkey.clone(),
            Arc::clone(&stats),
            Arc::clone(&shutdown_requested),
        )
        .await
        {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) if e.downcast_ref::<EscrowFlushError>().is_some() => {
                error!("Funds-critical escrow state could not be persisted: {}", e);
                return Err(e);
            }
            Err(e) => error!("Client closed with error {:?}", e),
        }
        if shutdown_requested.load(Ordering::Acquire) {
            info!("Shutdown requested, skipping reconnect");
            break;
        }
        info!("Client closed, reconnecting");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}
