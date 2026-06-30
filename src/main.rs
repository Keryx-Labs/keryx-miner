#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod miner;
mod pow;
mod target;
mod watch;

const WHITELIST: [&str; 4] = ["libkeryxcuda", "libkeryxopencl", "keryxcuda", "keryxopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

/// Attempt to install the CUDA runtime libraries candle needs, on a Debian/Ubuntu host (HiveOS).
///
/// OPoI GPU inference needs cuBLAS, cuBLASLt and cuRAND — candle creates handles for all three
/// when it opens the CUDA device. These ship with the CUDA toolkit but not with the bare NVIDIA
/// driver that mining rigs usually have. Rather than forcing miners to run apt by hand, we add
/// the NVIDIA CUDA repo and install `libcublas-12-2` (cuBLAS + cuBLASLt) and `libcurand-12-2`
/// ourselves, then register their directory with ldconfig. Runs as root on HiveOS, so no sudo.
///
/// Version 12-2 (not 12-6) is deliberate: the binary's candle kernels are compiled with the
/// CUDA 12.2 toolkit so they JIT on driver >= 535 (typical HiveOS), and the cuBLAS runtime must
/// match that minimum. Installing 12-6 here would pull a runtime needing driver >= 560.
/// Returns true on success.
#[cfg(target_os = "linux")]
fn install_cuda_libs() -> bool {
    use std::process::Command;
    // Only meaningful where apt-get exists (Debian/Ubuntu, incl. HiveOS).
    let has_apt = Command::new("sh")
        .args(["-c", "command -v apt-get"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
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
apt-get install -y -qq libcublas-12-2 libcurand-12-2
CUBLAS_PATH=$(find /usr/local /usr/lib -name 'libcublas.so.12' 2>/dev/null | head -1)
if [ -z "$CUBLAS_PATH" ]; then echo "libcublas.so.12 not found after install"; exit 1; fi
echo "$(dirname "$CUBLAS_PATH")" > /etc/ld.so.conf.d/keryx-cuda.conf
ldconfig
ldconfig -p | grep -q libcublas.so.12 || { echo "libcublas still not in loader cache"; exit 1; }
ldconfig -p | grep -q libcurand.so   || { echo "libcurand still not in loader cache"; exit 1; }
rm -f cuda-keyring.deb"#;
    Command::new("bash")
        .args(["-c", script])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
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

/// Query GPU stats via nvidia-smi and warn on power/VRAM issues for the selected model tier.
///
/// VRAM requirements (GGUF Q4_K_M weights only, not counting CUDA workspace):
///   Gemma-3-4B      →  ~2.7 GB
///   Dolphin-8B      →  ~4.9 GB
///   Qwen3-32B       → ~19.5 GB  (requires ≥24 GB card)
///   Llama-3.3-70B   → ~42.5 GB  (requires ≥48 GB card)
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.limit,power.max_limit,memory.total",
            "--format=csv,noheader,nounits",
        ])
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
        ("Llama-3.3-70B (--very-high)", 46_000)
    } else if needs_high {
        ("Qwen3-32B (--high)", 20_000)
    } else {
        ("Dolphin-8B (default)", 8_000)
    };

    if vram_mb < min_vram_mb {
        log::warn!(
            "⚠  {} needs ≥{} GB VRAM but only {} GB on this GPU — GPU inference for this tier \
             will OOM. Use a smaller tier (--high Qwen3-32B / --light Gemma-3-4B) or serve it \
             via a host/CPU path.",
            model_label,
            min_vram_mb / 1024,
            vram_mb / 1024,
        );
    } else {
        log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
    }
}

/// GPU 0 total VRAM (MB) via nvidia-smi, or None when nvidia-smi is unavailable or
/// unparseable (e.g. AMD-only machines). GPU 0 is the device the miner mines/serves on.
fn query_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
}

/// OPoI capability gate (layer A): drop the models this machine cannot actually
/// serve on GPU 0, so the `ai:cap` announcement never promises a model the miner
/// would fail to load. Skipped when nvidia-smi is unavailable (CPU-fallback setups
/// keep working).
fn filter_specs_by_vram(
    specs: &'static [&'static keryx_miner::models::ModelSpec],
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    let Some(gpu0_mb) = query_vram_mb() else {
        log::warn!("Cannot query GPU VRAM (nvidia-smi) — skipping the model capability gate.");
        return specs;
    };
    let kept: Vec<&'static keryx_miner::models::ModelSpec> = specs
        .iter()
        .copied()
        .filter(|spec| {
            if spec.min_vram_mb <= gpu0_mb {
                true
            } else {
                log::warn!(
                    "✗  '{}' needs ≥{} MB VRAM but only {} MB on GPU 0 — model NOT announced (ai:cap) and not downloaded.",
                    spec.name,
                    spec.min_vram_mb,
                    gpu0_mb,
                );
                false
            }
        })
        .collect();
    if kept.len() == specs.len() {
        specs
    } else {
        // Leaked once at startup to keep the &'static API of init_supported.
        Box::leak(kept.into_boxed_slice())
    }
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

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
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
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

/// Tokio async worker count. Async workload is minimal (gRPC/stratum + a few tasks);
/// default 2 avoids spawning one worker per logical CPU on large rigs.
fn tokio_worker_threads() -> usize {
    std::env::var("KERYX_ASYNC_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .clamp(1, 4)
}

/// Cap for `spawn_blocking` (inference, IPFS, model prefetch). Override with KERYX_BLOCKING_THREADS.
fn tokio_blocking_threads() -> usize {
    std::env::var("KERYX_BLOCKING_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .clamp(2, 16)
}

fn main() -> Result<(), Error> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(tokio_worker_threads())
        .max_blocking_threads(tokio_blocking_threads())
        .enable_all()
        .build()?;
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

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
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

        #[derive(serde::Deserialize)]
        struct ApiEscrowEntry {
            coinbase_txid: String,
            block_hash: String,
            confirm_daa: i64,
            amount_sompi: i64,
            output_index: i64,
        }

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone)
                .call()
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let entries: Vec<escrow::EscrowEntry> = api_entries
            .into_iter()
            .map(|a| escrow::EscrowEntry {
                coinbase_txid: a.coinbase_txid,
                block_hash: a.block_hash,
                confirm_daa: a.confirm_daa as u64,
                amount_sompi: a.amount_sompi as u64,
                output_index: a.output_index as u32,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                is_inference: false,
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!(
            "Recovered {} escrow entries — claimable: {:.4} KRX",
            count,
            total_sompi as f64 / 1e8
        );
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
    //   (no flag)    → Dolphin-8B   [default]
    //   --light      → Gemma-3-4B
    //   --high       → Qwen3-32B
    //   --very-high  → Llama-3.3-70B

    // Warn if GPU power limit is below safe threshold for the selected model tier.
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads.
    check_gpu_power_limit(opt.high || opt.very_high, opt.very_high);

    let tier = if opt.very_high {
        info!("--very-high mode: top tier — mines Llama-3.3-70B under PoM.");
        keryx_miner::models::Tier::VeryHigh
    } else if opt.high {
        info!("--high mode: high tier — mines Qwen3-32B under PoM.");
        keryx_miner::models::Tier::High
    } else if opt.light {
        info!("--light mode: baseline tier — mines Gemma-3-4B under PoM.");
        keryx_miner::models::Tier::Light
    } else if opt.very_light {
        info!("--very-light mode: smallest tier — mines Qwen3-1.7B under PoM (falls back to Gemma-3-4B before H2).");
        keryx_miner::models::Tier::VeryLight
    } else {
        info!("default mode: mines Dolphin-8B under PoM.");
        keryx_miner::models::Tier::Default
    };
    // Stage the FINAL lineup (post-H2) directly — `specs_for(VERY_LIGHT_ACTIVATION_DAA, ..)` always
    // returns the latest models for the tier. The legacy lineup is dead (OPoI-v2 is in the past),
    // and we deliberately do NOT also download the pre-H2 model for a tier whose model changes at
    // H2: the old `--very-high` 70B Q4_K_M is served by nobody (48 GB-only), so a 5090 miner just
    // pulls the new Q2_K_L straight away instead of paying two ~27 GB downloads + a hot-swap. A
    // tier whose post-H2 model isn't consensus-valid yet (very-light, very-high) simply produces no
    // block until H2 (its `pom_tier_index` is None pre-H2) — it idles, no wasted bandwidth.
    let specs_v2 = filter_specs_by_vram(
        keryx_miner::models::specs_for(keryx_miner::models::VERY_LIGHT_ACTIVATION_DAA, tier),
    );
    // PoM: pick the highest tier this miner serves that has a pinned R_T (the model it will
    // mine under possession). Captured before `specs_v2` is consumed; the index is built after
    // prefetch (below). `&'static ModelSpec` is Copy so this survives the moves.
    let pom_spec = if keryx_miner::pom::POM_ACTIVATION_DAA != u64::MAX {
        specs_v2
            .iter()
            .copied()
            .filter(|s| keryx_miner::models::is_pom_model(&s.model_id))
            .max_by_key(|s| s.min_vram_mb)
    } else {
        None
    };
    // Serve the uncensored lineup from the start. set_v2_lineup keeps the readiness-gated
    // crossing swap a consistent no-op (it would swap v2 -> v2).
    keryx_miner::slm::set_v2_lineup(specs_v2);
    keryx_miner::slm::init_supported(specs_v2);
    log::debug!(
        "OPoI Phase-3 active — {} uncensored model(s) staged (legacy lineup dropped, post-fork).",
        specs_v2.len(),
    );
    // Block until the uncensored lineup is fully downloaded before mining: never start hashing
    // while a model this miner will serve is still downloading.
    match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs_v2)).await {
        Ok(Ok(())) => info!("Model files ready — starting mining."),
        Ok(Err(e)) => {
            error!("OPoI v2 prefetch failed — refusing to mine without the post-hardfork lineup: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            error!("Model prefetch task panicked: {}", e);
            return Err(e.into());
        }
    }
    // PoM possession setup is fully LAZY: nothing GPU- or host-heavy happens at boot. During the
    // pre-PoM legacy phase the GPU + host stay free for the legacy lineup (mining + inference start
    // immediately). The possession index AND the GPU walk are built by the mining loop the first
    // time PoM is active (DAA >= POM_ACTIVATION_DAA). Here we only record cheap config.
    if let Some(spec) = pom_spec {
        let gpath = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        // Force the single-device split loader so the mining tier exposes its quant tensors for
        // zero-dup sharing. The PoM tier *index* is computed per block from the block DAA (it
        // shifts at the very-light H2 hardfork), so it is not recorded here — only the model.
        keryx_miner::slm::set_pom_force_split(true);
        keryx_miner::pom_gpu::set_mining_tier(spec.model_id, gpath);
        info!("PoM: configured to mine {} under possession; index + GPU walk load lazily when PoM activates (DAA {}).",
            spec.dir_name, keryx_miner::pom::POM_ACTIVATION_DAA);
    }

    // Verify GPU inference works before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    info!("Probing GPU inference (cuBLAS) before mining…");
    match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
        Ok(keryx_miner::slm::GpuProbe::Ok) => info!("GPU inference verified — cuBLAS loaded successfully."),
        Ok(keryx_miner::slm::GpuProbe::NoCuda) => {
            error!("No CUDA device detected — OPoI inference is GPU-only and is mandatory, cannot mine.");
            return Err("No CUDA device — cannot start OPoI mining".into());
        }
        Ok(keryx_miner::slm::GpuProbe::CublasMissing) => {
            warn!("CUDA GPU detected but a CUDA runtime lib is missing — installing them automatically (one-time)…");
            #[cfg(target_os = "linux")]
            {
                let installed = tokio::task::spawn_blocking(install_cuda_libs).await.unwrap_or(false);
                if !installed {
                    error!("Automatic CUDA lib install failed — install them manually then restart:");
                    error!("  apt-get install -y libcublas-12-2 libcurand-12-2");
                    return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
                }
                // Re-probe in-process. The dynamic loader may still hold a stale cache, so if
                // the freshly-installed libs aren't picked up here, exit cleanly and let the
                // supervisor (HiveOS/PM2) relaunch us with a fresh loader cache.
                match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
                    Ok(keryx_miner::slm::GpuProbe::Ok) => {
                        info!("CUDA libs installed — GPU inference verified, starting mining.");
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
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        info!("Client closed, reconnecting");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
