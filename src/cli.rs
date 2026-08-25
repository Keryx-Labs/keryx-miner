use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "keryx-miner", version, about = "A Keryx high performance GPU miner with OPoI inference\n\nUncensored model tiers — one model per tier (default: Gemma-4-12B):\n  --very-light Qwen3.5-9B-abliterated (Q5_K_M) — 8GB+ VRAM, smallest tier\n  --light      GLM-4-9B (Q6_K) — 12GB+ VRAM\n  (default)    Gemma-4-12B-abliterated (Q6_K) — 16GB+ VRAM\n  --high       Qwen3.6-27B (Q4_K_M) — 24GB+ VRAM\n  --very-high  Kimi-Linear-48B (Q4_K_M) — 32GB+ VRAM", term_width = 0)]
pub struct Opt {
    // ── OPoI / Inference ─────────────────────────────────────────────────────

    #[clap(
        long = "very-light",
        help = "Model tier: Qwen3.5-9B-abliterated (Q5_K_M) — 8GB+ GPU, smallest tier",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "very-high"]
    )]
    pub very_light: bool,

    #[clap(
        long = "light",
        help = "Model tier: GLM-4-9B (Q6_K) — 12GB+ VRAM",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very-light", "high", "very-high"]
    )]
    pub light: bool,

    #[clap(
        long = "high",
        help = "Model tier: Qwen3.6-27B (Q4_K_M) — 24GB+ VRAM",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very-light", "light", "very-high"]
    )]
    pub high: bool,

    #[clap(
        long = "very-high",
        help = "Model tier: Kimi-Linear-48B (Q4_K_M) — 32GB+ VRAM",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very-light", "light", "high"]
    )]
    pub very_high: bool,

    #[clap(
        long = "force-model",
        value_name = "TIER[,TIER...]",
        help = "Force the model tier per GPU (CUDA-driver order, CSV): e.g. --force-model light,very-high \
                → GPU0=light, GPU1=very-high. Names: very-light|light|default|high|very-high. Bypasses the \
                per-card VRAM check (an undersized card will OOM); unlisted/extra cards keep auto best-fit.",
        help_heading = "OPoI / Inference"
    )]
    pub force_model: Option<String>,

    #[clap(
        long = "ipfs-url",
        help = "IPFS Kubo API URL for uploading inference results",
        help_heading = "OPoI / Inference",
        default_value = "http://127.0.0.1:5001"
    )]
    pub ipfs_url: String,

    #[clap(
        long = "models-dir",
        help = "Directory where model files are stored/downloaded (overrides default <exe_dir>/models)",
        help_heading = "OPoI / Inference"
    )]
    pub models_dir: Option<String>,

    #[clap(
        long = "hiveos",
        help = "Enable HiveOS defaults (uses /hive/miners/custom/models when --models-dir is not set)",
        help_heading = "OPoI / Inference"
    )]
    pub hiveos: bool,

    #[clap(
        long = "resident-tree",
        help = "Hold the full Merkle tree in RAM for faster proof build (needs ~2x model size of system RAM; falls back to disk if unavailable)",
        help_heading = "OPoI / Inference"
    )]
    pub resident_tree: bool,

    #[clap(
        long = "escrow-key-file",
        help = "Path to the OPoI escrow private key file (auto-generated if absent)",
        help_heading = "OPoI / Inference",
        default_value = "escrow.key"
    )]
    pub escrow_key_file: String,

    #[clap(
        long = "escrow-cert",
        help = "Escrow delegation cert as 128 hex chars, for setups that cannot drop a file (HiveOS). Wins over --escrow-cert-file",
        help_heading = "OPoI / Inference"
    )]
    pub escrow_cert: Option<String>,

    #[clap(
        long = "escrow-cert-file",
        help = "Path to the escrow delegation cert produced by `keryx-cli delegate-escrow` (required from H6)",
        help_heading = "OPoI / Inference",
        default_value = "escrow.cert"
    )]
    pub escrow_cert_file: String,

    #[clap(
        long = "escrow-state-file",
        help = "Path to the escrow claim state file",
        help_heading = "OPoI / Inference",
        default_value = "escrow_state.json"
    )]
    pub escrow_state_file: String,

    #[clap(
        long = "recover-escrow",
        help = "Rebuild escrow_state.json by querying the Keryx public API. Exits after recovery.",
        help_heading = "OPoI / Inference"
    )]
    pub recover_escrow: bool,

    #[clap(
        long = "recover-escrow-api",
        help = "Base URL of the Keryx API to use for escrow recovery",
        help_heading = "OPoI / Inference",
        default_value = "https://keryx-labs.com"
    )]
    pub recover_escrow_api: String,

    // ── Mining ────────────────────────────────────────────────────────────────

    #[clap(short, long, help = "Enable debug logging level")]
    pub debug: bool,

    #[clap(short = 'a', long = "mining-address", help = "The Keryx address for the miner reward")]
    pub mining_address: Option<String>,

    #[clap(short = 's', long = "keryxd-address", default_value = "127.0.0.1", help = "The IP of the keryxd instance")]
    pub keryxd_address: String,

    #[clap(long = "devfund-percent", help = "The percentage of blocks to send to the devfund (minimum 2%)", default_value = "2", parse(try_from_str = parse_devfund_percent))]
    pub devfund_percent: u16,

    #[clap(short, long, help = "Keryxd port [default: Mainnet = 22110, Testnet = 22210]")]
    port: Option<u16>,

    #[clap(
        long,
        help = "Use testnet instead of mainnet: default port 22210 and testnet DAA activation gates (PoM/H3 from genesis, H4/H5 at 3000) [default: false]"
    )]
    testnet: bool,

    #[clap(short = 't', long = "threads", help = "Amount of CPU miner threads to launch [default: 0]")]
    pub num_threads: Option<u16>,

    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when keryxd says it is not synced",
        long_help = "Mine even when keryxd says it is not synced, only useful when passing `--allow-submit-block-when-not-synced` to keryxd  [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[cfg(feature = "block-celebration")]
    #[clap(
        long = "block-celebration",
        help = "Enable the TUI block-accepted animation and sound (requires a build with the block-celebration feature)",
        help_heading = "Terminal UI"
    )]
    pub block_celebration: bool,

    #[clap(skip)]
    pub devfund_address: String,

    #[clap(
        long = "stats-bind",
        help = "Stats API bind address (e.g. 0.0.0.0, 127.0.0.1)",
        help_heading = "Monitoring",
        default_value = "127.0.0.1"
    )]
    pub stats_bind: String,

    #[clap(
        long = "stats-port",
        help = "Stats API TCP port",
        help_heading = "Monitoring",
        default_value_t = 3338u16
    )]
    pub stats_port: u16,

    #[clap(
        long = "plain-log-file",
        help = "Write plain text logs to this file path",
        help_heading = "Monitoring"
    )]
    pub plain_log_file: Option<String>,
}

fn parse_devfund_percent(s: &str) -> Result<u16, &'static str> {
    let err = "devfund-percent should be --devfund-percent=XX.YY up to 2 numbers after the dot";
    let mut splited = s.split('.');
    let prefix = splited.next().ok_or(err)?;
    // if there's no postfix then it's 0.
    let postfix = splited.next().ok_or(err).unwrap_or("0");
    // error if there's more than a single dot
    if splited.next().is_some() {
        return Err(err);
    };
    // error if there are more than 2 numbers before or after the dot
    if prefix.len() > 2 || postfix.len() > 2 {
        return Err(err);
    }
    let postfix: u16 = postfix.parse().map_err(|_| err)?;
    let prefix: u16 = prefix.parse().map_err(|_| err)?;
    // can't be more than 99.99%,
    if prefix >= 100 || postfix >= 100 {
        return Err(err);
    }
    if prefix < 2 {
        // Force at least 2 percent
        return Ok(200u16);
    }
    // DevFund is out of 10_000
    Ok(prefix * 100 + postfix)
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        // Switch every DAA activation gate (PoM + PoW salts) to its testnet value before any
        // mining state is built — see `pom::set_testnet`.
        keryx_miner::pom::set_testnet(self.testnet);
        if self.recover_escrow {
            return Ok(());
        }
        if self.mining_address.is_none() {
            return Err("--mining-address is required".into());
        }
        if self.keryxd_address.is_empty() {
            self.keryxd_address = "127.0.0.1".to_string();
        }

        if !self.keryxd_address.contains("://") {
            let port_str = self.port().to_string();
            let (keryxd, port) = match self.keryxd_address.contains(':') {
                true => self.keryxd_address.split_once(':').expect("We checked for `:`"),
                false => (self.keryxd_address.as_str(), port_str.as_str()),
            };
            self.keryxd_address = format!("grpc://{}:{}", keryxd, port);
        }
        log::info!("keryxd address: {}", self.keryxd_address);

        if self.num_threads.is_none() {
            self.num_threads = Some(0);
        }

        let miner_network = self.mining_address.as_deref().and_then(|a| a.split(':').next());
        self.devfund_address = String::from("keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte");
        let devfund_network = self.devfund_address.split(':').next();
        if miner_network.is_some() && devfund_network.is_some() && miner_network != devfund_network {
            self.devfund_percent = 0;
            log::info!(
                "Mining address ({}) and devfund ({}) are not from the same network. Disabling devfund.",
                miner_network.unwrap(),
                devfund_network.unwrap()
            )
        }
        Ok(())
    }

    fn port(&mut self) -> u16 {
        *self.port.get_or_insert(if self.testnet { 22210 } else { 22110 })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn model_tier_conflicts_reference_valid_arguments() {
        Opt::command().debug_assert();
        assert!(Opt::try_parse_from(["keryx-miner", "--very-light", "--very-high"]).is_err());
    }

    #[cfg(feature = "block-celebration")]
    #[test]
    fn block_celebration_requires_runtime_opt_in() {
        let disabled = Opt::try_parse_from(["keryx-miner"]).unwrap();
        let enabled = Opt::try_parse_from(["keryx-miner", "--block-celebration"]).unwrap();
        assert!(!disabled.block_celebration);
        assert!(enabled.block_celebration);
    }
}
