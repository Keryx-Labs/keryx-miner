// Plan Step 7 (guard regression check): temporarily corrupt one shard's `tensor_names` in the
// running miner's manifest computation (drop one tensor) and confirm the N-guard in
// `ensure_shard_installed_inner` (src/pom_gpu.rs) refuses to mine — logs "gather N != shard index
// N — refusing to mine" — rather than silently producing a proof over the wrong data.
//
// Sequence: install the shard with its real, correct manifest (builds the host index, caches its
// N). Uninstall just the GPU-resident miner (leaves the host index cache untouched — mirrors a
// live device reloading its gather without the index ever being rebuilt). Re-register the same
// device with a manifest missing its last tensor, then try to reinstall: the gather now walks a
// different (shorter) tensor set than the cached index was built from, so N must mismatch and the
// guard must refuse. Never edits a source file — the "manifest" here is in-process state private
// to this disposable process, so there's nothing to revert once it exits.

use clap::Parser;
use keryx_miner::{gguf, models, pom, pom_gpu};
use std::fs::File;

#[derive(Parser, Debug)]
#[clap(name = "shard_guard_poc", about = "Plan Step 7: proves the N-guard refuses to mine when a shard's manifest is corrupted")]
struct Args {
    #[clap(long, default_value = r"E:\keryx-miner-v0.4.1-OPoI-win64-amd64\models\Kimi-Linear-48B\model.gguf")]
    gguf_path: String,
    #[clap(long, default_value_t = 8_000_000_000)]
    target_bytes: u64,
    /// CUDA device to use — pick an idle GPU, this loads real weights onto it.
    #[clap(long, default_value_t = 0)]
    device_id: u32,
    /// Gate PoM activation DAAs as mainnet instead of testnet.
    #[clap(long)]
    mainnet: bool,
}

fn main() {
    let args = Args::parse();
    pom::set_testnet(!args.mainnet);

    let device_id: u32 = args.device_id;
    let spec = &models::KIMI_LINEAR_48B;
    let gguf_path = args.gguf_path.clone();
    let target_bytes: u64 = args.target_bytes;

    let mut file = File::open(&gguf_path).expect("open gguf");
    let meta = gguf::GgufMeta::read(&mut file).expect("parse gguf meta");
    let shards = keryx_miner::shard::pack_shards(&meta, target_bytes).expect("pack shards");
    let good_manifest = shards.into_iter().find(|s| s.index == 0).expect("shard 0 in range");
    println!(
        "[guard-poc] shard 0: layers [{},{}) — {} tensors (correct manifest)",
        good_manifest.layer_lo,
        good_manifest.layer_hi,
        good_manifest.tensor_names.len()
    );

    let daa: u64 = pom::pom_v3_activation_daa();

    // Step A: install with the correct manifest — must succeed and cache the host index.
    pom_gpu::set_shard_for_device(device_id, spec.model_id, gguf_path.clone(), target_bytes, good_manifest.clone());
    let installed = pom_gpu::ensure_installed(device_id, daa);
    println!("[guard-poc] correct-manifest install -> {installed}");
    assert!(installed, "baseline install with the correct manifest must succeed");

    // Step B: release the GPU-resident miner only. The host index cache (built from the correct
    // manifest) is untouched — this is the load-bearing precondition for the guard to have
    // anything to check against.
    pom_gpu::uninstall(device_id);

    // Step C: corrupt the manifest — drop the last tensor, same shard/device/gguf otherwise.
    let mut bad_manifest = good_manifest.clone();
    let dropped = bad_manifest.tensor_names.pop().expect("shard has at least one tensor");
    println!("[guard-poc] corrupting manifest: dropped tensor \"{dropped}\" ({} -> {} tensors)", good_manifest.tensor_names.len(), bad_manifest.tensor_names.len());
    pom_gpu::set_shard_for_device(device_id, spec.model_id, gguf_path, target_bytes, bad_manifest);

    // Step D: reinstall attempt. The gather now walks a shorter tensor set than the cached index
    // (still built from the correct manifest) — the N-guard must refuse rather than silently
    // installing a miner whose proofs wouldn't match the index it claims.
    let reinstalled = pom_gpu::ensure_installed(device_id, daa);
    println!("[guard-poc] corrupted-manifest reinstall -> {reinstalled}");
    assert!(!reinstalled, "N-guard must refuse to mine when the gather no longer matches the cached shard index");

    println!("\n=== GUARD PROVEN: dropping a tensor from the manifest made the gather N diverge from the cached shard index N, and ensure_installed correctly refused rather than installing a miner over mismatched data. ===");
}
