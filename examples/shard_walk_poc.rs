// Concept PoC: split a real model into shards, have two independent GPUs each mine (PoM walk)
// their own shard, and self-verify the produced proof — all in-process, no live node required.
// Uses the same functions the real miner uses (pom_gpu::mine, pom_gpu::dump_v3,
// pom_v3::build_proof_v3/verify_proof_v3) with a synthetic pre-pow-hash/target so it doesn't
// depend on the network ever reaching the PoM activation DAA for real.

use clap::Parser;
use keryx_miner::{gguf, models, pom, pom_gpu, pom_v3};
use std::fs::File;

#[derive(Parser, Debug, Clone)]
#[clap(name = "shard_walk_poc", about = "Concept PoC: split a real GGUF into shards, mine+prove each on its own GPU")]
struct Args {
    #[clap(long, default_value = r"E:\keryx-miner-v0.4.1-OPoI-win64-amd64\models\Kimi-Linear-48B\model.gguf")]
    gguf_path: String,
    #[clap(long, default_value_t = 8_000_000_000)]
    target_bytes: u64,
    /// Gate PoM activation DAAs as mainnet instead of testnet.
    #[clap(long)]
    mainnet: bool,
}

fn run_shard(device_id: u32, shard_idx: u16, args: &Args) {
    let spec = &models::KIMI_LINEAR_48B;
    let gguf_path = args.gguf_path.clone();
    let target_bytes: u64 = args.target_bytes;

    let mut file = File::open(&gguf_path).expect("open gguf");
    let meta = gguf::GgufMeta::read(&mut file).expect("parse gguf meta");
    let shards = keryx_miner::shard::pack_shards(&meta, target_bytes).expect("pack shards");
    let manifest = shards.into_iter().find(|s| s.index == shard_idx).expect("shard index in range");
    println!("[gpu{device_id}] shard {shard_idx}: layers [{},{})", manifest.layer_lo, manifest.layer_hi);

    pom_gpu::set_shard_for_device(device_id, spec.model_id, gguf_path, target_bytes, manifest);

    let daa: u64 = pom::pom_v3_activation_daa(); // testnet gate, set by main() below
    println!("[gpu{device_id}] installing shard (expect a raw scoped upload, no llama engine)...");
    let installed = pom_gpu::ensure_installed(device_id, daa);
    println!("[gpu{device_id}] ensure_installed -> {installed}");
    assert!(installed, "shard install failed for device {device_id}");

    // Synthetic pre-pow-hash/timestamp/target: this PoC only needs to prove the walk+proof
    // round-trip on real shard data, not a real node-issued template.
    let pph = [0x11u8; 32];
    let timestamp: u64 = 1_700_000_000_000;
    let h3 = true;
    let h5_1 = true;
    let h5_2 = true;
    let target_le = [0xffu8; 32]; // trivially easy: any pow_value <= this

    println!("[gpu{device_id}] mining (real GPU walk over the resident shard)...");
    let mut nonce_cursor: u64 = 1;
    let batch: u64 = 512;
    let found = loop {
        if let Some(n) = pom_gpu::mine(device_id, &pph, timestamp, &target_le, nonce_cursor, batch, h3, true, h5_1, h5_2, true, false, false) {
            break n;
        }
        nonce_cursor = nonce_cursor.wrapping_add(batch);
    };
    println!("[gpu{device_id}] found nonce={found}");

    let tier = pom_gpu::current_tier(device_id, daa).expect("shard tier index");
    let index = pom_gpu::active_index_for_device(device_id, spec.model_id).expect("resident shard index");

    let seed = pom::pom_block_seed(&pph, timestamp, found, h3, h5_1, h5_2);
    let (states, snippets, final_state) =
        pom_gpu::dump_v3(device_id, &pph, timestamp, found, h3, h5_1, h5_2).expect("dump_v3 (host re-walk from GPU state)");
    let pow_value = pom::pom_pow_value(final_state, &pph, h3);
    assert!(pom::le_leq(&pow_value, &target_le), "pow_value must meet the (trivial) target");

    let v3 = pom_v3::build_proof_v3(tier, &pph, found, seed, &states, &snippets, index.as_ref()).expect("build_proof_v3");
    assert_eq!(pom_v3::fold64(&v3.roots[pom_v3::POM_V3_K]), final_state, "GPU fold must match host tree fold");
    let ok = pom_v3::verify_proof_v3(&pph, found, seed, &v3, &index.r_t, index.n_chunks);
    assert!(ok, "verify_proof_v3 must accept the proof built from this shard's own walk");

    println!("[gpu{device_id}] PROOF BUILT AND SELF-VERIFIED (verify_proof_v3 == true) — tier={tier}");
}

fn main() {
    let args = Args::parse();
    pom::set_testnet(!args.mainnet);
    let a0 = args.clone();
    let a1 = args.clone();
    let t0 = std::thread::spawn(move || run_shard(0, 0, &a0));
    let t1 = std::thread::spawn(move || run_shard(1, 1, &a1));
    t0.join().expect("gpu0 thread panicked");
    t1.join().expect("gpu1 thread panicked");
    println!("\n=== CONCEPT PROVEN: both GPUs independently mined their own shard of Kimi-Linear-48B and produced a self-verified PoM proof, with no live node involved. ===");
}
