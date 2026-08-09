use anyhow::{bail, Context, Result};
use std::path::Path;
use std::time::Duration;

static SMOKE_SPECS: [&keryx_miner::models::ModelSpec; 1] = [&keryx_miner::models::QWEN3_8B_ABLITERATED];

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1);
    let Some(gguf) = args.next() else {
        bail!("usage: cargo run --example llama_smoke -- <model.gguf> [gpu] [cycles] [pom-swap]");
    };
    let gpu = args.next().as_deref().unwrap_or("0").parse::<usize>().context("invalid GPU ordinal")?;
    let cycles = args.next().as_deref().unwrap_or("10").parse::<usize>().context("invalid cycle count")?;
    let pom_swap = args.next().as_deref() == Some("pom-swap");
    if !Path::new(&gguf).is_file() {
        bail!("GGUF does not exist: {gguf}");
    }
    keryx_miner::llama_engine::probe_library().map_err(anyhow::Error::msg)?;

    if pom_swap {
        return run_pom_swap(&gguf, gpu, cycles);
    }

    run_plain(&gguf, gpu, cycles)
}

fn run_plain(gguf: &str, gpu: usize, cycles: usize) -> Result<()> {
    for cycle in 1..=cycles {
        keryx_miner::llama_engine::ensure_loaded(gguf, gpu).with_context(|| format!("load cycle {cycle}/{cycles}"))?;
        let Some(tensors) = keryx_miner::llama_engine::tensors() else {
            keryx_miner::llama_engine::unload("smoke_tensor_failure");
            bail!("tensor discovery failed on cycle {cycle}/{cycles}");
        };
        let bytes = tensors.iter().map(|(_, _, bytes, _)| *bytes as u64).sum::<u64>();
        let device_tensors = tensors.iter().filter(|(_, _, _, device)| *device).count();
        println!(
            "cycle {cycle}/{cycles}: {} tensors, {} device tensors, {} MiB",
            tensors.len(),
            device_tensors,
            bytes / (1024 * 1024)
        );
        keryx_miner::llama_engine::unload("smoke_cycle_complete");
        std::thread::sleep(Duration::from_millis(250));
    }

    Ok(())
}

fn run_pom_swap(gguf: &str, gpu: usize, cycles: usize) -> Result<()> {
    let device_id = u32::try_from(gpu).context("GPU ordinal exceeds u32")?;
    let spec = &keryx_miner::models::QWEN3_8B_ABLITERATED;
    let daa = keryx_miner::pom::h5_activation_daa();
    keryx_miner::slm::init_supported(&SMOKE_SPECS);
    keryx_miner::pom_gpu::set_mining_tier(device_id, spec.model_id, gguf.to_string());

    for cycle in 1..=cycles {
        if !keryx_miner::pom_gpu::ensure_installed(device_id, daa) {
            bail!("PoM miner installation failed on cycle {cycle}/{cycles}");
        }
        println!("cycle {cycle}/{cycles}: PoM miner installed; releasing miner then llama");
        keryx_miner::pom_gpu::release_llama_for_gpu(device_id, "smoke_pom_swap");
        std::thread::sleep(Duration::from_millis(250));
    }

    Ok(())
}
