use keryx_miner::models::GEMMA_4_12B_ABLITERATED;
use keryx_miner::{pom_gpu, slm};

#[test]
#[ignore = "requires one CUDA GPU, real Gemma-4 GGUF, keryx-llama shared library, and KERYX_MODELS_DIR"]
fn gemma4_live_raw_pom_fallback_then_opoi() {
    let models_root = std::env::var("KERYX_MODELS_DIR")
        .expect("set KERYX_MODELS_DIR to the directory containing Gemma-4-12B-abliterated");
    let gguf = std::path::Path::new(&models_root)
        .join(GEMMA_4_12B_ABLITERATED.dir_name)
        .join("model.gguf");
    assert!(gguf.exists(), "real Gemma-4 GGUF is missing at {}", gguf.display());

    let gguf = gguf.to_string_lossy().into_owned();
    let specs: &'static [&'static keryx_miner::models::ModelSpec] =
        Box::leak(vec![&GEMMA_4_12B_ABLITERATED].into_boxed_slice());
    slm::init_supported(specs);
    pom_gpu::set_mining_tier(0, GEMMA_4_12B_ABLITERATED.model_id, gguf.clone());

    // This exercises the real H6 installation path. On Gemma, llama.cpp currently exposes
    // N=331,123,456 chunks while the canonical GGUF index is N=305,318,656. The miner must
    // therefore unload the zero-copy llama view and install a raw canonical PoM copy instead.
    assert!(
        pom_gpu::ensure_installed(0, u64::MAX),
        "Gemma-4 PoM fallback failed to install on GPU 0"
    );

    // The layout mismatch is a zero-copy PoM limitation, not an inference failure. The model
    // must remain advertised so OPoI does not suspend mining with `no models ready`.
    assert!(
        slm::loaded_model_ids().contains(&GEMMA_4_12B_ABLITERATED.model_id),
        "Gemma-4 was withdrawn from ai:cap after the PoM layout fallback"
    );

    // Inference has priority. This must evict/drain the raw PoM miner, reload Gemma in the
    // llama engine and produce a real non-empty response on the same GPU.
    let text = slm::load_and_run_inference(
        &GEMMA_4_12B_ABLITERATED.model_id,
        "Reply with only OK.",
        16,
    )
    .expect("Gemma-4 OPoI inference failed after the raw PoM fallback");
    assert!(!text.trim().is_empty(), "Gemma-4 returned an empty OPoI response");

    eprintln!("Gemma-4 live fallback PASS: PoM raw canonical path installed, ai:cap preserved, OPoI response={text:?}");
}
