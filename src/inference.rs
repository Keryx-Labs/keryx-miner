/// Inference module — Phase 2.
///
/// Detects AiRequest TXs (subnetwork 0x03) in block templates, runs LLM
/// inference, and embeds the inference tag in the coinbase extra_data.
/// Tag computation delegates to `keryx-inference` (fixed-point MLP, bit-exact).

/// Compute the Phase-2 inference tag for a coinbase.
pub fn compute_opoi_tag(nonce_hex: &str) -> String {
    let nonce = u64::from_str_radix(nonce_hex, 16).unwrap_or(0);
    keryx_inference::tag_fixed(nonce)
}

// ---------------------------------------------------------------------------
// AiRequest TX scanning helpers (binary format).
// ---------------------------------------------------------------------------

/// Parse an AiRequest from a hex-encoded transaction payload (gRPC format).
///
/// Returns `(stable_id_hex16, prompt_string, max_tokens)` on success.
/// `stable_id_hex16` is the first 8 bytes of blake2b-256(payload_bytes) encoded
/// as 16 lowercase hex chars — a deterministic deduplication key that does not
/// require the transaction ID (unavailable in GetBlockTemplateResponse).
pub fn extract_ai_request(payload_hex: &str) -> Option<(String, String, usize)> {
    let req = keryx_inference::AiRequestPayload::from_hex(payload_hex)?;
    let prompt = String::from_utf8(req.prompt).ok()?;
    let bytes = hex::decode(payload_hex).ok()?;
    let hash = blake2b_simd::blake2b(&bytes);
    let stable_id = hex::encode(&hash.as_bytes()[..8]);
    Some((stable_id, prompt, req.max_tokens as usize))
}
