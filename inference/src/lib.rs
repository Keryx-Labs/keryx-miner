/// Keryx OPoI (Optimistic Proof of Inference) — Phase 2 + Phase 3
///
/// Phase 2: fixed-point i32/i64 MLP — bit-exact on all hardware.
///   Tags are verified on-chain; collateral is slashed for fraud.
///   (The Phase-1 f32 MLP was reference-only, never consensus post-Phase 2 — removed.)
/// Phase 3 A: AiResponse/AiChallenge as on-chain txs, RocksDB slash state.
/// Phase 3 B: ZK fraud proof API (Groth16 stub — circuit VK lands in Phase 3 C).

pub mod ai_payload;
pub mod fraud_proof;
pub mod model_fixed;
pub mod task;

pub use ai_payload::{
    AiRequestPayload, AiResponder, AiResponsePayload, AiChallengePayload,
    MIN_AI_REQUEST_PAYLOAD_LEN, MAX_AI_REQUEST_PAYLOAD_LEN,
    MIN_AI_RESPONSE_PAYLOAD_LEN, MAX_AI_RESPONSE_PAYLOAD_LEN, AI_RESPONSE_PAYLOAD_V2_LEN,
    MIN_AI_CHALLENGE_PAYLOAD_LEN, MAX_AI_CHALLENGE_PAYLOAD_LEN,
    SUBNETWORK_ID_AI_REQUEST_HEX, SUBNETWORK_ID_AI_RESPONSE_HEX, SUBNETWORK_ID_AI_CHALLENGE_HEX,
};
pub use fraud_proof::{verify_fraud_proof, compute_ai_commitment, FraudProofResult, FRAUD_PROOF_LEN};

pub use task::{InferenceResult, InferenceTask};

// ── Phase 2 — Verification helpers ───────────────────────────────────────────

/// Minimum offset in a coinbase payload before the extra_data begins.
/// Layout: blue_score(8) + subsidy(8) + spk_version(2) + spk_len(1) = 19 bytes.
/// The real offset is 19 + spk_len, but scanning from 19 avoids the dense binary
/// header where a false `/ai:v1:` match is theoretically possible.
const COINBASE_MIN_BINARY_HEADER: usize = 19;

/// Scans raw coinbase payload bytes for an OPoI tag.
///
/// Looks for the ASCII byte sequence `/ai:v1:` followed by 16 lowercase hex chars,
/// preceded by `/{nonce_hex16}`.  Returns `(nonce, claimed_tag_hex16)` on success.
///
/// Searches only from byte offset `COINBASE_MIN_BINARY_HEADER` to skip the
/// binary-encoded fields that precede `extra_data`.
pub fn parse_opoi(payload: &[u8]) -> Option<(u64, String)> {
    const MARKER: &[u8] = b"/ai:v1:";
    const NONCE_HEX_LEN: usize = 16;
    const TAG_HEX_LEN: usize = 16;

    // Skip the fixed binary header to avoid spurious matches.
    let search_start = COINBASE_MIN_BINARY_HEADER.min(payload.len());
    let search_slice = &payload[search_start..];

    // Find the marker byte sequence.
    let relative_pos = search_slice.windows(MARKER.len()).position(|w| w == MARKER)?;
    let marker_pos = search_start + relative_pos; // absolute position in payload

    // Extract ai_tag (16 hex chars after the marker).
    let tag_start = marker_pos + MARKER.len();
    if payload.len() < tag_start + TAG_HEX_LEN {
        return None;
    }
    let tag_bytes = &payload[tag_start..tag_start + TAG_HEX_LEN];
    let claimed_tag = std::str::from_utf8(tag_bytes).ok()?;
    if !claimed_tag.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    // Extract nonce: the 16 hex chars immediately before the marker, preceded by '/'.
    // Layout: .../{nonce_hex16}/ai:v1:{tag_hex16}
    if marker_pos < NONCE_HEX_LEN + 1 {
        return None; // not enough room for "/{nonce}"
    }
    let slash_pos = marker_pos - NONCE_HEX_LEN - 1;
    if payload[slash_pos] != b'/' {
        return None;
    }
    let nonce_bytes = &payload[slash_pos + 1..marker_pos];
    let nonce_hex = std::str::from_utf8(nonce_bytes).ok()?;
    if !nonce_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let nonce = u64::from_str_radix(nonce_hex, 16).ok()?;

    Some((nonce, claimed_tag.to_string()))
}

// ── Phase 2 — Fixed-point verification ───────────────────────────────────────

/// Protocol version salt for Phase 2 OPoI tags.
///
/// XORed into the nonce before the fixed-point MLP forward pass. Any miner
/// compiled against an older keryx-inference (without this salt) will produce
/// wrong OPoI tags and have their blocks rejected at consensus level —
/// cryptographically enforced, impossible to spoof without updating the crate.
pub const PHASE2_OPOI_SALT: u64 = u64::from_le_bytes(*b"KERYX:2\0");

/// Runs the fixed-point MLP on `nonce` and returns the 32-byte output.
/// Bit-exact on every platform — used for on-chain tag verification in Phase 2.
pub fn run_inference_fixed(nonce: u64) -> [u8; 32] {
    let task = InferenceTask::from_nonce(nonce);
    model_fixed::forward(&task.input)
}

/// Returns the 16-char hex OPoI tag produced by the fixed-point model for `nonce`.
/// The nonce is salted with `PHASE2_OPOI_SALT` before inference — miners compiled
/// against older versions of this crate will produce incompatible tags.
pub fn tag_fixed(nonce: u64) -> String {
    let output = run_inference_fixed(nonce ^ PHASE2_OPOI_SALT);
    hex::encode(&output[..8])
}

/// Verifies that `claimed_hex16` matches the fixed-point model output for `nonce`.
/// This is the authoritative check used by consensus during Phase 2.
pub fn verify_tag_fixed(nonce: u64, claimed_hex16: &str) -> bool {
    tag_fixed(nonce) == claimed_hex16
}

/// Generates OPoI extra_data bytes for use in test coinbase payloads.
/// Produces `/{nonce_hex16}/ai:v1:{tag_hex16}` which `parse_opoi` can parse.
pub fn gen_opoi_extra_data(nonce: u64) -> Vec<u8> {
    let tag = tag_fixed(nonce);
    format!("/{:016x}/ai:v1:{}", nonce, tag).into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_inference_is_deterministic() {
        let r1 = run_inference_fixed(0xDEAD_BEEF_CAFE_1337);
        let r2 = run_inference_fixed(0xDEAD_BEEF_CAFE_1337);
        assert_eq!(r1, r2, "same nonce must produce same output");
    }

    #[test]
    fn different_nonces_produce_different_outputs() {
        assert_ne!(run_inference_fixed(1), run_inference_fixed(2), "different nonces should differ");
    }

    #[test]
    fn tag_fixed_is_16_chars() {
        assert_eq!(tag_fixed(42).len(), 16);
    }

    // ── Phase 2 tests ─────────────────────────────────────────────────────────

    fn make_coinbase_payload(nonce: u64, tag: &str) -> Vec<u8> {
        // Binary header: blue_score(8) + subsidy(8) + spk_version(2) + spk_len(1) + spk(34)
        let mut payload = vec![0u8; 19 + 34]; // 53 bytes binary prefix
        // Append ASCII extra_data
        let extra = format!("0.2.1/2025-01-01/{:016x}/ai:v1:{}", nonce, tag);
        payload.extend_from_slice(extra.as_bytes());
        payload
    }

    #[test]
    fn parse_opoi_finds_valid_tag() {
        let nonce = 0xABCD_1234_5678_EF01u64;
        let tag = tag_fixed(nonce);
        let payload = make_coinbase_payload(nonce, &tag);

        let parsed = parse_opoi(&payload);
        assert!(parsed.is_some(), "should find OPoI tag");
        let (parsed_nonce, parsed_tag) = parsed.unwrap();
        assert_eq!(parsed_nonce, nonce);
        assert_eq!(parsed_tag, tag);
    }

    #[test]
    fn parse_opoi_returns_none_when_missing() {
        let payload: Vec<u8> = b"\x00\x00\x00\x00\x00\x00\x00\x00plain/extra/data/without/ai/tag".to_vec();
        assert!(parse_opoi(&payload).is_none());
    }

    #[test]
    fn verify_tag_fixed_accepts_correct_tag() {
        let nonce = 99u64;
        assert!(verify_tag_fixed(nonce, &tag_fixed(nonce)));
    }

    #[test]
    fn verify_tag_fixed_rejects_wrong_tag() {
        assert!(!verify_tag_fixed(42, "0000000000000000"));
    }
}
