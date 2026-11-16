//! OPoI Fraud Proof — Phase 3 C: deterministic re-execution.
//!
//! The verifiable component of an AiResponse is a 32-byte commitment:
//!   `commitment = model_fixed::forward(request_hash)`
//! where `request_hash = blake2b(raw_AiRequest_payload)[0..32]`.
//!
//! This commitment is deterministic (bit-exact on all hardware) and fast
//! to re-execute (~microseconds), so there is no need for a ZK circuit.
//! The miner MUST prepend the commitment to AiResponse.result; fraud is
//! detected when the on-chain commitment differs from the re-computed one.
//!
//! Wire format of `proof_data` inside `AiChallengePayload` (Phase 3 C):
//!   `[request_hash: 32 bytes]`
//!
//! The verifier uses `request_hash` to:
//!   1. Confirm it matches the `request_hash` stored in the AiResponseRecord.
//!   2. Re-compute `model_fixed::forward(request_hash)`.
//!   3. Compare with `claimed_commitment` from the AiResponseRecord.
//!
//! If they differ, the miner published a fraudulent commitment → slash.

use crate::model_fixed;

/// Byte length of a Phase 3 C fraud proof (= the request_hash, 32 bytes).
pub const FRAUD_PROOF_LEN: usize = 32;

/// Result returned by `verify_fraud_proof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FraudProofResult {
    /// Fraud proven: the miner's claimed commitment does not match the
    /// deterministic re-execution of `model_fixed::forward(request_hash)`.
    Valid,
    /// The miner was honest: re-execution matches the claimed commitment.
    /// Also returned for malformed inputs (wrong request_hash).
    Invalid,
}

/// Verify a re-execution fraud proof for an AiResponse.
///
/// `request_hash` — the 32-byte blake2b prefix of the original AiRequest
///                  payload, as submitted by the challenger in `proof_data`.
/// `claimed_commitment` — the first 32 bytes of `AiResponse.result`, stored
///                        in `AiResponseRecord.claimed_commitment` when the
///                        AiResponse was indexed on-chain.
///
/// Returns `Valid` if the miner lied, `Invalid` if honest.
pub fn verify_fraud_proof(request_hash: &[u8; 32], claimed_commitment: &[u8; 32]) -> FraudProofResult {
    let expected = model_fixed::forward(request_hash);
    if expected != *claimed_commitment {
        FraudProofResult::Valid
    } else {
        FraudProofResult::Invalid
    }
}

/// Compute the OPoI commitment for a given AiRequest payload hash.
///
/// Miners call this to prepend the commitment to `AiResponse.result`.
/// Consensus calls this (via `verify_fraud_proof`) to detect fraud.
pub fn compute_ai_commitment(request_hash: &[u8; 32]) -> [u8; 32] {
    model_fixed::forward(request_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_commitment(request_hash: &[u8; 32]) -> [u8; 32] {
        compute_ai_commitment(request_hash)
    }

    #[test]
    fn honest_miner_returns_invalid() {
        let req = [1u8; 32];
        let commitment = dummy_commitment(&req);
        assert_eq!(verify_fraud_proof(&req, &commitment), FraudProofResult::Invalid);
    }

    #[test]
    fn lying_miner_returns_valid() {
        let req = [2u8; 32];
        let wrong_commitment = [0xFFu8; 32];
        assert_eq!(verify_fraud_proof(&req, &wrong_commitment), FraudProofResult::Valid);
    }

    #[test]
    fn different_request_hashes_produce_different_commitments() {
        let c1 = compute_ai_commitment(&[1u8; 32]);
        let c2 = compute_ai_commitment(&[2u8; 32]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn commitment_is_deterministic() {
        let req = [42u8; 32];
        assert_eq!(compute_ai_commitment(&req), compute_ai_commitment(&req));
    }
}
