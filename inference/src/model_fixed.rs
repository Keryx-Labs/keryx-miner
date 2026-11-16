//! Fixed-point MLP for OPoI Phase 2 — deterministic on all hardware.
//!
//! Architecture: 32 → 256 → 128 → 32 (ReLU / ReLU / identity)
//!
//! All arithmetic is i32/i64 — no floating-point operations anywhere.
//! This guarantees bit-exact results on every CPU regardless of SIMD,
//! FMA extensions, or compiler rounding mode.  Used as the canonical
//! reference model for fraud-proof verification: a challenger computes
//! the expected tag here and the consensus verifier checks it on-chain.
//!
//! Weight layout: w[output_neuron * n_inputs + input_neuron]
//!
//! Scale convention: each hidden-layer accumulator is right-shifted by
//! NORM_SHIFT (= 10 bits) to keep activations in a bounded integer range
//! before being fed into the next layer.

use crate::task::MODEL_SEED;

/// Normalization right-shift applied after each hidden-layer accumulation.
/// Divides by 2^10 = 1024 to prevent unbounded magnitude growth.
const NORM_SHIFT: u32 = 10;

/// Layer dimensions.
const N_IN:  usize = 32;
const N_H1:  usize = 256;
const N_H2:  usize = 128;
const N_OUT: usize = 32;

/// He initialization scale for each layer (dimensionless integer units).
/// he_scale = floor(sqrt(2 / n_inputs) × 2^NORM_SHIFT) = floor(sqrt(2/N) × 1024)
///
/// Layer 1 (N =  32): sqrt(2/32)  × 1024 = 0.25    × 1024 = 256  (exact)
/// Layer 2 (N = 256): sqrt(2/256) × 1024 ≈ 0.08839 × 1024 = 90   (floor of 90.51)
/// Layer 3 (N = 128): sqrt(2/128) × 1024 = 0.125   × 1024 = 128  (exact)
const HE_L1: i64 = 256;
const HE_L2: i64 = 90;
const HE_L3: i64 = 128;

/// Advance the LCG and return a weight in the range (-he_scale, +he_scale).
///
/// Uses the same LCG constants as model.rs but extracts the upper 32 bits
/// (rather than bits 41..64) — intentionally producing a different weight
/// distribution so the fixed-point model is clearly distinct from the f32 one.
#[inline]
fn lcg_weight(state: &mut u64, he_scale: i64) -> i32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);

    // Upper 32 bits reinterpreted as i32 → uniform in [-2^31, 2^31).
    let raw = ((*state >> 32) as u32) as i32 as i64;

    // Scale to (-he_scale, +he_scale): raw/2^31 × he_scale
    // Integer approximation: (raw × he_scale) >> 31
    ((raw * he_scale) >> 31) as i32
}

/// Build the weight matrix for a layer (all weights generated from the shared seed).
fn make_weights(rows: usize, cols: usize, layer_id: u64, he: i64) -> Vec<i32> {
    let mut s = MODEL_SEED.wrapping_add(layer_id.wrapping_mul(0xDEAD_BEEF_CAFE_1337));
    (0..rows * cols).map(|_| lcg_weight(&mut s, he)).collect()
}

/// Run the fixed-point forward pass.
///
/// Returns 32 output bytes.  The first 8 are used as the Phase 2 OPoI tag
/// (encoded as 16 lowercase hex chars appended to the coinbase `extra_data`).
pub fn forward(input: &[u8; 32]) -> [u8; 32] {
    // Centre input: raw [0, 255] → signed [-128, 127]
    let x: Vec<i64> = input.iter().map(|&b| b as i64 - 128).collect();

    // ── Layer 1: N_IN → N_H1, ReLU ──────────────────────────────────────────
    //
    // Accumulator worst-case: N_IN × 128 × HE_L1 = 32 × 128 × 256 = 1,048,576
    // After >> NORM_SHIFT: max activation ≈ 1,024  (range: [0, 1024])
    let w1 = make_weights(N_H1, N_IN, 0, HE_L1);
    let h1: Vec<i64> = (0..N_H1)
        .map(|i| {
            let acc: i64 = (0..N_IN)
                .map(|j| x[j] * w1[i * N_IN + j] as i64)
                .sum();
            (acc >> NORM_SHIFT).max(0)
        })
        .collect();

    // ── Layer 2: N_H1 → N_H2, ReLU ──────────────────────────────────────────
    //
    // Accumulator worst-case: N_H1 × 1_024 × HE_L2 = 256 × 1_024 × 90 = 23,592,960
    // After >> NORM_SHIFT: max activation ≈ 23,040  (range: [0, 23040])
    let w2 = make_weights(N_H2, N_H1, 1, HE_L2);
    let h2: Vec<i64> = (0..N_H2)
        .map(|i| {
            let acc: i64 = (0..N_H1)
                .map(|j| h1[j] * w2[i * N_H1 + j] as i64)
                .sum();
            (acc >> NORM_SHIFT).max(0)
        })
        .collect();

    // ── Layer 3: N_H2 → N_OUT, identity ─────────────────────────────────────
    //
    // Accumulator worst-case: N_H2 × 23_040 × HE_L3 = 128 × 23_040 × 128 = 377,487,360
    // Fits comfortably in i64.  No normalisation needed — we fold to bytes directly.
    let w3 = make_weights(N_OUT, N_H2, 2, HE_L3);
    let h3: Vec<i64> = (0..N_OUT)
        .map(|i| {
            (0..N_H2)
                .map(|j| h2[j] * w3[i * N_H2 + j] as i64)
                .sum()
        })
        .collect();

    // ── Output → bytes ───────────────────────────────────────────────────────
    //
    // XOR-fold all 8 bytes of each i64 output for maximum bit diffusion.
    // Result is deterministic and in [0, 255] per output neuron.
    let mut out = [0u8; 32];
    for (i, &v) in h3.iter().enumerate() {
        let b = v.to_le_bytes();
        out[i] = b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ b[6] ^ b[7];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::InferenceTask;

    #[test]
    fn fixed_point_is_deterministic() {
        let task = InferenceTask::from_nonce(0xDEAD_BEEF_CAFE_1337);
        let r1 = forward(&task.input);
        let r2 = forward(&task.input);
        assert_eq!(r1, r2, "same input must produce identical output");
    }

    #[test]
    fn different_nonces_differ() {
        let t1 = InferenceTask::from_nonce(1);
        let t2 = InferenceTask::from_nonce(2);
        assert_ne!(forward(&t1.input), forward(&t2.input));
    }

    #[test]
    fn output_is_32_bytes() {
        let task = InferenceTask::from_nonce(42);
        let out = forward(&task.input);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn output_not_all_zeros() {
        let task = InferenceTask::from_nonce(0);
        let out = forward(&task.input);
        assert!(out.iter().any(|&b| b != 0), "output must not be all zeros");
    }

    /// Regression test: pins the exact output for nonce=42.
    /// Any accidental change to weights or arithmetic will fail this test immediately.
    #[test]
    fn fixed_point_regression_nonce_42() {
        let task = InferenceTask::from_nonce(42);
        let out = forward(&task.input);
        let expected: [u8; 32] = [
            182, 147, 169, 135, 251, 232, 129,  16,
            221, 172,  47, 152,   9,  81, 226, 160,
              1,  54, 235,  28, 221, 139, 125, 111,
            176, 173, 146,  73, 168, 229, 102, 209,
        ];
        assert_eq!(out, expected);
        assert_eq!(hex::encode(&out[..8]), "b693a987fbe88110");
    }

    #[test]
    #[ignore]
    fn gen_fixed_point_vector() {
        let task = InferenceTask::from_nonce(42);
        let out = forward(&task.input);
        println!("nonce=42 fixed-point output: {:?}", out);
        println!("hex tag: {}", hex::encode(&out[..8]));
    }
}
