//! OPoI task and result types.

/// Seed for the deterministic MLP model weights.
/// ASCII for "KeryxOP" — identifies this as the Optimistic Proof of Inference v1 model.
pub const MODEL_SEED: u64 = 0x4B_65_72_79_78_4F_50_00;

/// An inference task derived from a miner's per-block nonce.
/// The nonce is already random and unique per block template request.
#[derive(Debug, Clone)]
pub struct InferenceTask {
    /// 32-byte input tensor, derived from the block nonce.
    pub input: [u8; 32],
}

/// The result of running inference on an `InferenceTask`.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// 32-byte output of the MLP forward pass.
    pub output: [u8; 32],
}

impl InferenceTask {
    /// Constructs a task from a 64-bit miner nonce.
    /// The nonce is placed in the first 8 bytes; remaining bytes are filled
    /// with a fixed pattern derived from `MODEL_SEED` for full determinism.
    pub fn from_nonce(nonce: u64) -> Self {
        let mut input = [0u8; 32];
        input[..8].copy_from_slice(&nonce.to_le_bytes());
        // Fill [8..32] with repeating MODEL_SEED bytes so the full input is deterministic
        // given only the nonce — no external state needed.
        let seed_bytes = MODEL_SEED.to_le_bytes();
        for chunk in input[8..].chunks_mut(8) {
            let n = chunk.len();
            chunk.copy_from_slice(&seed_bytes[..n]);
        }
        Self { input }
    }
}

impl InferenceResult {
    /// Returns a 16-character lowercase hex string of the first 8 bytes.
    /// Used as the `ai:v1:<tag>` suffix in the coinbase `extra_data` field.
    pub fn as_hex8(&self) -> String {
        hex::encode(&self.output[..8])
    }
}
