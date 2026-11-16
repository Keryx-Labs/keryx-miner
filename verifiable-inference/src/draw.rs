//! Draws from a response seed.

use crate::merkle::Hash;

pub const AUDIT_THRESHOLD: u16 = 16_384;
pub const POINTS: u32 = 2;

pub fn audited(seed: &Hash) -> bool {
    u16::from_le_bytes([seed[0], seed[1]]) < AUDIT_THRESHOLD
}

/// Draws of point `i`.
pub fn point(seed: &Hash, i: u32) -> [u64; 4] {
    let mut buf = [0u8; 36];
    buf[..32].copy_from_slice(seed);
    buf[32..].copy_from_slice(&i.to_le_bytes());
    let h = blake3::hash(&buf);
    let b = h.as_bytes();
    std::array::from_fn(|j| u64::from_le_bytes(b[8 * j..8 * j + 8].try_into().unwrap()))
}
