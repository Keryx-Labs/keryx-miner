use super::statum_codec::MiningNotify;
use crate::{Error, Uint256};
use num::Float;

pub(super) struct MiningJob {
    pub id: String,
    pub header_hash: [u64; 4],
    pub timestamp: u64,
    pub daa_score: u64,
    pub block_bits: Option<u32>,
    pub task_json: Option<String>,
}

impl TryFrom<MiningNotify> for MiningJob {
    type Error = Error;

    fn try_from(notify: MiningNotify) -> Result<Self, Error> {
        let (id, header_hash, timestamp, daa_score, block_bits, task_json) = match notify {
            MiningNotify::MiningNotifyWithTaskV3((id, hash, time, daa, bits, task)) => {
                (id, hash, time, daa, Some(bits), Some(task))
            }
            MiningNotify::MiningNotifyShortV3((id, hash, time, daa, bits)) => (id, hash, time, daa, Some(bits), None),
            MiningNotify::MiningNotifyWithTask((id, hash, time, daa, task)) => (id, hash, time, daa, None, Some(task)),
            MiningNotify::MiningNotifyShortV2((id, hash, time, daa)) => (id, hash, time, daa, None, None),
            _ => return Err("Keryx mining.notify must include the actual DAA score".into()),
        };
        Ok(Self { id, header_hash, timestamp, daa_score, block_bits, task_json })
    }
}

pub(super) fn difficulty_to_target(difficulty: f32) -> Result<Uint256, Error> {
    if !difficulty.is_finite() || difficulty <= 0.0 || !difficulty.recip().is_finite() {
        return Err("Invalid pool difficulty".into());
    }
    let (mantissa, exponent, _) = difficulty.recip().integer_decode();
    let mantissa = mantissa * 0xffff;
    let exponent = 208i32 + i32::from(exponent);
    if exponent < 0 {
        return Err("Target is too small".into());
    }
    let start = exponent as usize / 64;
    let remainder = exponent as usize % 64;
    let mut words = [0u64; 4];
    if start >= words.len() || (start == 3 && mantissa.leading_zeros() < remainder as u32) {
        return Err("Target is too big".into());
    }
    words[start] = mantissa << remainder;
    if remainder != 0 && start < 3 {
        words[start + 1] = mantissa >> (64 - remainder);
    }
    Ok(Uint256::new(words))
}

pub(super) fn effective_target(pool_target: Uint256, block_bits: Option<u32>) -> Result<Uint256, Error> {
    let Some(bits) = block_bits else {
        return Ok(pool_target);
    };
    let size = bits >> 24;
    let mantissa = bits & 0x007fffff;
    if bits & 0x00800000 != 0 || size > 34 || (size > 33 && mantissa > 0xff) || (size > 32 && mantissa > 0xffff) {
        return Err("Invalid block target".into());
    }
    let target = crate::target::u256_from_compact_target(bits);
    if target == Uint256::default() {
        return Err("Invalid block target".into());
    }
    Ok(pool_target.max(target))
}

pub(super) fn nonce_range(extranonce: &str, nonce_size: u32) -> Result<(u64, u64), Error> {
    if nonce_size > 8
        || extranonce.len() > (8 - nonce_size) as usize * 2
        || !extranonce.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err("Invalid Stratum extranonce range".into());
    }
    let prefix = if extranonce.is_empty() { 0 } else { u64::from_str_radix(extranonce, 16)? };
    if nonce_size == 8 {
        Ok((u64::MAX, 0))
    } else {
        let bits = nonce_size * 8;
        Ok(((1u64 << bits) - 1, prefix << bits))
    }
}
