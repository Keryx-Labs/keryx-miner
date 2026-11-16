//! Request-bound prefix: the layer-0 check that ties a trace to its request.

use crate::draw;
use crate::manifest;
use crate::merkle::{Hash, b3_fold, b3_verify};
use crate::opening::PrefixPoint;
use crate::verify::{Fail, PosShape, ResponseCtx, TierCtx, expected_ubatch, layout_for, open_bytes, open_entry, open_row, x_len};
use crate::{q8, quant};

/// Prefix tokens, one per byte of the request id.
pub const PREFIX_LEN: u32 = 32;
/// Draw index of the prefix point.
pub const PREFIX_POINT: u32 = draw::POINTS;
/// Largest accepted `|q - v / d|` over the committed Q8_1 input.
pub const THRESHOLD: f64 = 1.0;

/// Consensus data of a tier's prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixCtx {
    /// Token id of each byte value.
    pub table: [u32; 256],
    /// Prompt position of the first prefix token.
    pub offset: u32,
    /// Manifest index of the embedding table entry.
    pub embd_entry: u32,
    /// Manifest index of the layer-0 input norm entry.
    pub norm_entry: u32,
    /// RMS norm epsilon.
    pub eps: f64,
    /// Factor applied to the embedding before the norm.
    pub embd_scale: f64,
}

/// Whether a prompt of `n_prompt` positions holds the whole prefix.
pub fn fits(n_prompt: u32, offset: u32) -> bool {
    n_prompt >= offset.saturating_add(PREFIX_LEN)
}

/// The prefix position a seed draws.
pub fn position(seed: &Hash, offset: u32) -> u64 {
    offset as u64 + draw::point(seed, PREFIX_POINT)[0] % PREFIX_LEN as u64
}

/// The token the request id puts at prefix position `t`.
pub fn expected_token(ctx: &PrefixCtx, request_txid: &Hash, t: u64) -> Option<u32> {
    let i = t.checked_sub(ctx.offset as u64)?;
    (i < PREFIX_LEN as u64).then(|| ctx.table[request_txid[i as usize] as usize])
}

/// The layer-0 input of token `tok`: the scaled, RMS-normalized embedding row times the norm.
pub fn expected_input(embd_row: &[f32], norm: &[f32], eps: f64, embd_scale: f64) -> Vec<f64> {
    let e: Vec<f64> = embd_row.iter().map(|&v| v as f64 * embd_scale).collect();
    let ms = e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64;
    let inv = 1.0 / libm::sqrt(ms + eps);
    e.iter().zip(norm).map(|(v, &w)| v * inv * w as f64).collect()
}

/// `max |q - v / d|` of the committed row against `v`, per 32-value block; `None` when a zero
/// scale faces a non-zero expectation.
pub fn deviation(x: &[u8], layout: u8, v: &[f64]) -> Option<f64> {
    let blocks = q8::blocks(x, layout, v.len())?;
    let vmax = v.iter().fold(0f64, |m, a| m.max(a.abs()));
    let mut worst = 0f64;
    for (b, (d, q)) in blocks.iter().enumerate() {
        let vals = &v[b * 32..((b + 1) * 32).min(v.len())];
        if !d.is_finite() || !q8::canonical(*d, q) || vals.iter().any(|a| !a.is_finite()) {
            return None;
        }
        if *d == 0.0 {
            if vals.iter().any(|a| a.abs() > 1e-6 * vmax) {
                return None;
            }
            continue;
        }
        for (j, a) in vals.iter().enumerate() {
            worst = worst.max((q[j] as f64 - a / *d as f64).abs());
        }
    }
    Some(worst)
}

/// Verify the prefix point of an opening.
pub fn verify_prefix_point(
    tier: &TierCtx,
    ctx: &PrefixCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    request_txid: &Hash,
    p: &PrefixPoint,
) -> Result<u64, Fail> {
    verify_prefix_point_at(tier, ctx, resp, seed, request_txid, p, |_| PosShape::stage1(tier.n_products), false)
}

pub(crate) fn verify_prefix_point_at(
    tier: &TierCtx,
    ctx: &PrefixCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    request_txid: &Hash,
    p: &PrefixPoint,
    shape_for: impl Fn(u32) -> PosShape,
    fragment: bool,
) -> Result<u64, Fail> {
    if resp.n_pos == 0 || resp.n_prompt == 0 || resp.n_prompt > resp.n_pos || !fits(resp.n_prompt, ctx.offset) {
        return Err(Fail::Shape);
    }
    if tier.n_products == 0 || tier.n_products > tier.n_entries {
        return Err(Fail::Shape);
    }
    let t = position(seed, ctx.offset);
    let tok = expected_token(ctx, request_txid, t).ok_or(Fail::Shape)?;

    let first = open_entry(tier, &p.entry, 0)?;
    if first.role != manifest::ROLE_PRODUCT || first.n_exp != 1 {
        return Err(Fail::Manifest);
    }
    let embd = open_entry(tier, &p.embd.0, ctx.embd_entry as u64)?;
    let norm = open_entry(tier, &p.norm.0, ctx.norm_entry as u64)?;
    if embd.role != manifest::ROLE_EMBD || norm.role != manifest::ROLE_NORM || norm.wtype != 0 {
        return Err(Fail::Manifest);
    }
    if embd.k != first.k || norm.k != first.k || embd.n_exp != 1 || norm.n_exp != 1 {
        return Err(Fail::Manifest);
    }

    let shape = shape_for(tok);
    let n_pos_leaves = shape.n_leaves;
    let layout = layout_for(p.n_ubatch, first.wtype);
    if p.x.len() as u64 != x_len(first.k, layout) {
        return Err(Fail::Shape);
    }
    let pos_root = b3_fold(n_pos_leaves, 0, blake3::hash(&p.x).as_bytes(), &p.x_path).ok_or(Fail::Trace)?;
    if !b3_verify(&resp.trace_root, resp.n_pos as u64, t, &pos_root, &p.pos_path) {
        return Err(Fail::Trace);
    }
    if p.n_ubatch != expected_ubatch(t, resp.n_prompt) {
        return Err(Fail::Mismatch);
    }
    if !b3_verify(&pos_root, n_pos_leaves, shape.meta_index, &shape.meta_hash(t, p.n_ubatch), &p.meta_path) {
        return Err(Fail::Trace);
    }

    let k = first.k as usize;
    let row = open_row(tier, &embd, &p.embd.1, tok as u64)?;
    let e = quant::dequant_row(embd.wtype, &row, k).ok_or(Fail::Unsupported)?;
    let (b0, nb) = if fragment {
        crate::chain::norm_fragment(crate::chain::link_block(seed, PREFIX_POINT, &p.x, layout, k as u64).ok_or(Fail::Shape)?, k as u64)
    } else {
        (0, (k as u64).div_ceil(32))
    };
    let (lo, len) = (b0 * 32, (nb * 32).min(k as u64 - b0 * 32));
    let w_bytes = open_bytes(tier, &norm, &p.norm.1, 4 * lo, 4 * len)?;
    let w: Vec<f32> = w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let dev = if fragment {
        let s = crate::chain::rms_scaled(&e, ctx.eps, ctx.embd_scale);
        let v: Vec<f64> = s[lo as usize..(lo + len) as usize].iter().zip(&w).map(|(a, &b)| a * b as f64).collect();
        crate::chain::block_deviation(&p.x, layout, k, b0 as usize, &v)
    } else {
        deviation(&p.x, layout, &expected_input(&e, &w, ctx.eps, ctx.embd_scale))
    };
    match dev {
        Some(dev) if dev <= THRESHOLD => Ok(t),
        _ => Err(Fail::Mismatch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16;

    #[test]
    fn the_draw_stays_inside_the_prefix() {
        for s in 0..64u8 {
            let t = position(&[s; 32], 7);
            assert!((7..39).contains(&t));
        }
        assert!(fits(39, 7));
        assert!(!fits(38, 7));
    }

    #[test]
    fn expected_token_follows_the_table() {
        let mut table = [0u32; 256];
        for (b, id) in table.iter_mut().enumerate() {
            *id = 1000 + b as u32;
        }
        let ctx = PrefixCtx { table, offset: 6, embd_entry: 1, norm_entry: 2, eps: 1e-6, embd_scale: 1.0 };
        let mut txid = [0u8; 32];
        txid[3] = 0x41;
        assert_eq!(expected_token(&ctx, &txid, 9), Some(1000 + 0x41));
        assert_eq!(expected_token(&ctx, &txid, 5), None);
        assert_eq!(expected_token(&ctx, &txid, 37), Some(1000));
        assert_eq!(expected_token(&ctx, &txid, 38), None);
    }

    #[test]
    fn deviation_of_a_rounded_row_is_below_half() {
        let v: Vec<f64> = (0..64).map(|i| ((i * 37 % 23) as f64 - 11.0) * 0.01).collect();
        let mut leaf = Vec::new();
        for blk in v.chunks(32) {
            let amax = blk.iter().fold(0f64, |m, a| m.max(a.abs()));
            let d = f16::round((amax / 127.0) as f32);
            leaf.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            leaf.extend_from_slice(&[0, 0]);
            for a in blk {
                leaf.push(libm::round(a / d as f64) as i8 as u8);
            }
        }
        let dev = deviation(&leaf, q8::MMVQ, &v).unwrap();
        assert!(dev <= 0.5 + 0.2, "{dev}");
        let shifted: Vec<f64> = v.iter().map(|a| a + 0.05).collect();
        assert!(deviation(&leaf, q8::MMVQ, &shifted).unwrap() > 1.0);
        let mut nan = v.clone();
        nan[5] = f64::NAN;
        assert_eq!(deviation(&leaf, q8::MMVQ, &nan), None);
        let zero = vec![0u8; 72];
        assert_eq!(deviation(&zero, q8::MMVQ, &vec![0.0; 64]), Some(0.0));
        assert_eq!(deviation(&zero, q8::MMVQ, &v), None);
    }
}
