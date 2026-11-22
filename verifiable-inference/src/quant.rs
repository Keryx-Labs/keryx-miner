//! Weight row dequantization.

use crate::f16;

pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q2_K: u32 = 10;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ2_XXS: u32 = 16;
pub const BF16: u32 = 30;

/// (bytes per block, values per block).
pub fn block_size(wtype: u32) -> Option<(usize, usize)> {
    match wtype {
        Q4_0 => Some((18, 32)),
        Q5_0 => Some((22, 32)),
        F16 => Some((2, 1)),
        Q8_0 => Some((34, 32)),
        Q2_K => Some((84, 256)),
        IQ2_XXS => Some((66, 256)),
        Q4_K => Some((144, 256)),
        Q5_K => Some((176, 256)),
        Q6_K => Some((210, 256)),
        BF16 => Some((2, 1)),
        _ => None,
    }
}

pub fn row_bytes(wtype: u32, k: usize) -> Option<usize> {
    block_size(wtype).map(|(bb, bv)| k / bv * bb)
}

/// 6-bit scale and min of sub-block `j` of a Q4_K/Q5_K block.
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) } else { ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `v` rounded to the nearest bfloat16, ties to even.
pub fn round_bf16(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    let bits = v.to_bits();
    let r = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(r & 0xffff_0000)
}

/// Dequantize one weight row of `k` values.
pub fn dequant_row(wtype: u32, row: &[u8], k: usize) -> Option<Vec<f32>> {
    let (bb, bv) = block_size(wtype)?;
    if row.len() < k / bv * bb || !k.is_multiple_of(bv) {
        return None;
    }
    let mut y = vec![0f32; k];
    for (b, out) in y.chunks_mut(bv).enumerate() {
        let blk = &row[b * bb..(b + 1) * bb];
        match wtype {
            BF16 => out[0] = bf16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
            F16 => out[0] = f16::read(blk, 0),
            Q2_K => {
                let (sc, qs) = (&blk[0..16], &blk[16..80]);
                let d = f16::read(blk, 80);
                let dmin = f16::read(blk, 82);
                for (v, o) in out.iter_mut().enumerate() {
                    let g = v / 16;
                    let q = (qs[32 * (v / 128) + v % 32] >> (2 * ((v / 32) % 4))) & 3;
                    *o = d * (sc[g] & 0xf) as f32 * q as f32 - dmin * (sc[g] >> 4) as f32;
                }
            }
            IQ2_XXS => {
                let d = f16::read(blk, 0);
                for ib in 0..8 {
                    let q = &blk[2 + 8 * ib..2 + 8 * ib + 8];
                    let signs = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
                    let db = d * (0.5 + (signs >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid = crate::iq2::IQ2XXS_GRID[q[l] as usize].to_le_bytes();
                        let sg = crate::iq2::KSIGNS_IQ2XS[((signs >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            let v = db * grid[j] as f32;
                            out[32 * ib + 8 * l + j] = if sg & (1 << j) != 0 { -v } else { v };
                        }
                    }
                }
            }
            Q8_0 => {
                let d = f16::read(blk, 0);
                for j in 0..32 {
                    out[j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
            Q4_0 => {
                let d = f16::read(blk, 0);
                for j in 0..16 {
                    out[j] = ((blk[2 + j] & 0xf) as i32 - 8) as f32 * d;
                    out[j + 16] = ((blk[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
            Q5_0 => {
                let d = f16::read(blk, 0);
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    out[j] = (((blk[6 + j] & 0xf) as u32 | xh0) as i32 - 16) as f32 * d;
                    out[j + 16] = (((blk[6 + j] >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
                }
            }
            Q4_K => {
                let d = f16::read(blk, 0);
                let dmin = f16::read(blk, 2);
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                for g in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * g, sc);
                    let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
                    let q = &qs[32 * g..32 * g + 32];
                    for l in 0..32 {
                        out[64 * g + l] = d1 * (q[l] & 0xf) as f32 - mm1;
                        out[64 * g + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
                    }
                }
            }
            Q5_K => {
                let d = f16::read(blk, 0);
                let dmin = f16::read(blk, 2);
                let sc = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                for g in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * g, sc);
                    let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
                    let (u1, u2) = (1u8 << (2 * g), 2u8 << (2 * g));
                    let q = &qs[32 * g..32 * g + 32];
                    for l in 0..32 {
                        let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        out[64 * g + l] = d1 * ((q[l] & 0xf) + h1) as f32 - mm1;
                        out[64 * g + 32 + l] = d2 * ((q[l] >> 4) + h2) as f32 - mm2;
                    }
                }
            }
            Q6_K => {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = f16::read(blk, 208);
                for h in 0..2 {
                    let (ql, qh, sc) = (&ql[64 * h..], &qh[32 * h..], &sc[8 * h..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let o = 128 * h;
                        out[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        out[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        out[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        out[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
            _ => return None,
        }
    }
    Some(y)
}

/// Q4_K/Q5_K row in tiled form.
pub fn k_tile_form(wtype: u32, row: &[u8], k: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if wtype != Q4_K && wtype != Q5_K {
        return None;
    }
    let (bb, _) = block_size(wtype)?;
    let mut w_eff = vec![0f32; k];
    let mut b_h = vec![0f32; k / 32];
    for b in 0..k / 256 {
        let blk = &row[b * bb..(b + 1) * bb];
        let d = f16::read(blk, 0);
        let dmin = f16::read(blk, 2);
        let sc = &blk[4..16];
        for sb in 0..8 {
            let (s, m) = scale_min_k4(sb, sc);
            let a_h = f16::round(d * f16::round(s as f32));
            b_h[b * 8 + sb] = f16::round(dmin * f16::round(m as f32));
            let g = sb / 2;
            for l in 0..32 {
                let q = if wtype == Q4_K {
                    let v = blk[16 + 32 * g + l];
                    if sb.is_multiple_of(2) { v & 0xf } else { v >> 4 }
                } else {
                    let v = blk[48 + 32 * g + l];
                    let h = blk[16 + l];
                    let (lo, bit) = if sb.is_multiple_of(2) { (v & 0xf, 1u8 << (2 * g)) } else { (v >> 4, 2u8 << (2 * g)) };
                    lo + if h & bit != 0 { 16 } else { 0 }
                };
                w_eff[256 * b + 32 * sb + l] = if a_h == 0.0 { 0.0 } else { a_h * q as f32 };
            }
        }
    }
    Some((w_eff, b_h))
}

/// Q2_K row in tiled form.
pub fn q2k_tile_form(row: &[u8], k: usize) -> Option<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if !k.is_multiple_of(256) || row.len() < k / 256 * 84 {
        return None;
    }
    let mut q = vec![0u8; k];
    let mut a = vec![0f32; k / 16];
    let mut m = vec![0f32; k / 16];
    for b in 0..k / 256 {
        let blk = &row[b * 84..(b + 1) * 84];
        let d = f16::read(blk, 80);
        let dmin = f16::read(blk, 82);
        for g in 0..16 {
            a[16 * b + g] = f16::round(d * (blk[g] & 0xf) as f32);
            m[16 * b + g] = f16::round(dmin * (blk[g] >> 4) as f32);
        }
        for v in 0..256 {
            q[256 * b + v] = (blk[16 + 32 * (v / 128) + v % 32] >> (2 * ((v / 32) % 4))) & 3;
        }
    }
    Some((q, a, m))
}

/// Block scales and offset of a Q4_0/Q5_0 row.
pub fn offset_block_scales(wtype: u32, row: &[u8], k: usize) -> Option<(Vec<f32>, f64)> {
    let (bb, off) = match wtype {
        Q4_0 => (18, 8.0),
        Q5_0 => (22, 16.0),
        _ => return None,
    };
    Some(((0..k / 32).map(|b| f16::read(row, b * bb)).collect(), off))
}
