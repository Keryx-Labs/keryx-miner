//! Q8_1 activation rows.

use crate::f16;

/// Scale and quantized values of each 32-value block of a row.
pub fn blocks(leaf: &[u8], layout: u8, k: usize) -> Option<Vec<(f32, [i8; 32])>> {
    let mut out = Vec::with_capacity(k.div_ceil(32));
    match layout {
        MMVQ => {
            if leaf.len() < k.div_ceil(32) * 36 {
                return None;
            }
            for blk in leaf.chunks_exact(36).take(k.div_ceil(32)) {
                out.push((f16::read(blk, 0), std::array::from_fn(|j| blk[4 + j] as i8)));
            }
        }
        MMQ_D4 | MMQ_DS4 | MMQ_D2S6 => {
            if leaf.len() < k.div_ceil(128) * 144 {
                return None;
            }
            for blk in leaf.chunks_exact(144).take(k.div_ceil(128)) {
                for sb in 0..4 {
                    let d = match layout {
                        MMQ_D4 => f32::from_le_bytes([blk[4 * sb], blk[4 * sb + 1], blk[4 * sb + 2], blk[4 * sb + 3]]),
                        MMQ_DS4 => f16::read(blk, 4 * sb),
                        _ => f16::read(blk, 2 * (sb / 2)),
                    };
                    out.push((d, std::array::from_fn(|j| blk[16 + 32 * sb + j] as i8)));
                }
            }
        }
        _ => return None,
    }
    out.truncate(k.div_ceil(32));
    Some(out)
}

/// Whether a block is one the quantizer writes: a zero scale with zero values, or a positive
/// scale with a value at full magnitude.
pub fn canonical(d: f32, q: &[i8; 32]) -> bool {
    (d == 0.0 && q.iter().all(|&v| v == 0)) || (d > 0.0 && q.iter().any(|&v| v.unsigned_abs() == 127))
}

/// Largest `|s - d * sum(q)|` of a block, in units of its scale.
const SUM_SLACK: f64 = 24.0;

/// Whether every stored partial sum of a row agrees with the values of its block.
pub fn sums_consistent(leaf: &[u8], layout: u8, k: usize) -> bool {
    let (Some((_, sums)), Some(blocks)) = (decode(leaf, layout, k), blocks(leaf, layout, k)) else {
        return false;
    };
    let Some(sums) = sums else { return true };
    sums.iter().zip(&blocks).all(|(&s, (d, q))| {
        let sq = q.iter().map(|&v| v as f64).sum::<f64>() * *d as f64;
        (s as f64 - sq).abs() <= SUM_SLACK * d.abs() as f64 + (s as f64).abs() / 1024.0
    })
}

pub const MMVQ: u8 = 0;
pub const MMQ_D4: u8 = 1;
pub const MMQ_DS4: u8 = 2;
pub const MMQ_D2S6: u8 = 3;

/// Values and stored partial sums of a row.
pub fn decode(leaf: &[u8], layout: u8, k: usize) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
    let mut x = Vec::with_capacity(k);
    let mut s = Vec::new();
    match layout {
        MMVQ => {
            if leaf.len() < k.div_ceil(32) * 36 {
                return None;
            }
            for blk in leaf.chunks_exact(36).take(k.div_ceil(32)) {
                let d = f16::read(blk, 0);
                s.push(f16::read(blk, 2));
                for j in 0..32 {
                    x.push((blk[4 + j] as i8) as f32 * d);
                }
            }
        }
        MMQ_D4 | MMQ_DS4 | MMQ_D2S6 => {
            if leaf.len() < k.div_ceil(128) * 144 {
                return None;
            }
            for blk in leaf.chunks_exact(144).take(k.div_ceil(128)) {
                for sb in 0..4 {
                    let d = match layout {
                        MMQ_D4 => f32::from_le_bytes([blk[4 * sb], blk[4 * sb + 1], blk[4 * sb + 2], blk[4 * sb + 3]]),
                        MMQ_DS4 => {
                            s.push(f16::read(blk, 4 * sb + 2));
                            f16::read(blk, 4 * sb)
                        }
                        _ => f16::read(blk, 2 * (sb / 2)),
                    };
                    for j in 0..32 {
                        x.push((blk[16 + 32 * sb + j] as i8) as f32 * d);
                    }
                }
            }
        }
        _ => return None,
    }
    x.truncate(k);
    let has_s = layout == MMVQ || layout == MMQ_DS4;
    if has_s {
        s.truncate(k / 32);
    }
    Some((x, has_s.then_some(s)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::q8_1_row;

    fn row() -> Vec<u8> {
        q8_1_row(&(0..512).map(|i| ((i * 37 % 101) as f64 - 50.0) * 0.013).collect::<Vec<_>>())
    }

    fn set_sum(x: &mut [u8], block: usize, s: f32) {
        x[block * 36 + 2..block * 36 + 4].copy_from_slice(&f16::from_f32(s).to_le_bytes());
    }

    #[test]
    fn a_zero_scale_needs_zero_values() {
        let mut q = [0i8; 32];
        assert!(canonical(0.0, &q));
        q[3] = 5;
        assert!(!canonical(0.0, &q));
        q[3] = 127;
        assert!(canonical(0.5, &q));
    }

    #[test]
    fn honest_sums_pass() {
        assert!(sums_consistent(&row(), MMVQ, 512));
    }

    #[test]
    fn a_shifted_sum_fails() {
        let mut x = row();
        let (d, s) = (f16::read(&x, 36 * 3), f16::read(&x, 36 * 3 + 2));
        set_sum(&mut x, 3, s + 100.0 * d);
        assert!(!sums_consistent(&x, MMVQ, 512));
    }

    #[test]
    fn a_zero_block_needs_a_zero_sum() {
        let mut x = q8_1_row(&vec![0.0; 512]);
        assert!(sums_consistent(&x, MMVQ, 512));
        set_sum(&mut x, 0, 0.5);
        assert!(!sums_consistent(&x, MMVQ, 512));
    }
}
