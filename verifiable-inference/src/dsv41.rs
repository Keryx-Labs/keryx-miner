//! Compressed attention rows of DeepSeek-V4.1: a block of `ratio` consecutive tokens pooled into
//! one key/value row, read by the attention of the layers that share it.

use crate::attn::{Yarn, rms_norm, rope_norm_yarn};

/// Pooled latent of a block: per value, the members' outputs weighted by the softmax of their
/// gate outputs (no gate: one member), then RMS-normed and scaled by `norm`.
pub fn comp_latent(kvs: &[&[f32]], gates: Option<&[&[f32]]>, norm: &[f32], eps: f64) -> Option<Vec<f64>> {
    let d = norm.len();
    if kvs.is_empty() || kvs.iter().any(|k| k.len() != d) {
        return None;
    }
    let mut out = vec![0f64; d];
    match gates {
        None if kvs.len() == 1 => out.iter_mut().zip(kvs[0]).for_each(|(o, &v)| *o = v as f64),
        Some(g) if g.len() == kvs.len() && g.iter().all(|r| r.len() == d) => {
            for (i, o) in out.iter_mut().enumerate() {
                let m = g.iter().map(|r| r[i] as f64).fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = g.iter().map(|r| libm::exp(r[i] as f64 - m)).collect();
                let z: f64 = e.iter().sum();
                *o = kvs.iter().zip(&e).map(|(k, w)| k[i] as f64 * w).sum::<f64>() / z;
            }
        }
        _ => return None,
    }
    rms_norm(&mut out, Some(norm), eps);
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// Rotary part of a compressed row: the last `n_rot` values of `x` rotated at `pos`.
pub fn rope_tail(x: &mut [f64], pos: u64, n_rot: usize, base: f64, yarn: &Yarn) {
    let off = x.len() - n_rot;
    rope_norm_yarn(x, off, pos, n_rot, base, yarn, false);
}

/// Values of `x` as a Q8_1 input holds them: blocks of 32 scaled by half the largest magnitude
/// over 127, the scale stored in half precision.
pub fn q8_1_values(x: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for b in x.chunks(32) {
        let amax = b.iter().fold(0f32, |m, &v| m.max((v as f32).abs()));
        let d = amax / 127.0;
        let dh = crate::f16::round(d) as f64;
        out.extend(b.iter().map(|&v| if amax == 0.0 { 0.0 } else { libm::roundf(v as f32 / d) as f64 * dh }));
    }
    out
}

/// The row a compressed cache stores for a block starting at `pos`: the rotated latent, then
/// the indexer key `rope(rms_norm(W · q8_1(latent)) · k_norm)` when `ik` is given (rows of
/// `w_type` of `latent.len()` values, one per key value).
#[allow(clippy::too_many_arguments)]
pub fn comp_row(latent: &[f64], ik: Option<(&[u8], u32, &[f32])>, pos: u64, n_rot: usize, base: f64, yarn: &Yarn, eps: f64) -> Option<Vec<f64>> {
    let d = latent.len();
    let mut row = latent.to_vec();
    rope_tail(&mut row, pos, n_rot, base, yarn);
    if let Some((w, w_type, k_norm)) = ik {
        let m = k_norm.len();
        let rb = w.len().checked_div(m).filter(|&r| r > 0 && r * m == w.len())?;
        let x = q8_1_values(latent);
        let mut key = (0..m)
            .map(|r| crate::quant::dequant_row(w_type, &w[r * rb..(r + 1) * rb], d).map(|wr| wr.iter().zip(&x).map(|(&a, &b)| a as f64 * b).sum::<f64>()))
            .collect::<Option<Vec<f64>>>()?;
        rms_norm(&mut key, Some(k_norm), eps);
        rope_tail(&mut key, pos, n_rot, base, yarn);
        row.extend(key);
    }
    row.iter().all(|v| v.is_finite()).then_some(row)
}

/// Whether the stored row `bytes` (`row_type`) holds `want` at the block starting at `pos`: one
/// quantization step per value, plus the drift of single-precision angles and sums.
pub fn row_matches(bytes: &[u8], row_type: u32, want: &[f64], pos: u64) -> bool {
    let Some(got) = crate::quant::dequant_row(row_type, bytes, want.len()) else { return false };
    let rel = 2e-5 + 4e-8 * pos as f64;
    got.chunks(32).zip(want.chunks(32)).all(|(g, w)| {
        let amax = w.iter().fold(0f64, |m, v| m.max(v.abs()));
        let step = if row_type == crate::quant::Q8_0 { amax / 127.0 * 1.01 } else { amax * 2e-3 };
        g.iter().zip(w).all(|(&a, &b)| (a as f64 - b).abs() <= step + rel * amax + 1e-30)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// ggml's pooled, normed and rotated rows of 3 blocks at ratio 2 and ratio 1.
    #[test]
    fn compressed_rows_match_ggml() {
        let f = f32s(include_bytes!("../testdata/dsv41-comp.bin"));
        let yarn = Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 };
        let (d, nb) = (512usize, 3usize);
        let mut o = 0;
        for ratio in [2usize, 1] {
            let take = |o: &mut usize, n: usize| {
                let v = f[*o..*o + n].to_vec();
                *o += n;
                v
            };
            let (kv, sc, w, want) = (take(&mut o, d * ratio * nb), take(&mut o, d * ratio * nb), take(&mut o, d), take(&mut o, d * nb));
            for b in 0..nb {
                let kvs: Vec<&[f32]> = (0..ratio).map(|j| &kv[(b * ratio + j) * d..(b * ratio + j + 1) * d]).collect();
                let gs: Vec<&[f32]> = (0..ratio).map(|j| &sc[(b * ratio + j) * d..(b * ratio + j + 1) * d]).collect();
                let gates = (ratio > 1).then_some(&gs[..]);
                let mut x = comp_latent(&kvs, gates, &w, 1e-20).unwrap();
                rope_tail(&mut x, ((b + 1000) * ratio) as u64, 64, 160_000.0, &yarn);
                let dev = x.iter().zip(&want[b * d..(b + 1) * d]).fold(0f64, |m, (a, &e)| m.max((a - e as f64).abs()));
                assert!(dev < 1e-4, "ratio {ratio} block {b}: {dev}");
            }
        }
        assert_eq!(o, f.len());
    }

    fn vals(seed: u64, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
    }

    fn q8_0(v: &[f64]) -> Vec<u8> {
        let mut out = Vec::new();
        for b in v.chunks(32) {
            let amax = b.iter().fold(0f32, |m, &a| m.max((a as f32).abs()));
            let d = crate::f16::round(amax / 127.0);
            out.extend_from_slice(&crate::f16::from_f32(d).to_le_bytes());
            out.extend(b.iter().map(|&a| (if d == 0.0 { 0.0 } else { libm::roundf(a as f32 / d) }) as i8 as u8));
        }
        out
    }

    struct Block {
        kvs: Vec<Vec<f32>>,
        gates: Vec<Vec<f32>>,
        norm: Vec<f32>,
        w: Vec<u8>,
        k_norm: Vec<f32>,
        yarn: Yarn,
    }

    fn block() -> Block {
        let rows: Vec<f64> = vals(9, 128 * 512, 0.05).iter().map(|&x| x as f64).collect();
        Block {
            kvs: (0..2).map(|j| vals(10 + j, 512, 2.0)).collect(),
            gates: (0..2).map(|j| vals(20 + j, 512, 3.0)).collect(),
            norm: (0..512).map(|i| 0.5 + 0.01 * (i % 37) as f32).collect(),
            w: rows.chunks(512).flat_map(q8_0).collect(),
            k_norm: (0..128).map(|i| 0.8 + 0.02 * (i % 11) as f32).collect(),
            yarn: Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 },
        }
    }

    impl Block {
        fn expect(&self, gates_swapped: bool, pos: u64) -> Vec<f64> {
            let kvs: Vec<&[f32]> = self.kvs.iter().map(|v| &v[..]).collect();
            let mut gs: Vec<&[f32]> = self.gates.iter().map(|v| &v[..]).collect();
            if gates_swapped {
                gs.reverse();
            }
            let latent = comp_latent(&kvs, Some(&gs), &self.norm, 1e-20).unwrap();
            comp_row(&latent, Some((&self.w, crate::quant::Q8_0, &self.k_norm)), pos, 64, 160_000.0, &self.yarn, 1e-20).unwrap()
        }
    }

    #[test]
    fn an_honest_compressed_row_matches_and_forgeries_do_not() {
        let b = block();
        let pos = 2_048;
        let honest = b.expect(false, pos);
        assert_eq!(honest.len(), 640);
        let stored = q8_0(&honest);
        assert!(row_matches(&stored, crate::quant::Q8_0, &honest, pos));
        // a stored value one step off still passes: the engine may round the other way
        let mut flip = stored.clone();
        flip[2 + 5] = flip[2 + 5].wrapping_add(1);
        assert!(row_matches(&flip, crate::quant::Q8_0, &honest, pos));
        let mut far = stored.clone();
        far[2 + 5] = far[2 + 5].wrapping_add(3);
        assert!(!row_matches(&far, crate::quant::Q8_0, &honest, pos));
        assert!(!row_matches(&q8_0(&b.expect(true, pos)), crate::quant::Q8_0, &honest, pos));
        assert!(!row_matches(&q8_0(&b.expect(false, pos + 2)), crate::quant::Q8_0, &honest, pos));
        let mut no_key = honest.clone();
        no_key[512..].iter_mut().for_each(|v| *v = 0.0);
        assert!(!row_matches(&q8_0(&no_key), crate::quant::Q8_0, &honest, pos));
    }

    #[test]
    fn pooling_needs_one_gate_per_member() {
        let (a, b, w) = (vec![1.0f32; 512], vec![2.0f32; 512], vec![1.0f32; 512]);
        assert!(comp_latent(&[&a, &b], None, &w, 1e-20).is_none());
        assert!(comp_latent(&[&a, &b], Some(&[&a]), &w, 1e-20).is_none());
        let x = comp_latent(&[&a, &b], Some(&[&w, &w]), &w, 1e-20).unwrap();
        assert!(x.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }
}
