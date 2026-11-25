//! Compressed attention rows of DeepSeek-V4.1: a block of `ratio` consecutive tokens pooled into
//! one key/value row, read by the attention of the layers that share it.

use crate::attn::{Yarn, rms_norm, rope_norm_yarn};
use crate::chain::{ChainCtx, f32_weights, f32s, seq_layout};
use crate::manifest::{self, Chain};
use crate::merkle::{Hash, b3_fold, b3_verify};
use crate::opening::{CompOpeningPoint, LeafBytes};
use crate::verify::{Fail, ResponseCtx, TierCtx, open_bytes, open_entry};

/// Draw index of the compressed row point.
pub const COMP_POINT: u32 = crate::dsa::SEL_POINT + 1;

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

fn leaf_root(n_leaves: u64, index: u64, leaf: &LeafBytes) -> Result<Hash, Fail> {
    b3_fold(n_leaves, index, blake3::hash(&leaf.bytes).as_bytes(), &leaf.leaf_path).ok_or(Fail::Trace)
}

/// Verify the compressed row point of a response under `seed`: the drawn row of the compressed
/// cache of the drawn source layer is the pooled latent of its block's members, normed and
/// rotated, with its indexer key.
pub fn verify_comp_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &CompOpeningPoint) -> Result<(), Fail> {
    let (nl, n_pos) = (ctx.n_layers as u64, resp.n_pos as u64);
    if ctx.n_comp == 0 || nl == 0 || n_pos == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = crate::draw::point(seed, COMP_POINT);
    let slot = d[1] % ctx.n_comp as u64;
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let c = match open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?.chain {
        Chain::Layer(ll) if ll.layer as u64 == layer => ll.comp.ok_or(Fail::Manifest)?,
        _ => return Err(Fail::Manifest),
    };
    let ratio = c.ratio as u64;
    if c.slot as u64 != slot || c.src as u64 != layer || ratio == 0 {
        return Err(Fail::Manifest);
    }
    let n_blocks = n_pos / ratio;
    if n_blocks == 0 {
        return Ok(());
    }
    let k = d[0] % n_blocks;
    let gated = c.gate != manifest::NONE;
    if gated != (ratio > 1) || op.entries.len() != 1 + gated as usize || op.members.len() as u64 != ratio {
        return Err(Fail::Shape);
    }
    let kv = open_entry(tier, &op.entries[0], c.kv as u64)?;
    let dim = kv.m;
    if kv.role != manifest::ROLE_PRODUCT || c.n_rot as u64 > dim || !dim.is_multiple_of(32) {
        return Err(Fail::Manifest);
    }
    if gated {
        let g = open_entry(tier, &op.entries[1], c.gate as u64)?;
        if g.role != manifest::ROLE_PRODUCT || g.m != dim {
            return Err(Fail::Manifest);
        }
    }
    let with_key = c.ik != manifest::NONE;
    if op.weights.len() != 1 + 2 * with_key as usize {
        return Err(Fail::Shape);
    }
    let norm = f32_weights(tier, &op.weights[0], c.norm, dim, 0, dim)?;
    let key = if with_key {
        let me = open_entry(tier, &op.weights[1].0, c.ik as u64)?;
        if me.role != manifest::ROLE_MATRIX || me.k != dim || me.m == 0 || !me.m.is_multiple_of(32) {
            return Err(Fail::Manifest);
        }
        let w = open_bytes(tier, &me, &op.weights[1].1, 0, me.n_bytes)?;
        let kn = f32_weights(tier, &op.weights[2], c.k_norm, me.m, 0, me.m)?;
        Some((w, me.wtype, kn))
    } else {
        None
    };
    let shape = crate::chain::shape_any(tier, ctx);
    let (mut kvs, mut gates) = (Vec::new(), Vec::new());
    for (j, m) in op.members.iter().enumerate() {
        let p = k * ratio + j as u64;
        let root = leaf_root(shape.n_leaves, 2 * c.kv as u64 + 1, &m.k)?;
        if !b3_verify(&op.pos_root, n_pos, p, &root, &m.pos_path) {
            return Err(Fail::Trace);
        }
        if gated && leaf_root(shape.n_leaves, 2 * c.gate as u64 + 1, &m.g)? != root {
            return Err(Fail::Trace);
        }
        if m.k.bytes.len() as u64 != 4 * dim || (gated && m.g.bytes.len() as u64 != 4 * dim) || (!gated && !m.g.bytes.is_empty()) {
            return Err(Fail::Shape);
        }
        kvs.push(f32s(&m.k.bytes));
        if gated {
            gates.push(f32s(&m.g.bytes));
        }
    }
    let kr: Vec<&[f32]> = kvs.iter().map(|v| &v[..]).collect();
    let gr: Vec<&[f32]> = gates.iter().map(|v| &v[..]).collect();
    let eps = f32::from_bits(c.eps) as f64;
    let latent = comp_latent(&kr, gated.then_some(&gr[..]), &norm, eps).ok_or(Fail::Mismatch)?;
    let base = f32::from_bits(c.rope_base) as f64;
    let want = comp_row(&latent, key.as_ref().map(|(w, t, kn)| (&w[..], *t, &kn[..])), k * ratio, c.n_rot as usize, base, &c.yarn(), eps).ok_or(Fail::Mismatch)?;
    let rb = want.len() as u64 / 32 * 34;
    let layout = seq_layout(ctx);
    let leaf = layout.comp(slot).ok_or(Fail::Shape)?;
    let bytes = crate::seq::open_span(&op.seq_root, &layout, leaf, n_blocks * rb, k * rb, (k + 1) * rb, &op.row).ok_or(Fail::Trace)?;
    row_matches(bytes, crate::quant::Q8_0, &want, k * ratio).then_some(()).ok_or(Fail::Mismatch)
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

    mod point {
        use super::*;
        use crate::manifest::{AttnLinks, CompLinks, Entry, LayerLinks, ProductLink};
        use crate::merkle::{B3_CHUNK, b3_chunks_proof, b3_path, b3_root, rt_range_proof};
        use crate::opening::{ChunkRange, EntryOpening, SelMember};
        use crate::seq::{SeqLayout, SeqSpan};

        const N: usize = 6;
        const NONE: u32 = manifest::NONE;

        struct Kit {
            weights: Vec<u8>,
            chunk_leaves: Vec<Hash>,
            entries: Vec<Vec<u8>>,
            entry_leaves: Vec<Hash>,
            tier: TierCtx,
            ctx: ChainCtx,
            pos_leaves: Vec<Vec<Vec<u8>>>,
            pos_roots: Vec<Hash>,
            pos_root: Hash,
            seq_leaves: Vec<Vec<u8>>,
            seq_root: Hash,
            offs: [usize; 3],
        }

        fn entry(name: &str, role: u8, wtype: u32, k: usize, m: usize, off: usize, n: usize, chain: Chain) -> Vec<u8> {
            let e = Entry {
                name: name.into(), wtype, k: k as u64, m: m as u64, n_exp: 1, chunk_base: (off / 32) as u64, n_bytes: n as u64,
                row_bytes: if m > 0 { (n / m.max(1)) as u32 } else { 0 }, role, glu_op: 0, glu_form: manifest::FORM_NONE, partner: NONE, bias: NONE, chain,
            };
            let mut b = crate::testkit::serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            b
        }

        fn build(swap: bool) -> Kit {
            let blk = block();
            let mut weights = Vec::new();
            let mut offs = [0usize; 3];
            for (o, b) in offs.iter_mut().zip([blk.norm.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>(), blk.w.clone(), blk.k_norm.iter().flat_map(|a| a.to_le_bytes()).collect()]) {
                *o = weights.len();
                weights.extend(b);
            }
            let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let attn = AttnLinks {
                q: 0, k: 0, v: NONE, q_norm: NONE, k_norm: NONE, n_head: 1, n_head_kv: 1, head_dim: 512, rope_kind: 0, n_rot: 64, rope_base: 0,
                kq_scale: 1f32.to_bits(), swa_window: 0, softcap: 0, rope_freqs: NONE, v_norm: 0, gated: 0, slot: 0, mla_rank: 0, absorb: NONE, out: NONE,
            };
            let y = blk.yarn;
            let comp = CompLinks {
                src: 0, slot: 0, ratio: 2, kv: 0, gate: 1, norm: 3, ik: 4, k_norm: 5, n_rot: 64, rope_base: 160_000f32.to_bits(),
                freq_scale: (y.freq_scale as f32).to_bits(), ext_factor: 1f32.to_bits(), attn_factor: (y.attn_factor as f32).to_bits(),
                beta_fast: 32f32.to_bits(), beta_slow: 1f32.to_bits(), n_ctx_orig: 65_536, eps: 1e-20f32.to_bits(),
            };
            let none_link = Chain::Product(ProductLink { layer: 0, link: manifest::LINK_NONE, norm: NONE, src: NONE });
            let layer = LayerLinks { layer: 0, proj: 0, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: Some(attn), recur: None, hc: None, dsa: None, comp: Some(comp) };
            let entries = vec![
                entry("blk.0.attn_compressor_kv.weight", manifest::ROLE_PRODUCT, crate::quant::Q8_0, 64, 512, 0, 0, none_link),
                entry("blk.0.attn_compressor_gate.weight", manifest::ROLE_PRODUCT, crate::quant::Q8_0, 64, 512, 0, 0, none_link),
                entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, 0, Chain::Layer(layer)),
                entry("blk.0.attn_compressor_norm.weight", manifest::ROLE_NORM, 0, 512, 1, offs[0], 2048, Chain::None),
                entry("blk.0.indexer.attn_k.weight", manifest::ROLE_MATRIX, crate::quant::Q8_0, 512, 128, offs[1], blk.w.len(), Chain::None),
                entry("blk.0.indexer.k_norm.weight", manifest::ROLE_NORM, 0, 128, 1, offs[2], 512, Chain::None),
            ];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx { r_t: rt_range_proof(&chunk_leaves, 0, 1).0, n_chunks: chunk_leaves.len() as u64, m_t: rt_range_proof(&entry_leaves, 0, 1).0, n_entries: entries.len() as u64, n_products: 2 };
            let ctx = ChainCtx { n_sel: 0, n_comp: 1, n_layers: 1, layer_base: 2, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 1, n_recur: 0, seq_heads: 1 };
            let kvs: Vec<Vec<f32>> = (0..N).map(|t| vals(100 + t as u64, 512, 2.0)).collect();
            let gates: Vec<Vec<f32>> = (0..N).map(|t| vals(200 + t as u64, 512, 3.0)).collect();
            let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
            let n_leaves = crate::chain::n_pos_leaves(2, 1, 0) as usize;
            let pos_leaves: Vec<Vec<Vec<u8>>> = (0..N)
                .map(|t| {
                    let mut l = vec![Vec::new(); n_leaves];
                    l[1] = f(&kvs[t]);
                    l[3] = f(&gates[t]);
                    l[n_leaves - 1] = crate::chain::meta(t as u32, N as u32, 7).to_vec();
                    l
                })
                .collect();
            let pos_roots: Vec<Hash> = pos_leaves.iter().map(|l| b3_root(&l.iter().map(|b| *blake3::hash(b).as_bytes()).collect::<Vec<_>>())).collect();
            let pos_root = b3_root(&pos_roots);
            let rows: Vec<u8> = (0..N / 2)
                .flat_map(|k| {
                    let kv: Vec<&[f32]> = (0..2).map(|j| &kvs[2 * k + j][..]).collect();
                    let mut gs: Vec<&[f32]> = (0..2).map(|j| &gates[2 * k + j][..]).collect();
                    if swap {
                        gs.reverse();
                    }
                    let lat = comp_latent(&kv, Some(&gs), &blk.norm, 1e-20).unwrap();
                    q8_0(&comp_row(&lat, Some((&blk.w, crate::quant::Q8_0, &blk.k_norm)), 2 * k as u64, 64, 160_000.0, &y, 1e-20).unwrap())
                })
                .collect();
            let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: 1, n_comp: 1 };
            let mut seq_leaves = vec![Vec::new(); layout.n_leaves() as usize];
            seq_leaves[layout.comp(0).unwrap() as usize] = rows;
            let seq_root = b3_root(&seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect::<Vec<_>>());
            Kit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos_leaves, pos_roots, pos_root, seq_leaves, seq_root, offs }
        }

        impl Kit {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn range(&self, off: usize, n: usize) -> ChunkRange {
                let (first, count) = (off / 32, n / 32);
                ChunkRange { first_chunk: first as u64, chunks: self.weights[off..off + n].to_vec(), proof: rt_range_proof(&self.chunk_leaves, first as u64, count as u64).1 }
            }

            fn resp(&self) -> ResponseCtx {
                ResponseCtx { trace_root: crate::seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: N as u32, n_pos: N as u32 }
            }

            fn leaf(&self, t: usize, i: usize) -> LeafBytes {
                let hashes: Vec<Hash> = self.pos_leaves[t].iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                LeafBytes { bytes: self.pos_leaves[t][i].clone(), leaf_path: b3_path(&hashes, i) }
            }

            fn open(&self, seed: &Hash) -> CompOpeningPoint {
                let k = (crate::draw::point(seed, COMP_POINT)[0] % (N as u64 / 2)) as usize;
                let rb = 640 / 32 * 34;
                let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: 1, n_comp: 1 };
                let leaf = layout.comp(0).unwrap() as usize;
                let data = &self.seq_leaves[leaf];
                let (lo, hi) = (k * rb, (k + 1) * rb);
                let (c0, c1) = (lo / B3_CHUNK, (hi - 1) / B3_CHUNK);
                let hashes: Vec<Hash> = self.seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                let row = SeqSpan {
                    first_chunk: c0 as u32,
                    bytes: data[c0 * B3_CHUNK..((c1 + 1) * B3_CHUNK).min(data.len())].to_vec(),
                    chunk_proof: b3_chunks_proof(data, c0 as u64, (c1 - c0 + 1) as u64),
                    leaf_path: b3_path(&hashes, leaf),
                };
                CompOpeningPoint {
                    response_txid: [1; 32],
                    pos_root: self.pos_root,
                    seq_root: self.seq_root,
                    layer: self.eo(2),
                    entries: vec![self.eo(0), self.eo(1)],
                    weights: vec![(self.eo(3), self.range(self.offs[0], 2048)), (self.eo(4), self.range(self.offs[1], 128 * 544)), (self.eo(5), self.range(self.offs[2], 512))],
                    row,
                    members: (0..2).map(|j| SelMember { pos_path: b3_path(&self.pos_roots, 2 * k + j), k: self.leaf(2 * k + j, 1), g: self.leaf(2 * k + j, 3) }).collect(),
                }
            }
        }

        fn seeds() -> impl Iterator<Item = Hash> {
            (0..24u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
        }

        #[test]
        fn an_honest_compressed_row_point_passes_and_round_trips() {
            let kit = build(false);
            for s in seeds() {
                let op = kit.open(&s);
                assert_eq!(CompOpeningPoint::decode(&op.encode()), Some(op.clone()));
                assert!(op.encode().len() < crate::opening::MAX_BODY_LEN);
                verify_comp_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op).unwrap_or_else(|e| panic!("{e:?}"));
            }
        }

        #[test]
        fn a_row_of_misweighted_members_is_refused() {
            let kit = build(true);
            for s in seeds() {
                assert_eq!(verify_comp_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &kit.open(&s)), Err(Fail::Mismatch));
            }
        }

        #[test]
        fn members_of_another_block_are_refused() {
            let kit = build(false);
            let s = seeds().next().unwrap();
            let mut op = kit.open(&s);
            op.members.swap(0, 1);
            assert_eq!(verify_comp_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Trace));
            let mut op = kit.open(&s);
            op.members.pop();
            assert_eq!(verify_comp_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Shape));
            let short = ResponseCtx { n_pos: 1, n_prompt: 1, ..kit.resp() };
            assert_eq!(verify_comp_point(&kit.tier, &kit.ctx, &short, &s, &kit.open(&s)), Ok(()));
        }
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
