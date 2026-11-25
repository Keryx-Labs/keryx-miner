//! N-gram table lookups (Engram) of DeepSeek-V4.1: the rows a token reads, from its committed
//! token history, the token map and the table, each opened against its root.

use crate::manifest::EngramLinks;
use crate::merkle::{Hash, rt_verify_range};
use crate::opening::ChunkRange;

/// Table row of each head of `e` for a token whose mapped id and those of its predecessors are
/// `ids` (`ids[s]` the token `s` positions back, `None` when blocked: before the sequence start
/// or outside the map); a blocked shift blocks every longer one.
pub fn engram_rows(e: &EngramLinks, ids: &[Option<u32>]) -> Option<Vec<u32>> {
    let (n_gram, n_head) = (e.n_gram as usize, e.n_head as usize);
    if ids.len() != n_gram || n_gram < 2 {
        return None;
    }
    let mut toks = vec![0u64; n_gram];
    let mut blocked = false;
    for (s, t) in toks.iter_mut().enumerate() {
        blocked = blocked || ids[s].is_none();
        *t = if blocked { e.pad as u64 } else { ids[s].unwrap() as u64 };
    }
    let mut out = Vec::with_capacity(e.n_cols());
    let mut rolling = toks[0].wrapping_mul(e.mult[0]);
    for o in 1..n_gram {
        rolling ^= toks[o].wrapping_mul(e.mult[o]);
        for h in 0..n_head {
            let col = (o - 1) * n_head + h;
            let p = e.primes[col] as u64;
            if p < 2 {
                return None;
            }
            out.push((rolling % p + e.offsets[col] as u64) as u32);
        }
    }
    Some(out)
}

/// Mapped id of token `tok` from the 32 B chunk of the map holding it; `None` when the chunk is
/// not that of the committed map. Tokens past the map are blocked (`Some(None)`).
pub fn mapped_id(e: &EngramLinks, tok: u32, chunk: &ChunkRange) -> Option<Option<u32>> {
    if tok >= e.map_len {
        return Some(None);
    }
    let n_chunks = (4 * e.map_len as u64).div_ceil(32);
    let c = tok as u64 * 4 / 32;
    if chunk.first_chunk != c || chunk.chunks.len() != 32 {
        return None;
    }
    let leaf: Hash = *blake3::hash(&chunk.chunks).as_bytes();
    if !rt_verify_range(&e.map_root, n_chunks, c, &[leaf], &chunk.proof) {
        return None;
    }
    let o = (tok as usize * 4) % 32;
    Some(Some(u32::from_le_bytes(chunk.chunks[o..o + 4].try_into().unwrap())))
}

/// Values of table row `row` from the chunks covering it; `None` when they are not those of the
/// committed table.
pub fn table_row(e: &EngramLinks, row: u32, r: &ChunkRange) -> Option<Vec<f32>> {
    let d = e.head_dim as usize;
    let rb = crate::quant::row_bytes(e.table_type, d)? as u64;
    let (lo, hi) = (row as u64 * rb, (row as u64 + 1) * rb);
    let (first, last) = (lo / 32, (hi - 1) / 32);
    if last >= e.table_chunks || r.first_chunk != first || r.chunks.len() as u64 != (last - first + 1) * 32 {
        return None;
    }
    let leaves: Vec<Hash> = r.chunks.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    if !rt_verify_range(&e.table_root, e.table_chunks, first, &leaves, &r.proof) {
        return None;
    }
    let o = (lo - first * 32) as usize;
    crate::quant::dequant_row(e.table_type, &r.chunks[o..o + rb as usize], d)
}

/// Draw index of the n-gram lookup point.
pub const ENGRAM_POINT: u32 = crate::dsv41::COMP_POINT + 1;

/// Gate of a stream: the sigmoid of the signed square root of `dot` (at least 1e-6 in magnitude,
/// the sign of zero positive).
pub fn stream_gate(dot: f64) -> f64 {
    let m = dot.abs().max(1e-6).sqrt();
    let x = if dot < 0.0 { -m } else { m };
    1.0 / (1.0 + libm::exp(-x))
}

/// Verify the n-gram lookup point of a response under `seed`: the drawn chunk of one stream after
/// the lookup of the drawn module is the stream before it plus the value gated by the normed dot
/// of the stream with its key.
pub fn verify_engram_point(tier: &crate::verify::TierCtx, ctx: &crate::chain::ChainCtx, resp: &crate::verify::ResponseCtx, seed: &Hash, op: &crate::opening::EngramOpening) -> Result<(), crate::verify::Fail> {
    use crate::chain::{f32_weights, leaf_chunk, chunk_leaf, finite_f32s};
    use crate::manifest::{self, Chain};
    use crate::merkle::{B3_CHUNK, b3_fold, b3_verify};
    use crate::verify::{Fail, open_entry};
    let (nl, n_pos) = (ctx.n_layers as u64, resp.n_pos as u64);
    if ctx.n_engram == 0 || nl == 0 || n_pos == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = crate::draw::point(seed, ENGRAM_POINT);
    let module = d[1] % ctx.n_engram as u64;
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let ll = match open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?.chain {
        Chain::Layer(ll) if ll.layer as u64 == layer => ll,
        _ => return Err(Fail::Manifest),
    };
    let (e, hc) = (ll.engram.ok_or(Fail::Manifest)?, ll.hc.ok_or(Fail::Manifest)?);
    if e.module as u64 != module {
        return Err(Fail::Manifest);
    }
    let out_leaf = crate::chain::engram_leaf(tier.n_products, nl, crate::chain::n_router(ctx), ctx.n_sel as u64, module);
    if e.out_leaf as u64 != out_leaf {
        return Err(Fail::Manifest);
    }
    let src = match layer {
        0 => return Err(Fail::Unsupported),
        l => crate::chain::resid_leaf(tier.n_products, l - 1, crate::chain::Resid::Out),
    };
    let kv = open_entry(tier, &op.kv, e.kv as u64)?;
    let n = hc.n_hc as u64;
    let dim = kv.m.checked_div(n + 1).filter(|&x| x > 0 && x * (n + 1) == kv.m).ok_or(Fail::Manifest)?;
    let per = (4 * dim).checked_div(B3_CHUNK as u64).filter(|&p| p > 0 && p * B3_CHUNK as u64 == 4 * dim).ok_or(Fail::Manifest)?;
    if kv.role != manifest::ROLE_PRODUCT || op.x.len() as u64 != per || op.key.len() as u64 != per {
        return Err(Fail::Shape);
    }
    let t = d[0] % n_pos;
    let (s, jc) = (d[2] % n, (d[2] / n) % per);
    let shape = crate::chain::shape_any(tier, ctx);
    let total = 4 * n * dim;
    let h = chunk_leaf(&op.dst, total, s * per + jc)?;
    let root = b3_fold(shape.n_leaves, out_leaf, &h, &op.dst.leaf_path).ok_or(Fail::Trace)?;
    if !b3_verify(&op.pos_root, n_pos, t, &root, &op.pos_path) {
        return Err(Fail::Trace);
    }
    let dst = finite_f32s(&op.dst.chunk)?;
    let mut x = Vec::with_capacity(dim as usize);
    let mut key = Vec::with_capacity(dim as usize);
    for c in 0..per {
        x.extend(leaf_chunk(&root, &shape, src, &op.x[c as usize], total, s * per + c)?);
        key.extend(leaf_chunk(&root, &shape, 2 * e.kv as u64 + 1, &op.key[c as usize], 4 * kv.m, s * per + c)?);
    }
    let value = leaf_chunk(&root, &shape, 2 * e.kv as u64 + 1, &op.value, 4 * kv.m, n * per + jc)?;
    let g = f32_weights(tier, &op.gate, e.gate, n * dim, s * dim, dim)?;
    let rn = |v: &[f32]| {
        let mut w: Vec<f64> = v.iter().map(|&a| a as f64).collect();
        crate::attn::rms_norm(&mut w, None, ctx.eps);
        w
    };
    let (rx, rk) = (rn(&x), rn(&key));
    let scale = 1.0 / (dim as f64).sqrt();
    let dot: f64 = rx.iter().zip(&rk).zip(&g).map(|((a, b), &c)| a * b * c as f64).sum::<f64>() * scale;
    let mag: f64 = rx.iter().zip(&rk).zip(&g).map(|((a, b), &c)| (a * b * c as f64).abs()).sum::<f64>() * scale;
    let gate = stream_gate(dot);
    let dg = 0.25 * (1e-5 * mag).sqrt() + 1e-6;
    let off = (jc * B3_CHUNK as u64 / 4) as usize;
    if dst.len() != value.len() {
        return Err(Fail::Shape);
    }
    for (j, (&got, &v)) in dst.iter().zip(&value).enumerate() {
        let (xs, v) = (x[off + j] as f64, v as f64);
        let want = xs + v * gate;
        if (got as f64 - want).abs() > 2e-6 * (xs.abs() + (v * gate).abs()) + v.abs() * dg + 1e-30 {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::rt_range_proof;

    fn links(map: &[u32], table: &[u8]) -> EngramLinks {
        let map_bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).chain(std::iter::repeat(0)).take(map.len().div_ceil(8) * 32).collect();
        let ml: Vec<Hash> = map_bytes.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let tl: Vec<Hash> = table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let mut e = EngramLinks {
            module: 0, kv: 0, gate: 0, n_gram: 4, n_head: 2, head_dim: 32, table_type: crate::quant::IQ4_NL, table_root: rt_range_proof(&tl, 0, 1).0,
            table_chunks: tl.len() as u64, map_root: rt_range_proof(&ml, 0, 1).0, map_len: map.len() as u32, pad: 2, out_leaf: 0,
            mult: [0; crate::manifest::ENGRAM_MAX_GRAM], primes: [0; crate::manifest::ENGRAM_MAX_COLS], offsets: [0; crate::manifest::ENGRAM_MAX_COLS],
        };
        e.mult[..4].copy_from_slice(&[76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391]);
        e.primes[..6].copy_from_slice(&[11, 13, 17, 19, 23, 29]);
        e.offsets[..6].copy_from_slice(&[0, 11, 24, 41, 60, 83]);
        e
    }

    /// The hash as llama.cpp writes it, transcribed independently.
    fn reference(e: &EngramLinks, ids: &[i64]) -> Vec<u32> {
        let mut toks = [0u64; 4];
        let mut blocked = ids[0] < 0;
        toks[0] = if blocked { e.pad as u64 } else { ids[0] as u64 };
        for s in 1..4 {
            blocked = blocked || ids[s] < 0;
            toks[s] = if blocked { e.pad as u64 } else { ids[s] as u64 };
        }
        let mut out = vec![0u32; 6];
        let mut rolling = toks[0].wrapping_mul(e.mult[0]);
        for o in 1..4 {
            rolling ^= toks[o].wrapping_mul(e.mult[o]);
            for h in 0..2 {
                let col = (o - 1) * 2 + h;
                out[col] = (rolling % e.primes[col] as u64 + e.offsets[col] as u64) as u32;
            }
        }
        out
    }

    #[test]
    fn rows_follow_the_hash_and_blocking() {
        let e = links(&[0], &[0; 32]);
        for ids in [[5i64, 9, 3, 7], [5, -1, 3, 7], [-1, 9, 3, 7], [5, 9, 3, -1], [0, 0, 0, 0]] {
            let opt: Vec<Option<u32>> = ids.iter().map(|&x| (x >= 0).then_some(x as u32)).collect();
            assert_eq!(engram_rows(&e, &opt).unwrap(), reference(&e, &ids), "{ids:?}");
        }
        let a = engram_rows(&e, &[Some(5), None, Some(3), Some(7)]).unwrap();
        let b = engram_rows(&e, &[Some(5), None, Some(4), Some(1)]).unwrap();
        assert_eq!(a, b, "a blocked shift hides the tokens behind it");
        assert!(engram_rows(&e, &[Some(5), Some(9)]).is_none());
    }

    #[test]
    fn the_map_and_the_table_open_against_their_roots() {
        let map: Vec<u32> = (0..40).map(|t| (t * 7) % 23).collect();
        let row = vec![0x3cu8, 0x00].into_iter().chain((0..16).map(|i| i as u8 * 17)).collect::<Vec<u8>>();
        let table: Vec<u8> = row.iter().copied().cycle().take(18 * 16).collect();
        let e = links(&map, &table);
        let map_bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ml: Vec<Hash> = map_bytes.chunks(32).map(|c| { let mut p = c.to_vec(); p.resize(32, 0); *blake3::hash(&p).as_bytes() }).collect();
        let chunk = |t: u32| {
            let c = t as usize * 4 / 32;
            let mut b = map_bytes[c * 32..((c + 1) * 32).min(map_bytes.len())].to_vec();
            b.resize(32, 0);
            ChunkRange { first_chunk: c as u64, chunks: b, proof: rt_range_proof(&ml, c as u64, 1).1 }
        };
        assert_eq!(mapped_id(&e, 13, &chunk(13)), Some(Some(map[13])));
        assert_eq!(mapped_id(&e, 13, &chunk(21)), None);
        assert_eq!(mapped_id(&e, 99, &chunk(13)), Some(None));
        let tl: Vec<Hash> = table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let range = |r: u64| {
            let (lo, hi) = (r * 18, (r + 1) * 18);
            let (f, l) = (lo / 32, (hi - 1) / 32);
            ChunkRange { first_chunk: f, chunks: table[(f * 32) as usize..((l + 1) * 32) as usize].to_vec(), proof: rt_range_proof(&tl, f, l - f + 1).1 }
        };
        let want = crate::quant::dequant_row(crate::quant::IQ4_NL, &row, 32).unwrap();
        assert_eq!(table_row(&e, 5, &range(5)), Some(want));
        assert_eq!(table_row(&e, 5, &range(6)), None);
        let mut bad = range(5);
        bad.chunks[3] ^= 1;
        assert_eq!(table_row(&e, 5, &bad), None);
    }

    mod link {
        use super::*;
        use crate::chain::{ChainCtx, verify_chain_point};
        use crate::manifest::{self, Chain, Entry, LayerLinks, ProductLink};
        use crate::merkle::{B3_CHUNK, b3_chunk_cv, b3_path, b3_root};
        use crate::opening::{EntryOpening, LinkOpening, Point, PrevToken};
        use crate::testkit::{in_layout, layout_at, q8_1_row};
        use crate::verify::{Fail, ResponseCtx, TierCtx};

        const N: usize = 6;
        const D: usize = 256;
        const TRB: usize = D / 32 * 18;
        const COLS: usize = 6;
        const K: usize = COLS * D;
        const M: usize = 8;
        const ROWS: usize = 208;
        const NONE: u32 = manifest::NONE;

        fn bytes(seed: u64, n: usize) -> Vec<u8> {
            (0..n).map(|i| blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0]).collect()
        }

        struct Kit {
            weights: Vec<u8>,
            chunk_leaves: Vec<Hash>,
            entries: Vec<Vec<u8>>,
            entry_leaves: Vec<Hash>,
            tier: TierCtx,
            ctx: ChainCtx,
            e: EngramLinks,
            map: Vec<u32>,
            table: Vec<u8>,
            tokens: Vec<u32>,
            pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
            pos_roots: Vec<Hash>,
        }

        fn ids_at(e: &EngramLinks, map: &[u32], tokens: &[u32], t: usize, unigram: bool) -> Vec<u32> {
            let ids: Vec<Option<u32>> = (0..4).map(|s| if s > t || (unigram && s > 0) { None } else { map.get(tokens[t - s] as usize).copied() }).collect();
            engram_rows(e, &ids).unwrap()
        }

        fn build(unigram: bool) -> Kit {
            let map: Vec<u32> = (0..16).map(|t| (t * 5 + 3) % 13).collect();
            let table: Vec<u8> = (0..ROWS * D / 32).flat_map(|r| {
                let mut b = crate::f16::from_f32(0.01 + 0.0005 * (r % 7) as f32).to_le_bytes().to_vec();
                b.extend(bytes(1000 + r as u64, 16));
                b
            }).collect();
            let mut e = links(&map, &table);
            e.head_dim = D as u32;
            e.primes[..COLS].copy_from_slice(&[31, 37, 29, 41, 23, 43]);
            e.offsets[..COLS].copy_from_slice(&[0, 31, 68, 97, 138, 161]);
            let rb = K / 32 * 34;
            let weights: Vec<u8> = (0..M).flat_map(|r| (0..K / 32).flat_map(move |b| {
                let mut blk = crate::f16::from_f32(0.0004).to_le_bytes().to_vec();
                blk.extend(bytes(2000 + (r * 64 + b) as u64, 32));
                blk
            })).collect();
            assert_eq!(weights.len(), M * rb);
            let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let mut product = Entry {
                name: "blk.0.engram_kv.weight".into(), wtype: crate::quant::Q8_0, k: K as u64, m: M as u64, n_exp: 1, chunk_base: 0, n_bytes: weights.len() as u64,
                row_bytes: rb as u32, role: manifest::ROLE_PRODUCT, glu_op: 0, glu_form: manifest::FORM_NONE, partner: NONE, bias: NONE,
                chain: Chain::Product(ProductLink { layer: 0, link: manifest::LINK_ENGRAM, norm: NONE, src: NONE }),
            };
            let mut p0 = crate::testkit::serialize(&product);
            p0.extend_from_slice(&manifest::chain_bytes(&product.chain));
            product.chain = Chain::Layer(LayerLinks {
                layer: 0, proj: 0, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: None, recur: None, hc: None, dsa: None, comp: None, engram: Some(e),
            });
            product.name = "blk.0".into();
            product.role = manifest::ROLE_LAYER;
            let mut l0 = crate::testkit::serialize(&product);
            l0.extend_from_slice(&manifest::chain_bytes(&product.chain));
            let entries = vec![p0, l0];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx {
                r_t: crate::merkle::rt_range_proof(&chunk_leaves, 0, 1).0, n_chunks: chunk_leaves.len() as u64,
                m_t: crate::merkle::rt_range_proof(&entry_leaves, 0, 1).0, n_entries: 2, n_products: 1,
            };
            let ctx = ChainCtx { n_sel: 0, n_comp: 0, n_engram: 0, n_layers: 1, layer_base: 1, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 };
            let tokens: Vec<u32> = (0..N as u32).map(|i| (i * 7 + 2) % 16).collect();
            let n_leaves = crate::chain::n_pos_leaves(1, 1, 0) as usize;
            let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..N)
                .map(|t| {
                    let rows = ids_at(&e, &map, &tokens, t, unigram);
                    let v: Vec<f64> = rows.iter().flat_map(|&r| crate::quant::dequant_row(crate::quant::IQ4_NL, &table[r as usize * TRB..(r as usize + 1) * TRB], D).unwrap()).map(|x| x as f64).collect();
                    let layout = layout_at(t as u64, N as u32);
                    let x = in_layout(&q8_1_row(&v), layout.1);
                    let y: Vec<f32> = (0..M).map(|r| crate::dot::canonical_dot(crate::quant::Q8_0, &weights[r * rb..(r + 1) * rb], K, &x, layout.1).unwrap().y_hat as f32).collect();
                    let mut leaves = vec![Vec::new(); n_leaves];
                    leaves[0] = x;
                    leaves[1] = y.iter().flat_map(|a| a.to_le_bytes()).collect();
                    leaves[n_leaves - 1] = crate::chain::meta(t as u32, layout.0, tokens[t]).to_vec();
                    let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                    (leaves, hashes)
                })
                .collect();
            let pos_roots: Vec<Hash> = pos.iter().map(|(_, h)| b3_root(h)).collect();
            Kit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, e, map, table, tokens, pos, pos_roots }
        }

        impl Kit {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: crate::merkle::rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn map_chunk(&self, tok: u32) -> ChunkRange {
                let mb: Vec<u8> = self.map.iter().flat_map(|v| v.to_le_bytes()).chain(std::iter::repeat(0)).take(self.map.len().div_ceil(8) * 32).collect();
                let ml: Vec<Hash> = mb.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
                let c = tok as usize * 4 / 32;
                ChunkRange { first_chunk: c as u64, chunks: mb[c * 32..c * 32 + 32].to_vec(), proof: crate::merkle::rt_range_proof(&ml, c as u64, 1).1 }
            }

            fn row_range(&self, r: u32) -> ChunkRange {
                let tl: Vec<Hash> = self.table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
                let (lo, hi) = (r as u64 * TRB as u64, (r as u64 + 1) * TRB as u64);
                let (f, l) = (lo / 32, (hi - 1) / 32);
                ChunkRange { first_chunk: f, chunks: self.table[(f * 32) as usize..((l + 1) * 32) as usize].to_vec(), proof: crate::merkle::rt_range_proof(&tl, f, l - f + 1).1 }
            }

            fn open(&self, seed: &Hash, lie: bool) -> (Point, u32, LinkOpening) {
                let d = crate::draw::point(seed, 0);
                let t = (d[0] % N as u64) as usize;
                let row = (d[2] % M as u64) as usize;
                let rb = (K / 32 * 34) as u64;
                let (leaves, hashes) = &self.pos[t];
                let y = &leaves[1];
                let yc = (row * 4 / B3_CHUNK) as u64;
                let cvs: Vec<Hash> = (0..y.len().div_ceil(B3_CHUNK)).map(|k| b3_chunk_cv(&y[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(y.len())], k as u64)).collect();
                let off = row as u64 * rb;
                let (first, last) = (off / 32, (off + rb - 1) / 32);
                let p = Point {
                    n_ubatch: layout_at(t as u64, N as u32).0,
                    pos_path: b3_path(&self.pos_roots, t),
                    meta_path: b3_path(hashes, hashes.len() - 1),
                    entry: self.eo(0),
                    partner: None,
                    bias: None,
                    rows: vec![ChunkRange { first_chunk: first, chunks: self.weights[(first * 32) as usize..((last + 1) * 32) as usize].to_vec(), proof: crate::merkle::rt_range_proof(&self.chunk_leaves, first, last - first + 1).1 }],
                    x: leaves[0].clone(),
                    x_path: b3_path(hashes, 0),
                    y_total_len: y.len() as u32,
                    y_chunk_index: yc as u32,
                    y_chunk: y[(yc as usize) * B3_CHUNK..((yc as usize + 1) * B3_CHUNK).min(y.len())].to_vec(),
                    y_chunk_path: if cvs.len() > 1 { b3_path(&cvs, yc as usize) } else { Vec::new() },
                    y_path: b3_path(hashes, 1),
                };
                let block = crate::chain::link_block(seed, 0, &p.x, layout_at(t as u64, N as u32).1, K as u64).unwrap();
                let col = (block * 32 / D as u64) as usize;
                let rows = ids_at(&self.e, &self.map, &self.tokens, t, false);
                let shifts = t.min(3);
                let prev: Vec<PrevToken> = (1..=shifts)
                    .map(|s| {
                        let q = t - s;
                        let tok = if lie && s == 1 { (self.tokens[q] + 1) % 16 } else { self.tokens[q] };
                        PrevToken { tok, meta_path: b3_path(&self.pos[q].1, self.pos[q].1.len() - 1), pos_path: b3_path(&self.pos_roots, q) }
                    })
                    .collect();
                let map = (0..=shifts).map(|s| self.map_chunk(if lie && s == 1 { prev[0].tok } else { self.tokens[t - s] })).collect();
                (p, self.tokens[t], LinkOpening::Engram { layer: self.eo(1), prev, map, row: self.row_range(rows[col]) })
            }

            fn check(&self, seed: &Hash, lie: bool) -> (usize, Result<(), Fail>) {
                let (p, tok, link) = self.open(seed, lie);
                let resp = ResponseCtx { trace_root: b3_root(&self.pos_roots), n_prompt: N as u32, n_pos: N as u32 };
                let t = (crate::draw::point(seed, 0)[0] % N as u64) as usize;
                (t, verify_chain_point(&self.tier, &self.ctx, &resp, seed, 0, &p, tok, &link, None).map(|_| ()))
            }
        }

        fn seeds() -> impl Iterator<Item = Hash> {
            (0..48u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
        }

        #[test]
        fn an_honest_lookup_input_passes_and_round_trips() {
            let kit = build(false);
            let mut ts = std::collections::BTreeSet::new();
            for s in seeds() {
                let (_, _, link) = kit.open(&s, false);
                let mut w = crate::opening::Opening { response_txid: [0; 32], points: vec![kit.open(&s, false).0], prefix: Some(Default::default()), links: vec![(7, link.clone())], experts: vec![None], resid: None, roots: None };
                w.points.push(kit.open(&s, false).0);
                w.links.push((7, link));
                w.experts.push(None);
                assert_eq!(crate::opening::Opening::decode(&w.encode()), Some(w));
                let (t, r) = kit.check(&s, false);
                ts.insert(t);
                r.unwrap_or_else(|e| panic!("t {t}: {e:?}"));
            }
            assert!(ts.contains(&0) && ts.iter().any(|&t| t >= 3), "{ts:?}");
        }

        #[test]
        fn an_input_of_the_token_alone_is_refused() {
            let kit = build(true);
            for s in seeds() {
                let (t, r) = kit.check(&s, false);
                if t == 0 {
                    assert!(r.is_ok());
                } else {
                    assert_eq!(r, Err(Fail::Mismatch), "t {t}");
                }
            }
        }

        #[test]
        fn another_earlier_token_is_refused() {
            let kit = build(false);
            for s in seeds() {
                let (t, r) = kit.check(&s, true);
                if t > 0 {
                    assert_eq!(r, Err(Fail::Trace), "t {t}");
                }
            }
        }
    }

    mod resid {
        use super::*;
        use crate::chain::ChainCtx;
        use crate::manifest::{self, Chain, Entry, HcLinks, LayerLinks};
        use crate::merkle::{B3_CHUNK, b3_chunk_cv, b3_path, b3_root};
        use crate::opening::{EngramOpening, EntryOpening, LeafChunk};
        use crate::verify::{Fail, ResponseCtx, TierCtx};

        const N: usize = 5;
        const NHC: usize = 2;
        const D: usize = 256;
        const NONE: u32 = manifest::NONE;

        #[derive(Clone, Copy, PartialEq)]
        enum Fault {
            None,
            /// The gate without the square root.
            PlainGate,
            /// The value added without its gate.
            Ungated,
            /// Each stream gated by the other stream's dot.
            SwappedGates,
        }

        fn vals(seed: u64, n: usize, a: f32) -> Vec<f32> {
            (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
        }

        struct Kit {
            weights: Vec<u8>,
            chunk_leaves: Vec<Hash>,
            entries: Vec<Vec<u8>>,
            entry_leaves: Vec<Hash>,
            tier: TierCtx,
            ctx: ChainCtx,
            pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
            pos_roots: Vec<Hash>,
        }

        fn entry(name: &str, role: u8, wtype: u32, k: usize, m: usize, n: usize, chain: Chain) -> Vec<u8> {
            let e = Entry {
                name: name.into(), wtype, k: k as u64, m: m as u64, n_exp: 1, chunk_base: 0, n_bytes: n as u64, row_bytes: n as u32, role, glu_op: 0,
                glu_form: manifest::FORM_NONE, partner: NONE, bias: NONE, chain,
            };
            let mut b = crate::testkit::serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            b
        }

        fn build(fault: Fault) -> Kit {
            let gates: Vec<f32> = vals(5, NHC * D, 1.5);
            let weights: Vec<u8> = gates.iter().flat_map(|a| a.to_le_bytes()).collect();
            let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let hc = HcLinks { attn_fn: 0, ffn_fn: 0, attn_scale: [0; 3], ffn_scale: [0; 3], attn_base: NONE, ffn_base: NONE, n_hc: NHC as u32, sinkhorn_iters: 20, eps: 0 , lag: 0, prev_fn: NONE, prev_scale: [0; 3], prev_base: NONE};
            let mut e = links(&[0], &[0; 32]);
            e.module = 0;
            e.kv = 0;
            e.gate = 3;
            e.out_leaf = crate::chain::engram_leaf(1, 2, 0, 0, 0) as u32;
            let layer = |l: u32, eg: Option<EngramLinks>| {
                Chain::Layer(LayerLinks { layer: l, proj: 0, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: None, recur: None, hc: Some(hc), dsa: None, comp: None, engram: eg })
            };
            let entries = vec![
                entry("blk.1.engram_kv.weight", manifest::ROLE_PRODUCT, crate::quant::Q8_0, 512, (NHC + 1) * D, 0, Chain::None),
                entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, layer(0, None)),
                entry("blk.1", manifest::ROLE_LAYER, 0, 0, 0, 0, layer(1, Some(e))),
                entry("blk.1.engram_gate.weight", manifest::ROLE_NORM, 0, NHC * D, 1, weights.len(), Chain::None),
            ];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx { r_t: crate::merkle::rt_range_proof(&chunk_leaves, 0, 1).0, n_chunks: chunk_leaves.len() as u64, m_t: crate::merkle::rt_range_proof(&entry_leaves, 0, 1).0, n_entries: 4, n_products: 1 };
            let ctx = ChainCtx { n_sel: 0, n_comp: 0, n_engram: 1, n_layers: 2, layer_base: 1, embd_entry: 0, eps: 1e-20, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 };
            let n_leaves = crate::chain::n_pos_leaves(1, 2, 0) as usize + 1;
            let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
            let rn = |v: &[f32]| {
                let mut w: Vec<f64> = v.iter().map(|&a| a as f64).collect();
                crate::attn::rms_norm(&mut w, None, 1e-20);
                w
            };
            let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..N)
                .map(|t| {
                    let x = vals(100 + t as u64, NHC * D, 2.0);
                    let kv = vals(200 + t as u64, (NHC + 1) * D, 1.0);
                    let value = &kv[NHC * D..];
                    let dots: Vec<f64> = (0..NHC)
                        .map(|s| {
                            let (rx, rk) = (rn(&x[s * D..(s + 1) * D]), rn(&kv[s * D..(s + 1) * D]));
                            rx.iter().zip(&rk).zip(&gates[s * D..(s + 1) * D]).map(|((a, b), &c)| a * b * c as f64).sum::<f64>() / (D as f64).sqrt()
                        })
                        .collect();
                    let out: Vec<f32> = (0..NHC)
                        .flat_map(|s| {
                            let dot = if fault == Fault::SwappedGates { dots[NHC - 1 - s] } else { dots[s] };
                            let g = match fault {
                                Fault::PlainGate => 1.0 / (1.0 + libm::exp(-dot)),
                                Fault::Ungated => 1.0,
                                _ => stream_gate(dot),
                            };
                            let x = &x;
                            (0..D).map(move |j| (x[s * D + j] as f64 + value[j] as f64 * g) as f32).collect::<Vec<_>>()
                        })
                        .collect();
                    let mut leaves = vec![Vec::new(); n_leaves];
                    leaves[1] = f(&kv);
                    leaves[crate::chain::resid_leaf(1, 0, crate::chain::Resid::Out) as usize] = f(&x);
                    leaves[e.out_leaf as usize] = f(&out);
                    leaves[n_leaves - 1] = crate::chain::meta(t as u32, 1, 7).to_vec();
                    let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                    (leaves, hashes)
                })
                .collect();
            let pos_roots: Vec<Hash> = pos.iter().map(|(_, h)| b3_root(h)).collect();
            Kit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos, pos_roots }
        }

        impl Kit {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: crate::merkle::rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn chunk(&self, t: usize, i: usize, c: usize) -> LeafChunk {
                let b = &self.pos[t].0[i];
                let n = b.len().div_ceil(B3_CHUNK);
                let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
                LeafChunk { total_len: b.len() as u32, index: c as u32, chunk: b[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(b.len())].to_vec(), chunk_path: b3_path(&cvs, c), leaf_path: b3_path(&self.pos[t].1, i) }
            }

            fn resp(&self) -> ResponseCtx {
                ResponseCtx { trace_root: crate::seq::trace_root(&b3_root(&self.pos_roots), &[5; 32]), n_prompt: N as u32, n_pos: N as u32 }
            }

            fn open(&self, seed: &Hash) -> EngramOpening {
                let d = crate::draw::point(seed, ENGRAM_POINT);
                let t = (d[0] % N as u64) as usize;
                let (s, jc) = ((d[2] % NHC as u64) as usize, 0usize);
                let out = crate::chain::engram_leaf(1, 2, 0, 0, 0) as usize;
                let src = crate::chain::resid_leaf(1, 0, crate::chain::Resid::Out) as usize;
                EngramOpening {
                    response_txid: [1; 32],
                    pos_root: b3_root(&self.pos_roots),
                    seq_root: [5; 32],
                    layer: self.eo(2),
                    kv: self.eo(0),
                    gate: (self.eo(3), ChunkRange { first_chunk: (s * D * 4 / 32) as u64, chunks: self.weights[s * D * 4..(s + 1) * D * 4].to_vec(), proof: crate::merkle::rt_range_proof(&self.chunk_leaves, (s * D * 4 / 32) as u64, (D * 4 / 32) as u64).1 }),
                    pos_path: b3_path(&self.pos_roots, t),
                    dst: self.chunk(t, out, s + jc),
                    x: vec![self.chunk(t, src, s)],
                    key: vec![self.chunk(t, 1, s)],
                    value: self.chunk(t, 1, NHC + jc),
                }
            }
        }

        fn seeds() -> impl Iterator<Item = Hash> {
            (0..32u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
        }

        #[test]
        fn an_honest_lookup_residual_passes_and_round_trips() {
            let kit = build(Fault::None);
            for s in seeds() {
                let op = kit.open(&s);
                assert_eq!(EngramOpening::decode(&op.encode()), Some(op.clone()));
                verify_engram_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op).unwrap_or_else(|e| panic!("{e:?}"));
            }
        }

        #[test]
        fn a_misgated_residual_is_refused() {
            for fault in [Fault::PlainGate, Fault::Ungated, Fault::SwappedGates] {
                let kit = build(fault);
                for s in seeds() {
                    assert_eq!(verify_engram_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &kit.open(&s)), Err(Fail::Mismatch));
                }
            }
        }

        #[test]
        fn a_stream_of_another_position_is_refused() {
            let kit = build(Fault::None);
            let s = seeds().next().unwrap();
            let mut op = kit.open(&s);
            let t = (crate::draw::point(&s, ENGRAM_POINT)[0] % N as u64) as usize;
            op.x = vec![kit.chunk((t + 1) % N, crate::chain::resid_leaf(1, 0, crate::chain::Resid::Out) as usize, 0)];
            assert_eq!(verify_engram_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Trace));
        }
    }

    #[test]
    fn the_envelope_round_trips() {
        use crate::manifest::{Chain, LayerLinks, NONE, chain_bytes};
        let e = links(&[1, 2, 3], &[0; 64]);
        let inner = LayerLinks { layer: 1, proj: 4, down: 9, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: None, recur: None, hc: None, dsa: None, comp: None, engram: None };
        let c = Chain::Layer(LayerLinks { engram: Some(e), ..inner });
        let mut ent = vec![1u8, 0, b'x'];
        ent.extend_from_slice(&[0u8; 44]);
        ent.extend_from_slice(&[crate::manifest::ROLE_LAYER, 0, crate::manifest::FORM_NONE]);
        ent.extend_from_slice(&NONE.to_le_bytes());
        ent.extend_from_slice(&NONE.to_le_bytes());
        ent.extend_from_slice(&chain_bytes(&c));
        assert_eq!(crate::manifest::Entry::parse(&ent).map(|(x, n)| (x.chain, n)), Some((c, ent.len())));
        let mut cut = ent.clone();
        cut.truncate(ent.len() - 30);
        assert!(crate::manifest::Entry::parse(&cut).is_none());
    }
}
