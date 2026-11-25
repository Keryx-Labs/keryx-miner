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

    use crate::engramkit::links;

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
        use crate::verify::Fail;

        use crate::engramkit::build_link as build;

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
        use crate::opening::EngramOpening;
        use crate::verify::Fail;

        use crate::engramkit::{build_resid as build, ResidFault as Fault, R_N as N};

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
