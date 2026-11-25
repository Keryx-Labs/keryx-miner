//! Compressed attention rows of DeepSeek-V4.1: a block of `ratio` consecutive tokens pooled into
//! one key/value row, read by the attention of the layers that share it.

use crate::attn::{Yarn, rms_norm, rope_norm_yarn};
use crate::chain::{ChainCtx, f32_weights, f32s, seq_layout};
use crate::manifest::{self, Chain};
use crate::merkle::{Hash, b3_fold, b3_verify};
use crate::opening::{CompAttnOpening, CompOpeningPoint, CompSelOpening, LeafBytes};
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

/// The partial of a head's sink: a logit `s` (on the scale of the scaled scores) with no value.
pub fn sink_partial(d: usize, s: f64) -> crate::attn::Partial {
    crate::attn::Partial { o: vec![0.0; d], m: s, l: 1.0 }
}

/// `p` with the sink `s` of its head in the softmax denominator.
pub fn with_sink(p: &crate::attn::Partial, s: f64) -> Option<crate::attn::Partial> {
    crate::attn_blocks::combine(&[p.clone(), sink_partial(p.o.len(), s)])
}

/// A head output with the rotary part of its last `n_rot` values undone at the query position.
pub fn derope(o: &mut [f64], pos: u64, n_rot: usize, base: f64, yarn: &Yarn) {
    let off = o.len() - n_rot;
    rope_norm_yarn(o, off, pos, n_rot, base, yarn, true);
}

/// Bytes a selected compressed row costs a leaf phase: the row and the proof of its span.
const COMP_ROW_COST: usize = 2560;

/// Drawn attention point of a compressed attention layer.
struct CompSite {
    a: manifest::AttnLinks,
    c: manifest::CompLinks,
    t: u64,
    qh: u64,
    w0: u64,
    n_w: u64,
    sel: Vec<u64>,
    tree: crate::attn_blocks::BlockTree,
    d: [u64; 4],
}

impl CompSite {
    /// Rotary base and scaling of the layer.
    fn rope(&self) -> (f64, Yarn) {
        if self.c.ratio > 0 { (f32::from_bits(self.c.rope_base) as f64, self.c.yarn()) } else { (f32::from_bits(self.a.rope_base) as f64, Yarn::NONE) }
    }
}

fn comp_site(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &CompAttnOpening) -> Result<CompSite, Fail> {
    let (nl, n_pos, n_attn) = (ctx.n_layers as u64, resp.n_pos as u64, ctx.n_attn as u64);
    if n_pos == 0 || nl == 0 || n_attn == 0 || ctx.seq_heads == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = crate::draw::point(seed, crate::chain::ATTN_POINT);
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let (a, c) = match open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?.chain {
        Chain::Layer(ll) if ll.layer as u64 == layer => (ll.attn.ok_or(Fail::Manifest)?, ll.comp.ok_or(Fail::Manifest)?),
        _ => return Err(Fail::Manifest),
    };
    if a.slot as u64 != d[1] % n_attn {
        return Err(Fail::Manifest);
    }
    let hd = a.head_dim as u64;
    if a.n_head == 0 || a.n_head_kv != 1 || a.v != manifest::NONE || a.mla_rank != 0 || hd == 0 || !hd.is_multiple_of(32) || a.swa_window == 0 || a.n_rot as u64 > hd {
        return Err(Fail::Manifest);
    }
    let t = crate::chain::resid_position(ctx, resp, layer, d[0])?;
    let qh = d[2] % a.n_head as u64;
    let w0 = (t + 1).saturating_sub(a.swa_window as u64);
    let n_w = t + 1 - w0;
    let sel = match (c.ratio, &op.sel) {
        (0, None) => Vec::new(),
        (0, Some(_)) | (_, None) => return Err(Fail::Shape),
        (ratio, Some(s)) => {
            let shape = crate::chain::shape_any(tier, ctx);
            let leaf = crate::chain::sel_leaf(tier.n_products, nl, crate::chain::n_router(ctx), c.sel as u64);
            let root = b3_fold(shape.n_leaves, leaf, blake3::hash(&s.bytes).as_bytes(), &s.leaf_path).ok_or(Fail::Trace)?;
            if c.sel >= ctx.n_sel || !b3_verify(&op.pos_root, n_pos, t, &root, &s.pos_path) {
                return Err(Fail::Trace);
            }
            let n_top = c.n_top as usize;
            if n_top == 0 || s.bytes.len() != 4 + 4 * n_top {
                return Err(Fail::Mismatch);
            }
            let slots: Vec<u32> = s.bytes[4..].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            let n_vis = (t + 1) / ratio as u64;
            let m = (n_vis as usize).min(n_top);
            let ids: Vec<u64> = slots[..m].iter().map(|&x| x as u64).collect();
            if slots[m..].iter().any(|&x| x != u32::MAX) || ids.windows(2).any(|w| w[0] >= w[1]) || ids.last().is_some_and(|&x| x >= n_vis) {
                return Err(Fail::Mismatch);
            }
            ids
        }
    };
    let (cl, b) = crate::attn_blocks::block_params(COMP_ROW_COST, crate::attn_blocks::partial_len(hd as usize)).ok_or(Fail::Manifest)?;
    let tree = crate::attn_blocks::BlockTree::new(0, n_w + sel.len() as u64, cl, b).ok_or(Fail::Manifest)?;
    Ok(CompSite { a, c, t, qh, w0, n_w, sel, tree, d })
}

/// Verify phase `path.len()` of the attention point of a compressed attention layer; `path`
/// holds the child drawn after each earlier phase and `parent_root` the partials root the
/// previous phase committed.
pub fn verify_comp_attn_phase(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    path: &[u64],
    parent_root: Option<&Hash>,
    op: &CompAttnOpening,
) -> Result<crate::chain::AttnStep, Fail> {
    use crate::attn_blocks::{check_leaf, check_node, child_weights, decode_partials, open_partial, partials_root};
    let s = comp_site(tier, ctx, resp, seed, op)?;
    let (a, c) = (&s.a, &s.c);
    let (hd, n_pos) = (a.head_dim as usize, resp.n_pos as u64);
    let depth = s.tree.depth() as usize;
    if op.path != path || path.len() > depth {
        return Err(Fail::Shape);
    }
    let parent = match (path.split_last(), parent_root) {
        (None, None) if op.parent.is_empty() && op.parent_path.is_empty() => None,
        (Some((&i, up)), Some(root)) => {
            let n = s.tree.children(up).ok_or(Fail::Shape)?.len() as u64;
            Some(open_partial(root, n, i, &op.parent, &op.parent_path, hd).ok_or(Fail::Trace)?)
        }
        _ => return Err(Fail::Shape),
    };
    let (root_phase, last) = (path.is_empty(), path.len() == depth);
    let mut keep = CompAttnOpening {
        response_txid: op.response_txid,
        pos_root: op.pos_root,
        seq_root: op.seq_root,
        layer: op.layer.clone(),
        sel: op.sel.clone(),
        path: op.path.clone(),
        parent: op.parent.clone(),
        parent_path: op.parent_path.clone(),
        ..Default::default()
    };
    if root_phase || last {
        (keep.q_entry, keep.q, keep.pos_path) = (op.q_entry.clone(), op.q.clone(), op.pos_path.clone());
    }
    if root_phase {
        (keep.o_entry, keep.out, keep.sinks, keep.lse) = (op.o_entry.clone(), op.out.clone(), op.sinks.clone(), op.lse.clone());
    }
    if last {
        (keep.window, keep.comp, keep.anchor_pos_path, keep.anchor_k, keep.norms) =
            (op.window.clone(), op.comp.clone(), op.anchor_pos_path.clone(), op.anchor_k.clone(), op.norms.clone());
    } else {
        keep.partials = op.partials.clone();
    }
    if *op != keep {
        return Err(Fail::Shape);
    }
    let shape = crate::chain::shape_any(tier, ctx);
    let (base, yarn) = s.rope();
    let n_rot = a.n_rot as usize;
    let layout = crate::chain::seq_layout(ctx);
    // the root: children with the head's sink, de-rotated, against the output product's input
    let root_ok = |parts: &[crate::attn::Partial]| -> Result<bool, Fail> {
        let oe = open_entry(tier, &op.o_entry, a.out as u64)?;
        let gs = (oe.k as usize).checked_div(hd).filter(|&g| g > 0 && g * hd == oe.k as usize).ok_or(Fail::Manifest)?;
        if oe.role != manifest::ROLE_HEADS || oe.n_exp as usize * gs != a.n_head as usize {
            return Err(Fail::Manifest);
        }
        let lay = crate::verify::layout_for(crate::verify::expected_ubatch(s.t, resp.n_prompt), oe.wtype);
        let root_t = b3_fold(shape.n_leaves, 2 * a.out as u64, blake3::hash(&op.out.bytes).as_bytes(), &op.out.leaf_path).ok_or(Fail::Trace)?;
        if !b3_verify(&op.pos_root, n_pos, s.t, &root_t, &op.pos_path) {
            return Err(Fail::Trace);
        }
        let x_one = crate::verify::x_len(oe.k, lay) as usize;
        if op.out.bytes.len() != oe.n_exp as usize * x_one {
            return Err(Fail::Shape);
        }
        let g = s.qh as usize / gs;
        let xg = &op.out.bytes[g * x_one..(g + 1) * x_one];
        let sink = f32_weights(tier, &op.sinks, c.sinks, a.n_head as u64, s.qh, 1)?[0] as f64;
        let nh = a.n_head as u64;
        let leaf = layout.lse(a.slot as u64, 0).ok_or(Fail::Shape)?;
        let lo = (s.t * nh + s.qh) * 4;
        let mb = crate::seq::open_span(&op.seq_root, &layout, leaf, n_pos * nh * 4, lo, lo + 4, &op.lse).ok_or(Fail::Trace)?;
        let mass = f32::from_le_bytes(mb.try_into().map_err(|_| Fail::Shape)?) as f64;
        let mut all = parts.to_vec();
        all.push(sink_partial(hd, sink));
        let Some(comb) = crate::attn_blocks::combine(&all) else { return Ok(false) };
        if !mass.is_finite() || (crate::attn_blocks::lse(&comb) - mass).abs() > crate::attn_blocks::root_lse_tol(comb.m) {
            return Ok(false);
        }
        let mut o = comb.o;
        derope(&mut o, s.t, n_rot, base, &yarn);
        Ok(crate::attn_blocks::root_within(xg, lay, oe.k as usize, (s.qh as usize % gs) * hd / 32, &o))
    };
    let children = if last {
        None
    } else {
        let n = s.tree.children(path).ok_or(Fail::Shape)?.len();
        Some(decode_partials(&op.partials, hd).filter(|p| p.len() == n).ok_or(Fail::Shape)?)
    };
    match (&parent, &children) {
        (Some(p), Some(ch)) => check_node(p, ch).then_some(()).ok_or(Fail::Mismatch)?,
        (None, Some(ch)) => root_ok(ch)?.then_some(()).ok_or(Fail::Mismatch)?,
        _ => {
            // a leaf: the query, the keys of the node, an anchor of its window rows
            let qe = open_entry(tier, &op.q_entry, a.q as u64)?;
            if qe.m != a.n_head as u64 * hd as u64 {
                return Err(Fail::Manifest);
            }
            let q_leaf = 2 * a.q as u64 + 1;
            let root_t = crate::chain::anchor_root(&op.pos_root, n_pos, s.t, &shape, q_leaf, &op.q, 4 * qe.m, &op.pos_path)?;
            let mut q: Vec<f64> = crate::chain::head_from_chunks(&root_t, &shape, q_leaf, &op.q, 4 * qe.m, s.qh as usize * hd, hd)?.iter().map(|&x| x as f64).collect();
            rope_tail(&mut q, s.t, n_rot, base, &yarn);
            let (v0, v1) = s.tree.node(path).ok_or(Fail::Shape)?;
            let rb = (hd / 32 * 34) as u64;
            let (ws, we) = (v0.min(s.n_w), v1.min(s.n_w));
            let mut keys: Vec<Vec<f32>> = Vec::new();
            if ws < we {
                let leaf = layout.kcache(a.slot as u64, 0).ok_or(Fail::Shape)?;
                let bytes = crate::seq::open_span(&op.seq_root, &layout, leaf, n_pos * rb, (s.w0 + ws) * rb, (s.w0 + we) * rb, &op.window).ok_or(Fail::Trace)?;
                let rows: Vec<&[u8]> = bytes.chunks_exact(rb as usize).collect();
                for r in &rows {
                    keys.push(crate::quant::dequant_row(crate::quant::Q8_0, r, hd).ok_or(Fail::Shape)?);
                }
                let ps = s.w0 + ws + s.d[3] % (we - ws);
                let k_leaf = 2 * a.k as u64 + 1;
                let ar = crate::chain::anchor_root(&op.pos_root, n_pos, ps, &shape, k_leaf, &op.anchor_k, 4 * hd as u64, &op.anchor_pos_path)?;
                let y = crate::chain::head_from_chunks(&ar, &shape, k_leaf, &op.anchor_k, 4 * hd as u64, 0, hd)?;
                let [nw] = <[_; 1]>::try_from(op.norms.clone()).map_err(|_| Fail::Shape)?;
                let norm = f32_weights(tier, &nw, a.k_norm, hd as u64, 0, hd as u64)?;
                let mut want: Vec<f64> = y.iter().map(|&x| x as f64).collect();
                rms_norm(&mut want, Some(&norm), f32::from_bits(c.eps) as f64);
                rope_tail(&mut want, ps, n_rot, base, &yarn);
                if !row_matches(rows[(ps - s.w0 - ws) as usize], crate::quant::Q8_0, &want, ps) {
                    return Err(Fail::Mismatch);
                }
            } else if op.window != crate::seq::SeqSpan::default() || !op.anchor_k.is_empty() || !op.norms.is_empty() {
                return Err(Fail::Shape);
            }
            let (cs, ce) = (v0.max(s.n_w) - s.n_w, v1.max(s.n_w) - s.n_w);
            let ids = &s.sel[cs as usize..ce as usize];
            if op.comp.len() != ids.len() {
                return Err(Fail::Shape);
            }
            if !ids.is_empty() {
                let crb = (c.row_len as u64 / 32) * 34;
                let n_rows = n_pos / c.ratio as u64;
                let leaf = layout.comp(c.slot as u64).ok_or(Fail::Shape)?;
                for (&id, span) in ids.iter().zip(&op.comp) {
                    let b = crate::seq::open_span(&op.seq_root, &layout, leaf, n_rows * crb, id * crb, id * crb + rb, span).ok_or(Fail::Trace)?;
                    keys.push(crate::quant::dequant_row(crate::quant::Q8_0, b, hd).ok_or(Fail::Shape)?);
                }
            }
            let scale = f32::from_bits(a.kq_scale) as f64;
            let ok = match parent {
                Some(p) => check_leaf(&q, scale, &keys, &keys, &p),
                None => crate::attn::ARITHS.iter().any(|&ar| crate::attn::partial(ar, &q, scale, &keys, &keys).is_some_and(|p| root_ok(&[p]).unwrap_or(false))),
            };
            if !ok {
                return Err(Fail::Mismatch);
            }
        }
    }
    Ok(match children {
        None => crate::chain::AttnStep::Done,
        Some(ch) => crate::chain::AttnStep::Next { root: partials_root(&op.partials, hd).ok_or(Fail::Shape)?, n: ch.len() as u64, weights: child_weights(&ch).ok_or(Fail::Shape)? },
    })
}

/// Threshold and kept ids of a selection leaf of `n_slots` slots over `n_vis` candidates: the
/// first `min(n_vis, n_slots)` slots strictly increasing below `n_vis`, the rest `u32::MAX`.
pub fn parse_kept(b: &[u8], n_slots: usize, n_vis: u64) -> Option<(f64, Vec<u64>)> {
    if n_slots == 0 || b.len() != 4 + 4 * n_slots {
        return None;
    }
    let thr = f32::from_le_bytes(b[..4].try_into().unwrap()) as f64;
    let slots: Vec<u32> = b[4..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let m = (n_vis as usize).min(n_slots);
    let ids: Vec<u64> = slots[..m].iter().map(|&x| x as u64).collect();
    let ok = thr.is_finite() && slots[m..].iter().all(|&x| x == u32::MAX) && ids.windows(2).all(|w| w[0] < w[1]) && ids.last().is_none_or(|&x| x < n_vis);
    ok.then_some((thr, ids))
}

/// Indexer score of a compressed row's key `k` for the rotated query heads `q` (`w.len()` heads
/// end to end) and prescaled head weights `w`, and its magnitude.
pub fn index_score(q: &[f64], w: &[f64], k: &[f32]) -> (f64, f64) {
    let d = k.len();
    let (mut s, mut mag) = (0f64, 0f64);
    for (h, &wh) in w.iter().enumerate() {
        let dot: f64 = q[h * d..(h + 1) * d].iter().zip(k).map(|(&a, &b)| a * b as f64).sum();
        s += wh * dot.max(0.0);
        mag += wh.abs() * q[h * d..(h + 1) * d].iter().zip(k).map(|(&a, &b)| (a * b as f64).abs()).sum::<f64>();
    }
    (s, mag)
}

/// Verify the selection point of a compressed attention tier under `seed`: a layer's kept rows
/// are a top-k of its indexer scores (within its candidate blocks), or a layer's candidate blocks
/// are a top-k of their best row score with the newest block pinned; checked on drawn rows or
/// blocks on both sides of the committed threshold.
pub fn verify_comp_sel_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &CompSelOpening) -> Result<(), Fail> {
    use crate::dsa::{SEL_DRAWS, SEL_POINT, drawn_pools};
    let (nl, n_pos) = (ctx.n_layers as u64, resp.n_pos as u64);
    if ctx.n_sel == 0 || ctx.n_comp == 0 || nl == 0 || n_pos == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = crate::draw::point(seed, SEL_POINT);
    let s = d[1] % ctx.n_sel as u64;
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let c = match open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?.chain {
        Chain::Layer(ll) if ll.layer as u64 == layer => ll.comp.ok_or(Fail::Manifest)?,
        _ => return Err(Fail::Manifest),
    };
    let ratio = c.ratio as u64;
    let picks = c.pick == 1 && c.cand as u64 == s;
    let keeps = c.sel as u64 == s && c.topk_src as u64 == layer;
    if ratio == 0 || c.iq == manifest::NONE || picks == keeps {
        return Err(Fail::Manifest);
    }
    let (bs, cand_n, n_top) = (c.cand_bs as u64, c.cand_n as u64, c.n_top as u64);
    if picks && (bs == 0 || cand_n == 0) {
        return Err(Fail::Manifest);
    }
    let first = if picks { (cand_n * bs + 1) * ratio - 1 } else { (n_top + 1) * ratio - 1 };
    if first >= n_pos {
        return Ok(());
    }
    let t = first + d[0] % (n_pos - first);
    let n_vis = (t + 1) / ratio;
    let [eq, ew] = <[_; 2]>::try_from(op.entries.clone()).map_err(|_| Fail::Shape)?;
    let (iqe, iwe) = (open_entry(tier, &eq, c.iq as u64)?, open_entry(tier, &ew, c.iw as u64)?);
    let n_ih = iwe.m;
    let dk = iqe.m.checked_div(n_ih).filter(|&x| x > 0 && x * n_ih == iqe.m).ok_or(Fail::Manifest)?;
    let lat = (c.row_len as u64).checked_sub(dk).filter(|&l| l.is_multiple_of(32) && dk.is_multiple_of(32) && c.n_rot as u64 <= dk).ok_or(Fail::Manifest)?;
    if iqe.role != manifest::ROLE_PRODUCT || iwe.role != manifest::ROLE_PRODUCT {
        return Err(Fail::Manifest);
    }
    let shape = crate::chain::shape_any(tier, ctx);
    let sel_leaf = |i: u32| crate::chain::sel_leaf(tier.n_products, nl, crate::chain::n_router(ctx), i as u64);
    let root_t = leaf_root(shape.n_leaves, sel_leaf(s as u32), &op.sel)?;
    if leaf_root(shape.n_leaves, 2 * c.iq as u64 + 1, &op.iq)? != root_t || leaf_root(shape.n_leaves, 2 * c.iw as u64 + 1, &op.iw)? != root_t || !b3_verify(&op.pos_root, n_pos, t, &root_t, &op.pos_path) {
        return Err(Fail::Trace);
    }
    if op.iq.bytes.len() as u64 != 4 * iqe.m || op.iw.bytes.len() as u64 != 4 * n_ih {
        return Err(Fail::Shape);
    }
    let (base, yarn) = (f32::from_bits(c.rope_base) as f64, c.yarn());
    let mut q: Vec<f64> = f32s(&op.iq.bytes).iter().map(|&x| x as f64).collect();
    for h in q.chunks_mut(dk as usize) {
        rope_tail(h, t, c.n_rot as usize, base, &yarn);
    }
    let scale = 1.0 / ((dk * n_ih) as f64).sqrt();
    let w: Vec<f64> = f32s(&op.iw.bytes).iter().map(|&x| x as f64 * scale).collect();
    let tol = |mag: f64| (1e-5 + 4e-8 * t as f64) * mag + 1e-30;
    let crb = c.row_len as u64 / 32 * 34;
    let layout = seq_layout(ctx);
    let comp_leaf = layout.comp(c.slot as u64).ok_or(Fail::Shape)?;
    let n_rows = n_pos / ratio;
    // scores of the rows [j0, j1) from one opened span
    let scores = |span: &crate::seq::SeqSpan, j0: u64, j1: u64| -> Result<Vec<(f64, f64)>, Fail> {
        let b = crate::seq::open_span(&op.seq_root, &layout, comp_leaf, n_rows * crb, j0 * crb, j1 * crb, span).ok_or(Fail::Trace)?;
        b.chunks_exact(crb as usize)
            .map(|r| crate::quant::dequant_row(crate::quant::Q8_0, r, c.row_len as usize).map(|v| index_score(&q, &w, &v[lat as usize..])).ok_or(Fail::Shape))
            .collect()
    };
    if picks {
        let nbv = n_vis.div_ceil(bs);
        let (thr, blocks) = parse_kept(&op.sel.bytes, cand_n as usize, nbv).ok_or(Fail::Mismatch)?;
        let pinned = (n_vis - 1) / bs;
        if !blocks.contains(&pinned) || !op.cand.bytes.is_empty() {
            return Err(if op.cand.bytes.is_empty() { Fail::Mismatch } else { Fail::Shape });
        }
        let drawn = drawn_pools(seed, &blocks, nbv);
        if op.rows.len() != drawn.len() {
            return Err(Fail::Shape);
        }
        for (i, (&b, span)) in drawn.iter().zip(&op.rows).enumerate() {
            let sc = scores(span, b * bs, ((b + 1) * bs).min(n_vis))?;
            let (best, mag) = sc.iter().fold((f64::NEG_INFINITY, 0f64), |(m, g), &(x, y)| (m.max(x), g.max(y)));
            let ok = if i < SEL_DRAWS { b == pinned || best + tol(mag) >= thr } else { best - tol(mag) <= thr };
            if !ok {
                return Err(Fail::Mismatch);
            }
        }
        return Ok(());
    }
    let (thr, ids) = parse_kept(&op.sel.bytes, n_top as usize, n_vis).ok_or(Fail::Mismatch)?;
    // the layer picking the candidate blocks keeps its rows over all its scores
    let eligible: Box<dyn Fn(u64) -> bool> = if c.cand == manifest::NONE || c.pick == 1 {
        if !op.cand.bytes.is_empty() {
            return Err(Fail::Shape);
        }
        Box::new(|_| true)
    } else {
        if leaf_root(shape.n_leaves, sel_leaf(c.cand), &op.cand)? != root_t {
            return Err(Fail::Trace);
        }
        let (_, blocks) = parse_kept(&op.cand.bytes, cand_n as usize, n_vis.div_ceil(bs)).ok_or(Fail::Mismatch)?;
        Box::new(move |j: u64| blocks.binary_search(&(j / bs)).is_ok())
    };
    let drawn = drawn_pools(seed, &ids, n_vis);
    if op.rows.len() != drawn.len() {
        return Err(Fail::Shape);
    }
    for (i, (&j, span)) in drawn.iter().zip(&op.rows).enumerate() {
        let (sc, mag) = scores(span, j, j + 1)?[0];
        let ok = if i < SEL_DRAWS { eligible(j) && sc + tol(mag) >= thr } else { !eligible(j) || sc - tol(mag) <= thr };
        if !ok {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
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
                sinks: NONE, sel: NONE, n_top: 0, row_len: 640, iq: NONE, iw: NONE, topk_src: NONE, cand: NONE, cand_n: 0, cand_bs: 0, pick: 0,
            };
            let none_link = Chain::Product(ProductLink { layer: 0, link: manifest::LINK_NONE, norm: NONE, src: NONE });
            let layer = LayerLinks { layer: 0, proj: 0, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: Some(attn), recur: None, hc: None, dsa: None, comp: Some(comp), engram: None };
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
            let ctx = ChainCtx { n_sel: 0, n_comp: 1, n_engram: 0, n_layers: 1, layer_base: 2, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 1, n_recur: 0, seq_heads: 1 };
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

    /// ggml heads over 9 keys that are also the values, with a sink logit each (one dominating),
    /// output de-rotated at position 3000.
    #[test]
    fn sink_heads_match_ggml() {
        let f = f32s(include_bytes!("../testdata/dsv41-head.bin"));
        let (d, h, nk) = (512usize, 4usize, 9usize);
        let (q, k, sk, want) = (&f[..d * h], &f[d * h..d * h + d * nk], &f[d * h + d * nk..d * h + d * nk + h], &f[d * h + d * nk + h..]);
        let keys: Vec<Vec<f32>> = k.chunks_exact(d).map(|c| c.to_vec()).collect();
        let yarn = Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 };
        for hd in 0..h {
            let qh: Vec<f64> = q[hd * d..(hd + 1) * d].iter().map(|&x| x as f64).collect();
            let p = crate::attn::partial(crate::attn::Arith::Exact, &qh, 1.0 / (d as f64).sqrt(), &keys, &keys).unwrap();
            let mut o = with_sink(&p, sk[hd] as f64).unwrap().o;
            derope(&mut o, 3000, 64, 160_000.0, &yarn);
            let dev = o.iter().zip(&want[hd * d..(hd + 1) * d]).fold(0f64, |m, (a, &b)| m.max((a - b as f64).abs()));
            assert!(dev < 1e-6 + 4e-8 * 3000.0, "head {hd}: {dev}");
            let without = p.o.iter().zip(&want[hd * d..(hd + 1) * d]).fold(0f64, |m, (a, &b)| m.max((a - b as f64).abs()));
            assert!(without > 1e-3, "head {hd}: the sink and the de-rotation must matter");
        }
    }

    mod comp_attn {
        use super::*;
        use crate::attn::{Arith, Partial};
        use crate::attn_blocks::{BlockTree, block_params, children_partials, encode_partials, partial_len, partial_opening, set_test_budget};
        use crate::chain::{ATTN_POINT, AttnStep};
        use crate::manifest::{AttnLinks, CompLinks, Entry, LayerLinks, ProductLink};
        use crate::merkle::{B3_CHUNK, b3_chunk_cv, b3_chunks_proof, b3_path, b3_root, rt_range_proof};
        use crate::opening::{ChunkRange, EntryOpening, LeafChunk, SelOpening};
        use crate::seq::{SeqLayout, SeqSpan};
        use crate::testkit::{in_layout, layout_at, q8_1_row};

        const N: usize = 12;
        const NH: usize = 2;
        const HD: usize = 512;
        const WIN: u32 = 4;
        const NTOP: usize = 2;
        const NONE: u32 = manifest::NONE;
        const SEL: usize = 8;
        const LAYOUT: SeqLayout = SeqLayout { n_attn: 1, n_recur: 0, heads: 1, n_comp: 1 };
        const SINKS: [f32; NH] = [0.3, -0.7];
        const BUDGET: usize = 2 * COMP_ROW_COST;

        #[derive(Clone, Copy, PartialEq)]
        enum Fault {
            None,
            /// Every visible compressed row attended, not only the selected ones.
            Dense,
            /// No sink in the softmax.
            NoSink,
            /// A window row of no committed projection.
            ForgedWindow,
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
            pos_root: Hash,
            seq_leaves: Vec<Vec<u8>>,
            seq_hashes: Vec<Hash>,
            seq_root: Hash,
            q: Vec<Vec<f32>>,
            win_keys: Vec<Vec<f32>>,
            comp_keys: Vec<Vec<f32>>,
        }

        fn yarn() -> Yarn {
            Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 }
        }

        fn entry(name: &str, role: u8, k: usize, m: usize, n_exp: u32, off: usize, n: usize, chain: Chain) -> Vec<u8> {
            let product = matches!(role, manifest::ROLE_PRODUCT | manifest::ROLE_HEADS);
            let e = Entry {
                name: name.into(), wtype: if product { crate::quant::Q8_0 } else { 0 }, k: k as u64, m: m as u64, n_exp, chunk_base: (off / 32) as u64,
                n_bytes: n as u64, row_bytes: if product { (k / 32 * 34) as u32 } else { n as u32 }, role, glu_op: 0, glu_form: manifest::FORM_NONE,
                partner: NONE, bias: NONE, chain,
            };
            let mut b = crate::testkit::serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            b
        }

        fn selection(t: usize) -> Vec<u64> {
            let n_vis = (t + 1) / 2;
            (n_vis.saturating_sub(NTOP)..n_vis).map(|k| k as u64).collect()
        }

        fn window(t: usize) -> std::ops::RangeInclusive<usize> {
            (t + 1).saturating_sub(WIN as usize)..=t
        }

        fn build(fault: Fault) -> Kit {
            let norm: Vec<f32> = (0..HD).map(|i| 0.6 + 0.01 * (i % 29) as f32).collect();
            let mut weights: Vec<u8> = norm.iter().flat_map(|a| a.to_le_bytes()).collect();
            let sinks_off = weights.len();
            weights.extend((0..8).flat_map(|i| SINKS.get(i).copied().unwrap_or(0.0).to_le_bytes()));
            let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let y = yarn();
            let attn = AttnLinks {
                q: 0, k: 1, v: NONE, q_norm: NONE, k_norm: 4, n_head: NH as u32, n_head_kv: 1, head_dim: HD as u32, rope_kind: 0, n_rot: 64,
                rope_base: 10_000f32.to_bits(), kq_scale: (1.0f32 / (HD as f32).sqrt()).to_bits(), swa_window: WIN, softcap: 0, rope_freqs: NONE,
                v_norm: 0, gated: 0, slot: 0, mla_rank: 0, absorb: NONE, out: 2,
            };
            let comp = CompLinks {
                src: 0, slot: 0, ratio: 2, kv: NONE, gate: NONE, norm: NONE, ik: NONE, k_norm: NONE, n_rot: 64, rope_base: 160_000f32.to_bits(),
                freq_scale: (y.freq_scale as f32).to_bits(), ext_factor: 1f32.to_bits(), attn_factor: (y.attn_factor as f32).to_bits(),
                beta_fast: 32f32.to_bits(), beta_slow: 1f32.to_bits(), n_ctx_orig: 65_536, eps: 1e-20f32.to_bits(), sinks: 5, sel: 0, n_top: NTOP as u32, row_len: 640, iq: NONE, iw: NONE, topk_src: 0, cand: NONE, cand_n: 0, cand_bs: 0, pick: 0,
            };
            let none_link = Chain::Product(ProductLink { layer: 0, link: manifest::LINK_NONE, norm: NONE, src: NONE });
            let layer = LayerLinks { layer: 0, proj: 2, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: Some(attn), recur: None, hc: None, dsa: None, comp: Some(comp), engram: None };
            let entries = vec![
                entry("blk.0.attn_q_b.weight", manifest::ROLE_PRODUCT, 64, NH * HD, 1, 0, 0, none_link),
                entry("blk.0.attn_kv.weight", manifest::ROLE_PRODUCT, 64, HD, 1, 0, 0, none_link),
                entry("blk.0.attn_output_a.weight", manifest::ROLE_HEADS, NH * HD, 16, 1, 0, 0, none_link),
                entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, 0, Chain::Layer(layer)),
                entry("blk.0.attn_kv_a_norm.weight", manifest::ROLE_NORM, HD, 1, 1, 0, 4 * HD, Chain::None),
                entry("blk.0.attn_sinks.weight", manifest::ROLE_NORM, NH, 1, 1, sinks_off, 32, Chain::None),
            ];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx { r_t: rt_range_proof(&chunk_leaves, 0, 1).0, n_chunks: chunk_leaves.len() as u64, m_t: rt_range_proof(&entry_leaves, 0, 1).0, n_entries: entries.len() as u64, n_products: 3 };
            let ctx = ChainCtx { n_sel: 1, n_comp: 1, n_engram: 0, n_layers: 1, layer_base: 3, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 1, n_recur: 0, seq_heads: 1 };
            let kv_y: Vec<Vec<f32>> = (0..N).map(|t| vals(300 + t as u64, HD, 2.0)).collect();
            let q: Vec<Vec<f32>> = (0..N).map(|t| vals(400 + t as u64, NH * HD, 1.0)).collect();
            let win_rows: Vec<Vec<u8>> = (0..N)
                .map(|t| {
                    let src = if fault == Fault::ForgedWindow { vals(900 + t as u64, HD, 2.0) } else { kv_y[t].clone() };
                    let mut k: Vec<f64> = src.iter().map(|&x| x as f64).collect();
                    rms_norm(&mut k, Some(&norm), 1e-20);
                    rope_tail(&mut k, t as u64, 64, 160_000.0, &y);
                    q8_0(&k)
                })
                .collect();
            let win_keys: Vec<Vec<f32>> = win_rows.iter().map(|r| crate::quant::dequant_row(crate::quant::Q8_0, r, HD).unwrap()).collect();
            let comp_rows: Vec<Vec<u8>> = (0..N / 2).map(|k| q8_0(&vals(500 + k as u64, 640, 1.0).iter().map(|&x| x as f64).collect::<Vec<_>>())).collect();
            let comp_keys: Vec<Vec<f32>> = comp_rows.iter().map(|r| crate::quant::dequant_row(crate::quant::Q8_0, &r[..HD / 32 * 34], HD).unwrap()).collect();
            let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
            let n_leaves = crate::chain::n_pos_leaves(3, 1, 0) as usize + 1;
            let mut lse_leaf = Vec::new();
            let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..N)
                .map(|t| {
                    let ids = selection(t);
                    let comp_ids: Vec<u64> = if fault == Fault::Dense { (0..(t as u64 + 1) / 2).collect() } else { ids.clone() };
                    let keys: Vec<Vec<f32>> = window(t).map(|p| win_keys[p].clone()).chain(comp_ids.iter().map(|&k| comp_keys[k as usize].clone())).collect();
                    let mut x = Vec::new();
                    for h in 0..NH {
                        let mut qh: Vec<f64> = q[t][h * HD..(h + 1) * HD].iter().map(|&v| v as f64).collect();
                        rope_tail(&mut qh, t as u64, 64, 160_000.0, &y);
                        let p = crate::attn::partial(Arith::Exact, &qh, 1.0 / (HD as f64).sqrt(), &keys, &keys).unwrap();
                        let c = if fault == Fault::NoSink { p } else { with_sink(&p, SINKS[h] as f64).unwrap() };
                        lse_leaf.extend((crate::attn_blocks::lse(&c) as f32).to_le_bytes());
                        let mut o = c.o;
                        derope(&mut o, t as u64, 64, 160_000.0, &y);
                        x.extend(o);
                    }
                    let mut leaves = vec![Vec::new(); n_leaves];
                    leaves[1] = f(&q[t]);
                    leaves[3] = f(&kv_y[t]);
                    leaves[4] = in_layout(&q8_1_row(&x), layout_at(t as u64, N as u32).1);
                    leaves[5] = f(&vals(600 + t as u64, 16, 1.0));
                    let mut sel = 0f32.to_le_bytes().to_vec();
                    sel.extend(ids.iter().map(|&k| k as u32).chain(std::iter::repeat(u32::MAX)).take(NTOP).flat_map(|v| v.to_le_bytes()));
                    leaves[SEL] = sel;
                    leaves[n_leaves - 1] = crate::chain::meta(t as u32, layout_at(t as u64, N as u32).0, 7).to_vec();
                    let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                    (leaves, hashes)
                })
                .collect();
            let pos_roots: Vec<Hash> = pos.iter().map(|(_, h)| b3_root(h)).collect();
            let pos_root = b3_root(&pos_roots);
            let mut seq_leaves = vec![Vec::new(); LAYOUT.n_leaves() as usize];
            seq_leaves[LAYOUT.kcache(0, 0).unwrap() as usize] = win_rows.concat();
            seq_leaves[LAYOUT.lse(0, 0).unwrap() as usize] = lse_leaf;
            seq_leaves[LAYOUT.comp(0).unwrap() as usize] = comp_rows.concat();
            let seq_hashes: Vec<Hash> = seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            let seq_root = b3_root(&seq_hashes);
            Kit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos, pos_roots, pos_root, seq_leaves, seq_hashes, seq_root, q, win_keys, comp_keys }
        }

        impl Kit {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn range(&self, off: usize, n: usize) -> ChunkRange {
                ChunkRange { first_chunk: (off / 32) as u64, chunks: self.weights[off..off + n].to_vec(), proof: rt_range_proof(&self.chunk_leaves, (off / 32) as u64, (n / 32) as u64).1 }
            }

            fn resp(&self) -> ResponseCtx {
                ResponseCtx { trace_root: crate::seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: N as u32, n_pos: N as u32 }
            }

            fn leaf_bytes(&self, t: usize, i: usize) -> LeafBytes {
                LeafBytes { bytes: self.pos[t].0[i].clone(), leaf_path: b3_path(&self.pos[t].1, i) }
            }

            fn leaf_chunk(&self, t: usize, i: usize, c: usize) -> LeafChunk {
                let b = &self.pos[t].0[i];
                let n = b.len().div_ceil(B3_CHUNK).max(1);
                let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
                LeafChunk {
                    total_len: b.len() as u32,
                    index: c as u32,
                    chunk: b[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(b.len())].to_vec(),
                    chunk_path: if n == 1 { Vec::new() } else { b3_path(&cvs, c) },
                    leaf_path: b3_path(&self.pos[t].1, i),
                }
            }

            fn span(&self, leaf: u64, lo: usize, hi: usize) -> SeqSpan {
                let data = &self.seq_leaves[leaf as usize];
                let (c0, c1) = (lo / B3_CHUNK, (hi - 1) / B3_CHUNK);
                SeqSpan {
                    first_chunk: c0 as u32,
                    bytes: data[c0 * B3_CHUNK..((c1 + 1) * B3_CHUNK).min(data.len())].to_vec(),
                    chunk_proof: b3_chunks_proof(data, c0 as u64, (c1 - c0 + 1) as u64),
                    leaf_path: b3_path(&self.seq_hashes, leaf as usize),
                }
            }

            fn tree(&self, t: usize) -> BlockTree {
                let (c, b) = block_params(COMP_ROW_COST, partial_len(HD)).unwrap();
                BlockTree::new(0, (window(t).count() + selection(t).len()) as u64, c, b).unwrap()
            }

            fn open(&self, seed: &Hash, path: &[u64], prev: Option<&[u8]>) -> CompAttnOpening {
                let d = crate::draw::point(seed, ATTN_POINT);
                let (t, qh) = ((d[0] % N as u64) as usize, (d[2] % NH as u64) as usize);
                let tree = self.tree(t);
                let ids = selection(t);
                let (parent, parent_path) = match (path.last(), prev) {
                    (Some(&i), Some(b)) => partial_opening(b, HD, i as usize).unwrap(),
                    _ => Default::default(),
                };
                let mut op = CompAttnOpening {
                    response_txid: [1; 32],
                    pos_root: self.pos_root,
                    seq_root: self.seq_root,
                    layer: self.eo(3),
                    sel: Some(SelOpening { bytes: self.pos[t].0[SEL].clone(), leaf_path: b3_path(&self.pos[t].1, SEL), pos_path: b3_path(&self.pos_roots, t) }),
                    path: path.to_vec(),
                    parent,
                    parent_path,
                    ..Default::default()
                };
                let last = path.len() as u32 == tree.depth();
                if path.is_empty() || last {
                    op.q_entry = self.eo(0);
                    op.q = vec![self.leaf_chunk(t, 1, 2 * qh), self.leaf_chunk(t, 1, 2 * qh + 1)];
                    op.pos_path = b3_path(&self.pos_roots, t);
                }
                if path.is_empty() {
                    op.o_entry = self.eo(2);
                    op.out = self.leaf_bytes(t, 4);
                    op.sinks = (self.eo(5), self.range(2048, 32));
                    let lo = (t * NH + qh) * 4;
                    op.lse = self.span(LAYOUT.lse(0, 0).unwrap(), lo, lo + 4);
                }
                let w: Vec<usize> = window(t).collect();
                if last {
                    let (v0, v1) = tree.node(path).unwrap();
                    let n_w = w.len() as u64;
                    let (ws, we) = (v0.min(n_w) as usize, v1.min(n_w) as usize);
                    let rb = HD / 32 * 34;
                    if ws < we {
                        op.window = self.span(LAYOUT.kcache(0, 0).unwrap(), (w[0] + ws) * rb, (w[0] + we) * rb);
                        let ps = w[0] + ws + (d[3] % (we - ws) as u64) as usize;
                        op.anchor_pos_path = b3_path(&self.pos_roots, ps);
                        op.anchor_k = vec![self.leaf_chunk(ps, 3, 0), self.leaf_chunk(ps, 3, 1)];
                        op.norms = vec![(self.eo(4), self.range(0, 4 * HD))];
                    }
                    let (cs, ce) = ((v0.max(n_w) - n_w) as usize, (v1.max(n_w) - n_w) as usize);
                    op.comp = ids[cs..ce].iter().map(|&k| self.span(LAYOUT.comp(0).unwrap(), k as usize * 680, k as usize * 680 + rb)).collect();
                } else {
                    let mut qv: Vec<f64> = self.q[t][qh * HD..(qh + 1) * HD].iter().map(|&v| v as f64).collect();
                    rope_tail(&mut qv, t as u64, 64, 160_000.0, &yarn());
                    let keys: Vec<Vec<f32>> = w.iter().map(|&p| self.win_keys[p].clone()).chain(ids.iter().map(|&k| self.comp_keys[k as usize].clone())).collect();
                    let parts: Vec<Partial> = children_partials(Arith::Exact, &qv, 1.0 / (HD as f64).sqrt(), &keys, &keys, 0, &tree, path).unwrap();
                    op.partials = encode_partials(&parts);
                }
                op
            }

            /// Every phase of every path the point of `seed` can take; the deepest path reached.
            fn walk(&self, seed: &Hash) -> Result<usize, (Vec<u64>, Fail)> {
                let mut stack: Vec<(Vec<u64>, Option<Vec<u8>>, Option<Hash>)> = vec![(Vec::new(), None, None)];
                let mut deepest = 0;
                while let Some((path, prev, root)) = stack.pop() {
                    let op = self.open(seed, &path, prev.as_deref());
                    assert_eq!(CompAttnOpening::decode(&op.encode()).as_ref(), Some(&op));
                    match verify_comp_attn_phase(&self.tier, &self.ctx, &self.resp(), seed, &path, root.as_ref(), &op) {
                        Ok(AttnStep::Done) => deepest = deepest.max(path.len()),
                        Ok(AttnStep::Next { root, n, .. }) => stack.extend((0..n).map(|i| ([path.clone(), vec![i]].concat(), Some(op.partials.clone()), Some(root)))),
                        Err(e) => return Err((path, e)),
                    }
                }
                Ok(deepest)
            }
        }

        fn seeds() -> impl Iterator<Item = Hash> {
            (0..40u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
        }

        fn draw_t(s: &Hash) -> usize {
            (crate::draw::point(s, ATTN_POINT)[0] % N as u64) as usize
        }

        #[test]
        fn honest_compressed_attention_passes_every_phase() {
            set_test_budget(BUDGET);
            let kit = build(Fault::None);
            let deep = seeds().map(|s| kit.walk(&s).unwrap_or_else(|e| panic!("{e:?}"))).max();
            assert!(deep >= Some(1), "{deep:?}");
        }

        #[test]
        fn honest_compressed_attention_passes_whole() {
            let kit = build(Fault::None);
            for s in seeds() {
                kit.walk(&s).unwrap_or_else(|e| panic!("{e:?}"));
            }
        }

        #[test]
        fn attention_beyond_the_selection_or_without_the_sink_fails_at_the_root() {
            set_test_budget(BUDGET);
            for fault in [Fault::Dense, Fault::NoSink] {
                let kit = build(fault);
                for s in seeds() {
                    let t = draw_t(&s);
                    let dense_differs = (t + 1) / 2 > NTOP;
                    let r = kit.walk(&s);
                    if fault == Fault::NoSink || dense_differs {
                        assert_eq!(r, Err((Vec::new(), Fail::Mismatch)), "t {t}");
                    } else {
                        assert!(r.is_ok());
                    }
                }
            }
        }

        #[test]
        fn a_window_row_of_no_committed_projection_fails_at_its_leaf() {
            set_test_budget(BUDGET);
            let kit = build(Fault::ForgedWindow);
            let fails = seeds().filter(|s| matches!(kit.walk(s), Err((p, Fail::Mismatch)) if !p.is_empty())).count();
            assert!(fails > 0);
        }

        #[test]
        fn another_selection_is_refused() {
            set_test_budget(BUDGET);
            let kit = build(Fault::None);
            let s = seeds().find(|s| (draw_t(s) + 1) / 2 > NTOP).unwrap();
            let mut op = kit.open(&s, &[], None);
            op.sel.as_mut().unwrap().bytes[4] ^= 1;
            assert_eq!(verify_comp_attn_phase(&kit.tier, &kit.ctx, &kit.resp(), &s, &[], None, &op).err(), Some(Fail::Trace));
            op.sel = None;
            assert_eq!(verify_comp_attn_phase(&kit.tier, &kit.ctx, &kit.resp(), &s, &[], None, &op).err(), Some(Fail::Shape));
        }
    }

    mod comp_sel {
        use super::*;
        use crate::dsa::{SEL_POINT, drawn_pools};
        use crate::manifest::{AttnLinks, CompLinks, Entry, LayerLinks, ProductLink};
        use crate::merkle::{B3_CHUNK, b3_chunks_proof, b3_path, b3_root, rt_range_proof};
        use crate::opening::EntryOpening;
        use crate::seq::{SeqLayout, SeqSpan};

        const N: usize = 12;
        const NIH: usize = 2;
        const DK: usize = 128;
        const LAT: usize = 32;
        const NTOP: usize = 2;
        const CN: usize = 2;
        const BS: usize = 2;
        const NONE: u32 = manifest::NONE;
        const LAYOUT: SeqLayout = SeqLayout { n_attn: 2, n_recur: 0, heads: 1, n_comp: 1 };

        #[derive(Clone, Copy, PartialEq)]
        enum Fault {
            None,
            /// Layer 0 keeps its two lowest rows.
            LowRows,
            /// Layer 0 picks its lowest blocks besides the pinned one.
            LowBlocks,
            /// Layer 1 keeps a row outside the candidate blocks.
            OutsideCandidates,
        }

        struct Kit {
            entries: Vec<Vec<u8>>,
            entry_leaves: Vec<Hash>,
            tier: TierCtx,
            ctx: ChainCtx,
            pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
            pos_roots: Vec<Hash>,
            pos_root: Hash,
            seq_leaves: Vec<Vec<u8>>,
            seq_hashes: Vec<Hash>,
            seq_root: Hash,
        }

        fn entry(name: &str, role: u8, k: usize, m: usize, chain: Chain) -> Vec<u8> {
            let e = Entry {
                name: name.into(), wtype: crate::quant::Q8_0, k: k as u64, m: m as u64, n_exp: 1, chunk_base: 0, n_bytes: 0, row_bytes: 0, role, glu_op: 0,
                glu_form: manifest::FORM_NONE, partner: NONE, bias: NONE, chain,
            };
            let mut b = crate::testkit::serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            b
        }

        fn links(layer: u32, iq: u32, sel: u32, pick: u32) -> CompLinks {
            let y = block().yarn;
            CompLinks {
                src: 0, slot: 0, ratio: 1, kv: NONE, gate: NONE, norm: NONE, ik: NONE, k_norm: NONE, n_rot: 64, rope_base: 160_000f32.to_bits(),
                freq_scale: (y.freq_scale as f32).to_bits(), ext_factor: 1f32.to_bits(), attn_factor: (y.attn_factor as f32).to_bits(), beta_fast: 32f32.to_bits(),
                beta_slow: 1f32.to_bits(), n_ctx_orig: 65_536, eps: 1e-20f32.to_bits(), sinks: NONE, sel, n_top: NTOP as u32, row_len: (LAT + DK) as u32,
                iq, iw: iq + 1, topk_src: layer, cand: 1, cand_n: CN as u32, cand_bs: BS as u32, pick,
            }
        }

        fn top(scores: &[f64], keep: impl Fn(usize) -> bool, n: usize, pinned: Option<usize>) -> (f64, Vec<usize>) {
            let mut order: Vec<usize> = (0..scores.len()).filter(|&j| keep(j) && Some(j) != pinned).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let room = n - pinned.is_some() as usize;
            let mut kept: Vec<usize> = order.iter().take(room).copied().chain(pinned).collect();
            kept.sort();
            let thr = match (order.get(room.saturating_sub(1)), order.get(room)) {
                (Some(&a), Some(&b)) if room > 0 => (scores[a] + scores[b]) / 2.0,
                (Some(&a), None) => scores[a],
                _ => 0.0,
            };
            (thr, kept)
        }

        fn leaf(thr: f64, kept: &[usize], slots: usize) -> Vec<u8> {
            let mut b = (thr as f32).to_le_bytes().to_vec();
            b.extend(kept.iter().map(|&k| k as u32).chain(std::iter::repeat(u32::MAX)).take(slots).flat_map(|v| v.to_le_bytes()));
            b
        }

        fn build(fault: Fault) -> Kit {
            let a_links = |slot: u32| AttnLinks {
                q: 0, k: 0, v: NONE, q_norm: NONE, k_norm: NONE, n_head: 1, n_head_kv: 1, head_dim: 512, rope_kind: 0, n_rot: 64, rope_base: 0,
                kq_scale: 1f32.to_bits(), swa_window: 4, softcap: 0, rope_freqs: NONE, v_norm: 0, gated: 0, slot, mla_rank: 0, absorb: NONE, out: NONE,
            };
            let none_link = Chain::Product(ProductLink { layer: 0, link: manifest::LINK_NONE, norm: NONE, src: NONE });
            let layer = |l: u32, c: CompLinks| {
                Chain::Layer(LayerLinks { layer: l, proj: 0, down: 0, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: Some(a_links(l)), recur: None, hc: None, dsa: None, comp: Some(c), engram: None })
            };
            let entries = vec![
                entry("blk.0.indexer.attn_q_b.weight", manifest::ROLE_PRODUCT, 64, NIH * DK, none_link),
                entry("blk.0.indexer.proj.weight", manifest::ROLE_PRODUCT, 64, NIH, none_link),
                entry("blk.1.indexer.attn_q_b.weight", manifest::ROLE_PRODUCT, 64, NIH * DK, none_link),
                entry("blk.1.indexer.proj.weight", manifest::ROLE_PRODUCT, 64, NIH, none_link),
                entry("blk.0", manifest::ROLE_LAYER, 0, 0, layer(0, links(0, 0, 0, 1))),
                entry("blk.1", manifest::ROLE_LAYER, 0, 0, layer(1, links(1, 2, 2, 0))),
            ];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx { r_t: [0; 32], n_chunks: 1, m_t: rt_range_proof(&entry_leaves, 0, 1).0, n_entries: entries.len() as u64, n_products: 4 };
            let ctx = ChainCtx { n_sel: 3, n_comp: 1, n_engram: 0, n_layers: 2, layer_base: 4, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 2, n_recur: 0, seq_heads: 1 };
            let y = block().yarn;
            let rows: Vec<Vec<u8>> = (0..N).map(|j| q8_0(&vals(700 + j as u64, LAT + DK, 1.0).iter().map(|&x| x as f64).collect::<Vec<_>>())).collect();
            let keys: Vec<Vec<f32>> = rows.iter().map(|r| crate::quant::dequant_row(crate::quant::Q8_0, r, LAT + DK).unwrap()[LAT..].to_vec()).collect();
            let iq: Vec<[Vec<f32>; 2]> = (0..N).map(|t| [vals(800 + t as u64, NIH * DK, 1.0), vals(820 + t as u64, NIH * DK, 1.0)]).collect();
            let iw: Vec<[Vec<f32>; 2]> = (0..N).map(|t| [vals(840 + t as u64, NIH, 1.0).iter().map(|x| x.abs() + 0.2).collect(), vals(860 + t as u64, NIH, 1.0).iter().map(|x| x.abs() + 0.2).collect()]).collect();
            let score = |t: usize, l: usize, j: usize| {
                let mut q: Vec<f64> = iq[t][l].iter().map(|&x| x as f64).collect();
                for h in q.chunks_mut(DK) {
                    rope_tail(h, t as u64, 64, 160_000.0, &y);
                }
                let w: Vec<f64> = iw[t][l].iter().map(|&x| x as f64 / ((DK * NIH) as f64).sqrt()).collect();
                index_score(&q, &w, &keys[j]).0
            };
            let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
            let n_leaves = crate::chain::n_pos_leaves(4, 2, 0) as usize + 3;
            let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..N)
                .map(|t| {
                    let n_vis = t + 1;
                    let s0: Vec<f64> = (0..n_vis).map(|j| score(t, 0, j)).collect();
                    let s1: Vec<f64> = (0..n_vis).map(|j| score(t, 1, j)).collect();
                    let bmax: Vec<f64> = (0..n_vis.div_ceil(BS)).map(|b| (b * BS..((b + 1) * BS).min(n_vis)).map(|j| s0[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
                    let pinned = (n_vis - 1) / BS;
                    let (bthr, blocks) = if fault == Fault::LowBlocks && bmax.len() > CN {
                        let neg: Vec<f64> = bmax.iter().map(|x| -x).collect();
                        let (th, k) = top(&neg, |_| true, CN, Some(pinned));
                        (-th, k)
                    } else {
                        top(&bmax, |_| true, CN, Some(pinned))
                    };
                    let (thr0, kept0) = if fault == Fault::LowRows && n_vis > NTOP {
                        let neg: Vec<f64> = s0.iter().map(|x| -x).collect();
                        let (th, k) = top(&neg, |_| true, NTOP, None);
                        (-th, k)
                    } else {
                        top(&s0, |_| true, NTOP, None)
                    };
                    let in_cand = |j: usize| blocks.contains(&(j / BS));
                    let (thr1, mut kept1) = top(&s1, in_cand, NTOP, None);
                    if fault == Fault::OutsideCandidates && n_vis > CN * BS {
                        if let Some(out) = (0..n_vis).find(|&j| !in_cand(j)) {
                            kept1[0] = out;
                            kept1.sort();
                        }
                    }
                    let mut leaves = vec![Vec::new(); n_leaves];
                    leaves[1] = f(&iq[t][0]);
                    leaves[3] = f(&iw[t][0]);
                    leaves[5] = f(&iq[t][1]);
                    leaves[7] = f(&iw[t][1]);
                    let base = crate::chain::sel_leaf(4, 2, 0, 0) as usize;
                    leaves[base] = leaf(thr0, &kept0, NTOP);
                    leaves[base + 1] = leaf(bthr, &blocks, CN);
                    leaves[base + 2] = leaf(thr1, &kept1, NTOP);
                    leaves[n_leaves - 1] = crate::chain::meta(t as u32, 1, 7).to_vec();
                    let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                    (leaves, hashes)
                })
                .collect();
            let pos_roots: Vec<Hash> = pos.iter().map(|(_, h)| b3_root(h)).collect();
            let pos_root = b3_root(&pos_roots);
            let mut seq_leaves = vec![Vec::new(); LAYOUT.n_leaves() as usize];
            seq_leaves[LAYOUT.comp(0).unwrap() as usize] = rows.concat();
            let seq_hashes: Vec<Hash> = seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            let seq_root = b3_root(&seq_hashes);
            Kit { entries, entry_leaves, tier, ctx, pos, pos_roots, pos_root, seq_leaves, seq_hashes, seq_root }
        }

        impl Kit {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn resp(&self) -> ResponseCtx {
                ResponseCtx { trace_root: crate::seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: N as u32, n_pos: N as u32 }
            }

            fn lb(&self, t: usize, i: usize) -> LeafBytes {
                LeafBytes { bytes: self.pos[t].0[i].clone(), leaf_path: b3_path(&self.pos[t].1, i) }
            }

            fn span(&self, lo: usize, hi: usize) -> SeqSpan {
                let leaf = LAYOUT.comp(0).unwrap() as usize;
                let data = &self.seq_leaves[leaf];
                let (c0, c1) = (lo / B3_CHUNK, (hi - 1) / B3_CHUNK);
                SeqSpan {
                    first_chunk: c0 as u32,
                    bytes: data[c0 * B3_CHUNK..((c1 + 1) * B3_CHUNK).min(data.len())].to_vec(),
                    chunk_proof: b3_chunks_proof(data, c0 as u64, (c1 - c0 + 1) as u64),
                    leaf_path: b3_path(&self.seq_hashes, leaf),
                }
            }

            /// The opening of `seed`, `None` when it draws nothing to check.
            fn open(&self, seed: &Hash) -> Option<CompSelOpening> {
                let d = crate::draw::point(seed, SEL_POINT);
                let s = (d[1] % 3) as usize;
                let (layer, l) = [(4usize, 0usize), (4, 0), (5, 1)][s];
                let picks = s == 1;
                let first = if picks { CN * BS } else { NTOP };
                let t = first + (d[0] % (N - first) as u64) as usize;
                let n_vis = t + 1;
                let base = crate::chain::sel_leaf(4, 2, 0, 0) as usize;
                let rb = (LAT + DK) / 32 * 34;
                let kept = |b: &[u8]| b[4..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).filter(|&x| x != u32::MAX).map(|x| x as u64).collect::<Vec<_>>();
                let sel = &self.pos[t].0[base + s];
                let rows = if picks {
                    drawn_pools(seed, &kept(sel), n_vis.div_ceil(BS) as u64)
                        .iter()
                        .map(|&b| self.span(b as usize * BS * rb, ((b as usize + 1) * BS).min(n_vis) * rb))
                        .collect()
                } else {
                    drawn_pools(seed, &kept(sel), n_vis as u64).iter().map(|&j| self.span(j as usize * rb, (j as usize + 1) * rb)).collect()
                };
                Some(CompSelOpening {
                    response_txid: [1; 32],
                    pos_root: self.pos_root,
                    seq_root: self.seq_root,
                    layer: self.eo(layer),
                    entries: vec![self.eo(2 * l), self.eo(2 * l + 1)],
                    pos_path: b3_path(&self.pos_roots, t),
                    sel: self.lb(t, base + s),
                    iq: self.lb(t, 4 * l + 1),
                    iw: self.lb(t, 4 * l + 3),
                    cand: if s == 2 { self.lb(t, base + 1) } else { LeafBytes::default() },
                    rows,
                })
            }

            fn verify(&self, seed: &Hash) -> Result<(), Fail> {
                verify_comp_sel_point(&self.tier, &self.ctx, &self.resp(), seed, &self.open(seed).unwrap())
            }
        }

        fn seeds() -> impl Iterator<Item = Hash> {
            (0..96u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
        }

        fn drawn_leaf(s: &Hash) -> u64 {
            crate::draw::point(s, SEL_POINT)[1] % 3
        }

        #[test]
        fn honest_selections_and_candidates_pass_and_round_trip() {
            let kit = build(Fault::None);
            let mut seen = [0; 3];
            for s in seeds() {
                let op = kit.open(&s).unwrap();
                assert_eq!(CompSelOpening::decode(&op.encode()), Some(op.clone()));
                kit.verify(&s).unwrap_or_else(|e| panic!("leaf {}: {e:?}", drawn_leaf(&s)));
                seen[drawn_leaf(&s) as usize] += 1;
            }
            assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
        }

        #[test]
        fn forged_selections_are_caught() {
            for (fault, leaf) in [(Fault::LowRows, 0), (Fault::LowBlocks, 1), (Fault::OutsideCandidates, 2)] {
                let kit = build(fault);
                let draws: Vec<Result<(), Fail>> = seeds().filter(|s| drawn_leaf(s) == leaf).map(|s| kit.verify(&s)).collect();
                let caught = draws.iter().filter(|r| **r == Err(Fail::Mismatch)).count();
                assert!(caught * 3 > draws.len(), "leaf {leaf}: {caught} of {}", draws.len());
                assert!(draws.iter().all(|r| r.is_ok() || *r == Err(Fail::Mismatch)));
            }
        }

        #[test]
        fn a_tampered_row_or_query_is_refused() {
            let kit = build(Fault::None);
            let s = seeds().next().unwrap();
            let mut op = kit.open(&s).unwrap();
            op.rows[0].bytes[3] ^= 1;
            assert_eq!(verify_comp_sel_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Trace));
            let mut op = kit.open(&s).unwrap();
            op.iq.bytes[0] ^= 1;
            assert_eq!(verify_comp_sel_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Trace));
            let mut op = kit.open(&s).unwrap();
            op.rows.pop();
            assert_eq!(verify_comp_sel_point(&kit.tier, &kit.ctx, &kit.resp(), &s, &op), Err(Fail::Shape));
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
