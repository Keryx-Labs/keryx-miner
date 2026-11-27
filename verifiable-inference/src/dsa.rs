//! Indexer selection of a sparse attention layer: the committed pools are a top-k of the indexer
//! scores, checked on drawn pools against the committed threshold.

use crate::chain::{ChainCtx, dsa_pools, n_router, sel_leaf, shape_any};
use crate::draw;
use crate::manifest::{self, Chain, DsaLinks};
use crate::merkle::{Hash, b3_fold, b3_verify};
use crate::opening::{LeafBytes, SelOpeningPoint};
use crate::verify::{Fail, ResponseCtx, TierCtx, open_entry};

/// Draw index of the selection point.
pub const SEL_POINT: u32 = crate::decision::DECISION_POINT + 1;
/// Pools drawn on each side of the threshold.
pub const SEL_DRAWS: usize = 5;
/// Rounding flips the tolerance absorbs per pool: a value by one step, or a block scale by one
/// half-precision step.
const FLIPS: usize = 16;
/// Relative change of a block scale flip.
const SCALE_REL: f64 = 1.0 / 1024.0;
/// Relative accumulation slack of the kernel.
const ACC_REL: f64 = 1e-5;

/// Positions of layer `layer` whose selection leaves out some visible pool, under `dsa`.
pub fn eligible(ctx: &ChainCtx, resp: &ResponseCtx, layer: u64, dsa: &DsaLinks) -> std::ops::Range<u64> {
    let n_pos = resp.n_pos as u64;
    let mut first = (dsa.n_top as u64 + 1) * dsa.kpool as u64 - 1;
    if ctx.trim_last && layer + 1 == ctx.n_layers as u64 {
        first = first.max((resp.n_prompt as u64).saturating_sub(1));
    }
    first.min(n_pos)..n_pos
}

/// Values of a block-quantized Q8_0 row as the cache stores them, and the step of each value.
fn q8_0_cache(v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut q, mut s) = (Vec::with_capacity(v.len()), Vec::with_capacity(v.len()));
    for b in v.chunks(32) {
        let amax = b.iter().fold(0f64, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d16 = crate::f16::round(d as f32) as f64;
        q.extend(b.iter().map(|&x| (x * id).round() * d16));
        s.extend(std::iter::repeat_n(d16, b.len()));
    }
    (q, s)
}

fn layer_norm(y: &[f32], w: &[f32], b: &[f32], eps: f64) -> Vec<f64> {
    let n = y.len() as f64;
    let mean = y.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = y.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let r = 1.0 / (var + eps).sqrt();
    y.iter().zip(w).zip(b).map(|((&x, &w), &b)| (x as f64 - mean) * r * w as f64 + b as f64).collect()
}

/// A rounding flip and the change it can cause on each value of the pooled key it reaches.
pub type Flip = Vec<(usize, f64)>;

/// Pooled key of a pool from the key and gate outputs of its members, as the cache holds it, and
/// the flips that can move it.
pub fn pooled_key(keys: &[&[f32]], gates: &[&[f32]], w: &[f32], b: &[f32], ape: &[f32], eps: f64) -> Option<(Vec<f64>, Vec<Flip>)> {
    let (kp, d) = (keys.len(), w.len());
    if kp < 2 || gates.len() != kp || b.len() != d || ape.len() != kp * d || keys.iter().chain(gates).any(|r| r.len() != d) {
        return None;
    }
    let ks: Vec<(Vec<f64>, Vec<f64>)> = keys.iter().map(|k| q8_0_cache(&layer_norm(k, w, b, eps))).collect();
    let gs: Vec<(Vec<f64>, Vec<f64>)> = gates.iter().map(|g| q8_0_cache(&g.iter().map(|&x| x as f64).collect::<Vec<_>>())).collect();
    let mut pooled = vec![0f64; d];
    // Sensitivity of each pooled value to member j's key and gate.
    let (mut dk, mut dg) = (vec![vec![0f64; d]; kp], vec![vec![0f64; d]; kp]);
    for i in 0..d {
        let logits: Vec<f64> = (0..kp).map(|j| gs[j].0[i] + ape[j * d + i] as f64).collect();
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = logits.iter().map(|&l| libm::exp(l - m)).collect();
        let z: f64 = e.iter().sum();
        pooled[i] = (0..kp).map(|j| e[j] / z * ks[j].0[i]).sum();
        for j in 0..kp {
            let p = e[j] / z;
            dk[j][i] = p;
            dg[j][i] = 1.5 * p * (1.0 - p) * (ks[j].0[i] - pooled[i]).abs();
        }
    }
    let (q, step) = q8_0_cache(&pooled);
    if !pooled.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut flips: Vec<Flip> = Vec::with_capacity((2 * kp + 1) * (d + d / 32));
    let mut add = |vals: &[f64], step: &[f64], sens: &[f64]| {
        flips.extend((0..d).map(|i| vec![(i, sens[i] * step[i])]));
        for b0 in (0..d).step_by(32) {
            flips.push((b0..(b0 + 32).min(d)).map(|i| (i, sens[i] * vals[i].abs() * SCALE_REL)).collect());
        }
    };
    for j in 0..kp {
        add(&ks[j].0, &ks[j].1, &dk[j]);
        add(&gs[j].0, &gs[j].1, &dg[j]);
    }
    add(&q, &step, &vec![1.0; d]);
    Some((q, flips))
}

/// Indexer score of a pooled key for the query heads `q` (`w.len()` heads end to end) and the
/// prescaled head weights `w`, and its tolerance given the `flips` of the key.
pub fn score(q: &[f32], w: &[f64], key: &[f64], flips: &[Flip]) -> Option<(f64, f64)> {
    let d = key.len();
    if d == 0 || q.len() != w.len() * d || flips.iter().flatten().any(|&(i, _)| i >= d) {
        return None;
    }
    let (mut s, mut mag) = (0f64, 0f64);
    let mut c = vec![0f64; d];
    for (h, &wh) in w.iter().enumerate() {
        let qh = &q[h * d..(h + 1) * d];
        let dot: f64 = qh.iter().zip(key).map(|(&a, &b)| a as f64 * b).sum();
        s += wh * dot.max(0.0);
        mag += wh.abs() * qh.iter().zip(key).map(|(&a, &b)| (a as f64 * b).abs()).sum::<f64>();
        c.iter_mut().zip(qh).for_each(|(c, &a)| *c += wh.abs() * (a as f64).abs());
    }
    let mut eff: Vec<f64> = flips.iter().map(|f| f.iter().map(|&(i, e)| c[i] * e).sum()).collect();
    let n = FLIPS.min(eff.len());
    if n > 0 {
        eff.select_nth_unstable_by(n - 1, |a, b| b.total_cmp(a));
    }
    let tol = eff[..n].iter().sum::<f64>() + ACC_REL * mag + f64::MIN_POSITIVE;
    (s.is_finite() && tol.is_finite()).then_some((s, tol))
}

/// Pools drawn by `seed` at position `t`: `SEL_DRAWS` among the chosen `pools`, then as many among
/// the visible pools left out.
pub fn drawn_pools(seed: &Hash, pools: &[u64], n_pool: u64) -> Vec<u64> {
    let n_out = n_pool - pools.len() as u64;
    let r = |i: usize| {
        let h = blake3::hash(&[seed.as_slice(), b"dsa-pool", &(i as u32).to_le_bytes()].concat());
        u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
    };
    let mut out: Vec<u64> = (0..SEL_DRAWS).map(|i| pools[(r(i) % pools.len() as u64) as usize]).collect();
    for i in 0..SEL_DRAWS {
        let mut p = r(SEL_DRAWS + i) % n_out;
        for &s in pools {
            if s <= p {
                p += 1;
            }
        }
        out.push(p);
    }
    out
}

/// Position root of position `t` reached by `leaf` at index `index`.
fn leaf_root(n_leaves: u64, index: u64, leaf: &LeafBytes) -> Result<Hash, Fail> {
    b3_fold(n_leaves, index, blake3::hash(&leaf.bytes).as_bytes(), &leaf.leaf_path).ok_or(Fail::Trace)
}

fn f32_leaf(b: &[u8], n: u64) -> Result<Vec<f32>, Fail> {
    (b.len() as u64 == 4 * n).then(|| crate::chain::f32s(b)).ok_or(Fail::Shape)
}

/// Verify the selection point of a response under `seed`.
pub fn verify_sel_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &SelOpeningPoint) -> Result<(), Fail> {
    let (nl, n_pos, n_attn) = (ctx.n_layers as u64, resp.n_pos as u64, ctx.n_attn as u64);
    if n_pos == 0 || nl == 0 || n_attn == 0 || ctx.n_sel == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = draw::point(seed, SEL_POINT);
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let le = open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?;
    let (a, dsa) = match le.chain {
        Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => (ll.attn.ok_or(Fail::Manifest)?, ll.dsa.ok_or(Fail::Manifest)?),
        _ => return Err(Fail::Manifest),
    };
    if a.slot as u64 != d[1] % n_attn || dsa.sel >= ctx.n_sel {
        return Err(Fail::Manifest);
    }
    let span = eligible(ctx, resp, layer, &dsa);
    if span.is_empty() {
        return Ok(());
    }
    let t = span.start + d[0] % (span.end - span.start);
    let [eq, ek, eg, ew] = <[_; 4]>::try_from(op.entries.clone()).map_err(|_| Fail::Shape)?;
    let [iq, ik, ig, iw] = [(eq, dsa.iq), (ek, dsa.ik), (eg, dsa.ig), (ew, dsa.w)].map(|(e, rank)| open_entry(tier, &e, rank as u64));
    let (iq, ik, ig, iw) = (iq?, ik?, ig?, iw?);
    if [&iq, &ik, &ig, &iw].iter().any(|e| e.role != manifest::ROLE_PRODUCT) {
        return Err(Fail::Manifest);
    }
    let (dk, nh) = (ik.m, iw.m);
    if dk == 0 || nh == 0 || ig.m != dk || iq.m != nh * dk {
        return Err(Fail::Manifest);
    }
    let [wn, wb, wa] = <[_; 3]>::try_from(op.norms.clone()).map_err(|_| Fail::Shape)?;
    let kp = dsa.kpool as u64;
    let k_norm = crate::chain::f32_weights(tier, &wn, dsa.k_norm, dk, 0, dk)?;
    let k_bias = crate::chain::f32_weights(tier, &wb, dsa.k_bias, dk, 0, dk)?;
    let ape = crate::chain::f32_weights(tier, &wa, dsa.ape, kp * dk, 0, kp * dk)?;
    let pshape = shape_any(tier, ctx);
    let nleaves = pshape.n_leaves;
    let sel_i = sel_leaf(tier.n_products, nl, n_router(ctx), dsa.sel as u64);
    let root_t = leaf_root(nleaves, sel_i, &op.sel)?;
    if leaf_root(nleaves, 2 * dsa.iq as u64 + 1, &op.iq)? != root_t || leaf_root(nleaves, 2 * dsa.w as u64 + 1, &op.w)? != root_t {
        return Err(Fail::Trace);
    }
    if !b3_verify(&op.pos_root, n_pos, t, &root_t, &op.pos_path) {
        return Err(Fail::Trace);
    }
    let (thr, pools) = dsa_pools(&dsa, t, &op.sel.bytes).ok_or(Fail::Mismatch)?;
    let thr = thr as f64;
    if !thr.is_finite() {
        return Err(Fail::Mismatch);
    }
    let q = f32_leaf(&op.iq.bytes, iq.m)?;
    let scale = 1.0 / ((dk * nh) as f64).sqrt();
    let w: Vec<f64> = f32_leaf(&op.w.bytes, nh)?.iter().map(|&x| x as f64 * scale).collect();
    let drawn = drawn_pools(seed, &pools, (t + 1) / kp);
    let range = match op.part {
        0 => 0..drawn.len(),
        crate::opening::SEL_PART_SELECTED => 0..SEL_DRAWS,
        crate::opening::SEL_PART_OUTSIDE => SEL_DRAWS..drawn.len(),
        _ => return Err(Fail::Shape),
    };
    if op.members.len() as u64 != range.len() as u64 * kp {
        return Err(Fail::Shape);
    }
    let eps = f32::from_bits(dsa.k_eps) as f64;
    for (i, (&p, ms)) in drawn[range.clone()].iter().zip(op.members.chunks_exact(kp as usize)).enumerate().map(|(j, x)| (j + range.start, x)) {
        let mut keys = Vec::with_capacity(kp as usize);
        let mut gates = Vec::with_capacity(kp as usize);
        for (j, m) in ms.iter().enumerate() {
            let r = leaf_root(nleaves, 2 * dsa.ik as u64 + 1, &m.k)?;
            if leaf_root(nleaves, 2 * dsa.ig as u64 + 1, &m.g)? != r || !b3_verify(&op.pos_root, n_pos, p * kp + j as u64, &r, &m.pos_path) {
                return Err(Fail::Trace);
            }
            keys.push(f32_leaf(&m.k.bytes, dk)?);
            gates.push(f32_leaf(&m.g.bytes, dk)?);
        }
        let (kr, gr): (Vec<&[f32]>, Vec<&[f32]>) = (keys.iter().map(|v| &v[..]).collect(), gates.iter().map(|v| &v[..]).collect());
        let (key, dev) = pooled_key(&kr, &gr, &k_norm, &k_bias, &ape, eps).ok_or(Fail::Mismatch)?;
        let (s, tol) = score(&q, &w, &key, &dev).ok_or(Fail::Mismatch)?;
        let ok = if i < SEL_DRAWS { s + tol >= thr } else { s - tol <= thr };
        if !ok {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
}
