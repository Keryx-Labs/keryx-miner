//! Stage-2 links between committed activations.

use crate::draw;
use crate::manifest::{self, Chain};
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_leaf_from_chunk, b3_verify};
use crate::opening::{ChunkRange, EntryOpening, ExpertIds, HcMoe, HcSource, LeafBytes, LeafChunk, LinkOpening, Point, ResidAdd, ResidOpening, ResidPoint, ResidSrc};
use crate::attn::Partial;
use crate::attn_blocks::{BlockTree, PrunedTree, block_params, check_leaf, check_node, check_root_lse, child_weights, decode_partials, lse, open_partial, partial_len, partials_root, root_lse_tol};
use crate::opening::{AttnOpening, PrefixPoint};
use crate::prefix::{PrefixCtx, THRESHOLD, expected_input, verify_prefix_point_at};
use crate::verify::{Fail, PosShape, Report, ResponseCtx, TierCtx, open_bytes, open_entry, open_row, verify_point_at};
use crate::{q8, quant};

/// Stage byte of a trace carrying the chain leaves.
pub const STAGE_CHAIN: u8 = 2;
/// Draw index of the residual point.
pub const RESID_POINT: u32 = draw::POINTS + 1;
/// Draw index of the attention point.
pub const ATTN_POINT: u32 = RESID_POINT + 1;

/// Consensus chain data of a tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainCtx {
    pub n_layers: u32,
    /// Manifest index of the layer-0 `ROLE_LAYER` entry; layer `l` is at `layer_base + l`.
    pub layer_base: u32,
    pub embd_entry: u32,
    pub eps: f64,
    pub embd_scale: f64,
    /// The last layer keeps only the output positions.
    pub trim_last: bool,
    /// Active experts per position of an expert product; 0 = no expert products.
    pub n_used: u32,
    /// Attention layers carrying key/value cache leaves; 0 = none.
    pub n_attn: u32,
    /// Recurrent layers carrying sequence leaves; 0 = none.
    pub n_recur: u32,
    /// Heads per layer in the sequence tree (the largest key/value or recurrent head count).
    pub seq_heads: u32,
    /// Selection leaves per position, one per attention layer with a full sparse indexer.
    pub n_sel: u32,
    /// Compressed row leaves of the sequence tree, one per layer producing compressed rows.
    pub n_comp: u32,
    /// Leaves per position of the streams after each n-gram lookup, after the selection leaves.
    pub n_engram: u32,
}
/// Relative tolerance of a residual addition, per element.
pub const EPS_RESID: f64 = 1e-6;

/// Residual leaves of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resid {
    Attn,
    Out,
}

/// Leaves of a stage-2 position: `X`/`Y` per product, two residuals per layer, `n_router` router
/// leaves, then the meta leaf.
pub fn n_pos_leaves(n_products: u64, n_layers: u64, n_router: u64) -> u64 {
    2 * n_products + 2 * n_layers + n_router + 1
}

/// Leaf index of the router scores of `layer`.
pub fn router_leaf(n_products: u64, n_layers: u64, layer: u64) -> u64 {
    2 * n_products + 2 * n_layers + layer
}

/// Leaf index of a residual of `layer`.
pub fn resid_leaf(n_products: u64, layer: u64, kind: Resid) -> u64 {
    2 * n_products + 2 * layer + if kind == Resid::Attn { 0 } else { 1 }
}

/// Leaf index of the meta leaf.
pub fn meta_leaf(n_products: u64, n_layers: u64, n_router: u64) -> u64 {
    2 * n_products + 2 * n_layers + n_router
}

/// Router leaves of a position.
pub fn n_router(ctx: &ChainCtx) -> u64 {
    if ctx.n_used > 0 { ctx.n_layers as u64 } else { 0 }
}

/// Meta leaf bytes: position, batch size, token consumed at the position.
pub fn meta(t: u32, n_ubatch: u32, tok: u32) -> [u8; 12] {
    let mut m = [0u8; 12];
    m[..4].copy_from_slice(&t.to_le_bytes());
    m[4..8].copy_from_slice(&n_ubatch.to_le_bytes());
    m[8..].copy_from_slice(&tok.to_le_bytes());
    m
}

/// Blocks of 32 values covered by a norm weight fragment.
pub const NORM_FRAGMENT_BLOCKS: u64 = 8;
/// Blocks of 32 `f32` values in a leaf chunk.
pub const CHUNK_BLOCKS: u64 = (B3_CHUNK / 128) as u64;
const FRAGMENT_DRAW: u32 = 64;

/// First block and block count of the norm weight fragment covering `block` over `k` values.
pub fn norm_fragment(block: u64, k: u64) -> (u64, u64) {
    let blocks = k.div_ceil(32);
    let n = NORM_FRAGMENT_BLOCKS.min(blocks);
    (block.min(blocks - n), n)
}

/// Block of the product input `x` (`k` values in `layout`) that the link of point `i` checks:
/// drawn uniformly, or on even odds by block scale.
pub fn link_block(seed: &Hash, i: u32, x: &[u8], layout: u8, k: u64) -> Option<u64> {
    if !k.is_multiple_of(32) {
        return None;
    }
    let blocks = q8::blocks(x, layout, k as usize)?;
    let n = blocks.len() as u64;
    if n == 0 {
        return None;
    }
    let r = draw::point(seed, FRAGMENT_DRAW + i);
    if r[1] & 1 == 0 {
        return Some(r[0] % n);
    }
    let peak = blocks.iter().fold(0f32, |m, (d, _)| m.max(d.abs()));
    if !peak.is_finite() {
        return None;
    }
    let w: Vec<u64> = blocks.iter().map(|(d, _)| 1 + if peak > 0.0 { (d.abs() as f64 / peak as f64 * 65535.0) as u64 } else { 0 }).collect();
    let mut target = r[2] % w.iter().sum::<u64>();
    for (b, &wb) in w.iter().enumerate() {
        if target < wb {
            return Some(b as u64);
        }
        target -= wb;
    }
    None
}

/// Largest block scale of an unlinked product input over its median block scale.
const PEAK_OVER_MEDIAN: f32 = 1024.0;

/// Whether no block scale of slot `slot` of `x` (rows of `k` values; a single row serves every
/// slot) exceeds [`PEAK_OVER_MEDIAN`] times the median block scale of that slot.
pub fn peak_bounded(x: &[u8], layout: u8, k: u64, slot: u64) -> bool {
    let x_one = crate::verify::x_len(k, layout) as usize;
    if !k.is_multiple_of(32) || x_one == 0 || x.is_empty() || !x.len().is_multiple_of(x_one) {
        return false;
    }
    let slot = if x.len() == x_one { 0 } else { slot as usize };
    let Some(row) = x.chunks_exact(x_one).nth(slot) else { return false };
    let Some(blocks) = q8::blocks(row, layout, k as usize) else { return false };
    let mut d: Vec<f32> = blocks.iter().map(|(b, _)| b.abs()).collect();
    if d.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let peak = d.iter().fold(0f32, |m, &v| m.max(v));
    let mid = d.len() / 2;
    let median = *d.select_nth_unstable_by(mid, f32::total_cmp).1;
    median == 0.0 || peak <= PEAK_OVER_MEDIAN * median
}

/// `v / rms(v)` in f64.
/// RMS range of a whole residual vector past the embedding.
pub const RESID_RMS_MIN: f64 = 1e-3;
pub const RESID_RMS_MAX: f64 = 1024.0;
/// Lower RMS bound of one chunk of a residual vector.
pub const RESID_CHUNK_RMS_MIN: f64 = 1e-4;

/// RMS of `v`, in f64.
fn rms(v: &[f32]) -> f64 {
    libm::sqrt(v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / v.len().max(1) as f64)
}

pub fn rms_scaled(v: &[f32], eps: f64, scale: f64) -> Vec<f64> {
    let e: Vec<f64> = v.iter().map(|&a| a as f64 * scale).collect();
    let inv = 1.0 / libm::sqrt(e.iter().map(|a| a * a).sum::<f64>() / e.len() as f64 + eps);
    e.iter().map(|a| a * inv).collect()
}

/// Checks the input blocks `[b0, b0 + n)` against `rms_scaled(src)` times the weight fragment `w`.
fn fragment_link(x: &[u8], layout: u8, k: usize, scaled: &[f64], w: &[f32], b0: u64, n: u64) -> Result<(), Fail> {
    let (lo, hi) = ((b0 * 32) as usize, (((b0 + n) * 32) as usize).min(k));
    if scaled.len() != k || w.len() != hi - lo {
        return Err(Fail::Shape);
    }
    let v: Vec<f64> = scaled[lo..hi].iter().zip(w).map(|(a, &b)| a * b as f64).collect();
    within(block_deviation(x, layout, k, b0 as usize, &v))
}

/// `v * w / rms(v)` in f64.
pub fn rms_norm(v: &[f32], w: &[f32], eps: f64) -> Vec<f64> {
    expected_input(v, w, eps, 1.0)
}

/// `max |q - v / d|` over the blocks of `x` starting at block `first_block`, against `v`.
pub fn block_deviation(x: &[u8], layout: u8, k: usize, first_block: usize, v: &[f64]) -> Option<f64> {
    let blocks = q8::blocks(x, layout, k)?;
    let n = v.len().div_ceil(32);
    let vmax = v.iter().fold(0f64, |m, a| m.max(a.abs()));
    let mut worst = 0f64;
    for (b, (d, q)) in blocks.get(first_block..first_block.checked_add(n)?)?.iter().enumerate() {
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

/// Worst deviation of `v` against the Q8 blocks of `x` from `first_block`, and the most values of
/// one block past `base`.
pub fn block_outliers(x: &[u8], layout: u8, k: usize, first_block: usize, v: &[f64], base: f64) -> Option<(f64, usize)> {
    let blocks = q8::blocks(x, layout, k)?;
    let n = v.len().div_ceil(32);
    let vmax = v.iter().fold(0f64, |m, a| m.max(a.abs()));
    let (mut worst, mut spare) = (0f64, 0usize);
    for (b, (d, q)) in blocks.get(first_block..first_block.checked_add(n)?)?.iter().enumerate() {
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
        let mut over = 0;
        for (j, a) in vals.iter().enumerate() {
            let dev = (q[j] as f64 - a / *d as f64).abs();
            worst = worst.max(dev);
            over += (dev > base) as usize;
        }
        spare = spare.max(over);
    }
    Some((worst, spare))
}

fn within(dev: Option<f64>) -> Result<(), Fail> {
    match dev {
        Some(d) if d <= THRESHOLD => Ok(()),
        _ => Err(Fail::Mismatch),
    }
}

/// A product input normed from a committed residual: `x = Q8(rms_norm(src) * w)`.
pub fn norm_link(x: &[u8], layout: u8, src: &[f32], w: &[f32], eps: f64) -> Result<(), Fail> {
    if src.len() != w.len() || src.is_empty() {
        return Err(Fail::Shape);
    }
    within(block_deviation(x, layout, src.len(), 0, &rms_norm(src, w, eps)))
}

/// A product input taken from the committed GLU output `h`, for the blocks covering `h` from
/// block `first_block`.
pub fn glu_link(x: &[u8], layout: u8, k: usize, first_block: usize, h: &[f32]) -> Result<(), Fail> {
    if h.is_empty() {
        return Err(Fail::Shape);
    }
    let v: Vec<f64> = h.iter().map(|&a| a as f64).collect();
    within(block_deviation(x, layout, k, first_block, &v))
}

/// A residual addition, element by element: `dst = scale * (add + src)`.
pub fn residual_link(dst: &[f32], add: &[f64], src: &[f32], scale: f64) -> Result<(), Fail> {
    if dst.is_empty() || dst.len() != add.len() || dst.len() != src.len() {
        return Err(Fail::Shape);
    }
    for ((&d, &a), &s) in dst.iter().zip(add).zip(src) {
        let s = s as f64;
        let tol = EPS_RESID * scale.abs() * (a.abs() + s.abs()) + f32::MIN_POSITIVE as f64;
        if !tol.is_finite() || !(d as f64 - scale * (a + s)).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
}

/// Leaf index of selection `i` of a position, after the router leaves.
pub const fn sel_leaf(n_products: u64, n_layers: u64, n_router: u64, i: u64) -> u64 {
    2 * n_products + 2 * n_layers + n_router + i
}

/// Leaf layout of a stage-2 position consuming `tok`.
pub fn shape(tier: &TierCtx, ctx: &ChainCtx, tok: u32) -> PosShape {
    let n_layers = ctx.n_layers as u64;
    let n_sel = ctx.n_sel as u64 + ctx.n_engram as u64;
    PosShape {
        n_leaves: n_pos_leaves(tier.n_products, n_layers, n_router(ctx)) + n_sel,
        meta_index: meta_leaf(tier.n_products, n_layers, n_router(ctx)) + n_sel,
        tok: Some(tok),
        n_used: ctx.n_used,
    }
}

pub(crate) fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// [`f32s`] of committed activations, which must all be finite.
pub(crate) fn finite_f32s(b: &[u8]) -> Result<Vec<f32>, Fail> {
    let v = f32s(b);
    if v.iter().all(|x| x.is_finite()) { Ok(v) } else { Err(Fail::Mismatch) }
}

pub(crate) fn f32_weights(tier: &TierCtx, w: &(EntryOpening, ChunkRange), index: u32, k: u64, offset: u64, n: u64) -> Result<Vec<f32>, Fail> {
    let e = open_entry(tier, &w.0, index as u64)?;
    if e.role != manifest::ROLE_NORM || e.wtype != 0 || e.k != k {
        return Err(Fail::Manifest);
    }
    Ok(f32s(&open_bytes(tier, &e, &w.1, 4 * offset, 4 * n)?))
}

fn embedding(tier: &TierCtx, ctx: &ChainCtx, w: &(EntryOpening, ChunkRange), tok: u32, k: u64) -> Result<Vec<f32>, Fail> {
    let e = open_entry(tier, &w.0, ctx.embd_entry as u64)?;
    if e.role != manifest::ROLE_EMBD || e.k != k || e.n_exp != 1 {
        return Err(Fail::Manifest);
    }
    let row = open_row(tier, &e, &w.1, tok as u64)?;
    quant::dequant_row(e.wtype, &row, k as usize).ok_or(Fail::Unsupported)
}

fn whole_leaf(pos_root: &Hash, shape: &PosShape, index: u64, l: &LeafBytes, len: u64) -> Result<Vec<f32>, Fail> {
    if l.bytes.len() as u64 != len {
        return Err(Fail::Shape);
    }
    if !b3_verify(pos_root, shape.n_leaves, index, blake3::hash(&l.bytes).as_bytes(), &l.leaf_path) {
        return Err(Fail::Trace);
    }
    finite_f32s(&l.bytes)
}

fn chunk_leaf_raw(c: &ExpertIds, total_len: u64, index: u64) -> Result<Hash, Fail> {
    b3_leaf_from_chunk(total_len, index, &c.chunk, &c.chunk_path).ok_or(Fail::Trace)
}

pub(crate) fn chunk_leaf(c: &LeafChunk, total_len: u64, index: u64) -> Result<Hash, Fail> {
    if c.total_len as u64 != total_len || c.index as u64 != index {
        return Err(Fail::Shape);
    }
    b3_leaf_from_chunk(total_len, index, &c.chunk, &c.chunk_path).ok_or(Fail::Trace)
}

pub(crate) fn leaf_chunk(pos_root: &Hash, shape: &PosShape, leaf: u64, c: &LeafChunk, total_len: u64, index: u64) -> Result<Vec<f32>, Fail> {
    let h = chunk_leaf(c, total_len, index)?;
    if !b3_verify(pos_root, shape.n_leaves, leaf, &h, &c.leaf_path) {
        return Err(Fail::Trace);
    }
    finite_f32s(&c.chunk)
}

fn n_chunks(total_len: u64) -> u64 {
    total_len.div_ceil(B3_CHUNK as u64).max(1)
}

/// Relative tolerance of a residual built from a hyper-connection mix.
pub const EPS_HC: f64 = 1e-5;

/// Weights of a hyper-connection mix: `pre[i]`, `post[i]`, `comb[dst + n * src]`.
#[derive(Clone, Debug, PartialEq)]
pub struct HcMix {
    pub pre: Vec<f64>,
    pub post: Vec<f64>,
    pub comb: Vec<f64>,
}

/// The mix of `n` streams from the mix product output `mixes`, its 3 scales and its biases:
/// `pre = sigmoid + eps`, `post = 2 sigmoid`, `comb` = row softmax then Sinkhorn, as the graph.
pub fn hc_mix(mixes: &[f32], scale: &[f32], base: &[f32], n: usize, iters: u32, eps: f64) -> Option<HcMix> {
    if n == 0 || mixes.len() != (2 + n) * n || base.len() != mixes.len() || scale.len() != 3 || iters == 0 {
        return None;
    }
    let sig = |x: f64| 1.0 / (1.0 + libm::exp(-x));
    let at = |i: usize, s: f32| mixes[i] as f64 * s as f64 + base[i] as f64;
    let pre: Vec<f64> = (0..n).map(|i| sig(at(i, scale[0])) + eps).collect();
    let post: Vec<f64> = (0..n).map(|i| 2.0 * sig(at(n + i, scale[1]))).collect();
    let mut c: Vec<f64> = (0..n * n).map(|k| at(2 * n + k, scale[2])).collect();
    for src in 0..n {
        let col = &mut c[n * src..n * (src + 1)];
        let top = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let z: f64 = col.iter().map(|v| libm::exp(v - top)).sum();
        for v in col.iter_mut() {
            *v = libm::exp(*v - top) / z + eps;
        }
    }
    let by_dst = |c: &mut Vec<f64>| {
        for dst in 0..n {
            let sum: f64 = (0..n).map(|src| c[dst + n * src]).sum::<f64>() + eps;
            for src in 0..n {
                c[dst + n * src] /= sum;
            }
        }
    };
    let by_src = |c: &mut Vec<f64>| {
        for src in 0..n {
            let sum: f64 = (0..n).map(|dst| c[dst + n * src]).sum::<f64>() + eps;
            for dst in 0..n {
                c[dst + n * src] /= sum;
            }
        }
    };
    by_dst(&mut c);
    for _ in 1..iters {
        by_src(&mut c);
        by_dst(&mut c);
    }
    let all = pre.iter().chain(&post).chain(&c);
    if all.clone().any(|v| !v.is_finite()) {
        return None;
    }
    Some(HcMix { pre, post, comb: c })
}

/// The hyper-connection links of `layer` and the mix weights of its half `kind`.
fn hc_half(tier: &TierCtx, ctx: &ChainCtx, layer: u64, kind: Resid, le: &EntryOpening) -> Result<(manifest::HcLinks, u32, [u32; 3], u32, Option<u64>), Fail> {
    let e = open_entry(tier, le, ctx.layer_base as u64 + layer)?;
    let (hc, ll) = match e.chain {
        Chain::Layer(ll) if e.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => (ll.hc.ok_or(Fail::Manifest)?, ll),
        _ => return Err(Fail::Manifest),
    };
    let src = attn_source(tier, ctx, &ll)?;
    Ok(match kind {
        Resid::Attn => (hc, hc.attn_fn, hc.attn_scale, hc.attn_base, src),
        Resid::Out => (hc, hc.ffn_fn, hc.ffn_scale, hc.ffn_base, src),
    })
}

/// Leaf of the streams after the n-gram lookup `e` of a position.
pub const fn engram_leaf(n_products: u64, n_layers: u64, n_router: u64, n_sel: u64, e: u64) -> u64 {
    sel_leaf(n_products, n_layers, n_router, n_sel + e)
}

/// Leaf of the streams the attention half of layer `ll` reads: after its n-gram lookup, else the
/// output of the layer before (`None` at layer 0: the repeated embedding).
pub(crate) fn attn_source(tier: &TierCtx, ctx: &ChainCtx, ll: &manifest::LayerLinks) -> Result<Option<u64>, Fail> {
    let (np, nl) = (tier.n_products, ctx.n_layers as u64);
    match ll.engram {
        Some(e) => {
            let leaf = engram_leaf(np, nl, n_router(ctx), ctx.n_sel as u64, e.module as u64);
            (e.module < ctx.n_engram && e.out_leaf as u64 == leaf).then_some(Some(leaf)).ok_or(Fail::Manifest)
        }
        None if ll.layer == 0 => Ok(None),
        None => Ok(Some(resid_leaf(np, ll.layer as u64 - 1, Resid::Out))),
    }
}

/// The mix of the half whose mix product is `fn_rank`, from its committed output at the position.
#[allow(clippy::too_many_arguments)]
fn hc_mix_at(
    tier: &TierCtx,
    pos_root: &Hash,
    shape: &PosShape,
    hc: &manifest::HcLinks,
    fn_rank: u32,
    mixes: &LeafBytes,
    scale: [u32; 3],
    base: (&(EntryOpening, ChunkRange), u32),
) -> Result<HcMix, Fail> {
    let n = hc.n_hc as u64;
    let m = (2 + n) * n;
    let y = whole_leaf(pos_root, shape, 2 * fn_rank as u64 + 1, mixes, 4 * m)?;
    let s = scale.map(f32::from_bits);
    let b = f32_weights(tier, base.0, base.1, m, 0, m)?;
    hc_mix(&y, &s, &b, n as usize, hc.sinkhorn_iters, f32::from_bits(hc.eps) as f64).ok_or(Fail::Mismatch)
}

/// The `n` streams of a hyper-connection source as one flat vector: committed, or the token
/// embedding repeated.
fn hc_streams(tier: &TierCtx, ctx: &ChainCtx, pos_root: &Hash, shape: &PosShape, src: &HcSource, leaf: Option<u64>, n: u64, dim: u64, tok: u32) -> Result<Vec<f32>, Fail> {
    match (src, leaf) {
        (HcSource::Streams(b), Some(l)) => whole_leaf(pos_root, shape, l, b, 4 * n * dim),
        (HcSource::Token(w), None) => {
            let row: Vec<f32> = embedding(tier, ctx, w, tok, dim)?.iter().map(|&v| (v as f64 * ctx.embd_scale) as f32).collect();
            Ok((0..n).flat_map(|_| row.iter().copied()).collect())
        }
        _ => Err(Fail::Shape),
    }
}

/// Verify point `i` of a stage-2 opening and the link of its input.
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_point(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    i: u32,
    p: &Point,
    tok: u32,
    link: &LinkOpening,
    ids: Option<&ExpertIds>,
) -> Result<Report, Fail> {
    verify_chain_point_half(tier, ctx, resp, seed, i, p, tok, link, ids, crate::opening::HALF_WHOLE)
}

/// Verify `half` of point `i` of a stage-2 opening: the whole point, its weight rows alone
/// (`link` empty), or the link of its input alone (`p` without weight rows).
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_point_half(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    i: u32,
    p: &Point,
    tok: u32,
    link: &LinkOpening,
    ids: Option<&ExpertIds>,
    half: u8,
) -> Result<Report, Fail> {
    use crate::opening::{HALF_LINK, HALF_PRODUCT, HALF_WHOLE};
    let shape = shape(tier, ctx, tok);
    let (report, o) = match half {
        HALF_WHOLE | HALF_PRODUCT => verify_point_at(tier, resp, seed, i, p, &shape, ids)?,
        HALF_LINK => crate::verify::verify_point_input(tier, resp, seed, i, p, &shape, ids)?,
        _ => return Err(Fail::Shape),
    };
    let Chain::Product(pl) = o.input.chain else {
        return Err(Fail::Manifest);
    };
    let (k, np, n_layers) = (o.input.k, tier.n_products, ctx.n_layers as u64);
    let layer = pl.layer as u64;
    if pl.link != manifest::LINK_HEAD && pl.link != manifest::LINK_HC_HEAD && layer >= n_layers {
        return Err(Fail::Manifest);
    }
    if half == HALF_PRODUCT {
        return if matches!(link, LinkOpening::None) { Ok(report) } else { Err(Fail::Shape) };
    }
    match (pl.link, link) {
        (manifest::LINK_NONE, LinkOpening::None) => {
            if !peak_bounded(&p.x, o.layout, k, o.slot) {
                return Err(Fail::Mismatch);
            }
        }
        (manifest::LINK_ATTN_IN, LinkOpening::Token { embd, norm }) if layer == 0 => {
            let (b0, n) = norm_fragment(link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)?, k);
            let w = f32_weights(tier, norm, pl.norm, k, b0 * 32, (n * 32).min(k - b0 * 32))?;
            let scaled = rms_scaled(&embedding(tier, ctx, embd, tok, k)?, ctx.eps, ctx.embd_scale);
            fragment_link(&p.x, o.layout, k as usize, &scaled, &w, b0, n)?;
        }
        (manifest::LINK_ATTN_IN | manifest::LINK_FFN_IN | manifest::LINK_HEAD, LinkOpening::Norm { src, norm }) => {
            let leaf = match pl.link {
                manifest::LINK_ATTN_IN if layer > 0 => resid_leaf(np, layer - 1, Resid::Out),
                manifest::LINK_FFN_IN => resid_leaf(np, layer, Resid::Attn),
                manifest::LINK_HEAD if n_layers > 0 => resid_leaf(np, n_layers - 1, Resid::Out),
                _ => return Err(Fail::Shape),
            };
            let h = whole_leaf(&o.pos_root, &shape, leaf, src, 4 * k)?;
            if !(RESID_RMS_MIN..=RESID_RMS_MAX).contains(&rms(&h)) {
                return Err(Fail::Mismatch);
            }
            let x_one = crate::verify::x_len(k, o.layout) as usize;
            let xs = if p.x.len() == x_one { &p.x[..] } else { p.x.get(o.slot as usize * x_one..(o.slot as usize + 1) * x_one).ok_or(Fail::Shape)? };
            let (b0, n) = norm_fragment(link_block(seed, i, xs, o.layout, k).ok_or(Fail::Shape)?, k);
            let w = f32_weights(tier, norm, pl.norm, k, b0 * 32, (n * 32).min(k - b0 * 32))?;
            fragment_link(xs, o.layout, k as usize, &rms_scaled(&h, ctx.eps, 1.0), &w, b0, n)?;
        }
        (manifest::LINK_ENGRAM, LinkOpening::Engram { layer: le, prev, map, row }) if o.input.n_exp == 1 => {
            let e = match open_entry(tier, le, ctx.layer_base as u64 + layer)?.chain {
                Chain::Layer(ll) if ll.layer as u64 == layer => ll.engram.ok_or(Fail::Manifest)?,
                _ => return Err(Fail::Manifest),
            };
            let (n_gram, d) = (e.n_gram as u64, e.head_dim as u64);
            if report.rank != e.kv as u64 || k != e.n_cols() as u64 * d || !d.is_multiple_of(32) {
                return Err(Fail::Manifest);
            }
            let t = report.t;
            let shifts = t.min(n_gram - 1);
            if prev.len() as u64 != shifts || map.len() as u64 != 1 + shifts {
                return Err(Fail::Shape);
            }
            // the tokens back from t, each bound to its position by the meta leaf
            let mut toks = vec![tok];
            for (s, pt) in prev.iter().enumerate() {
                let q = t - 1 - s as u64;
                let meta = shape_any(tier, ctx);
                let h = PosShape { tok: Some(pt.tok), ..meta }.meta_hash(q, crate::verify::expected_ubatch(q, resp.n_prompt));
                let root = b3_fold(meta.n_leaves, meta.meta_index, &h, &pt.meta_path).ok_or(Fail::Trace)?;
                if !b3_verify(&resp.trace_root, resp.n_pos as u64, q, &root, &pt.pos_path) {
                    return Err(Fail::Trace);
                }
                toks.push(pt.tok);
            }
            let mut ids = toks.iter().zip(map).map(|(&tk, m)| crate::engram::mapped_id(&e, tk, m).ok_or(Fail::Trace)).collect::<Result<Vec<_>, _>>()?;
            ids.resize(n_gram as usize, None);
            let rows = crate::engram::engram_rows(&e, &ids).ok_or(Fail::Manifest)?;
            let block = link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)?;
            let col = (block * 32 / d) as usize;
            let vals = crate::engram::table_row(&e, rows[col], row).ok_or(Fail::Weights)?;
            glu_link(&p.x, o.layout, k as usize, col * (d as usize / 32), &vals)?;
        }
        (manifest::LINK_HC_HEAD, LinkOpening::Norm { src, norm }) if n_layers > 0 && o.input.n_exp == 1 => {
            let n_hc = pl.src as u64;
            if !(2..=16).contains(&n_hc) {
                return Err(Fail::Manifest);
            }
            let r = whole_leaf(&o.pos_root, &shape, resid_leaf(np, n_layers - 1, Resid::Out), src, 4 * n_hc * k)?;
            let ku = k as usize;
            let h: Vec<f32> = (0..ku).map(|j| ((0..n_hc as usize).map(|s| r[s * ku + j] as f64).sum::<f64>() / n_hc as f64) as f32).collect();
            if !(RESID_RMS_MIN..=RESID_RMS_MAX).contains(&rms(&h)) {
                return Err(Fail::Mismatch);
            }
            let (b0, n) = norm_fragment(link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)?, k);
            let w = f32_weights(tier, norm, pl.norm, k, b0 * 32, (n * 32).min(k - b0 * 32))?;
            fragment_link(&p.x, o.layout, k as usize, &rms_scaled(&h, ctx.eps, 1.0), &w, b0, n)?;
        }
        (manifest::LINK_NORMED, LinkOpening::Norm { src, norm }) if o.input.n_exp == 1 && pl.src < tier.n_products as u32 => {
            let h = whole_leaf(&o.pos_root, &shape, 2 * pl.src as u64 + 1, src, 4 * k)?;
            let (b0, n) = norm_fragment(link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)?, k);
            let w = f32_weights(tier, norm, pl.norm, k, b0 * 32, (n * 32).min(k - b0 * 32))?;
            fragment_link(&p.x, o.layout, k as usize, &rms_scaled(&h, ctx.eps, 1.0), &w, b0, n)?;
        }
        (manifest::LINK_GLU, LinkOpening::Source { src, y, ids: Some(ids) }) if o.input.n_exp > 1 => {
            let se = open_entry(tier, src, pl.src as u64)?;
            if se.glu_form != manifest::FORM_PAIR_GATE || se.n_exp != o.input.n_exp || se.m != k {
                return Err(Fail::Manifest);
            }
            let n_used = ctx.n_used as u64;
            let body = n_used * k * 4;
            let per_slot = k * 4 / B3_CHUNK as u64;
            if per_slot == 0 || (k * 4) % B3_CHUNK as u64 != 0 {
                return Err(Fail::Shape);
            }
            let total = body + 4 * n_used;
            let gate_y = 2 * pl.src as u64 + 1;
            let ids_leaf = chunk_leaf_raw(ids, total, body / B3_CHUNK as u64)?;
            if !b3_verify(&o.pos_root, shape.n_leaves, gate_y, &ids_leaf, &y.leaf_path) {
                return Err(Fail::Trace);
            }
            let s = 4 * o.slot as usize;
            let id = ids.chunk.get(s..s + 4).ok_or(Fail::Shape)?;
            if u32::from_le_bytes(id.try_into().unwrap()) as u64 != o.expert {
                return Err(Fail::Mismatch);
            }
            let x_one = crate::verify::x_len(k, o.layout) as usize;
            let xs = p.x.get(o.slot as usize * x_one..(o.slot as usize + 1) * x_one).ok_or(Fail::Shape)?;
            let c = o.slot * per_slot + link_block(seed, i, xs, o.layout, k).ok_or(Fail::Shape)? / CHUNK_BLOCKS;
            let h = leaf_chunk(&o.pos_root, &shape, gate_y, y, total, c)?;
            glu_link(xs, o.layout, k as usize, ((c - o.slot * per_slot) * B3_CHUNK as u64 / 128) as usize, &h)?;
        }
        (manifest::LINK_GLU | manifest::LINK_DIRECT, LinkOpening::Source { src, y, ids: None }) if o.input.n_exp == 1 => {
            let se = open_entry(tier, src, pl.src as u64)?;
            let m_out = match (pl.link, se.glu_form) {
                (manifest::LINK_GLU, manifest::FORM_PAIR_GATE) => se.m,
                (manifest::LINK_GLU, manifest::FORM_SPLIT) => se.m / 2,
                (manifest::LINK_DIRECT, manifest::FORM_NONE) if se.role == manifest::ROLE_PRODUCT => se.m,
                (manifest::LINK_DIRECT, manifest::FORM_NONE) if se.role == manifest::ROLE_HEADS => se.m * se.n_exp as u64,
                _ => return Err(Fail::Manifest),
            };
            if m_out != k || (se.n_exp != 1 && se.role != manifest::ROLE_HEADS) {
                return Err(Fail::Manifest);
            }
            let total = 4 * m_out;
            let c = link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)? / CHUNK_BLOCKS;
            let h = leaf_chunk(&o.pos_root, &shape, 2 * pl.src as u64 + 1, y, total, c)?;
            glu_link(&p.x, o.layout, k as usize, (c * B3_CHUNK as u64 / 128) as usize, &h)?;
        }
        (manifest::LINK_HEAD_SLICE, LinkOpening::Source { src, y, ids: None }) if o.input.role == manifest::ROLE_HEADS => {
            let se = open_entry(tier, src, pl.src as u64)?;
            let n_head = o.input.n_exp as u64;
            if se.role != manifest::ROLE_PRODUCT || se.n_exp != 1 || n_head < 2 || se.m % n_head != 0 {
                return Err(Fail::Manifest);
            }
            let stride = se.m / n_head;
            if k > stride || !k.is_multiple_of(32) || !stride.is_multiple_of(32) {
                return Err(Fail::Manifest);
            }
            let per_chunk = (B3_CHUNK / 4) as u64;
            let start = o.slot * stride;
            let x_one = crate::verify::x_len(k, o.layout) as usize;
            let xs = p.x.get(o.slot as usize * x_one..(o.slot as usize + 1) * x_one).ok_or(Fail::Shape)?;
            let c = (start + link_block(seed, i, xs, o.layout, k).ok_or(Fail::Shape)? * 32) / per_chunk;
            let h = leaf_chunk(&o.pos_root, &shape, 2 * pl.src as u64 + 1, y, 4 * se.m, c)?;
            let (lo, hi) = ((c * per_chunk).max(start), ((c + 1) * per_chunk).min(start + k));
            let vals = h.get((lo - c * per_chunk) as usize..(hi - c * per_chunk) as usize).ok_or(Fail::Shape)?;
            glu_link(xs, o.layout, k as usize, ((lo - start) / 32) as usize, vals)?;
        }
        (manifest::LINK_HC_ATTN_IN | manifest::LINK_HC_FFN_IN, LinkOpening::HcIn { layer: le, src })
        | (manifest::LINK_HC_ATTN_PRE | manifest::LINK_HC_FFN_PRE, LinkOpening::HcPre { layer: le, src, .. }) => {
            let attn = matches!(pl.link, manifest::LINK_HC_ATTN_IN | manifest::LINK_HC_ATTN_PRE);
            let half = if attn { Resid::Attn } else { Resid::Out };
            let (hc, fn_rank, si, bi, attn_src) = hc_half(tier, ctx, layer, half, le)?;
            let leaf = if attn { attn_src } else { Some(resid_leaf(np, layer, Resid::Attn)) };
            let nh = hc.n_hc as u64;
            let (b0, n) = norm_fragment(link_block(seed, i, &p.x, o.layout, k).ok_or(Fail::Shape)?, k);
            let len = (n * 32).min(k - b0 * 32);
            let x: Vec<f32> = match link {
                LinkOpening::HcIn { .. } => {
                    if pl.norm != manifest::NONE || pl.src != manifest::NONE || nh == 0 || !k.is_multiple_of(nh) || report.rank != fn_rank as u64 {
                        return Err(Fail::Manifest);
                    }
                    hc_streams(tier, ctx, &o.pos_root, &shape, src, leaf, nh, k / nh, tok)?
                }
                LinkOpening::HcPre { mixes, base, .. } => {
                    let r = hc_streams(tier, ctx, &o.pos_root, &shape, src, leaf, nh, k, tok)?;
                    let k = k as usize;
                    match hc.pre_source(attn) {
                        // the first single-pass attention input is stream 0
                        None => {
                            if pl.src != manifest::NONE || !mixes.bytes.is_empty() || *base != Default::default() {
                                return Err(Fail::Shape);
                            }
                            r[..k].to_vec()
                        }
                        Some((src_fn, src_scale, src_base)) => {
                            if pl.src != src_fn || (hc.lag == 0 && (src_fn != fn_rank || src_scale != si || src_base != bi)) {
                                return Err(Fail::Manifest);
                            }
                            let mix = hc_mix_at(tier, &o.pos_root, &shape, &hc, src_fn, mixes, src_scale, (base, src_base))?;
                            (0..k).map(|j| (0..nh as usize).map(|s| mix.pre[s] * r[s * k + j] as f64).sum::<f64>() as f32).collect()
                        }
                    }
                }
                _ => unreachable!(),
            };
            if leaf.is_some() && !(RESID_RMS_MIN..=RESID_RMS_MAX).contains(&rms(&x)) {
                return Err(Fail::Mismatch);
            }
            let w = match link {
                LinkOpening::HcPre { norm, .. } => f32_weights(tier, norm, pl.norm, k, b0 * 32, len)?,
                _ => vec![1f32; len as usize],
            };
            fragment_link(&p.x, o.layout, k as usize, &rms_scaled(&x, ctx.eps, 1.0), &w, b0, n)?;
        }
        _ => return Err(Fail::Shape),
    }
    Ok(report)
}

/// Verify the prefix point of a stage-2 opening.
pub fn verify_chain_prefix_point(
    tier: &TierCtx,
    ctx: &ChainCtx,
    prefix: &PrefixCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    request_txid: &Hash,
    p: &PrefixPoint,
) -> Result<u64, Fail> {
    verify_prefix_point_at(tier, prefix, resp, seed, request_txid, p, |tok| shape(tier, ctx, tok), true)
}

/// Tolerance on router probabilities.
pub const EPS_ROUTE: f64 = 1e-5;
/// Smallest weight sum of the routed experts.
const ROUTE_SUM_FLOOR: f64 = 6.103515625e-5;

/// Mixing weights of the committed experts `ids` from the committed router scores, or `None` when
/// `ids` is not the top `ids.len()` of `sigmoid(scores) + bias`, or of `softmax(scores)` without a
/// bias.
pub fn route_weights(scores: &[f32], bias: Option<&[f32]>, ids: &[u32], w_scale: f64) -> Option<Vec<f64>> {
    route_weights_gated(scores, bias, ids, w_scale, 0)
}

/// `route_weights` under the expert gating function `gating` (`MoeLinks::gating`).
pub fn route_weights_gated(scores: &[f32], bias: Option<&[f32]>, ids: &[u32], w_scale: f64, gating: u32) -> Option<Vec<f64>> {
    if bias.is_some_and(|b| b.len() != scores.len()) || ids.is_empty() || ids.len() > scores.len() {
        return None;
    }
    let p: Vec<f64> = match (gating, bias) {
        (manifest::GATING_SQRT_SOFTPLUS, _) => scores
            .iter()
            .map(|&x| {
                let x = x as f64;
                libm::sqrt(if x > 20.0 { x } else { libm::log1p(libm::exp(x)) })
            })
            .collect(),
        (0, Some(_)) => scores.iter().map(|&x| 1.0 / (1.0 + libm::exp(-(x as f64)))).collect(),
        (0, None) => {
            let top = scores.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
            let e: Vec<f64> = scores.iter().map(|&x| libm::exp(x as f64 - top)).collect();
            let z: f64 = e.iter().sum();
            e.iter().map(|v| v / z).collect()
        }
        _ => return None,
    };
    if p.iter().any(|v| !v.is_finite()) || bias.is_some_and(|b| b.iter().any(|v| !v.is_finite())) {
        return None;
    }
    let sel: Vec<f64> = match bias {
        Some(b) => p.iter().zip(b).map(|(a, &v)| a + v as f64).collect(),
        None => p.clone(),
    };
    let mut chosen = vec![false; p.len()];
    let mut prev = f64::INFINITY;
    for &id in ids {
        let i = id as usize;
        if i >= p.len() || chosen[i] || sel[i] > prev + EPS_ROUTE {
            return None;
        }
        chosen[i] = true;
        prev = sel[i];
    }
    let floor = ids.iter().map(|&i| sel[i as usize]).fold(f64::INFINITY, f64::min);
    if sel.iter().enumerate().any(|(i, &v)| !chosen[i] && v > floor + EPS_ROUTE) {
        return None;
    }
    let sum = ids.iter().map(|&i| p[i as usize]).sum::<f64>().max(ROUTE_SUM_FLOOR);
    Some(ids.iter().map(|&i| p[i as usize] / sum * w_scale).collect())
}

/// Relative tolerance of the router scores and of the routed sum.
pub const EPS_MOE: f64 = 1e-5;
/// Relative tolerance of a recomputed router score.
pub const EPS_ROUTER: f64 = 1.0 / 512.0;

struct MoeCheck<'a> {
    tier: &'a TierCtx,
    ctx: &'a ChainCtx,
    shape: &'a PosShape,
    pos_root: &'a Hash,
    layer: u64,
    down: u64,
    dim: u64,
    c: u64,
    seed: &'a Hash,
}

/// Relative rounding of a router input a half-precision (BF16) weight matrix may be multiplied with.
const BF16_INPUT_ROUND: f64 = 1.0 / 256.0;

/// Row `row` of `dim` weights of a router (f32, or BF16 widened exactly), and the relative
/// rounding of the input its product may carry.
fn router_row(tier: &TierCtx, re: &manifest::Entry, range: &ChunkRange, row: u64, dim: u64) -> Result<(Vec<f32>, f64), Fail> {
    match re.wtype {
        0 => Ok((f32s(&open_bytes(tier, re, range, row * 4 * dim, 4 * dim)?), 0.0)),
        quant::BF16 => {
            let b = open_bytes(tier, re, range, row * 2 * dim, 2 * dim)?;
            Ok((quant::dequant_row(quant::BF16, &b, dim as usize).ok_or(Fail::Shape)?, BF16_INPUT_ROUND))
        }
        _ => Err(Fail::Manifest),
    }
}

impl MoeCheck<'_> {
    fn verify(&self, moe: &manifest::MoeLinks, m: &crate::opening::MoeAdd, dst: &[f32], src: &[f32], off: usize, n: usize) -> Result<(), Fail> {
        let (tier, ctx, np) = (self.tier, self.ctx, self.tier.n_products);
        let n_used = ctx.n_used as u64;
        if n_used == 0 || m.slots.len() as u64 != n_used {
            return Err(Fail::Shape);
        }
        let re = open_entry(tier, &m.router_row.0, moe.router as u64)?;
        if re.role != manifest::ROLE_ROUTER || re.k != self.dim {
            return Err(Fail::Manifest);
        }
        let n_exp = re.m;
        let scores = whole_leaf(self.pos_root, self.shape, router_leaf(np, ctx.n_layers as u64, self.layer), &m.scores, 4 * n_exp)?;
        let bias = match moe.probs_bias {
            manifest::NONE => None,
            b => Some(f32_weights(tier, &m.bias, b, n_exp, 0, n_exp)?),
        };

        let body = n_used * self.dim * 4;
        let per_slot = self.dim * 4 / B3_CHUNK as u64;
        if !body.is_multiple_of(B3_CHUNK as u64) || !(self.dim * 4).is_multiple_of(B3_CHUNK as u64) {
            return Err(Fail::Shape);
        }
        let total = body + 4 * n_used;
        let y_leaf = 2 * moe.down_exps as u64 + 1;
        let ids_leaf = chunk_leaf_raw(&m.ids, total, body / B3_CHUNK as u64)?;
        if !b3_verify(self.pos_root, self.shape.n_leaves, y_leaf, &ids_leaf, &m.slots[0].leaf_path) {
            return Err(Fail::Trace);
        }
        let ids: Vec<u32> = m.ids.chunk.get(..4 * n_used as usize).ok_or(Fail::Shape)?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let w = route_weights_gated(&scores, bias.as_deref(), &ids, f32::from_bits(moe.w_scale) as f64, moe.gating).ok_or(Fail::Mismatch)?;

        let h = whole_leaf(self.pos_root, self.shape, resid_leaf(np, self.layer, Resid::Attn), &m.resid, 4 * self.dim)?;
        let w_norm = f32_weights(tier, &m.norm, moe.ffn_norm, self.dim, 0, self.dim)?;
        let v = rms_norm(&h, &w_norm, ctx.eps);
        let gate = match (moe.shexp_gate, &m.shexp_gate) {
            (manifest::NONE, None) => None,
            (gi, Some((ge, gr))) if gi != manifest::NONE => {
                let e = open_entry(tier, ge, gi as u64)?;
                if e.role != manifest::ROLE_ROUTER || e.wtype != 0 || e.k != self.dim || e.m != 1 {
                    return Err(Fail::Manifest);
                }
                let wg = f32s(&open_bytes(tier, &e, gr, 0, 4 * self.dim)?);
                let (mut dot, mut mag) = (0f64, 0f64);
                for (&wj, &vj) in wg.iter().zip(&v) {
                    dot += wj as f64 * vj;
                    mag += (wj as f64 * vj).abs();
                }
                let g = 1.0 / (1.0 + libm::exp(-dot));
                if !g.is_finite() || !mag.is_finite() {
                    return Err(Fail::Mismatch);
                }
                Some((g, g * (1.0 - g) * EPS_ROUTER * mag))
            }
            _ => return Err(Fail::Shape),
        };
        let row = draw::point(self.seed, RESID_POINT)[3] % n_exp;
        let (wr, in_round) = router_row(tier, &re, &m.router_row.1, row, self.dim)?;
        let (mut dot, mut mag) = (0f64, 0f64);
        for (&wj, &vj) in wr.iter().zip(&v) {
            dot += wj as f64 * vj;
            mag += (wj as f64 * vj).abs();
        }
        if !(scores[row as usize] as f64 - dot).abs().partial_cmp(&((EPS_ROUTER + in_round) * mag + f32::MIN_POSITIVE as f64)).is_some_and(std::cmp::Ordering::is_le) {
            return Err(Fail::Mismatch);
        }

        let ys = (0..n_used)
            .map(|s| leaf_chunk(self.pos_root, self.shape, y_leaf, &m.slots[s as usize], total, s * per_slot + self.c))
            .collect::<Result<Vec<_>, _>>()?;
        let sh = leaf_chunk(self.pos_root, self.shape, 2 * self.down + 1, &m.shexp, 4 * self.dim, self.c)?;
        if ys.iter().any(|y| y.len() != n) || sh.len() != n || dst.len() != n || src.len() != n || off + n > self.dim as usize {
            return Err(Fail::Shape);
        }
        let (g, g_tol) = gate.unwrap_or((1.0, 0.0));
        for j in 0..n {
            let shared = sh[j] as f64 * g;
            let (mut sum, mut mag) = (shared + src[j] as f64, shared.abs() + (src[j] as f64).abs());
            for (ws, y) in w.iter().zip(&ys) {
                sum += ws * y[j] as f64;
                mag += (ws * y[j] as f64).abs();
            }
            let tol = EPS_MOE * mag + (sh[j] as f64).abs() * g_tol + f32::MIN_POSITIVE as f64;
            if !mag.is_finite() || !(dst[j] as f64 - sum).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
                return Err(Fail::Mismatch);
            }
        }
        Ok(())
    }
}

/// Whether a product of role `role` may feed a residual of a layer, `trimmed` = the last layer
/// kept only at the output positions.
fn layer_role(role: u8, trimmed: bool) -> bool {
    role == manifest::ROLE_PRODUCT || (trimmed && role == manifest::ROLE_OUTPUT)
}

/// Position of a residual point of `layer`.
pub fn resid_position(ctx: &ChainCtx, resp: &ResponseCtx, layer: u64, d0: u64) -> Result<u64, Fail> {
    let n_pos = resp.n_pos as u64;
    if ctx.trim_last && layer + 1 == ctx.n_layers as u64 {
        let first = (resp.n_prompt as u64).checked_sub(1).filter(|&f| f < n_pos).ok_or(Fail::Shape)?;
        return Ok(first + d0 % (n_pos - first));
    }
    Ok(d0 % n_pos)
}

/// Residual site `2 * layer + kind` drawn from `d1` among `n_sites`, site `j` weighted `n_sites - j`.
pub fn resid_site(d1: u64, n_sites: u64) -> u64 {
    let mut x = d1 % (n_sites * (n_sites + 1) / 2).max(1);
    for j in 0..n_sites {
        if x < n_sites - j {
            return j;
        }
        x -= n_sites - j;
    }
    0
}

/// Verify the residual point of a stage-2 opening.
pub fn verify_resid_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, p: &ResidPoint) -> Result<(), Fail> {
    let (np, n_layers) = (tier.n_products, ctx.n_layers as u64);
    if resp.n_pos == 0 || n_layers == 0 {
        return Err(Fail::Shape);
    }
    let d = draw::point(seed, RESID_POINT);
    let sel = resid_site(d[1], 2 * n_layers);
    let (layer, kind) = (sel / 2, if sel.is_multiple_of(2) { Resid::Attn } else { Resid::Out });
    let t = resid_position(ctx, resp, layer, d[0])?;
    let shape = shape(tier, ctx, p.tok);

    let le = open_entry(tier, &p.layer, ctx.layer_base as u64 + layer)?;
    let ll = match le.chain {
        Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => ll,
        _ => return Err(Fail::Manifest),
    };
    let rank = if kind == Resid::Attn { ll.proj } else { ll.down } as u64;
    let pe = open_entry(tier, &p.product, rank)?;
    let trimmed = ctx.trim_last && layer + 1 == n_layers;
    if !layer_role(pe.role, trimmed) || pe.glu_form != manifest::FORM_NONE || pe.n_exp != 1 {
        return Err(Fail::Manifest);
    }
    if let Some(hc) = ll.hc {
        return verify_hc_resid(tier, ctx, resp, seed, p, (layer, kind, t, d[2]), &ll, &hc, rank, pe.m);
    }
    let dim = pe.m;
    let total = 4 * dim;
    let c = d[2] % n_chunks(total);

    let dst_leaf = chunk_leaf(&p.dst, total, c)?;
    let pos_root = b3_fold(shape.n_leaves, resid_leaf(np, layer, kind), &dst_leaf, &p.dst.leaf_path).ok_or(Fail::Trace)?;
    if p.n_ubatch != crate::verify::expected_ubatch(t, resp.n_prompt) {
        return Err(Fail::Mismatch);
    }
    if !b3_verify(&resp.trace_root, resp.n_pos as u64, t, &pos_root, &p.pos_path)
        || !b3_verify(&pos_root, shape.n_leaves, shape.meta_index, &shape.meta_hash(t, p.n_ubatch), &p.meta_path)
    {
        return Err(Fail::Trace);
    }
    let dst = finite_f32s(&p.dst.chunk)?;
    if !(RESID_CHUNK_RMS_MIN..=RESID_RMS_MAX).contains(&rms(&dst)) {
        return Err(Fail::Mismatch);
    }
    let (off, n) = (c as usize * B3_CHUNK / 4, dst.len());

    let src = match (kind, &p.src) {
        (Resid::Attn, ResidSrc::Token(e, r)) if layer == 0 => {
            let row = embedding(tier, ctx, &(e.clone(), r.clone()), p.tok, dim)?;
            row[off..off + n].iter().map(|&v| (v as f64 * ctx.embd_scale) as f32).collect()
        }
        (Resid::Attn, ResidSrc::Chunk(sc)) if layer > 0 => leaf_chunk(&pos_root, &shape, resid_leaf(np, layer - 1, Resid::Out), sc, total, c)?,
        (Resid::Out, ResidSrc::Chunk(sc)) => leaf_chunk(&pos_root, &shape, resid_leaf(np, layer, Resid::Attn), sc, total, c)?,
        _ => return Err(Fail::Shape),
    };
    let post = if kind == Resid::Attn { ll.post_attn } else { ll.post_ffn };
    if ll.moe.is_some() && kind == Resid::Out && post == manifest::NONE && !matches!(p.add, ResidAdd::Moe(_)) {
        return Err(Fail::Shape);
    }
    if let ResidAdd::Moe(m) = &p.add {
        let moe = ll.moe.filter(|_| kind == Resid::Out && post == manifest::NONE).ok_or(Fail::Shape)?;
        let ctx2 = MoeCheck { tier, ctx, shape: &shape, pos_root: &pos_root, layer, down: rank, dim, c, seed };
        return ctx2.verify(&moe, m, &dst, &src, off, n);
    }
    let add: Vec<f64> = match (post, &p.add) {
        (manifest::NONE, ResidAdd::Chunk(ac)) => leaf_chunk(&pos_root, &shape, 2 * rank + 1, ac, total, c)?.iter().map(|&v| v as f64).collect(),
        (pn, ResidAdd::Normed { y, norm }) if pn != manifest::NONE => {
            let yv = whole_leaf(&pos_root, &shape, 2 * rank + 1, y, total)?;
            let w = f32_weights(tier, norm, pn, dim, off as u64, n as u64)?;
            rms_scaled(&yv, ctx.eps, 1.0)[off..off + n].iter().zip(&w).map(|(a, &b)| a * b as f64).collect()
        }
        _ => return Err(Fail::Shape),
    };
    let scale = match ll.scale {
        manifest::NONE => 1.0,
        _ if kind == Resid::Attn => 1.0,
        bits => f32::from_bits(bits) as f64,
    };
    if !scale.is_finite() {
        return Err(Fail::Manifest);
    }
    residual_link(&dst, &add, &src, scale)
}

/// Largest `|v - q d|` of a Q8_1 value, in units of its block scale: half a step, plus the f16
/// rounding of the scale over 127 steps.
const Q8_ROUND: f64 = 0.6;

/// The output of a hyper-connection MoE half on chunk `c` of its streams: `sum_e w_e down_e +
/// shexp`, the routed experts and their weights checked against the router input committed by the
/// gate product of the experts. Returns the values and their magnitudes.
#[allow(clippy::too_many_arguments)]
fn hc_moe_out(
    tier: &TierCtx,
    ctx: &ChainCtx,
    pos_root: &Hash,
    shape: &PosShape,
    seed: &Hash,
    (layer, dim, c, n_ubatch): (u64, u64, u64, u32),
    moe: &manifest::MoeLinks,
    m: &HcMoe,
    shexp: &[f32],
) -> Result<(Vec<f64>, Vec<f64>), Fail> {
    let np = tier.n_products;
    let n_used = ctx.n_used as u64;
    if n_used == 0 || m.slots.len() as u64 != n_used || moe.shexp_gate != manifest::NONE {
        return Err(Fail::Shape);
    }
    let re = open_entry(tier, &m.router_row.0, moe.router as u64)?;
    if re.role != manifest::ROLE_ROUTER || re.k != dim {
        return Err(Fail::Manifest);
    }
    let n_exp = re.m;
    let scores = whole_leaf(pos_root, shape, router_leaf(np, ctx.n_layers as u64, layer), &m.scores, 4 * n_exp)?;
    let bias = match moe.probs_bias {
        manifest::NONE => None,
        b => Some(f32_weights(tier, &m.bias, b, n_exp, 0, n_exp)?),
    };
    let per_slot = dim * 4 / B3_CHUNK as u64;
    if per_slot == 0 || !(dim * 4).is_multiple_of(B3_CHUNK as u64) {
        return Err(Fail::Shape);
    }
    let body = n_used * dim * 4;
    let total = body + 4 * n_used;
    let y_leaf = 2 * moe.down_exps as u64 + 1;
    let ids_leaf = chunk_leaf_raw(&m.ids, total, body / B3_CHUNK as u64)?;
    if !b3_verify(pos_root, shape.n_leaves, y_leaf, &ids_leaf, &m.slots[0].leaf_path) {
        return Err(Fail::Trace);
    }
    let ids: Vec<u32> = m.ids.chunk.get(..4 * n_used as usize).ok_or(Fail::Shape)?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let w = route_weights_gated(&scores, bias.as_deref(), &ids, f32::from_bits(moe.w_scale) as f64, moe.gating).ok_or(Fail::Mismatch)?;

    let ge = open_entry(tier, &m.gate, moe.gate_exps as u64)?;
    if ge.k != dim || ge.n_exp < 2 {
        return Err(Fail::Manifest);
    }
    let layout = crate::verify::input_layout(ge.role, n_ubatch, ge.wtype);
    let x_one = crate::verify::x_len(dim, layout) as usize;
    if !b3_verify(pos_root, shape.n_leaves, 2 * moe.gate_exps as u64, blake3::hash(&m.x.bytes).as_bytes(), &m.x.leaf_path) {
        return Err(Fail::Trace);
    }
    let blocks = q8::blocks(m.x.bytes.get(..x_one).ok_or(Fail::Shape)?, layout, dim as usize).ok_or(Fail::Shape)?;
    let row = draw::point(seed, RESID_POINT)[3] % n_exp;
    let (wr, in_round) = router_row(tier, &re, &m.router_row.1, row, dim)?;
    let (mut dot, mut mag, mut round) = (0f64, 0f64, 0f64);
    for (j, &wj) in wr.iter().enumerate() {
        let (d, q) = &blocks[j / 32];
        let v = q[j % 32] as f64 * *d as f64;
        dot += wj as f64 * v;
        mag += (wj as f64 * v).abs();
        round += (wj as f64).abs() * (*d as f64).abs() * Q8_ROUND;
    }
    let tol = (EPS_ROUTER + in_round) * mag + round + f32::MIN_POSITIVE as f64;
    if !(scores[row as usize] as f64 - dot).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
        return Err(Fail::Mismatch);
    }
    let ys = (0..n_used)
        .map(|s| leaf_chunk(pos_root, shape, y_leaf, &m.slots[s as usize], total, s * per_slot + c))
        .collect::<Result<Vec<_>, _>>()?;
    if ys.iter().any(|y| y.len() != shexp.len()) {
        return Err(Fail::Shape);
    }
    let out = (0..shexp.len()).map(|j| shexp[j] as f64 + w.iter().zip(&ys).map(|(a, y)| a * y[j] as f64).sum::<f64>()).collect();
    let mags = (0..shexp.len()).map(|j| (shexp[j] as f64).abs() + w.iter().zip(&ys).map(|(a, y)| (a * y[j] as f64).abs()).sum::<f64>()).collect();
    Ok((out, mags))
}

/// Chunk of a hyper-connection residual drawn from `d2`: the stream, the chunk within the stream,
/// and the chunks per stream, for streams of `dim` values.
pub fn hc_resid_chunk(d2: u64, n_hc: u64, dim: u64) -> Option<(u64, u64, u64)> {
    let per = (4 * dim).checked_div(B3_CHUNK as u64).filter(|&p| p > 0 && (4 * dim).is_multiple_of(B3_CHUNK as u64))?;
    let c = d2 % (n_hc * per);
    Some((c / per, c % per, per))
}

/// A hyper-connection residual, element by element on the drawn chunk:
/// `R'[s] = post[s] * o + sum_src comb[s][src] * R[src]`.
#[allow(clippy::too_many_arguments)]
fn verify_hc_resid(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    p: &ResidPoint,
    site: (u64, Resid, u64, u64),
    ll: &manifest::LayerLinks,
    hc: &manifest::HcLinks,
    rank: u64,
    dim: u64,
) -> Result<(), Fail> {
    let (layer, kind, t, d2) = site;
    let np = tier.n_products;
    let shape = shape(tier, ctx, p.tok);
    let n = hc.n_hc as u64;
    let (s, jc, per) = hc_resid_chunk(d2, n, dim).ok_or(Fail::Shape)?;
    let total = 4 * n * dim;
    let c = s * per + jc;
    let ResidAdd::Hc(h) = &p.add else { return Err(Fail::Shape) };
    let dst_leaf = chunk_leaf(&p.dst, total, c)?;
    let pos_root = b3_fold(shape.n_leaves, resid_leaf(np, layer, kind), &dst_leaf, &p.dst.leaf_path).ok_or(Fail::Trace)?;
    if p.n_ubatch != crate::verify::expected_ubatch(t, resp.n_prompt) {
        return Err(Fail::Mismatch);
    }
    if !b3_verify(&resp.trace_root, resp.n_pos as u64, t, &pos_root, &p.pos_path)
        || !b3_verify(&pos_root, shape.n_leaves, shape.meta_index, &shape.meta_hash(t, p.n_ubatch), &p.meta_path)
    {
        return Err(Fail::Trace);
    }
    let dst = finite_f32s(&p.dst.chunk)?;
    if !(RESID_CHUNK_RMS_MIN..=RESID_RMS_MAX).contains(&rms(&dst)) {
        return Err(Fail::Mismatch);
    }
    let len = dst.len();
    let off = (jc * B3_CHUNK as u64 / 4) as usize;
    let o = leaf_chunk(&pos_root, &shape, 2 * rank + 1, &h.o, 4 * dim, jc)?;
    let (o, o_mag): (Vec<f64>, Vec<f64>) = match (kind, ll.moe, &h.moe) {
        (Resid::Out, Some(moe), Some(hm)) => hc_moe_out(tier, ctx, &pos_root, &shape, seed, (layer, dim, jc, p.n_ubatch), &moe, hm, &o)?,
        (Resid::Out, Some(_), None) | (_, _, Some(_)) => return Err(Fail::Shape),
        _ => (o.iter().map(|&v| v as f64).collect(), o.iter().map(|&v| (v as f64).abs()).collect()),
    };
    let srcs: Vec<Vec<f32>> = match (kind, layer, &p.src) {
        (Resid::Attn, 0, ResidSrc::Token(e, r)) if h.streams.is_empty() => {
            let row = embedding(tier, ctx, &(e.clone(), r.clone()), p.tok, dim)?;
            let v: Vec<f32> = row.get(off..off + len).ok_or(Fail::Shape)?.iter().map(|&x| (x as f64 * ctx.embd_scale) as f32).collect();
            vec![v; n as usize]
        }
        (_, _, ResidSrc::Chunk(first)) if h.streams.len() as u64 + 1 == n && !(kind == Resid::Attn && layer == 0) => {
            let leaf = if kind == Resid::Attn { attn_source(tier, ctx, ll)?.ok_or(Fail::Shape)? } else { resid_leaf(np, layer, Resid::Attn) };
            std::iter::once(first)
                .chain(&h.streams)
                .enumerate()
                .map(|(src, sc)| leaf_chunk(&pos_root, &shape, leaf, sc, total, src as u64 * per + jc))
                .collect::<Result<_, _>>()?
        }
        _ => return Err(Fail::Shape),
    };
    let (fn_rank, si, bi) = match kind {
        Resid::Attn => (hc.attn_fn, hc.attn_scale, hc.attn_base),
        Resid::Out => (hc.ffn_fn, hc.ffn_scale, hc.ffn_base),
    };
    let mix = hc_mix_at(tier, &pos_root, &shape, hc, fn_rank, &h.mixes, si, (&h.base, bi))?;
    if o.len() != len || srcs.iter().any(|v| v.len() != len) {
        return Err(Fail::Shape);
    }
    let su = s as usize;
    for j in 0..len {
        let mut want = mix.post[su] * o[j];
        let mut mag = mix.post[su].abs() * o_mag[j];
        for (src, v) in srcs.iter().enumerate() {
            let term = mix.comb[su + n as usize * src] * v[j] as f64;
            want += term;
            mag += term.abs();
        }
        let tol = EPS_HC * mag + f32::MIN_POSITIVE as f64;
        if !tol.is_finite() || !(dst[j] as f64 - want).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
}

/// Verify a residual opening: its residual point, then one row of the product whose output the
/// residual adds and, on a routed layer, one row of a routed expert.
pub fn verify_resid_opening(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &ResidOpening) -> Result<(), Fail> {
    let p = &op.resid;
    verify_resid_point(tier, ctx, resp, seed, p)?;
    let (np, n_layers) = (tier.n_products, ctx.n_layers as u64);
    let d = draw::point(seed, RESID_POINT);
    let sel = resid_site(d[1], 2 * n_layers);
    let (layer, kind) = (sel / 2, if sel.is_multiple_of(2) { Resid::Attn } else { Resid::Out });
    let shape = shape(tier, ctx, p.tok);
    let ll = match open_entry(tier, &p.layer, ctx.layer_base as u64 + layer)?.chain {
        Chain::Layer(ll) => ll,
        _ => return Err(Fail::Manifest),
    };
    let rank = if kind == Resid::Attn { ll.proj } else { ll.down } as u64;
    let pe = open_entry(tier, &p.product, rank)?;
    let (dim, per) = (pe.m, (B3_CHUNK / 4) as u64);
    let (c, dst_c, dst_total) = match ll.hc {
        Some(hc) => {
            let (s, jc, cps) = hc_resid_chunk(d[2], hc.n_hc as u64, dim).ok_or(Fail::Shape)?;
            (jc, s * cps + jc, 4 * hc.n_hc as u64 * dim)
        }
        None => {
            let c = d[2] % n_chunks(4 * dim);
            (c, c, 4 * dim)
        }
    };
    let dst_leaf = chunk_leaf(&p.dst, dst_total, dst_c)?;
    let pos_root = b3_fold(shape.n_leaves, resid_leaf(np, layer, kind), &dst_leaf, &p.dst.leaf_path).ok_or(Fail::Trace)?;
    let lo = c * per;
    let row = lo + (d[3] >> 32) % per.min(dim.checked_sub(lo).filter(|&n| n > 0).ok_or(Fail::Shape)?);
    let y_at = |b: &[u8], i: u64| -> Result<f64, Fail> {
        let i = 4 * i as usize;
        Ok(f32::from_le_bytes(b.get(i..i + 4).ok_or(Fail::Shape)?.try_into().unwrap()) as f64)
    };
    let y = match &p.add {
        ResidAdd::Chunk(ac) => y_at(&ac.chunk, row - lo)?,
        ResidAdd::Normed { y, .. } => y_at(&y.bytes, row)?,
        ResidAdd::Moe(m) => y_at(&m.shexp.chunk, row - lo)?,
        ResidAdd::Hc(h) => y_at(&h.o.chunk, row - lo)?,
    };
    let f = &op.feed;
    feed_row(tier, &shape, &pos_root, p.n_ubatch, (&pe, rank), (0, 0, 1), row, y, (&f.x, &f.x_path), &f.row, f.bias.as_ref())?;
    match (&p.add, &op.expert) {
        (ResidAdd::Moe(m), Some(er)) => {
            let moe = ll.moe.ok_or(Fail::Shape)?;
            let de = open_entry(tier, &er.entry, moe.down_exps as u64)?;
            let trimmed = ctx.trim_last && layer + 1 == n_layers;
            if !layer_role(de.role, trimmed) || de.glu_form != manifest::FORM_NONE || de.n_exp < 2 || de.m != dim {
                return Err(Fail::Manifest);
            }
            let n_used = ctx.n_used as u64;
            let slot = (d[3] >> 48) % n_used.max(1);
            let s = 4 * slot as usize;
            let id = u32::from_le_bytes(m.ids.chunk.get(s..s + 4).ok_or(Fail::Shape)?.try_into().unwrap()) as u64;
            let y = y_at(&m.slots.get(slot as usize).ok_or(Fail::Shape)?.chunk, row - lo)?;
            feed_row(tier, &shape, &pos_root, p.n_ubatch, (&de, moe.down_exps as u64), (id, slot, n_used), row, y, (&er.x, &er.x_path), &er.row, None)
        }
        (ResidAdd::Hc(h), Some(er)) if h.moe.is_some() => {
            let (moe, hm) = (ll.moe.ok_or(Fail::Shape)?, h.moe.as_ref().unwrap());
            let de = open_entry(tier, &er.entry, moe.down_exps as u64)?;
            if de.role != manifest::ROLE_PRODUCT || de.glu_form != manifest::FORM_NONE || de.n_exp < 2 || de.m != dim {
                return Err(Fail::Manifest);
            }
            let n_used = ctx.n_used as u64;
            let slot = (d[3] >> 48) % n_used.max(1);
            let s = 4 * slot as usize;
            let id = u32::from_le_bytes(hm.ids.chunk.get(s..s + 4).ok_or(Fail::Shape)?.try_into().unwrap()) as u64;
            let y = y_at(&hm.slots.get(slot as usize).ok_or(Fail::Shape)?.chunk, row - lo)?;
            feed_row(tier, &shape, &pos_root, p.n_ubatch, (&de, moe.down_exps as u64), (id, slot, n_used), row, y, (&er.x, &er.x_path), &er.row, None)
        }
        (ResidAdd::Moe(_), None) | (_, Some(_)) => Err(Fail::Shape),
        (_, None) => Ok(()),
    }
}

/// Checks `y` against row `row` of expert `exp` of product `e` over its committed input, slot
/// `slot` of `n_slots` (a single input row serves every slot).
#[allow(clippy::too_many_arguments)]
fn feed_row(
    tier: &TierCtx,
    shape: &PosShape,
    pos_root: &Hash,
    n_ubatch: u32,
    (e, rank): (&manifest::Entry, u64),
    (exp, slot, n_slots): (u64, u64, u64),
    row: u64,
    y: f64,
    (x, x_path): (&[u8], &[Hash]),
    w: &ChunkRange,
    bias: Option<&(EntryOpening, ChunkRange)>,
) -> Result<(), Fail> {
    let layout = crate::verify::input_layout(e.role, n_ubatch, e.wtype);
    let x_one = crate::verify::x_len(e.k, layout);
    let xs = match x.len() as u64 {
        n if n == x_one => x,
        n if n_slots > 1 && n == n_slots * x_one => &x[(slot * x_one) as usize..((slot + 1) * x_one) as usize],
        _ => return Err(Fail::Shape),
    };
    if !b3_verify(pos_root, shape.n_leaves, 2 * rank, blake3::hash(x).as_bytes(), x_path) {
        return Err(Fail::Trace);
    }
    let k = e.k as usize;
    if !q8::sums_consistent(xs, layout, k) {
        return Err(Fail::Mismatch);
    }
    let bytes = crate::verify::open_expert_row(tier, e, w, exp, row)?;
    let r = crate::dot::canonical_dot(e.wtype, &bytes, k, xs, layout).ok_or(Fail::Unsupported)?;
    let b = match (e.bias, bias) {
        (manifest::NONE, None) => 0.0,
        (bi, Some((be, br))) if bi != manifest::NONE && n_slots == 1 => {
            let bias_entry = open_entry(tier, be, bi as u64)?;
            if bias_entry.role != manifest::ROLE_BIAS || bias_entry.wtype != 0 || bias_entry.k != e.m {
                return Err(Fail::Manifest);
            }
            f32::from_le_bytes(open_bytes(tier, &bias_entry, br, row * 4, 4)?.try_into().unwrap()) as f64
        }
        _ => return Err(Fail::Shape),
    };
    let (y_hat, tol) = (r.y_hat + b, crate::verify::EPS * (r.scale + b.abs()));
    if !(y.is_finite() && y_hat.is_finite() && tol.is_finite()) || !(y - y_hat).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
        return Err(Fail::Mismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::q8_1_row;

    const D: usize = 512;
    const EPS: f64 = 1e-6;

    fn vec_of(seed: u32, scale: f32) -> Vec<f32> {
        (0..D).map(|i| scale * ((((i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2001) as f32 / 1000.0 - 1.0)).collect()
    }

    fn norm_w() -> Vec<f32> {
        (0..D).map(|i| 1.0 + 0.001 * (i % 17) as f32).collect()
    }

    fn honest_input(src: &[f32]) -> Vec<u8> {
        q8_1_row(&rms_norm(src, &norm_w(), EPS))
    }

    #[test]
    fn layout_of_a_stage2_position() {
        assert_eq!(n_pos_leaves(3, 2, 0), 11);
        assert_eq!(n_pos_leaves(3, 2, 2), 13);
        assert_eq!(router_leaf(3, 2, 1), 11);
        assert_eq!(meta_leaf(3, 2, 2), 12);
        assert_eq!(resid_leaf(3, 0, Resid::Attn), 6);
        assert_eq!(resid_leaf(3, 1, Resid::Out), 9);
        assert_eq!(meta_leaf(3, 2, 0), 10);
        assert_eq!(&meta(7, 512, 42)[8..], &42u32.to_le_bytes());
    }

    #[test]
    fn honest_norm_link_passes() {
        let h = vec_of(1, 3.0);
        assert_eq!(norm_link(&honest_input(&h), q8::MMVQ, &h, &norm_w(), EPS), Ok(()));
    }

    #[test]
    fn empty_input_fails() {
        let h = vec_of(1, 3.0);
        let x = vec![0u8; D / 32 * 36];
        assert_eq!(norm_link(&x, q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn a_non_canonical_scale_fails() {
        let h = vec_of(1, 3.0);
        let honest = honest_input(&h);
        let rescaled = |f: f32, flip: bool| -> Vec<u8> {
            let mut x = honest.clone();
            for blk in x.chunks_exact_mut(36) {
                let d = crate::f16::read(blk, 0) * f;
                blk[..2].copy_from_slice(&crate::f16::from_f32(if flip { -d } else { d }).to_le_bytes());
                for j in 0..32 {
                    let q = libm::roundf(blk[4 + j] as i8 as f32 / f) as i8;
                    blk[4 + j] = (if flip { -q } else { q }) as u8;
                }
            }
            x
        };
        assert_eq!(norm_link(&rescaled(64.0, false), q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
        assert_eq!(norm_link(&rescaled(1.0, true), q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn input_of_another_source_fails() {
        let (h, other) = (vec_of(1, 3.0), vec_of(2, 3.0));
        assert_eq!(norm_link(&honest_input(&other), q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn one_input_fails_at_other_positions() {
        let c = honest_input(&vec_of(9, 2.0));
        let failed = (1..=8).filter(|&s| norm_link(&c, q8::MMVQ, &vec_of(s, 3.0), &norm_w(), EPS).is_err()).count();
        assert_eq!(failed, 8);
    }

    #[test]
    fn wrong_norm_weight_fails() {
        let h = vec_of(1, 3.0);
        let w2: Vec<f32> = (0..D).map(|i| 1.0 + 0.05 * (i % 13) as f32).collect();
        assert_eq!(norm_link(&honest_input(&h), q8::MMVQ, &h, &w2, EPS), Err(Fail::Mismatch));
    }

    fn spiked_row(spike: f64) -> Vec<u8> {
        let mut h: Vec<f64> = vec_of(5, 0.5).iter().map(|&a| a as f64).collect();
        for v in h[7 * 32..8 * 32].iter_mut() {
            *v *= spike;
        }
        q8_1_row(&h)
    }

    #[test]
    fn the_link_draw_lands_on_a_spiked_block() {
        let x = spiked_row(1e4);
        let seeds = 0..400u32;
        let hits = seeds.clone().filter(|s| link_block(blake3::hash(&s.to_le_bytes()).as_bytes(), 0, &x, q8::MMVQ, D as u64) == Some(7)).count();
        assert!(hits > 400 / 3, "{hits}");
        let flat = spiked_row(1.0);
        let n = (D / 32) as u64;
        assert!(seeds.map(|s| link_block(blake3::hash(&s.to_le_bytes()).as_bytes(), 0, &flat, q8::MMVQ, D as u64).unwrap()).all(|b| b < n));
    }

    #[test]
    fn an_unlinked_input_is_bounded_by_its_median_block() {
        assert!(peak_bounded(&spiked_row(1.0), q8::MMVQ, D as u64, 0));
        assert!(peak_bounded(&spiked_row(1000.0), q8::MMVQ, D as u64, 0));
        assert!(!peak_bounded(&spiked_row(2000.0), q8::MMVQ, D as u64, 0));
        assert!(peak_bounded(&q8_1_row(&vec![0.0; D]), q8::MMVQ, D as u64, 0));
        let heads = [spiked_row(1.0), spiked_row(1.0), spiked_row(2000.0)].concat();
        assert!(peak_bounded(&heads, q8::MMVQ, D as u64, 1));
        assert!(!peak_bounded(&heads, q8::MMVQ, D as u64, 2), "the drawn slot is the one bounded");
        assert!(!peak_bounded(&heads, q8::MMVQ, D as u64, 3));
        let loud: Vec<f64> = vec_of(5, 500.0).iter().map(|&a| a as f64).collect();
        let shifted = [q8_1_row(&loud), spiked_row(1e5)].concat();
        assert!(!peak_bounded(&shifted, q8::MMVQ, D as u64, 1), "other slots do not move the median of the drawn one");
        assert!(peak_bounded(&spiked_row(1.0), q8::MMVQ, D as u64, 5), "a single row serves every slot");
    }

    #[test]
    fn glu_link_checks_the_covered_blocks() {
        let h = vec_of(3, 0.5);
        let x = q8_1_row(&h.iter().map(|&a| a as f64).collect::<Vec<_>>());
        assert_eq!(glu_link(&x, q8::MMVQ, D, 4, &h[128..256]), Ok(()));
        assert_eq!(glu_link(&x, q8::MMVQ, D, 5, &h[128..256]), Err(Fail::Mismatch));
        assert_eq!(glu_link(&x, q8::MMVQ, D, 15, &h[128..256]), Err(Fail::Mismatch));
    }

    #[test]
    fn an_infinite_activation_never_passes_a_link() {
        assert_eq!(residual_link(&[f32::INFINITY], &[f64::NEG_INFINITY], &[0.0], 1.0), Err(Fail::Mismatch));
        assert_eq!(residual_link(&[f32::NEG_INFINITY], &[0.0], &[f32::INFINITY], 1.0), Err(Fail::Mismatch));
        let leaf: Vec<u8> = [1.0f32, f32::INFINITY, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(finite_f32s(&leaf), Err(Fail::Mismatch));
        let leaf: Vec<u8> = [1.0f32, f32::NAN].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(finite_f32s(&leaf), Err(Fail::Mismatch));
        let leaf: Vec<u8> = [1.0f32, -2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(finite_f32s(&leaf), Ok(vec![1.0, -2.5]));
    }

    #[test]
    fn non_finite_values_fail() {
        let h = vec_of(1, 3.0);
        let mut v: Vec<f64> = rms_norm(&h, &norm_w(), EPS);
        let x = q8_1_row(&v);
        v[3] = f64::NAN;
        assert_eq!(block_deviation(&x, q8::MMVQ, D, 0, &v), None);
        let add: Vec<f64> = h.iter().map(|&a| a as f64).collect();
        let mut dst: Vec<f32> = h.iter().map(|a| 2.0 * a).collect();
        dst[0] = f32::NAN;
        assert_eq!(residual_link(&dst, &add, &h, 1.0), Err(Fail::Mismatch));
    }

    #[test]
    fn outliers_count_the_values_of_a_block_past_the_base() {
        let v: Vec<f64> = rms_norm(&vec_of(1, 3.0), &norm_w(), EPS);
        let x = q8_1_row(&v);
        let d: Vec<f64> = q8::blocks(&x, q8::MMVQ, D).unwrap().iter().map(|b| b.0 as f64).collect();
        let shift = |v: &[f64], n: usize| -> Vec<f64> { v.iter().enumerate().map(|(i, a)| if i % 32 < n { a + 1.6 * d[i / 32] } else { *a }).collect() };
        let (worst, spare) = block_outliers(&x, q8::MMVQ, D, 0, &v, 1.0).unwrap();
        assert!(worst <= 0.5 + 1e-9 && spare == 0);
        let (worst, spare) = block_outliers(&x, q8::MMVQ, D, 0, &shift(&v, 3), 1.0).unwrap();
        assert!(worst <= 2.1 + 1e-9 && spare == 3);
        let (worst, spare) = block_outliers(&x, q8::MMVQ, D, 0, &shift(&v, 32), 1.0).unwrap();
        assert!(worst <= 2.1 + 1e-9 && spare == 32);
    }

    #[test]
    fn routing_follows_the_top_scores() {
        let scores: Vec<f32> = (0..16).map(|i| ((i * 7) % 16) as f32 * 0.25 - 2.0).collect();
        let bias = vec![0.0f32; 16];
        let mut order: Vec<u32> = (0..16).collect();
        order.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
        let top = &order[..4];
        let w = route_weights(&scores, Some(&bias), top, 2.0).unwrap();
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        assert!(route_weights(&scores, Some(&bias), &order[1..5], 2.0).is_none());
        assert!(route_weights(&scores, Some(&bias), &[top[1], top[0], top[2], top[3]], 2.0).is_none());
        assert!(route_weights(&scores, Some(&bias), &[top[0], top[0], top[2], top[3]], 2.0).is_none());
        let mut biased = bias.clone();
        biased[order[10] as usize] = 5.0;
        let w2 = route_weights(&scores, Some(&biased), &[order[10], top[0], top[1], top[2]], 2.0).unwrap();
        assert!((w2.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    }

    /// `hc_mix` against llama.cpp's fused mix kernels on GLM-5.3-Flash, layer 0, first token of a
    /// probe: mixes recomputed from the GGUF, expected values as the probe printed them.
    #[test]
    fn hc_mix_matches_the_fused_kernels_on_glm() {
        let mixes: [f32; 24] = [2.715549469e+01, -8.747535706e+00, 8.463341713e+00, 7.669409180e+01, -5.148453712e+00, -3.682340240e+01, -1.652810669e+01, -1.402769089e+01, 8.885743713e+01, -2.163059235e+01, -3.372954559e+01, -3.353501129e+01, 1.280406647e+02, -1.879361343e+01, -4.886148071e+01, -6.059747314e+01, -2.376058960e+01, -9.349130630e+00, -1.406003284e+01, 4.707107925e+01, -2.038398552e+01, 5.488075638e+01, -9.178322792e+00, -2.542195702e+01];
        let scale: [f32; 3] = [6.140246615e-02, 4.462148994e-02, 8.887463808e-02];
        let base: [f32; 24] = [-7.583102226e+00, 5.753003359e-01, -7.202132225e+00, -7.759641647e+00, 2.692349434e+00, 1.496213675e+00, -1.532154322e+00, -1.147554278e+00, 8.127574256e-05, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -3.258989891e-04, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -3.044366604e-04, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -8.000000000e+00, -2.892836346e-04];
        let m = hc_mix(&mixes, &scale, &base, 4, 20, 1e-6).unwrap();
        let near = |a: &[f64], b: &[f64], tol: f64| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol);
        assert!(near(&m.pre, &[0.0027, 0.5099, 0.0013, 0.0451], 2e-3), "{:?}", m.pre);
        assert!(near(&m.post, &[1.8431, 0.9270, 0.1870, 0.2901], 5e-3), "{:?}", m.post);
        let printed = [[0.9129, 0.0015, 0.0041, 0.0048], [0.0871, 0.8810, 0.0005, 0.0005], [0.0, 0.0002, 0.9943, 0.0888], [0.0, 0.1173, 0.0011, 0.9059]];
        for (src, row) in printed.iter().enumerate() {
            let got: Vec<f64> = (0..4).map(|dst| m.comb[dst + 4 * src]).collect();
            assert!(near(&got, row, 5e-3), "src {src}: {got:?} vs {row:?}");
        }
        for dst in 0..4 {
            assert!(((0..4).map(|src| m.comb[dst + 4 * src]).sum::<f64>() - 1.0).abs() < 1e-5);
        }
        assert!(hc_mix(&mixes[..23], &scale, &base, 4, 20, 1e-6).is_none());
    }

    #[test]
    fn softmax_routing_takes_the_top_scores() {
        let scores = [0.5f32, 3.0, -1.0, 2.0, 2.5, 0.0];
        let w = route_weights(&scores, None, &[1, 4, 3], 1.0).unwrap();
        let e: Vec<f64> = [3.0f64, 2.5, 2.0].iter().map(|s| s.exp()).collect();
        let z: f64 = e.iter().sum();
        for (a, b) in w.iter().zip(&e) {
            assert!((a - b / z).abs() < 1e-12);
        }
        assert!(route_weights(&scores, None, &[1, 4, 0], 1.0).is_none());
        assert!(route_weights(&scores, None, &[4, 1, 3], 1.0).is_none());
        assert!(route_weights(&scores, Some(&[0.0; 5]), &[1], 1.0).is_none());
        let sp = |x: f32| libm::sqrt(libm::log1p(libm::exp(x as f64)));
        let w = route_weights_gated(&scores, None, &[1, 4, 3], 1.5, manifest::GATING_SQRT_SOFTPLUS).unwrap();
        let z = sp(scores[1]) + sp(scores[4]) + sp(scores[3]);
        assert!((w[0] - 1.5 * sp(scores[1]) / z).abs() < 1e-12);
        assert!(route_weights_gated(&scores, None, &[1, 4, 3], 1.0, 7).is_none());
    }

    #[test]
    fn residual_addition() {
        let (y, h) = (vec_of(4, 0.3), vec_of(5, 2.0));
        let add: Vec<f64> = y.iter().map(|&a| a as f64).collect();
        let r: Vec<f32> = y.iter().zip(&h).map(|(a, b)| a + b).collect();
        assert_eq!(residual_link(&r, &add, &h, 1.0), Ok(()));
        assert_eq!(residual_link(&h, &add, &h, 1.0), Err(Fail::Mismatch));
        let zero = vec![0f64; D];
        assert_eq!(residual_link(&r, &zero, &h, 1.0), Err(Fail::Mismatch));
        let scaled: Vec<f32> = r.iter().map(|a| a * 0.75).collect();
        assert_eq!(residual_link(&scaled, &add, &h, 0.75), Ok(()));
        assert_eq!(residual_link(&scaled, &add, &h, 1.0), Err(Fail::Mismatch));
    }

    #[test]
    fn post_norm_residual() {
        let (y, h, w) = (vec_of(6, 4.0), vec_of(7, 2.0), norm_w());
        let add = rms_norm(&y, &w, EPS);
        let r: Vec<f32> = add.iter().zip(&h).map(|(a, &b)| (*a as f32) + b).collect();
        assert_eq!(residual_link(&r, &add, &h, 1.0), Ok(()));
        let raw: Vec<f64> = y.iter().map(|&a| a as f64).collect();
        assert_eq!(residual_link(&r, &raw, &h, 1.0), Err(Fail::Mismatch));
    }
}

/// Outcome of an accepted attention phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttnStep {
    Done,
    /// The next phase opens one of the `n` children whose partials this phase committed under
    /// `root`, drawn under `weights`.
    Next { root: Hash, n: u64, weights: Vec<u16> },
}

/// Verify the attention point of a response opened in a single phase.
pub fn verify_attn_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &AttnOpening) -> Result<(), Fail> {
    match verify_attn_phase(tier, ctx, resp, seed, &[], None, op)? {
        AttnStep::Done => Ok(()),
        AttnStep::Next { .. } => Err(Fail::Unsupported),
    }
}

/// Drawn attention point: layer links, query position, draws and the block tree of the attended
/// positions.
struct AttnSite {
    ll: manifest::LayerLinks,
    a: manifest::AttnLinks,
    slot: u64,
    t: u64,
    d: [u64; 4],
    tree: BlockTree,
    /// Values per partial.
    dv: usize,
    /// Attended positions of a sparse attention layer, from the tree's first position.
    keep: Option<Vec<bool>>,
}

impl AttnSite {
    fn node(&self, path: &[u64]) -> Option<(u64, u64)> {
        match &self.keep {
            Some(k) => PrunedTree { tree: self.tree, keep: k }.node(path),
            None => self.tree.node(path),
        }
    }

    fn children(&self, path: &[u64]) -> Option<Vec<(u64, u64)>> {
        match &self.keep {
            Some(k) => PrunedTree { tree: self.tree, keep: k }.children(path),
            None => self.tree.children(path),
        }
    }

    fn attended(&self, p: u64) -> bool {
        self.keep.as_ref().is_none_or(|k| PrunedTree { tree: self.tree, keep: k }.attended(p))
    }
}

/// Length of a selection leaf under `dsa`.
pub const fn sel_len(dsa: &manifest::DsaLinks) -> usize {
    4 + 4 * dsa.n_top as usize
}

/// Threshold and pools of a selection leaf: an f32 score, then `n_top` slots holding the chosen
/// pools (strictly increasing, below the complete pools of `t`, as many as the indexer keeps),
/// the slots left over set to `u32::MAX`.
pub fn dsa_pools(dsa: &manifest::DsaLinks, t: u64, sel: &[u8]) -> Option<(f32, Vec<u64>)> {
    let kpool = dsa.kpool as u64;
    if kpool < 2 || dsa.n_top == 0 || sel.len() != sel_len(dsa) {
        return None;
    }
    let thr = f32::from_le_bytes(sel[..4].try_into().unwrap());
    let slots: Vec<u32> = sel[4..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_pool = (t + 1) / kpool;
    let m = n_pool.min(dsa.n_top as u64) as usize;
    let pools: Vec<u64> = slots[..m].iter().map(|&p| p as u64).collect();
    if slots[m..].iter().any(|&p| p != u32::MAX) || pools.windows(2).any(|w| w[0] >= w[1]) || pools.last().is_some_and(|&p| p >= n_pool) {
        return None;
    }
    Some((thr, pools))
}

/// Positions of `[0, t]` a query at `t` attends under `dsa`, given its selection leaf `sel`.
pub fn dsa_attended(dsa: &manifest::DsaLinks, t: u64, sel: &[u8]) -> Option<Vec<bool>> {
    let kpool = dsa.kpool as u64;
    let (_, pools) = dsa_pools(dsa, t, sel)?;
    let n_pool = (t + 1) / kpool;
    let mut keep = vec![false; t as usize + 1];
    for p in pools {
        keep[(p * kpool) as usize..((p + 1) * kpool) as usize].iter_mut().for_each(|k| *k = true);
    }
    if dsa.tail != 0 {
        keep[(n_pool * kpool) as usize..].iter_mut().for_each(|k| *k = true);
    }
    keep.iter().any(|&k| k).then_some(keep)
}

fn attn_site(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &AttnOpening) -> Result<AttnSite, Fail> {
    let (nl, n_pos, n_attn) = (ctx.n_layers as u64, resp.n_pos as u64, ctx.n_attn as u64);
    if n_pos == 0 || nl == 0 || n_attn == 0 || n_attn > nl || ctx.seq_heads == 0 {
        return Err(Fail::Unsupported);
    }
    if crate::seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = draw::point(seed, ATTN_POINT);
    let slot = d[1] % n_attn;
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let le = open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?;
    let (ll, a) = match le.chain {
        Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => (ll, ll.attn.ok_or(Fail::Shape)?),
        _ => return Err(Fail::Manifest),
    };
    if ll.comp.is_some() {
        return Err(Fail::Shape);
    }
    if a.slot as u64 != slot {
        return Err(Fail::Manifest);
    }
    let t = resid_position(ctx, resp, layer, d[0])?;
    if a.softcap != 0 || a.v_norm > 1 || a.gated > 1 {
        return Err(Fail::Unsupported);
    }
    let hd = a.head_dim as usize;
    if hd == 0 || !hd.is_multiple_of(32) {
        return Err(Fail::Manifest);
    }
    let (start, row, dv) = if a.mla_rank != 0 {
        (0, hd / 32 * 34, a.mla_rank as usize)
    } else {
        let start = if a.swa_window > 0 { (t + 1).saturating_sub(a.swa_window as u64) } else { 0 };
        (start, 2 * (hd / 32 * 34), hd)
    };
    let (c, b) = block_params(row, partial_len(dv)).ok_or(Fail::Manifest)?;
    let tree = BlockTree::new(start, t + 1, c, b).ok_or(Fail::Manifest)?;
    let keep = match (&ll.dsa, &op.sel) {
        (None, None) => None,
        (Some(dsa), Some(sel)) => {
            if dsa.sel >= ctx.n_sel || start != 0 || a.mla_rank == 0 {
                return Err(Fail::Manifest);
            }
            let pshape = shape_any(tier, ctx);
            let leaf = sel_leaf(tier.n_products, nl, n_router(ctx), dsa.sel as u64);
            let root_t = b3_fold(pshape.n_leaves, leaf, blake3::hash(&sel.bytes).as_bytes(), &sel.leaf_path).ok_or(Fail::Trace)?;
            if !b3_verify(&op.pos_root, n_pos, t, &root_t, &sel.pos_path) {
                return Err(Fail::Trace);
            }
            Some(dsa_attended(dsa, t, &sel.bytes).ok_or(Fail::Mismatch)?)
        }
        _ => return Err(Fail::Shape),
    };
    Ok(AttnSite { ll, a, slot, t, d, tree, dv, keep })
}

/// Verify phase `path.len()` of the attention point of a response; `path` holds the child drawn
/// after each earlier phase and `parent_root` the partials root the previous phase committed.
pub fn verify_attn_phase(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    path: &[u64],
    parent_root: Option<&Hash>,
    op: &AttnOpening,
) -> Result<AttnStep, Fail> {
    let s = attn_site(tier, ctx, resp, seed, op)?;
    let depth = s.tree.depth() as usize;
    if op.path != path || path.len() > depth {
        return Err(Fail::Shape);
    }
    let parent = match (path.split_last(), parent_root) {
        (None, None) if op.parent.is_empty() && op.parent_path.is_empty() => None,
        (Some((&i, up)), Some(root)) => {
            let n = s.children(up).ok_or(Fail::Shape)?.len() as u64;
            Some(open_partial(root, n, i, &op.parent, &op.parent_path, s.dv).ok_or(Fail::Trace)?)
        }
        _ => return Err(Fail::Shape),
    };
    let last = path.len() == depth;
    let mut keep = AttnOpening {
        response_txid: op.response_txid,
        pos_root: op.pos_root,
        seq_root: op.seq_root,
        layer: op.layer.clone(),
        path: op.path.clone(),
        parent: op.parent.clone(),
        parent_path: op.parent_path.clone(),
        sel: op.sel.clone(),
        ..Default::default()
    };
    if path.is_empty() || last {
        keep.q_entry = op.q_entry.clone();
        keep.pos_path = op.pos_path.clone();
        keep.q = op.q.clone();
    }
    if path.is_empty() {
        keep.o_entry = op.o_entry.clone();
        keep.out = op.out.clone();
        keep.lse = op.lse.clone();
    }
    if last {
        keep.norms = op.norms.clone();
        keep.kcache = op.kcache.clone();
        keep.vcache = op.vcache.clone();
        keep.anchor_pos_path = op.anchor_pos_path.clone();
        keep.anchor_k = op.anchor_k.clone();
        keep.anchor_v = op.anchor_v.clone();
        keep.absorb = op.absorb.clone();
    } else {
        keep.partials = op.partials.clone();
    }
    if *op != keep {
        return Err(Fail::Shape);
    }
    let children = if last {
        None
    } else {
        let n = s.children(path).ok_or(Fail::Shape)?.len();
        Some(decode_partials(&op.partials, s.dv).filter(|p| p.len() == n).ok_or(Fail::Shape)?)
    };
    match (&parent, &children) {
        (Some(p), Some(ch)) => check_node(p, ch).then_some(()).ok_or(Fail::Mismatch)?,
        _ if s.a.mla_rank != 0 => mla_phase(tier, ctx, resp, op, &s, path, parent.as_ref(), children.as_deref())?,
        _ => gqa_phase(tier, ctx, resp, op, &s, path, parent.as_ref(), children.as_deref())?,
    }
    Ok(match children {
        None => AttnStep::Done,
        Some(ch) => AttnStep::Next { root: partials_root(&op.partials, s.dv).ok_or(Fail::Shape)?, n: ch.len() as u64, weights: child_weights(&ch).ok_or(Fail::Shape)? },
    })
}

/// Attended query of the drawn head of the attention point of `seed` and its score scale, from
/// the query fields of `op`: the query entry, norms, query chunks at the drawn position and their
/// path, and the latent query of a latent-cache layer.
pub fn attn_query(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &AttnOpening) -> Result<(Vec<f64>, f64), Fail> {
    let s = attn_site(tier, ctx, resp, seed, op)?;
    let q = if s.a.mla_rank != 0 { mla_query(tier, ctx, resp, op, &s)? } else { gqa_query(tier, ctx, resp, op, &s)? };
    Ok((q, f32::from_bits(s.a.kq_scale) as f64))
}

pub(crate) fn gqa_shape(a: &manifest::AttnLinks) -> crate::attn::AttnShape {
    crate::attn::AttnShape {
        n_head: a.n_head as usize,
        n_head_kv: a.n_head_kv as usize,
        head_dim: a.head_dim as usize,
        rope_kind: a.rope_kind,
        n_rot: a.n_rot as usize,
        rope_base: f32::from_bits(a.rope_base) as f64,
        kq_scale: f32::from_bits(a.kq_scale) as f64,
    }
}

/// Query norm, key norm and RoPE frequency weights a grouped-query layer declares.
type GqaNorms = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>);

fn gqa_norms(tier: &TierCtx, a: &manifest::AttnLinks, norms: &[(EntryOpening, ChunkRange)]) -> Result<GqaNorms, Fail> {
    let hd = a.head_dim as u64;
    let declared = [(a.q_norm, hd), (a.k_norm, hd), (a.rope_freqs, (a.n_rot / 2) as u64)];
    let declared: Vec<(u32, u64)> = declared.into_iter().filter(|&(i, _)| i != manifest::NONE).collect();
    if norms.len() != declared.len() {
        return Err(Fail::Shape);
    }
    let mut weights = declared.iter().zip(norms).map(|(&(i, k), w)| f32_weights(tier, w, i, k, 0, k)).collect::<Result<Vec<_>, _>>()?.into_iter();
    let mut take = |i: u32| if i != manifest::NONE { weights.next() } else { None };
    Ok((take(a.q_norm), take(a.k_norm), take(a.rope_freqs)))
}

/// Key/value head and query head of a grouped-query point.
fn gqa_heads(s: &AttnSite, g: usize) -> (usize, usize) {
    let kv = (s.d[2] % s.a.n_head_kv as u64) as usize;
    (kv, kv * g + (s.d[2] / s.a.n_head_kv as u64 % g as u64) as usize)
}

fn gqa_query(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, op: &AttnOpening, s: &AttnSite) -> Result<Vec<f64>, Fail> {
    let a = &s.a;
    let shape = gqa_shape(a);
    let (g, hd, qw) = (shape.group().ok_or(Fail::Manifest)?, shape.head_dim, 1 + a.gated as usize);
    let qe = open_entry(tier, &op.q_entry, a.q as u64)?;
    if qe.m as usize != shape.n_head * hd * qw {
        return Err(Fail::Manifest);
    }
    let (q_norm, k_norm, freqs) = gqa_norms(tier, a, &op.norms)?;
    let prep = crate::attn::Prep { q_norm: q_norm.as_deref(), k_norm: k_norm.as_deref(), v_norm: a.v_norm == 1, freqs: freqs.as_deref(), eps: ctx.eps };
    if !prep.fits(&shape) {
        return Err(Fail::Manifest);
    }
    let (pshape, q_leaf) = (shape_any(tier, ctx), 2 * a.q as u64 + 1);
    let root_t = anchor_root(&op.pos_root, resp.n_pos as u64, s.t, &pshape, q_leaf, &op.q, 4 * qe.m, &op.pos_path)?;
    let raw = head_from_chunks(&root_t, &pshape, q_leaf, &op.q, 4 * qe.m, gqa_heads(s, g).1 * hd * qw, hd)?;
    Ok(prep.query(&shape, &raw, s.t))
}

fn mla_query(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, op: &AttnOpening, s: &AttnSite) -> Result<Vec<f64>, Fail> {
    let a = &s.a;
    let (nh, hd, r) = (a.n_head as usize, a.head_dim as usize, a.mla_rank as usize);
    let rope = hd.checked_sub(r).filter(|_| r.is_multiple_of(32) && nh > 0).ok_or(Fail::Manifest)?;
    let qe = open_entry(tier, &op.q_entry, a.q as u64)?;
    if !qe.m.is_multiple_of(nh as u64) {
        return Err(Fail::Manifest);
    }
    let qh_w = (qe.m / nh as u64) as usize;
    let nope = qh_w.checked_sub(rope).filter(|&n| n > 0).ok_or(Fail::Manifest)?;
    let (pshape, q_leaf, qh) = (shape_any(tier, ctx), 2 * a.q as u64 + 1, (s.d[2] % nh as u64) as usize);
    let root_t = anchor_root(&op.pos_root, resp.n_pos as u64, s.t, &pshape, q_leaf, &op.q, 4 * qe.m, &op.pos_path)?;
    let mut q: Vec<f64> = head_from_chunks(&root_t, &pshape, 2 * a.absorb as u64 + 1, &op.absorb, 4 * (nh * r) as u64, qh * r, r)?
        .iter()
        .map(|&x| x as f64)
        .collect();
    if rope > 0 {
        q.extend(head_from_chunks(&root_t, &pshape, q_leaf, &op.q, 4 * qe.m, qh * qh_w + nope, rope)?.iter().map(|&x| x as f64));
    }
    Ok(q)
}

/// Layout of the attention-output input at the drawn position and the position root its opened
/// leaf reaches.
fn out_root(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, op: &AttnOpening, s: &AttnSite, rank: u32, wtype: u32) -> Result<(u8, Hash), Fail> {
    let layout = crate::verify::layout_for(crate::verify::expected_ubatch(s.t, resp.n_prompt), wtype);
    let root_t = b3_fold(shape_any(tier, ctx).n_leaves, 2 * rank as u64, blake3::hash(&op.out.bytes).as_bytes(), &op.out.leaf_path).ok_or(Fail::Trace)?;
    b3_verify(&op.pos_root, resp.n_pos as u64, s.t, &root_t, &op.pos_path).then_some((layout, root_t)).ok_or(Fail::Trace)
}

/// The output check (whole opening or root phase) or the leaf check of a grouped-query layer.
#[allow(clippy::too_many_arguments)]
fn gqa_phase(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    op: &AttnOpening,
    s: &AttnSite,
    path: &[u64],
    parent: Option<&Partial>,
    children: Option<&[Partial]>,
) -> Result<(), Fail> {
    use crate::attn::{ARITHS, Prep};
    let (a, n_pos) = (&s.a, resp.n_pos as u64);
    if a.absorb != manifest::NONE || a.out != manifest::NONE || !op.absorb.is_empty() {
        return Err(Fail::Shape);
    }
    let shape = gqa_shape(a);
    let g = shape.group().ok_or(Fail::Manifest)?;
    let hd = shape.head_dim;
    let (kv, qh) = gqa_heads(s, g);
    let qe = open_entry(tier, &op.q_entry, a.q as u64)?;
    let qw = 1 + a.gated as usize;
    if qe.m as usize != shape.n_head * hd * qw {
        return Err(Fail::Manifest);
    }
    let pshape = shape_any(tier, ctx);
    let q_leaf = 2 * a.q as u64 + 1;
    let (root_t, layout) = if path.is_empty() {
        let oe = open_entry(tier, &op.o_entry, s.ll.proj as u64)?;
        if oe.k as usize != shape.n_head * hd {
            return Err(Fail::Manifest);
        }
        let (layout, root_t) = out_root(tier, ctx, resp, op, s, s.ll.proj, oe.wtype)?;
        if op.out.bytes.len() as u64 != crate::verify::x_len(oe.k, layout) {
            return Err(Fail::Shape);
        }
        (root_t, layout)
    } else {
        (anchor_root(&op.pos_root, n_pos, s.t, &pshape, q_leaf, &op.q, 4 * qe.m, &op.pos_path)?, 0)
    };
    if let Some(ch) = children {
        let gate = if qw > 1 { Some(head_from_chunks(&root_t, &pshape, q_leaf, &op.q, 4 * qe.m, qh * hd * qw + hd, hd)?) } else { None };
        let mass = committed_lse(ctx, op, s, n_pos, kv as u64, g, qh - kv * g)?;
        return check_root_lse(&op.out.bytes, layout, shape.n_head * hd, qh, ch, gate.as_deref(), mass).then_some(()).ok_or(Fail::Mismatch);
    }
    let (q_norm, k_norm, freqs) = gqa_norms(tier, a, &op.norms)?;
    let prep = Prep { q_norm: q_norm.as_deref(), k_norm: k_norm.as_deref(), v_norm: a.v_norm == 1, freqs: freqs.as_deref(), eps: ctx.eps };
    if !prep.fits(&shape) {
        return Err(Fail::Manifest);
    }
    let (p0, p1) = s.node(path).ok_or(Fail::Shape)?;
    let sl = seq_layout(ctx);
    let krows = cache_rows(ctx, &op.seq_root, sl.kcache(s.slot, kv as u64), n_pos, hd, p0, p1, &op.kcache)?;
    let vrows = cache_rows(ctx, &op.seq_root, sl.vcache(s.slot, kv as u64), n_pos, hd, p0, p1, &op.vcache)?;
    let keys = krows.iter().map(|r| dequant_head(r, hd)).collect::<Result<Vec<_>, _>>()?;
    let values = vrows.iter().map(|r| dequant_head(r, hd)).collect::<Result<Vec<_>, _>>()?;
    let ok = match parent {
        Some(p) => check_leaf(&gqa_query(tier, ctx, resp, op, s)?, shape.kq_scale, &keys, &values, p),
        None => {
            let raw = head_from_chunks(&root_t, &pshape, q_leaf, &op.q, 4 * qe.m, qh * hd * qw, hd * qw)?;
            let q = prep.query(&shape, &raw[..hd], s.t);
            let gate = (qw > 1).then(|| &raw[hd..]);
            let mass = committed_lse(ctx, op, s, n_pos, kv as u64, g, qh - kv * g)?;
            ARITHS.iter().any(|&ar| {
                crate::attn::partial(ar, &q, shape.kq_scale, &keys, &values).is_some_and(|p| check_root_lse(&op.out.bytes, layout, shape.n_head * hd, qh, &[p], gate, mass))
            })
        }
    };
    if !ok {
        return Err(Fail::Mismatch);
    }
    let i = (s.d[3] % (p1 - p0)) as usize;
    let ps = p0 + i as u64;
    let kv_total = 4 * (shape.n_head_kv * hd) as u64;
    let root_ps = anchor_root(&op.pos_root, n_pos, ps, &pshape, 2 * a.k as u64 + 1, &op.anchor_k, kv_total, &op.anchor_pos_path)?;
    let k_proj = head_from_chunks(&root_ps, &pshape, 2 * a.k as u64 + 1, &op.anchor_k, kv_total, kv * hd, hd)?;
    let v_proj = if a.v == a.k {
        if !op.anchor_v.is_empty() {
            return Err(Fail::Shape);
        }
        k_proj.clone()
    } else {
        head_from_chunks(&root_ps, &pshape, 2 * a.v as u64 + 1, &op.anchor_v, kv_total, kv * hd, hd)?
    };
    let k_c = prep.key(&shape, &k_proj, ps);
    let v_c = prep.value(&v_proj);
    (q8_0_matches(krows[i], &k_c) && q8_0_matches(vrows[i], &v_c)).then_some(()).ok_or(Fail::Mismatch)
}

/// The output check (whole opening or root phase) or the leaf check of a latent-cache layer: one
/// drawn query head against the shared latent cache.
#[allow(clippy::too_many_arguments)]
fn mla_phase(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    op: &AttnOpening,
    s: &AttnSite,
    path: &[u64],
    parent: Option<&Partial>,
    children: Option<&[Partial]>,
) -> Result<(), Fail> {
    use crate::attn::ARITHS;
    let (a, n_pos) = (&s.a, resp.n_pos as u64);
    let (nh, hd, r) = (a.n_head as usize, a.head_dim as usize, a.mla_rank as usize);
    let rope = hd.checked_sub(r).filter(|_| r.is_multiple_of(32)).ok_or(Fail::Manifest)?;
    let none = manifest::NONE;
    if a.n_head_kv != 1 || a.n_rot != 0 || a.swa_window != 0 || a.gated != 0 || a.v_norm != 0 || a.q_norm != none || a.rope_freqs != none {
        return Err(Fail::Unsupported);
    }
    if a.v != none || a.k_norm == none || a.absorb == none || nh == 0 {
        return Err(Fail::Manifest);
    }
    let qe = open_entry(tier, &op.q_entry, a.q as u64)?;
    if !qe.m.is_multiple_of(nh as u64) {
        return Err(Fail::Manifest);
    }
    (qe.m / nh as u64).checked_sub(rope as u64).filter(|&n| n > 0).ok_or(Fail::Manifest)?;
    if !op.anchor_v.is_empty() || op.vcache != crate::seq::SeqSpan::default() {
        return Err(Fail::Shape);
    }
    let pshape = shape_any(tier, ctx);
    let qh = (s.d[2] % nh as u64) as usize;
    let out = if path.is_empty() {
        let out_rank = if a.out == none { s.ll.proj } else { a.out };
        let oe = open_entry(tier, &op.o_entry, out_rank as u64)?;
        if oe.k as usize != r || oe.n_exp as usize != nh || oe.role != manifest::ROLE_HEADS {
            return Err(Fail::Manifest);
        }
        let (layout, _) = out_root(tier, ctx, resp, op, s, out_rank, oe.wtype)?;
        let x_one = crate::verify::x_len(oe.k, layout) as usize;
        if op.out.bytes.len() != nh * x_one {
            return Err(Fail::Shape);
        }
        Some((&op.out.bytes[qh * x_one..(qh + 1) * x_one], layout))
    } else {
        None
    };
    if let Some(ch) = children {
        let (x, layout) = out.ok_or(Fail::Shape)?;
        let mass = committed_lse(ctx, op, s, n_pos, 0, nh, qh)?;
        return check_root_lse(x, layout, r, 0, ch, None, mass).then_some(()).ok_or(Fail::Mismatch);
    }
    let [w] = <[(EntryOpening, ChunkRange); 1]>::try_from(op.norms.clone()).map_err(|_| Fail::Shape)?;
    let k_norm = f32_weights(tier, &w, a.k_norm, r as u64, 0, r as u64)?;
    let q = mla_query(tier, ctx, resp, op, s)?;
    let (p0, p1) = s.node(path).ok_or(Fail::Shape)?;
    let rows = cache_rows(ctx, &op.seq_root, seq_layout(ctx).kcache(s.slot, 0), n_pos, hd, p0, p1, &op.kcache)?;
    let all = rows.iter().map(|row| dequant_head(row, hd)).collect::<Result<Vec<_>, _>>()?;
    let keys: Vec<Vec<f32>> = all.into_iter().zip(p0..p1).filter(|(_, p)| s.attended(*p)).map(|(k, _)| k).collect();
    let values: Vec<Vec<f32>> = keys.iter().map(|k| k[..r].to_vec()).collect();
    let scale = f32::from_bits(a.kq_scale) as f64;
    let ok = match (parent, out) {
        (Some(p), _) => check_leaf(&q, scale, &keys, &values, p),
        (None, Some((x, layout))) => {
            let mass = committed_lse(ctx, op, s, n_pos, 0, nh, qh)? as f64;
            ARITHS.iter().any(|&ar| {
                crate::attn::partial(ar, &q, scale, &keys, &values)
                    .is_some_and(|p| (lse(&p) - mass).abs() <= root_lse_tol(p.m) && crate::attn_blocks::root_within(x, layout, r, 0, &p.o))
            })
        }
        (None, None) => false,
    };
    if !ok {
        return Err(Fail::Mismatch);
    }
    let i = (s.d[3] % (p1 - p0)) as usize;
    let ps = p0 + i as u64;
    let root_ps = anchor_root(&op.pos_root, n_pos, ps, &pshape, 2 * a.k as u64 + 1, &op.anchor_k, 4 * hd as u64, &op.anchor_pos_path)?;
    let kv_a = head_from_chunks(&root_ps, &pshape, 2 * a.k as u64 + 1, &op.anchor_k, 4 * hd as u64, 0, hd)?;
    let mut key: Vec<f64> = kv_a.iter().map(|&x| x as f64).collect();
    crate::attn::rms_norm(&mut key[..r], Some(&k_norm), ctx.eps);
    q8_0_matches(rows[i], &key).then_some(()).ok_or(Fail::Mismatch)
}

/// Sequence tree layout of a tier.
/// Committed log mass of query head `qi` of the group of `g` sharing key/value head `kv`, at the
/// drawn position.
#[allow(clippy::too_many_arguments)]
fn committed_lse(ctx: &ChainCtx, op: &AttnOpening, s: &AttnSite, n_pos: u64, kv: u64, g: usize, qi: usize) -> Result<f32, Fail> {
    let sl = seq_layout(ctx);
    let leaf = sl.lse(s.slot, kv).ok_or(Fail::Shape)?;
    let lo = (s.t * g as u64 + qi as u64) * 4;
    let b = crate::seq::open_span(&op.seq_root, &sl, leaf, n_pos * g as u64 * 4, lo, lo + 4, &op.lse).ok_or(Fail::Trace)?;
    Ok(f32::from_le_bytes(b.try_into().map_err(|_| Fail::Shape)?))
}

pub fn seq_layout(ctx: &ChainCtx) -> crate::seq::SeqLayout {
    crate::seq::SeqLayout { n_attn: ctx.n_attn as u64, n_recur: ctx.n_recur as u64, heads: ctx.seq_heads as u64, n_comp: ctx.n_comp as u64 }
}

/// Q8_0 head rows of positions `[p0, p1)` of the cache sequence leaf `leaf`, from `span`.
#[allow(clippy::too_many_arguments)]
fn cache_rows<'a>(ctx: &ChainCtx, seq_root: &Hash, leaf: Option<u64>, n_pos: u64, hd: usize, p0: u64, p1: u64, span: &'a crate::seq::SeqSpan) -> Result<Vec<&'a [u8]>, Fail> {
    let bb = (hd / 32 * 34) as u64;
    let leaf = leaf.ok_or(Fail::Shape)?;
    let bytes = crate::seq::open_span(seq_root, &seq_layout(ctx), leaf, n_pos * bb, p0 * bb, p1 * bb, span).ok_or(Fail::Trace)?;
    Ok(bytes.chunks_exact(bb as usize).collect())
}

fn dequant_head(row: &[u8], hd: usize) -> Result<Vec<f32>, Fail> {
    quant::dequant_row(quant::Q8_0, row, hd).ok_or(Fail::Shape)
}

/// Root of position `p` reached by the first of `chunks` of the f32 leaf `leaf`, checked against
/// the position tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchor_root(pos_root: &Hash, n_pos: u64, p: u64, shape: &PosShape, leaf: u64, chunks: &[LeafChunk], total: u64, path: &[Hash]) -> Result<Hash, Fail> {
    let c = chunks.first().ok_or(Fail::Shape)?;
    let h = chunk_leaf(c, total, c.index as u64)?;
    let root = b3_fold(shape.n_leaves, leaf, &h, &c.leaf_path).ok_or(Fail::Trace)?;
    b3_verify(pos_root, n_pos, p, &root, path).then_some(root).ok_or(Fail::Trace)
}

/// Values `[lo, lo + n)` of the f32 leaf `leaf` at the position of `root`, from its opened chunks.
pub(crate) fn head_from_chunks(root: &Hash, shape: &PosShape, leaf: u64, chunks: &[LeafChunk], total: u64, lo: usize, n: usize) -> Result<Vec<f32>, Fail> {
    let mut out = vec![0f32; n];
    let mut covered = vec![false; n];
    for c in chunks {
        let vals = leaf_chunk(root, shape, leaf, c, total, c.index as u64)?;
        let base = c.index as usize * B3_CHUNK / 4;
        for (j, v) in vals.iter().enumerate() {
            if let Some(i) = (base + j).checked_sub(lo).filter(|&i| i < n) {
                out[i] = *v;
                covered[i] = true;
            }
        }
    }
    covered.iter().all(|&c| c).then_some(out).ok_or(Fail::Shape)
}

/// Whether the Q8_0 blocks `b` are a canonical quantization of `v` within one step per value.
fn q8_0_matches(b: &[u8], v: &[f64]) -> bool {
    if b.len() != v.len().div_ceil(32) * 34 {
        return false;
    }
    let vmax = v.iter().fold(0f64, |m, a| m.max(a.abs()));
    b.chunks_exact(34).zip(v.chunks(32)).all(|(blk, vals)| {
        let d = crate::f16::read(blk, 0);
        let q: [i8; 32] = std::array::from_fn(|j| blk[2 + j] as i8);
        if !d.is_finite() || !q8::canonical(d, &q) {
            return false;
        }
        if d == 0.0 {
            return vals.iter().all(|a| a.abs() <= 1e-6 * vmax);
        }
        vals.iter().enumerate().all(|(j, a)| (q[j] as f64 - a / d as f64).abs() <= THRESHOLD)
    })
}

/// Leaf layout of a position, independent of its token.
pub(crate) fn shape_any(tier: &TierCtx, ctx: &ChainCtx) -> PosShape {
    shape(tier, ctx, 0)
}

