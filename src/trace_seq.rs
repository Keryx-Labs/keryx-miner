//! Attention and recurrent openings of a response that commits sequence leaves.

use crate::llama_engine::SeqCapture;
use crate::pom::WeightIndex;
use crate::trace::{bytes_range, leaf_bytes, leaf_chunk, Captures, SeqTrace, TierManifest, TraceRecord};
use keryx_verifiable_inference::attn_blocks::{block_params, partial_len, partial_opening, BlockTree};
use keryx_verifiable_inference::chain::{self, ATTN_POINT};
use keryx_verifiable_inference::draw;
use keryx_verifiable_inference::manifest::{self, AttnLinks, Chain, LayerLinks, RecurLinks, RECUR_KDA};
use keryx_verifiable_inference::merkle::{b3_chunk_cv, b3_path, b3_range_proof, b3_root, Hash, B3_CHUNK};
use keryx_verifiable_inference::opening::{AttnOpening, LeafChunk, RecurOpening, RecurTap};
use keryx_verifiable_inference::quant::{dequant_row, Q8_0};
use keryx_verifiable_inference::recur::{RECUR_POINT, RECUR_ROWS};
use keryx_verifiable_inference::seq::{self, SeqSpan};
use keryx_verifiable_inference::verify::ResponseCtx;

const C: u64 = B3_CHUNK as u64;

/// A sequence leaf span to capture: the leaf and the whole chunks covering `[lo, hi)` of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanPlan {
    pub leaf: u32,
    pub total: u64,
    pub lo: u64,
    pub hi: u64,
}

impl SpanPlan {
    fn new(leaf: u64, total: u64, lo: u64, hi: u64) -> Option<Self> {
        (lo < hi && hi <= total).then(|| Self { leaf: leaf as u32, total, lo: lo / C * C, hi: total.min(hi.div_ceil(C) * C) })
    }

    /// The `(leaf, lo, hi)` the engine captures.
    pub fn capture(&self) -> (u32, u64, u64) {
        (self.leaf, self.lo, self.hi)
    }

    fn n_chunks(&self) -> u64 {
        self.total.div_ceil(C).max(1)
    }
}

/// Largest chunk count among `plans`, the capacity each capture's chaining values need.
pub fn max_chunks(plans: &[SpanPlan]) -> u32 {
    plans.iter().map(|p| p.n_chunks()).max().unwrap_or(1) as u32
}

/// The span of `plan` from its capture, checked against the committed sequence leaf.
fn span(plan: &SpanPlan, cap: &SeqCapture, seq: &SeqTrace) -> Result<SeqSpan, String> {
    let leaf_hash = *seq.leaves.get(plan.leaf as usize).ok_or("no such sequence leaf")?;
    if cap.bytes.len() as u64 != plan.hi - plan.lo {
        return Err(format!("leaf {}: {} bytes captured for {}..{}", plan.leaf, cap.bytes.len(), plan.lo, plan.hi));
    }
    let n = plan.n_chunks();
    let (first, count) = (plan.lo / C, (plan.hi - plan.lo).div_ceil(C));
    let chunk_proof = if n == 1 {
        if *blake3::hash(&cap.bytes).as_bytes() != leaf_hash {
            return Err(format!("leaf {}: the replayed single chunk does not hash to the committed leaf", plan.leaf));
        }
        Vec::new()
    } else {
        if cap.cvs.len() as u64 != n {
            return Err(format!("leaf {}: {} chunk values captured for {} chunks", plan.leaf, cap.cvs.len(), n));
        }
        if b3_root(&cap.cvs) != leaf_hash {
            return Err(format!("leaf {}: the replayed chunks do not hash to the committed leaf", plan.leaf));
        }
        if let Some(i) = cap.bytes.chunks(B3_CHUNK).enumerate().position(|(i, c)| b3_chunk_cv(c, first + i as u64) != cap.cvs[(first + i as u64) as usize]) {
            return Err(format!("leaf {}: captured chunk {} does not match its chaining value", plan.leaf, first + i as u64));
        }
        b3_range_proof(&cap.cvs, first, count)
    };
    Ok(SeqSpan { first_chunk: first as u32, bytes: cap.bytes.clone(), chunk_proof, leaf_path: b3_path(&seq.leaves, plan.leaf as usize) })
}

/// Chunks of the f32 leaf `leaf` at position `t` covering its values `[lo, lo + n)`.
fn value_chunks(rec: &TraceRecord, t: u64, leaf: u32, bytes: &[u8], lo: u64, n: u64) -> Option<Vec<LeafChunk>> {
    let (c0, c1) = (4 * lo / C, (4 * (lo + n) - 1) / C);
    (c0..=c1).map(|c| leaf_chunk(rec, t, leaf, bytes, c)).collect()
}

fn layer_links(m: &TierManifest, layer: u64) -> Option<LayerLinks> {
    match m.entry(m.chain?.layer_base as u64 + layer).chain {
        Chain::Layer(ll) => Some(ll),
        _ => None,
    }
}

fn find_layer<T>(m: &TierManifest, pick: impl Fn(&LayerLinks) -> Option<T>) -> Option<(u64, LayerLinks, T)> {
    let ctx = m.chain?;
    (0..ctx.n_layers as u64).find_map(|l| {
        let ll = layer_links(m, l)?;
        pick(&ll).map(|x| (l, ll, x))
    })
}

/// Dequantized cache rows, one per position.
pub type Rows = Vec<Vec<f32>>;

/// What the attention point of a seed opens.
#[derive(Clone, Copy, Debug)]
pub struct AttnDraw {
    pub layer: u64,
    pub slot: u64,
    pub t: u64,
    /// Key/value head, or the query head of a latent-cache layer.
    pub head: u64,
    /// Query head whose partials the phases open.
    pub qh: u64,
    pub start: u64,
    pub tree: BlockTree,
    /// Values per partial.
    pub dv: usize,
    anchor_draw: u64,
    pub links: AttnLinks,
    pub proj: u32,
}

/// The attention point of `seed`, drawn as the node draws it.
pub fn draw_attn(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<AttnDraw> {
    let ctx = m.chain?;
    if ctx.n_attn == 0 {
        return None;
    }
    let d = draw::point(seed, ATTN_POINT);
    let slot = d[1] % ctx.n_attn as u64;
    let (layer, ll, a) = find_layer(m, |ll| ll.attn.filter(|a| a.slot as u64 == slot))?;
    let t = chain::resid_position(&ctx, &rec.response_ctx(), layer, d[0]).ok()?;
    let mla = a.mla_rank > 0;
    let (head, qh) = if mla {
        (d[2] % a.n_head as u64, d[2] % a.n_head as u64)
    } else {
        let (nkv, g) = (a.n_head_kv as u64, (a.n_head / a.n_head_kv) as u64);
        let kv = d[2] % nkv;
        (kv, kv * g + d[2] / nkv % g)
    };
    let start = if !mla && a.swa_window > 0 { (t + 1).saturating_sub(a.swa_window as u64) } else { 0 };
    let row = a.head_dim as usize / 32 * 34;
    let (row, dv) = if mla { (row, a.mla_rank as usize) } else { (2 * row, a.head_dim as usize) };
    let (c, b) = block_params(row, partial_len(dv))?;
    let tree = BlockTree::new(start, t + 1, c, b)?;
    let proj = if mla && a.out != manifest::NONE { a.out } else { ll.proj };
    Some(AttnDraw { layer, slot, t, head, qh, start, tree, dv, anchor_draw: d[3], links: a, proj })
}

impl AttnDraw {
    /// Whether the phase of `path` opens a leaf.
    pub fn leaf(&self, path: &[u64]) -> bool {
        path.len() as u32 == self.tree.depth()
    }

    /// Anchor position of the leaf of `path`.
    pub fn anchor(&self, path: &[u64]) -> Option<u64> {
        let (p0, p1) = self.tree.node(path)?;
        Some(p0 + self.anchor_draw % (p1 - p0))
    }

    /// The `(position, leaf)` captures the phase of `path` needs: the query, the output at the
    /// first phase, the anchor at the last.
    pub fn captures(&self, path: &[u64]) -> Option<Vec<(u32, u32)>> {
        let a = &self.links;
        let t = self.t as u32;
        let mut out = vec![(t, 2 * a.q + 1)];
        if path.is_empty() {
            out.push((t, 2 * self.proj));
        }
        if a.mla_rank > 0 {
            out.push((t, 2 * a.absorb + 1));
        }
        if self.leaf(path) {
            let ps = self.anchor(path)? as u32;
            out.push((ps, 2 * a.k + 1));
            if a.mla_rank == 0 && a.v != a.k {
                out.push((ps, 2 * a.v + 1));
            }
        }
        Some(out)
    }

    /// The cache spans of the positions of the node of `path`: keys, then values (none for a
    /// latent cache).
    pub fn spans(&self, m: &TierManifest, n_pos: u64, path: &[u64]) -> Option<Vec<SpanPlan>> {
        let layout = chain::seq_layout(&m.chain?);
        let a = &self.links;
        let bb = (a.head_dim / 32 * 34) as u64;
        let kv = if a.mla_rank > 0 { 0 } else { self.head };
        let (p0, p1) = self.tree.node(path)?;
        let (lo, hi, total) = (p0 * bb, p1 * bb, n_pos * bb);
        let mut out = vec![SpanPlan::new(layout.kcache(self.slot, kv)?, total, lo, hi)?];
        if a.mla_rank == 0 {
            out.push(SpanPlan::new(layout.vcache(self.slot, kv)?, total, lo, hi)?);
        }
        if path.is_empty() {
            let (g, qi) = if a.mla_rank > 0 { (a.n_head as u64, self.qh) } else { let g = (a.n_head / a.n_head_kv) as u64; (g, self.qh - kv * g) };
            let at = (self.t * g + qi) * 4;
            out.push(SpanPlan::new(layout.lse(self.slot, kv)?, n_pos * g * 4, at, at + 4)?);
        }
        Some(out)
    }

    /// Dequantized keys and values of the node of `path`, from the cache spans of `op`.
    pub fn rows(&self, op: &AttnOpening, path: &[u64]) -> Option<(Rows, Rows)> {
        let hd = self.links.head_dim as usize;
        let bb = hd / 32 * 34;
        let (p0, p1) = self.tree.node(path)?;
        let rows = |s: &SeqSpan| -> Option<Rows> {
            let off = (p0 as usize * bb).checked_sub(s.first_chunk as usize * B3_CHUNK)?;
            let b = s.bytes.get(off..off + (p1 - p0) as usize * bb)?;
            b.chunks_exact(bb).map(|r| dequant_row(Q8_0, r, hd)).collect()
        };
        let keys = rows(&op.kcache)?;
        let values = if self.links.mla_rank > 0 { keys.iter().map(|k| k[..self.dv].to_vec()).collect() } else { rows(&op.vcache)? };
        Some((keys, values))
    }

    /// `op` reduced to the fields the phase of `path` carries, with `partials` before the last.
    pub fn phase_fields(&self, op: &AttnOpening, path: &[u64], partials: Vec<u8>) -> AttnOpening {
        if self.leaf(path) {
            return op.clone();
        }
        let base = AttnOpening {
            response_txid: op.response_txid,
            pos_root: op.pos_root,
            seq_root: op.seq_root,
            layer: op.layer.clone(),
            path: op.path.clone(),
            parent: op.parent.clone(),
            parent_path: op.parent_path.clone(),
            partials,
            ..Default::default()
        };
        if path.is_empty() {
            AttnOpening { q_entry: op.q_entry.clone(), o_entry: op.o_entry.clone(), pos_path: op.pos_path.clone(), q: op.q.clone(), out: op.out.clone(), lse: op.lse.clone(), ..base }
        } else {
            base
        }
    }
}

/// Draw weights of the children whose partials the attention phase `body` committed, as the node
/// computes them.
pub fn attn_child_weights(tier: u8, rec: &TraceRecord, seed: &Hash, body: &[u8]) -> Option<Vec<u16>> {
    use keryx_verifiable_inference::attn_blocks::{child_weights, decode_partials};
    use keryx_verifiable_inference::opening::inner_body;
    let m = TierManifest::for_attn(tier)?;
    let dv = draw_attn(&m, rec, seed)?.dv;
    let op = inner_body(body).as_deref().and_then(AttnOpening::decode)?;
    child_weights(&decode_partials(&op.partials, dv)?)
}

/// Children of the node of `path` of the attention point of `seed`.
pub fn attn_children(tier: u8, rec: &TraceRecord, seed: &Hash, path: &[u64]) -> Option<u64> {
    let m = TierManifest::for_attn(tier)?;
    Some(draw_attn(&m, rec, seed)?.tree.children(path)?.len() as u64)
}

/// Every field the phase of `path` of `d` may need: the query fields at every phase, the output
/// at the first, the anchor at the last, the cache spans of the node, and the parent from `prev`,
/// the partials of the previous phase. Partials are left to the caller.
#[allow(clippy::too_many_arguments)]
pub fn attn_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    index: &WeightIndex,
    d: &AttnDraw,
    caps: &Captures,
    spans: &[SeqCapture],
    response_txid: Hash,
    path: &[u64],
    prev: Option<&[u8]>,
) -> Result<AttnOpening, String> {
    let a = &d.links;
    let t = d.t;
    let cap = |pos: u64, leaf: u32| caps.get(&(pos as u32, leaf));
    let (hd, mla, leaf) = (a.head_dim as u64, a.mla_rank > 0, d.leaf(path));
    let ps = if leaf { Some(d.anchor(path).ok_or("no anchor")?) } else { None };
    let q_leaf = 2 * a.q + 1;
    let q_bytes = cap(t, q_leaf).ok_or("no query capture")?;
    let qe = m.entry(a.q as u64);
    let (q, absorb, anchor_k, anchor_v, norms) = if mla {
        let (nh, r) = (a.n_head as u64, a.mla_rank as u64);
        let qh_w = qe.m / nh;
        let rope = hd - r;
        let nope = qh_w - rope;
        let q = value_chunks(rec, t, q_leaf, q_bytes, d.head * qh_w + nope, rope).ok_or("query chunks")?;
        let absorb = value_chunks(rec, t, 2 * a.absorb + 1, cap(t, 2 * a.absorb + 1).ok_or("no latent query capture")?, d.head * r, r).ok_or("latent query chunks")?;
        let ak = match ps {
            Some(ps) => value_chunks(rec, ps, 2 * a.k + 1, cap(ps, 2 * a.k + 1).ok_or("no anchor capture")?, 0, hd).ok_or("anchor chunks")?,
            None => Vec::new(),
        };
        let norm = bytes_range(m, index, a.k_norm as u64, 0, 4 * r).ok_or("latent norm weights")?;
        (q, absorb, ak, Vec::new(), vec![norm])
    } else {
        let qw = 1 + a.gated as u64;
        let q = value_chunks(rec, t, q_leaf, q_bytes, d.qh * hd * qw, hd * qw).ok_or("query chunks")?;
        let (ak, av) = match ps {
            Some(ps) => {
                let ak = value_chunks(rec, ps, 2 * a.k + 1, cap(ps, 2 * a.k + 1).ok_or("no key anchor capture")?, d.head * hd, hd).ok_or("key anchor chunks")?;
                let av = if a.v == a.k {
                    Vec::new()
                } else {
                    value_chunks(rec, ps, 2 * a.v + 1, cap(ps, 2 * a.v + 1).ok_or("no value anchor capture")?, d.head * hd, hd).ok_or("value anchor chunks")?
                };
                (ak, av)
            }
            None => (Vec::new(), Vec::new()),
        };
        let declared = [(a.q_norm, hd), (a.k_norm, hd), (a.rope_freqs, (a.n_rot / 2) as u64)];
        let norms = declared
            .into_iter()
            .filter(|&(i, _)| i != manifest::NONE)
            .map(|(i, k)| bytes_range(m, index, i as u64, 0, 4 * k))
            .collect::<Option<Vec<_>>>()
            .ok_or("norm weights")?;
        (q, Vec::new(), ak, av, norms)
    };
    let plans = d.spans(m, rec.n_pos as u64, path).ok_or("no cache spans")?;
    if spans.len() != plans.len() {
        return Err(format!("{} spans captured for {}", spans.len(), plans.len()));
    }
    let kcache = span(&plans[0], &spans[0], seq)?;
    let vcache = if mla { SeqSpan::default() } else { span(&plans[1], &spans[1], seq)? };
    let lse = if path.is_empty() { span(plans.last().ok_or("no mass span")?, spans.last().ok_or("no mass capture")?, seq)? } else { SeqSpan::default() };
    let (parent, parent_path) = match (path.last(), prev) {
        (Some(&i), Some(b)) => partial_opening(b, d.dv, i as usize).ok_or("no parent partial")?,
        (None, None) => Default::default(),
        _ => return Err("the parent partials do not match the path".into()),
    };
    let base = m.chain.ok_or("no chain context")?.layer_base as u64;
    Ok(AttnOpening {
        response_txid,
        pos_root: rec.trace_root,
        seq_root: seq.root,
        layer: m.entry_opening(base + d.layer),
        q_entry: m.entry_opening(a.q as u64),
        o_entry: if path.is_empty() { m.entry_opening(d.proj as u64) } else { Default::default() },
        norms,
        pos_path: rec.pos_path(t),
        q,
        out: if path.is_empty() { leaf_bytes(rec, t, 2 * d.proj, cap(t, 2 * d.proj).ok_or("no output capture")?) } else { Default::default() },
        kcache,
        vcache,
        anchor_pos_path: ps.map(|ps| rec.pos_path(ps)).unwrap_or_default(),
        anchor_k,
        anchor_v,
        absorb,
        path: path.to_vec(),
        parent,
        parent_path,
        partials: Vec::new(),
        lse,
    })
}

/// What the recurrent point of a seed opens.
#[derive(Clone, Copy, Debug)]
pub struct RecurDraw {
    pub layer: u64,
    pub slot: u64,
    pub head: u64,
    pub segment: u64,
    pub first: u64,
    pub end: u64,
    pub row0: u64,
    pub anchor: u64,
    pub links: RecurLinks,
    pub proj: u32,
}

/// The recurrent point of `seed`, drawn as the node draws it.
pub fn draw_recur(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<RecurDraw> {
    let ctx = m.chain?;
    if ctx.n_recur == 0 {
        return None;
    }
    let n_pos = rec.n_pos as u64;
    let d = draw::point(seed, RECUR_POINT);
    let slot = d[1] % ctx.n_recur as u64;
    let (layer, ll, r) = find_layer(m, |ll| ll.recur.filter(|r| r.slot as u64 == slot))?;
    let iv = r.interval as u64;
    let head = d[2] % r.n_v_heads as u64;
    let segment = d[0] % n_pos.div_ceil(iv);
    let (first, end) = (segment * iv, ((segment + 1) * iv).min(n_pos));
    let row0 = d[3] % (r.head_dim as u64 / RECUR_ROWS) * RECUR_ROWS;
    let anchor = first + (d[3] >> 32) % (end - first);
    Some(RecurDraw { layer, slot, head, segment, first, end, row0, anchor, links: r, proj: ll.proj })
}

impl RecurDraw {
    fn kda(&self) -> bool {
        self.links.kind == RECUR_KDA
    }

    /// Value offsets of the query, key and value channels of the drawn head in their projections.
    fn offsets(&self) -> [u64; 3] {
        let r = &self.links;
        let (h, hd, nk) = (self.head, r.head_dim as u64, r.n_k_heads as u64);
        if self.kda() {
            [h * hd; 3]
        } else {
            let hk = h % nk;
            [hk * hd, (nk + hk) * hd, 2 * nk * hd + h * hd]
        }
    }

    fn taps(&self) -> std::ops::RangeInclusive<u64> {
        let n = (self.anchor + 1).min(self.links.conv_len as u64);
        self.anchor + 1 - n..=self.anchor
    }

    /// The `(position, leaf)` captures the opening needs.
    pub fn captures(&self) -> Vec<(u32, u32)> {
        let r = &self.links;
        let mut out = Vec::new();
        for p in self.taps() {
            for src in if self.kda() { vec![r.q, r.k, r.v] } else { vec![r.q] } {
                out.push((p as u32, 2 * src + 1));
            }
        }
        let p = self.anchor as u32;
        out.extend([(p, 2 * r.beta + 1), (p, 2 * r.decay + 1), (p, 2 * r.gate + 1), (p, 2 * self.proj)]);
        out
    }

    /// The sequence spans the opening needs: records, then the start and end state rows that exist.
    pub fn spans(&self, m: &TierManifest, n_pos: u64) -> Option<Vec<SpanPlan>> {
        let layout = chain::seq_layout(&m.chain?);
        let r = &self.links;
        let (hd, iv) = (r.head_dim as u64, r.interval as u64);
        let rl = seq::record_len(r.kind, hd as usize) as u64;
        let mut out = vec![SpanPlan::new(layout.record(self.slot, self.head)?, n_pos * rl, self.first * rl, self.end * rl)?];
        let ck = layout.checkpoint(self.slot, self.head)?;
        let ck_len = seq::checkpoint_len(hd, n_pos, iv);
        for (on, m_ck) in [(self.segment > 0, self.segment), (self.end - self.first == iv, self.segment + 1)] {
            if on {
                let (lo, hi) = seq::checkpoint_rows(hd, m_ck, self.row0, RECUR_ROWS);
                out.push(SpanPlan::new(ck, ck_len, lo, hi)?);
            }
        }
        Some(out)
    }
}

/// The recurrent opening of `d`.
#[allow(clippy::too_many_arguments)]
pub fn recur_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    index: &WeightIndex,
    d: &RecurDraw,
    caps: &Captures,
    spans: &[SeqCapture],
    response_txid: Hash,
) -> Result<RecurOpening, String> {
    let r = &d.links;
    let cap = |pos: u64, leaf: u32| caps.get(&(pos as u32, leaf));
    let (hd, cl) = (r.head_dim as u64, r.conv_len as u64);
    let offs = d.offsets();
    let srcs = [r.q, r.k, r.v];
    let mut taps = Vec::new();
    for p in d.taps() {
        let lc = |x: usize| -> Option<LeafChunk> {
            let leaf = 2 * srcs[x] + 1;
            leaf_chunk(rec, p, leaf, cap(p, leaf)?, 4 * offs[x] / C)
        };
        let (q, k, v) = (lc(0).ok_or("query tap")?, lc(1).ok_or("key tap")?, lc(2).ok_or("value tap")?);
        taps.push(RecurTap { pos_path: rec.pos_path(p), q, k, v });
    }
    let p = d.anchor;
    let gw = if d.kda() { hd } else { 1 };
    let chunk_at = |leaf: u32, off: u64| leaf_chunk(rec, p, leaf, cap(p, leaf)?, 4 * off / C);
    let conv_entries = [r.conv_q, r.conv_k, r.conv_v];
    let mut weights = (0..3)
        .map(|x| bytes_range(m, index, conv_entries[x] as u64, 4 * offs[x] * cl, 4 * hd * cl))
        .collect::<Option<Vec<_>>>()
        .ok_or("convolution weights")?;
    weights.push(bytes_range(m, index, r.dt as u64, 4 * d.head * gw, 4 * gw).ok_or("decay bias")?);
    weights.push(bytes_range(m, index, r.a as u64, 4 * d.head, 4).ok_or("decay scale")?);
    weights.push(bytes_range(m, index, r.norm as u64, 0, 4 * hd).ok_or("output norm")?);
    let plans = d.spans(m, rec.n_pos as u64).ok_or("no recurrent spans")?;
    if spans.len() != plans.len() {
        return Err(format!("{} spans captured for {}", spans.len(), plans.len()));
    }
    let mut it = plans.iter().zip(spans);
    let (rp, rc) = it.next().ok_or("no record span")?;
    let records = span(rp, rc, seq)?;
    let start_rows = if d.segment > 0 {
        let (sp, sc) = it.next().ok_or("no start rows")?;
        span(sp, sc, seq)?
    } else {
        SeqSpan::default()
    };
    let end_rows = if d.end - d.first == r.interval as u64 {
        let (sp, sc) = it.next().ok_or("no end rows")?;
        span(sp, sc, seq)?
    } else {
        SeqSpan::default()
    };
    let base = m.chain.ok_or("no chain context")?.layer_base as u64;
    Ok(RecurOpening {
        response_txid,
        pos_root: rec.trace_root,
        seq_root: seq.root,
        layer: m.entry_opening(base + d.layer),
        out_entry: m.entry_opening(d.proj as u64),
        decay_entry: m.entry_opening(r.decay as u64),
        weights,
        records,
        start_rows,
        end_rows,
        taps,
        beta: chunk_at(2 * r.beta + 1, d.head).ok_or("beta chunk")?,
        decay: chunk_at(2 * r.decay + 1, d.head * gw).ok_or("decay chunk")?,
        gate: chunk_at(2 * r.gate + 1, d.head * hd).ok_or("gate chunk")?,
        out: leaf_bytes(rec, p, 2 * d.proj, cap(p, 2 * d.proj).ok_or("no output capture")?),
    })
}

/// The response context the node checks attention and recurrent points against.
pub fn seq_response_ctx(rec: &TraceRecord, seq: &SeqTrace) -> ResponseCtx {
    ResponseCtx { trace_root: seq::trace_root(&rec.trace_root, &seq.root), n_prompt: rec.n_prompt, n_pos: rec.n_pos }
}

const BUNDLE: u8 = 0xB0;

/// Opening bodies sent as one transaction each, kept together until sent.
pub fn bundle(bodies: &[Vec<u8>]) -> Vec<u8> {
    if bodies.len() == 1 {
        return bodies[0].clone();
    }
    let mut out = vec![BUNDLE, bodies.len() as u8];
    for b in bodies {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

/// The bodies of [`bundle`].
pub fn unbundle(b: &[u8]) -> Option<Vec<Vec<u8>>> {
    if b.first() != Some(&BUNDLE) {
        return Some(vec![b.to_vec()]);
    }
    let n = *b.get(1)? as usize;
    let mut o = 2;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize;
        out.push(b.get(o + 4..o + 4 + len)?.to_vec());
        o += 4 + len;
    }
    (o == b.len()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_attention_layer_of_every_tier_opens_in_at_most_three_phases() {
        for tier in 0..5u8 {
            let m = TierManifest::for_attn(tier).unwrap();
            let ctx = m.chain.unwrap();
            let mut shapes = std::collections::BTreeSet::new();
            for l in 0..ctx.n_layers as u64 {
                let Some(a) = layer_links(&m, l).and_then(|ll| ll.attn) else { continue };
                let row = a.head_dim as usize / 32 * 34;
                let (row, dv) = if a.mla_rank > 0 { (row, a.mla_rank as usize) } else { (2 * row, a.head_dim as usize) };
                let (c, b) = block_params(row, partial_len(dv)).unwrap_or_else(|| panic!("tier {tier} layer {l}: no block parameters"));
                let start = 32_768u64.saturating_sub(if a.swa_window > 0 { a.swa_window as u64 } else { 32_768 });
                let depth = BlockTree::new(start, 32_768, c, b).unwrap().depth();
                assert!(depth <= 2, "tier {tier} layer {l}: depth {depth}");
                shapes.insert((a.head_dim, a.mla_rank, a.swa_window, c, b, depth + 1));
            }
            assert!(!shapes.is_empty(), "tier {tier}: no attention layer");
            eprintln!("tier {tier}: (head_dim, mla_rank, window, c, b, phases at 32k) {shapes:?}");
        }
    }

    #[test]
    fn bundles_round_trip_and_single_bodies_stay_bare() {
        let one = vec![vec![3u8, 1, 2]];
        assert_eq!(bundle(&one), one[0]);
        assert_eq!(unbundle(&one[0]), Some(one));
        let three = vec![vec![7u8; 10], vec![5u8; 3000], vec![6u8; 1]];
        let b = bundle(&three);
        assert_eq!(unbundle(&b), Some(three));
        assert!(unbundle(&b[..b.len() - 1]).is_none());
    }

    #[test]
    fn span_plans_cover_whole_chunks() {
        let p = SpanPlan::new(3, 5000, 1100, 2100).unwrap();
        assert_eq!((p.lo, p.hi), (1024, 3072));
        let q = SpanPlan::new(3, 5000, 4500, 5000).unwrap();
        assert_eq!((q.lo, q.hi), (4096, 5000));
        assert!(SpanPlan::new(3, 5000, 10, 10).is_none() && SpanPlan::new(3, 5000, 10, 5001).is_none());
    }
}
