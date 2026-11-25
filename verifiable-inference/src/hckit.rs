//! Two-layer synthetic tier with hyper-connection streams, and its stage-2 traces, for tests.

use crate::chain::{self, ChainCtx, Resid};
use crate::dot;
use crate::draw;
use crate::f16;
use crate::manifest::{self, Chain, Entry, HcLinks, LayerLinks, ProductLink};
use crate::merkle::{B3_CHUNK, Hash, b3_chunk_cv, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, FeedRow, HcAdd, HcSource, LeafBytes, LeafChunk, LinkOpening, Point, PrefixPoint, ResidAdd, ResidOpening, ResidPoint, ResidSrc};
use crate::prefix::{self, PrefixCtx};
use crate::q8;
use crate::quant;
use crate::testkit::{in_layout, layout_at, q8_1_row, serialize};
use crate::verify::{ResponseCtx, TierCtx};

pub const D: usize = 512;
pub const N_HC: usize = 4;
const K_HC: usize = N_HC * D;
const N_MIX: usize = (2 + N_HC) * N_HC;
const N_LAYERS: usize = 2;
const VOCAB: usize = 256;
const EPS: f64 = 1e-6;
const HC_EPS: f32 = 1e-6;
const ITERS: u32 = 20;

const PER_LAYER: usize = 4;
const HC_ATTN_FN: usize = 0;
const ATTN_X: usize = 1;
const FFN_X: usize = 2;
const HC_FFN_FN: usize = 3;
const N_PRODUCTS: usize = N_LAYERS * PER_LAYER;
const EMBD: usize = N_PRODUCTS;
const NORMS: usize = EMBD + 1;
const NORMS_PER_LAYER: usize = 4;
const SCALE: [f32; 3] = [0.5, 0.4, 0.6];
const LAYERS: usize = NORMS + N_LAYERS * NORMS_PER_LAYER;

fn rank(l: usize, p: usize) -> usize {
    PER_LAYER * l + p
}

/// Norm entries of layer `l`: attention norm, FFN norm, then the mix biases of each half.
fn norm(l: usize, i: usize) -> usize {
    NORMS + NORMS_PER_LAYER * l + i
}

fn q8_0_rows(seed: u64, rows: usize, k: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k / 32 * 34);
    for r in 0..rows {
        for b in 0..k / 32 {
            out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
            let h = blake3::hash(&[seed.to_le_bytes(), ((r * k / 32 + b) as u64).to_le_bytes()].concat());
            out.extend_from_slice(h.as_bytes());
        }
    }
    out
}

pub struct HcTier {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<(Entry, Vec<u8>)>,
    pub entry_leaves: Vec<Hash>,
    pub r_t: Hash,
    pub m_t: Hash,
    /// Single-pass mixes: each half reads the `pre` of the mix before it.
    pub lag: bool,
}

pub struct Position {
    pub tok: u32,
    pub leaves: Vec<Vec<u8>>,
    pub hashes: Vec<Hash>,
}

pub struct HcTrace {
    pub n_prompt: u32,
    pub positions: Vec<Position>,
    pub pos_roots: Vec<Hash>,
    pub trace_root: Hash,
}

impl HcTrace {
    /// Micro-batch size of each position.
    pub fn ubatch(&self) -> Vec<u32> {
        (0..self.positions.len() as u64).map(|t| layout_at(t, self.n_prompt).0).collect()
    }
}

/// What a trace computes dishonestly.
#[derive(Clone, Copy, Default)]
pub struct Faults {
    /// The residual combination uses `comb[src][dst]` instead of `comb[dst][src]`.
    pub transposed_comb: bool,
    /// The first input of a half reads the streams with `post` instead of `pre`.
    pub post_as_pre: bool,
    /// Under single-pass mixes, each half reads the `pre` of its own mix.
    pub own_pre: bool,
}

impl Default for HcTier {
    fn default() -> Self {
        Self::new()
    }
}

impl HcTier {
    pub fn new() -> Self {
        Self::with_lag(false)
    }

    /// A tier of single-pass mixes when `lag`.
    pub fn with_lag(lag: bool) -> Self {
        let mut weights: Vec<u8> = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();
        let push = |weights: &mut Vec<u8>, bytes: Vec<u8>| {
            let off = weights.len();
            weights.extend_from_slice(&bytes);
            while !weights.len().is_multiple_of(32) {
                weights.push(0);
            }
            (off, bytes.len())
        };
        let base = |name: String, role: u8, wtype: u32, k: usize, m: usize, off: usize, n_bytes: usize, rb: usize, chain: Chain| Entry {
            name,
            wtype,
            k: k as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base: (off / 32) as u64,
            n_bytes: n_bytes as u64,
            row_bytes: rb as u32,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain,
        };
        let link = |layer: usize, link: u8, norm: u32, src: u32| Chain::Product(ProductLink { layer: layer as u32, link, norm, src });
        for l in 0..N_LAYERS {
            for p in 0..PER_LAYER {
                let (name, k, m, chain) = match p {
                    ATTN_X => {
                        let src = match (lag, l) {
                            (false, _) => rank(l, HC_ATTN_FN) as u32,
                            (true, 0) => manifest::NONE,
                            (true, l) => rank(l - 1, HC_FFN_FN) as u32,
                        };
                        ("attn_x", D, D, link(l, manifest::LINK_HC_ATTN_PRE, norm(l, 0) as u32, src))
                    }
                    FFN_X => ("ffn_x", D, D, link(l, manifest::LINK_HC_FFN_PRE, norm(l, 1) as u32, rank(l, if lag { HC_ATTN_FN } else { HC_FFN_FN }) as u32)),
                    HC_ATTN_FN => ("hc_attn_fn", K_HC, N_MIX, link(l, manifest::LINK_HC_ATTN_IN, manifest::NONE, manifest::NONE)),
                    _ => ("hc_ffn_fn", K_HC, N_MIX, link(l, manifest::LINK_HC_FFN_IN, manifest::NONE, manifest::NONE)),
                };
                let scale = if p >= HC_ATTN_FN { 0.003 } else { 0.004 };
                let (off, n) = push(&mut weights, q8_0_rows(100 * (l as u64 + 1) + p as u64, m, k, scale));
                entries.push(base(format!("blk.{l}.{name}.weight"), manifest::ROLE_PRODUCT, quant::Q8_0, k, m, off, n, k / 32 * 34, chain));
            }
        }
        let (off, n) = push(&mut weights, q8_0_rows(3, VOCAB, D, 0.02));
        entries.push(base("token_embd.weight".into(), manifest::ROLE_EMBD, quant::Q8_0, D, VOCAB, off, n, D / 32 * 34, Chain::None));
        for l in 0..N_LAYERS {
            for (i, (name, k)) in [("attn_norm", D), ("ffn_norm", D), ("hc_attn_base", N_MIX), ("hc_ffn_base", N_MIX)].iter().enumerate() {
                let v: Vec<f32> = match (*name, *k) {
                    (_, N_MIX) => (0..N_MIX).map(|j| 0.3 * ((j * 7 + l + i) % 5) as f32 - 0.6).collect(),
                    _ => (0..*k).map(|j| 1.0 + 0.001 * ((j + 3 * i) % 17) as f32).collect(),
                };
                let (off, n) = push(&mut weights, v.iter().flat_map(|a| a.to_le_bytes()).collect());
                entries.push(base(format!("blk.{l}.{name}.weight"), manifest::ROLE_NORM, 0, *k, 1, off, n, 4 * k, Chain::None));
            }
        }
        for l in 0..N_LAYERS {
            let hc = HcLinks {
                attn_fn: rank(l, HC_ATTN_FN) as u32,
                ffn_fn: rank(l, HC_FFN_FN) as u32,
                attn_scale: SCALE.map(f32::to_bits),
                ffn_scale: SCALE.map(f32::to_bits),
                attn_base: norm(l, 2) as u32,
                ffn_base: norm(l, 3) as u32,
                n_hc: N_HC as u32,
                sinkhorn_iters: ITERS,
                eps: HC_EPS.to_bits(),
                lag: lag as u32,
                prev_fn: if lag && l > 0 { rank(l - 1, HC_FFN_FN) as u32 } else { manifest::NONE },
                prev_scale: if lag && l > 0 { SCALE.map(f32::to_bits) } else { [0; 3] },
                prev_base: if lag && l > 0 { norm(l - 1, 3) as u32 } else { manifest::NONE },
            };
            let ll = LayerLinks {
                layer: l as u32,
                proj: rank(l, ATTN_X) as u32,
                down: rank(l, FFN_X) as u32,
                post_attn: manifest::NONE,
                post_ffn: manifest::NONE,
                scale: manifest::NONE,
                moe: None,
                attn: None,
                recur: None,
                hc: Some(hc), dsa: None, comp: None, engram: None,
            };
            entries.push(base(format!("blk.{l}"), manifest::ROLE_LAYER, 0, 0, 0, 0, 0, 0, Chain::Layer(ll)));
        }
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let entries: Vec<(Entry, Vec<u8>)> = entries
            .into_iter()
            .map(|e| {
                let mut b = serialize(&e);
                b.extend_from_slice(&manifest::chain_bytes(&e.chain));
                (e, b)
            })
            .collect();
        let entry_leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        let m_t = rt_range_proof(&entry_leaves, 0, 1).0;
        Self { weights, chunk_leaves, entries, entry_leaves, r_t, m_t, lag }
    }

    pub fn tier(&self) -> TierCtx {
        TierCtx { r_t: self.r_t, n_chunks: self.chunk_leaves.len() as u64, m_t: self.m_t, n_entries: self.entries.len() as u64, n_products: N_PRODUCTS as u64 }
    }

    pub fn chain_ctx(&self) -> ChainCtx {
        ChainCtx { n_sel: 0, n_comp: 0, n_engram: 0,
            n_layers: N_LAYERS as u32,
            layer_base: LAYERS as u32,
            embd_entry: EMBD as u32,
            eps: EPS,
            embd_scale: 1.0,
            trim_last: false,
            n_used: 0,
            n_attn: 0,
            n_recur: 0,
            seq_heads: 0,
        }
    }

    fn entry(&self, i: usize) -> &Entry {
        &self.entries[i].0
    }

    fn tensor(&self, i: usize) -> &[u8] {
        let e = self.entry(i);
        let off = e.chunk_base as usize * 32;
        &self.weights[off..off + e.n_bytes as usize]
    }

    fn row(&self, i: usize, r: usize) -> &[u8] {
        let rb = self.entry(i).row_bytes as usize;
        &self.tensor(i)[r * rb..(r + 1) * rb]
    }

    fn f32_tensor(&self, i: usize) -> Vec<f32> {
        self.tensor(i).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn matvec(&self, i: usize, x: &[u8]) -> Vec<f32> {
        let k = self.entry(i).k as usize;
        (0..self.entry(i).m as usize).map(|r| dot::canonical_dot(quant::Q8_0, self.row(i, r), k, x, q8::MMVQ).unwrap().y_hat as f32).collect()
    }

    fn mix(&self, l: usize, half: Resid, mixes: &[f32]) -> chain::HcMix {
        let b = if half == Resid::Attn { norm(l, 2) } else { norm(l, 3) };
        chain::hc_mix(mixes, &SCALE, &self.f32_tensor(b), N_HC, ITERS, HC_EPS as f64).unwrap()
    }

    /// A stage-2 trace of a pass over `tokens`, one position per token.
    pub fn trace(&self, tokens: &[u32], n_prompt: u32, faults: Faults) -> HcTrace {
        let np = N_PRODUCTS as u64;
        let n_leaves = chain::n_pos_leaves(np, N_LAYERS as u64, 0) as usize;
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|a| a.to_le_bytes()).collect() };
        let mut positions = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let layout = layout_at(t as u64, n_prompt).1;
            let mut leaves = vec![Vec::new(); n_leaves];
            let e = quant::dequant_row(quant::Q8_0, self.row(EMBD, tok as usize), D).unwrap();
            let mut r: Vec<f32> = (0..N_HC).flat_map(|_| e.iter().copied()).collect();
            // the pre of the previous mix (single pass): stream 0 before the first
            let mut prev_pre: Vec<f64> = (0..N_HC).map(|s| (s == 0) as u8 as f64).collect();
            for l in 0..N_LAYERS {
                for (half, fn_p, x_p, nrm) in [(Resid::Attn, HC_ATTN_FN, ATTN_X, norm(l, 0)), (Resid::Out, HC_FFN_FN, FFN_X, norm(l, 1))] {
                    let x_hc = q8_1_row(&chain::rms_scaled(&r, EPS, 1.0));
                    let mixes = self.matvec(rank(l, fn_p), &x_hc);
                    let m = self.mix(l, half, &mixes);
                    let lagged = self.lag && !faults.own_pre;
                    let w_in = if faults.post_as_pre { m.post.clone() } else if lagged { prev_pre.clone() } else { m.pre.clone() };
                    // the FFN half reads the attention mix's pre, the next attention half this FFN mix's pre
                    prev_pre = m.pre.clone();
                    let x: Vec<f32> = (0..D).map(|j| (0..N_HC).map(|s| w_in[s] * r[s * D + j] as f64).sum::<f64>() as f32).collect();
                    let x_q = q8_1_row(&chain::rms_norm(&x, &self.f32_tensor(nrm), EPS));
                    let o = self.matvec(rank(l, x_p), &x_q);
                    let next: Vec<f32> = (0..N_HC)
                        .flat_map(|dst| {
                            let (m, r, o) = (&m, &r, &o);
                            (0..D).map(move |j| {
                                let c = |src: usize| if faults.transposed_comb { m.comb[src + N_HC * dst] } else { m.comb[dst + N_HC * src] };
                                (m.post[dst] * o[j] as f64 + (0..N_HC).map(|src| c(src) * r[src * D + j] as f64).sum::<f64>()) as f32
                            })
                        })
                        .collect();
                    leaves[2 * rank(l, fn_p)] = in_layout(&x_hc, layout);
                    leaves[2 * rank(l, fn_p) + 1] = bytes(&mixes);
                    leaves[2 * rank(l, x_p)] = in_layout(&x_q, layout);
                    leaves[2 * rank(l, x_p) + 1] = bytes(&o);
                    leaves[chain::resid_leaf(np, l as u64, half) as usize] = bytes(&next);
                    r = next;
                }
            }
            leaves[chain::meta_leaf(np, N_LAYERS as u64, 0) as usize] = chain::meta(t as u32, layout_at(t as u64, n_prompt).0, tok).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let trace_root = b3_root(&pos_roots);
        HcTrace { n_prompt, positions, pos_roots, trace_root }
    }

    fn entry_opening(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].1.clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    fn bytes_range(&self, i: usize, offset: u64, len: u64) -> ChunkRange {
        let base = self.entry(i).chunk_base;
        let (first, last) = (base + offset / 32, base + (offset + len - 1) / 32);
        ChunkRange {
            first_chunk: first,
            chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
            proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
        }
    }

    fn whole(&self, i: usize) -> (EntryOpening, ChunkRange) {
        (self.entry_opening(i), self.bytes_range(i, 0, self.entry(i).n_bytes))
    }

    fn row_of(&self, i: usize, r: u64) -> (EntryOpening, ChunkRange) {
        let rb = self.entry(i).row_bytes as u64;
        (self.entry_opening(i), self.bytes_range(i, r * rb, rb))
    }

    fn norm_fragment_of(&self, i: usize, block: u64) -> (EntryOpening, ChunkRange) {
        let k = self.entry(i).k;
        let (b0, n) = chain::norm_fragment(block, k);
        (self.entry_opening(i), self.bytes_range(i, 4 * b0 * 32, 4 * (n * 32).min(k - b0 * 32)))
    }

    /// The streams a half of `layer` reads at `pos`.
    fn source(&self, pos: &Position, layer: usize, half: Resid) -> HcSource {
        let np = N_PRODUCTS as u64;
        match (half, layer) {
            (Resid::Attn, 0) => HcSource::Token(self.row_of(EMBD, pos.tok as u64)),
            (Resid::Attn, l) => HcSource::Streams(leaf_bytes(pos, chain::resid_leaf(np, l as u64 - 1, Resid::Out))),
            (Resid::Out, l) => HcSource::Streams(leaf_bytes(pos, chain::resid_leaf(np, l as u64, Resid::Attn))),
        }
    }

    /// Point `i` drawn by `seed`, with its token and input link.
    pub fn open_point(&self, trace: &HcTrace, seed: &Hash, i: u32) -> (Point, u32, LinkOpening) {
        let d = draw::point(seed, i);
        let np = N_PRODUCTS as u64;
        let drawn = (d[1] % np) as usize;
        let t = d[0] % trace.positions.len() as u64;
        let e = self.entry(drawn);
        let row = d[2] % e.m;
        let pos = &trace.positions[t as usize];
        let yc = row * 4 / B3_CHUNK as u64;
        let ych = leaf_chunk(pos, 2 * drawn as u64 + 1, yc);
        let point = Point {
            n_ubatch: layout_at(t, trace.n_prompt).0,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            entry: self.entry_opening(drawn),
            partner: None,
            bias: None,
            rows: vec![self.row_of(drawn, row).1],
            x: pos.leaves[2 * drawn].clone(),
            x_path: b3_path(&pos.hashes, 2 * drawn),
            y_total_len: pos.leaves[2 * drawn + 1].len() as u32,
            y_chunk_index: yc as u32,
            y_chunk: ych.chunk,
            y_chunk_path: ych.chunk_path,
            y_path: b3_path(&pos.hashes, 2 * drawn + 1),
        };
        let Chain::Product(pl) = e.chain else { panic!("product without link") };
        let l = pl.layer as usize;
        let layout = crate::verify::input_layout(e.role, point.n_ubatch, e.wtype);
        let block = chain::link_block(seed, i, &point.x, layout, e.k).expect("link block");
        let layer = self.entry_opening(LAYERS + l);
        let link = match pl.link {
            manifest::LINK_HC_ATTN_IN => LinkOpening::HcIn { layer, src: self.source(pos, l, Resid::Attn) },
            manifest::LINK_HC_FFN_IN => LinkOpening::HcIn { layer, src: self.source(pos, l, Resid::Out) },
            link => {
                let (half, fn_p, b) = if link == manifest::LINK_HC_ATTN_PRE { (Resid::Attn, HC_ATTN_FN, 2) } else { (Resid::Out, HC_FFN_FN, 3) };
                // the mix whose pre the input reads: its own, or the one before it under single-pass mixes
                let source = match (self.lag, half, l) {
                    (false, _, _) => Some((l, fn_p, b)),
                    (true, Resid::Attn, 0) => None,
                    (true, Resid::Attn, l) => Some((l - 1, HC_FFN_FN, 3)),
                    (true, Resid::Out, l) => Some((l, HC_ATTN_FN, 2)),
                };
                let (mixes, base) = match source {
                    Some((sl, sp, sb)) => (leaf_bytes(pos, 2 * rank(sl, sp) as u64 + 1), self.whole(norm(sl, sb))),
                    None => (Default::default(), Default::default()),
                };
                LinkOpening::HcPre { layer, src: self.source(pos, l, half), mixes, base, norm: self.norm_fragment_of(pl.norm as usize, block) }
            }
        };
        (point, pos.tok, link)
    }

    /// Residual point drawn by `seed`.
    pub fn open_resid(&self, trace: &HcTrace, seed: &Hash) -> ResidPoint {
        let d = draw::point(seed, chain::RESID_POINT);
        let np = N_PRODUCTS as u64;
        let resp = response(trace);
        let sel = chain::resid_site(d[1], 2 * N_LAYERS as u64);
        let (layer, kind) = ((sel / 2) as usize, if sel.is_multiple_of(2) { Resid::Attn } else { Resid::Out });
        let t = chain::resid_position(&self.chain_ctx(), &resp, layer as u64, d[0]).expect("residual position");
        let pos = &trace.positions[t as usize];
        let (product, fn_p, b_i) = if kind == Resid::Attn { (rank(layer, ATTN_X), HC_ATTN_FN, 2) } else { (rank(layer, FFN_X), HC_FFN_FN, 3) };
        let (s, jc, per) = chain::hc_resid_chunk(d[2], N_HC as u64, D as u64).expect("chunk");
        let dst_leaf = chain::resid_leaf(np, layer as u64, kind);
        let (src, streams) = match self.source(pos, layer, kind) {
            HcSource::Token((e, r)) => (ResidSrc::Token(e, r), Vec::new()),
            HcSource::Streams(_) => {
                let leaf = if kind == Resid::Attn { chain::resid_leaf(np, layer as u64 - 1, Resid::Out) } else { chain::resid_leaf(np, layer as u64, Resid::Attn) };
                let mut cs: Vec<LeafChunk> = (0..N_HC as u64).map(|src| leaf_chunk(pos, leaf, src * per + jc)).collect();
                let first = cs.remove(0);
                (ResidSrc::Chunk(first), cs)
            }
        };
        ResidPoint {
            n_ubatch: layout_at(t, trace.n_prompt).0,
            tok: pos.tok,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            layer: self.entry_opening(LAYERS + layer),
            product: self.entry_opening(product),
            dst: leaf_chunk(pos, dst_leaf, s * per + jc),
            src,
            add: ResidAdd::Hc(Box::new(HcAdd {
                o: leaf_chunk(pos, 2 * product as u64 + 1, jc),
                streams,
                mixes: leaf_bytes(pos, 2 * rank(layer, fn_p) as u64 + 1),
                base: self.whole(norm(layer, b_i)),
                moe: None,
            })),
        }
    }

    /// Prefix context: identity byte table, the layer-0 mix product read without norm weight.
    pub fn prefix_ctx(&self, offset: u32) -> PrefixCtx {
        PrefixCtx { table: std::array::from_fn(|b| b as u32), offset, embd_entry: EMBD as u32, norm_entry: manifest::NONE, eps: EPS, embd_scale: 1.0 }
    }

    /// Prefix point drawn by `seed`, opening the embedding row of `tok` (the position's own token
    /// when `None`).
    pub fn open_prefix(&self, trace: &HcTrace, seed: &Hash, offset: u32, tok: Option<u32>) -> PrefixPoint {
        let np = N_PRODUCTS as u64;
        let t = prefix::position(seed, offset);
        let pos = &trace.positions[t as usize];
        PrefixPoint {
            n_ubatch: layout_at(t, trace.n_prompt).0,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            entry: self.entry_opening(0),
            x: pos.leaves[0].clone(),
            x_path: b3_path(&pos.hashes, 0),
            embd: self.row_of(EMBD, tok.unwrap_or(pos.tok) as u64),
            norm: Default::default(),
        }
    }

    /// Residual opening drawn by `seed`: the residual point and one row of its feeding product.
    pub fn open_resid_piece(&self, trace: &HcTrace, seed: &Hash) -> ResidOpening {
        let resid = self.open_resid(trace, seed);
        let d = draw::point(seed, chain::RESID_POINT);
        let product = resid.product.index as usize;
        let per = (B3_CHUNK / 4) as u64;
        let (_, jc, _) = chain::hc_resid_chunk(d[2], N_HC as u64, D as u64).expect("chunk");
        let lo = jc * per;
        let row = lo + (d[3] >> 32) % per.min(self.entry(product).m - lo);
        let sel = chain::resid_site(d[1], 2 * N_LAYERS as u64);
        let t = chain::resid_position(&self.chain_ctx(), &response(trace), sel / 2, d[0]).expect("residual position");
        let pos = &trace.positions[t as usize];
        let feed = FeedRow { x: pos.leaves[2 * product].clone(), x_path: b3_path(&pos.hashes, 2 * product), row: self.row_of(product, row).1, bias: None };
        ResidOpening { response_txid: [0; 32], roots: None, resid, feed, expert: None }
    }
}

pub fn response(trace: &HcTrace) -> ResponseCtx {
    ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 }
}

fn leaf_bytes(pos: &Position, i: u64) -> LeafBytes {
    LeafBytes { bytes: pos.leaves[i as usize].clone(), leaf_path: b3_path(&pos.hashes, i as usize) }
}

fn leaf_chunk(pos: &Position, i: u64, c: u64) -> LeafChunk {
    let b = &pos.leaves[i as usize];
    let n = b.len().div_ceil(B3_CHUNK).max(1);
    let chunk = b[c as usize * B3_CHUNK..((c as usize + 1) * B3_CHUNK).min(b.len())].to_vec();
    let chunk_path = if n == 1 {
        Vec::new()
    } else {
        let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
        b3_path(&cvs, c as usize)
    };
    LeafChunk { total_len: b.len() as u32, index: c as u32, chunk, chunk_path, leaf_path: b3_path(&pos.hashes, i as usize) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{verify_chain_point, verify_resid_opening, verify_resid_point};
    use crate::verify::Fail;

    fn tokens() -> Vec<u32> {
        (0..12u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect()
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..64u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    #[test]
    fn honest_hyper_connection_trace_passes_every_draw() {
        let tier = HcTier::new();
        let trace = tier.trace(&tokens(), 12, Faults::default());
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), response(&trace));
        let mut kinds = [0usize; 4];
        let mut sites = std::collections::BTreeSet::new();
        for seed in seeds() {
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                kinds[match (&link, p.entry.index as usize % PER_LAYER) {
                    (LinkOpening::HcIn { src: HcSource::Token(_), .. }, _) => 0,
                    (LinkOpening::HcIn { .. }, _) => 1,
                    (LinkOpening::HcPre { src: HcSource::Token(_), .. }, _) => 2,
                    _ => 3,
                }] += 1;
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).unwrap_or_else(|e| panic!("point {i}: {e:?}"));
            }
            let r = tier.open_resid(&trace, &seed);
            sites.insert((r.layer.index, r.product.index, matches!(r.src, ResidSrc::Token(..)), r.dst.index));
            verify_resid_point(&tc, &cc, &rc, &seed, &r).unwrap_or_else(|e| panic!("resid: {e:?}"));
            verify_resid_opening(&tc, &cc, &rc, &seed, &tier.open_resid_piece(&trace, &seed)).unwrap_or_else(|e| panic!("resid piece: {e:?}"));
        }
        assert!(kinds.iter().all(|&n| n > 0), "{kinds:?}");
        assert!(sites.iter().any(|s| s.2) && sites.iter().any(|s| !s.2) && sites.iter().any(|s| s.3 % 2 == 1), "{sites:?}");
    }

    #[test]
    fn a_transposed_stream_mix_is_refused_by_the_residual_point() {
        let tier = HcTier::new();
        let trace = tier.trace(&tokens(), 12, Faults { transposed_comb: true, ..Default::default() });
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), response(&trace));
        let mut refused = 0;
        for seed in seeds() {
            let r = tier.open_resid(&trace, &seed);
            // The streams entering layer 0 are all equal: any column-stochastic mix of them is the same.
            let equal_streams = matches!(r.src, ResidSrc::Token(..));
            match verify_resid_point(&tc, &cc, &rc, &seed, &r) {
                Err(Fail::Mismatch) => refused += 1,
                Ok(()) if equal_streams => {}
                other => panic!("equal streams {equal_streams}: {other:?}"),
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_half_input_mixed_with_the_wrong_weights_is_refused_by_its_link() {
        let tier = HcTier::new();
        let trace = tier.trace(&tokens(), 12, Faults { post_as_pre: true, ..Default::default() });
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), response(&trace));
        let mut refused = 0;
        for seed in seeds() {
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                let pre = matches!(link, LinkOpening::HcPre { .. });
                // At layer 0 the streams span only the embedding and the attention output: two mixes
                // can give the same normed input.
                let equal_streams = p.entry.index as usize / PER_LAYER == 0;
                match verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None) {
                    Err(Fail::Mismatch) if pre => refused += !equal_streams as usize,
                    Ok(_) if !pre || equal_streams => {}
                    other => panic!("point {i} (pre {pre}): {other:?}"),
                }
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn the_prefix_binds_the_repeated_embedding_of_the_request_tokens() {
        let tier = HcTier::new();
        let offset = 2;
        let request: Hash = std::array::from_fn(|i| (i * 29 + 11) as u8);
        let mut toks = tokens();
        toks.resize(offset as usize, 0);
        toks.extend(request.iter().map(|&b| b as u32));
        toks.extend([7, 9, 13]);
        let n = toks.len() as u32;
        let trace = tier.trace(&toks, n, Faults::default());
        let (tc, cc, rc, pc) = (tier.tier(), tier.chain_ctx(), response(&trace), tier.prefix_ctx(offset));
        let other: Hash = std::array::from_fn(|i| request[i] ^ 0x5a);
        for seed in seeds() {
            let t = chain::verify_chain_prefix_point(&tc, &cc, &pc, &rc, &seed, &request, &tier.open_prefix(&trace, &seed, offset, None)).unwrap();
            assert!((offset as u64..offset as u64 + prefix::PREFIX_LEN as u64).contains(&t));
            let want = prefix::expected_token(&pc, &other, t).unwrap();
            let forged = tier.open_prefix(&trace, &seed, offset, Some(want));
            // The meta leaf commits the position's token: another request's token does not open it.
            assert_eq!(chain::verify_chain_prefix_point(&tc, &cc, &pc, &rc, &seed, &other, &forged), Err(Fail::Trace));
        }
        let with_norm = PrefixCtx { norm_entry: norm(0, 0) as u32, ..pc };
        let p = tier.open_prefix(&trace, &seeds().next().unwrap(), offset, None);
        assert!(chain::verify_chain_prefix_point(&tc, &cc, &with_norm, &rc, &seeds().next().unwrap(), &request, &p).is_err());
    }

    #[test]
    fn honest_single_pass_mixes_pass_every_draw() {
        let tier = HcTier::with_lag(true);
        let trace = tier.trace(&tokens(), 12, Faults::default());
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), response(&trace));
        let mut pre = [0usize; 3];
        for seed in seeds() {
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                if let LinkOpening::HcPre { mixes, .. } = &link {
                    pre[if mixes.bytes.is_empty() { 0 } else if p.entry.index as usize % PER_LAYER == ATTN_X { 1 } else { 2 }] += 1;
                }
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).unwrap_or_else(|e| panic!("point {i}: {e:?}"));
            }
            verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed)).unwrap();
        }
        assert!(pre.iter().all(|&n| n > 0), "{pre:?}");
    }

    #[test]
    fn a_half_reading_its_own_pre_under_single_pass_mixes_is_refused() {
        let tier = HcTier::with_lag(true);
        let trace = tier.trace(&tokens(), 12, Faults { own_pre: true, ..Faults::default() });
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), response(&trace));
        let mut refused = 0;
        for seed in seeds() {
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                // at layer 0 every stream is the embedding: any pre mixes it to the same normed input
                if matches!(link, LinkOpening::HcPre { .. }) && p.entry.index as usize != rank(0, ATTN_X) {
                    let r = verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None);
                    assert_eq!(r.err(), Some(Fail::Mismatch));
                    refused += 1;
                }
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_single_pass_layer_round_trips_and_keeps_plain_layers_unchanged() {
        let plain = HcTier::new();
        let lagged = HcTier::with_lag(true);
        let layer = |t: &HcTier, l: usize| t.entries[LAYERS + l].1.clone();
        assert_ne!(layer(&plain, 1), layer(&lagged, 1));
        let parsed = Entry::parse(&layer(&lagged, 1)).unwrap().0;
        assert!(matches!(parsed.chain, Chain::Layer(ll) if ll.hc.is_some_and(|h| h.lag == 1 && h.prev_fn == rank(0, HC_FFN_FN) as u32)));
        assert!(matches!(Entry::parse(&layer(&plain, 1)).unwrap().0.chain, Chain::Layer(ll) if ll.hc.is_some_and(|h| h.lag == 0)));
    }

    #[test]
    fn a_layer_wrapped_in_hyper_connections_round_trips() {
        let tier = HcTier::new();
        let (e, b) = &tier.entries[LAYERS + 1];
        let (parsed, len) = Entry::parse(b).unwrap();
        assert_eq!((len, &parsed.chain), (b.len(), &e.chain));
        assert!(matches!(parsed.chain, Chain::Layer(ll) if ll.hc.is_some_and(|h| h.n_hc == N_HC as u32 && h.attn_fn == rank(1, HC_ATTN_FN) as u32)));
    }
}

/// One layer of hyper-connection streams with a routed-expert FFN, one position, for the C5 checks.
#[cfg(test)]
mod moe_tests {
    use super::*;
    use crate::chain::{route_weights_gated, verify_resid_opening, verify_resid_point};
    use crate::opening::{ExpertIds, ExpertRow, HcMoe};
    use crate::verify::Fail;

    const E: usize = 16;
    const U: usize = 4;
    const W_SCALE: f32 = 2.5;
    const RB: usize = D / 32 * 34;

    fn vecf(seed: u64, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
    }

    fn bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|a| a.to_le_bytes()).collect()
    }

    #[derive(Clone, Copy, Default)]
    struct Fault {
        /// The router scores are committed from another input than the experts' gate input.
        scores_off_input: bool,
        /// One routed expert's output is bumped after the residual was built.
        expert_bump: bool,
        /// The routed experts are combined without the weight scale.
        unscaled: bool,
        /// The router weights are stored in BF16 (not a fault).
        bf16_router: bool,
        /// The manifest gates the experts by sqrt-softplus (not a fault).
        sqrt_softplus: bool,
        /// The experts are combined with sigmoid weights whatever the manifest says.
        sigmoid_weights: bool,
    }

    struct Kit {
        weights: Vec<u8>,
        chunk_leaves: Vec<Hash>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        tier: TierCtx,
        ctx: ChainCtx,
        pos: Position,
        trace_root: Hash,
        offs: [usize; 5],
        ids: Vec<u32>,
        /// Bytes per router weight.
        rw: usize,
    }

    fn entry(name: &str, role: u8, wtype: u32, k: usize, m: usize, n_exp: u32, off: usize, n: usize, rb: usize, chain: Chain) -> Vec<u8> {
        let e = Entry {
            name: name.into(),
            wtype,
            k: k as u64,
            m: m as u64,
            n_exp,
            chunk_base: (off / 32) as u64,
            n_bytes: n as u64,
            row_bytes: rb as u32,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain,
        };
        let mut b = serialize(&e);
        b.extend_from_slice(&manifest::chain_bytes(&e.chain));
        b
    }

    fn build(f: Fault) -> Kit {
        let mut weights: Vec<u8> = Vec::new();
        let mut push = |b: Vec<u8>| {
            let off = weights.len();
            weights.extend_from_slice(&b);
            while !weights.len().is_multiple_of(32) {
                weights.push(0);
            }
            off
        };
        let router: Vec<f32> = vecf(1, E * D, 0.05)
            .into_iter()
            .map(|w| if f.bf16_router { quant::dequant_row(quant::BF16, &((w.to_bits() + 0x8000) >> 16).to_le_bytes()[..2], 1).unwrap()[0] } else { w })
            .collect();
        let bias = vecf(2, E, 0.01);
        let norm_w: Vec<f32> = (0..D).map(|i| 1.0 + 0.001 * (i % 7) as f32).collect();
        let base: Vec<f32> = (0..N_MIX).map(|j| 0.3 * ((j * 7) % 5) as f32 - 0.6).collect();
        let router_bytes: Vec<u8> = if f.bf16_router { router.iter().flat_map(|w| ((w.to_bits() >> 16) as u16).to_le_bytes()).collect() } else { bytes(&router) };
        let rw = if f.bf16_router { 2 } else { 4 };
        let router_off = push(router_bytes);
        let bias_off = push(bytes(&bias));
        let norm_off = push(bytes(&norm_w));
        let base_off = push(bytes(&base));
        let exps_off = push(q8_0_rows(40, E * D, D, 0.004));
        let shexp_off = push(q8_0_rows(41, D, D, 0.004));
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();

        let none = manifest::NONE;
        let scale = [0.5f32, 0.4, 0.6];
        let moe = manifest::MoeLinks { down_exps: 1, router: 5, probs_bias: 6, w_scale: W_SCALE.to_bits(), ffn_norm: 7, gate_exps: 3, shexp_gate: none, gating: if f.sqrt_softplus { manifest::GATING_SQRT_SOFTPLUS } else { 0 } };
        let hc = manifest::HcLinks {
            attn_fn: 4,
            ffn_fn: 4,
            attn_scale: scale.map(f32::to_bits),
            ffn_scale: scale.map(f32::to_bits),
            attn_base: 8,
            ffn_base: 8,
            n_hc: N_HC as u32,
            sinkhorn_iters: ITERS,
            eps: HC_EPS.to_bits(),
            lag: 0,
            prev_fn: manifest::NONE,
            prev_scale: [0; 3],
            prev_base: manifest::NONE,
        };
        let q = quant::Q8_0;
        let entries = vec![
            entry("blk.0.attn_output.weight", manifest::ROLE_PRODUCT, q, D, D, 1, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_down_exps.weight", manifest::ROLE_PRODUCT, q, D, D, E as u32, exps_off, E * D * RB, RB, Chain::None),
            entry("blk.0.ffn_down_shexp.weight", manifest::ROLE_PRODUCT, q, D, D, 1, shexp_off, D * RB, RB, Chain::None),
            entry("blk.0.ffn_gate_exps.weight", manifest::ROLE_PRODUCT, q, D, D, E as u32, 0, 0, 0, Chain::None),
            entry("blk.0.hc_ffn_fn.weight", manifest::ROLE_PRODUCT, q, K_HC, N_MIX, 1, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_gate_inp.weight", manifest::ROLE_ROUTER, if f.bf16_router { quant::BF16 } else { 0 }, D, E, 1, router_off, rw * E * D, rw * D, Chain::None),
            entry("blk.0.exp_probs_b.bias", manifest::ROLE_NORM, 0, E, 1, 1, bias_off, 4 * E, 4 * E, Chain::None),
            entry("blk.0.ffn_norm.weight", manifest::ROLE_NORM, 0, D, 1, 1, norm_off, 4 * D, 4 * D, Chain::None),
            entry("blk.0.hc_ffn_base.weight", manifest::ROLE_NORM, 0, N_MIX, 1, 1, base_off, 4 * N_MIX, 4 * N_MIX, Chain::None),
            entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, 0, 0, 0, Chain::Layer(LayerLinks {
                layer: 0, proj: 0, down: 2, post_attn: none, post_ffn: none, scale: none, moe: Some(moe), attn: None, recur: None, hc: Some(hc), dsa: None, comp: None, engram: None,
            })),
        ];
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let np = 5u64;
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: entries.len() as u64,
            n_products: np,
        };
        let ctx = ChainCtx { n_sel: 0, n_comp: 0, n_engram: 0, n_layers: 1, layer_base: 9, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: false, n_used: U as u32, n_attn: 0, n_recur: 0, seq_heads: 0 };

        // FFN half of the layer: streams after attention, their mix, then the routed FFN.
        let r: Vec<f32> = vecf(3, N_HC * D, 2.0);
        let mixes = vecf(4, N_MIX, 3.0);
        let mix = chain::hc_mix(&mixes, &scale, &base, N_HC, ITERS, HC_EPS as f64).unwrap();
        let xin: Vec<f32> = (0..D).map(|j| (0..N_HC).map(|s| mix.pre[s] * r[s * D + j] as f64).sum::<f64>() as f32).collect();
        let v = chain::rms_norm(&xin, &norm_w, EPS);
        let router_in: Vec<f64> = if f.scores_off_input { vecf(31, D, 1.0).iter().map(|&a| a as f64).collect() } else { v.clone() };
        let scores: Vec<f32> = (0..E).map(|e| router[e * D..(e + 1) * D].iter().zip(&router_in).map(|(&w, &x)| w as f64 * x).sum::<f64>() as f32).collect();
        let sel: Vec<f64> = scores.iter().zip(&bias).map(|(&s, &b)| 1.0 / (1.0 + (-(s as f64)).exp()) + b as f64).collect();
        let mut ids: Vec<u32> = (0..E as u32).collect();
        ids.sort_by(|&a, &b| sel[b as usize].total_cmp(&sel[a as usize]));
        ids.truncate(U);
        let gating = if f.sqrt_softplus && !f.sigmoid_weights { manifest::GATING_SQRT_SOFTPLUS } else { 0 };
        let w = route_weights_gated(&scores, Some(&bias), &ids, if f.unscaled { 1.0 } else { W_SCALE as f64 }, gating).unwrap();
        let row_dot = |off: usize, x: &[u8]| crate::dot::canonical_dot(q, &weights[off..off + RB], D, x, q8::MMVQ).unwrap().y_hat as f32;
        let qx = |v: Vec<f32>| q8_1_row(&v.iter().map(|&a| a as f64).collect::<Vec<_>>());
        let xs: Vec<Vec<u8>> = (0..U).map(|s| qx(vecf(50 + s as u64, D, 1.0))).collect();
        let x_sh = qx(vecf(60, D, 1.0));
        let ys: Vec<Vec<f32>> = (0..U).map(|s| (0..D).map(|j| row_dot(exps_off + (ids[s] as usize * D + j) * RB, &xs[s])).collect()).collect();
        let sh: Vec<f32> = (0..D).map(|j| row_dot(shexp_off + j * RB, &x_sh)).collect();
        let o: Vec<f64> = (0..D).map(|j| sh[j] as f64 + w.iter().zip(&ys).map(|(a, y)| a * y[j] as f64).sum::<f64>()).collect();
        let out: Vec<f32> = (0..N_HC)
            .flat_map(|dst| (0..D).map(move |j| (dst, j)))
            .map(|(dst, j)| (mix.post[dst] * o[j] + (0..N_HC).map(|src| mix.comb[dst + N_HC * src] * r[src * D + j] as f64).sum::<f64>()) as f32)
            .collect();
        let mut y_exps: Vec<u8> = ys.iter().flat_map(|y| bytes(y)).collect();
        if f.expert_bump {
            for s in 0..U {
                for j in 0..D {
                    let i = (s * D + j) * 4;
                    let v = f32::from_le_bytes(y_exps[i..i + 4].try_into().unwrap()) + 0.5;
                    y_exps[i..i + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        for id in &ids {
            y_exps.extend_from_slice(&id.to_le_bytes());
        }
        let n_leaves = chain::n_pos_leaves(np, 1, 1) as usize;
        let mut leaves = vec![Vec::new(); n_leaves];
        leaves[2] = xs.concat();
        leaves[3] = y_exps;
        leaves[4] = x_sh;
        leaves[5] = bytes(&sh);
        leaves[6] = q8_1_row(&v);
        leaves[9] = bytes(&mixes);
        leaves[chain::resid_leaf(np, 0, Resid::Attn) as usize] = bytes(&r);
        leaves[chain::resid_leaf(np, 0, Resid::Out) as usize] = bytes(&out);
        leaves[chain::router_leaf(np, 1, 0) as usize] = bytes(&scores);
        leaves[chain::meta_leaf(np, 1, 1) as usize] = chain::meta(0, 1, 7).to_vec();
        let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let trace_root = b3_root(&[b3_root(&hashes)]);
        Kit {
            weights,
            chunk_leaves,
            entries,
            entry_leaves,
            tier,
            ctx,
            pos: Position { tok: 7, leaves, hashes },
            trace_root,
            offs: [router_off, bias_off, base_off, exps_off, shexp_off],
            rw,
            ids,
        }
    }

    impl Kit {
        fn eo(&self, i: usize) -> EntryOpening {
            EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
        }

        fn range(&self, off: usize, len: usize) -> ChunkRange {
            let (first, last) = ((off / 32) as u64, ((off + len - 1) / 32) as u64);
            ChunkRange {
                first_chunk: first,
                chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
                proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
            }
        }

        fn resp(&self) -> ResponseCtx {
            ResponseCtx { trace_root: self.trace_root, n_prompt: 1, n_pos: 1 }
        }

        /// The residual point of `seed`, when it draws the FFN residual.
        fn open(&self, seed: &Hash) -> Option<ResidPoint> {
            let d = draw::point(seed, chain::RESID_POINT);
            if chain::resid_site(d[1], 2) != 1 {
                return None;
            }
            let np = 5u64;
            let (s, jc, per) = chain::hc_resid_chunk(d[2], N_HC as u64, D as u64).unwrap();
            let attn = chain::resid_leaf(np, 0, Resid::Attn);
            let mut streams: Vec<LeafChunk> = (0..N_HC as u64).map(|src| leaf_chunk(&self.pos, attn, src * per + jc)).collect();
            let first = streams.remove(0);
            let row = d[3] % E as u64;
            let ids_chunk = leaf_chunk(&self.pos, 3, (U * D * 4 / B3_CHUNK) as u64);
            let [router_off, bias_off, base_off, ..] = self.offs;
            Some(ResidPoint {
                n_ubatch: 1,
                tok: 7,
                pos_path: Vec::new(),
                meta_path: b3_path(&self.pos.hashes, chain::meta_leaf(np, 1, 1) as usize),
                layer: self.eo(9),
                product: self.eo(2),
                dst: leaf_chunk(&self.pos, chain::resid_leaf(np, 0, Resid::Out), s * per + jc),
                src: ResidSrc::Chunk(first),
                add: ResidAdd::Hc(Box::new(HcAdd {
                    o: leaf_chunk(&self.pos, 5, jc),
                    streams,
                    mixes: leaf_bytes(&self.pos, 9),
                    base: (self.eo(8), self.range(base_off, 4 * N_MIX)),
                    moe: Some(Box::new(HcMoe {
                        scores: leaf_bytes(&self.pos, chain::router_leaf(np, 1, 0)),
                        bias: (self.eo(6), self.range(bias_off, 4 * E)),
                        ids: ExpertIds { chunk: ids_chunk.chunk, chunk_path: ids_chunk.chunk_path },
                        slots: (0..U as u64).map(|sl| leaf_chunk(&self.pos, 3, sl * per + jc)).collect(),
                        router_row: (self.eo(5), self.range(router_off + row as usize * self.rw * D, self.rw * D)),
                        gate: self.eo(3),
                        x: leaf_bytes(&self.pos, 6),
                    })),
                })),
            })
        }

        fn piece(&self, seed: &Hash) -> Option<ResidOpening> {
            let resid = self.open(seed)?;
            let d = draw::point(seed, chain::RESID_POINT);
            let [.., exps_off, shexp_off] = self.offs;
            let per = (B3_CHUNK / 4) as u64;
            let (_, jc, _) = chain::hc_resid_chunk(d[2], N_HC as u64, D as u64).unwrap();
            let row = (jc * per + (d[3] >> 32) % per.min(D as u64 - jc * per)) as usize;
            let id = self.ids[((d[3] >> 48) % U as u64) as usize] as usize;
            let feed = FeedRow { x: self.pos.leaves[4].clone(), x_path: b3_path(&self.pos.hashes, 4), row: self.range(shexp_off + row * RB, RB), bias: None };
            let expert = ExpertRow { entry: self.eo(1), x: self.pos.leaves[2].clone(), x_path: b3_path(&self.pos.hashes, 2), row: self.range(exps_off + (id * D + row) * RB, RB) };
            Some(ResidOpening { response_txid: [0; 32], roots: None, resid, feed, expert: Some(expert) })
        }

        fn verdicts(&self) -> Vec<Result<(), Fail>> {
            (0..256u32)
                .filter_map(|s| {
                    let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                    self.open(&seed).map(|p| verify_resid_point(&self.tier, &self.ctx, &self.resp(), &seed, &p))
                })
                .collect()
        }
    }

    #[test]
    fn an_honest_routed_ffn_half_passes_its_residual_and_rows() {
        let k = build(Fault::default());
        let mut n = 0;
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let Some(p) = k.open(&seed) else { continue };
            verify_resid_point(&k.tier, &k.ctx, &k.resp(), &seed, &p).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            verify_resid_opening(&k.tier, &k.ctx, &k.resp(), &seed, &k.piece(&seed).unwrap()).unwrap_or_else(|e| panic!("seed {s} piece: {e:?}"));
            n += 1;
        }
        assert!(n > 20, "{n}");
    }

    #[test]
    fn a_bf16_router_passes_and_still_catches_forged_scores() {
        let k = build(Fault { bf16_router: true, ..Default::default() });
        let mut n = 0;
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let Some(p) = k.open(&seed) else { continue };
            verify_resid_point(&k.tier, &k.ctx, &k.resp(), &seed, &p).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            n += 1;
        }
        assert!(n > 20, "{n}");
        let v = build(Fault { bf16_router: true, scores_off_input: true, ..Default::default() }).verdicts();
        let refused = v.iter().filter(|r| **r == Err(Fail::Mismatch)).count();
        assert!(refused * 4 >= v.len() * 3, "{refused} of {}", v.len());
    }

    #[test]
    fn sqrt_softplus_gated_experts_pass_and_sigmoid_weights_are_refused() {
        let k = build(Fault { sqrt_softplus: true, ..Default::default() });
        let mut n = 0;
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let Some(p) = k.open(&seed) else { continue };
            verify_resid_point(&k.tier, &k.ctx, &k.resp(), &seed, &p).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            n += 1;
        }
        assert!(n > 20, "{n}");
        let v = build(Fault { sqrt_softplus: true, sigmoid_weights: true, ..Default::default() }).verdicts();
        assert!(!v.is_empty() && v.iter().all(|r| *r == Err(Fail::Mismatch)), "{v:?}");
    }

    #[test]
    fn router_scores_off_the_committed_input_are_refused() {
        // One router row is replayed per audit: a forged score vector is caught on most rows.
        let v = build(Fault { scores_off_input: true, ..Default::default() }).verdicts();
        let refused = v.iter().filter(|r| **r == Err(Fail::Mismatch)).count();
        assert!(v.iter().all(|r| r.is_ok() || *r == Err(Fail::Mismatch)));
        assert!(refused * 4 >= v.len() * 3, "{refused} of {}", v.len());
    }

    #[test]
    fn expert_outputs_off_the_residual_are_refused() {
        let v = build(Fault { expert_bump: true, ..Default::default() }).verdicts();
        assert!(!v.is_empty() && v.iter().all(|r| *r == Err(Fail::Mismatch)), "{v:?}");
    }

    #[test]
    fn experts_combined_without_their_scale_are_refused() {
        let v = build(Fault { unscaled: true, ..Default::default() }).verdicts();
        assert!(!v.is_empty() && v.iter().all(|r| *r == Err(Fail::Mismatch)), "{v:?}");
    }
}
