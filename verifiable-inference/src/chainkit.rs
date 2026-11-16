//! Two-layer synthetic tier and stage-2 traces for tests.

use crate::chain::{self, ChainCtx, Resid};
use crate::dot;
use crate::manifest::{self, Chain, Entry, LayerLinks, ProductLink};
use crate::draw;
use crate::merkle::{B3_CHUNK, Hash, b3_chunk_cv, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, LeafBytes, LeafChunk, LinkOpening, Point, PrefixPoint, ResidAdd, ResidPoint, ResidSrc};
use crate::prefix::{self, PrefixCtx};
use crate::prefix::expected_input;
use crate::q8;
use crate::quant;
use crate::f16;
use crate::testkit::{in_layout, layout_at, q8_1_row, serialize};
use crate::verify::TierCtx;

pub const D: usize = 512;
const HEAD_M: usize = 32;
const Q_M: usize = 32;
const N_LAYERS: usize = 2;
pub const VOCAB: usize = 256;
pub const EPS: f64 = 1e-6;

const PER_LAYER: usize = 7;
const ATTN_Q: usize = 0;
const ATTN_OUTPUT: usize = 1;
const FFN_DOWN: usize = 2;
const FFN_GATE: usize = 3;
const FFN_UP: usize = 4;
const SSM_F_A: usize = 5;
const SSM_F_B: usize = 6;
const N_PRODUCTS: usize = N_LAYERS * PER_LAYER + 1;
const OUTPUT: usize = N_LAYERS * PER_LAYER;
const EMBD: usize = N_PRODUCTS;
const NORMS: usize = EMBD + 1;
const OUTPUT_NORM: usize = NORMS + 2 * N_LAYERS;
const LAYERS: usize = OUTPUT_NORM + 1;

fn q8_0_rows(seed: u64, rows: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * D / 32 * 34);
    for r in 0..rows {
        for b in 0..D / 32 {
            out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
            let h = blake3::hash(&[seed.to_le_bytes(), ((r * D / 32 + b) as u64).to_le_bytes()].concat());
            out.extend_from_slice(h.as_bytes());
        }
    }
    out
}

/// A Q8_1 row of `D` zero values with block scale `d`.
fn zero_q8_row(d: f32) -> Vec<u8> {
    (0..D / 32)
        .flat_map(|_| {
            let mut b = f16::from_f32(d).to_le_bytes().to_vec();
            b.extend_from_slice(&[0u8; 34]);
            b
        })
        .collect()
}

/// A Q8_1 row of `D` values with block scale `d` and one value per block at full magnitude.
#[cfg(test)]
fn peaked_q8_row(d: f32) -> Vec<u8> {
    let mut x = zero_q8_row(d);
    for blk in x.chunks_exact_mut(36) {
        blk[4] = 127;
    }
    x
}

fn rank(l: usize, p: usize) -> usize {
    l * PER_LAYER + p
}

fn attn_norm(l: usize) -> usize {
    NORMS + 2 * l
}

fn ffn_norm(l: usize) -> usize {
    NORMS + 2 * l + 1
}

pub struct Tier {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<(Entry, Vec<u8>)>,
    pub entry_leaves: Vec<Hash>,
    pub r_t: Hash,
    pub m_t: Hash,
}

/// Committed leaves of one position.
pub struct Position {
    pub tok: u32,
    pub leaves: Vec<Vec<u8>>,
    pub hashes: Vec<Hash>,
}

pub struct Trace {
    pub n_prompt: u32,
    pub positions: Vec<Position>,
    pub pos_roots: Vec<Hash>,
    pub trace_root: Hash,
    /// Batch size committed at every position instead of the expected one.
    pub ubatch: Option<u32>,
}

impl Trace {
    /// Committed batch size and product input layout at position `t`.
    pub fn layout(&self, t: u64) -> (u32, u8) {
        layout_of(self.ubatch, t, self.n_prompt)
    }
}

fn layout_of(forced: Option<u32>, t: u64, n_prompt: u32) -> (u32, u8) {
    match forced {
        Some(ub) => (ub, crate::verify::layout_for(ub, quant::Q8_0)),
        None => layout_at(t, n_prompt),
    }
}

impl Default for Tier {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier {
    pub fn new() -> Self {
        let row_bytes = (D / 32 * 34) as u32;
        let mut weights = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();
        let base = |name: String, role: u8, wtype: u32, m: usize, off: usize, n_bytes: usize, rb: u32| Entry {
            name,
            wtype,
            k: D as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base: (off / 32) as u64,
            n_bytes: n_bytes as u64,
            row_bytes: rb,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let link = |layer: usize, link: u8, norm: usize, src: usize| {
            Chain::Product(ProductLink { layer: layer as u32, link, norm: norm as u32, src: src as u32 })
        };
        let none = manifest::NONE as usize;

        let push = |weights: &mut Vec<u8>, bytes: Vec<u8>| {
            let off = weights.len();
            weights.extend_from_slice(&bytes);
            (off, bytes.len())
        };
        for l in 0..N_LAYERS {
            let seed = 100 * (l as u64 + 1);
            for p in 0..PER_LAYER {
                let (name, m) = match p {
                    ATTN_OUTPUT => ("attn_output", D),
                    ATTN_Q => ("attn_k", Q_M),
                    FFN_DOWN => ("ffn_down", D),
                    FFN_GATE => ("ffn_gate", D),
                    FFN_UP => ("ffn_up", D),
                    SSM_F_A => ("ssm_f_a", D),
                    _ => ("ssm_f_b", D),
                };
                let (off, n) = push(&mut weights, q8_0_rows(seed + p as u64, m, 0.0004));
                let mut e = base(format!("blk.{l}.{name}.weight"), manifest::ROLE_PRODUCT, quant::Q8_0, m, off, n, row_bytes);
                e.chain = match p {
                    ATTN_OUTPUT => link(l, manifest::LINK_NONE, none, none),
                    ATTN_Q => link(l, manifest::LINK_ATTN_IN, attn_norm(l), none),
                    FFN_DOWN => link(l, manifest::LINK_GLU, none, rank(l, FFN_GATE)),
                    SSM_F_A => link(l, manifest::LINK_ATTN_IN, attn_norm(l), none),
                    SSM_F_B => link(l, manifest::LINK_DIRECT, none, rank(l, SSM_F_A)),
                    _ => link(l, manifest::LINK_FFN_IN, ffn_norm(l), none),
                };
                if p == FFN_GATE || p == FFN_UP {
                    e.glu_op = manifest::GLU_SWIGLU;
                    e.glu_form = if p == FFN_GATE { manifest::FORM_PAIR_GATE } else { manifest::FORM_PAIR_UP };
                    e.partner = rank(l, if p == FFN_GATE { FFN_UP } else { FFN_GATE }) as u32;
                }
                entries.push(e);
            }
        }
        let (off, n) = push(&mut weights, q8_0_rows(900, HEAD_M, 0.0004));
        let mut head = base("output.weight".into(), manifest::ROLE_OUTPUT, quant::Q8_0, HEAD_M, off, n, row_bytes);
        head.chain = link(N_LAYERS, manifest::LINK_HEAD, OUTPUT_NORM, none);
        entries.push(head);

        let (off, n) = push(&mut weights, q8_0_rows(3, VOCAB, 0.02));
        entries.push(base("token_embd.weight".into(), manifest::ROLE_EMBD, quant::Q8_0, VOCAB, off, n, row_bytes));
        for (i, name) in ["blk.0.attn_norm", "blk.0.ffn_norm", "blk.1.attn_norm", "blk.1.ffn_norm", "output_norm"].iter().enumerate() {
            let w: Vec<u8> = (0..D).flat_map(|j| (1.0 + 0.001 * ((j + 3 * i) % 17) as f32).to_le_bytes()).collect();
            let (off, n) = push(&mut weights, w);
            entries.push(base(format!("{name}.weight"), manifest::ROLE_NORM, 0, 1, off, n, 4 * D as u32));
        }
        for l in 0..N_LAYERS {
            let mut e = base(format!("blk.{l}"), manifest::ROLE_LAYER, 0, 0, 0, 0, 0);
            e.chain = Chain::Layer(LayerLinks {
                layer: l as u32,
                proj: rank(l, ATTN_OUTPUT) as u32,
                down: rank(l, FFN_DOWN) as u32,
                post_attn: manifest::NONE,
                post_ffn: manifest::NONE,
                scale: manifest::NONE,
                moe: None,
                attn: None,
                recur: None,
            });
            entries.push(e);
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
        Self { weights, chunk_leaves, entries, entry_leaves, r_t, m_t }
    }

    pub fn tier(&self) -> TierCtx {
        TierCtx {
            r_t: self.r_t,
            n_chunks: self.chunk_leaves.len() as u64,
            m_t: self.m_t,
            n_entries: self.entries.len() as u64,
            n_products: N_PRODUCTS as u64,
        }
    }

    pub fn chain_ctx(&self) -> ChainCtx {
        ChainCtx {
            n_layers: N_LAYERS as u32,
            layer_base: LAYERS as u32,
            embd_entry: EMBD as u32,
            eps: EPS,
            embd_scale: 1.0,
            trim_last: false,
            n_used: 0,
            n_attn: 0, n_recur: 0, seq_heads: 0,
        }
    }

    pub(crate) fn entry(&self, i: usize) -> &Entry {
        &self.entries[i].0
    }

    fn tensor(&self, i: usize) -> &[u8] {
        let e = self.entry(i);
        let off = e.chunk_base as usize * 32;
        &self.weights[off..off + e.n_bytes as usize]
    }

    pub(crate) fn row(&self, i: usize, r: usize) -> &[u8] {
        let rb = self.entry(i).row_bytes as usize;
        &self.tensor(i)[r * rb..(r + 1) * rb]
    }

    fn f32_tensor(&self, i: usize) -> Vec<f32> {
        self.tensor(i).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn matvec(&self, i: usize, x: &[u8]) -> Vec<f64> {
        (0..self.entry(i).m as usize)
            .map(|r| dot::canonical_dot(quant::Q8_0, self.row(i, r), D, x, q8::MMVQ).unwrap().y_hat)
            .collect()
    }

    fn embedding(&self, tok: u32) -> Vec<f32> {
        quant::dequant_row(quant::Q8_0, self.row(EMBD, tok as usize), D).unwrap()
    }

    /// A stage-2 trace of an honest pass over `tokens`, one position per token.
    pub fn trace(&self, tokens: &[u32], n_prompt: u32) -> Trace {
        self.trace_declaring(tokens, n_prompt, None)
    }

    /// [`Self::trace`] committing batch size `ubatch` at every position when given.
    pub fn trace_declaring(&self, tokens: &[u32], n_prompt: u32, ubatch: Option<u32>) -> Trace {
        self.trace_pinned(tokens, n_prompt, ubatch, None)
    }

    /// [`Self::trace_declaring`] with residual `(layer, kind)` committed as `pin` at every position;
    /// with `via_add`, the product output feeding it is committed as `pin - src`.
    fn trace_pinned(&self, tokens: &[u32], n_prompt: u32, ubatch: Option<u32>, pin: Option<(usize, Resid, &[f32], bool)>) -> Trace {
        let hit = |l: usize, kind: Resid| pin.filter(|&(pl, pk, _, _)| pl == l && pk == kind);
        let pinned = |l: usize, kind: Resid, v: Vec<f32>| hit(l, kind).map_or(v, |(_, _, c, _)| c.to_vec());
        let add = |l: usize, kind: Resid, y: Vec<f32>, src: &[f32]| match hit(l, kind) {
            Some((_, _, c, true)) => c.iter().zip(src).map(|(a, b)| a - b).collect(),
            _ => y,
        };
        let np = N_PRODUCTS as u64;
        let n_leaves = chain::n_pos_leaves(np, N_LAYERS as u64, 0) as usize;
        let as_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&a| a as f32).collect() };
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|a| a.to_le_bytes()).collect() };
        let silu = |g: f64| g / (1.0 + libm::exp(-g));

        let empty = |leaves: &mut [Vec<u8>], r: usize| {
            leaves[2 * r].clear();
            leaves[2 * r + 1].clear();
        };

        let mut positions = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let mut leaves = vec![Vec::new(); n_leaves];
            let mut h = self.embedding(tok);
            for l in 0..N_LAYERS {
                let x_in = if l == 0 {
                    q8_1_row(&expected_input(&h, &self.f32_tensor(attn_norm(l)), EPS, 1.0))
                } else {
                    q8_1_row(&chain::rms_norm(&h, &self.f32_tensor(attn_norm(l)), EPS))
                };
                let y_q = self.matvec(rank(l, ATTN_Q), &x_in);
                let y_fa = as_f32(&self.matvec(rank(l, SSM_F_A), &x_in));
                let x_fb = q8_1_row(&y_fa.iter().map(|&a| a as f64).collect::<Vec<_>>());
                let y_fb = as_f32(&self.matvec(rank(l, SSM_F_B), &x_fb));
                let mix: Vec<f64> = (0..D).map(|i| y_q[i % Q_M]).collect();
                let x_ao = q8_1_row(&mix);
                let y_ao = add(l, Resid::Attn, as_f32(&self.matvec(rank(l, ATTN_OUTPUT), &x_ao)), &h);
                let r: Vec<f32> = pinned(l, Resid::Attn, y_ao.iter().zip(&h).map(|(a, b)| a + b).collect());

                let x_f = q8_1_row(&chain::rms_norm(&r, &self.f32_tensor(ffn_norm(l)), EPS));
                let g = self.matvec(rank(l, FFN_GATE), &x_f);
                let u = self.matvec(rank(l, FFN_UP), &x_f);
                let hglu: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| (silu(a) * b) as f32).collect();
                let x_d = q8_1_row(&hglu.iter().map(|&a| a as f64).collect::<Vec<_>>());
                let y_d = add(l, Resid::Out, as_f32(&self.matvec(rank(l, FFN_DOWN), &x_d)), &r);
                let out: Vec<f32> = pinned(l, Resid::Out, y_d.iter().zip(&r).map(|(a, b)| a + b).collect());

                for (p, x, y) in [
                    (ATTN_Q, &x_in, bytes(&as_f32(&y_q))),
                    (ATTN_OUTPUT, &x_ao, bytes(&y_ao)),
                    (FFN_GATE, &x_f, bytes(&hglu)),
                    (FFN_UP, &x_f, Vec::new()),
                    (FFN_DOWN, &x_d, bytes(&y_d)),
                    (SSM_F_A, &x_in, bytes(&y_fa)),
                    (SSM_F_B, &x_fb, bytes(&y_fb)),
                ] {
                    leaves[2 * rank(l, p)] = in_layout(x, layout_of(ubatch, t as u64, n_prompt).1);
                    leaves[2 * rank(l, p) + 1] = y;
                }
                leaves[chain::resid_leaf(np, l as u64, Resid::Attn) as usize] = bytes(&r);
                leaves[chain::resid_leaf(np, l as u64, Resid::Out) as usize] = bytes(&out);
                h = out;
            }
            if t + 1 >= n_prompt as usize {
                let x_o = q8_1_row(&chain::rms_norm(&h, &self.f32_tensor(OUTPUT_NORM), EPS));
                leaves[2 * OUTPUT + 1] = bytes(&as_f32(&self.matvec(OUTPUT, &x_o)));
                leaves[2 * OUTPUT] = x_o;
            } else {
                empty(&mut leaves, OUTPUT);
            }
            leaves[chain::meta_leaf(np, N_LAYERS as u64, 0) as usize] = chain::meta(t as u32, layout_of(ubatch, t as u64, n_prompt).0, tok).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let trace_root = b3_root(&pos_roots);
        Trace { n_prompt, positions, pos_roots, trace_root, ubatch }
    }

    /// A trace over `tokens` with every product input `zero_q8_row(d)`, every output zero and
    /// every residual equal to the token embedding.
    pub fn scaled_zero_trace(&self, tokens: &[u32], n_prompt: u32, d: f32) -> Trace {
        let np = N_PRODUCTS as u64;
        let n_leaves = chain::n_pos_leaves(np, N_LAYERS as u64, 0) as usize;
        let x = zero_q8_row(d);
        let mut positions = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let mut leaves = vec![Vec::new(); n_leaves];
            let h: Vec<u8> = self.embedding(tok).iter().flat_map(|a| a.to_le_bytes()).collect();
            for l in 0..N_LAYERS {
                for p in 0..PER_LAYER {
                    let r = rank(l, p);
                    leaves[2 * r] = in_layout(&x, layout_at(t as u64, n_prompt).1);
                    leaves[2 * r + 1] = if p == FFN_UP { Vec::new() } else { vec![0u8; 4 * self.entry(r).m as usize] };
                }
                leaves[chain::resid_leaf(np, l as u64, Resid::Attn) as usize] = h.clone();
                leaves[chain::resid_leaf(np, l as u64, Resid::Out) as usize] = h.clone();
            }
            if t + 1 >= n_prompt as usize {
                leaves[2 * OUTPUT] = x.clone();
                leaves[2 * OUTPUT + 1] = vec![0u8; 4 * HEAD_M];
            }
            leaves[chain::meta_leaf(np, N_LAYERS as u64, 0) as usize] = chain::meta(t as u32, layout_at(t as u64, n_prompt).0, tok).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let trace_root = b3_root(&pos_roots);
        Trace { n_prompt, positions, pos_roots, trace_root, ubatch: None }
    }
}

impl Tier {
    pub fn entry_opening(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].1.clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    fn range(&self, first: u64, last: u64) -> ChunkRange {
        ChunkRange {
            first_chunk: first,
            chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
            proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
        }
    }

    fn bytes_range(&self, i: usize, offset: u64, len: u64) -> ChunkRange {
        let base = self.entry(i).chunk_base;
        self.range(base + offset / 32, base + (offset + len - 1) / 32)
    }


    fn norm_fragment_of(&self, i: usize, block: u64) -> (EntryOpening, ChunkRange) {
        let k = self.entry(i).k;
        let (b0, n) = chain::norm_fragment(block, k);
        let len = (n * 32).min(k - b0 * 32);
        (self.entry_opening(i), self.bytes_range(i, 4 * b0 * 32, 4 * len))
    }

    fn row_of(&self, i: usize, r: u64) -> (EntryOpening, ChunkRange) {
        let rb = self.entry(i).row_bytes as u64;
        (self.entry_opening(i), self.bytes_range(i, r * rb, rb))
    }
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

fn n_chunks(len: usize) -> u64 {
    len.div_ceil(B3_CHUNK).max(1) as u64
}

impl Tier {
    /// Point `i` drawn by `seed`, with its token and input link.
    pub fn open_point(&self, trace: &Trace, seed: &Hash, i: u32) -> (Point, u32, LinkOpening) {
        let d = draw::point(seed, i);
        let np = N_PRODUCTS as u64;
        let n_pos = trace.positions.len() as u64;
        let drawn = (d[1] % np) as usize;
        let t = if drawn == OUTPUT {
            let first = trace.n_prompt as u64 - 1;
            first + d[0] % (n_pos - first)
        } else {
            d[0] % n_pos
        };
        let gate = if self.entry(drawn).glu_form == manifest::FORM_PAIR_UP { self.entry(drawn).partner as usize } else { drawn };
        let ge = self.entry(gate);
        let row = d[2] % ge.m;
        let pos = &trace.positions[t as usize];

        let pair = ge.glu_form == manifest::FORM_PAIR_GATE;
        let mut rows = vec![self.row_of(gate, row).1];
        if pair {
            rows.push(self.row_of(ge.partner as usize, row).1);
        }
        let y = &pos.leaves[2 * gate + 1];
        let yc = (row * 4 / B3_CHUNK as u64) as usize;
        let ych = leaf_chunk(pos, 2 * gate as u64 + 1, yc as u64);
        let point = Point {
            n_ubatch: trace.layout(t).0,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            entry: self.entry_opening(drawn),
            partner: (drawn != gate || pair).then(|| self.entry_opening(if drawn == gate { ge.partner as usize } else { gate })),
            bias: None,
            rows,
            x: pos.leaves[2 * gate].clone(),
            x_path: b3_path(&pos.hashes, 2 * gate),
            y_total_len: y.len() as u32,
            y_chunk_index: yc as u32,
            y_chunk: ych.chunk,
            y_chunk_path: ych.chunk_path,
            y_path: b3_path(&pos.hashes, 2 * gate + 1),
        };

        let Chain::Product(pl) = ge.chain else { panic!("product without link") };
        let layer = pl.layer as u64;
        let layout = crate::verify::input_layout(self.entry(drawn).role, point.n_ubatch, ge.wtype);
        let block = || chain::link_block(seed, i, &point.x, layout, ge.k).expect("link block");
        let link = match pl.link {
            manifest::LINK_NONE => LinkOpening::None,
            manifest::LINK_ATTN_IN if layer == 0 => {
                LinkOpening::Token { embd: self.row_of(EMBD, pos.tok as u64), norm: self.norm_fragment_of(pl.norm as usize, block()) }
            }
            manifest::LINK_GLU | manifest::LINK_DIRECT => {
                let src = pl.src as u64;
                let c = block() / chain::CHUNK_BLOCKS;
                LinkOpening::Source { src: self.entry_opening(src as usize), y: leaf_chunk(pos, 2 * src + 1, c), ids: None }
            }
            link => {
                let leaf = match link {
                    manifest::LINK_ATTN_IN => chain::resid_leaf(np, layer - 1, Resid::Out),
                    manifest::LINK_FFN_IN => chain::resid_leaf(np, layer, Resid::Attn),
                    _ => chain::resid_leaf(np, N_LAYERS as u64 - 1, Resid::Out),
                };
                LinkOpening::Norm { src: leaf_bytes(pos, leaf), norm: self.norm_fragment_of(pl.norm as usize, block()) }
            }
        };
        (point, pos.tok, link)
    }
}

impl Tier {
    /// Prefix context of the synthetic tier: identity byte table.
    pub fn prefix_ctx(&self, offset: u32) -> PrefixCtx {
        PrefixCtx {
            table: std::array::from_fn(|b| b as u32),
            offset,
            embd_entry: EMBD as u32,
            norm_entry: attn_norm(0) as u32,
            eps: EPS,
            embd_scale: 1.0,
        }
    }

    /// Stage-2 prefix point drawn by `seed` at `offset`.
    pub fn open_prefix(&self, trace: &Trace, seed: &Hash, offset: u32) -> PrefixPoint {
        let np = N_PRODUCTS as u64;
        let t = prefix::position(seed, offset) as usize;
        let pos = &trace.positions[t];
        let k = self.entry(attn_norm(0)).k;
        let layout = crate::verify::input_layout(self.entry(0).role, trace.layout(t as u64).0, self.entry(0).wtype);
        let block = chain::link_block(seed, prefix::PREFIX_POINT, &pos.leaves[0], layout, k).expect("link block");
        let (b0, n) = chain::norm_fragment(block, k);
        PrefixPoint {
            n_ubatch: trace.layout(t as u64).0,
            pos_path: b3_path(&trace.pos_roots, t),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            entry: self.entry_opening(0),
            x: pos.leaves[0].clone(),
            x_path: b3_path(&pos.hashes, 0),
            embd: self.row_of(EMBD, pos.tok as u64),
            norm: (self.entry_opening(attn_norm(0)), self.bytes_range(attn_norm(0), 4 * b0 * 32, 4 * (n * 32).min(k - b0 * 32))),
        }
    }

    /// Residual point drawn by `seed`.
    pub fn open_resid(&self, trace: &Trace, seed: &Hash) -> ResidPoint {
        self.open_resid_with(&self.chain_ctx(), trace, seed)
    }

    /// Residual point drawn by `seed` under `ctx`.
    pub fn open_resid_with(&self, ctx: &ChainCtx, trace: &Trace, seed: &Hash) -> ResidPoint {
        let d = draw::point(seed, chain::RESID_POINT);
        let np = N_PRODUCTS as u64;
        let resp = crate::verify::ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 };
        let sel = chain::resid_site(d[1], 2 * N_LAYERS as u64);
        let t = chain::resid_position(ctx, &resp, sel / 2, d[0]).expect("residual position");
        let sel = chain::resid_site(d[1], 2 * N_LAYERS as u64);
        let (layer, kind) = (sel / 2, if sel.is_multiple_of(2) { Resid::Attn } else { Resid::Out });
        let pos = &trace.positions[t as usize];
        let product = if kind == Resid::Attn { rank(layer as usize, ATTN_OUTPUT) } else { rank(layer as usize, FFN_DOWN) };
        let dst_leaf = chain::resid_leaf(np, layer, kind);
        let c = d[2] % n_chunks(pos.leaves[dst_leaf as usize].len());
        let src = match kind {
            Resid::Attn if layer == 0 => {
                let (e, r) = self.row_of(EMBD, pos.tok as u64);
                ResidSrc::Token(e, r)
            }
            Resid::Attn => ResidSrc::Chunk(leaf_chunk(pos, chain::resid_leaf(np, layer - 1, Resid::Out), c)),
            Resid::Out => ResidSrc::Chunk(leaf_chunk(pos, chain::resid_leaf(np, layer, Resid::Attn), c)),
        };
        ResidPoint {
            n_ubatch: trace.layout(t).0,
            tok: pos.tok,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            layer: self.entry_opening(LAYERS + layer as usize),
            product: self.entry_opening(product),
            dst: leaf_chunk(pos, dst_leaf, c),
            src,
            add: ResidAdd::Chunk(leaf_chunk(pos, 2 * product as u64 + 1, c)),
        }
    }

    /// Residual opening drawn by `seed`: the residual point and one row of its feeding product.
    pub fn open_resid_piece(&self, trace: &Trace, seed: &Hash) -> crate::opening::ResidOpening {
        let resid = self.open_resid(trace, seed);
        let d = draw::point(seed, chain::RESID_POINT);
        let product = resid.product.index as usize;
        let per = (B3_CHUNK / 4) as u64;
        let c = resid.dst.index as u64;
        let row = c * per + (d[3] >> 32) % per.min(self.entry(product).m - c * per);
        let resp = crate::verify::ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 };
        let sel = chain::resid_site(d[1], 2 * N_LAYERS as u64);
        let t = chain::resid_position(&self.chain_ctx(), &resp, sel / 2, d[0]).expect("residual position");
        let pos = &trace.positions[t as usize];
        let feed = crate::opening::FeedRow {
            x: pos.leaves[2 * product].clone(),
            x_path: b3_path(&pos.hashes, 2 * product),
            row: self.row_of(product, row).1,
            bias: None,
        };
        crate::opening::ResidOpening { response_txid: [0; 32], roots: None, resid, feed, expert: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{verify_chain_point, verify_resid_point};
    use crate::verify::ResponseCtx;

    fn honest() -> (Tier, Trace) {
        let tier = Tier::new();
        let tokens: Vec<u32> = (0..12u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect();
        let trace = tier.trace(&tokens, 8);
        for p in &trace.positions {
            for (i, l) in p.leaves.iter().enumerate() {
                let f32_leaf = (i < 2 * N_PRODUCTS && i % 2 == 1) || (i >= 2 * N_PRODUCTS && i < chain::meta_leaf(N_PRODUCTS as u64, N_LAYERS as u64, 0) as usize);
                if f32_leaf {
                    assert!(l.chunks_exact(4).all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite()), "leaf {i}");
                }
            }
        }
        (tier, trace)
    }

    fn resp(trace: &Trace) -> ResponseCtx {
        ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 }
    }

    #[test]
    fn honest_trace_passes_every_draw() {
        let (tier, trace) = honest();
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut links = [0usize; 5];
        for s in 0..64u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                let kind = match &link {
                    LinkOpening::None => 0,
                    LinkOpening::Token { .. } => 1,
                    LinkOpening::Norm { .. } => 2,
                    LinkOpening::Source { .. } => 3,
                };
                links[kind] += 1;
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).unwrap_or_else(|e| panic!("seed {s} point {i}: {e:?}"));
            }
            verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed)).unwrap_or_else(|e| panic!("seed {s} resid: {e:?}"));
            chain::verify_resid_opening(&tc, &cc, &rc, &seed, &tier.open_resid_piece(&trace, &seed)).unwrap_or_else(|e| panic!("seed {s} resid piece: {e:?}"));
        }
        assert!(links[..4].iter().all(|&n| n > 0), "{links:?}");
    }

    #[test]
    fn a_product_output_off_its_input_is_refused_by_the_residual_piece() {
        let tier = Tier::new();
        let tokens: Vec<u32> = (0..12u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect();
        let honest = tier.trace(&tokens, 8);
        let (tc, cc) = (tier.tier(), tier.chain_ctx());
        for kind in [Resid::Attn, Resid::Out] {
            let leaf = chain::resid_leaf(N_PRODUCTS as u64, 0, kind) as usize;
            let c: Vec<f32> = chain::f32s(&honest.positions[0].leaves[leaf]);
            let forged = tier.trace_pinned(&tokens, 8, None, Some((0, kind, &c, true)));
            let rc = resp(&forged);
            let site = if kind == Resid::Attn { 0 } else { 1 };
            let (mut on_site, mut refused) = (0, 0);
            for s in 0..256u32 {
                let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                let d = draw::point(&seed, chain::RESID_POINT);
                let point = verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&forged, &seed));
                assert_eq!(point, Ok(()), "seed {s}");
                if chain::resid_site(d[1], 2 * N_LAYERS as u64) != site || d[0] % forged.positions.len() as u64 == 0 {
                    continue;
                }
                on_site += 1;
                let piece = chain::verify_resid_opening(&tc, &cc, &rc, &seed, &tier.open_resid_piece(&forged, &seed));
                refused += (piece == Err(crate::verify::Fail::Mismatch)) as u32;
            }
            assert!(on_site > 0 && refused == on_site, "{kind:?}: {refused} of {on_site}");
        }
    }

    #[test]
    fn residual_sites_lean_towards_the_first_layers() {
        let n = 8u64;
        let mut hits = vec![0u64; n as usize];
        for d1 in 0..n * (n + 1) / 2 {
            hits[chain::resid_site(d1, n) as usize] += 1;
        }
        assert_eq!(hits, (1..=n).rev().collect::<Vec<_>>());
        assert_eq!(chain::resid_site(u64::MAX, 1), 0);
    }

    #[test]
    fn stage2_opening_round_trips() {
        use crate::opening::{Opening, PrefixPoint};
        let (tier, trace) = honest();
        let seed = [7u8; 32];
        let (points, links): (Vec<_>, Vec<_>) = (0..draw::POINTS)
            .map(|i| {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                (p, (tok, link))
            })
            .unzip();
        let o = Opening {
            response_txid: [3; 32],
            points,
            prefix: Some(PrefixPoint::default()),
            experts: vec![None; links.len()],
            links,
            resid: Some(tier.open_resid(&trace, &seed)),
            roots: None,
        };
        let body = o.encode();
        assert_eq!(body[0], crate::opening::VERSION_CHAIN);
        assert_eq!(Opening::decode(&body), Some(o.clone()));
        let split = Opening { roots: Some(([4; 32], [5; 32])), ..o };
        let body = split.encode();
        assert_eq!(body[0], crate::opening::VERSION_ROOTS);
        assert_eq!(Opening::decode(&body), Some(split.clone()));
        let seeded = crate::opening::seeded(&body, &[6; 32]).unwrap();
        assert_eq!(Opening::decode(&seeded), Some(split));
    }

    #[test]
    fn a_response_with_oversized_input_scales_is_refused() {
        use crate::chain::verify_chain_prefix_point;
        let tier = Tier::new();
        let request = *blake3::hash(b"request").as_bytes();
        let mut tokens: Vec<u32> = request.iter().map(|&b| b as u32).collect();
        tokens.extend((0..8u32).map(|i| (i * 37 + 5) % VOCAB as u32));
        let (tc, cc, pc) = (tier.tier(), tier.chain_ctx(), tier.prefix_ctx(0));
        let accepted = |trace: &Trace| -> Vec<u32> {
            let rc = resp(trace);
            (0..48u32)
                .filter(|s| {
                    let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                    (0..draw::POINTS).all(|i| {
                        let (p, tok, link) = tier.open_point(trace, &seed, i);
                        verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).is_ok()
                    }) && verify_chain_prefix_point(&tc, &cc, &pc, &rc, &seed, &request, &tier.open_prefix(trace, &seed, 0)).is_ok()
                        && verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(trace, &seed)).is_ok()
                })
                .collect()
        };
        assert_eq!(accepted(&tier.trace(&tokens, prefix::PREFIX_LEN)).len(), 48);
        let forged = accepted(&tier.scaled_zero_trace(&tokens, prefix::PREFIX_LEN, 1.0e4));
        assert!(forged.is_empty(), "accepted for seeds {forged:?}");
    }

    #[test]
    fn a_zeroed_input_is_refused() {
        let (tier, mut trace) = honest();
        let r = rank(1, ATTN_Q);
        for pos in &mut trace.positions {
            let x = vec![0u8; pos.leaves[2 * r].len()];
            let y = vec![0u8; pos.leaves[2 * r + 1].len()];
            pos.leaves[2 * r] = x;
            pos.leaves[2 * r + 1] = y;
            pos.hashes[2 * r] = *blake3::hash(&pos.leaves[2 * r]).as_bytes();
            pos.hashes[2 * r + 1] = *blake3::hash(&pos.leaves[2 * r + 1]).as_bytes();
        }
        trace.pos_roots = trace.positions.iter().map(|p| b3_root(&p.hashes)).collect();
        trace.trace_root = b3_root(&trace.pos_roots);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut refused = 0;
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            for i in 0..draw::POINTS {
                if draw::point(&seed, i)[1] % N_PRODUCTS as u64 != r as u64 {
                    continue;
                }
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                assert!(verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).is_err(), "seed {s} point {i}");
                refused += 1;
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_zero_or_huge_residual_fails_the_norm_links_reading_it() {
        for scale in [0.0f32, 1e6] {
            let (tier, mut trace) = honest();
            let leaf = chain::resid_leaf(N_PRODUCTS as u64, 0, Resid::Out) as usize;
            for pos in &mut trace.positions {
                let v: Vec<u8> = pos.leaves[leaf].chunks_exact(4).flat_map(|b| (f32::from_le_bytes(b.try_into().unwrap()) * scale).to_le_bytes()).collect();
                pos.leaves[leaf] = v;
                pos.hashes[leaf] = *blake3::hash(&pos.leaves[leaf]).as_bytes();
            }
            trace.pos_roots = trace.positions.iter().map(|p| b3_root(&p.hashes)).collect();
            trace.trace_root = b3_root(&trace.pos_roots);
            let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
            let mut refused = 0;
            for s in 0..64u32 {
                let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                for i in 0..draw::POINTS {
                    let (p, tok, link) = tier.open_point(&trace, &seed, i);
                    if let LinkOpening::Norm { src, .. } = &link {
                        if src.bytes.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 0.0) || scale > 1.0 {
                            if matches!(verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None), Err(crate::verify::Fail::Mismatch)) {
                                refused += 1;
                            }
                        }
                    }
                }
            }
            assert!(refused > 0, "scale {scale}");
        }
    }

    #[test]
    fn a_changed_residual_is_refused() {
        let (tier, mut trace) = honest();
        let leaf = chain::resid_leaf(N_PRODUCTS as u64, 0, Resid::Attn) as usize;
        for pos in &mut trace.positions {
            let v = f32::from_le_bytes(pos.leaves[leaf][..4].try_into().unwrap()) + 0.5;
            pos.leaves[leaf][..4].copy_from_slice(&v.to_le_bytes());
            pos.hashes[leaf] = *blake3::hash(&pos.leaves[leaf]).as_bytes();
        }
        trace.pos_roots = trace.positions.iter().map(|p| b3_root(&p.hashes)).collect();
        trace.trace_root = b3_root(&trace.pos_roots);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut refused = 0;
        for s in 0..64u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, chain::RESID_POINT);
            let hits = chain::resid_site(d[1], 4) < 2 && d[2].is_multiple_of(2);
            let r = verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed));
            if hits {
                assert_eq!(r, Err(crate::verify::Fail::Mismatch), "seed {s}");
                refused += 1;
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_trimmed_last_layer_adds_its_output_only_product() {
        let (mut tier, trace) = honest();
        let down = rank(N_LAYERS - 1, FFN_DOWN);
        let (e, bytes) = &mut tier.entries[down];
        e.role = manifest::ROLE_OUTPUT;
        let nl = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        bytes[2 + nl + 44] = manifest::ROLE_OUTPUT;
        tier.entry_leaves = tier.entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        tier.m_t = rt_range_proof(&tier.entry_leaves, 0, 1).0;
        let (tc, rc) = (tier.tier(), resp(&trace));
        let trimmed = ChainCtx { trim_last: true, ..tier.chain_ctx() };
        let (mut ok, mut refused) = (0, 0);
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let sel = chain::resid_site(draw::point(&seed, chain::RESID_POINT)[1], 2 * N_LAYERS as u64);
            if sel != 2 * N_LAYERS as u64 - 1 {
                continue;
            }
            let r = verify_resid_point(&tc, &trimmed, &rc, &seed, &tier.open_resid_with(&trimmed, &trace, &seed));
            assert_eq!(r, Ok(()), "seed {s}");
            ok += 1;
            let untrimmed = tier.chain_ctx();
            let r = verify_resid_point(&tc, &untrimmed, &rc, &seed, &tier.open_resid_with(&untrimmed, &trace, &seed));
            assert_eq!(r, Err(crate::verify::Fail::Manifest), "seed {s}");
            refused += 1;
        }
        assert!(ok > 0 && refused > 0);
    }
}

#[cfg(test)]
mod expert_tests {
    use super::*;
    use crate::opening::ExpertIds;
    use crate::verify::{PosShape, ResponseCtx, verify_point_at};

    const N_EXP: usize = 4;
    const N_USED: usize = 2;
    const M: usize = 128;

    #[test]
    fn expert_product_point_verifies() {
        let w = q8_0_rows(77, N_EXP * M, 0.0004);
        let chunk_leaves: Vec<Hash> = w.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let rb = D / 32 * 34;
        let e = Entry {
            name: "blk.0.ffn_down_exps.weight".into(),
            wtype: quant::Q8_0,
            k: D as u64,
            m: M as u64,
            n_exp: N_EXP as u32,
            chunk_base: 0,
            n_bytes: w.len() as u64,
            row_bytes: rb as u32,
            role: manifest::ROLE_PRODUCT,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let eb = serialize(&e);
        let m_t = rt_range_proof(&[Entry::leaf(&eb)], 0, 1).0;
        let tier = TierCtx { r_t, n_chunks: chunk_leaves.len() as u64, m_t, n_entries: 1, n_products: 1 };

        let ids: [u32; N_USED] = [3, 1];
        let xs: Vec<Vec<u8>> = (0..N_USED).map(|s| q8_1_row(&(0..D).map(|i| ((i * 7 + s * 13) % 23) as f64 * 0.01 - 0.1).collect::<Vec<_>>())).collect();
        let mut y = Vec::new();
        for (s, &id) in ids.iter().enumerate() {
            for r in 0..M {
                let row = &w[(id as usize * M + r) * rb..(id as usize * M + r + 1) * rb];
                let v = dot::canonical_dot(quant::Q8_0, row, D, &xs[s], q8::MMVQ).unwrap().y_hat as f32;
                y.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &id in &ids {
            y.extend_from_slice(&id.to_le_bytes());
        }
        let x: Vec<u8> = xs.concat();
        let meta = [0u8, 0, 0, 0, 1, 0, 0, 0];
        let leaves = [*blake3::hash(&x).as_bytes(), *blake3::hash(&y).as_bytes(), *blake3::hash(&meta).as_bytes()];
        let pos_root = b3_root(&leaves);
        let resp = ResponseCtx { trace_root: b3_root(&[pos_root]), n_prompt: 1, n_pos: 1 };
        let shape = PosShape { n_leaves: 3, meta_index: 2, tok: None, n_used: N_USED as u32 };
        let pos = Position { tok: 0, leaves: vec![x.clone(), y.clone(), meta.to_vec()], hashes: leaves.to_vec() };

        let mut seen = [false; N_USED];
        for s in 0..32u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, 0);
            let (slot, row) = ((d[3] >> 32) % N_USED as u64, d[2] % M as u64);
            seen[slot as usize] = true;
            let exp = ids[slot as usize] as u64;
            let off = (exp * M as u64 + row) * rb as u64;
            let y_off = (slot * M as u64 + row) * 4;
            let yc = leaf_chunk(&pos, 1, y_off / B3_CHUNK as u64);
            let first = leaf_chunk(&pos, 1, (N_USED * M * 4 / B3_CHUNK) as u64);
            let p = Point {
                n_ubatch: 1,
                pos_path: Vec::new(),
                meta_path: b3_path(&leaves, 2),
                entry: EntryOpening { index: 0, bytes: eb.clone(), path: Vec::new() },
                partner: None,
                bias: None,
                rows: vec![ChunkRange {
                    first_chunk: off / 32,
                    chunks: w[(off / 32 * 32) as usize..((off + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                    proof: rt_range_proof(&chunk_leaves, off / 32, (off + rb as u64 - 1) / 32 - off / 32 + 1).1,
                }],
                x: x.clone(),
                x_path: b3_path(&leaves, 0),
                y_total_len: y.len() as u32,
                y_chunk_index: yc.index,
                y_chunk: yc.chunk,
                y_chunk_path: yc.chunk_path,
                y_path: b3_path(&leaves, 1),
            };
            let e_ids = ExpertIds { chunk: first.chunk, chunk_path: first.chunk_path };
            verify_point_at(&tier, &resp, &seed, 0, &p, &shape, Some(&e_ids)).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            assert!(verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).is_err());
        }
        assert!(seen.iter().all(|&b| b));
    }
}

#[cfg(test)]
mod head_link_tests {
    use super::*;
    use crate::chain::{ChainCtx, verify_chain_point};
    use crate::opening::{LinkOpening, Point};
    use crate::verify::ResponseCtx;

    const NH: usize = 4;
    const STRIDE: usize = D + 32;
    const M1: usize = D / NH;
    const NP: u64 = 3;

    struct H {
        offs: Vec<usize>,
        chunk_leaves: Vec<Hash>,
        all: Vec<u8>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        tier: TierCtx,
        ctx: ChainCtx,
        pos: Position,
        trace_root: Hash,
    }

    fn dot_rows(w: &[u8], rows: std::ops::Range<usize>, x: &[u8]) -> Vec<f32> {
        let rb = D / 32 * 34;
        rows.map(|r| dot::canonical_dot(quant::Q8_0, &w[r * rb..(r + 1) * rb], D, x, q8::MMVQ).unwrap().y_hat as f32).collect()
    }

    /// `shift`: each head reads its slice 32 values late; `zero_out`: the output projection reads zeros.
    fn build(shift: bool, zero_out: bool) -> H {
        build_with(shift, zero_out, 1.0)
    }

    /// The harness with the fourth input block of the unlinked product scaled by `spike`.
    fn build_with(shift: bool, zero_out: bool, spike: f64) -> H {
        let none = manifest::NONE;
        let weights = vec![q8_0_rows(11, NH * STRIDE, 0.0004), q8_0_rows(12, NH * M1, 0.0004), q8_0_rows(13, D, 0.0004)];
        let mut all = Vec::new();
        let mut offs = Vec::new();
        for w in &weights {
            offs.push(all.len());
            all.extend_from_slice(w);
        }
        let chunk_leaves: Vec<Hash> = all.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let rb = D / 32 * 34;
        let entry = |name: &str, role: u8, m: usize, n_exp: u32, i: usize, link: u8, src: u32| {
            let e = Entry {
                name: name.into(),
                wtype: quant::Q8_0,
                k: D as u64,
                m: m as u64,
                n_exp,
                chunk_base: (offs[i] / 32) as u64,
                n_bytes: weights[i].len() as u64,
                row_bytes: rb as u32,
                role,
                glu_op: 0,
                glu_form: manifest::FORM_NONE,
                partner: none,
                bias: none,
                chain: Chain::Product(ProductLink { layer: 0, link, norm: none, src }),
            };
            let mut b = serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            b
        };
        let entries = vec![
            entry("blk.0.attn_q.weight", manifest::ROLE_PRODUCT, NH * STRIDE, 1, 0, manifest::LINK_NONE, none),
            entry("blk.0.attn_k_b.weight", manifest::ROLE_HEADS, M1, NH as u32, 1, manifest::LINK_HEAD_SLICE, 0),
            entry("blk.0.attn_output.weight", manifest::ROLE_PRODUCT, D, 1, 2, manifest::LINK_DIRECT, 1),
        ];
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: NP,
            n_products: NP,
        };
        let ctx = ChainCtx { n_layers: 1, layer_base: 3, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 };
        let x0 = q8_1_row(&(0..D).map(|i| (((i * 7) % 23) as f64 * 0.01 - 0.1) * if i / 32 == 3 { spike } else { 1.0 }).collect::<Vec<_>>());
        let y0 = dot_rows(&weights[0], 0..NH * STRIDE, &x0);
        let off = if shift { 32 } else { 0 };
        let xs: Vec<Vec<u8>> = (0..NH).map(|h| q8_1_row(&y0[h * STRIDE + off..h * STRIDE + off + D].iter().map(|&a| a as f64).collect::<Vec<_>>())).collect();
        let y1: Vec<f32> = (0..NH).flat_map(|h| dot_rows(&weights[1], h * M1..(h + 1) * M1, &xs[h])).collect();
        let x2 = q8_1_row(&if zero_out { vec![0.0; D] } else { y1.iter().map(|&a| a as f64).collect() });
        let y2 = dot_rows(&weights[2], 0..D, &x2);
        let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
        let n_leaves = 2 * NP as usize + 2 + 1;
        let mut leaves = vec![Vec::new(); n_leaves];
        leaves[0] = x0;
        leaves[1] = f(&y0);
        leaves[2] = xs.concat();
        leaves[3] = f(&y1);
        leaves[4] = x2;
        leaves[5] = f(&y2);
        leaves[n_leaves - 1] = chain::meta(0, 1, 7).to_vec();
        let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let pos = Position { tok: 7, leaves, hashes };
        let trace_root = b3_root(&[b3_root(&pos.hashes)]);
        H { offs, chunk_leaves, all, entries, entry_leaves, tier, ctx, pos, trace_root }
    }

    impl H {
        fn eo(&self, i: usize) -> EntryOpening {
            EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
        }

        fn open(&self, seed: &Hash) -> (Point, LinkOpening) {
            let d = draw::point(seed, 0);
            let rank = (d[1] % NP) as usize;
            let (m, n_exp) = [(NH * STRIDE, 1), (M1, NH), (D, 1)][rank];
            let slot = if n_exp > 1 { (d[3] >> 32) % n_exp as u64 } else { 0 } as usize;
            let row = (d[2] % m as u64) as usize;
            let rb = (D / 32 * 34) as u64;
            let off = self.offs[rank] as u64 + (slot * m + row) as u64 * rb;
            let y = &self.pos.leaves[2 * rank + 1];
            let yc = leaf_chunk(&self.pos, 2 * rank as u64 + 1, ((slot * m + row) * 4 / B3_CHUNK) as u64);
            let p = Point {
                n_ubatch: 1,
                pos_path: Vec::new(),
                meta_path: b3_path(&self.pos.hashes, self.pos.hashes.len() - 1),
                entry: self.eo(rank),
                partner: None,
                bias: None,
                rows: vec![ChunkRange {
                    first_chunk: off / 32,
                    chunks: self.all[(off / 32 * 32) as usize..((off + rb - 1) / 32 * 32 + 32) as usize].to_vec(),
                    proof: rt_range_proof(&self.chunk_leaves, off / 32, (off + rb - 1) / 32 - off / 32 + 1).1,
                }],
                x: self.pos.leaves[2 * rank].clone(),
                x_path: b3_path(&self.pos.hashes, 2 * rank),
                y_total_len: y.len() as u32,
                y_chunk_index: yc.index,
                y_chunk: yc.chunk,
                y_chunk_path: yc.chunk_path,
                y_path: b3_path(&self.pos.hashes, 2 * rank + 1),
            };
            let x_one = crate::verify::x_len(D as u64, crate::q8::MMVQ) as usize;
            let xs = &p.x[slot * x_one..(slot + 1) * x_one];
            let block = chain::link_block(seed, 0, xs, crate::q8::MMVQ, D as u64).expect("link block") as usize;
            let link = match rank {
                0 => LinkOpening::None,
                1 => {
                    let c = (slot * STRIDE + block * 32) * 4 / B3_CHUNK;
                    LinkOpening::Source { src: self.eo(0), y: leaf_chunk(&self.pos, 1, c as u64), ids: None }
                }
                _ => {
                    let c = block as u64 / chain::CHUNK_BLOCKS;
                    LinkOpening::Source { src: self.eo(1), y: leaf_chunk(&self.pos, 3, c), ids: None }
                }
            };
            (p, link)
        }

        fn check(&self, seed: &Hash) -> (usize, Result<(), crate::verify::Fail>) {
            let (p, link) = self.open(seed);
            let resp = ResponseCtx { trace_root: self.trace_root, n_prompt: 1, n_pos: 1 };
            let rank = (draw::point(seed, 0)[1] % NP) as usize;
            (rank, verify_chain_point(&self.tier, &self.ctx, &resp, seed, 0, &p, 7, &link, None).map(|_| ()))
        }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..96u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    #[test]
    fn honest_head_slices_and_head_outputs_pass() {
        let h = build(false, false);
        let mut seen = [false; 3];
        for s in seeds() {
            let (rank, r) = h.check(&s);
            seen[rank] = true;
            r.unwrap_or_else(|e| panic!("rank {rank}: {e:?}"));
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn a_misplaced_head_slice_is_refused() {
        let h = build(true, false);
        for s in seeds() {
            let (rank, r) = h.check(&s);
            assert_eq!(r.is_err(), rank == 1, "rank {rank}");
        }
    }

    #[test]
    fn an_unlinked_input_with_a_spiked_block_is_refused() {
        for (spike, refused) in [(500.0, false), (4000.0, true)] {
            let h = build_with(false, false, spike);
            let mut seen = false;
            for s in seeds() {
                let (rank, r) = h.check(&s);
                if rank == 0 {
                    seen = true;
                    assert_eq!(r.is_err(), refused, "spike {spike}");
                }
            }
            assert!(seen);
        }
    }

    #[test]
    fn an_output_projection_cut_from_the_heads_is_refused() {
        let h = build(false, true);
        for s in seeds() {
            let (rank, r) = h.check(&s);
            assert_eq!(r.is_err(), rank == 2, "rank {rank}");
        }
    }
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    use crate::chain::{route_weights, verify_resid_point};
    use crate::opening::{ExpertIds, MoeAdd};
    use crate::verify::ResponseCtx;

    const E: usize = 16;
    const U: usize = 4;
    const SCALE: f32 = 2.5;

    fn vecf(seed: u64, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
    }

    fn bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|a| a.to_le_bytes()).collect()
    }

    struct Moe {
        weights: Vec<u8>,
        chunk_leaves: Vec<Hash>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        tier: TierCtx,
        ctx: ChainCtx,
        pos: Position,
        trace_root: Hash,
        n_ubatch: u32,
        exps_off: usize,
        shexp_off: usize,
        ids: Vec<u32>,
    }

    /// Product output committed with no tie to its input.
    #[derive(Clone, Copy, PartialEq)]
    enum Forge {
        Shexp,
        Experts,
    }

    const RB: usize = D / 32 * 34;

    fn entry(name: &str, role: u8, k: usize, m: usize, n_exp: u32, off: usize, n: usize, rb: usize, chain: Chain) -> Vec<u8> {
        let e = Entry {
            name: name.into(),
            wtype: if matches!(role, manifest::ROLE_PRODUCT | manifest::ROLE_OUTPUT) { quant::Q8_0 } else { 0 },
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

    fn build(order_swap: bool, bump_slot: Option<usize>, batch: u32) -> Moe {
        build_with(order_swap, bump_slot, batch, None)
    }

    /// `gate_scale`: router input committed as `peaked_q8_row(gate_scale)` under scores of no input.
    fn build_with(order_swap: bool, bump_slot: Option<usize>, batch: u32, gate_scale: Option<f32>) -> Moe {
        build_full(order_swap, bump_slot, batch, gate_scale, None)
    }

    fn build_full(order_swap: bool, bump_slot: Option<usize>, batch: u32, gate_scale: Option<f32>, forge: Option<Forge>) -> Moe {
        let router = vecf(1, E * D, 0.05);
        let bias = vecf(2, E, 0.01);
        let norm: Vec<f32> = (0..D).map(|i| 1.0 + 0.001 * (i % 7) as f32).collect();
        let mut weights = bytes(&router);
        let bias_off = weights.len();
        weights.extend_from_slice(&bytes(&bias));
        let norm_off = weights.len();
        weights.extend_from_slice(&bytes(&norm));
        let exps_off = weights.len();
        weights.extend_from_slice(&q8_0_rows(40, E * D, 0.0004));
        let shexp_off = weights.len();
        weights.extend_from_slice(&q8_0_rows(41, D, 0.0004));
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();

        let none = manifest::NONE;
        let last = if batch > 1 { manifest::ROLE_OUTPUT } else { manifest::ROLE_PRODUCT };
        let moe = manifest::MoeLinks { down_exps: 1, router: 4, probs_bias: 5, w_scale: SCALE.to_bits(), ffn_norm: 6, gate_exps: 3 };
        let entries = vec![
            entry("blk.0.attn_output.weight", manifest::ROLE_PRODUCT, D, D, 1, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_down_exps.weight", last, D, D, E as u32, exps_off, E * D * RB, RB, Chain::None),
            entry("blk.0.ffn_down_shexp.weight", last, D, D, 1, shexp_off, D * RB, RB, Chain::None),
            entry("blk.0.ffn_gate_exps.weight", last, D, D, E as u32, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_gate_inp.weight", manifest::ROLE_ROUTER, D, E, 1, 0, 4 * E * D, 4 * D, Chain::None),
            entry("blk.0.exp_probs_b.bias", manifest::ROLE_NORM, E, 1, 1, bias_off, 4 * E, 4 * E, Chain::None),
            entry("blk.0.ffn_norm.weight", manifest::ROLE_NORM, D, 1, 1, norm_off, 4 * D, 4 * D, Chain::None),
            entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, 0, 0, Chain::Layer(LayerLinks {
                layer: 0, proj: 0, down: 2, post_attn: none, post_ffn: none, scale: none, moe: Some(moe), attn: None, recur: None,
            })),
        ];
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: entries.len() as u64,
            n_products: 4,
        };
        let ctx = ChainCtx { n_layers: 1, layer_base: 7, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: batch > 1, n_used: U as u32, n_attn: 0, n_recur: 0, seq_heads: 0 };

        let r = vecf(3, D, 2.0);
        let x = chain::rms_norm(&r, &norm, EPS);
        let scores: Vec<f32> = match gate_scale {
            Some(_) => vecf(30, E, 3.0),
            None => (0..E).map(|e| router[e * D..(e + 1) * D].iter().zip(&x).map(|(&w, &v)| w as f64 * v).sum::<f64>() as f32).collect(),
        };
        let sel: Vec<f64> = scores.iter().zip(&bias).map(|(&s, &b)| 1.0 / (1.0 + (-(s as f64)).exp()) + b as f64).collect();
        let mut ids: Vec<u32> = (0..E as u32).collect();
        ids.sort_by(|&a, &b| sel[b as usize].total_cmp(&sel[a as usize]));
        ids.truncate(U);
        let w = route_weights(&scores, &bias, &ids, SCALE as f64).unwrap();
        let row_dot = |off: usize, x: &[u8]| crate::dot::canonical_dot(quant::Q8_0, &weights[off..off + RB], D, x, q8::MMVQ).unwrap().y_hat as f32;
        let q = |v: Vec<f32>| q8_1_row(&v.iter().map(|&a| a as f64).collect::<Vec<_>>());
        let xs: Vec<Vec<u8>> = (0..U).map(|s| q(vecf(50 + s as u64, D, 1.0))).collect();
        let x_sh = q(vecf(60, D, 1.0));
        let ys: Vec<Vec<f32>> = match forge {
            Some(Forge::Experts) => (0..U).map(|s| vecf(10 + s as u64, D, 0.3)).collect(),
            _ => (0..U).map(|s| (0..D).map(|j| row_dot(exps_off + (ids[s] as usize * D + j) * RB, &xs[s])).collect()).collect(),
        };
        let sh: Vec<f32> = match forge {
            Some(Forge::Shexp) => vecf(20, D, 0.3),
            _ => (0..D).map(|j| row_dot(shexp_off + j * RB, &x_sh)).collect(),
        };
        let out: Vec<f32> = (0..D).map(|j| (w.iter().zip(&ys).map(|(a, y)| a * y[j] as f64).sum::<f64>() + sh[j] as f64 + r[j] as f64) as f32).collect();
        let mut y_exps: Vec<u8> = ys.iter().flat_map(|y| bytes(y)).collect();
        if let Some(s) = bump_slot {
            let v = f32::from_le_bytes(y_exps[s * D * 4..s * D * 4 + 4].try_into().unwrap()) + 0.25;
            y_exps[s * D * 4..s * D * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut committed = ids.clone();
        if order_swap {
            committed.swap(0, U - 1);
        }
        for id in &committed {
            y_exps.extend_from_slice(&id.to_le_bytes());
        }
        let np = 4u64;
        let n_leaves = chain::n_pos_leaves(np, 1, 1) as usize;
        let mut leaves = vec![Vec::new(); n_leaves];
        leaves[6] = match gate_scale {
            Some(d) => peaked_q8_row(d),
            None => q8_1_row(&x),
        };
        leaves[2] = xs.concat();
        leaves[3] = y_exps;
        leaves[4] = x_sh;
        leaves[5] = bytes(&sh);
        leaves[chain::resid_leaf(np, 0, Resid::Attn) as usize] = bytes(&r);
        leaves[chain::resid_leaf(np, 0, Resid::Out) as usize] = bytes(&out);
        leaves[chain::router_leaf(np, 1, 0) as usize] = bytes(&scores);
        leaves[chain::meta_leaf(np, 1, 1) as usize] = chain::meta(0, 1, 7).to_vec();
        let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let trace_root = b3_root(&[b3_root(&hashes)]);
        Moe { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos: Position { tok: 7, leaves, hashes }, trace_root, n_ubatch: 1, exps_off, shexp_off, ids: committed }
    }

    impl Moe {
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

        fn open(&self, seed: &Hash) -> Option<ResidPoint> {
            let d = draw::point(seed, chain::RESID_POINT);
            if chain::resid_site(d[1], 2) != 1 {
                return None;
            }
            let np = 4u64;
            let c = d[2] % (D as u64 * 4 / B3_CHUNK as u64);
            let per = (D * 4 / B3_CHUNK) as u64;
            let row = d[3] % E as u64;
            let ids_chunk = leaf_chunk(&self.pos, 3, (U * D * 4 / B3_CHUNK) as u64);
            Some(ResidPoint {
                n_ubatch: self.n_ubatch,
                tok: 7,
                pos_path: Vec::new(),
                meta_path: b3_path(&self.pos.hashes, chain::meta_leaf(np, 1, 1) as usize),
                layer: self.eo(7),
                product: self.eo(2),
                dst: leaf_chunk(&self.pos, chain::resid_leaf(np, 0, Resid::Out), c),
                src: ResidSrc::Chunk(leaf_chunk(&self.pos, chain::resid_leaf(np, 0, Resid::Attn), c)),
                add: ResidAdd::Moe(Box::new(MoeAdd {
                    scores: leaf_bytes(&self.pos, chain::router_leaf(np, 1, 0)),
                    bias: (self.eo(5), self.range(4 * E * D, 4 * E)),
                    ids: ExpertIds { chunk: ids_chunk.chunk, chunk_path: ids_chunk.chunk_path },
                    slots: (0..U as u64).map(|s| leaf_chunk(&self.pos, 3, s * per + c)).collect(),
                    shexp: leaf_chunk(&self.pos, 5, c),
                    router_row: (self.eo(4), self.range(row as usize * 4 * D, 4 * D)),
                    resid: leaf_bytes(&self.pos, chain::resid_leaf(np, 0, Resid::Attn)),
                    norm: (self.eo(6), self.range(4 * E * D + 4 * E, 4 * D)),
                })),
            })
        }

        fn check(&self, seed: &Hash) -> Option<Result<(), crate::verify::Fail>> {
            let resp = ResponseCtx { trace_root: self.trace_root, n_prompt: 1, n_pos: 1 };
            let p = self.open(seed)?;
            Some(verify_resid_point(&self.tier, &self.ctx, &resp, seed, &p))
        }

        fn open_piece(&self, seed: &Hash) -> Option<crate::opening::ResidOpening> {
            let resid = self.open(seed)?;
            let d = draw::point(seed, chain::RESID_POINT);
            let per = (B3_CHUNK / 4) as u64;
            let c = resid.dst.index as u64;
            let row = (c * per + (d[3] >> 32) % per.min(D as u64 - c * per)) as usize;
            let id = self.ids[((d[3] >> 48) % U as u64) as usize] as usize;
            let feed = crate::opening::FeedRow { x: self.pos.leaves[4].clone(), x_path: b3_path(&self.pos.hashes, 4), row: self.range(self.shexp_off + row * RB, RB), bias: None };
            let expert = crate::opening::ExpertRow {
                entry: self.eo(1),
                x: self.pos.leaves[2].clone(),
                x_path: b3_path(&self.pos.hashes, 2),
                row: self.range(self.exps_off + (id * D + row) * RB, RB),
            };
            Some(crate::opening::ResidOpening { response_txid: [0; 32], roots: None, resid, feed, expert: Some(expert) })
        }

        fn check_piece(&self, seed: &Hash) -> Option<Result<(), crate::verify::Fail>> {
            let resp = ResponseCtx { trace_root: self.trace_root, n_prompt: 1, n_pos: 1 };
            Some(crate::chain::verify_resid_opening(&self.tier, &self.ctx, &resp, seed, &self.open_piece(seed)?))
        }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..64u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    #[test]
    fn honest_moe_layer_passes() {
        let m = build(false, None, 1);
        let n = seeds().filter_map(|s| m.check(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
        let n = seeds().filter_map(|s| m.check_piece(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
    }

    #[test]
    fn an_expert_output_off_its_input_is_refused_by_the_residual_piece() {
        for forge in [Forge::Shexp, Forge::Experts] {
            let m = build_full(false, None, 1, None, Some(forge));
            let points: Vec<_> = seeds().filter_map(|s| m.check(&s)).collect();
            assert!(!points.is_empty() && points.iter().all(Result::is_ok));
            let pieces: Vec<_> = seeds().filter_map(|s| m.check_piece(&s)).collect();
            assert!(pieces.iter().all(|r| *r == Err(crate::verify::Fail::Mismatch)), "{pieces:?}");
        }
    }

    #[test]
    fn a_moe_piece_must_carry_its_expert_row() {
        let m = build(false, None, 1);
        let resp = ResponseCtx { trace_root: m.trace_root, n_prompt: 1, n_pos: 1 };
        let mut n = 0;
        for s in seeds() {
            let Some(mut op) = m.open_piece(&s) else { continue };
            op.expert = None;
            assert_eq!(crate::chain::verify_resid_opening(&m.tier, &m.ctx, &resp, &s, &op), Err(crate::verify::Fail::Shape));
            n += 1;
        }
        assert!(n > 0);
    }

    #[test]
    fn router_scores_over_an_oversized_input_scale_are_refused() {
        let m = build_with(false, None, 1, Some(1.0e4));
        let results: Vec<_> = seeds().filter_map(|s| m.check(&s)).collect();
        assert!(!results.is_empty() && results.iter().all(|r| r.is_err()), "{results:?}");
    }

    #[test]
    fn a_moe_residual_must_carry_its_experts() {
        let m = build(false, None, 1);
        let resp = ResponseCtx { trace_root: m.trace_root, n_prompt: 1, n_pos: 1 };
        let mut n = 0;
        for s in seeds() {
            let Some(mut p) = m.open(&s) else { continue };
            let c = draw::point(&s, chain::RESID_POINT)[2] % (D as u64 * 4 / B3_CHUNK as u64);
            p.add = ResidAdd::Chunk(leaf_chunk(&m.pos, 5, c));
            assert_eq!(verify_resid_point(&m.tier, &m.ctx, &resp, &s, &p), Err(crate::verify::Fail::Shape));
            n += 1;
        }
        assert!(n > 0);
    }

    #[test]
    fn swapped_experts_are_refused() {
        let m = build(true, None, 1);
        let results: Vec<_> = seeds().filter_map(|s| m.check(&s)).collect();
        assert!(!results.is_empty() && results.iter().all(|r| *r == Err(crate::verify::Fail::Mismatch)));
    }

    #[test]
    fn a_changed_expert_output_is_refused() {
        let m = build(false, Some(1), 1);
        let mut refused = 0;
        for s in seeds() {
            let d = draw::point(&s, chain::RESID_POINT);
            if let Some(r) = m.check(&s)
                && d[2].is_multiple_of(2)
            {
                assert_eq!(r, Err(crate::verify::Fail::Mismatch));
                refused += 1;
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_trimmed_last_layer_reads_its_expert_input_as_one_row() {
        let m = build(false, None, 32);
        let n = seeds().filter_map(|s| m.check(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
        let n = seeds().filter_map(|s| m.check_piece(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
    }
}

#[cfg(test)]
mod head_tests {
    use super::*;
    use crate::verify::{PosShape, ResponseCtx, verify_point_at};

    const H: usize = 4;
    const M: usize = 16;

    #[test]
    fn per_head_product_point_verifies() {
        let w = q8_0_rows(91, H * M, 0.0004);
        let chunk_leaves: Vec<Hash> = w.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let rb = D / 32 * 34;
        let e = Entry {
            name: "blk.3.attn_k_b.weight".into(),
            wtype: quant::Q8_0,
            k: D as u64,
            m: M as u64,
            n_exp: H as u32,
            chunk_base: 0,
            n_bytes: w.len() as u64,
            row_bytes: rb as u32,
            role: manifest::ROLE_HEADS,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let eb = serialize(&e);
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&[Entry::leaf(&eb)], 0, 1).0,
            n_entries: 1,
            n_products: 1,
        };
        let xs: Vec<Vec<u8>> = (0..H).map(|h| q8_1_row(&(0..D).map(|i| ((i * 5 + h * 11) % 19) as f64 * 0.01 - 0.09).collect::<Vec<_>>())).collect();
        let mut y = Vec::new();
        for (h, xh) in xs.iter().enumerate() {
            for r in 0..M {
                let row = &w[(h * M + r) * rb..(h * M + r + 1) * rb];
                y.extend_from_slice(&(dot::canonical_dot(quant::Q8_0, row, D, xh, q8::MMVQ).unwrap().y_hat as f32).to_le_bytes());
            }
        }
        let x = xs.concat();
        let meta = [0u8, 0, 0, 0, 1, 0, 0, 0];
        let leaves = [*blake3::hash(&x).as_bytes(), *blake3::hash(&y).as_bytes(), *blake3::hash(&meta).as_bytes()];
        let resp = ResponseCtx { trace_root: b3_root(&[b3_root(&leaves)]), n_prompt: 1, n_pos: 1 };
        let shape = PosShape { n_leaves: 3, meta_index: 2, tok: None, n_used: 0 };
        let pos = Position { tok: 0, leaves: vec![x.clone(), y.clone(), meta.to_vec()], hashes: leaves.to_vec() };
        let mut seen = [false; H];
        for s in 0..48u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, 0);
            let (h, row) = ((d[3] >> 32) % H as u64, d[2] % M as u64);
            seen[h as usize] = true;
            let off = (h * M as u64 + row) * rb as u64;
            let yc = leaf_chunk(&pos, 1, (h * M as u64 + row) * 4 / B3_CHUNK as u64);
            let mut p = Point {
                n_ubatch: 1,
                pos_path: Vec::new(),
                meta_path: b3_path(&leaves, 2),
                entry: EntryOpening { index: 0, bytes: eb.clone(), path: Vec::new() },
                partner: None,
                bias: None,
                rows: vec![ChunkRange {
                    first_chunk: off / 32,
                    chunks: w[(off / 32 * 32) as usize..((off + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                    proof: rt_range_proof(&chunk_leaves, off / 32, (off + rb as u64 - 1) / 32 - off / 32 + 1).1,
                }],
                x: x.clone(),
                x_path: b3_path(&leaves, 0),
                y_total_len: y.len() as u32,
                y_chunk_index: yc.index,
                y_chunk: yc.chunk,
                y_chunk_path: yc.chunk_path,
                y_path: b3_path(&leaves, 1),
            };
            verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            let other = ((h + 1) % H as u64) * M as u64 + row;
            let oo = other * rb as u64;
            p.rows = vec![ChunkRange {
                first_chunk: oo / 32,
                chunks: w[(oo / 32 * 32) as usize..((oo + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                proof: rt_range_proof(&chunk_leaves, oo / 32, (oo + rb as u64 - 1) / 32 - oo / 32 + 1).1,
            }];
            assert!(verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).is_err());
        }
        assert!(seen.iter().all(|&b| b));
    }
}

#[cfg(test)]
mod block_task_tests {
    use super::*;
    use crate::block_task::{TASK_LEN, draw_seed, task_tokens, verify_block_task};
    use crate::opening::Opening;

    fn opening(tier: &Tier, trace: &Trace, block_hash: &Hash) -> Opening {
        let seed = draw_seed(block_hash);
        let (points, links): (Vec<_>, Vec<_>) = (0..draw::POINTS)
            .map(|i| {
                let (p, tok, link) = tier.open_point(trace, &seed, i);
                (p, (tok, link))
            })
            .unzip();
        Opening {
            response_txid: *block_hash,
            experts: vec![None; points.len()],
            points,
            prefix: Some(tier.open_prefix(trace, &seed, 0)),
            links,
            resid: Some(tier.open_resid(trace, &seed)),
            roots: None,
        }
    }

    #[test]
    fn honest_block_task_passes() {
        let tier = Tier::new();
        let window_seed = [0x5Au8; 32];
        let pc = tier.prefix_ctx(0);
        let trace = tier.trace(&task_tokens(&pc.table, &window_seed), TASK_LEN);
        for s in 0..48u32 {
            let block = *blake3::hash(&s.to_le_bytes()).as_bytes();
            verify_block_task(&tier.tier(), &tier.chain_ctx(), &pc, &trace.trace_root, &window_seed, &block, &opening(&tier, &trace, &block))
                .unwrap_or_else(|e| panic!("block {s}: {e:?}"));
        }
    }

    fn accepted_blocks(tier: &Tier, trace: &Trace, window_seed: &Hash) -> Vec<u32> {
        let pc = tier.prefix_ctx(0);
        (0..48u32)
            .filter(|s| {
                let block = *blake3::hash(&s.to_le_bytes()).as_bytes();
                verify_block_task(&tier.tier(), &tier.chain_ctx(), &pc, &trace.trace_root, window_seed, &block, &opening(tier, trace, &block)).is_ok()
            })
            .collect()
    }

    #[test]
    fn a_task_with_oversized_input_scales_is_refused() {
        let tier = Tier::new();
        let window_seed = [0x5Au8; 32];
        let trace = tier.scaled_zero_trace(&task_tokens(&tier.prefix_ctx(0).table, &window_seed), TASK_LEN, 1.0e4);
        let accepted = accepted_blocks(&tier, &trace, &window_seed);
        assert!(accepted.is_empty(), "accepted for blocks {accepted:?}");
    }

    #[test]
    fn a_task_with_undersized_input_scales_is_refused() {
        let tier = Tier::new();
        let window_seed = [0x5Au8; 32];
        let trace = tier.scaled_zero_trace(&task_tokens(&tier.prefix_ctx(0).table, &window_seed), TASK_LEN, 1.0e-3);
        let accepted = accepted_blocks(&tier, &trace, &window_seed);
        assert!(accepted.is_empty(), "accepted for blocks {accepted:?}");
    }

    #[test]
    fn a_task_declaring_another_batch_size_is_refused() {
        let tier = Tier::new();
        let window_seed = [0x5Au8; 32];
        let tokens = task_tokens(&tier.prefix_ctx(0).table, &window_seed);
        assert_eq!(accepted_blocks(&tier, &tier.trace(&tokens, TASK_LEN), &window_seed).len(), 48);
        let accepted = accepted_blocks(&tier, &tier.trace_declaring(&tokens, TASK_LEN, Some(1)), &window_seed);
        assert!(accepted.is_empty(), "accepted for blocks {accepted:?}");
    }

    #[test]
    fn a_task_of_another_window_is_refused() {
        let tier = Tier::new();
        let pc = tier.prefix_ctx(0);
        let trace = tier.trace(&task_tokens(&pc.table, &[0x11u8; 32]), TASK_LEN);
        for s in 0..48u32 {
            let block = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let r = verify_block_task(&tier.tier(), &tier.chain_ctx(), &pc, &trace.trace_root, &[0x22u8; 32], &block, &opening(&tier, &trace, &block));
            assert!(r.is_err(), "block {s}");
        }
    }
}


#[cfg(test)]
mod attn_tests {
    use super::*;
    use crate::attn::{AttnShape, Prep, ROPE_NEOX, ROPE_NORM, apply_gate, group_output};
    use crate::attn_blocks::{BlockTree, block_params, partial_len, partial_opening, pick_partials, set_test_budget};
    use crate::chain::{ATTN_POINT, AttnStep, n_pos_leaves, verify_attn_phase, verify_attn_point};
    use crate::manifest::AttnLinks;
    use crate::merkle::b3_chunks_proof;
    use crate::opening::AttnOpening;
    use crate::seq::{self, SeqLayout, SeqSpan};

    /// Sequence leaves of one attention layer: key, value and log mass leaf of each head.
    struct Seq {
        layout: SeqLayout,
        leaves: Vec<Vec<u8>>,
        hashes: Vec<Hash>,
        root: Hash,
    }

    /// Log mass leaf of each key/value head: per position, the log mass of each query head of the
    /// group over the attended positions.
    fn lse_leaves(g: usize, scale: f64, attended: impl Fn(usize) -> std::ops::RangeInclusive<usize>, queries: &[Vec<Vec<f64>>], keys: &[Vec<Vec<f32>>], values: &[Vec<Vec<f32>>]) -> Vec<Vec<u8>> {
        let n_kv = keys[0].len();
        (0..n_kv)
            .map(|kv| {
                (0..queries.len())
                    .flat_map(|t| {
                        let ks: Vec<Vec<f32>> = attended(t).map(|p| keys[p][kv].clone()).collect();
                        let vs: Vec<Vec<f32>> = attended(t).map(|p| values[p][kv].clone()).collect();
                        (0..g).flat_map(move |qi| {
                            let p = crate::attn::partial(crate::attn::Arith::Exact, &queries[t][kv * g + qi], scale, &ks, &vs).unwrap();
                            (crate::attn_blocks::lse(&p) as f32).to_le_bytes()
                        }).collect::<Vec<u8>>()
                    })
                    .collect()
            })
            .collect()
    }

    impl Seq {
        fn new(heads: usize, kcache: &[Vec<Vec<u8>>], vcache: Option<&[Vec<Vec<u8>>]>, lse: &[Vec<u8>]) -> Seq {
            let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: heads as u64 };
            let mut leaves = vec![Vec::new(); layout.n_leaves() as usize];
            for h in 0..kcache[0].len() {
                leaves[layout.kcache(0, h as u64).unwrap() as usize] = kcache.iter().flat_map(|p| p[h].clone()).collect();
                if let Some(v) = vcache {
                    leaves[layout.vcache(0, h as u64).unwrap() as usize] = v.iter().flat_map(|p| p[h].clone()).collect();
                }
            }
            for (h, b) in lse.iter().enumerate() {
                leaves[layout.lse(0, h as u64).unwrap() as usize] = b.clone();
            }
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            let root = b3_root(&hashes);
            Seq { layout, leaves, hashes, root }
        }

        /// Span of the log mass of query head `qi` of the group of `g` of head `kv` at `t`.
        fn lse_span(&self, kv: u64, g: usize, t: usize, qi: usize) -> SeqSpan {
            let lo = (t * g + qi) * 4;
            self.span(self.layout.lse(0, kv).unwrap(), lo, lo + 4)
        }

        fn span(&self, leaf: u64, lo: usize, hi: usize) -> SeqSpan {
            let data = &self.leaves[leaf as usize];
            let (c0, c1) = (lo / B3_CHUNK, (hi - 1) / B3_CHUNK);
            SeqSpan {
                first_chunk: c0 as u32,
                bytes: data[c0 * B3_CHUNK..((c1 + 1) * B3_CHUNK).min(data.len())].to_vec(),
                chunk_proof: b3_chunks_proof(data, c0 as u64, (c1 - c0 + 1) as u64),
                leaf_path: b3_path(&self.hashes, leaf as usize),
            }
        }
    }
    use crate::verify::{Fail, ResponseCtx};

    const NH: usize = 8;
    const NKV: usize = 2;
    const HD: usize = 64;
    const QD: usize = NH * HD;
    const KD: usize = NKV * HD;
    const NP: u64 = 4;
    const N_POS: usize = 6;
    const KQ: f32 = 0.125;
    const BASE: f32 = 10_000.0;
    const WINDOW: u32 = 4;

    fn vals(seed: u64, n: usize) -> Vec<f32> {
        (0..n).map(|i| blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0).collect()
    }

    fn q8_0(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for blk in v.chunks(32) {
            let amax = blk.iter().fold(0f32, |m, a| m.max(a.abs()));
            let d = f16::round(amax / 127.0);
            out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            out.extend(blk.iter().map(|&a| (if d == 0.0 { 0.0 } else { libm::roundf(a / d) }) as i8 as u8));
        }
        out
    }

    fn entry(name: &str, role: u8, k: usize, m: usize, off: usize, n: usize, chain: Chain) -> Vec<u8> {
        let e = Entry {
            name: name.into(),
            wtype: if role == manifest::ROLE_PRODUCT { quant::Q8_0 } else { 0 },
            k: k as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base: (off / 32) as u64,
            n_bytes: n as u64,
            row_bytes: n as u32,
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

    /// Attention variant of the synthetic layer.
    #[derive(Clone, Copy, Default)]
    struct Cfg {
        /// The attention output is committed as zeros.
        zero_out: bool,
        /// The cache holds keys of no committed projection, the output follows it.
        forge_cache: bool,
        /// NEOX RoPE and q/k norms.
        qk_norm: bool,
        /// Unweighted value norm.
        v_norm: bool,
        /// RoPE over the whole head with frequency divisors.
        freqs: bool,
        /// Sliding window of `WINDOW` positions.
        swa: bool,
        /// The value projection is the key projection.
        shared_kv: bool,
        /// Query heads interleaved with their output gates.
        gated: bool,
        /// A layer without attention before the attention layer.
        mixed: bool,
        /// The output is committed without its gate.
        skip_gate: bool,
    }

    const GLM: Cfg = Cfg { zero_out: false, forge_cache: false, qk_norm: false, v_norm: false, freqs: false, swa: false, shared_kv: false, gated: false, mixed: false, skip_gate: false };
    const GEMMA_SWA: Cfg = Cfg { qk_norm: true, v_norm: true, freqs: true, swa: true, ..GLM };
    const GEMMA_GLOBAL: Cfg = Cfg { qk_norm: true, v_norm: true, freqs: true, shared_kv: true, ..GLM };
    const QWEN: Cfg = Cfg { qk_norm: true, gated: true, mixed: true, ..GLM };

    fn shape(c: Cfg) -> AttnShape {
        let rope_kind = if c.qk_norm { ROPE_NEOX } else { ROPE_NORM };
        let n_rot = if c.freqs { HD } else { HD / 2 };
        AttnShape { n_head: NH, n_head_kv: NKV, head_dim: HD, rope_kind, n_rot, rope_base: BASE as f64, kq_scale: KQ as f64 }
    }

    struct At {
        cfg: Cfg,
        weights: Vec<u8>,
        chunk_leaves: Vec<Hash>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        tier: TierCtx,
        ctx: ChainCtx,
        positions: Vec<Position>,
        pos_roots: Vec<Hash>,
        pos_root: Hash,
        seq: Seq,
        /// Attended query of each head at each position.
        queries: Vec<Vec<Vec<f64>>>,
        /// Dequantized cache rows of each key/value head at each position.
        keys: Vec<Vec<Vec<f32>>>,
        values: Vec<Vec<Vec<f32>>>,
    }

    /// Builds the phase of a path from the partials of the previous phase.
    type OpenPhase<'a> = &'a dyn Fn(&[u64], Option<&[u8]>) -> AttnOpening;
    /// A path to verify, with the partials and root of the phase before it.
    type Pending = (Vec<u64>, Option<Vec<u8>>, Option<Hash>);

    /// Every phase of the attention point of `seed`, walked over every path of its block tree.
    /// Returns the deepest phase, or the first path that fails.
    fn walk(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, open: OpenPhase) -> Result<usize, (Vec<u64>, Fail)> {
        let mut stack: Vec<Pending> = vec![(Vec::new(), None, None)];
        let mut deepest = 0;
        while let Some((path, prev, root)) = stack.pop() {
            let op = open(&path, prev.as_deref());
            match verify_attn_phase(tier, ctx, resp, seed, &path, root.as_ref(), &op) {
                Ok(AttnStep::Done) => deepest = deepest.max(path.len()),
                Ok(AttnStep::Next { root, n, .. }) => stack.extend((0..n).map(|i| ([path.clone(), vec![i]].concat(), Some(op.partials.clone()), Some(root)))),
                Err(e) => return Err((path, e)),
            }
        }
        Ok(deepest)
    }

    fn window_start(c: Cfg, t: usize) -> usize {
        if c.swa { (t + 1).saturating_sub(WINDOW as usize) } else { 0 }
    }

    fn build(c: Cfg) -> At {
        let none = manifest::NONE;
        let q_norm: Vec<f32> = (0..HD).map(|i| 0.5 + 0.03 * (i % 11) as f32).collect();
        let k_norm: Vec<f32> = (0..HD).map(|i| 1.5 - 0.02 * (i % 13) as f32).collect();
        let freqs: Vec<f32> = (0..HD / 2).map(|i| if i < HD / 8 { 1.0 + 0.25 * (i % 3) as f32 } else { 1e30 }).collect();
        let mut weights = Vec::new();
        let mut offs = [0usize; 3];
        for (o, w) in offs.iter_mut().zip([&q_norm, &k_norm, &freqs]) {
            *o = weights.len();
            weights.extend(w.iter().flat_map(|a| a.to_le_bytes()));
        }
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let nl = 1 + c.mixed as usize;
        let mut next = 4 + nl as u32;
        let mut alloc = |on: bool| if on { next += 1; next - 1 } else { none };
        let (qn, kn, rf) = (alloc(c.qk_norm), alloc(c.qk_norm), alloc(c.freqs));
        let al = nl as u32 - 1;
        let attn = AttnLinks {
            q: 0, k: 1, v: if c.shared_kv { 1 } else { 2 }, q_norm: qn, k_norm: kn, n_head: NH as u32, n_head_kv: NKV as u32,
            head_dim: HD as u32, rope_kind: shape(c).rope_kind, n_rot: shape(c).n_rot as u32, rope_base: BASE.to_bits(),
            kq_scale: KQ.to_bits(), swa_window: if c.swa { WINDOW } else { 0 }, softcap: 0, rope_freqs: rf, v_norm: c.v_norm as u32,
            gated: c.gated as u32, slot: 0, mla_rank: 0, absorb: none, out: none,
        };
        let qw = 1 + c.gated as usize;
        let mut entries = vec![
            entry(&format!("blk.{al}.attn_q.weight"), manifest::ROLE_PRODUCT, 512, QD * qw, 0, 0, Chain::None),
            entry(&format!("blk.{al}.attn_k.weight"), manifest::ROLE_PRODUCT, 512, KD, 0, 0, Chain::None),
            entry(&format!("blk.{al}.attn_v.weight"), manifest::ROLE_PRODUCT, 512, KD, 0, 0, Chain::None),
            entry(&format!("blk.{al}.attn_output.weight"), manifest::ROLE_PRODUCT, QD, 512, 0, 0, Chain::None),
        ];
        if c.mixed {
            entries.push(entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, Chain::Layer(LayerLinks {
                layer: 0, proj: none, down: none, post_attn: none, post_ffn: none, scale: none, moe: None, attn: None, recur: None,
            })));
        }
        entries.push(entry(&format!("blk.{al}"), manifest::ROLE_LAYER, 0, 0, 0, 0, Chain::Layer(LayerLinks {
            layer: al, proj: 3, down: none, post_attn: none, post_ffn: none, scale: none, moe: None, attn: Some(attn), recur: None,
        })));
        if c.qk_norm {
            entries.push(entry("blk.0.attn_q_norm.weight", manifest::ROLE_NORM, HD, 1, offs[0], 4 * HD, Chain::None));
            entries.push(entry("blk.0.attn_k_norm.weight", manifest::ROLE_NORM, HD, 1, offs[1], 4 * HD, Chain::None));
        }
        if c.freqs {
            entries.push(entry("rope_freqs.weight", manifest::ROLE_NORM, HD / 2, 1, offs[2], 2 * HD, Chain::None));
        }
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: entries.len() as u64,
            n_products: NP,
        };
        let ctx = ChainCtx { n_layers: nl as u32, layer_base: 4, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 1, n_recur: 0, seq_heads: NKV as u32 };
        let prep = Prep {
            q_norm: c.qk_norm.then_some(&q_norm[..]),
            k_norm: c.qk_norm.then_some(&k_norm[..]),
            v_norm: c.v_norm,
            freqs: c.freqs.then_some(&freqs[..]),
            eps: EPS,
        };
        let n_leaves = n_pos_leaves(NP, nl as u64, 0) as usize;
        let qg_proj: Vec<Vec<f32>> = (0..N_POS).map(|t| vals(100 + t as u64, QD * qw)).collect();
        let split = |qg: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let (mut q, mut g) = (Vec::new(), Vec::new());
            for h in qg.chunks_exact(HD * qw) {
                q.extend_from_slice(&h[..HD]);
                g.extend_from_slice(&h[HD..]);
            }
            (q, g)
        };
        let k_proj: Vec<Vec<f32>> = (0..N_POS).map(|p| vals(200 + p as u64, KD)).collect();
        let v_proj: Vec<Vec<f32>> = if c.shared_kv { k_proj.clone() } else { (0..N_POS).map(|p| vals(300 + p as u64, KD)).collect() };
        let f32s = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let heads = |p: usize, f: &dyn Fn(&[f32], usize) -> Vec<f64>, src: &[f32]| -> Vec<Vec<u8>> {
            (0..NKV).map(|h| q8_0(&f32s(f(&src[h * HD..(h + 1) * HD], p)))).collect()
        };
        let kcache: Vec<Vec<Vec<u8>>> = (0..N_POS)
            .map(|p| {
                let src = if c.forge_cache { vals(900 + p as u64, KD) } else { k_proj[p].clone() };
                heads(p, &|x, p| prep.key(&shape(c), x, p as u64), &src)
            })
            .collect();
        let vcache: Vec<Vec<Vec<u8>>> = (0..N_POS).map(|p| heads(p, &|x, _| prep.value(x), &v_proj[p])).collect();
        let deq = |row: &[u8]| quant::dequant_row(quant::Q8_0, row, HD).unwrap();
        let mut positions = Vec::new();
        for t in 0..N_POS {
            let mut out = Vec::new();
            let st = window_start(c, t);
            let (q_proj, gate) = split(&qg_proj[t]);
            for kv in 0..NKV {
                let keys: Vec<Vec<f32>> = (st..=t).map(|p| deq(&kcache[p][kv])).collect();
                let values: Vec<Vec<f32>> = (st..=t).map(|p| deq(&vcache[p][kv])).collect();
                let mut g = group_output(&shape(c), &prep, &q_proj, kv, t as u64, &keys, &values).unwrap();
                if c.gated && !c.skip_gate {
                    let gs = NH / NKV * HD;
                    apply_gate(&mut g, &gate[kv * gs..(kv + 1) * gs]).unwrap();
                }
                out.extend(g);
            }
            let mut leaves = vec![Vec::new(); n_leaves];
            leaves[1] = qg_proj[t].iter().flat_map(|a| a.to_le_bytes()).collect();
            leaves[3] = k_proj[t].iter().flat_map(|a| a.to_le_bytes()).collect();
            leaves[5] = v_proj[t].iter().flat_map(|a| a.to_le_bytes()).collect();
            leaves[6] = q8_1_row(&if c.zero_out { vec![0.0; QD] } else { out });
            leaves[n_leaves - 1] = chain::meta(t as u32, N_POS as u32, 7).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok: 7, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let pos_root = b3_root(&pos_roots);
        let queries: Vec<Vec<Vec<f64>>> = (0..N_POS).map(|t| (0..NH).map(|h| prep.query(&shape(c), &split(&qg_proj[t]).0[h * HD..(h + 1) * HD], t as u64)).collect()).collect();
        let keys: Vec<Vec<Vec<f32>>> = kcache.iter().map(|p| p.iter().map(|r| deq(r)).collect()).collect();
        let values: Vec<Vec<Vec<f32>>> = vcache.iter().map(|p| p.iter().map(|r| deq(r)).collect()).collect();
        let masses = lse_leaves(NH / NKV, KQ as f64, |t| window_start(c, t)..=t, &queries, &keys, &values);
        let seq = Seq::new(NKV, &kcache, Some(&vcache), &masses);
        At { cfg: c, weights, chunk_leaves, entries, entry_leaves, tier, ctx, positions, pos_roots, pos_root, seq, queries, keys, values }
    }

    impl At {
        fn eo(&self, i: usize) -> EntryOpening {
            EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
        }

        fn norm(&self, i: usize) -> (EntryOpening, ChunkRange) {
            let e = Entry::parse(&self.entries[i]).unwrap().0;
            let (first, last) = (e.chunk_base, e.chunk_base + (e.n_bytes - 1) / 32);
            let range = ChunkRange {
                first_chunk: first,
                chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
                proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
            };
            (self.eo(i), range)
        }

        fn resp(&self) -> ResponseCtx {
            ResponseCtx { trace_root: seq::trace_root(&self.pos_root, &self.seq.root), n_prompt: N_POS as u32, n_pos: N_POS as u32 }
        }

        /// Spans of the key and value caches of head `kv` over positions `[p0, p1)`.
        fn caches(&self, kv: u64, p0: usize, p1: usize) -> (SeqSpan, SeqSpan) {
            let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: NKV as u64 };
            let bb = HD / 32 * 34;
            (self.seq.span(layout.kcache(0, kv).unwrap(), p0 * bb, p1 * bb), self.seq.span(layout.vcache(0, kv).unwrap(), p0 * bb, p1 * bb))
        }

        fn open(&self, seed: &Hash) -> AttnOpening {
            let d = draw::point(seed, ATTN_POINT);
            let t = (d[0] % N_POS as u64) as usize;
            let kv = d[2] % NKV as u64;
            let st = window_start(self.cfg, t);
            let ps = st + (d[3] % (t - st + 1) as u64) as usize;
            let nl = self.ctx.n_layers as u64;
            let pos = &self.positions[t];
            let la = 4 + nl as usize - 1;
            let qw = 1 + self.cfg.gated as u64;
            let (kcache, vcache) = self.caches(kv, st, t + 1);
            let g = NH / NKV;
            let lse = self.seq.lse_span(kv, g, t, (d[2] / NKV as u64 % g as u64) as usize);
            AttnOpening {
                response_txid: [1; 32],
                pos_root: self.pos_root,
                seq_root: self.seq.root,
                layer: self.eo(la),
                q_entry: self.eo(0),
                o_entry: self.eo(3),
                norms: (la + 1..self.entries.len()).map(|i| self.norm(i)).collect(),
                pos_path: b3_path(&self.pos_roots, t),
                q: (kv * qw..(kv + 1) * qw).map(|c| leaf_chunk(pos, 1, c)).collect(),
                out: leaf_bytes(pos, 6),
                kcache,
                vcache,
                anchor_pos_path: b3_path(&self.pos_roots, ps),
                anchor_k: vec![leaf_chunk(&self.positions[ps], 3, 0)],
                anchor_v: if self.cfg.shared_kv { Vec::new() } else { vec![leaf_chunk(&self.positions[ps], 5, 0)] },
                absorb: Vec::new(),
                lse,
                ..Default::default()
            }
        }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..64u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    impl At {
        /// Block tree of the attended positions of `t` under the current phase budget.
        fn tree(&self, t: usize) -> BlockTree {
            let (c, b) = block_params(2 * (HD / 32 * 34), partial_len(HD)).unwrap();
            BlockTree::new(window_start(self.cfg, t) as u64, t as u64 + 1, c, b).unwrap()
        }

        /// Phase `path.len()` of the attention point of `seed`, `prev` the partials of the previous phase.
        fn open_phase(&self, seed: &Hash, path: &[u64], prev: Option<&[u8]>) -> AttnOpening {
            let d = draw::point(seed, ATTN_POINT);
            let t = (d[0] % N_POS as u64) as usize;
            let kv = d[2] % NKV as u64;
            let g = (NH / NKV) as u64;
            let qh = (kv * g + d[2] / NKV as u64 % g) as usize;
            let tree = self.tree(t);
            let whole = self.open(seed);
            if tree.depth() == 0 {
                return whole;
            }
            let (parent, parent_path) = match (path.last(), prev) {
                (Some(&i), Some(b)) => partial_opening(b, HD, i as usize).unwrap(),
                _ => Default::default(),
            };
            let base = AttnOpening {
                response_txid: whole.response_txid,
                pos_root: whole.pos_root,
                seq_root: whole.seq_root,
                layer: whole.layer.clone(),
                path: path.to_vec(),
                parent,
                parent_path,
                ..Default::default()
            };
            if path.len() as u32 == tree.depth() {
                let (p0, p1) = tree.node(path).unwrap();
                let ps = (p0 + d[3] % (p1 - p0)) as usize;
                let (kcache, vcache) = self.caches(kv, p0 as usize, p1 as usize);
                return AttnOpening {
                    q_entry: whole.q_entry,
                    pos_path: whole.pos_path,
                    q: whole.q,
                    norms: whole.norms,
                    kcache,
                    vcache,
                    anchor_pos_path: b3_path(&self.pos_roots, ps),
                    anchor_k: vec![leaf_chunk(&self.positions[ps], 3, 0)],
                    anchor_v: if self.cfg.shared_kv { Vec::new() } else { vec![leaf_chunk(&self.positions[ps], 5, 0)] },
                    ..base
                };
            }
            let st = window_start(self.cfg, t);
            let keys: Vec<Vec<f32>> = (st..=t).map(|p| self.keys[p][kv as usize].clone()).collect();
            let values: Vec<Vec<f32>> = (st..=t).map(|p| self.values[p][kv as usize].clone()).collect();
            let partials = pick_partials(&self.queries[t][qh], KQ as f64, &keys, &values, st as u64, &tree, path, |_| true).unwrap();
            if path.is_empty() {
                AttnOpening { q_entry: whole.q_entry, o_entry: whole.o_entry, pos_path: whole.pos_path, q: whole.q, out: whole.out, partials, lse: whole.lse, ..base }
            } else {
                AttnOpening { partials, ..base }
            }
        }

        fn walk(&self, seed: &Hash) -> Result<usize, (Vec<u64>, Fail)> {
            walk(&self.tier, &self.ctx, &self.resp(), seed, &|path, prev| self.open_phase(seed, path, prev))
        }
    }

    /// Phase budget giving leaves of two positions and nodes of two children.
    const SMALL_BUDGET: usize = 2 * 2 * (HD / 32 * 34);

    #[test]
    fn honest_phased_points_pass_on_every_path() {
        set_test_budget(SMALL_BUDGET);
        for c in ALL {
            let a = build(c);
            let deepest = seeds().map(|s| a.walk(&s).unwrap_or_else(|e| panic!("{e:?}"))).max().unwrap();
            assert_eq!(deepest, 2);
        }
    }

    #[test]
    fn a_phased_disconnected_output_fails_at_the_root() {
        set_test_budget(SMALL_BUDGET);
        for c in ALL {
            let a = build(Cfg { zero_out: true, ..c });
            for s in seeds() {
                assert_eq!(a.walk(&s), Err((Vec::new(), Fail::Mismatch)));
            }
        }
    }

    #[test]
    fn a_phased_inconsistent_cache_fails_at_every_leaf() {
        set_test_budget(SMALL_BUDGET);
        for c in ALL {
            let a = build(Cfg { forge_cache: true, ..c });
            for s in seeds() {
                let t = (draw::point(&s, ATTN_POINT)[0] % N_POS as u64) as usize;
                let (path, e) = a.walk(&s).unwrap_err();
                assert_eq!((path.len() as u32, e), (a.tree(t).depth(), Fail::Mismatch));
            }
        }
    }

    /// A seed whose point takes three phases, with the honest phases of its first path.
    fn deep(a: &At) -> (Hash, Vec<AttnOpening>, Vec<Hash>) {
        let s = seeds().find(|s| a.tree((draw::point(s, ATTN_POINT)[0] % N_POS as u64) as usize).depth() == 2).unwrap();
        let (mut ops, mut roots, mut prev) = (Vec::new(), Vec::new(), None::<Vec<u8>>);
        for k in 0..3 {
            let op = a.open_phase(&s, &vec![0; k], prev.as_deref());
            if let Ok(AttnStep::Next { root, .. }) = verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], roots.last(), &op) {
                roots.push(root);
            }
            prev = Some(op.partials.clone());
            ops.push(op);
        }
        (s, ops, roots)
    }

    #[test]
    fn a_phase_must_carry_its_committed_parent() {
        set_test_budget(SMALL_BUDGET);
        let a = build(GLM);
        let (s, ops, roots) = deep(&a);
        let check = |k: usize, op: &AttnOpening| verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], roots.get(k.wrapping_sub(1)), op);
        assert_eq!(check(1, &ops[1]).map(|x| matches!(x, AttnStep::Next { .. })), Ok(true));
        assert_eq!(check(2, &ops[2]), Ok(AttnStep::Done));
        let mut flipped = ops[2].clone();
        flipped.parent[3] ^= 1;
        assert_eq!(check(2, &flipped), Err(Fail::Trace));
        let mut sibling = ops[2].clone();
        (sibling.parent, sibling.parent_path) = partial_opening(&ops[1].partials, HD, 1).unwrap();
        assert_eq!(check(2, &sibling), Err(Fail::Trace));
        let mut orphan = ops[1].clone();
        orphan.parent.clear();
        assert_eq!(check(1, &orphan), Err(Fail::Trace));
        assert_eq!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[0], None, &ops[1]), Err(Fail::Shape));
        assert_eq!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[], None, &ops[1]), Err(Fail::Shape));
    }

    #[test]
    fn a_phase_carries_only_its_own_fields() {
        set_test_budget(SMALL_BUDGET);
        let a = build(QWEN);
        let (s, ops, roots) = deep(&a);
        let check = |k: usize, op: &AttnOpening| verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], roots.get(k.wrapping_sub(1)), op);
        let mut root_with_cache = ops[0].clone();
        root_with_cache.kcache = ops[2].kcache.clone();
        assert_eq!(check(0, &root_with_cache), Err(Fail::Shape));
        let mut node_with_output = ops[1].clone();
        node_with_output.out = ops[0].out.clone();
        assert_eq!(check(1, &node_with_output), Err(Fail::Shape));
        let mut leaf_with_partials = ops[2].clone();
        leaf_with_partials.partials = ops[1].partials.clone();
        assert_eq!(check(2, &leaf_with_partials), Err(Fail::Shape));
        let mut short = ops[1].clone();
        short.partials.truncate(partial_len(HD));
        assert_eq!(check(1, &short), Err(Fail::Shape));
        let mut redrawn = ops[1].clone();
        redrawn.path = vec![1];
        assert_eq!(check(1, &redrawn), Err(Fail::Shape));
        let mut deeper = ops[1].clone();
        deeper.path = vec![0, 0];
        assert_eq!(check(1, &deeper), Err(Fail::Shape));
        assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &ops[0]), Err(Fail::Unsupported));
    }

    #[test]
    fn inconsistent_partials_fail_where_they_are_combined() {
        set_test_budget(SMALL_BUDGET);
        let a = build(GEMMA_GLOBAL);
        let (s, ops, roots) = deep(&a);
        let check = |k: usize, op: &AttnOpening| verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], roots.get(k.wrapping_sub(1)), op);
        let m = 2 * HD;
        for (k, op) in ops.iter().enumerate().take(2) {
            let mut forged = op.clone();
            let v = f32::from_le_bytes(forged.partials[m..m + 4].try_into().unwrap()) + 0.5;
            forged.partials[m..m + 4].copy_from_slice(&v.to_le_bytes());
            assert_eq!(check(k, &forged), Err(Fail::Mismatch), "phase {k}");
        }
    }

    /// Children of an honest root phase with child 0 replaced by one that carries the combined
    /// output under a dominating mass: the output stays, the mass moves.
    fn dominated(op: &AttnOpening, d: usize) -> AttnOpening {
        use crate::attn::Partial;
        use crate::attn_blocks::{combine, decode_partials, encode_partials};
        let mut ch = decode_partials(&op.partials, d).unwrap();
        let whole = combine(&ch).unwrap();
        ch[0] = Partial { o: whole.o, m: whole.m + 60.0, l: 1.0 };
        AttnOpening { partials: encode_partials(&ch), ..op.clone() }
    }

    #[test]
    fn the_committed_mass_binds_the_root_phase() {
        set_test_budget(SMALL_BUDGET);
        for c in ALL {
            let a = build(c);
            let (s, ops, roots) = deep(&a);
            let check = |k: usize, op: &AttnOpening| verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], roots.get(k.wrapping_sub(1)), op);
            assert!(matches!(check(0, &ops[0]), Ok(AttnStep::Next { .. })));
            assert_eq!(check(0, &dominated(&ops[0], HD)), Err(Fail::Mismatch));
            let mut no_mass = ops[0].clone();
            no_mass.lse = SeqSpan::default();
            assert_eq!(check(0, &no_mass), Err(Fail::Trace));
            let mut node_with_mass = ops[1].clone();
            node_with_mass.lse = ops[0].lse.clone();
            assert_eq!(check(1, &node_with_mass), Err(Fail::Shape));
        }
    }

    const ALL: [Cfg; 4] = [GLM, GEMMA_SWA, GEMMA_GLOBAL, QWEN];

    #[test]
    fn an_honest_attention_point_passes() {
        for c in ALL {
            let a = build(c);
            for s in seeds() {
                verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)).unwrap_or_else(|e| panic!("{e:?}"));
            }
        }
    }

    #[test]
    fn a_disconnected_attention_is_refused() {
        for c in ALL {
            let a = build(Cfg { zero_out: true, ..c });
            for s in seeds() {
                assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)), Err(Fail::Mismatch));
            }
        }
    }

    #[test]
    fn a_cache_of_no_committed_projection_is_refused() {
        for c in ALL {
            let a = build(Cfg { forge_cache: true, ..c });
            for s in seeds() {
                assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)), Err(Fail::Mismatch));
            }
        }
    }

    #[test]
    fn a_short_cache_span_is_refused() {
        for c in ALL {
            let a = build(c);
            let s = seeds().next().unwrap();
            let mut op = a.open(&s);
            op.kcache.bytes.pop();
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Trace));
            let mut op = a.open(&s);
            op.vcache.bytes.push(0);
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Trace));
        }
    }

    #[test]
    fn a_tampered_cache_is_refused() {
        for c in ALL {
            let a = build(c);
            let s = seeds().next().unwrap();
            let mut op = a.open(&s);
            op.kcache.bytes[5] ^= 1;
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Trace));
        }
    }

    #[test]
    fn honest_sliding_window_points_reach_past_the_window() {
        let a = build(GEMMA_SWA);
        let past: Vec<Hash> = seeds().filter(|s| draw::point(s, ATTN_POINT)[0] % N_POS as u64 >= WINDOW as u64).collect();
        assert!(!past.is_empty());
        for s in past {
            verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)).unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    #[test]
    fn missing_or_swapped_norms_are_refused() {
        let a = build(GEMMA_SWA);
        let s = seeds().next().unwrap();
        let mut op = a.open(&s);
        op.norms.pop();
        assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Shape));
        let mut op = a.open(&s);
        op.norms.swap(0, 1);
        assert!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op).is_err());
    }

    #[test]
    fn a_shared_value_takes_no_value_anchor() {
        let a = build(GEMMA_GLOBAL);
        let s = seeds().next().unwrap();
        let mut op = a.open(&s);
        op.anchor_v = op.anchor_k.clone();
        assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Shape));
    }

    #[test]
    fn an_ungated_output_is_refused() {
        let a = build(Cfg { skip_gate: true, ..QWEN });
        for s in seeds() {
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)), Err(Fail::Mismatch));
        }
    }

    #[test]
    fn a_layer_without_attention_is_refused() {
        let a = build(QWEN);
        let s = seeds().next().unwrap();
        let mut op = a.open(&s);
        op.layer = a.eo(4);
        assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Shape));
    }

    #[test]
    fn a_trace_root_without_its_sequence_half_is_refused() {
        let a = build(GLM);
        let s = seeds().next().unwrap();
        let mut op = a.open(&s);
        op.seq_root[0] ^= 1;
        assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Trace));
    }

    mod mla {
        use super::*;
        use crate::attn::{head_output, rms_norm};

        const NH: usize = 4;
        const NOPE: usize = 32;
        const ROPE: usize = 32;
        const QW: usize = NOPE + ROPE;
        const R: usize = 512;
        const HD: usize = R + ROPE;
        const VD: usize = 32;
        const NP: u64 = 5;
        const KQ: f32 = 0.125;

        struct Mla {
            weights: Vec<u8>,
            chunk_leaves: Vec<Hash>,
            entries: Vec<Vec<u8>>,
            entry_leaves: Vec<Hash>,
            tier: TierCtx,
            ctx: ChainCtx,
            positions: Vec<Position>,
            pos_roots: Vec<Hash>,
            pos_root: Hash,
            seq: Seq,
            /// Latent query of each head at each position.
            queries: Vec<Vec<Vec<f64>>>,
            /// Dequantized latent cache rows.
            keys: Vec<Vec<f32>>,
        }

        /// `zero_out`: the latent output is committed as zeros; `forge_cache`: the cache holds keys
        /// of no committed projection, the output follows it.
        fn build(zero_out: bool, forge_cache: bool) -> Mla {
            let none = manifest::NONE;
            let norm: Vec<f32> = (0..R).map(|i| 0.8 + 0.01 * (i % 17) as f32).collect();
            let weights: Vec<u8> = norm.iter().flat_map(|a| a.to_le_bytes()).collect();
            let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let attn = AttnLinks {
                q: 0, k: 1, v: none, q_norm: none, k_norm: 6, n_head: NH as u32, n_head_kv: 1, head_dim: HD as u32, rope_kind: 0,
                n_rot: 0, rope_base: 0, kq_scale: KQ.to_bits(), swa_window: 0, softcap: 0, rope_freqs: none, v_norm: 0, gated: 0,
                slot: 0, mla_rank: R as u32, absorb: 2, out: 3,
            };
            let heads = |name: &str, k: usize, m: usize| {
                let mut e = Entry::parse(&entry(name, manifest::ROLE_PRODUCT, k, m, 0, 0, Chain::None)).unwrap().0;
                e.role = manifest::ROLE_HEADS;
                e.n_exp = NH as u32;
                let mut b = serialize(&e);
                b.extend_from_slice(&manifest::chain_bytes(&e.chain));
                b
            };
            let entries = vec![
                entry("blk.0.attn_q.weight", manifest::ROLE_PRODUCT, 512, NH * QW, 0, 0, Chain::None),
                entry("blk.0.attn_kv_a_mqa.weight", manifest::ROLE_PRODUCT, 512, HD, 0, 0, Chain::None),
                heads("blk.0.attn_k_b.weight", NOPE, R),
                heads("blk.0.attn_v_b.weight", R, VD),
                entry("blk.0.attn_output.weight", manifest::ROLE_PRODUCT, NH * VD, 512, 0, 0, Chain::None),
                entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, Chain::Layer(LayerLinks {
                    layer: 0, proj: 4, down: none, post_attn: none, post_ffn: none, scale: none, moe: None, attn: Some(attn), recur: None,
                })),
                entry("blk.0.attn_kv_a_norm.weight", manifest::ROLE_NORM, R, 1, 0, 4 * R, Chain::None),
            ];
            let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
            let tier = TierCtx {
                r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
                n_chunks: chunk_leaves.len() as u64,
                m_t: rt_range_proof(&entry_leaves, 0, 1).0,
                n_entries: entries.len() as u64,
                n_products: NP,
            };
            let ctx = ChainCtx { n_layers: 1, layer_base: 5, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 1, n_recur: 0, seq_heads: 1 };
            let n_leaves = n_pos_leaves(NP, 1, 0) as usize;
            let q: Vec<Vec<f32>> = (0..N_POS).map(|t| vals(100 + t as u64, NH * QW)).collect();
            let kv_a: Vec<Vec<f32>> = (0..N_POS).map(|p| vals(200 + p as u64, HD)).collect();
            let absorbed: Vec<Vec<f32>> = (0..N_POS).map(|t| vals(300 + t as u64, NH * R)).collect();
            let cache: Vec<Vec<u8>> = (0..N_POS)
                .map(|p| {
                    let src = if forge_cache { vals(900 + p as u64, HD) } else { kv_a[p].clone() };
                    let mut k: Vec<f64> = src.iter().map(|&x| x as f64).collect();
                    rms_norm(&mut k[..R], Some(&norm), EPS);
                    q8_0(&k.iter().map(|&x| x as f32).collect::<Vec<_>>())
                })
                .collect();
            let keys: Vec<Vec<f32>> = cache.iter().map(|c| quant::dequant_row(quant::Q8_0, c, HD).unwrap()).collect();
            let mut positions = Vec::new();
            let mut queries = Vec::new();
            for t in 0..N_POS {
                let mut x = Vec::new();
                queries.push(Vec::new());
                for h in 0..NH {
                    let mut qh: Vec<f64> = absorbed[t][h * R..(h + 1) * R].iter().map(|&a| a as f64).collect();
                    qh.extend(q[t][h * QW + NOPE..(h + 1) * QW].iter().map(|&a| a as f64));
                    queries[t].push(qh.clone());
                    let ks = &keys[..=t];
                    let vs: Vec<Vec<f32>> = ks.iter().map(|k| k[..R].to_vec()).collect();
                    let lat = head_output(&qh, ks, &vs, KQ as f64).unwrap();
                    x.extend(q8_1_row(&if zero_out { vec![0.0; R] } else { lat }));
                }
                let mut leaves = vec![Vec::new(); n_leaves];
                leaves[1] = q[t].iter().flat_map(|a| a.to_le_bytes()).collect();
                leaves[3] = kv_a[t].iter().flat_map(|a| a.to_le_bytes()).collect();
                leaves[5] = absorbed[t].iter().flat_map(|a| a.to_le_bytes()).collect();
                leaves[6] = x;
                leaves[n_leaves - 1] = chain::meta(t as u32, N_POS as u32, 7).to_vec();
                let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                positions.push(Position { tok: 7, leaves, hashes });
            }
            let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
            let pos_root = b3_root(&pos_roots);
            let rows: Vec<Vec<Vec<u8>>> = cache.iter().map(|c| vec![c.clone()]).collect();
            let per_pos_keys: Vec<Vec<Vec<f32>>> = keys.iter().map(|k| vec![k.clone()]).collect();
            let per_pos_values: Vec<Vec<Vec<f32>>> = keys.iter().map(|k| vec![k[..R].to_vec()]).collect();
            let masses = lse_leaves(NH, KQ as f64, |t| 0..=t, &queries, &per_pos_keys, &per_pos_values);
            let seq = Seq::new(1, &rows, None, &masses);
            Mla { weights, chunk_leaves, entries, entry_leaves, tier, ctx, positions, pos_roots, pos_root, seq, queries, keys }
        }

        impl Mla {
            fn eo(&self, i: usize) -> EntryOpening {
                EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
            }

            fn resp(&self) -> ResponseCtx {
                ResponseCtx { trace_root: seq::trace_root(&self.pos_root, &self.seq.root), n_prompt: N_POS as u32, n_pos: N_POS as u32 }
            }

            fn open(&self, seed: &Hash) -> AttnOpening {
                let d = draw::point(seed, ATTN_POINT);
                let t = (d[0] % N_POS as u64) as usize;
                let qh = d[2] % NH as u64;
                let ps = (d[3] % (t as u64 + 1)) as usize;
                let pos = &self.positions[t];
                let n = self.chunk_leaves.len() as u64;
                let range = ChunkRange { first_chunk: 0, chunks: self.weights.clone(), proof: rt_range_proof(&self.chunk_leaves, 0, n).1 };
                let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: 1 };
                let bb = HD / 32 * 34;
                AttnOpening {
                    response_txid: [1; 32],
                    pos_root: self.pos_root,
                    seq_root: self.seq.root,
                    layer: self.eo(5),
                    q_entry: self.eo(0),
                    o_entry: self.eo(3),
                    norms: vec![(self.eo(6), range)],
                    pos_path: b3_path(&self.pos_roots, t),
                    q: vec![leaf_chunk(pos, 1, 0)],
                    out: leaf_bytes(pos, 6),
                    kcache: self.seq.span(layout.kcache(0, 0).unwrap(), 0, (t + 1) * bb),
                    vcache: SeqSpan::default(),
                    anchor_pos_path: b3_path(&self.pos_roots, ps),
                    anchor_k: (0..3).map(|c| leaf_chunk(&self.positions[ps], 3, c)).collect(),
                    anchor_v: Vec::new(),
                    absorb: (2 * qh..2 * qh + 2).map(|c| leaf_chunk(pos, 5, c)).collect(),
                    lse: self.seq.lse_span(0, NH, t, qh as usize),
                    ..Default::default()
                }
            }
        }

        /// Phase budget giving leaves of two positions and nodes of two children.
        const SMALL_BUDGET: usize = 2 * partial_len(R);

        impl Mla {
            fn tree(&self, t: usize) -> BlockTree {
                let (c, b) = block_params(HD / 32 * 34, partial_len(R)).unwrap();
                BlockTree::new(0, t as u64 + 1, c, b).unwrap()
            }

            /// Phase `path.len()` of the attention point of `seed`, `prev` the partials of the previous phase.
            fn open_phase(&self, seed: &Hash, path: &[u64], prev: Option<&[u8]>) -> AttnOpening {
                let d = draw::point(seed, ATTN_POINT);
                let t = (d[0] % N_POS as u64) as usize;
                let qh = (d[2] % NH as u64) as usize;
                let tree = self.tree(t);
                let whole = self.open(seed);
                if tree.depth() == 0 {
                    return whole;
                }
                let (parent, parent_path) = match (path.last(), prev) {
                    (Some(&i), Some(b)) => partial_opening(b, R, i as usize).unwrap(),
                    _ => Default::default(),
                };
                let base = AttnOpening {
                    response_txid: whole.response_txid,
                    pos_root: whole.pos_root,
                    seq_root: whole.seq_root,
                    layer: whole.layer.clone(),
                    path: path.to_vec(),
                    parent,
                    parent_path,
                    ..Default::default()
                };
                if path.len() as u32 == tree.depth() {
                    let (p0, p1) = tree.node(path).unwrap();
                    let ps = (p0 + d[3] % (p1 - p0)) as usize;
                    let bb = HD / 32 * 34;
                    let layout = SeqLayout { n_attn: 1, n_recur: 0, heads: 1 };
                    return AttnOpening {
                        q_entry: whole.q_entry,
                        pos_path: whole.pos_path,
                        q: whole.q,
                        norms: whole.norms,
                        kcache: self.seq.span(layout.kcache(0, 0).unwrap(), p0 as usize * bb, p1 as usize * bb),
                        anchor_pos_path: b3_path(&self.pos_roots, ps),
                        anchor_k: (0..3).map(|c| leaf_chunk(&self.positions[ps], 3, c)).collect(),
                        absorb: whole.absorb,
                        ..base
                    };
                }
                let keys = &self.keys[..=t];
                let values: Vec<Vec<f32>> = keys.iter().map(|k| k[..R].to_vec()).collect();
                let partials = pick_partials(&self.queries[t][qh], KQ as f64, keys, &values, 0, &tree, path, |_| true).unwrap();
                if path.is_empty() {
                    AttnOpening { q_entry: whole.q_entry, o_entry: whole.o_entry, pos_path: whole.pos_path, q: whole.q, out: whole.out, partials, lse: whole.lse, ..base }
                } else {
                    AttnOpening { partials, ..base }
                }
            }

            fn walk(&self, seed: &Hash) -> Result<usize, (Vec<u64>, Fail)> {
                walk(&self.tier, &self.ctx, &self.resp(), seed, &|path, prev| self.open_phase(seed, path, prev))
            }
        }

        #[test]
        fn honest_phased_latent_points_pass_on_every_path() {
            set_test_budget(SMALL_BUDGET);
            let a = build(false, false);
            assert_eq!(seeds().map(|s| a.walk(&s).unwrap_or_else(|e| panic!("{e:?}"))).max(), Some(2));
        }

        #[test]
        fn a_phased_disconnected_latent_output_fails_at_the_root() {
            set_test_budget(SMALL_BUDGET);
            let a = build(true, false);
            for s in seeds() {
                assert_eq!(a.walk(&s), Err((Vec::new(), Fail::Mismatch)));
            }
        }

        #[test]
        fn a_phased_inconsistent_latent_cache_fails_at_every_leaf() {
            set_test_budget(SMALL_BUDGET);
            let a = build(false, true);
            for s in seeds() {
                let t = (draw::point(&s, ATTN_POINT)[0] % N_POS as u64) as usize;
                let (path, e) = a.walk(&s).unwrap_err();
                assert_eq!((path.len() as u32, e), (a.tree(t).depth(), Fail::Mismatch));
            }
        }

        #[test]
        fn a_phased_latent_leaf_takes_the_drawn_head_query() {
            set_test_budget(SMALL_BUDGET);
            let a = build(false, false);
            let s = seeds().find(|s| a.tree((draw::point(s, ATTN_POINT)[0] % N_POS as u64) as usize).depth() == 2).unwrap();
            let d = draw::point(&s, ATTN_POINT);
            let (t, other) = ((d[0] % N_POS as u64) as usize, (d[2] + 1) % NH as u64);
            let (mut prev, mut root) = (None::<Vec<u8>>, None);
            for k in 0..2 {
                let op = a.open_phase(&s, &vec![0; k], prev.as_deref());
                match verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &vec![0; k], root.as_ref(), &op) {
                    Ok(AttnStep::Next { root: r, .. }) => root = Some(r),
                    x => panic!("{x:?}"),
                }
                prev = Some(op.partials);
            }
            let mut leaf = a.open_phase(&s, &[0, 0], prev.as_deref());
            assert_eq!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[0, 0], root.as_ref(), &leaf), Ok(AttnStep::Done));
            leaf.absorb = (2 * other..2 * other + 2).map(|c| leaf_chunk(&a.positions[t], 5, c)).collect();
            assert!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[0, 0], root.as_ref(), &leaf).is_err());
        }

        #[test]
        fn the_committed_mass_binds_the_latent_root_phase() {
            set_test_budget(SMALL_BUDGET);
            let a = build(false, false);
            let s = seeds().find(|s| a.tree((draw::point(s, ATTN_POINT)[0] % N_POS as u64) as usize).depth() >= 1).unwrap();
            let root = a.open_phase(&s, &[], None);
            assert!(matches!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[], None, &root), Ok(AttnStep::Next { .. })));
            assert_eq!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[], None, &super::dominated(&root, R)), Err(Fail::Mismatch));
            let mut no_mass = root.clone();
            no_mass.lse = SeqSpan::default();
            assert_eq!(verify_attn_phase(&a.tier, &a.ctx, &a.resp(), &s, &[], None, &no_mass), Err(Fail::Trace));
        }

        #[test]
        fn an_honest_latent_attention_point_passes() {
            let a = build(false, false);
            for s in seeds() {
                verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)).unwrap_or_else(|e| panic!("{e:?}"));
            }
        }

        #[test]
        fn a_disconnected_latent_output_is_refused() {
            let a = build(true, false);
            for s in seeds() {
                assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)), Err(Fail::Mismatch));
            }
        }

        #[test]
        fn a_latent_cache_of_no_committed_projection_is_refused() {
            let a = build(false, true);
            for s in seeds() {
                assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &a.open(&s)), Err(Fail::Mismatch));
            }
        }

        #[test]
        fn a_tampered_latent_cache_is_refused() {
            let a = build(false, false);
            let s = seeds().next().unwrap();
            let mut op = a.open(&s);
            op.kcache.bytes[7] ^= 1;
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Trace));
        }

        #[test]
        fn a_latent_layer_takes_no_value_cache() {
            let a = build(false, false);
            let s = seeds().next().unwrap();
            let mut op = a.open(&s);
            op.vcache = op.kcache.clone();
            assert_eq!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op), Err(Fail::Shape));
        }

        #[test]
        fn another_head_query_is_refused() {
            let a = build(false, false);
            let s = seeds().next().unwrap();
            let qh = draw::point(&s, ATTN_POINT)[2] % NH as u64;
            let other = (qh + 1) % NH as u64;
            let mut op = a.open(&s);
            let t = (draw::point(&s, ATTN_POINT)[0] % N_POS as u64) as usize;
            op.absorb = (2 * other..2 * other + 2).map(|c| leaf_chunk(&a.positions[t], 5, c)).collect();
            assert!(verify_attn_point(&a.tier, &a.ctx, &a.resp(), &s, &op).is_err());
        }
    }
}

#[cfg(test)]
mod recur_tests {
    use super::*;
    use crate::chain::ChainCtx;
    use crate::manifest::{RECUR_GDA, RECUR_KDA, RecurLinks};
    use crate::merkle::b3_chunks_proof;
    use crate::opening::{RecurOpening, RecurTap};
    use crate::recur::{RECUR_POINT, RECUR_ROWS, verify_recur_point};
    use crate::seq::{self, Record, SeqLayout, SeqSpan};
    use crate::verify::{Fail, ResponseCtx};

    const HD: usize = 32;
    const NV: usize = 8;
    const IV: usize = 4;
    const CL: usize = 4;
    const N_POS: usize = 10;

    fn vals(seed: u64, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
    }

    fn sig(x: f64) -> f64 {
        1.0 / (1.0 + libm::exp(-x))
    }

    fn l2(x: &mut [f64]) {
        let r = 1.0 / libm::sqrt(x.iter().map(|a| a * a).sum::<f64>().max(EPS * EPS));
        x.iter_mut().for_each(|a| *a *= r);
    }

    #[derive(Clone, Copy, Default)]
    struct Cfg {
        kda: bool,
        /// The state restarts from zero at every position.
        reset: bool,
        /// Keys of no committed projection, the recurrence following them.
        fake_k: bool,
        zero_out: bool,
        /// Checkpoints shifted away from the recurrence.
        bad_ckpt: bool,
        /// The decay product carries the decay bias.
        fused_dt: bool,
        /// One huge output column per head, the other columns set from it, the output committed from them.
        spike_o: bool,
        /// A huge write strength at the first position, the recurrence following it.
        big_beta: bool,
        /// A growing decay at the first position, the recurrence following it.
        growing_decay: bool,
        /// A huge value at the first position, the recurrence following it.
        huge_v: bool,
    }

    struct Rt {
        cfg: Cfg,
        weights: Vec<u8>,
        chunk_leaves: Vec<Hash>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        offs: Vec<usize>,
        tier: TierCtx,
        ctx: ChainCtx,
        links: RecurLinks,
        out_rank: usize,
        positions: Vec<Position>,
        pos_roots: Vec<Hash>,
        pos_root: Hash,
        seq_leaves: Vec<Vec<u8>>,
        seq_hashes: Vec<Hash>,
        seq_root: Hash,
    }

    fn entry(name: &str, role: u8, k: usize, m: usize, off: usize, n: usize) -> Vec<u8> {
        let e = Entry {
            name: name.into(),
            wtype: if role == manifest::ROLE_PRODUCT { quant::Q8_0 } else { 0 },
            k: k as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base: (off / 32) as u64,
            n_bytes: n as u64,
            row_bytes: n as u32,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let mut b = serialize(&e);
        b.extend_from_slice(&manifest::chain_bytes(&e.chain));
        b
    }

    fn build(cfg: Cfg) -> Rt {
        let none = manifest::NONE;
        let kda = cfg.kda;
        let nk = if kda { NV } else { 2 };
        let qkv_m = (2 * nk + NV) * HD;
        let gw = if kda { HD } else { 1 };
        // Products: query/key/value sources, beta, decay, gate, output projection.
        let names: Vec<(&str, usize)> = if kda {
            vec![("attn_q", NV * HD), ("attn_k", NV * HD), ("attn_v", NV * HD), ("ssm_beta", NV), ("ssm_f_b", NV * HD), ("ssm_g_b", NV * HD), ("attn_output", 512)]
        } else {
            vec![("attn_qkv", qkv_m), ("ssm_beta", NV), ("ssm_alpha", NV), ("attn_gate", NV * HD), ("ssm_out", 512)]
        };
        let np = names.len();
        let (src, beta_r, decay_r, gate_r, out_r) = if kda { ([0, 1, 2], 3, 4, 5, 6) } else { ([0, 0, 0], 1, 2, 3, 4) };
        let conv_n = if kda { NV * HD } else { qkv_m };
        let mut wv: Vec<(String, Vec<f32>)> = Vec::new();
        let n_conv = if kda { 3 } else { 1 };
        for c in 0..n_conv {
            wv.push((format!("blk.0.ssm_conv1d_{c}.weight"), vals(40 + c as u64, CL * conv_n, 0.6)));
        }
        wv.push(("blk.0.ssm_dt.bias".into(), vals(50, NV * gw, 0.5)));
        wv.push(("blk.0.ssm_a".into(), (0..NV).map(|i| -(0.2 + 0.1 * i as f32)).collect()));
        wv.push(("blk.0.ssm_norm.weight".into(), (0..HD).map(|i| 1.0 + 0.01 * i as f32).collect()));
        let mut weights = Vec::new();
        let mut offs = Vec::new();
        for (_, w) in &wv {
            offs.push(weights.len());
            weights.extend(w.iter().flat_map(|a| a.to_le_bytes()));
        }
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let wbase = np + 1;
        let conv_idx = |c: usize| (wbase + if kda { c } else { 0 }) as u32;
        let links = RecurLinks {
            kind: if kda { RECUR_KDA } else { RECUR_GDA },
            slot: 0,
            n_v_heads: NV as u32,
            n_k_heads: nk as u32,
            head_dim: HD as u32,
            q: src[0],
            k: src[1],
            v: src[2],
            conv_q: conv_idx(0),
            conv_k: conv_idx(1),
            conv_v: conv_idx(2),
            beta: beta_r,
            decay: decay_r,
            dt: (wbase + n_conv) as u32,
            a: (wbase + n_conv + 1) as u32,
            norm: (wbase + n_conv + 2) as u32,
            gate: gate_r,
            interval: IV as u32,
            conv_len: CL as u32,
        };
        let mut entries: Vec<Vec<u8>> = names
            .iter()
            .map(|&(n, m)| entry(&format!("blk.0.{n}.weight"), manifest::ROLE_PRODUCT, if n.ends_with("output") || n == "ssm_out" { NV * HD } else { 512 }, m, 0, 0))
            .collect();
        let mut layer = Entry::parse(&entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0)).unwrap().0;
        layer.chain = Chain::Layer(LayerLinks {
            layer: 0, proj: out_r, down: none, post_attn: none, post_ffn: none, scale: none, moe: None, attn: None, recur: Some(links),
        });
        let mut lb = serialize(&layer);
        lb.extend_from_slice(&manifest::chain_bytes(&layer.chain));
        entries.push(lb);
        for (i, (n, w)) in wv.iter().enumerate() {
            entries.push(entry(n, manifest::ROLE_NORM, w.len(), 1, offs[i], 4 * w.len()));
        }
        if cfg.fused_dt {
            let mut e = Entry::parse(&entries[decay_r as usize]).unwrap().0;
            e.bias = links.dt;
            let mut b = serialize(&e);
            b.extend_from_slice(&manifest::chain_bytes(&e.chain));
            entries[decay_r as usize] = b;
        }
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: entries.len() as u64,
            n_products: np as u64,
        };
        let ctx = ChainCtx {
            n_layers: 1, layer_base: np as u32, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0,
            n_recur: 1, seq_heads: NV as u32,
        };

        // Committed projection outputs per position.
        let ys: Vec<Vec<Vec<f32>>> = (0..N_POS)
            .map(|t| names.iter().enumerate().map(|(r, &(_, m))| vals(1000 * (r as u64 + 1) + t as u64, m, 1.0)).collect())
            .collect();
        let conv_w = |c: usize| &wv[if kda { c } else { 0 }].1;
        let (dt, a, norm) = (&wv[n_conv].1, &wv[n_conv + 1].1, &wv[n_conv + 2].1);
        let conv = |c: usize, t: usize, ch: usize| -> f64 {
            let x = (0..CL).map(|j| {
                let tp = t as i64 - (CL - 1) as i64 + j as i64;
                if tp < 0 { 0.0 } else { ys[tp as usize][src[c] as usize][ch] as f64 }
            });
            let s: f64 = x.enumerate().map(|(j, v)| conv_w(c)[ch * CL + j] as f64 * v).sum();
            s * sig(s)
        };
        let mut state = vec![vec![vec![0f64; HD]; HD]; NV];
        let mut records: Vec<Vec<u8>> = vec![Vec::new(); NV];
        let mut ckpts: Vec<Vec<u8>> = vec![Vec::new(); NV];
        let mut outs: Vec<Vec<f64>> = Vec::new();
        for (t, y_t) in ys.iter().enumerate() {
            let mut y_all = Vec::new();
            for h in 0..NV {
                let hk = h % nk;
                let (qo, ko, vo) = if kda { (h * HD, h * HD, h * HD) } else { (hk * HD, (nk + hk) * HD, 2 * nk * HD + h * HD) };
                let mut q: Vec<f64> = (0..HD).map(|i| conv(0, t, qo + i)).collect();
                let mut k: Vec<f64> = if cfg.fake_k { vals(7000 + t as u64 * 31 + h as u64, HD, 1.0).iter().map(|&x| x as f64).collect() } else { (0..HD).map(|i| conv(1, t, ko + i)).collect() };
                let mut v: Vec<f64> = (0..HD).map(|i| conv(2, t, vo + i)).collect();
                if cfg.huge_v && t == 0 {
                    v[0] = 2.0 * crate::recur::RECUR_V_MAX;
                }
                l2(&mut q);
                l2(&mut k);
                let beta = if cfg.big_beta && t == 0 { 4.0 } else { sig(y_t[beta_r as usize][h] as f64) };
                let mut g: Vec<f64> = (0..gw).map(|i| a[h] as f64 * {
                    let x = y_t[decay_r as usize][h * gw + i] as f64 + if cfg.fused_dt { 0.0 } else { dt[h * gw + i] as f64 };
                    if x > 20.0 { x } else { libm::log1p(libm::exp(x)) }
                }).collect();
                if cfg.growing_decay && t == 0 {
                    g.iter_mut().for_each(|x| *x = 1.0 / 128.0);
                }
                let eg = |i: usize| libm::exp(if kda { g[i] } else { g[0] });
                let s = &mut state[h];
                if cfg.reset {
                    s.iter_mut().for_each(|r| r.iter_mut().for_each(|x| *x = 0.0));
                }
                let mut o = vec![0f64; HD];
                for col in 0..HD {
                    let kv: f64 = (0..HD).map(|i| eg(i) * s[col][i] * k[i]).sum();
                    let d = (v[col] - kv) * beta;
                    for i in 0..HD {
                        s[col][i] = eg(i) * s[col][i] + k[i] * d;
                    }
                    o[col] = (0..HD).map(|i| s[col][i] * q[i]).sum::<f64>() / (HD as f64).sqrt();
                }
                if cfg.spike_o {
                    let honest = libm::sqrt(o.iter().map(|x| x * x).sum::<f64>() / HD as f64);
                    o.iter_mut().for_each(|x| *x += 10.0 * honest);
                    o[0] = 60_000.0;
                }
                let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
                let rec = Record { q: f(&q), k: f(&k), v: f(&v), o: f(&o), beta: beta as f32, g: f(&g) };
                records[h].extend(rec.encode(links.kind));
                if (t + 1) % IV == 0 {
                    for row in s.iter() {
                        ckpts[h].extend(row.iter().flat_map(|&x| ((x + if cfg.bad_ckpt { 0.5 } else { 0.0 }) as f32).to_le_bytes()));
                    }
                }
                let ms = o.iter().map(|x| x * x).sum::<f64>() / HD as f64;
                let rn = 1.0 / libm::sqrt(ms + EPS);
                for i in 0..HD {
                    let z = ys[t][gate_r as usize][h * HD + i] as f64;
                    let act = if kda { sig(z) } else { z * sig(z) };
                    y_all.push(o[i] * rn * norm[i] as f64 * act);
                }
            }
            y_all.resize(512, 0.0);
            outs.push(if cfg.zero_out { vec![0.0; 512] } else { y_all });
        }
        let n_leaves = 2 * np + 2 + 1;
        let mut positions = Vec::new();
        for t in 0..N_POS {
            let mut leaves = vec![Vec::new(); n_leaves];
            for r in 0..np {
                leaves[2 * r + 1] = ys[t][r].iter().flat_map(|a| a.to_le_bytes()).collect();
            }
            leaves[2 * out_r as usize] = q8_1_row(&outs[t]);
            leaves[n_leaves - 1] = chain::meta(t as u32, 1, 7).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok: 7, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let pos_root = b3_root(&pos_roots);
        let layout = SeqLayout { n_attn: 0, n_recur: 1, heads: NV as u64 };
        let mut seq_leaves = vec![Vec::new(); layout.n_leaves() as usize];
        for h in 0..NV {
            seq_leaves[layout.record(0, h as u64).unwrap() as usize] = records[h].clone();
            seq_leaves[layout.checkpoint(0, h as u64).unwrap() as usize] = ckpts[h].clone();
        }
        let seq_hashes: Vec<Hash> = seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let seq_root = b3_root(&seq_hashes);
        Rt { cfg, weights, chunk_leaves, entries, entry_leaves, offs, tier, ctx, links, out_rank: out_r as usize, positions, pos_roots, pos_root, seq_leaves, seq_hashes, seq_root }
    }

    impl Rt {
        fn eo(&self, i: usize) -> EntryOpening {
            EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
        }

        fn wrange(&self, entry: usize, off: usize, n: usize) -> (EntryOpening, ChunkRange) {
            let wi = entry - self.ctx.layer_base as usize - 1;
            let (lo, hi) = (self.offs[wi] + 4 * off, self.offs[wi] + 4 * (off + n));
            let (first, last) = (lo / 32, (hi - 1) / 32);
            let range = ChunkRange {
                first_chunk: first as u64,
                chunks: self.weights[first * 32..(last + 1) * 32].to_vec(),
                proof: rt_range_proof(&self.chunk_leaves, first as u64, (last - first + 1) as u64).1,
            };
            (self.eo(entry), range)
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

        fn resp(&self) -> ResponseCtx {
            ResponseCtx { trace_root: seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: 1, n_pos: N_POS as u32 }
        }

        fn open(&self, seed: &Hash) -> RecurOpening {
            let l = self.links;
            let (kda, nk, hd) = (self.cfg.kda, l.n_k_heads as usize, HD);
            let d = draw::point(seed, RECUR_POINT);
            let h = (d[2] % NV as u64) as usize;
            let hk = h % nk;
            let s = (d[0] % N_POS.div_ceil(IV) as u64) as usize;
            let (c, e) = (s * IV, ((s + 1) * IV).min(N_POS));
            let r0 = d[3] % (hd as u64 / RECUR_ROWS) * RECUR_ROWS;
            let p = c + ((d[3] >> 32) % (e - c) as u64) as usize;
            let layout = SeqLayout { n_attn: 0, n_recur: 1, heads: NV as u64 };
            let rl = seq::record_len(l.kind, hd);
            let rows = |m: usize| {
                let (lo, hi) = seq::checkpoint_rows(hd as u64, m as u64, r0, RECUR_ROWS);
                self.span(layout.checkpoint(0, h as u64).unwrap(), lo as usize, hi as usize)
            };
            let offs = if kda { [h * hd, h * hd, h * hd] } else { [hk * hd, (nk + hk) * hd, 2 * nk * hd + h * hd] };
            let n_taps = (p + 1).min(CL);
            let chunk_of = |off: usize| (off * 4 / B3_CHUNK) as u64;
            let srcs = [l.q, l.k, l.v];
            let taps = (p + 1 - n_taps..=p)
                .map(|t| {
                    let pos = &self.positions[t];
                    let lc = |x: usize| leaf_chunk(pos, 2 * srcs[x] as u64 + 1, chunk_of(offs[x]));
                    RecurTap { pos_path: b3_path(&self.pos_roots, t), q: lc(0), k: lc(1), v: lc(2) }
                })
                .collect();
            let gw = if kda { hd } else { 1 };
            let pos = &self.positions[p];
            let conv_e = [l.conv_q, l.conv_k, l.conv_v];
            let mut weights: Vec<(EntryOpening, ChunkRange)> = (0..3).map(|x| self.wrange(conv_e[x] as usize, offs[x] * CL, hd * CL)).collect();
            weights.push(self.wrange(l.dt as usize, h * gw, gw));
            weights.push(self.wrange(l.a as usize, h, 1));
            weights.push(self.wrange(l.norm as usize, 0, hd));
            RecurOpening {
                response_txid: [1; 32],
                pos_root: self.pos_root,
                seq_root: self.seq_root,
                layer: self.eo(self.ctx.layer_base as usize),
                out_entry: self.eo(self.out_rank),
                decay_entry: self.eo(l.decay as usize),
                weights,
                records: self.span(layout.record(0, h as u64).unwrap(), c * rl, e * rl),
                start_rows: if s == 0 { SeqSpan::default() } else { rows(s) },
                end_rows: if e - c == IV { rows(s + 1) } else { SeqSpan::default() },
                taps,
                beta: leaf_chunk(pos, 2 * l.beta as u64 + 1, chunk_of(h)),
                decay: leaf_chunk(pos, 2 * l.decay as u64 + 1, chunk_of(h * gw)),
                gate: leaf_chunk(pos, 2 * l.gate as u64 + 1, chunk_of(h * hd)),
                out: leaf_bytes(pos, 2 * self.out_rank as u64),
            }
        }

        fn check(&self, seed: &Hash) -> Result<(), Fail> {
            verify_recur_point(&self.tier, &self.ctx, &self.resp(), seed, &self.open(seed))
        }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..64u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    const GDA: Cfg = Cfg { kda: false, reset: false, fake_k: false, zero_out: false, bad_ckpt: false, fused_dt: false, spike_o: false, big_beta: false, growing_decay: false, huge_v: false };
    const KINDS: [Cfg; 3] = [GDA, Cfg { kda: true, ..GDA }, Cfg { kda: true, fused_dt: true, ..GDA }];

    #[test]
    fn an_honest_recurrent_point_passes() {
        for c in KINDS {
            let rt = build(c);
            let mut segs = [false; 3];
            for s in seeds() {
                segs[(draw::point(&s, RECUR_POINT)[0] % 3) as usize] = true;
                rt.check(&s).unwrap_or_else(|e| panic!("kda {} fused {}: {e:?}", c.kda, c.fused_dt));
            }
            assert!(segs.iter().all(|&b| b));
        }
    }

    #[test]
    fn a_trace_built_token_by_token_is_refused() {
        for c in KINDS {
            let rt = build(Cfg { reset: true, ..c });
            for s in seeds() {
                assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
            }
        }
    }

    #[test]
    fn keys_of_no_committed_projection_are_refused() {
        for c in KINDS {
            let rt = build(Cfg { fake_k: true, ..c });
            for s in seeds() {
                assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
            }
        }
    }

    #[test]
    fn a_disconnected_output_is_refused() {
        for c in KINDS {
            let rt = build(Cfg { zero_out: true, ..c });
            for s in seeds() {
                assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
            }
        }
    }

    #[test]
    fn shifted_checkpoints_are_refused() {
        for c in KINDS {
            let rt = build(Cfg { bad_ckpt: true, ..c });
            for s in seeds() {
                assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
            }
        }
    }

    #[test]
    fn a_huge_output_column_is_refused() {
        for c in KINDS {
            let rt = build(Cfg { spike_o: true, ..c });
            for s in seeds() {
                assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {} fused {}", c.kda, c.fused_dt);
            }
        }
    }

    #[test]
    fn a_write_strength_beyond_the_sigmoid_is_refused() {
        for c in KINDS {
            let rt = build(Cfg { big_beta: true, ..c });
            let mut first = 0;
            for s in seeds() {
                let segment = draw::point(&s, crate::recur::RECUR_POINT)[0] % (N_POS as u64).div_ceil(IV as u64);
                if segment == 0 {
                    first += 1;
                    assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
                }
            }
            assert!(first > 0);
        }
    }

    #[test]
    fn a_growing_decay_or_a_huge_value_is_refused() {
        for c in KINDS {
            for cfg in [Cfg { growing_decay: true, ..c }, Cfg { huge_v: true, ..c }] {
                let rt = build(cfg);
                let mut first = 0;
                for s in seeds() {
                    if draw::point(&s, crate::recur::RECUR_POINT)[0] % (N_POS as u64).div_ceil(IV as u64) == 0 {
                        first += 1;
                        assert_eq!(rt.check(&s), Err(Fail::Mismatch), "kda {}", c.kda);
                    }
                }
                assert!(first > 0);
            }
        }
    }

    #[test]
    fn the_decay_bias_follows_the_decay_product_entry() {
        for c in KINDS {
            let rt = build(c);
            let s = seeds().next().unwrap();
            let mut op = rt.open(&s);
            op.decay_entry = rt.eo(rt.links.gate as usize);
            assert!(verify_recur_point(&rt.tier, &rt.ctx, &rt.resp(), &s, &op).is_err());
            let mut op = rt.open(&s);
            op.decay_entry = rt.eo(rt.links.dt as usize);
            assert!(verify_recur_point(&rt.tier, &rt.ctx, &rt.resp(), &s, &op).is_err());
        }
        let honest = build(Cfg { kda: true, ..GDA });
        let fused = build(Cfg { kda: true, fused_dt: true, ..GDA });
        let s = seeds().next().unwrap();
        let mut op = fused.open(&s);
        op.decay_entry = honest.eo(honest.links.decay as usize);
        assert!(verify_recur_point(&fused.tier, &fused.ctx, &fused.resp(), &s, &op).is_err());
    }

    #[test]
    fn tampered_records_and_short_windows_are_refused() {
        for c in KINDS {
            let rt = build(c);
            let s = seeds().next().unwrap();
            let mut op = rt.open(&s);
            op.records.bytes[3] ^= 1;
            assert_eq!(verify_recur_point(&rt.tier, &rt.ctx, &rt.resp(), &s, &op), Err(Fail::Trace));
            let s = seeds().find(|s| {
                let d = draw::point(s, RECUR_POINT);
                let seg = (d[0] % 3) as usize;
                let (c0, e) = (seg * IV, ((seg + 1) * IV).min(N_POS));
                c0 + ((d[3] >> 32) % (e - c0) as u64) as usize >= CL
            }).unwrap();
            let mut op = rt.open(&s);
            op.taps.remove(0);
            assert_eq!(verify_recur_point(&rt.tier, &rt.ctx, &rt.resp(), &s, &op), Err(Fail::Shape));
            let mut op = rt.open(&s);
            op.seq_root[0] ^= 1;
            assert_eq!(verify_recur_point(&rt.tier, &rt.ctx, &rt.resp(), &s, &op), Err(Fail::Trace));
        }
    }
}
