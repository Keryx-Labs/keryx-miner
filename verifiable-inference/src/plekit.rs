//! Two-layer synthetic tier with per-layer inputs, and its stage-2 traces, for tests.

use crate::chain::{self, ChainCtx, PosLeaves, Resid};
use crate::chainkit::{Position, Trace, leaf_bytes, leaf_chunk, n_chunks};
use crate::dot;
use crate::draw;
use crate::f16;
use crate::manifest::{self, Chain, Entry, LayerLinks, PleLayer, ProductLink};
use crate::merkle::{Hash, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, FeedRow, LinkOpening, PlePoint, Point, ResidAdd, ResidOpening, ResidPoint, ResidSrc};
use crate::q8;
use crate::quant;
use crate::testkit::{in_layout, layout_at, q8_1_row, serialize};
use crate::verify::{ResponseCtx, TierCtx};

const D: usize = 512;
const W: usize = 256;
const N_LAYERS: usize = 2;
const VOCAB: usize = 64;
const EPS: f64 = 1e-6;
const SCALE: f32 = 0.75;
const EMBD_SCALE: f32 = 2.0;

const PER_LAYER: usize = 4;
const ATTN_OUTPUT: usize = 0;
const FFN_DOWN: usize = 1;
const INP_GATE: usize = 2;
const PROJ: usize = 3;
const N_PRODUCTS: usize = N_LAYERS * PER_LAYER;
const EMBD: usize = N_PRODUCTS;
const POST_NORM: usize = EMBD + 1;
const PLE_NORM: usize = POST_NORM + N_LAYERS;
const PLE_EMBD: usize = PLE_NORM + 1;
const PLE_PROJ: usize = PLE_EMBD + 1;
const LAYERS: usize = PLE_PROJ + 1;

fn rank(l: usize, p: usize) -> usize {
    l * PER_LAYER + p
}

/// Q8_0 rows of `k` values.
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

/// A deliberate fault in a forged trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The gate input is not the residual before the per-layer input.
    GateInput,
    /// The projection input is built from other per-layer inputs than the committed ones.
    PleInput,
    /// The residual before the per-layer input does not add the FFN output.
    PeIn,
    /// The layer output skips the layer scale.
    OutScale,
    /// The committed per-layer inputs are off their projection, norm and table.
    Inputs,
    /// The committed projection is shifted off its weights (its inputs follow it).
    Proj,
    /// The projection reads another token's embedding.
    ProjToken,
}

fn proj_scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

fn ple_scales() -> (f32, f32) {
    ((W as f32).sqrt(), 1.0 / 2f32.sqrt())
}

pub struct PleTier {
    weights: Vec<u8>,
    chunk_leaves: Vec<Hash>,
    entries: Vec<(Entry, Vec<u8>)>,
    entry_leaves: Vec<Hash>,
    r_t: Hash,
    m_t: Hash,
}

impl PleTier {
    pub fn new() -> Self {
        let mut weights = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();
        let base = |name: String, role: u8, wtype: u32, k: usize, m: usize, off: usize, n_bytes: usize, rb: usize| Entry {
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
            chain: Chain::None,
        };
        let mut push = |bytes: Vec<u8>| {
            let off = weights.len();
            weights.extend_from_slice(&bytes);
            (off, bytes.len())
        };
        let link = |layer: usize, link: u8, src: u32| Chain::Product(ProductLink { layer: layer as u32, link, norm: manifest::NONE, src });
        for l in 0..N_LAYERS {
            for p in 0..PER_LAYER {
                let (name, k, m) = match p {
                    ATTN_OUTPUT => ("attn_output", D, D),
                    FFN_DOWN => ("ffn_down", D, D),
                    INP_GATE => ("inp_gate", D, W),
                    _ => ("proj", W, D),
                };
                let (off, n) = push(q8_0_rows(100 * (l as u64 + 1) + p as u64, m, k, 0.0004));
                let mut e = base(format!("blk.{l}.{name}.weight"), manifest::ROLE_PRODUCT, quant::Q8_0, k, m, off, n, k / 32 * 34);
                e.chain = match p {
                    INP_GATE => link(l, manifest::LINK_RESID_IN, manifest::NONE),
                    PROJ => link(l, manifest::LINK_PLE, rank(l, INP_GATE) as u32),
                    _ => link(l, manifest::LINK_NONE, manifest::NONE),
                };
                entries.push(e);
            }
        }
        let (off, n) = push(q8_0_rows(3, VOCAB, D, 0.02));
        entries.push(base("token_embd.weight".into(), manifest::ROLE_EMBD, quant::Q8_0, D, VOCAB, off, n, D / 32 * 34));
        for l in 0..N_LAYERS {
            let w: Vec<u8> = (0..D).flat_map(|j| (1.0 + 0.002 * ((j + 5 * l) % 13) as f32).to_le_bytes()).collect();
            let (off, n) = push(w);
            entries.push(base(format!("blk.{l}.post_norm.weight"), manifest::ROLE_NORM, 0, D, 1, off, n, 4 * D));
        }
        let w: Vec<u8> = (0..W).flat_map(|j| (0.9 + 0.003 * (j % 29) as f32).to_le_bytes()).collect();
        let (off, n) = push(w);
        entries.push(base("per_layer_proj_norm.weight".into(), manifest::ROLE_NORM, 0, W, 1, off, n, 4 * W));
        let (off, n) = push(q8_0_rows(7, VOCAB, W * N_LAYERS, 0.01));
        entries.push(base("per_layer_token_embd.weight".into(), manifest::ROLE_PLE_EMBD, quant::Q8_0, W * N_LAYERS, VOCAB, off, n, W * N_LAYERS / 32 * 34));
        let bf16: Vec<u8> = (0..W * N_LAYERS * D)
            .flat_map(|i| {
                let h = blake3::hash(&(i as u64).to_le_bytes());
                let v = (h.as_bytes()[0] as f32 - 127.5) * 0.0004;
                ((quant::round_bf16(v).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let (off, n) = push(bf16);
        let mut pe = base("per_layer_model_proj.weight".into(), manifest::ROLE_PLE_PROJ, quant::BF16, D, W * N_LAYERS, off, n, 2 * D);
        let (es, os) = ple_scales();
        pe.chain = Chain::Ple(manifest::PleLinks {
            embd: PLE_EMBD as u32,
            norm: PLE_NORM as u32,
            width: W as u32,
            embd_scale: es.to_bits(),
            proj_scale: proj_scale().to_bits(),
            out_scale: os.to_bits(),
        });
        entries.push(pe);
        for l in 0..N_LAYERS {
            let mut e = base(format!("blk.{l}"), manifest::ROLE_LAYER, 0, 0, 0, 0, 0, 0);
            e.chain = Chain::Layer(LayerLinks {
                layer: l as u32,
                proj: rank(l, ATTN_OUTPUT) as u32,
                down: rank(l, FFN_DOWN) as u32,
                post_attn: manifest::NONE,
                post_ffn: manifest::NONE,
                scale: SCALE.to_bits(),
                moe: None,
                attn: None,
                recur: None,
                ple: Some(PleLayer { gate: rank(l, INP_GATE) as u32, proj: rank(l, PROJ) as u32, post_norm: (POST_NORM + l) as u32 }),
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
        TierCtx { r_t: self.r_t, n_chunks: self.chunk_leaves.len() as u64, m_t: self.m_t, n_entries: self.entries.len() as u64, n_products: N_PRODUCTS as u64 }
    }

    pub fn chain_ctx(&self) -> ChainCtx {
        ChainCtx {
            n_layers: N_LAYERS as u32,
            layer_base: LAYERS as u32,
            embd_entry: EMBD as u32,
            eps: EPS,
            embd_scale: EMBD_SCALE as f64,
            trim_last: false,
            n_used: 0,
            n_attn: 0,
            n_recur: 0,
            seq_heads: 0,
            ple_entry: PLE_PROJ as u32,
            ple_width: W as u32,
        }
    }

    fn leaves(&self) -> PosLeaves {
        PosLeaves::of(&self.tier(), &self.chain_ctx())
    }

    fn entry(&self, i: usize) -> &Entry {
        &self.entries[i].0
    }

    fn row(&self, i: usize, r: usize) -> &[u8] {
        let e = self.entry(i);
        let off = e.chunk_base as usize * 32 + r * e.row_bytes as usize;
        &self.weights[off..off + e.row_bytes as usize]
    }

    fn matvec(&self, i: usize, x: &[u8]) -> Vec<f32> {
        let e = self.entry(i);
        (0..e.m as usize).map(|r| dot::canonical_dot(quant::Q8_0, self.row(i, r), e.k as usize, x, q8::MMVQ).unwrap().y_hat as f32).collect()
    }

    fn f32_tensor(&self, i: usize) -> Vec<f32> {
        let e = self.entry(i);
        let off = e.chunk_base as usize * 32;
        self.weights[off..off + e.n_bytes as usize].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// The per-layer projection before its norm and the per-layer inputs of every layer, for
    /// `tok` at a position whose projection input is rounded to bf16 when `rounded`.
    fn per_layer(&self, tok: u32, rounded: bool, fault: Option<Fault>) -> (Vec<f32>, Vec<f32>) {
        let from = if fault == Some(Fault::ProjToken) { (tok + 1) % VOCAB as u32 } else { tok };
        let x: Vec<f32> = quant::dequant_row(quant::Q8_0, self.row(EMBD, from as usize), D)
            .unwrap()
            .iter()
            .map(|&v| v * EMBD_SCALE)
            .map(|v| if rounded { quant::round_bf16(v) } else { v })
            .collect();
        let mut proj: Vec<f32> = (0..W * N_LAYERS)
            .map(|j| {
                let w = quant::dequant_row(quant::BF16, self.row(PLE_PROJ, j), D).unwrap();
                let y: f64 = w.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
                y as f32 * proj_scale()
            })
            .collect();
        if fault == Some(Fault::Proj) {
            let shift = 0.1 * proj.iter().map(|v| v.abs()).sum::<f32>() / proj.len() as f32;
            proj.iter_mut().for_each(|v| *v += shift);
        }
        let norm = self.f32_tensor(PLE_NORM);
        let table = quant::dequant_row(quant::Q8_0, self.row(PLE_EMBD, tok as usize), W * N_LAYERS).unwrap();
        let (es, os) = ple_scales();
        let mut inputs = Vec::with_capacity(W * N_LAYERS);
        for l in 0..N_LAYERS {
            let normed = chain::rms_norm(&proj[l * W..(l + 1) * W], &norm, EPS);
            for i in 0..W {
                let v = ((normed[i] as f32 + table[l * W + i] * es) * os) * if fault == Some(Fault::Inputs) { 1.01 } else { 1.0 };
                inputs.push(v);
            }
        }
        (proj, inputs)
    }

    pub fn trace(&self, tokens: &[u32], n_prompt: u32, fault: Option<Fault>) -> Trace {
        let lv = self.leaves();
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|a| a.to_le_bytes()).collect() };
        let f64s = |v: &[f32]| -> Vec<f64> { v.iter().map(|&a| a as f64).collect() };
        let ones = vec![1.0f32; D];
        let gelu = |g: f32| crate::verify::glu(manifest::GLU_GEGLU, g as f64).unwrap().0;
        let mut positions = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let layout = layout_at(t as u64, n_prompt).1;
            let mut leaves = vec![Vec::new(); lv.n() as usize];
            let mut h: Vec<f32> = quant::dequant_row(quant::Q8_0, self.row(EMBD, tok as usize), D).unwrap().iter().map(|&v| v * EMBD_SCALE).collect();
            let (proj, all_inputs) = self.per_layer(tok, (t as u32) < n_prompt, fault);
            for l in 0..N_LAYERS {
                let x_ao = q8_1_row(&chain::rms_norm(&h, &ones, EPS));
                let y_ao = self.matvec(rank(l, ATTN_OUTPUT), &x_ao);
                let attn: Vec<f32> = h.iter().zip(&y_ao).map(|(a, b)| a + b).collect();

                let x_d = q8_1_row(&chain::rms_norm(&attn, &ones, EPS));
                let y_d = self.matvec(rank(l, FFN_DOWN), &x_d);
                let k = if fault == Some(Fault::PeIn) { 2.0 } else { 1.0 };
                let pe_in: Vec<f32> = attn.iter().zip(&y_d).map(|(a, b)| a + k * b).collect();

                let gate_src: Vec<f64> = f64s(&pe_in).iter().map(|v| if fault == Some(Fault::GateInput) { v * 1.5 } else { *v }).collect();
                let x_g = q8_1_row(&gate_src);
                let y_g = self.matvec(rank(l, INP_GATE), &x_g);
                let inp = &all_inputs[l * W..(l + 1) * W];
                let used: Vec<f32> = inp.iter().map(|&v| if fault == Some(Fault::PleInput) { v * 1.3 } else { v }).collect();
                let mut prod: Vec<f64> = y_g.iter().zip(&used).map(|(&g, &u)| (gelu(g) as f32 * u) as f64).collect();
                prod.resize(2 * W, 0.0);
                let x_p = q8_1_row(&prod);
                let y_p = self.matvec(rank(l, PROJ), &x_p);
                let normed = chain::rms_norm(&y_p, &self.f32_tensor(POST_NORM + l), EPS);
                let s = if fault == Some(Fault::OutScale) { 1.0 } else { SCALE as f64 };
                let out: Vec<f32> = pe_in.iter().zip(&normed).map(|(&a, &b)| (s * (a as f64 + b)) as f32).collect();

                for (p, x, y) in [(ATTN_OUTPUT, &x_ao, &y_ao), (FFN_DOWN, &x_d, &y_d), (INP_GATE, &x_g, &y_g), (PROJ, &x_p, &y_p)] {
                    leaves[2 * rank(l, p)] = in_layout(x, layout);
                    leaves[2 * rank(l, p) + 1] = bytes(y);
                }
                leaves[lv.resid(l as u64, Resid::Attn) as usize] = bytes(&attn);
                leaves[lv.resid(l as u64, Resid::PeIn) as usize] = bytes(&pe_in);
                leaves[lv.resid(l as u64, Resid::Out) as usize] = bytes(&out);
                h = out;
            }
            leaves[lv.ple_proj() as usize] = bytes(&proj);
            leaves[lv.ple_inputs() as usize] = bytes(&all_inputs);
            leaves[lv.meta() as usize] = chain::meta(t as u32, layout_at(t as u64, n_prompt).0, tok).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let trace_root = b3_root(&pos_roots);
        Trace { n_prompt, positions, pos_roots, trace_root, ubatch: None }
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

    fn row_of(&self, i: usize, r: u64) -> (EntryOpening, ChunkRange) {
        let rb = self.entry(i).row_bytes as u64;
        (self.entry_opening(i), self.bytes_range(i, r * rb, rb))
    }

    /// Point `i` drawn by `seed`, with its token and input link.
    pub fn open_point(&self, trace: &Trace, seed: &Hash, i: u32) -> (Point, u32, LinkOpening) {
        let lv = self.leaves();
        let d = draw::point(seed, i);
        let drawn = (d[1] % N_PRODUCTS as u64) as usize;
        let t = d[0] % trace.positions.len() as u64;
        let e = self.entry(drawn);
        let row = d[2] % e.m;
        let pos = &trace.positions[t as usize];
        let y = &pos.leaves[2 * drawn + 1];
        let yc = row * 4 / crate::merkle::B3_CHUNK as u64;
        let ych = leaf_chunk(pos, 2 * drawn as u64 + 1, yc);
        let point = Point {
            n_ubatch: trace.layout(t).0,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, lv.meta() as usize),
            entry: self.entry_opening(drawn),
            partner: None,
            bias: None,
            rows: vec![self.row_of(drawn, row).1],
            x: pos.leaves[2 * drawn].clone(),
            x_path: b3_path(&pos.hashes, 2 * drawn),
            y_total_len: y.len() as u32,
            y_chunk_index: yc as u32,
            y_chunk: ych.chunk,
            y_chunk_path: ych.chunk_path,
            y_path: b3_path(&pos.hashes, 2 * drawn + 1),
        };
        let Chain::Product(pl) = e.chain else { panic!("product without link") };
        let layer = pl.layer as u64;
        let layout = crate::verify::input_layout(e.role, point.n_ubatch, e.wtype);
        let c = chain::link_block(seed, i, &point.x, layout, e.k).expect("link block") / chain::CHUNK_BLOCKS;
        let link = match pl.link {
            manifest::LINK_RESID_IN => LinkOpening::Resid(leaf_chunk(pos, lv.resid(layer, Resid::PeIn), c)),
            manifest::LINK_PLE => {
                let src = pl.src as u64;
                let per_layer = (4 * W / crate::merkle::B3_CHUNK) as u64;
                LinkOpening::Ple {
                    src: self.entry_opening(src as usize),
                    y: leaf_chunk(pos, 2 * src + 1, c),
                    inputs: leaf_chunk(pos, lv.ple_inputs(), layer * per_layer + c),
                }
            }
            _ => LinkOpening::None,
        };
        (point, pos.tok, link)
    }

    fn resid_draw(&self, trace: &Trace, seed: &Hash) -> (u64, Resid, u64) {
        let lv = self.leaves();
        let d = draw::point(seed, chain::RESID_POINT);
        let (layer, kind) = lv.resid_at(chain::resid_site(d[1], lv.per_layer() * N_LAYERS as u64));
        let resp = ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 };
        let t = chain::resid_position(&self.chain_ctx(), &resp, layer, d[0]).expect("residual position");
        (layer, kind, t)
    }

    /// Residual point drawn by `seed`.
    pub fn open_resid(&self, trace: &Trace, seed: &Hash) -> ResidPoint {
        let lv = self.leaves();
        let d = draw::point(seed, chain::RESID_POINT);
        let (layer, kind, t) = self.resid_draw(trace, seed);
        let pos = &trace.positions[t as usize];
        let ll = match self.entry(LAYERS + layer as usize).chain {
            Chain::Layer(ll) => ll,
            _ => unreachable!(),
        };
        let rule = chain::resid_rule(&ll, kind, true).expect("rule");
        let product = rule.rank as usize;
        let dst_leaf = lv.resid(layer, kind);
        let c = d[2] % n_chunks(pos.leaves[dst_leaf as usize].len());
        let src = match rule.src {
            None if layer == 0 => {
                let (e, r) = self.row_of(EMBD, pos.tok as u64);
                ResidSrc::Token(e, r)
            }
            None => ResidSrc::Chunk(leaf_chunk(pos, lv.resid(layer - 1, Resid::Out), c)),
            Some(from) => ResidSrc::Chunk(leaf_chunk(pos, lv.resid(layer, from), c)),
        };
        let add = if rule.post == manifest::NONE {
            ResidAdd::Chunk(leaf_chunk(pos, 2 * product as u64 + 1, c))
        } else {
            let per = (crate::merkle::B3_CHUNK / 4) as u64;
            let n = per.min(D as u64 - c * per);
            ResidAdd::Normed { y: leaf_bytes(pos, 2 * product as u64 + 1), norm: (self.entry_opening(rule.post as usize), self.bytes_range(rule.post as usize, 4 * c * per, 4 * n)) }
        };
        ResidPoint {
            n_ubatch: trace.layout(t).0,
            tok: pos.tok,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, lv.meta() as usize),
            layer: self.entry_opening(LAYERS + layer as usize),
            product: self.entry_opening(product),
            dst: leaf_chunk(pos, dst_leaf, c),
            src,
            add,
        }
    }

    /// Residual opening drawn by `seed`: the residual point and one row of its feeding product.
    pub fn open_resid_piece(&self, trace: &Trace, seed: &Hash) -> ResidOpening {
        let resid = self.open_resid(trace, seed);
        let d = draw::point(seed, chain::RESID_POINT);
        let product = resid.product.index as usize;
        let per = (crate::merkle::B3_CHUNK / 4) as u64;
        let c = resid.dst.index as u64;
        let row = c * per + (d[3] >> 32) % per.min(self.entry(product).m - c * per);
        let (_, _, t) = self.resid_draw(trace, seed);
        let pos = &trace.positions[t as usize];
        let feed = FeedRow { x: pos.leaves[2 * product].clone(), x_path: b3_path(&pos.hashes, 2 * product), row: self.row_of(product, row).1, bias: None };
        let q = draw::point(seed, chain::PLE_POINT);
        let (layer, jj) = (q[1] % N_LAYERS as u64, q[2] % W as u64);
        let lv = self.leaves();
        let slice = (W / 32 * 34) as u64;
        let table_off = pos.tok as u64 * self.entry(PLE_EMBD).row_bytes as u64 + layer * slice;
        let ple = PlePoint {
            proj_entry: self.entry_opening(PLE_PROJ),
            proj_row: self.row_of(PLE_PROJ, layer * W as u64 + jj).1,
            embd: self.row_of(EMBD, pos.tok as u64),
            table: (self.entry_opening(PLE_EMBD), self.bytes_range(PLE_EMBD, table_off, slice)),
            norm: (self.entry_opening(PLE_NORM), self.bytes_range(PLE_NORM, 0, 4 * W as u64)),
            proj: leaf_chunk(pos, lv.ple_proj(), layer),
            inputs: leaf_chunk(pos, lv.ple_inputs(), layer),
        };
        ResidOpening { response_txid: [0; 32], roots: None, resid, feed, expert: None, ple: Some(ple) }
    }
}

impl Default for PleTier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{verify_chain_point, verify_resid_opening, verify_resid_point};
    use crate::verify::Fail;

    fn tokens() -> Vec<u32> {
        (0..12u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect()
    }

    fn resp(trace: &Trace) -> ResponseCtx {
        ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..96u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    #[test]
    fn honest_ple_trace_passes_every_draw() {
        let tier = PleTier::new();
        let trace = tier.trace(&tokens(), 8, None);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let (mut links, mut sites) = ([0usize; 3], [0usize; 3]);
        for seed in seeds() {
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                links[match link {
                    LinkOpening::Resid(_) => 0,
                    LinkOpening::Ple { .. } => 1,
                    _ => 2,
                }] += 1;
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).unwrap_or_else(|e| panic!("point {i}: {e:?}"));
            }
            let (_, kind, _) = tier.resid_draw(&trace, &seed);
            sites[kind as usize] += 1;
            verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed)).unwrap_or_else(|e| panic!("resid {kind:?}: {e:?}"));
            verify_resid_opening(&tc, &cc, &rc, &seed, &tier.open_resid_piece(&trace, &seed)).unwrap_or_else(|e| panic!("piece {kind:?}: {e:?}"));
        }
        assert!(links.iter().all(|&n| n > 0), "{links:?}");
        assert!(sites.iter().all(|&n| n > 0), "{sites:?}");
    }

    #[test]
    fn a_forged_per_layer_input_link_is_refused() {
        let tier = PleTier::new();
        let (tc, cc) = (tier.tier(), tier.chain_ctx());
        for (fault, product) in [(Fault::GateInput, INP_GATE), (Fault::PleInput, PROJ)] {
            let trace = tier.trace(&tokens(), 8, Some(fault));
            let rc = resp(&trace);
            let mut hit = 0;
            for seed in seeds() {
                for i in 0..draw::POINTS {
                    let (p, tok, link) = tier.open_point(&trace, &seed, i);
                    let r = verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None);
                    if p.entry.index as usize % PER_LAYER == product {
                        assert_eq!(r.map(|_| ()), Err(Fail::Mismatch), "{fault:?}");
                        hit += 1;
                    }
                }
            }
            assert!(hit > 0, "{fault:?}");
        }
    }

    #[test]
    fn a_forged_per_layer_residual_is_refused() {
        let tier = PleTier::new();
        let (tc, cc) = (tier.tier(), tier.chain_ctx());
        for (fault, site) in [(Fault::PeIn, Resid::PeIn), (Fault::OutScale, Resid::Out)] {
            let trace = tier.trace(&tokens(), 8, Some(fault));
            let rc = resp(&trace);
            let mut hit = 0;
            for seed in seeds() {
                let (_, kind, _) = tier.resid_draw(&trace, &seed);
                let r = verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed));
                if kind == site {
                    assert_eq!(r, Err(Fail::Mismatch), "{fault:?}");
                    hit += 1;
                } else {
                    assert_eq!(r, Ok(()), "{fault:?} on {kind:?}");
                }
            }
            assert!(hit > 0, "{fault:?}");
        }
    }

    #[test]
    fn a_forged_per_layer_projection_or_input_is_refused() {
        let tier = PleTier::new();
        let (tc, cc) = (tier.tier(), tier.chain_ctx());
        for fault in [Fault::Inputs, Fault::Proj, Fault::ProjToken] {
            let trace = tier.trace(&tokens(), 8, Some(fault));
            let rc = resp(&trace);
            for seed in seeds() {
                assert_eq!(verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed)), Ok(()), "{fault:?}");
                assert_eq!(verify_resid_opening(&tc, &cc, &rc, &seed, &tier.open_resid_piece(&trace, &seed)), Err(Fail::Mismatch), "{fault:?}");
            }
        }
    }

    #[test]
    fn the_per_layer_point_is_required_exactly_with_per_layer_inputs() {
        let tier = PleTier::new();
        let trace = tier.trace(&tokens(), 8, None);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        for seed in seeds().take(16) {
            let mut piece = tier.open_resid_piece(&trace, &seed);
            let b = piece.encode();
            assert_eq!(ResidOpening::decode(&b).as_ref(), Some(&piece));
            piece.ple = None;
            assert_eq!(verify_resid_opening(&tc, &cc, &rc, &seed, &piece), Err(Fail::Shape));
            assert!(b.len() > piece.encode().len());
        }
    }

    #[test]
    fn prompt_positions_round_the_projection_input_to_bf16() {
        let tier = PleTier::new();
        let tok = tokens()[0];
        let (f32_proj, _) = tier.per_layer(tok, false, None);
        let (bf16_proj, _) = tier.per_layer(tok, true, None);
        assert!(f32_proj.iter().zip(&bf16_proj).any(|(a, b)| a != b));
    }

    #[test]
    fn a_layer_without_its_per_layer_input_is_refused() {
        let tier = PleTier::new();
        let trace = tier.trace(&tokens(), 8, None);
        let mut cc = tier.chain_ctx();
        cc.ple_width = 0;
        let rc = resp(&trace);
        for seed in seeds().take(16) {
            assert!(verify_resid_point(&tier.tier(), &cc, &rc, &seed, &tier.open_resid(&trace, &seed)).is_err());
        }
    }
}
