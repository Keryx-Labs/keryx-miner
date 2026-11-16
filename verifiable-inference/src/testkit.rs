//! Synthetic tier and responses for tests.

use crate::dot;
use crate::draw;
use crate::f16;
use crate::manifest::{self, Entry};
use crate::merkle::{Hash, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, Opening, Point, PrefixPoint};
use crate::prefix::{self, PrefixCtx};
use crate::q8;
use crate::quant;
use crate::verify::{TierCtx, expected_ubatch, layout_for};

const K: usize = 512;
const M: usize = 4;

/// Vocabulary of the synthetic model: token id = byte value.
pub const VOCAB: usize = 256;

/// One Q8_0 product of `M` rows of `K` weights, a Q8_0 embedding table of `VOCAB` rows and an
/// F32 layer-0 norm, laid out one after the other in the weight file.
pub struct Model {
    weights: Vec<u8>,
    chunk_leaves: Vec<Hash>,
    entries: Vec<(Entry, Vec<u8>)>,
    entry_leaves: Vec<Hash>,
    r_t: Hash,
    m_t: Hash,
}

pub(crate) fn serialize(entry: &Entry) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
    b.extend_from_slice(entry.name.as_bytes());
    b.extend_from_slice(&entry.wtype.to_le_bytes());
    b.extend_from_slice(&entry.k.to_le_bytes());
    b.extend_from_slice(&entry.m.to_le_bytes());
    b.extend_from_slice(&entry.n_exp.to_le_bytes());
    b.extend_from_slice(&entry.chunk_base.to_le_bytes());
    b.extend_from_slice(&entry.n_bytes.to_le_bytes());
    b.extend_from_slice(&entry.row_bytes.to_le_bytes());
    b.extend_from_slice(&[entry.role, entry.glu_op, entry.glu_form]);
    b.extend_from_slice(&entry.partner.to_le_bytes());
    b.extend_from_slice(&entry.bias.to_le_bytes());
    b
}

fn q8_0_rows(seed: u64, rows: usize, scale: f32) -> Vec<u8> {
    let (bb, bv) = quant::block_size(quant::Q8_0).unwrap();
    let mut out = Vec::with_capacity(rows * K / bv * bb);
    for r in 0..rows {
        for b in 0..K / bv {
            out.extend_from_slice(&f16::from_f32(scale + 0.001 * (r * 7 + b) as f32).to_le_bytes());
            for j in 0..bv {
                out.push(byte(seed, (r * K / bv + b) * bv + j));
            }
        }
    }
    out
}

/// Q8_1 row (`mmvq` layout) of `v`, rounded the way the kernel does.
/// `leaf`, a row of 32-value MMVQ blocks, in the byte layout of `layout`.
pub(crate) fn in_layout(leaf: &[u8], layout: u8) -> Vec<u8> {
    if layout == q8::MMVQ {
        return leaf.to_vec();
    }
    let mut out = Vec::with_capacity(leaf.len());
    for g in leaf.chunks(4 * 36) {
        for b in g.chunks(36) {
            if layout == q8::MMQ_D4 {
                out.extend_from_slice(&f16::read(b, 0).to_le_bytes());
            } else {
                out.extend_from_slice(&b[..4]);
            }
        }
        for b in g.chunks(36) {
            out.extend_from_slice(&b[4..36]);
        }
    }
    out
}

/// Committed batch size and input layout of a Q8_0 product at position `t`.
pub(crate) fn layout_at(t: u64, n_prompt: u32) -> (u32, u8) {
    let ub = expected_ubatch(t, n_prompt);
    (ub, layout_for(ub, quant::Q8_0))
}

pub(crate) fn q8_1_row(v: &[f64]) -> Vec<u8> {
    let mut leaf = Vec::with_capacity(v.len() / 32 * 36);
    for blk in v.chunks(32) {
        let amax = blk.iter().fold(0f64, |m, a| m.max(a.abs()));
        let d = f16::round((amax / 127.0) as f32);
        let qs: Vec<i8> = blk.iter().map(|a| if d == 0.0 { 0 } else { libm::round(a / d as f64) as i8 }).collect();
        let s: f32 = qs.iter().map(|&q| q as f32 * d).sum();
        leaf.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        leaf.extend_from_slice(&f16::from_f32(s).to_le_bytes());
        leaf.extend(qs.iter().map(|&q| q as u8));
    }
    leaf
}

/// A traced response over the synthetic model.
pub struct Response {
    pub trace_root: Hash,
    pub n_prompt: u32,
    pub n_pos: u32,
    x: Vec<Vec<u8>>,
    y: Vec<Vec<u8>>,
    pos_leaves: Vec<Vec<Hash>>,
    pos_roots: Vec<Hash>,
}

fn byte(seed: u64, i: usize) -> u8 {
    blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0]
}

impl Model {
    pub fn new() -> Self {
        let (bb, bv) = quant::block_size(quant::Q8_0).unwrap();
        let row_bytes = (K / bv * bb) as u32;
        let product = q8_0_rows(1, M, 0.01);
        let embd = q8_0_rows(3, VOCAB, 0.02);
        let norm: Vec<u8> = (0..K).flat_map(|i| (1.0 + 0.001 * (i % 17) as f32).to_le_bytes()).collect();
        let mut weights = product.clone();
        weights.extend_from_slice(&embd);
        weights.extend_from_slice(&norm);
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let base = |name: &str, role: u8, wtype: u32, m: usize, chunk_base: u64, n_bytes: usize, rb: u32| Entry {
            name: name.into(),
            wtype,
            k: K as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base,
            n_bytes: n_bytes as u64,
            row_bytes: rb,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: manifest::Chain::None,
        };
        let entries: Vec<Entry> = vec![
            base("blk.0.ffn_down.weight", manifest::ROLE_PRODUCT, quant::Q8_0, M, 0, product.len(), row_bytes),
            base("token_embd.weight", manifest::ROLE_EMBD, quant::Q8_0, VOCAB, (product.len() / 32) as u64, embd.len(), row_bytes),
            base(
                "blk.0.attn_norm.weight",
                manifest::ROLE_NORM,
                0,
                1,
                ((product.len() + embd.len()) / 32) as u64,
                norm.len(),
                4 * K as u32,
            ),
        ];
        let entries: Vec<(Entry, Vec<u8>)> = entries
            .into_iter()
            .map(|e| {
                let b = serialize(&e);
                (e, b)
            })
            .collect();
        let entry_leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        let m_t = rt_range_proof(&entry_leaves, 0, 1).0;
        Self { weights, chunk_leaves, entries, entry_leaves, r_t, m_t }
    }

    fn entry(&self) -> &Entry {
        &self.entries[0].0
    }

    fn entry_opening(&self, index: usize) -> EntryOpening {
        EntryOpening {
            index: index as u32,
            bytes: self.entries[index].1.clone(),
            path: rt_range_proof(&self.entry_leaves, index as u64, 1).1,
        }
    }

    /// The bytes of chunks `first..=last`, proven against `R_T`.
    fn range(&self, first: u64, last: u64) -> ChunkRange {
        ChunkRange {
            first_chunk: first,
            chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
            proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
        }
    }

    /// Prefix context: identity table, two manifest entries after the product.
    pub fn prefix_ctx(&self, offset: u32) -> PrefixCtx {
        PrefixCtx { table: std::array::from_fn(|b| b as u32), offset, embd_entry: 1, norm_entry: 2, eps: 1e-6, embd_scale: 1.0 }
    }

    /// Layer-0 input of token `tok`, as the kernel commits it.
    fn prefix_leaf(&self, tok: u32) -> Vec<u8> {
        let (e, rb) = (&self.entries[1].0, self.entries[1].0.row_bytes as usize);
        let row_off = e.chunk_base as usize * 32 + tok as usize * rb;
        let row = quant::dequant_row(quant::Q8_0, &self.weights[row_off..row_off + rb], K).unwrap();
        let n = &self.entries[2].0;
        let w: Vec<f32> = self.weights[n.chunk_base as usize * 32..n.chunk_base as usize * 32 + 4 * K]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        q8_1_row(&prefix::expected_input(&row, &w, 1e-6, 1.0))
    }

    pub fn tier(&self) -> TierCtx {
        TierCtx { r_t: self.r_t, n_chunks: self.chunk_leaves.len() as u64, m_t: self.m_t, n_entries: 3, n_products: 1 }
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.entry().row_bytes as usize;
        &self.weights[r * rb..(r + 1) * rb]
    }

    /// A response of `n_pos` positions; `honest == false` shifts every output.
    pub fn response(&self, n_prompt: u32, n_pos: u32, honest: bool) -> Response {
        self.response_with_prefix(n_prompt, n_pos, honest, None)
    }

    /// [`Self::response`] whose prefix positions hold the layer-0 inputs of the tokens
    /// `(request id, offset)` puts there.
    pub fn response_with_prefix(&self, n_prompt: u32, n_pos: u32, honest: bool, prefix: Option<(&Hash, u32)>) -> Response {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut pos_leaves = Vec::new();
        for t in 0..n_pos as usize {
            let prefix_tok = prefix.and_then(|(txid, off)| {
                let i = (t as u64).checked_sub(off as u64)?;
                (i < prefix::PREFIX_LEN as u64).then(|| txid[i as usize] as u32)
            });
            let leaf = if let Some(tok) = prefix_tok {
                self.prefix_leaf(tok)
            } else {
                let mut leaf = Vec::with_capacity(K / 32 * 36);
                for b in 0..K / 32 {
                    let qs: Vec<u8> = (0..32).map(|j| byte(2 + t as u64, b * 32 + j)).collect();
                    let d = 0.02f32;
                    let s: f32 = qs.iter().map(|&q| (q as i8) as f32 * d).sum();
                    leaf.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                    leaf.extend_from_slice(&f16::from_f32(s).to_le_bytes());
                    leaf.extend_from_slice(&qs);
                }
                leaf
            };
            let mut out = Vec::with_capacity(M * 4);
            for r in 0..M {
                let rc = dot::canonical_dot(quant::Q8_0, self.row(r), K, &leaf, q8::MMVQ).unwrap();
                let v = rc.y_hat as f32 + if honest { 0.0 } else { 1.0 + rc.scale as f32 };
                out.extend_from_slice(&v.to_le_bytes());
            }
            let (ub, layout) = layout_at(t as u64, n_prompt);
            let leaf = in_layout(&leaf, layout);
            let mut meta = [0u8; 8];
            meta[..4].copy_from_slice(&(t as u32).to_le_bytes());
            meta[4..].copy_from_slice(&ub.to_le_bytes());
            pos_leaves.push(vec![*blake3::hash(&leaf).as_bytes(), *blake3::hash(&out).as_bytes(), *blake3::hash(&meta).as_bytes()]);
            x.push(leaf);
            y.push(out);
        }
        let pos_roots: Vec<Hash> = pos_leaves.iter().map(|l| b3_root(l)).collect();
        Response { trace_root: b3_root(&pos_roots), n_prompt, n_pos, x, y, pos_leaves, pos_roots }
    }

    /// The opening of `resp` under `seed`.
    pub fn open(&self, resp: &Response, txid: Hash, seed: &Hash) -> Opening {
        let points = (0..draw::POINTS)
            .map(|i| {
                let d = draw::point(seed, i);
                let t = (d[0] % resp.n_pos as u64) as usize;
                let row = d[2] % M as u64;
                let rb = self.entry().row_bytes as u64;
                Point {
                    n_ubatch: expected_ubatch(t as u64, resp.n_prompt),
                    pos_path: b3_path(&resp.pos_roots, t),
                    meta_path: b3_path(&resp.pos_leaves[t], 2),
                    entry: self.entry_opening(0),
                    partner: None,
                    bias: None,
                    rows: vec![self.range(row * rb / 32, (row * rb + rb - 1) / 32)],
                    x: resp.x[t].clone(),
                    x_path: b3_path(&resp.pos_leaves[t], 0),
                    y_total_len: (M * 4) as u32,
                    y_chunk_index: 0,
                    y_chunk: resp.y[t].clone(),
                    y_chunk_path: Vec::new(),
                    y_path: b3_path(&resp.pos_leaves[t], 1),
                }
            })
            .collect();
        Opening { response_txid: txid, points, prefix: None, links: Vec::new(), experts: Vec::new(), resid: None, roots: None }
    }

    /// [`Self::open`] with the prefix point of `(request id, offset)`.
    pub fn open_with_prefix(&self, resp: &Response, txid: Hash, seed: &Hash, request_txid: &Hash, offset: u32) -> Opening {
        let mut o = self.open(resp, txid, seed);
        let t = prefix::position(seed, offset) as usize;
        let tok = request_txid[t - offset as usize] as u64;
        let (e, n) = (&self.entries[1].0, &self.entries[2].0);
        let rb = e.row_bytes as u64;
        o.prefix = Some(PrefixPoint {
            n_ubatch: expected_ubatch(t as u64, resp.n_prompt),
            pos_path: b3_path(&resp.pos_roots, t),
            meta_path: b3_path(&resp.pos_leaves[t], 2),
            entry: self.entry_opening(0),
            x: resp.x[t].clone(),
            x_path: b3_path(&resp.pos_leaves[t], 0),
            embd: (self.entry_opening(1), self.range(e.chunk_base + tok * rb / 32, e.chunk_base + (tok * rb + rb - 1) / 32)),
            norm: (self.entry_opening(2), self.range(n.chunk_base, n.chunk_base + (4 * K as u64 - 1) / 32)),
        });
        o
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::verify_prefix_point;
    use crate::verify::{Fail, ResponseCtx, verify_point};

    #[test]
    fn synthetic_prefix_points_verify() {
        let m = Model::new();
        let tier = m.tier();
        let ctx = m.prefix_ctx(6);
        let req: Hash = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let other: Hash = std::array::from_fn(|i| (i * 7 + 4) as u8);
        let r = m.response_with_prefix(40, 48, true, Some((&req, 6)));
        let resp = ResponseCtx { trace_root: r.trace_root, n_prompt: r.n_prompt, n_pos: r.n_pos };
        for s in 0..32u8 {
            let seed = [s; 32];
            let o = Opening::decode(&m.open_with_prefix(&r, [5; 32], &seed, &req, 6).encode()).unwrap();
            let p = o.prefix.as_ref().unwrap();
            let t = verify_prefix_point(&tier, &ctx, &resp, &seed, &req, p).unwrap();
            assert!((6..38).contains(&t));
            // The row of another request's token does not match the committed input.
            assert!(verify_prefix_point(&tier, &ctx, &resp, &seed, &other, p).is_err());
            let forged = m.open_with_prefix(&r, [5; 32], &seed, &other, 6);
            assert_eq!(verify_prefix_point(&tier, &ctx, &resp, &seed, &other, forged.prefix.as_ref().unwrap()), Err(Fail::Mismatch));
            for (i, pt) in o.points.iter().enumerate() {
                verify_point(&tier, &resp, &seed, i as u32, pt).unwrap();
            }
        }
        // A trace made for another request, opened with this request's rows, still fails.
        let r2 = m.response_with_prefix(40, 48, true, Some((&other, 6)));
        let resp2 = ResponseCtx { trace_root: r2.trace_root, n_prompt: r2.n_prompt, n_pos: r2.n_pos };
        let seed = [9u8; 32];
        let o = m.open_with_prefix(&r2, [5; 32], &seed, &req, 6);
        assert_eq!(verify_prefix_point(&tier, &ctx, &resp2, &seed, &req, o.prefix.as_ref().unwrap()), Err(Fail::Mismatch));
        // A prompt too short for the prefix.
        let short = ResponseCtx { trace_root: r.trace_root, n_prompt: 30, n_pos: r.n_pos };
        let o = m.open_with_prefix(&r, [5; 32], &seed, &req, 6);
        assert_eq!(verify_prefix_point(&tier, &ctx, &short, &seed, &req, o.prefix.as_ref().unwrap()), Err(Fail::Shape));
        // Version 1 openings still decode without a prefix.
        assert!(Opening::decode(&m.open(&r, [5; 32], &seed).encode()).unwrap().prefix.is_none());
    }

    #[test]
    fn seeded_envelope_round_trips() {
        use crate::opening::{VERSION_SEEDED, seed_block, seeded};
        let m = Model::new();
        let r = m.response(3, 9, true);
        let req: Hash = [4; 32];
        let rp = m.response_with_prefix(40, 48, true, Some((&req, 6)));
        let block: Hash = [7; 32];
        for body in [m.open(&r, [5; 32], &[1; 32]).encode(), m.open_with_prefix(&rp, [5; 32], &[1; 32], &req, 6).encode()] {
            assert_eq!(seed_block(&body), None);
            let s = seeded(&body, &block).unwrap();
            assert_eq!(s[0], VERSION_SEEDED);
            assert_eq!(&s[1..33], &[5u8; 32]);
            assert_eq!(seed_block(&s), Some(block));
            assert_eq!(Opening::decode(&s), Opening::decode(&body));
            assert!(seeded(&s, &block).is_none());
            assert!(Opening::decode(&s[..s.len() - 1]).is_none());
        }
        let mut nested = seeded(&m.open(&r, [5; 32], &[1; 32]).encode(), &block).unwrap();
        nested[65] = VERSION_SEEDED;
        assert!(Opening::decode(&nested).is_none());
    }

    #[test]
    fn synthetic_openings_verify() {
        let m = Model::new();
        let tier = m.tier();
        for honest in [true, false] {
            let r = m.response(3, 9, honest);
            let ctx = ResponseCtx { trace_root: r.trace_root, n_prompt: r.n_prompt, n_pos: r.n_pos };
            for s in 0..32u8 {
                let seed = [s; 32];
                let o = Opening::decode(&m.open(&r, [5; 32], &seed).encode()).unwrap();
                for (i, p) in o.points.iter().enumerate() {
                    let res = verify_point(&tier, &ctx, &seed, i as u32, p);
                    assert_eq!(res.is_ok(), honest, "seed {s} point {i}: {res:?}");
                }
            }
        }
    }
}
