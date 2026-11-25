//! Synthetic n-gram lookup traces, shared by the node and miner tests.

use crate::chain::{verify_chain_point, ChainCtx};
use crate::engram::{engram_rows, stream_gate, ENGRAM_POINT};
use crate::manifest::{self, Chain, EngramLinks, Entry, HcLinks, LayerLinks, ProductLink};
use crate::merkle::{b3_chunk_cv, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use crate::opening::{ChunkRange, EngramOpening, EntryOpening, LeafChunk, LinkOpening, Point, PrevToken};
use crate::testkit::{in_layout, layout_at, q8_1_row};
use crate::verify::{Fail, ResponseCtx, TierCtx};

/// Links of a small n-gram layer over `map` and `table`.
pub fn links(map: &[u32], table: &[u8]) -> EngramLinks {
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

pub const N: usize = 6;
const D: usize = 256;
const TRB: usize = D / 32 * 18;
const COLS: usize = 6;
pub const K: usize = COLS * D;
const M: usize = 8;
const ROWS: usize = 208;
const NONE: u32 = manifest::NONE;

fn bytes(seed: u64, n: usize) -> Vec<u8> {
    (0..n).map(|i| blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0]).collect()
}

pub struct LinkKit {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<Vec<u8>>,
    pub entry_leaves: Vec<Hash>,
    pub tier: TierCtx,
    pub ctx: ChainCtx,
    pub e: EngramLinks,
    pub map: Vec<u32>,
    pub table: Vec<u8>,
    pub tokens: Vec<u32>,
    pub pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
    pub pos_roots: Vec<Hash>,
}

fn ids_at(e: &EngramLinks, map: &[u32], tokens: &[u32], t: usize, unigram: bool) -> Vec<u32> {
    let ids: Vec<Option<u32>> = (0..4).map(|s| if s > t || (unigram && s > 0) { None } else { map.get(tokens[t - s] as usize).copied() }).collect();
    engram_rows(e, &ids).unwrap()
}

/// One n-gram layer whose value product reads `COLS` table rows; with `unigram`, the input
/// is built from the token alone (refused).
pub fn build_link(unigram: bool) -> LinkKit {
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
    LinkKit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, e, map, table, tokens, pos, pos_roots }
}

impl LinkKit {
    pub fn eo(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: crate::merkle::rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    pub fn map_chunk(&self, tok: u32) -> ChunkRange {
        let mb: Vec<u8> = self.map.iter().flat_map(|v| v.to_le_bytes()).chain(std::iter::repeat(0)).take(self.map.len().div_ceil(8) * 32).collect();
        let ml: Vec<Hash> = mb.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let c = tok as usize * 4 / 32;
        ChunkRange { first_chunk: c as u64, chunks: mb[c * 32..c * 32 + 32].to_vec(), proof: crate::merkle::rt_range_proof(&ml, c as u64, 1).1 }
    }

    pub fn row_range(&self, r: u32) -> ChunkRange {
        let tl: Vec<Hash> = self.table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let (lo, hi) = (r as u64 * TRB as u64, (r as u64 + 1) * TRB as u64);
        let (f, l) = (lo / 32, (hi - 1) / 32);
        ChunkRange { first_chunk: f, chunks: self.table[(f * 32) as usize..((l + 1) * 32) as usize].to_vec(), proof: crate::merkle::rt_range_proof(&tl, f, l - f + 1).1 }
    }

    pub fn open(&self, seed: &Hash, lie: bool) -> (Point, u32, LinkOpening) {
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

    pub fn check(&self, seed: &Hash, lie: bool) -> (usize, Result<(), Fail>) {
        let (p, tok, link) = self.open(seed, lie);
        let resp = ResponseCtx { trace_root: b3_root(&self.pos_roots), n_prompt: N as u32, n_pos: N as u32 };
        let t = (crate::draw::point(seed, 0)[0] % N as u64) as usize;
        (t, verify_chain_point(&self.tier, &self.ctx, &resp, seed, 0, &p, tok, &link, None).map(|_| ()))
    }
}

// --- residual after the lookup ---

pub const R_N: usize = 5;
const R_NHC: usize = 2;
const R_D: usize = 256;
const R_NONE: u32 = manifest::NONE;

#[derive(Clone, Copy, PartialEq)]
pub enum ResidFault {
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

pub struct ResidKit {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<Vec<u8>>,
    pub entry_leaves: Vec<Hash>,
    pub tier: TierCtx,
    pub ctx: ChainCtx,
    pub pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
    pub pos_roots: Vec<Hash>,
}

fn entry(name: &str, role: u8, wtype: u32, k: usize, m: usize, n: usize, chain: Chain) -> Vec<u8> {
    let e = Entry {
        name: name.into(), wtype, k: k as u64, m: m as u64, n_exp: 1, chunk_base: 0, n_bytes: n as u64, row_bytes: n as u32, role, glu_op: 0,
        glu_form: manifest::FORM_NONE, partner: R_NONE, bias: R_NONE, chain,
    };
    let mut b = crate::testkit::serialize(&e);
    b.extend_from_slice(&manifest::chain_bytes(&e.chain));
    b
}

/// One n-gram layer adding its gated value to the streams, faulted as `fault`.
pub fn build_resid(fault: ResidFault) -> ResidKit {
    let gates: Vec<f32> = vals(5, R_NHC * R_D, 1.5);
    let weights: Vec<u8> = gates.iter().flat_map(|a| a.to_le_bytes()).collect();
    let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    let hc = HcLinks { attn_fn: 0, ffn_fn: 0, attn_scale: [0; 3], ffn_scale: [0; 3], attn_base: R_NONE, ffn_base: R_NONE, n_hc: R_NHC as u32, sinkhorn_iters: 20, eps: 0 , lag: 0, prev_fn: R_NONE, prev_scale: [0; 3], prev_base: R_NONE};
    let mut e = links(&[0], &[0; 32]);
    e.module = 0;
    e.kv = 0;
    e.gate = 3;
    e.out_leaf = crate::chain::engram_leaf(1, 2, 0, 0, 0) as u32;
    let layer = |l: u32, eg: Option<EngramLinks>| {
        Chain::Layer(LayerLinks { layer: l, proj: 0, down: 0, post_attn: R_NONE, post_ffn: R_NONE, scale: R_NONE, moe: None, attn: None, recur: None, hc: Some(hc), dsa: None, comp: None, engram: eg })
    };
    let entries = vec![
        entry("blk.1.engram_kv.weight", manifest::ROLE_PRODUCT, crate::quant::Q8_0, 512, (R_NHC + 1) * R_D, 0, Chain::None),
        entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, layer(0, None)),
        entry("blk.1", manifest::ROLE_LAYER, 0, 0, 0, 0, layer(1, Some(e))),
        entry("blk.1.engram_gate.weight", manifest::ROLE_NORM, 0, R_NHC * R_D, 1, weights.len(), Chain::None),
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
    let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..R_N)
        .map(|t| {
            let x = vals(100 + t as u64, R_NHC * R_D, 2.0);
            let kv = vals(200 + t as u64, (R_NHC + 1) * R_D, 1.0);
            let value = &kv[R_NHC * R_D..];
            let dots: Vec<f64> = (0..R_NHC)
                .map(|s| {
                    let (rx, rk) = (rn(&x[s * R_D..(s + 1) * R_D]), rn(&kv[s * R_D..(s + 1) * R_D]));
                    rx.iter().zip(&rk).zip(&gates[s * R_D..(s + 1) * R_D]).map(|((a, b), &c)| a * b * c as f64).sum::<f64>() / (R_D as f64).sqrt()
                })
                .collect();
            let out: Vec<f32> = (0..R_NHC)
                .flat_map(|s| {
                    let dot = if fault == ResidFault::SwappedGates { dots[R_NHC - 1 - s] } else { dots[s] };
                    let g = match fault {
                        ResidFault::PlainGate => 1.0 / (1.0 + libm::exp(-dot)),
                        ResidFault::Ungated => 1.0,
                        _ => stream_gate(dot),
                    };
                    let x = &x;
                    (0..R_D).map(move |j| (x[s * R_D + j] as f64 + value[j] as f64 * g) as f32).collect::<Vec<_>>()
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
    ResidKit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos, pos_roots }
}

impl ResidKit {
    pub fn eo(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: crate::merkle::rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    pub fn chunk(&self, t: usize, i: usize, c: usize) -> LeafChunk {
        let b = &self.pos[t].0[i];
        let n = b.len().div_ceil(B3_CHUNK);
        let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
        LeafChunk { total_len: b.len() as u32, index: c as u32, chunk: b[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(b.len())].to_vec(), chunk_path: b3_path(&cvs, c), leaf_path: b3_path(&self.pos[t].1, i) }
    }

    pub fn resp(&self) -> ResponseCtx {
        ResponseCtx { trace_root: crate::seq::trace_root(&b3_root(&self.pos_roots), &[5; 32]), n_prompt: R_N as u32, n_pos: R_N as u32 }
    }

    pub fn open(&self, seed: &Hash) -> EngramOpening {
        let d = crate::draw::point(seed, ENGRAM_POINT);
        let t = (d[0] % R_N as u64) as usize;
        let (s, jc) = ((d[2] % R_NHC as u64) as usize, 0usize);
        let out = crate::chain::engram_leaf(1, 2, 0, 0, 0) as usize;
        let src = crate::chain::resid_leaf(1, 0, crate::chain::Resid::Out) as usize;
        EngramOpening {
            response_txid: [1; 32],
            pos_root: b3_root(&self.pos_roots),
            seq_root: [5; 32],
            layer: self.eo(2),
            kv: self.eo(0),
            gate: (self.eo(3), ChunkRange { first_chunk: (s * R_D * 4 / 32) as u64, chunks: self.weights[s * R_D * 4..(s + 1) * R_D * 4].to_vec(), proof: crate::merkle::rt_range_proof(&self.chunk_leaves, (s * R_D * 4 / 32) as u64, (R_D * 4 / 32) as u64).1 }),
            pos_path: b3_path(&self.pos_roots, t),
            dst: self.chunk(t, out, s + jc),
            x: vec![self.chunk(t, src, s)],
            key: vec![self.chunk(t, 1, s)],
            value: self.chunk(t, 1, R_NHC + jc),
        }
    }
}
