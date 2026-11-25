//! Synthetic deepseek41 traces for the compressed row point, shared by the node and miner tests.

use crate::attn::Yarn;
use crate::chain::ChainCtx;
use crate::dsa::{drawn_pools, SEL_POINT};
use crate::dsv41::{comp_latent, comp_row, index_score, rope_tail, verify_comp_sel_point, COMP_POINT};
use crate::verify::Fail;
use crate::manifest::{self, AttnLinks, Chain, CompLinks, Entry, LayerLinks, ProductLink};
use crate::merkle::{b3_chunks_proof, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use crate::opening::{ChunkRange, CompOpeningPoint, CompSelOpening, EntryOpening, LeafBytes, SelMember};
use crate::seq::{SeqLayout, SeqSpan};
use crate::verify::{ResponseCtx, TierCtx};

fn vals(seed: u64, n: usize, a: f32) -> Vec<f32> {
    (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
}

/// Q8_0 blocks of `v`, rounded as ggml rounds them.
pub fn q8_0(v: &[f64]) -> Vec<u8> {
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
    norm: Vec<f32>,
    w: Vec<u8>,
    k_norm: Vec<f32>,
    yarn: Yarn,
}

fn block() -> Block {
    let rows: Vec<f64> = vals(9, 128 * 512, 0.05).iter().map(|&x| x as f64).collect();
    Block {
        norm: (0..512).map(|i| 0.5 + 0.01 * (i % 37) as f32).collect(),
        w: rows.chunks(512).flat_map(q8_0).collect(),
        k_norm: (0..128).map(|i| 0.8 + 0.02 * (i % 11) as f32).collect(),
        yarn: Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 },
    }
}

pub const N: usize = 6;
const NONE: u32 = manifest::NONE;

pub struct CompKit {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<Vec<u8>>,
    pub entry_leaves: Vec<Hash>,
    pub tier: TierCtx,
    pub ctx: ChainCtx,
    pub pos_leaves: Vec<Vec<Vec<u8>>>,
    pub pos_roots: Vec<Hash>,
    pub pos_root: Hash,
    pub seq_leaves: Vec<Vec<u8>>,
    pub seq_root: Hash,
    pub offs: [usize; 3],
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

/// One layer whose compressed rows (ratio 2) are pooled from `N` positions; gates swapped when
/// `swap` (the rows then disagree with the members).
pub fn build(swap: bool) -> CompKit {
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
    CompKit { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos_leaves, pos_roots, pos_root, seq_leaves, seq_root, offs }
}

impl CompKit {
    pub fn eo(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    pub fn range(&self, off: usize, n: usize) -> ChunkRange {
        let (first, count) = (off / 32, n / 32);
        ChunkRange { first_chunk: first as u64, chunks: self.weights[off..off + n].to_vec(), proof: rt_range_proof(&self.chunk_leaves, first as u64, count as u64).1 }
    }

    pub fn resp(&self) -> ResponseCtx {
        ResponseCtx { trace_root: crate::seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: N as u32, n_pos: N as u32 }
    }

    pub fn leaf(&self, t: usize, i: usize) -> LeafBytes {
        let hashes: Vec<Hash> = self.pos_leaves[t].iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        LeafBytes { bytes: self.pos_leaves[t][i].clone(), leaf_path: b3_path(&hashes, i) }
    }

    pub fn open(&self, seed: &Hash) -> CompOpeningPoint {
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

// --- two-level selection ---

pub const S_N: usize = 12;
const S_NIH: usize = 2;
const S_DK: usize = 128;
const S_LAT: usize = 32;
const S_NTOP: usize = 2;
const S_CN: usize = 2;
const S_BS: usize = 2;
const S_NONE: u32 = manifest::NONE;
const S_LAYOUT: SeqLayout = SeqLayout { n_attn: 2, n_recur: 0, heads: 1, n_comp: 1 };

#[derive(Clone, Copy, PartialEq)]
pub enum SelFault {
    None,
    /// Layer 0 keeps its two lowest rows.
    LowRows,
    /// Layer 0 picks its lowest blocks besides the pinned one.
    LowBlocks,
    /// Layer 1 keeps a row outside the candidate blocks.
    OutsideCandidates,
}

pub struct SelKit {
    pub entries: Vec<Vec<u8>>,
    pub entry_leaves: Vec<Hash>,
    pub tier: TierCtx,
    pub ctx: ChainCtx,
    pub pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)>,
    pub pos_roots: Vec<Hash>,
    pub pos_root: Hash,
    pub seq_leaves: Vec<Vec<u8>>,
    pub seq_hashes: Vec<Hash>,
    pub seq_root: Hash,
}

fn sel_entry(name: &str, role: u8, k: usize, m: usize, chain: Chain) -> Vec<u8> {
    let e = Entry {
        name: name.into(), wtype: crate::quant::Q8_0, k: k as u64, m: m as u64, n_exp: 1, chunk_base: 0, n_bytes: 0, row_bytes: 0, role, glu_op: 0,
        glu_form: manifest::FORM_NONE, partner: S_NONE, bias: S_NONE, chain,
    };
    let mut b = crate::testkit::serialize(&e);
    b.extend_from_slice(&manifest::chain_bytes(&e.chain));
    b
}

fn links(layer: u32, iq: u32, sel: u32, pick: u32) -> CompLinks {
    let y = block().yarn;
    CompLinks {
        src: 0, slot: 0, ratio: 1, kv: S_NONE, gate: S_NONE, norm: S_NONE, ik: S_NONE, k_norm: S_NONE, n_rot: 64, rope_base: 160_000f32.to_bits(),
        freq_scale: (y.freq_scale as f32).to_bits(), ext_factor: 1f32.to_bits(), attn_factor: (y.attn_factor as f32).to_bits(), beta_fast: 32f32.to_bits(),
        beta_slow: 1f32.to_bits(), n_ctx_orig: 65_536, eps: 1e-20f32.to_bits(), sinks: S_NONE, sel, n_top: S_NTOP as u32, row_len: (S_LAT + S_DK) as u32,
        iq, iw: iq + 1, topk_src: layer, cand: 1, cand_n: S_CN as u32, cand_bs: S_BS as u32, pick,
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

pub fn build_sel(fault: SelFault) -> SelKit {
    let a_links = |slot: u32| AttnLinks {
        q: 0, k: 0, v: S_NONE, q_norm: S_NONE, k_norm: S_NONE, n_head: 1, n_head_kv: 1, head_dim: 512, rope_kind: 0, n_rot: 64, rope_base: 0,
        kq_scale: 1f32.to_bits(), swa_window: 4, softcap: 0, rope_freqs: S_NONE, v_norm: 0, gated: 0, slot, mla_rank: 0, absorb: S_NONE, out: S_NONE,
    };
    let none_link = Chain::Product(ProductLink { layer: 0, link: manifest::LINK_NONE, norm: S_NONE, src: S_NONE });
    let layer = |l: u32, c: CompLinks| {
        Chain::Layer(LayerLinks { layer: l, proj: 0, down: 0, post_attn: S_NONE, post_ffn: S_NONE, scale: S_NONE, moe: None, attn: Some(a_links(l)), recur: None, hc: None, dsa: None, comp: Some(c), engram: None })
    };
    let entries = vec![
        sel_entry("blk.0.indexer.attn_q_b.weight", manifest::ROLE_PRODUCT, 64, S_NIH * S_DK, none_link),
        sel_entry("blk.0.indexer.proj.weight", manifest::ROLE_PRODUCT, 64, S_NIH, none_link),
        sel_entry("blk.1.indexer.attn_q_b.weight", manifest::ROLE_PRODUCT, 64, S_NIH * S_DK, none_link),
        sel_entry("blk.1.indexer.proj.weight", manifest::ROLE_PRODUCT, 64, S_NIH, none_link),
        sel_entry("blk.0", manifest::ROLE_LAYER, 0, 0, layer(0, links(0, 0, 0, 1))),
        sel_entry("blk.1", manifest::ROLE_LAYER, 0, 0, layer(1, links(1, 2, 2, 0))),
    ];
    let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
    let tier = TierCtx { r_t: [0; 32], n_chunks: 1, m_t: rt_range_proof(&entry_leaves, 0, 1).0, n_entries: entries.len() as u64, n_products: 4 };
    let ctx = ChainCtx { n_sel: 3, n_comp: 1, n_engram: 0, n_layers: 2, layer_base: 4, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 2, n_recur: 0, seq_heads: 1 };
    let y = block().yarn;
    let rows: Vec<Vec<u8>> = (0..S_N).map(|j| q8_0(&vals(700 + j as u64, S_LAT + S_DK, 1.0).iter().map(|&x| x as f64).collect::<Vec<_>>())).collect();
    let keys: Vec<Vec<f32>> = rows.iter().map(|r| crate::quant::dequant_row(crate::quant::Q8_0, r, S_LAT + S_DK).unwrap()[S_LAT..].to_vec()).collect();
    let iq: Vec<[Vec<f32>; 2]> = (0..S_N).map(|t| [vals(800 + t as u64, S_NIH * S_DK, 1.0), vals(820 + t as u64, S_NIH * S_DK, 1.0)]).collect();
    let iw: Vec<[Vec<f32>; 2]> = (0..S_N).map(|t| [vals(840 + t as u64, S_NIH, 1.0).iter().map(|x| x.abs() + 0.2).collect(), vals(860 + t as u64, S_NIH, 1.0).iter().map(|x| x.abs() + 0.2).collect()]).collect();
    let score = |t: usize, l: usize, j: usize| {
        let mut q: Vec<f64> = iq[t][l].iter().map(|&x| x as f64).collect();
        for h in q.chunks_mut(S_DK) {
            rope_tail(h, t as u64, 64, 160_000.0, &y);
        }
        let w: Vec<f64> = iw[t][l].iter().map(|&x| x as f64 / ((S_DK * S_NIH) as f64).sqrt()).collect();
        index_score(&q, &w, &keys[j]).0
    };
    let f = |v: &[f32]| v.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>();
    let n_leaves = crate::chain::n_pos_leaves(4, 2, 0) as usize + 3;
    let pos: Vec<(Vec<Vec<u8>>, Vec<Hash>)> = (0..S_N)
        .map(|t| {
            let n_vis = t + 1;
            let s0: Vec<f64> = (0..n_vis).map(|j| score(t, 0, j)).collect();
            let s1: Vec<f64> = (0..n_vis).map(|j| score(t, 1, j)).collect();
            let bmax: Vec<f64> = (0..n_vis.div_ceil(S_BS)).map(|b| (b * S_BS..((b + 1) * S_BS).min(n_vis)).map(|j| s0[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
            let pinned = (n_vis - 1) / S_BS;
            let (bthr, blocks) = if fault == SelFault::LowBlocks && bmax.len() > S_CN {
                let neg: Vec<f64> = bmax.iter().map(|x| -x).collect();
                let (th, k) = top(&neg, |_| true, S_CN, Some(pinned));
                (-th, k)
            } else {
                top(&bmax, |_| true, S_CN, Some(pinned))
            };
            let (thr0, kept0) = if fault == SelFault::LowRows && n_vis > S_NTOP {
                let neg: Vec<f64> = s0.iter().map(|x| -x).collect();
                let (th, k) = top(&neg, |_| true, S_NTOP, None);
                (-th, k)
            } else {
                top(&s0, |_| true, S_NTOP, None)
            };
            let in_cand = |j: usize| blocks.contains(&(j / S_BS));
            let (thr1, mut kept1) = top(&s1, in_cand, S_NTOP, None);
            if fault == SelFault::OutsideCandidates && n_vis > S_CN * S_BS {
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
            leaves[base] = leaf(thr0, &kept0, S_NTOP);
            leaves[base + 1] = leaf(bthr, &blocks, S_CN);
            leaves[base + 2] = leaf(thr1, &kept1, S_NTOP);
            leaves[n_leaves - 1] = crate::chain::meta(t as u32, 1, 7).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            (leaves, hashes)
        })
        .collect();
    let pos_roots: Vec<Hash> = pos.iter().map(|(_, h)| b3_root(h)).collect();
    let pos_root = b3_root(&pos_roots);
    let mut seq_leaves = vec![Vec::new(); S_LAYOUT.n_leaves() as usize];
    seq_leaves[S_LAYOUT.comp(0).unwrap() as usize] = rows.concat();
    let seq_hashes: Vec<Hash> = seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
    let seq_root = b3_root(&seq_hashes);
    SelKit { entries, entry_leaves, tier, ctx, pos, pos_roots, pos_root, seq_leaves, seq_hashes, seq_root }
}

impl SelKit {
    pub fn eo(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    pub fn resp(&self) -> ResponseCtx {
        ResponseCtx { trace_root: crate::seq::trace_root(&self.pos_root, &self.seq_root), n_prompt: S_N as u32, n_pos: S_N as u32 }
    }

    pub fn lb(&self, t: usize, i: usize) -> LeafBytes {
        LeafBytes { bytes: self.pos[t].0[i].clone(), leaf_path: b3_path(&self.pos[t].1, i) }
    }

    pub fn span(&self, lo: usize, hi: usize) -> SeqSpan {
        let leaf = S_LAYOUT.comp(0).unwrap() as usize;
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
    pub fn open(&self, seed: &Hash) -> Option<CompSelOpening> {
        let d = crate::draw::point(seed, SEL_POINT);
        let s = (d[1] % 3) as usize;
        let (layer, l) = [(4usize, 0usize), (4, 0), (5, 1)][s];
        let picks = s == 1;
        let first = if picks { S_CN * S_BS } else { S_NTOP };
        let t = first + (d[0] % (S_N - first) as u64) as usize;
        let n_vis = t + 1;
        let base = crate::chain::sel_leaf(4, 2, 0, 0) as usize;
        let rb = (S_LAT + S_DK) / 32 * 34;
        let kept = |b: &[u8]| b[4..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).filter(|&x| x != u32::MAX).map(|x| x as u64).collect::<Vec<_>>();
        let sel = &self.pos[t].0[base + s];
        let rows = if picks {
            drawn_pools(seed, &kept(sel), n_vis.div_ceil(S_BS) as u64)
                .iter()
                .map(|&b| self.span(b as usize * S_BS * rb, ((b as usize + 1) * S_BS).min(n_vis) * rb))
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

    pub fn verify(&self, seed: &Hash) -> Result<(), Fail> {
        verify_comp_sel_point(&self.tier, &self.ctx, &self.resp(), seed, &self.open(seed).unwrap())
    }
}
