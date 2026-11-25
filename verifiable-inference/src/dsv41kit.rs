//! Synthetic deepseek41 traces for the compressed row point, shared by the node and miner tests.

use crate::attn::Yarn;
use crate::chain::ChainCtx;
use crate::dsv41::{comp_latent, comp_row, COMP_POINT};
use crate::manifest::{self, AttnLinks, Chain, CompLinks, Entry, LayerLinks, ProductLink};
use crate::merkle::{b3_chunks_proof, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use crate::opening::{ChunkRange, CompOpeningPoint, EntryOpening, LeafBytes, SelMember};
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
