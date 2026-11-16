//! Inference trace of a sealed response: commitment, draw and opening. Byte-exact mirror of the
//! node's `collateral.rs` trace rules on top of the shared `keryx-verifiable-inference` crate.

use keryx_inference::TraceCommitment;
use keryx_verifiable_inference::chain::{self, ChainCtx, Resid};
use keryx_verifiable_inference::draw;
use keryx_verifiable_inference::manifest::{self, Chain, Entry};
use borsh::{BorshDeserialize, BorshSerialize};
use keryx_verifiable_inference::merkle::{b3_chunk_cv, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use keryx_verifiable_inference::opening::{
    self, ChunkRange, EntryOpening, ExpertIds, ExpertRow, FeedRow, LeafBytes, LeafChunk, LinkOpening, MoeAdd, Opening, Point, PrefixPoint,
    ResidAdd, ResidOpening, ResidPoint, ResidSrc,
};
use keryx_verifiable_inference::prefix::{self, PrefixCtx};
use keryx_verifiable_inference::verify::{self, ResponseCtx, TierCtx};
use std::collections::BTreeMap;

use crate::pom::WeightIndex;

pub const TRACE_DRAW_DOMAIN: &[u8] = b"KeryxTraceDrawV1";
pub const OPENING_SIG_DOMAIN: &[u8] = b"KeryxTraceOpeningV1";
/// DAA between a response's acceptance and the chain block that seeds its draw.
pub const TRACE_SEED_DELAY_DAA: u64 = 100;
/// Stage committed by this miner: products only (1a).
pub const STAGE_1A: u8 = 1;
/// Points opened per audited response.
pub const POINTS: u32 = draw::POINTS;

/// Serialized manifests of the tiers, in tier order; `None` = no manifest.
const MANIFESTS: [Option<&[u8]>; 5] = [
    Some(include_bytes!("../assets/trace-manifests/tier0.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier1.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier2.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier3.bin")),
    None,
];

/// Prefix contexts of the audited tiers, in tier order. MUST equal the node's `TRACE_PREFIXES`.
const PREFIXES: [Option<PrefixCtx>; 5] = [
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 6,
        embd_entry: 249,
        norm_entry: 250,
        eps: 9.999999974752427e-07,
        embd_scale: 1.0,
    }),
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 7,
        embd_entry: 361,
        norm_entry: 362,
        eps: 9.999999747378752e-06,
        embd_scale: 1.0,
    }),
    Some(PrefixCtx {
        table: [
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 107, 249, 250, 251, 252, 253,
            254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
            270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
            302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
            318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
            334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
            350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
            366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381,
            382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
            398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
            414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
            430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445,
            446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
            462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
            478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
        ],
        offset: 7,
        embd_entry: 329,
        norm_entry: 330,
        eps: 9.999999974752427e-07,
        embd_scale: 61.967735290527344,
    }),
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 6,
        embd_entry: 497,
        norm_entry: 498,
        eps: 9.999999974752427e-07,
        embd_scale: 1.0,
    }),
    None,
];

/// Stage committed once `trace_chain_activation` is live: products, chain leaves and links.
pub const STAGE_CHAIN: u8 = keryx_verifiable_inference::chain::STAGE_CHAIN;

/// Serialized stage-2 manifests (`KXMANIF2`), in tier order. MUST equal the node's
/// `TRACE_CHAINS_H15` manifests.
const CHAIN_MANIFESTS: [&[u8]; 5] = [
    include_bytes!("../assets/trace-manifests/chain-tier0.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier1.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier2.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier3.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier4.bin"),
];

/// Serialized manifests under the attention rule (attention and recurrent descriptors), in tier
/// order. MUST equal the node's `TRACE_ATTN_H_NEXT` manifests.
const ATTN_MANIFESTS: [&[u8]; 5] = [
    include_bytes!("../assets/trace-manifests/attn-tier0.bin"),
    include_bytes!("../assets/trace-manifests/attn-tier1.bin"),
    include_bytes!("../assets/trace-manifests/attn-tier2.bin"),
    include_bytes!("../assets/trace-manifests/attn-tier3.bin"),
    include_bytes!("../assets/trace-manifests/attn-tier4.bin"),
];

/// Attention layers, recurrent layers and sequence heads of each manifest under the attention
/// rule. MUST equal the node's `TRACE_ATTN_H_NEXT`.
const ATTN_SEQ: [(u32, u32, u32); 5] = [(8, 24, 32), (40, 0, 2), (48, 0, 8), (16, 48, 48), (7, 20, 32)];

/// Positions between recurrent state checkpoints when a manifest has no recurrent layer.
const DEFAULT_INTERVAL: u32 = 48;

/// Stage-2 chain context of each tier, with the prefix embedding and norm entries of its
/// manifest. MUST equal the node's `TRACE_CHAINS_H15`.
const CHAIN_CTX: [Option<(ChainCtx, u32, u32)>; 5] = [
    Some((ChainCtx { n_layers: 32, layer_base: 315, embd_entry: 249, eps: 9.999999974752427e-07, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 }, 249, 250)),
    Some((ChainCtx { n_layers: 40, layer_base: 523, embd_entry: 361, eps: 9.999999747378752e-06, embd_scale: 1.0, trim_last: true, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 }, 361, 362)),
    Some((ChainCtx { n_layers: 48, layer_base: 523, embd_entry: 329, eps: 9.999999974752427e-07, embd_scale: 61.967735290527344, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 }, 329, 330)),
    Some((ChainCtx { n_layers: 64, layer_base: 627, embd_entry: 497, eps: 9.999999974752427e-07, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 }, 497, 498)),
    Some((ChainCtx { n_layers: 27, layer_base: 503, embd_entry: 395, eps: 9.999999747378752e-06, embd_scale: 1.0, trim_last: true, n_used: 8, n_attn: 0, n_recur: 0, seq_heads: 0 }, 395, 396)),
];

/// Entries of the stage-2 manifest of `tier` and its product count.
pub fn chain_manifest(tier: u8) -> Option<keryx_verifiable_inference::manifest::Manifest> {
    keryx_verifiable_inference::manifest::read_file(CHAIN_MANIFESTS.get(tier as usize)?)
}

/// Stage-2 chain context of `tier`, and its prefix context on the stage-2 manifest.
pub fn chain_ctx(tier: u8) -> Option<(ChainCtx, PrefixCtx)> {
    let (ctx, embd, norm) = (*CHAIN_CTX.get(tier as usize)?)?;
    let prefix = PREFIXES.get(tier as usize)?.clone().map(|p| PrefixCtx { embd_entry: embd, norm_entry: norm, ..p });
    let prefix = match (tier, prefix) {
        (_, Some(p)) => p,
        (4, None) => PrefixCtx { table: PREFIXES[0].clone()?.table, offset: KIMI_PREFIX_OFFSET, embd_entry: embd, norm_entry: norm, eps: ctx.eps, embd_scale: ctx.embd_scale },
        _ => return None,
    };
    Some((ctx, prefix))
}

/// Tokens of the Kimi chat-template head before the request prefix (`<|im_system|>system<|im_middle|>Request id:`).
pub const KIMI_PREFIX_OFFSET: u32 = 6;


/// The prefix context of `tier`; `None` = no manifest.
pub fn prefix_ctx(tier: u8) -> Option<&'static PrefixCtx> {
    PREFIXES.get(tier as usize).and_then(|p| p.as_ref())
}

/// Whether a response accepted at `accepted_daa` is opened under the prefix rule.
pub fn prefix_active(accepted_daa: u64) -> bool {
    accepted_daa >= crate::pom::trace_prefix_activation_daa()
}

/// Whether a response generated at `daa` commits the stage-2 trace.
pub fn chain_active(daa: u64) -> bool {
    daa >= crate::pom::trace_chain_activation_daa()
}

/// Whether a response generated at `daa` commits sequence leaves and is opened under the
/// attention rule.
pub fn attention_active(daa: u64) -> bool {
    daa >= crate::pom::attention_activation_daa()
}

/// Whether every traced response accepted at `accepted_daa` is audited.
pub fn audit_all_active(accepted_daa: u64) -> bool {
    accepted_daa >= crate::pom::trace_audit_activation_daa()
}

/// The token ids the request id puts at the prefix positions of a traced prompt.
pub fn prefix_tokens(ctx: &PrefixCtx, request_hash: &[u8; 32]) -> Vec<i32> {
    request_hash.iter().map(|b| ctx.table[*b as usize] as i32).collect()
}

/// Manifest of an audited tier: the canonical product list and its root `M_T`.
pub struct TierManifest {
    pub entries: Vec<(Entry, Vec<u8>)>,
    pub n_products: u32,
    leaves: Vec<Hash>,
    pub m_t: Hash,
    /// Stage-2 chain context, `None` for a stage-1 manifest.
    pub chain: Option<ChainCtx>,
}

impl TierManifest {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let (entries, n_products) = manifest::read_file(data)?;
        if n_products == 0 || n_products as usize > entries.len() {
            return None;
        }
        let leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        let (m_t, _) = rt_range_proof(&leaves, 0, 1);
        Some(Self { entries, n_products, leaves, m_t, chain: None })
    }

    /// The stage-2 manifest of `tier` with its chain context.
    pub fn for_chain(tier: u8) -> Option<Self> {
        let (ctx, _) = chain_ctx(tier)?;
        let m = Self::parse(Self::raw_chain(tier)?)?;
        Some(Self { chain: Some(ctx), ..m })
    }

    pub fn raw_chain(tier: u8) -> Option<&'static [u8]> {
        CHAIN_MANIFESTS.get(tier as usize).copied()
    }

    /// The manifest of `tier` under the attention rule, with its chain context.
    pub fn for_attn(tier: u8) -> Option<Self> {
        let (mut ctx, _) = chain_ctx(tier)?;
        (ctx.n_attn, ctx.n_recur, ctx.seq_heads) = *ATTN_SEQ.get(tier as usize)?;
        let m = Self::parse(Self::raw_attn(tier)?)?;
        Some(Self { chain: Some(ctx), ..m })
    }

    pub fn raw_attn(tier: u8) -> Option<&'static [u8]> {
        ATTN_MANIFESTS.get(tier as usize).copied()
    }

    /// Sequence layout for the engine: one `(layer, kind, slot, heads, head_dim)` row per
    /// attention or recurrent layer.
    pub fn seq_layers(&self) -> Vec<crate::llama_engine::SeqLayer> {
        let mut out = Vec::new();
        for (e, _) in &self.entries {
            let Chain::Layer(ll) = e.chain else { continue };
            if let Some(a) = ll.attn {
                out.push([ll.layer, if a.mla_rank > 0 { 2 } else { 1 }, a.slot, a.n_head_kv, a.head_dim]);
            } else if let Some(r) = ll.recur {
                out.push([ll.layer, if r.kind == manifest::RECUR_KDA { 4 } else { 3 }, r.slot, r.n_v_heads, r.head_dim]);
            }
        }
        out
    }

    /// Positions between the recurrent state checkpoints of this manifest.
    pub fn seq_interval(&self) -> u32 {
        self.entries
            .iter()
            .find_map(|(e, _)| match e.chain {
                Chain::Layer(ll) => ll.recur.map(|r| r.interval),
                _ => None,
            })
            .unwrap_or(DEFAULT_INTERVAL)
    }

    /// The embedded manifest of `tier`; `None` = no manifest.
    pub fn for_tier(tier: u8) -> Option<Self> {
        MANIFESTS.get(tier as usize).copied().flatten().and_then(Self::parse)
    }

    /// The manifest the node holds for a response accepted at `accepted_daa`: the embedded one
    /// under the prefix rule, else its entries before the embedding and norm.
    pub fn for_tier_at(tier: u8, accepted_daa: u64) -> Option<Self> {
        let m = Self::for_tier(tier)?;
        if prefix_active(accepted_daa) { Some(m) } else { Some(m.legacy()) }
    }

    /// This manifest without the prefix entries (`M_T` of the node's `TRACE_MANIFESTS_H14`).
    pub fn legacy(&self) -> Self {
        let n = self.entries.iter().position(|(e, _)| e.role == manifest::ROLE_EMBD).unwrap_or(self.entries.len());
        let entries: Vec<(Entry, Vec<u8>)> = self.entries[..n].to_vec();
        let leaves: Vec<Hash> = self.leaves[..n].to_vec();
        let (m_t, _) = rt_range_proof(&leaves, 0, 1);
        Self { entries, n_products: self.n_products, leaves, m_t, chain: None }
    }

    pub fn raw_for_tier(tier: u8) -> Option<&'static [u8]> {
        MANIFESTS.get(tier as usize).copied().flatten()
    }

    /// Leaves of one position: `X(r)` at `2r`, `Y(r)` at `2r + 1`, the stage-2 chain leaves,
    /// then the meta leaf.
    pub fn n_leaves(&self) -> usize {
        match &self.chain {
            Some(c) => chain::n_pos_leaves(self.n_products as u64, c.n_layers as u64, if c.n_used > 0 { c.n_layers as u64 } else { 0 }) as usize,
            None => 2 * self.n_products as usize + 1,
        }
    }

    /// Consensus data of this manifest over the model root of `index`.
    pub fn tier_ctx(&self, index: &WeightIndex) -> TierCtx {
        TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: self.m_t, n_entries: self.entries.len() as u64, n_products: self.n_products as u64 }
    }

    pub fn entry(&self, index: u64) -> &Entry {
        &self.entries[index as usize].0
    }

    pub fn entry_opening(&self, index: u64) -> EntryOpening {
        EntryOpening {
            index: index as u32,
            bytes: self.entries[index as usize].1.clone(),
            path: rt_range_proof(&self.leaves, index, 1).1,
        }
    }
}

/// Trace of one response: the leaves the engine wrote, their position roots and `trace_root`.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct TraceRecord {
    pub stage: u8,
    pub n_prompt: u32,
    pub n_pos: u32,
    pub ubatch: Vec<u32>,
    n_leaves: usize,
    leaves: Vec<Hash>,
    pos_roots: Vec<Hash>,
    pub trace_root: Hash,
    pub tokens_root: Option<Hash>,
}

fn meta_leaf(t: u32, n_ubatch: u32) -> Hash {
    let mut meta = [0u8; 8];
    meta[..4].copy_from_slice(&t.to_le_bytes());
    meta[4..].copy_from_slice(&n_ubatch.to_le_bytes());
    *blake3::hash(&meta).as_bytes()
}

impl TraceRecord {
    /// Leaves as written by the engine, position-major (`n_pos × n_leaves`). The fill-ins of
    /// leaves the engine never writes are applied here; any other leaf left unwritten is an error.
    pub fn new(m: &TierManifest, n_prompt: u32, ubatch: Vec<u32>, mut leaves: Vec<Hash>) -> Option<Self> {
        let nl = m.n_leaves();
        let n_pos = ubatch.len();
        if n_pos == 0 || n_prompt == 0 || n_prompt as usize > n_pos || leaves.len() != n_pos * nl || n_pos > u32::MAX as usize {
            return None;
        }
        let empty = *blake3::hash(&[]).as_bytes();
        let np = m.n_products as usize;
        for t in 0..n_pos {
            let pos = &mut leaves[t * nl..(t + 1) * nl];
            for (r, (e, _)) in m.entries.iter().enumerate().take(np) {
                if e.glu_form == manifest::FORM_PAIR_UP {
                    pos[2 * r + 1] = empty;
                }
                if e.role == manifest::ROLE_OUTPUT && t + 1 < n_prompt as usize {
                    pos[2 * r] = empty;
                    pos[2 * r + 1] = empty;
                }
            }
            pos[nl - 1] = meta_leaf(t as u32, ubatch[t]);
        }
        if leaves.contains(&[0u8; 32]) {
            return None;
        }
        let pos_roots: Vec<Hash> = (0..n_pos).map(|t| b3_root(&leaves[t * nl..(t + 1) * nl])).collect();
        let trace_root = b3_root(&pos_roots);
        Some(Self { stage: STAGE_1A, n_prompt, n_pos: n_pos as u32, ubatch, n_leaves: nl, leaves, pos_roots, trace_root, tokens_root: None })
    }

    /// Stage-2 record over a chain manifest: `tokens` are the fed tokens, one per position.
    pub fn new_chain(m: &TierManifest, n_prompt: u32, ubatch: Vec<u32>, mut leaves: Vec<Hash>, tokens: &[i32]) -> Option<Self> {
        let ctx = m.chain?;
        let nl = m.n_leaves();
        let n_pos = ubatch.len();
        if n_pos == 0 || n_prompt == 0 || n_prompt as usize > n_pos || leaves.len() != n_pos * nl || n_pos > u32::MAX as usize || tokens.len() != n_pos {
            return None;
        }
        let empty = *blake3::hash(&[]).as_bytes();
        let np = m.n_products as u64;
        let last = ctx.n_layers.checked_sub(1)? as u64;
        for t in 0..n_pos {
            let pos = &mut leaves[t * nl..(t + 1) * nl];
            let before_output = t + 1 < n_prompt as usize;
            for (r, (e, _)) in m.entries.iter().enumerate().take(np as usize) {
                if e.glu_form == manifest::FORM_PAIR_UP {
                    pos[2 * r + 1] = empty;
                }
                if e.role == manifest::ROLE_OUTPUT && before_output {
                    pos[2 * r] = empty;
                    pos[2 * r + 1] = empty;
                }
            }
            if ctx.trim_last && before_output {
                pos[chain::resid_leaf(np, last, Resid::Attn) as usize] = empty;
                pos[chain::resid_leaf(np, last, Resid::Out) as usize] = empty;
            }
            if ctx.n_used > 0 {
                for l in 0..ctx.n_layers as u64 {
                    let routed = matches!(m.entry(ctx.layer_base as u64 + l).chain, Chain::Layer(ll) if ll.moe.is_some());
                    if !routed || (ctx.trim_last && l == last && before_output) {
                        pos[chain::router_leaf(np, ctx.n_layers as u64, l) as usize] = empty;
                    }
                }
            }
            pos[nl - 1] = *blake3::hash(&chain::meta(t as u32, ubatch[t], tokens[t] as u32)).as_bytes();
        }
        if leaves.contains(&[0u8; 32]) {
            return None;
        }
        let pos_roots: Vec<Hash> = (0..n_pos).map(|t| b3_root(&leaves[t * nl..(t + 1) * nl])).collect();
        let trace_root = b3_root(&pos_roots);
        let generated: Vec<u8> = tokens[n_prompt as usize..].iter().flat_map(|&t| (t as u32).to_le_bytes()).collect();
        let tokens_root = Some(*blake3::hash(&generated).as_bytes());
        Some(Self { stage: STAGE_CHAIN, n_prompt, n_pos: n_pos as u32, ubatch, n_leaves: nl, leaves, pos_roots, trace_root, tokens_root })
    }

    pub fn commitment(&self) -> TraceCommitment {
        TraceCommitment { trace_root: self.trace_root, n_prompt: self.n_prompt, n_pos: self.n_pos, stage: self.stage, tokens_root: self.tokens_root }
    }

    /// Path of position `t` in the position tree.
    pub fn pos_path(&self, t: u64) -> Vec<Hash> {
        b3_path(&self.pos_roots, t as usize)
    }

    pub(crate) fn response_ctx(&self) -> ResponseCtx {
        ResponseCtx { trace_root: self.trace_root, n_prompt: self.n_prompt, n_pos: self.n_pos }
    }

    pub fn leaves_at(&self, t: u64) -> &[Hash] {
        let t = t as usize;
        &self.leaves[t * self.n_leaves..(t + 1) * self.n_leaves]
    }

    pub fn leaf(&self, t: u64, index: u32) -> Hash {
        self.leaves_at(t)[index as usize]
    }
}

/// Draw seed of a response accepted before chain block `seed_block_hash`.
pub fn trace_seed(seed_block_hash: &[u8; 32], response_txid: &[u8; 32]) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(TRACE_DRAW_DOMAIN);
    hasher.update(seed_block_hash);
    hasher.update(response_txid);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

pub fn audited(seed: &Hash) -> bool {
    draw::audited(seed)
}

/// Window to open the next attention phase once its child is drawn; matches the node.
pub fn attn_phase_window_daa(tier: u8, max_tokens: u32, n_prompt: u32, n_pos: u32) -> u64 {
    const BASE_DAA: u64 = 3_000;
    const MAX_TOKENS_CAP: u32 = 4_096;
    const MAX_PROMPT: u32 = 131_072;
    let per_token: u64 = match tier {
        0..=2 => 2,
        3 => 3,
        _ => 4,
    };
    BASE_DAA + n_prompt.min(MAX_PROMPT) as u64 / 10 + n_pos.saturating_sub(n_prompt).min(max_tokens.min(MAX_TOKENS_CAP)) as u64 * per_token
}
const ATTN_CHILD_DOMAIN: &[u8] = b"KeryxAttnChildV1";

/// Seed of the child drawn after attention phase `phase` of a response, as the node draws it.
pub fn attn_child_seed(seed_block_hash: &[u8; 32], response_txid: &[u8; 32], phase: u8) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(ATTN_CHILD_DOMAIN);
    hasher.update(seed_block_hash);
    hasher.update(response_txid);
    hasher.update(&[phase]);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

/// Sequence leaves of a response: their hashes and the root of their tree.
#[derive(Clone, BorshSerialize, BorshDeserialize)]
pub struct SeqTrace {
    pub leaves: Vec<Hash>,
    pub root: Hash,
}

impl SeqTrace {
    pub fn new(leaves: Vec<Hash>) -> Option<Self> {
        if leaves.is_empty() {
            return None;
        }
        let root = b3_root(&leaves);
        Some(Self { leaves, root })
    }
}

/// What the miner keeps of a traced response to open it later.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct ResponseTrace {
    pub record: TraceRecord,
    /// Prompt then generated tokens, as fed to the engine.
    pub tokens: Vec<i32>,
    /// Sequence leaves, under the attention rule.
    pub seq: Option<SeqTrace>,
}

impl ResponseTrace {
    /// The commitment a response carries: the position root alone, or with the sequence root.
    pub fn commitment(&self) -> TraceCommitment {
        let mut c = self.record.commitment();
        if let Some(s) = &self.seq {
            c.trace_root = keryx_verifiable_inference::seq::trace_root(&self.record.trace_root, &s.root);
        }
        c
    }

    /// A small synthetic trace for store tests.
    #[cfg(test)]
    pub fn test_sample() -> Self {
        let record = TraceRecord {
            stage: STAGE_1A,
            tokens_root: None,
            n_prompt: 2,
            n_pos: 3,
            ubatch: vec![2, 1],
            n_leaves: 3,
            leaves: (0..9u8).map(|i| [i; 32]).collect(),
            pos_roots: (0..3u8).map(|i| [0x10 + i; 32]).collect(),
            trace_root: [0xaa; 32],
        };
        Self { record, tokens: vec![5, 6, 7], seq: None }
    }
}

/// Opening window after the seed block, in DAA. MUST equal the node's `service_window_daa_at`.
pub fn opening_window_daa(tier: u8, max_tokens: u32) -> u64 {
    const BASE_DAA: u64 = 3_000;
    const MAX_TOKENS_CAP: u32 = 4_096;
    let per_token_daa: u64 = match tier {
        0..=2 => 2,
        3 => 3,
        _ => 4,
    };
    BASE_DAA + max_tokens.min(MAX_TOKENS_CAP) as u64 * per_token_daa
}

/// Work of `prompt_passes` passes over a prompt of `n_prompt` positions and of `n_gen` generated
/// tokens, at the per-position rates of the node's windows.
pub fn work_daa(tier: u8, n_prompt: u32, n_gen: u32, prompt_passes: u64) -> u64 {
    const MAX_TOKENS_CAP: u32 = 4_096;
    const MAX_PROMPT: u32 = 131_072;
    let per_token: u64 = match tier {
        0..=2 => 2,
        3 => 3,
        _ => 4,
    };
    prompt_passes * (n_prompt.min(MAX_PROMPT) as u64 / 10) + n_gen.min(MAX_TOKENS_CAP) as u64 * per_token
}

/// Largest opening deadline extension the node grants from the network backlog; matches the node.
pub const OPENING_BACKLOG_CAP_DAA: u64 = 36_000;

/// Replays an open audit costs its responder, in units of serving work; matches the node.
pub const OPEN_AUDIT_WORK_PASSES: u64 = 3;

/// First estimate of the opening deadline of an audit of `work` serving work seeded at
/// `seed_daa` with its own window ending at `own`, chained behind the open audits of the same
/// key; the node's deadline replaces it once known.
pub fn chained_deadline(own: u64, open_ends: &[u64], seed_daa: u64, work: u64) -> u64 {
    open_ends.iter().copied().filter(|e| *e >= seed_daa).max().map_or(own, |end| own.max(end + OPEN_AUDIT_WORK_PASSES * work))
}

/// Serving work of one request under the node's v3 window. MUST equal the node's `service_work_daa`.
pub fn service_work_daa(tier: u8, n_prompt: u32, generated: u32) -> u64 {
    const RATE: [u64; 5] = [90, 180, 85, 70, 35];
    const MAX_TOKENS_CAP: u32 = 4_096;
    let rate = RATE[(tier as usize).min(RATE.len() - 1)];
    2 * (n_prompt as u64).div_ceil(100) + (rate * generated.min(MAX_TOKENS_CAP) as u64).div_ceil(100)
}

/// Window the first opening of an attention audit adds for its prompt; matches the node.
pub fn attn_first_prompt_daa(n_prompt: u32) -> u64 {
    const REPLAYS: u64 = 3;
    const MAX_PROMPT: u32 = 131_072;
    REPLAYS * (n_prompt.min(MAX_PROMPT) as u64 / 10)
}

/// One drawn point: the manifest entry, the gate of its pair, the position and the output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointSpec {
    pub rank: u64,
    pub g_rank: u64,
    pub t: u64,
    pub row: u64,
    /// Rows per leaf: active experts, heads, or 1.
    pub n_used: u64,
    pub slot: u64,
    /// The output leaf ends with the expert ids of its slots.
    pub experts: bool,
}

impl PointSpec {
    pub fn leaf_x(&self) -> u32 {
        2 * self.g_rank as u32
    }

    pub fn leaf_y(&self) -> u32 {
        2 * self.g_rank as u32 + 1
    }
}

/// Point `i` of the draw of `seed`, as the node will verify it.
pub fn draw_point(m: &TierManifest, n_prompt: u32, n_pos: u32, seed: &Hash, i: u32) -> Option<PointSpec> {
    if n_pos == 0 || n_prompt == 0 || n_prompt > n_pos {
        return None;
    }
    let d = draw::point(seed, i);
    let rank = d[1] % m.n_products as u64;
    let e0 = m.entry(rank);
    let n_traced = n_pos as u64;
    let t = if e0.role == manifest::ROLE_OUTPUT {
        let first_pos = n_prompt as u64 - 1;
        first_pos + d[0] % (n_traced - first_pos)
    } else {
        d[0] % n_traced
    };
    let g_rank = if e0.glu_form == manifest::FORM_PAIR_UP { e0.partner as u64 } else { rank };
    let g = m.entry(g_rank);
    let heads = e0.role == manifest::ROLE_HEADS;
    let experts = g.n_exp > 1 && !heads;
    if g.n_exp != 1 && m.chain.is_none() {
        return None;
    }
    let n_used = if experts {
        m.chain?.n_used as u64
    } else if heads {
        g.n_exp as u64
    } else {
        1
    };
    let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
    if m_out == 0 || n_used == 0 {
        return None;
    }
    let slot = if n_used > 1 { (d[3] >> 32) % n_used } else { 0 };
    Some(PointSpec { rank, g_rank, t, row: d[2] % m_out, n_used, slot, experts })
}

/// Chunk `c` of `b` and its path to the BLAKE3 root of `b`.
fn chunk_with_path(b: &[u8], c: usize) -> Option<(Vec<u8>, Vec<Hash>)> {
    let n = b.len().div_ceil(B3_CHUNK).max(1);
    if c >= n {
        return None;
    }
    let path = if n == 1 {
        Vec::new()
    } else {
        let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
        b3_path(&cvs, c)
    };
    Some((b[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(b.len())].to_vec(), path))
}

/// Chunks `first..=last` of the resident model with their range proof against `R_T`.
pub fn range_proof(index: &WeightIndex, first: u64, last: u64) -> Option<ChunkRange> {
    if first > last || last >= index.n_chunks {
        return None;
    }
    let mut chunks = Vec::with_capacity(((last - first + 1) * 32) as usize);
    for c in first..=last {
        chunks.extend_from_slice(&index.read_chunk_bytes(c));
    }
    let path_first = index.merkle_path(first);
    let path_last = if last == first { path_first.clone() } else { index.merkle_path(last) };
    let (mut lo, mut hi, mut n) = (first, last, index.n_chunks);
    let mut proof = Vec::new();
    let mut level = 0;
    while n > 1 {
        if lo % 2 == 1 {
            proof.push(*path_first.get(level)?);
        }
        if hi % 2 == 0 && hi + 1 < n {
            proof.push(*path_last.get(level)?);
        }
        lo /= 2;
        hi /= 2;
        n = n.div_ceil(2);
        level += 1;
    }
    Some(ChunkRange { first_chunk: first, chunks, proof })
}

/// The opening of `p` from the captured input bytes `x` and output vector `y` of its gate.
pub fn build_point(m: &TierManifest, rec: &TraceRecord, p: &PointSpec, x: Vec<u8>, y: &[u8], index: &WeightIndex) -> Option<Point> {
    let g = m.entry(p.g_rank);
    let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
    let body = p.n_used * m_out * 4;
    let total = body + if p.experts { 4 * p.n_used } else { 0 };
    if y.len() as u64 != total || p.t >= rec.n_pos as u64 || p.slot >= p.n_used {
        return None;
    }
    let exp = if p.experts {
        let o = (body + 4 * p.slot) as usize;
        u32::from_le_bytes(y[o..o + 4].try_into().ok()?) as u64
    } else if p.n_used > 1 {
        p.slot
    } else {
        0
    };
    if exp >= g.n_exp as u64 {
        return None;
    }
    let row_range = |e: &Entry, row: u64| -> Option<ChunkRange> {
        let (first, last, _) = e.row_chunks(exp, row)?;
        range_proof(index, first, last)
    };
    let mut rows = vec![row_range(g, p.row)?];
    let partner = match g.glu_form {
        manifest::FORM_NONE => None,
        manifest::FORM_SPLIT => {
            rows.push(row_range(g, p.row + m_out)?);
            None
        }
        manifest::FORM_PAIR_GATE => {
            let up_rank = g.partner as u64;
            rows.push(row_range(m.entry(up_rank), p.row)?);
            Some(m.entry_opening(if p.rank == p.g_rank { up_rank } else { p.g_rank }))
        }
        _ => return None,
    };
    let bias = if g.bias != manifest::NONE {
        let b = m.entry(g.bias as u64);
        let first = b.chunk_base + p.row * 4 / 32;
        Some((m.entry_opening(g.bias as u64), range_proof(index, first, first)?))
    } else {
        None
    };
    let pos_leaves = rec.leaves_at(p.t);
    let nl = rec.n_leaves;
    let c = ((p.slot * m_out + p.row) * 4 / B3_CHUNK as u64) as usize;
    let (y_chunk, y_chunk_path) = chunk_with_path(y, c)?;
    Some(Point {
        n_ubatch: rec.ubatch[p.t as usize],
        pos_path: b3_path(&rec.pos_roots, p.t as usize),
        meta_path: b3_path(pos_leaves, nl - 1),
        entry: m.entry_opening(p.rank),
        partner,
        bias,
        rows,
        x,
        x_path: b3_path(pos_leaves, p.leaf_x() as usize),
        y_total_len: y.len() as u32,
        y_chunk_index: c as u32,
        y_chunk,
        y_chunk_path,
        y_path: b3_path(pos_leaves, p.leaf_y() as usize),
    })
}

/// The prefix position the draw of `seed` opens.
pub fn draw_prefix(ctx: &PrefixCtx, seed: &Hash) -> u64 {
    prefix::position(seed, ctx.offset)
}

/// The prefix point at position `t` from the captured input bytes `x` of the first product.
pub fn build_prefix_point(m: &TierManifest, rec: &TraceRecord, ctx: &PrefixCtx, request_hash: &[u8; 32], t: u64, x: Vec<u8>, index: &WeightIndex) -> Option<PrefixPoint> {
    if t >= rec.n_pos as u64 {
        return None;
    }
    let tok = prefix::expected_token(ctx, request_hash, t)?;
    let embd = m.entry(ctx.embd_entry as u64);
    let norm = m.entry(ctx.norm_entry as u64);
    let (first, last, _) = embd.row_chunks(0, tok as u64)?;
    let norm_last = norm.chunk_base + (4 * norm.k - 1) / 32;
    let pos_leaves = rec.leaves_at(t);
    Some(PrefixPoint {
        n_ubatch: rec.ubatch[t as usize],
        pos_path: b3_path(&rec.pos_roots, t as usize),
        meta_path: b3_path(pos_leaves, rec.n_leaves - 1),
        entry: m.entry_opening(0),
        x,
        x_path: b3_path(pos_leaves, 0),
        embd: (m.entry_opening(ctx.embd_entry as u64), range_proof(index, first, last)?),
        norm: (m.entry_opening(ctx.norm_entry as u64), range_proof(index, norm.chunk_base, norm_last)?),
    })
}

/// Encoded opening body, `None` above the consensus size cap.
pub fn opening_body(response_txid: [u8; 32], points: Vec<Point>, prefix: Option<PrefixPoint>) -> Option<Vec<u8>> {
    let body = Opening { response_txid, points, prefix, links: Vec::new(), experts: Vec::new(), resid: None, roots: None }.encode();
    (body.len() + 33 <= opening::MAX_BODY_LEN).then_some(body)
}

/// The residual point of a stage-2 draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidDraw {
    pub t: u64,
    pub layer: u64,
    pub kind: Resid,
    /// Product whose output is added.
    pub rank: u64,
    pub c: u64,
    /// Routed experts added with it.
    pub moe: Option<manifest::MoeLinks>,
}

impl ResidDraw {
    pub fn dst_leaf(&self, np: u64) -> u32 {
        chain::resid_leaf(np, self.layer, self.kind) as u32
    }

    pub fn src_leaf(&self, np: u64) -> Option<u32> {
        match self.kind {
            Resid::Attn if self.layer == 0 => None,
            Resid::Attn => Some(chain::resid_leaf(np, self.layer - 1, Resid::Out) as u32),
            Resid::Out => Some(chain::resid_leaf(np, self.layer, Resid::Attn) as u32),
        }
    }
}

/// Everything a stage-2 opening of `seed` opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainDraw {
    /// Each point with the leaf its input link reads at the point's position.
    pub points: Vec<(PointSpec, Option<u32>)>,
    pub prefix_t: u64,
    pub resid: ResidDraw,
    pub n_layers: u64,
}

fn link_leaf(m: &TierManifest, ctx: &ChainCtx, g_rank: u64) -> Option<Option<u32>> {
    let np = m.n_products as u64;
    let Chain::Product(pl) = m.entry(g_rank).chain else { return None };
    let layer = pl.layer as u64;
    Some(match pl.link {
        manifest::LINK_NONE => None,
        manifest::LINK_ATTN_IN if layer == 0 => None,
        manifest::LINK_ATTN_IN => Some(chain::resid_leaf(np, layer - 1, Resid::Out) as u32),
        manifest::LINK_FFN_IN => Some(chain::resid_leaf(np, layer, Resid::Attn) as u32),
        manifest::LINK_HEAD => Some(chain::resid_leaf(np, ctx.n_layers.checked_sub(1)? as u64, Resid::Out) as u32),
        manifest::LINK_GLU | manifest::LINK_DIRECT | manifest::LINK_HEAD_SLICE => Some(2 * pl.src + 1),
        _ => return None,
    })
}

/// The stage-2 draw of `seed` over `rec`, as the node will verify it.
pub fn draw_chain(m: &TierManifest, pctx: &PrefixCtx, rec: &TraceRecord, seed: &Hash) -> Option<ChainDraw> {
    let ctx = m.chain?;
    let mut points = Vec::with_capacity(POINTS as usize);
    for i in 0..POINTS {
        let p = draw_point(m, rec.n_prompt, rec.n_pos, seed, i)?;
        points.push((p, link_leaf(m, &ctx, p.g_rank)?));
    }
    let n_layers = ctx.n_layers as u64;
    if n_layers == 0 {
        return None;
    }
    let d = draw::point(seed, chain::RESID_POINT);
    let sel = chain::resid_site(d[1], 2 * n_layers);
    let (layer, kind) = (sel / 2, if sel.is_multiple_of(2) { Resid::Attn } else { Resid::Out });
    let t = chain::resid_position(&ctx, &rec.response_ctx(), layer, d[0]).ok()?;
    let ll = match m.entry(ctx.layer_base as u64 + layer).chain {
        Chain::Layer(ll) if ll.layer as u64 == layer => ll,
        _ => return None,
    };
    let moe = ll.moe.filter(|_| kind == Resid::Out && ll.post_ffn == manifest::NONE);
    let rank = if kind == Resid::Attn { ll.proj } else { ll.down } as u64;
    let dim = m.entry(rank).m;
    let c = d[2] % (4 * dim).div_ceil(B3_CHUNK as u64).max(1);
    Some(ChainDraw { points, prefix_t: draw_prefix(pctx, seed), resid: ResidDraw { t, layer, kind, rank, c, moe }, n_layers })
}

impl ChainDraw {
    /// The `(position, leaf)` captures the opening needs, deduplicated.
    pub fn captures(&self, np: u64) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let mut add = |t: u64, leaf: u32| {
            if !out.contains(&(t as u32, leaf)) {
                out.push((t as u32, leaf));
            }
        };
        for (p, link) in &self.points {
            add(p.t, p.leaf_x());
            add(p.t, p.leaf_y());
            if let Some(l) = link {
                add(p.t, *l);
            }
        }
        add(self.prefix_t, 0);
        let r = &self.resid;
        add(r.t, r.dst_leaf(np));
        if let Some(l) = r.src_leaf(np) {
            add(r.t, l);
        }
        add(r.t, 2 * r.rank as u32);
        add(r.t, 2 * r.rank as u32 + 1);
        if let Some(moe) = &r.moe {
            add(r.t, 2 * moe.down_exps);
            add(r.t, 2 * moe.down_exps + 1);
            add(r.t, chain::router_leaf(np, self.n_layers, r.layer) as u32);
        }
        out
    }
}

/// Largest byte size leaf `leaf` of `m` can hold at a position.
pub fn leaf_cap(m: &TierManifest, leaf: u32) -> usize {
    let np = m.n_products as usize;
    let leaf = leaf as usize;
    if leaf < 2 * np {
        let e = m.entry((leaf / 2) as u64);
        let n = match (e.role, e.n_exp) {
            (manifest::ROLE_HEADS, h) => h as u64,
            (_, 1) => 1,
            _ => m.chain.map_or(1, |c| c.n_used as u64).max(1),
        };
        let bytes = if leaf.is_multiple_of(2) { e.k.div_ceil(512) * 512 / 32 * 36 } else { 4 * e.m + 4 };
        return (bytes * n) as usize;
    }
    m.entries.iter().take(np).map(|(e, _)| 4 * e.m.max(e.k) as usize).max().unwrap_or(0)
}

pub(crate) fn bytes_range(m: &TierManifest, index: &WeightIndex, entry: u64, offset: u64, len: u64) -> Option<(EntryOpening, ChunkRange)> {
    let e = m.entry(entry);
    if len == 0 {
        return None;
    }
    let first = e.chunk_base + offset / 32;
    let last = e.chunk_base + (offset + len - 1) / 32;
    Some((m.entry_opening(entry), range_proof(index, first, last)?))
}

fn row_opening(m: &TierManifest, index: &WeightIndex, entry: u64, row: u64) -> Option<(EntryOpening, ChunkRange)> {
    let (first, last, _) = m.entry(entry).row_chunks(0, row)?;
    Some((m.entry_opening(entry), range_proof(index, first, last)?))
}

fn norm_fragment(m: &TierManifest, index: &WeightIndex, entry: u64, block: u64) -> Option<(EntryOpening, ChunkRange)> {
    let k = m.entry(entry).k;
    let (b0, n) = chain::norm_fragment(block, k);
    bytes_range(m, index, entry, 4 * b0 * 32, 4 * (n * 32).min(k - b0 * 32))
}

pub(crate) fn leaf_bytes(rec: &TraceRecord, t: u64, leaf: u32, bytes: &[u8]) -> LeafBytes {
    LeafBytes { bytes: bytes.to_vec(), leaf_path: b3_path(rec.leaves_at(t), leaf as usize) }
}

pub(crate) fn leaf_chunk(rec: &TraceRecord, t: u64, leaf: u32, b: &[u8], c: u64) -> Option<LeafChunk> {
    let (chunk, chunk_path) = chunk_with_path(b, c as usize)?;
    Some(LeafChunk { total_len: b.len() as u32, index: c as u32, chunk, chunk_path, leaf_path: b3_path(rec.leaves_at(t), leaf as usize) })
}

/// The expert ids that close an expert output leaf `y` of `n_used` slots.
fn expert_ids(y: &[u8], n_used: u64) -> Option<ExpertIds> {
    let body = y.len().checked_sub(4 * n_used as usize)?;
    if body % B3_CHUNK != 0 {
        return None;
    }
    let (chunk, chunk_path) = chunk_with_path(y, body / B3_CHUNK)?;
    Some(ExpertIds { chunk, chunk_path })
}

/// Captured leaf bytes, by `(position, leaf)`.
pub type Captures = BTreeMap<(u32, u32), Vec<u8>>;

/// Whether every capture hashes to its committed leaf.
pub fn captures_match(rec: &TraceRecord, caps: &Captures) -> bool {
    caps.iter().all(|(&(t, leaf), b)| (t as u64) < rec.n_pos as u64 && *blake3::hash(b).as_bytes() == rec.leaf(t as u64, leaf))
}

/// Stage-2 opening of `d` without its residual point, and its residual opening; `None` when a part
/// cannot be built.
#[allow(clippy::too_many_arguments)]
pub fn chain_opening(
    m: &TierManifest,
    pctx: &PrefixCtx,
    rec: &TraceRecord,
    tokens: &[i32],
    seed: &Hash,
    request_hash: &[u8; 32],
    response_txid: [u8; 32],
    d: &ChainDraw,
    caps: &Captures,
    index: &WeightIndex,
) -> Option<(Opening, ResidOpening)> {
    let ctx = m.chain?;
    let np = m.n_products as u64;
    let cap = |t: u64, leaf: u32| caps.get(&(t as u32, leaf)).map(Vec::as_slice);
    let tok = |t: u64| tokens.get(t as usize).map(|&v| v as u32);

    let mut points = Vec::with_capacity(d.points.len());
    let mut links = Vec::with_capacity(d.points.len());
    let mut experts = Vec::with_capacity(d.points.len());
    for (i, (p, link_leaf)) in d.points.iter().enumerate() {
        let i = i as u32;
        let y = cap(p.t, p.leaf_y())?;
        let point = build_point(m, rec, p, cap(p.t, p.leaf_x())?.to_vec(), y, index)?;
        experts.push(if p.experts { Some(expert_ids(y, p.n_used)?) } else { None });
        let Chain::Product(pl) = m.entry(p.g_rank).chain else { return None };
        let g = m.entry(p.g_rank);
        let layout = verify::input_layout(g.role, point.n_ubatch, g.wtype);
        let x_one = verify::x_len(g.k, layout) as usize;
        let xs = if point.x.len() == x_one { &point.x[..] } else { point.x.get(p.slot as usize * x_one..(p.slot as usize + 1) * x_one)? };
        let block = chain::link_block(seed, i, xs, layout, g.k);
        let link = match (pl.link, link_leaf) {
            (manifest::LINK_NONE, None) => LinkOpening::None,
            (manifest::LINK_ATTN_IN, None) => LinkOpening::Token {
                embd: row_opening(m, index, ctx.embd_entry as u64, tok(p.t)? as u64)?,
                norm: norm_fragment(m, index, pl.norm as u64, chain::link_block(seed, i, &point.x, layout, g.k)?)?,
            },
            (manifest::LINK_GLU, Some(l)) if p.experts => {
                let y = cap(p.t, *l)?;
                let per_slot = m.entry(p.g_rank).k * 4 / B3_CHUNK as u64;
                if per_slot == 0 {
                    return None;
                }
                let c = p.slot * per_slot + block? / chain::CHUNK_BLOCKS;
                LinkOpening::Source { src: m.entry_opening(pl.src as u64), y: leaf_chunk(rec, p.t, *l, y, c)?, ids: Some(expert_ids(y, p.n_used)?) }
            }
            (manifest::LINK_HEAD_SLICE, Some(l)) => {
                let y = cap(p.t, *l)?;
                let (g, src) = (m.entry(p.g_rank), m.entry(pl.src as u64));
                let n_head = g.n_exp as u64;
                if n_head < 2 || src.m % n_head != 0 || g.k < 32 {
                    return None;
                }
                let per_chunk = (B3_CHUNK / 4) as u64;
                let c = (p.slot * (src.m / n_head) + block? * 32) / per_chunk;
                LinkOpening::Source { src: m.entry_opening(pl.src as u64), y: leaf_chunk(rec, p.t, *l, y, c)?, ids: None }
            }
            (manifest::LINK_GLU | manifest::LINK_DIRECT, Some(l)) => {
                let y = cap(p.t, *l)?;
                let c = block? / chain::CHUNK_BLOCKS;
                LinkOpening::Source { src: m.entry_opening(pl.src as u64), y: leaf_chunk(rec, p.t, *l, y, c)?, ids: None }
            }
            (_, Some(l)) => LinkOpening::Norm { src: leaf_bytes(rec, p.t, *l, cap(p.t, *l)?), norm: norm_fragment(m, index, pl.norm as u64, block?)? },
            _ => return None,
        };
        points.push(point);
        links.push((tok(p.t)?, link));
    }

    let t = d.prefix_t;
    if t >= rec.n_pos as u64 {
        return None;
    }
    let pos_leaves = rec.leaves_at(t);
    let ptok = prefix::expected_token(pctx, request_hash, t)?;
    let px = cap(t, 0)?;
    let e0 = m.entry(0);
    let pblock = chain::link_block(seed, prefix::PREFIX_POINT, px, verify::input_layout(e0.role, rec.ubatch[t as usize], e0.wtype), e0.k)?;
    let prefix = PrefixPoint {
        n_ubatch: rec.ubatch[t as usize],
        pos_path: b3_path(&rec.pos_roots, t as usize),
        meta_path: b3_path(pos_leaves, rec.n_leaves - 1),
        entry: m.entry_opening(0),
        x: cap(t, 0)?.to_vec(),
        x_path: b3_path(pos_leaves, 0),
        embd: row_opening(m, index, pctx.embd_entry as u64, ptok as u64)?,
        norm: norm_fragment(m, index, pctx.norm_entry as u64, pblock)?,
    };

    let r = &d.resid;
    let ll = match m.entry(ctx.layer_base as u64 + r.layer).chain {
        Chain::Layer(ll) => ll,
        _ => return None,
    };
    let rtok = tok(r.t)?;
    let dst = cap(r.t, r.dst_leaf(np))?;
    let src = match r.src_leaf(np) {
        None => {
            let (e, range) = row_opening(m, index, ctx.embd_entry as u64, rtok as u64)?;
            ResidSrc::Token(e, range)
        }
        Some(l) => ResidSrc::Chunk(leaf_chunk(rec, r.t, l, cap(r.t, l)?, r.c)?),
    };
    let y_leaf = 2 * r.rank as u32 + 1;
    let y = cap(r.t, y_leaf)?;
    let post = if r.kind == Resid::Attn { ll.post_attn } else { ll.post_ffn };
    let add = if let Some(moe) = &r.moe {
        let n_used = ctx.n_used as u64;
        let dim = m.entry(r.rank).m;
        let per_slot = dim * 4 / B3_CHUNK as u64;
        let router = m.entry(moe.router as u64);
        let n_exp = router.m;
        let down_leaf = 2 * moe.down_exps + 1;
        let down_y = cap(r.t, down_leaf)?;
        let slots = (0..n_used).map(|sl| leaf_chunk(rec, r.t, down_leaf, down_y, sl * per_slot + r.c)).collect::<Option<Vec<_>>>()?;
        let router_leaf = chain::router_leaf(np, ctx.n_layers as u64, r.layer) as u32;
        let row = draw::point(seed, chain::RESID_POINT)[3] % n_exp;
        let attn_leaf = chain::resid_leaf(np, r.layer, Resid::Attn) as u32;
        ResidAdd::Moe(Box::new(MoeAdd {
            scores: leaf_bytes(rec, r.t, router_leaf, cap(r.t, router_leaf)?),
            bias: bytes_range(m, index, moe.probs_bias as u64, 0, 4 * n_exp)?,
            ids: expert_ids(down_y, n_used)?,
            slots,
            shexp: leaf_chunk(rec, r.t, y_leaf, y, r.c)?,
            router_row: bytes_range(m, index, moe.router as u64, row * 4 * dim, 4 * dim)?,
            resid: leaf_bytes(rec, r.t, attn_leaf, cap(r.t, attn_leaf)?),
            norm: bytes_range(m, index, moe.ffn_norm as u64, 0, 4 * dim)?,
        }))
    } else if post == manifest::NONE {
        ResidAdd::Chunk(leaf_chunk(rec, r.t, y_leaf, y, r.c)?)
    } else {
        let off = r.c * B3_CHUNK as u64 / 4;
        let n = (dst.len() as u64 / 4).checked_sub(off)?.min(B3_CHUNK as u64 / 4);
        ResidAdd::Normed { y: leaf_bytes(rec, r.t, y_leaf, y), norm: bytes_range(m, index, post as u64, 4 * off, 4 * n)? }
    };
    let rpos = rec.leaves_at(r.t);
    let resid = ResidPoint {
        n_ubatch: rec.ubatch[r.t as usize],
        tok: rtok,
        pos_path: b3_path(&rec.pos_roots, r.t as usize),
        meta_path: b3_path(rpos, rec.n_leaves - 1),
        layer: m.entry_opening(ctx.layer_base as u64 + r.layer),
        product: m.entry_opening(r.rank),
        dst: leaf_chunk(rec, r.t, r.dst_leaf(np), dst, r.c)?,
        src,
        add,
    };

    let dr = draw::point(seed, chain::RESID_POINT);
    let (pe, per) = (m.entry(r.rank), (B3_CHUNK / 4) as u64);
    let lo = r.c * per;
    let row = lo + (dr[3] >> 32) % per.min(pe.m.checked_sub(lo).filter(|&n| n > 0)?);
    let feed = FeedRow {
        x: cap(r.t, 2 * r.rank as u32)?.to_vec(),
        x_path: b3_path(rpos, 2 * r.rank as usize),
        row: row_opening(m, index, r.rank, row)?.1,
        bias: if pe.bias == manifest::NONE { None } else { Some(bytes_range(m, index, pe.bias as u64, row * 4, 4)?) },
    };
    let expert = match &r.moe {
        Some(moe) => {
            let n_used = ctx.n_used as u64;
            let down_y = cap(r.t, 2 * moe.down_exps + 1)?;
            let slot = (dr[3] >> 48) % n_used.max(1);
            let s = down_y.len().checked_sub(4 * n_used as usize)? + 4 * slot as usize;
            let id = u32::from_le_bytes(down_y.get(s..s + 4)?.try_into().ok()?) as u64;
            let (first, last, _) = m.entry(moe.down_exps as u64).row_chunks(id, row)?;
            Some(ExpertRow {
                entry: m.entry_opening(moe.down_exps as u64),
                x: cap(r.t, 2 * moe.down_exps)?.to_vec(),
                x_path: b3_path(rpos, 2 * moe.down_exps as usize),
                row: range_proof(index, first, last)?,
            })
        }
        None => None,
    };
    let n = points.len();
    debug_assert_eq!(experts.len(), n);
    let opening = Opening { response_txid, points, prefix: Some(prefix), links, experts, resid: None, roots: None };
    Some((opening, ResidOpening { response_txid, roots: None, resid, feed, expert }))
}

/// Checks the trace and residual pieces of a stage-2 opening as the node does; `Err` names the
/// failing part.
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_opening(
    tier: &TierCtx,
    ctx: &ChainCtx,
    pctx: &PrefixCtx,
    rec: &TraceRecord,
    seed: &Hash,
    request_hash: &[u8; 32],
    body: &[u8],
    resid_body: &[u8],
) -> Result<(), String> {
    let o = Opening::decode(body).ok_or("malformed body")?;
    let r = ResidOpening::decode(resid_body).ok_or("malformed residual body")?;
    let resp = rec.response_ctx();
    let Some(pp) = &o.prefix else { return Err("stage-2 part missing".into()) };
    if o.resid.is_some() {
        return Err("residual point in the trace piece".into());
    }
    if r.response_txid != o.response_txid || r.roots != o.roots {
        return Err("residual piece of another opening".into());
    }
    if o.points.len() != POINTS as usize || o.links.len() != o.points.len() || o.experts.len() != o.points.len() {
        return Err("point count".into());
    }
    for (i, (p, (tok, link))) in o.points.iter().zip(&o.links).enumerate() {
        chain::verify_chain_point(tier, ctx, &resp, seed, i as u32, p, *tok, link, o.experts[i].as_ref()).map_err(|e| format!("point {i}: {e:?}"))?;
    }
    chain::verify_chain_prefix_point(tier, ctx, pctx, &resp, seed, request_hash, pp).map_err(|e| format!("prefix point: {e:?}"))?;
    chain::verify_resid_opening(tier, ctx, &resp, seed, &r).map_err(|e| format!("residual point: {e:?}"))
}

/// The message an opening is signed over.
pub fn opening_signature_message(body: &[u8]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(OPENING_SIG_DOMAIN);
    hasher.update(body);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadlines_chain_behind_the_open_audits_of_the_key() {
        let work = service_work_daa(0, 0, 4_096);
        assert_eq!(chained_deadline(5_000, &[], 100, work), 5_000);
        assert_eq!(chained_deadline(5_000, &[90], 100, work), 5_000, "a closed or missed audit no longer counts");
        assert_eq!(chained_deadline(5_000, &[4_000], 100, work), 4_000 + OPEN_AUDIT_WORK_PASSES * work);
        let far = 5_000 + OPENING_BACKLOG_CAP_DAA;
        assert_eq!(chained_deadline(5_000, &[far], 100, work), far + OPEN_AUDIT_WORK_PASSES * work, "the node, not the miner, bounds the chain");
    }

    #[test]
    fn work_follows_the_node_rates() {
        assert_eq!(work_daa(0, 1_000, 100, 3), 300 + 200);
        assert_eq!(work_daa(4, 1_000, 100, 1), 100 + 400);
        assert_eq!(work_daa(3, 1_000_000, 10_000, 2), 2 * 13_107 + 4_096 * 3);
        assert_eq!(work_daa(u8::MAX, 0, 10, 1), 40);
    }

    #[test]
    fn the_attention_phase_window_scales_with_the_replay() {
        assert_eq!(attn_phase_window_daa(0, 0, 0, 0), 3_000);
        assert_eq!(attn_phase_window_daa(2, 1_000, 30_000, 30_900), 7_800);
        assert_eq!(attn_phase_window_daa(4, 8_000, 0, 5_000), 19_384);
        assert_eq!(attn_phase_window_daa(3, 100, 2_000, 9_000), 3_500);
        assert_eq!(attn_phase_window_daa(4, u32::MAX, u32::MAX, u32::MAX), 16_107);
        assert_eq!(attn_first_prompt_daa(30_000), 9_000);
        assert_eq!(attn_first_prompt_daa(u32::MAX), 39_321);
    }

    #[test]
    fn the_child_seed_matches_the_node() {
        let seed = attn_child_seed(&[1u8; 32], &[2u8; 32], 1);
        assert_eq!(hex::encode(seed), "347b8cd184f5fdf5d3ef1828b99bb28d859b019df6b09ece1ec8167ea62e534b");
    }

    #[test]
    fn every_drawable_site_of_the_attention_manifests_is_supported() {
        use keryx_verifiable_inference::support::check_manifest;
        let entries = |m: &TierManifest| m.entries.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>();
        for tier in 0..5u8 {
            let m = TierManifest::for_attn(tier).unwrap();
            check_manifest(&entries(&m), &m.chain.unwrap(), 131_072).unwrap_or_else(|e| panic!("tier {tier}: {e}"));
        }
        let m = TierManifest::for_attn(2).unwrap();
        let mut bad = entries(&m);
        let i = bad.iter().position(|e| matches!(e.chain, Chain::Layer(ll) if ll.attn.is_some())).unwrap();
        if let Chain::Layer(ll) = &mut bad[i].chain {
            ll.attn.as_mut().unwrap().softcap = 1;
        }
        assert!(check_manifest(&bad, &m.chain.unwrap(), 131_072).is_err());
    }

    #[test]
    fn attention_manifests_match_the_node() {
        let roots = [
            "2f39af56c0e32524a44915e1ca6ae8db4dd2fc127dbf1c99ba9a2b8cac8602ee",
            "3bbe87ed68f0eb59e2b21be9996ff328d56447dfbe5b8b2e98cca6872427cb66",
            "0bf8505d055517e996744319017647752ab1a5bfc6a393b36d667414d48fbe69",
            "9179f634df5dcc732b7132ea54b09c98fa00c6483202a76a50f064a1d409381d",
            "95dd7a2f9266ed5961130e33ab75d92a035774dad5f74277b83b45a5fac72ad3",
        ];
        for (tier, root) in roots.iter().enumerate() {
            let m = TierManifest::for_attn(tier as u8).unwrap();
            let hex: String = m.m_t.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(&hex, root, "tier {tier}");
            let c = m.chain.unwrap();
            let layers = m.seq_layers();
            assert_eq!(layers.len() as u32, c.n_attn + c.n_recur, "tier {tier}");
            assert!(layers.iter().all(|l| l[3] > 0 && l[3] <= c.seq_heads && l[4] % 32 == 0));
            let attn = layers.iter().filter(|l| l[1] <= 2).count() as u32;
            assert_eq!(attn, c.n_attn);
            assert_eq!(m.seq_interval(), if c.n_recur > 0 { 32 } else { DEFAULT_INTERVAL });
        }
    }
    use keryx_verifiable_inference::merkle::{rt_range_proof, rt_verify_range};
    use keryx_verifiable_inference::verify::{verify_point, ResponseCtx, TierCtx};

    /// `M_T`, entries and products pinned by the node (`TRACE_MANIFESTS_PREFIX`).
    const PINNED: [(&str, u64, u64); 4] = [
        ("9b802ae5f4cafe073bd97b72222a77f44eb597491ab07235180c5ef8783f62bc", 251, 249),
        ("5d678ffbd6c4164cc8d4006168adc276541fdf3f181ea8a89cac70231e7d7d1a", 363, 241),
        ("c09c80b1815b93dba8b430f2648e50be87137de945f96ca82e4cfa2084b2a808", 331, 329),
        ("98dc87dc3cf1297232e89e58bfae501c1172ba05626807a086ca38cdaffb4c47", 499, 497),
    ];

    /// `M_T` and entries pinned by the node before the prefix rule (`TRACE_MANIFESTS_H14`).
    const PINNED_LEGACY: [(&str, u64); 4] = [
        ("4c6212d16c7fc573b372e57e60176190b63ed3637a7bdfa93fbf6a7ca2bfd492", 249),
        ("c586bd7c3ef5aeadd669a48e28beb135a9df43890bf4b30a26072cdebba1b045", 361),
        ("1550e2f3a709f1653663111196edeb8088a0c50873687e70ef77bbbcdaa949c8", 329),
        ("5ff3b42ff04356106ae7a56512321fae0f37bde171495b774bf944e4d0235c33", 497),
    ];

    /// `M_T`, entries and products of the stage-2 manifests pinned by the node (`TRACE_CHAINS_H15`).
    const PINNED_CHAIN: [(&str, u64, u64); 5] = [
        ("9f1ea28681bfa10a6afe4d68cbe27ebea8656f76369af1bfa2adbf71e10374d0", 347, 249),
        ("0431fa87e1a48b48151370c3fc7604de9b72461c9ac82881beeebb11febbfdeb", 563, 241),
        ("4463e82828cc458ec86753a2c792d8f0da50c0d13d9758b9a49ad030dc2bef0f", 571, 329),
        ("bc417110ea8e18db4fe6fe656bb6c7a47375585c86f9047ab7e40c28e11d0800", 691, 497),
        ("c846b9cf4d7fd0119653d517c11cec3ac096013ed07268839db6949f0de7db93", 530, 375),
    ];

    #[test]
    fn chain_manifests_match_the_node_pins() {
        for (tier, (m_t, n_entries, n_products)) in PINNED_CHAIN.iter().enumerate() {
            let (entries, np) = chain_manifest(tier as u8).expect("chain manifest");
            let leaves: Vec<_> = entries.iter().map(|(_, b)| keryx_verifiable_inference::manifest::Entry::leaf(b)).collect();
            assert_eq!(hex::encode(rt_range_proof(&leaves, 0, 1).0), *m_t, "tier {tier}");
            assert_eq!((entries.len() as u64, np as u64), (*n_entries, *n_products), "tier {tier}");
            let (ctx, prefix) = chain_ctx(tier as u8).expect("chain ctx");
            assert_eq!(entries[prefix.embd_entry as usize].0.role, keryx_verifiable_inference::manifest::ROLE_EMBD, "tier {tier}");
            assert_eq!(entries[prefix.norm_entry as usize].0.role, keryx_verifiable_inference::manifest::ROLE_NORM, "tier {tier}");
            assert_eq!(entries[ctx.layer_base as usize].0.role, keryx_verifiable_inference::manifest::ROLE_LAYER, "tier {tier}");
        }
    }

    #[test]
    fn embedded_manifests_match_the_node_pins() {
        for (tier, (m_t, n_entries, n_products)) in PINNED.iter().enumerate() {
            let m = TierManifest::for_tier(tier as u8).expect("manifest");
            assert_eq!(hex::encode(m.m_t), *m_t, "tier {tier}");
            assert_eq!(m.entries.len() as u64, *n_entries, "tier {tier}");
            assert_eq!(m.n_products as u64, *n_products, "tier {tier}");
            for i in 0..m.entries.len() as u64 {
                let eo = m.entry_opening(i);
                assert!(manifest::verify_entry(&m.m_t, *n_entries, i, &eo.bytes, &eo.path));
            }
            let ctx = prefix_ctx(tier as u8).expect("prefix context");
            let (embd, norm) = (m.entry(ctx.embd_entry as u64), m.entry(ctx.norm_entry as u64));
            assert_eq!((embd.role, norm.role), (manifest::ROLE_EMBD, manifest::ROLE_NORM), "tier {tier}");
            assert_eq!((embd.k, norm.k, norm.wtype), (m.entry(0).k, m.entry(0).k, 0), "tier {tier}");
            assert_eq!(m.entry(0).role, manifest::ROLE_PRODUCT, "tier {tier}");
            assert_eq!(ctx.table.iter().collect::<std::collections::BTreeSet<_>>().len(), 256, "tier {tier}");

            let (m_t, n_entries) = PINNED_LEGACY[tier];
            let legacy = m.legacy();
            assert_eq!(hex::encode(legacy.m_t), m_t, "tier {tier}");
            assert_eq!(legacy.entries.len() as u64, n_entries, "tier {tier}");
            assert_eq!(legacy.n_products, m.n_products, "tier {tier}");
            assert!(legacy.entries.iter().all(|(e, _)| e.role != manifest::ROLE_EMBD && e.role != manifest::ROLE_NORM));
        }
        assert!(TierManifest::for_tier(4).is_none());
        assert!(TierManifest::for_tier(9).is_none());
        assert!(prefix_ctx(4).is_none());
    }

    #[test]
    fn prefix_tokens_follow_the_tier_table() {
        let ctx = prefix_ctx(0).unwrap();
        let mut rh = [0u8; 32];
        rh[0] = 0x41;
        rh[31] = 0xff;
        let toks = prefix_tokens(ctx, &rh);
        assert_eq!(toks.len(), prefix::PREFIX_LEN as usize);
        assert_eq!(toks[0], ctx.table[0x41] as i32);
        assert_eq!(toks[31], ctx.table[0xff] as i32);
        assert_eq!(toks[1], ctx.table[0] as i32);
        for t in 0..64u8 {
            let seed = [t; 32];
            let p = draw_prefix(ctx, &seed);
            assert!((ctx.offset as u64..ctx.offset as u64 + 32).contains(&p));
            assert_eq!(prefix::expected_token(ctx, &rh, p), Some(toks[(p - ctx.offset as u64) as usize] as u32));
        }
    }

    #[test]
    fn range_proofs_match_the_reference_tree() {
        for n in [64u64, 65, 127, 1000, 4097, 70_001] {
            let data: Vec<u8> = (0..n * 32).map(|i| (i * 31 + 7) as u8).collect();
            let index = crate::pom::index_from_ram(data.clone());
            let leaves: Vec<Hash> = data.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let (root, _) = rt_range_proof(&leaves, 0, 1);
            assert_eq!(root, index.r_t, "n {n}");
            for first in [0, 1, n / 2, n.saturating_sub(2), n - 1] {
                for count in 1..=(n - first).min(5) {
                    let r = range_proof(&index, first, first + count - 1).unwrap();
                    let (_, expected) = rt_range_proof(&leaves, first, count);
                    assert_eq!(r.proof, expected, "n {n} first {first} count {count}");
                    let got: Vec<Hash> = r.chunks.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
                    assert!(rt_verify_range(&index.r_t, n, first, &got, &r.proof));
                }
            }
            assert!(range_proof(&index, n - 1, n).is_none());
        }
    }

    #[test]
    fn record_fills_in_the_canonical_leaves() {
        let m = TierManifest::for_tier(0).unwrap();
        let nl = m.n_leaves();
        let n_pos = 5;
        let mut leaves = vec![[1u8; 32]; n_pos * nl];
        // The engine never writes Y(up) of a pair, nor the head before the last prompt token.
        let empty = *blake3::hash(&[]).as_bytes();
        let rec = TraceRecord::new(&m, 3, vec![3, 3, 3, 1, 1], leaves.clone()).unwrap();
        assert_eq!(rec.n_pos, 5);
        let up = m.entries.iter().position(|(e, _)| e.glu_form == manifest::FORM_PAIR_UP).unwrap();
        let head = m.entries.iter().position(|(e, _)| e.role == manifest::ROLE_OUTPUT).unwrap();
        assert_eq!(rec.leaf(0, 2 * up as u32 + 1), empty);
        assert_eq!(rec.leaf(0, 2 * head as u32), empty);
        assert_eq!(rec.leaf(2, 2 * head as u32), [1u8; 32]);
        assert_eq!(rec.leaf(4, nl as u32 - 1), meta_leaf(4, 1));
        let c = rec.commitment();
        assert_eq!((c.n_prompt, c.n_pos, c.stage), (3, 5, STAGE_1A));
        // An unwritten product leaf is an incomplete trace.
        leaves[nl + 2] = [0u8; 32];
        assert!(TraceRecord::new(&m, 3, vec![3, 3, 3, 1, 1], leaves.clone()).is_none());
        assert!(TraceRecord::new(&m, 0, vec![1; 5], leaves.clone()).is_none());
        assert!(TraceRecord::new(&m, 6, vec![1; 5], leaves).is_none());
    }

    #[test]
    fn chain_record_fills_in_the_canonical_leaves() {
        let m = TierManifest::for_chain(1).unwrap();
        let ctx = m.chain.unwrap();
        assert!(ctx.trim_last);
        let nl = m.n_leaves();
        assert_eq!(nl, 2 * m.n_products as usize + 2 * ctx.n_layers as usize + 1);
        let (n_pos, n_prompt) = (5usize, 3u32);
        let mut leaves = vec![[1u8; 32]; n_pos * nl];
        let tokens = [10, 11, 12, 13, 14];
        let ubatch = vec![3, 3, 3, 1, 1];
        let empty = *blake3::hash(&[]).as_bytes();
        let rec = TraceRecord::new_chain(&m, n_prompt, ubatch.clone(), leaves.clone(), &tokens).unwrap();
        let np = m.n_products as u64;
        let last = ctx.n_layers as u64 - 1;
        let attn = chain::resid_leaf(np, last, Resid::Attn) as u32;
        let out = chain::resid_leaf(np, last, Resid::Out) as u32;
        assert_eq!((rec.leaf(0, attn), rec.leaf(1, out)), (empty, empty));
        assert_eq!((rec.leaf(2, attn), rec.leaf(4, out)), ([1u8; 32], [1u8; 32]));
        assert_eq!(rec.leaf(0, chain::resid_leaf(np, 0, Resid::Out) as u32), [1u8; 32]);
        assert_eq!(rec.leaf(3, nl as u32 - 1), *blake3::hash(&chain::meta(3, 1, 13)).as_bytes());
        let c = rec.commitment();
        let generated: Vec<u8> = [13u32, 14].iter().flat_map(|t| t.to_le_bytes()).collect();
        assert_eq!((c.stage, c.tokens_root), (STAGE_CHAIN, Some(*blake3::hash(&generated).as_bytes())));
        assert_eq!(c.to_bytes().len(), keryx_inference::AI_RESPONSE_TRACE_TOKENS_LEN);
        // A residual the engine did not write is an incomplete trace.
        leaves[nl + chain::resid_leaf(np, 2, Resid::Attn) as usize] = [0u8; 32];
        assert!(TraceRecord::new_chain(&m, n_prompt, ubatch.clone(), leaves.clone(), &tokens).is_none());
        assert!(TraceRecord::new_chain(&m, n_prompt, ubatch.clone(), vec![[1u8; 32]; n_pos * nl], &tokens[..4]).is_none());
        assert!(TraceRecord::new_chain(&TierManifest::for_tier(1).unwrap(), n_prompt, ubatch, vec![[1u8; 32]; n_pos * nl], &tokens).is_none());
    }

    #[test]
    fn chain_draw_opens_the_leaves_the_verifier_reads() {
        let (mut experts, mut heads, mut moe) = (0, 0, 0);
        for tier in 0..5u8 {
            let m = TierManifest::for_chain(tier).unwrap();
            let ctx = m.chain.unwrap();
            let (_, pctx) = chain_ctx(tier).unwrap();
            let nl = m.n_leaves();
            let (n_pos, n_prompt) = (300usize, 260u32);
            let tokens: Vec<i32> = (0..n_pos as i32).collect();
            let rec = TraceRecord::new_chain(&m, n_prompt, vec![1; n_pos], vec![[1u8; 32]; n_pos * nl], &tokens).unwrap();
            let np = m.n_products as u64;
            for counter in 0..256u64 {
                let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
                let d = draw_chain(&m, &pctx, &rec, &seed).unwrap_or_else(|| panic!("tier {tier} seed {counter}"));
                let caps = d.captures(np);
                assert!(caps.len() <= crate::llama_engine::REPLAY_MAX_CAPTURES);
                assert!(caps.iter().all(|&(t, l)| (t as u64) < rec.n_pos as u64 && (l as usize) < nl - 1));
                assert!(caps.iter().all(|&(_, l)| leaf_cap(&m, l) > 0 && leaf_cap(&m, l) <= 4 << 20));
                let r = d.resid;
                if ctx.trim_last && r.layer + 1 == ctx.n_layers as u64 {
                    assert!(r.t + 1 >= n_prompt as u64);
                }
                let dim = m.entry(r.rank).m;
                assert!(r.c < (4 * dim).div_ceil(B3_CHUNK as u64));
                for (p, link) in &d.points {
                    let Chain::Product(pl) = m.entry(p.g_rank).chain else { panic!("tier {tier}: product without link") };
                    assert_eq!(link.is_none(), pl.link == manifest::LINK_NONE || (pl.link == manifest::LINK_ATTN_IN && pl.layer == 0));
                    assert!(p.slot < p.n_used);
                    experts += p.experts as u32;
                    heads += (m.entry(p.rank).role == manifest::ROLE_HEADS) as u32;
                }
                if r.moe.is_some() {
                    assert_eq!(r.kind, Resid::Out);
                    assert!(caps.contains(&(r.t as u32, chain::router_leaf(np, ctx.n_layers as u64, r.layer) as u32)));
                    assert!(caps.contains(&(r.t as u32, chain::resid_leaf(np, r.layer, Resid::Attn) as u32)));
                    moe += 1;
                }
            }
        }
        assert!(experts > 0 && heads > 0 && moe > 0, "{experts} {heads} {moe}");
    }

    #[test]
    fn draw_matches_the_verifier() {
        let m = TierManifest::for_tier(1).unwrap();
        let seed = trace_seed(&[7u8; 32], &[9u8; 32]);
        let mut seen_out = false;
        for i in 0..64u32 {
            let s = blake3::hash(&i.to_le_bytes());
            let p = draw_point(&m, 10, 40, s.as_bytes(), 0).unwrap();
            let e = m.entry(p.rank);
            if e.role == manifest::ROLE_OUTPUT {
                assert!(p.t >= 9 && p.t < 40);
                seen_out = true;
            } else {
                assert!(p.t < 40);
            }
            let g = m.entry(p.g_rank);
            let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
            assert!(p.row < m_out);
            if e.glu_form == manifest::FORM_PAIR_UP {
                assert_eq!(p.g_rank, e.partner as u64);
            } else {
                assert_eq!(p.g_rank, p.rank);
            }
        }
        assert!(seen_out, "the head is drawn within 64 seeds");
        assert_eq!(audited(&seed), u16::from_le_bytes([seed[0], seed[1]]) < draw::AUDIT_THRESHOLD);
        assert!(draw_point(&m, 0, 40, &seed, 0).is_none());
        assert!(draw_point(&m, 41, 40, &seed, 0).is_none());
    }

    /// Bench dump of a traced response (`KXQ8C001`) and one capture, as written by `q8canon.cu`.
    fn read_dump(path: &std::path::Path) -> (u32, Vec<u32>, Vec<Hash>) {
        let b = std::fs::read(path).unwrap();
        assert_eq!(&b[..8], b"KXQ8C001");
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (n_pos, n_prompt) = (u(8) as usize, u(12));
        let ubatch = (0..n_pos).map(|i| u(24 + 4 * i)).collect();
        let o = 24 + 4 * n_pos;
        let leaves = b[o..].chunks_exact(32).map(|c| c.try_into().unwrap()).collect();
        (n_prompt, ubatch, leaves)
    }

    fn read_capture(path: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
        let cap = std::fs::read(path).unwrap();
        let xl = u32::from_le_bytes(cap[..4].try_into().unwrap()) as usize;
        let yl = u32::from_le_bytes(cap[4 + xl..8 + xl].try_into().unwrap()) as usize;
        (cap[4..4 + xl].to_vec(), cap[8 + xl..8 + xl + yl].to_vec())
    }

    /// Real openings from the bench traces, verified as the node would. Needs the bench
    /// directory (`KERYX_VI_BENCH`) and the tier GGUFs with their `pom-tree.bin`
    /// (`KERYX_VI_MODELS`); skipped otherwise.
    #[test]
    fn bench_openings_verify() {
        let (Ok(bench), Ok(models)) = (std::env::var("KERYX_VI_BENCH"), std::env::var("KERYX_VI_MODELS")) else {
            eprintln!("skipped: set KERYX_VI_BENCH and KERYX_VI_MODELS");
            return;
        };
        let bench = std::path::Path::new(&bench).join("spec/q8c");
        let cases: [(u8, &str, &str, &str, &str); 3] = [
            (1, "GLM-4-9B-0414", "glm-run1.bin", "glm-seeds.txt", "glmcap"),
            (2, "Gemma-4-12B-abliterated", "Gemma-4-12B-abliterated-run.bin", "Gemma-4-12B-abliterated-seeds.txt", "Gemma-4-12B-abliterated-cap"),
            (3, "Qwen3.8-27B", "Qwen3.8-27B-run.bin", "Qwen3.8-27B-seeds.txt", "Qwen3.8-27B-cap"),
        ];
        for (tier, model, run, seeds, cap_prefix) in cases {
            let gguf = std::path::Path::new(&models).join(model).join("model.gguf");
            if !gguf.exists() || !bench.join(run).exists() {
                eprintln!("skipped tier {tier}: files missing");
                continue;
            }
            let spec = crate::models::spec_for_gguf(gguf.to_str().unwrap()).expect("registry model");
            let index = WeightIndex::build_from_gguf(gguf.to_str().unwrap(), spec.model_id).unwrap();
            let m = TierManifest::for_tier(tier).unwrap();
            let (n_prompt, ubatch, leaves) = read_dump(&bench.join(run));
            let rec = TraceRecord::new(&m, n_prompt, ubatch, leaves).expect("complete trace");
            let tier_ctx = TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: m.m_t, n_entries: m.entries.len() as u64, n_products: m.n_products as u64 };
            let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
            let mut n_ok = 0;
            for line in std::fs::read_to_string(bench.join(seeds)).unwrap().lines() {
                let counter: u64 = line.split_whitespace().next().unwrap().parse().unwrap();
                let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
                let p = draw_point(&m, rec.n_prompt, rec.n_pos, &seed, 0).unwrap();
                let (x, y) = read_capture(&bench.join(format!("{cap_prefix}{counter}.bin")));
                assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(p.t, p.leaf_x()), "{model} seed {counter}: X leaf");
                assert_eq!(*blake3::hash(&y).as_bytes(), rec.leaf(p.t, p.leaf_y()), "{model} seed {counter}: Y leaf");
                let point = build_point(&m, &rec, &p, x, &y, &index).unwrap();
                let body = opening_body([3u8; 32], vec![point], None).unwrap();
                let decoded = Opening::decode(&body).unwrap();
                let report = verify_point(&tier_ctx, &resp, &seed, 0, &decoded.points[0]).unwrap_or_else(|e| panic!("{model} seed {counter}: {e:?}"));
                assert_eq!((report.t, report.rank, report.row), (p.t, p.rank, p.row));
                n_ok += 1;
            }
            eprintln!("{model}: {n_ok} openings verified");
            assert!(n_ok > 0);
        }
    }

    /// Traced generation, replay and opening on the resident model, verified as the node would.
    #[test]
    #[ignore = "requires one CUDA GPU, libkeryx-llama, and KERYX_TEST_MODEL_GPU0 (a tier GGUF)"]
    fn traced_generation_replays_and_verifies() {
        use crate::llama_engine;
        let gguf = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let spec = crate::models::spec_for_gguf(&gguf).expect("registry model");
        let tier = crate::models::pom_tier_index(&spec.model_id, u64::MAX).expect("tiered model");
        let m = TierManifest::for_tier(tier).expect("audited tier");
        let index = WeightIndex::build_from_gguf(&gguf, spec.model_id).unwrap();

        llama_engine::ensure_loaded(&gguf, 0).unwrap();
        let nl = llama_engine::trace_configure(&gguf, 0, TierManifest::raw_for_tier(tier).unwrap()).unwrap();
        assert_eq!(nl, m.n_leaves());
        let prompt = "Explain in two sentences why the sky is blue.";
        let t0 = std::time::Instant::now();
        let g = llama_engine::generate_traced(&gguf, 0, prompt, None, 24, 0, 0).unwrap();
        eprintln!("generated {} tokens ({} prompt) in {:.2?}: {:?}", g.tokens.len(), g.n_prompt, t0.elapsed(), g.text);
        assert_eq!(g.n_pos as usize, g.tokens.len());
        assert_eq!(g.leaves.len(), g.tokens.len() * nl);
        let rec = TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).expect("complete trace");
        let c = rec.commitment();
        assert_eq!((c.n_prompt, c.n_pos, c.stage), (g.n_prompt, g.n_pos, STAGE_1A));

        let tier_ctx = TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: m.m_t, n_entries: m.entries.len() as u64, n_products: m.n_products as u64 };
        let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
        let mut n_ok = 0;
        for counter in 0..8u64 {
            let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
            for i in 0..POINTS {
                let p = draw_point(&m, rec.n_prompt, rec.n_pos, &seed, i).unwrap();
                let t0 = std::time::Instant::now();
                let (x, y) = llama_engine::replay(&gguf, 0, &g.tokens, g.n_prompt, p.t as u32, p.leaf_x(), p.leaf_y()).unwrap();
                let replay = t0.elapsed();
                assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(p.t, p.leaf_x()), "seed {counter} point {i}: X leaf");
                assert_eq!(*blake3::hash(&y).as_bytes(), rec.leaf(p.t, p.leaf_y()), "seed {counter} point {i}: Y leaf");
                let point = build_point(&m, &rec, &p, x, &y, &index).unwrap();
                let body = opening_body([3u8; 32], vec![point], None).unwrap();
                let decoded = Opening::decode(&body).unwrap();
                let report = verify_point(&tier_ctx, &resp, &seed, i, &decoded.points[0]).unwrap_or_else(|e| panic!("seed {counter} point {i}: {e:?}"));
                assert_eq!((report.t, report.rank, report.row), (p.t, p.rank, p.row));
                eprintln!(
                    "seed {counter} point {i}: t {} rank {} row {} replay {:.2?} body {} B |y-y_hat| {:.2e} tol {:.2e}",
                    p.t, p.rank, p.row, replay, body.len(), (report.y - report.y_hat).abs(), report.tol
                );
                n_ok += 1;
            }
        }
        // Shape padding: the prompt grows to the floor and the answer gets its generated positions.
        let g = llama_engine::generate_traced(&gguf, 0, prompt, None, 8, 64, 1).unwrap();
        assert_eq!(g.n_prompt, 64);
        assert!(g.n_pos - g.n_prompt >= g.text.len().min(8) as u32, "{} generated for {} B", g.n_pos - g.n_prompt, g.text.len());
        assert_eq!(g.n_pos as usize, g.tokens.len());
        assert!(TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).is_some());

        // Prefix: the request id sits at the tier offset and its point verifies as the node would.
        let ctx = prefix_ctx(tier).expect("prefix context");
        let (head, rest) = crate::llm::format_prompt_parts(spec.name, prompt).expect("prefix template");
        let rh: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let toks = prefix_tokens(ctx, &rh);
        let prefix = llama_engine::PromptPrefix { tokens: &toks, head_tokens: ctx.offset, rest: &rest };
        let g = llama_engine::generate_traced(&gguf, 0, &head, Some(prefix), 8, 0, 0).unwrap();
        assert_eq!(&g.tokens[ctx.offset as usize..ctx.offset as usize + 32], toks.as_slice());
        let rec = TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).expect("complete trace");
        let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
        for counter in 0..4u64 {
            let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
            let t = draw_prefix(ctx, &seed);
            let (x, _) = llama_engine::replay(&gguf, 0, &g.tokens, g.n_prompt, t as u32, 0, 1).unwrap();
            assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(t, 0), "seed {counter}: prefix X leaf");
            let point = build_prefix_point(&m, &rec, ctx, &rh, t, x, &index).unwrap();
            let body = opening_body([3u8; 32], vec![], Some(point)).unwrap();
            let decoded = Opening::decode(&body).unwrap();
            let opened = prefix::verify_prefix_point(&tier_ctx, ctx, &resp, &seed, &rh, decoded.prefix.as_ref().unwrap())
                .unwrap_or_else(|e| panic!("seed {counter}: prefix {e:?}"));
            assert_eq!(opened, t);
            eprintln!("seed {counter}: prefix position {t} verified, body {} B", body.len());
        }
        llama_engine::unload(0);
        assert_eq!(n_ok, 8 * POINTS);
    }
}
