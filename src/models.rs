/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored five-tier lineup, active at `pom_v3_activation_daa()` (the H6 hardfork) — below
/// that DAA this binary refuses to mine (`pom_tier_index` = None). Every model is untied so the
/// in-process llama engine hosts walk + inference in one resident copy:
///   --very-light  Qwen3.5-9B-abliterated Q5_K_M (Alibaba)  — 8 GB
///   --light       GLM-4-9B-0414          Q6_K   (Zhipu)    — 12 GB
///   (default)     Gemma-4-12B-abliterated Q6_K  (Google)   — 16 GB
///   --high        Qwen3.6-27B            Q4_K_M (Alibaba)  — 24 GB
///   --very-high   Kimi-Linear-48B        Q4_K_M (Moonshot) — 32 GB
///
/// All GGUF weights are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF quantized — LLaMA architecture. llama-served.
    Gguf,
    /// GGUF quantized — GLM 4 architecture (tier 1). llama-served.
    GgufGlm4,
    /// GGUF quantized — Qwen3.5 hybrid-SSM architecture (tiers 0 and 3). llama-served.
    GgufQwen35,
    /// GGUF quantized — Kimi-Linear MoE architecture (tier 4). llama-served.
    GgufKimiLinear,
    /// GGUF quantized — Gemma 4 architecture (tier 2). llama-served.
    GgufGemma4,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    /// Empty for the whole lineup: llama uses the tokenizer embedded in the GGUF.
    pub tokenizer_cid: &'static str,
    /// Single entry: the model.gguf CID.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

// ── H6 lineup ───────────────────────────────────────────────────
// Active at `crate::pom::pom_v3_activation_daa()` (the H6 hardfork, matrix-walk era). Every model
// is UNTIED so the in-process llama engine hosts walk + inference in one resident copy;
// `libkeryx-llama.so` is REQUIRED to serve them.
// `tokenizer_cid` is empty: llama uses the tokenizer embedded in the GGUF, no separate file.
// model_id bytes MUST equal the node's `POM_TIERS_H6` constants (CIDv0[2..34] of the pinned GGUF).

/// Tier-0 model — Qwen3.5-9B-abliterated Q5_K_M (huihui-ai abliteration, mradermacher GGUF).
/// `model_id` MUST equal the node's `QWEN3_5_9B_ABLITERATED_MODEL_ID`.
pub const QWEN3_5_9B_ABLITERATED: ModelSpec = ModelSpec {
    name: "qwen3.5-9b-abliterated",
    model_id: [
        0xbd, 0x34, 0x56, 0x8c, 0xd8, 0x9f, 0x5f, 0x19,
        0xc6, 0xc3, 0xa6, 0xe1, 0xa6, 0x1b, 0x92, 0x9b,
        0xc8, 0x68, 0x70, 0x94, 0x09, 0xea, 0xad, 0x8e,
        0x67, 0x2d, 0x85, 0xf3, 0xc1, 0xeb, 0x57, 0x10,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["Qmb5E3zospd78SfiRHB9iZWNz29xuwRJufieZbWzEFBuGB"],
    dir_name: "Qwen3.5-9B-abliterated",
    // ~6.5 GB Q5_K_M weights + ~1.3 GB KV/workspace → 8 GB card.
    min_vram_mb: 8_000,
};

pub const GLM_4_9B_0414: ModelSpec = ModelSpec {
    name: "glm-4-9b-0414",
    // CIDv0[2..34] of model.gguf — GLM-4-9B-0414-abliterated Q6_K
    model_id: [
        0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c,
        0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e,
        0x30, 0x0c, 0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a,
        0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
    ],
    format: ModelFormat::GgufGlm4,
    tokenizer_cid: "",
    weight_cids: &["QmfBGGZumBR4XGFLLPjYozvhRSt3kXjrgsV3jXciCdAeM7"],
    dir_name: "GLM-4-9B-0414",
    // ~8.3 GB Q6_K weights + ~1.5 GB KV/workspace → 12 GB card (3060 12GB / 3080 12GB).
    min_vram_mb: 12_000,
};

pub const QWEN3_6_27B: ModelSpec = ModelSpec {
    name: "qwen3.6-27b",
    // CIDv0[2..34] of model.gguf — Qwen3.6-27B-abliterated-v2 Q4_K_M (mradermacher)
    model_id: [
        0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9,
        0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3,
        0x9f, 0x81, 0x42, 0x78, 0x52, 0x56, 0x04, 0x9e,
        0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmamoYQGGAkBaqiWuNmwxeC9AQnt9F7sLyX57VoqbJWeUV"],
    dir_name: "Qwen3.6-27B",
    // ~16.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const KIMI_LINEAR_48B: ModelSpec = ModelSpec {
    name: "kimi-linear-48b",
    // CIDv0[2..34] of model.gguf — Kimi-Linear-48B-A3B-Instruct-abliterated Q4_K_M (mradermacher, i1)
    model_id: [
        0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef,
        0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6,
        0xac, 0xda, 0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f,
        0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
    ],
    format: ModelFormat::GgufKimiLinear,
    tokenizer_cid: "",
    weight_cids: &["QmSVhtoNrL8bWJXZuEXMMWqty8qHScQMRuacuoa9ujsYqp"],
    dir_name: "Kimi-Linear-48B",
    // ~29.7 GB Q4_K_M weights (MoE, 3B active) + KV/workspace → needs a 32 GB card (5090),
    // so the top tier stays 5090-class.
    min_vram_mb: 30_000,
};

/// Tier-2 model — gemma-4-12B-it-abliterated Q6_K (huihui-ai abliteration, mradermacher
/// GGUF). `model_id` MUST equal the node's `GEMMA_4_12B_ABLITERATED_MODEL_ID`.
pub const GEMMA_4_12B_ABLITERATED: ModelSpec = ModelSpec {
    name: "gemma-4-12b-abliterated",
    model_id: [
        0x39, 0x99, 0x84, 0x04, 0x56, 0x00, 0xf7, 0xd5,
        0x8d, 0x1b, 0x2c, 0xf0, 0x1e, 0x6a, 0x4b, 0xf4,
        0x66, 0xfa, 0x15, 0xc7, 0xac, 0x31, 0xbd, 0x0d,
        0xd1, 0xa7, 0x1e, 0x00, 0x3b, 0x61, 0x7c, 0xc6,
    ],
    format: ModelFormat::GgufGemma4,
    tokenizer_cid: "",
    weight_cids: &["QmSDVicqRDwitecBaPitHsAePLUEamgL4KfrBWYHVWQyx9"],
    dir_name: "Gemma-4-12B-abliterated",
    // ~9.8 GB Q6_K weights + ~2 GB KV/workspace → 16 GB card (fills the 12→24 GB gap).
    min_vram_mb: 16_000,
};

/// Whether `model_id` is one of the Proof-of-Model tier models. DAA-independent — used at startup
/// to pick a mineable PoM model before any block DAA is known (the tier *index* is then computed
/// per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_5_9B_ABLITERATED.model_id
        || *model_id == GLM_4_9B_0414.model_id
        || *model_id == GEMMA_4_12B_ABLITERATED.model_id
        || *model_id == QWEN3_6_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
}

/// Mirror of the node's per-block tier table (`POM_TIERS_H6`), recomputed from the block DAA.
/// Below the H6 gate this binary refuses to mine (None) — it never produces a pre-H6-era block.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    if *model_id == QWEN3_5_9B_ABLITERATED.model_id {
        Some(0)
    } else if *model_id == GLM_4_9B_0414.model_id {
        Some(1)
    } else if *model_id == GEMMA_4_12B_ABLITERATED.model_id {
        Some(2)
    } else if *model_id == QWEN3_6_27B.model_id {
        Some(3)
    } else if *model_id == KIMI_LINEAR_48B.model_id {
        Some(4)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    VeryLight,
    Light,
    Default,
    High,
    VeryHigh,
}

/// True once the H6 hardfork has a scheduled DAA — startup staging (lineup + VRAM ladder) then
/// targets the H6 lineup.
pub fn h6_staged() -> bool {
    crate::pom::pom_v3_activation_daa() != u64::MAX
}

/// DAA marking the latest scheduled era for startup staging (VRAM ladder + initial mining model).
/// The miner does NOT idle until the crossing — it stages against the latest scheduled lineup,
/// prefetches every scheduled era (`pom_models_all_eras`), and hot-swaps the resident model at
/// the crossing (`pom_gpu::advance_mining_tier_if_due`).
pub fn staging_daa() -> u64 {
    crate::pom::pom_v3_activation_daa()
}

/// The single model a hardware `tier` mines AND serves at `daa` — matching the node's per-block
/// tier table (`pom_tiers`). The hardware tier is fixed; the model it must mine flips at a gate,
/// which is what arms `advance_mining_tier_if_due`.
/// `None` when the tier has no consensus-valid model in that era: it mines nothing there and
/// idles until its gate, rather than downloading and mining a model the node would reject. The
/// retirement of a crossed era is expressed by returning `None` for it.
pub fn pom_model_for_tier(daa: u64, tier: Tier) -> Option<&'static ModelSpec> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    Some(match tier {
        Tier::VeryLight => &QWEN3_5_9B_ABLITERATED,
        Tier::Light => &GLM_4_9B_0414,
        Tier::Default => &GEMMA_4_12B_ABLITERATED,
        Tier::High => &QWEN3_6_27B,
        Tier::VeryHigh => &KIMI_LINEAR_48B,
    })
}

/// Every PoM model a `tier` may still mine — the current-era model and, once a later era is
/// scheduled, its model too. Prefetched together at startup so the era crossing hot-swaps the
/// resident mining model without stalling on a mid-run download.
///
/// `chain_daa` is the network's virtual DAA score. An era spans `[gate, next_gate)`, and nothing
/// below the tip can still be mined, so an era the chain has already left needs no model. `None`
/// (node unreachable, or pool mining) keeps every scheduled era.
pub fn pom_models_all_eras(tier: Tier, chain_daa: Option<u64>) -> Vec<&'static ModelSpec> {
    let gates = vec![crate::pom::pom_v3_activation_daa(), staging_daa()];
    let mut out: Vec<&'static ModelSpec> = Vec::new();
    for gate in reachable_gates(gates, chain_daa) {
        let Some(s) = pom_model_for_tier(gate, tier) else { continue };
        if !out.iter().any(|x| x.model_id == s.model_id) {
            out.push(s);
        }
    }
    out
}

/// The era gates whose models can still be mined, sorted. An era spans `[gate, next_gate)`, so it
/// is dropped once the chain has passed `next_gate`. The last era is open-ended and always kept.
fn reachable_gates(mut gates: Vec<u64>, chain_daa: Option<u64>) -> Vec<u64> {
    gates.sort_unstable();
    gates.dedup();
    let Some(daa) = chain_daa else { return gates };
    let last = gates.len().saturating_sub(1);
    gates
        .iter()
        .enumerate()
        .filter(|(i, _)| *i == last || gates[i + 1] > daa)
        .map(|(_, gate)| *gate)
        .collect()
}

/// The single model a hardware tier mines AND serves at **startup staging** (the latest scheduled
/// era). A PoM GPU is bound to its tier; the era crossing swaps the resident model in place.
///
/// Infallible by construction: the latest scheduled era always carries a model for all five
/// tiers. Retiring a tier outright would have to shrink the VRAM ladder in the same change, and
/// this panic is where a half-done retirement would surface.
pub fn spec_for_tier(tier: Tier) -> &'static ModelSpec {
    pom_model_for_tier(staging_daa(), tier).expect("the staging era carries every tier")
}

/// Resolves a model name/id.
pub const REGISTRY: &[&ModelSpec] = &[
    &QWEN3_5_9B_ABLITERATED,
    &GLM_4_9B_0414,
    &GEMMA_4_12B_ABLITERATED,
    &QWEN3_6_27B,
    &KIMI_LINEAR_48B,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the eras the chain can still reach are kept. Synthetic gates keep this independent of
    /// the network switch.
    #[test]
    fn eras_already_left_are_dropped() {
        // Testnet shape: H4/H5 born with the chain, H6 scheduled later.
        let testnet = vec![0, 0, 108_000];
        // Chain past the H6 gate: the pre-H6 era can never be mined again.
        assert_eq!(reachable_gates(testnet.clone(), Some(200_000)), vec![108_000]);
        // Chain still below it: both eras are live, the crossing must not stall on a download.
        assert_eq!(reachable_gates(testnet.clone(), Some(50_000)), vec![0, 108_000]);
        // Unknown tip (node unreachable, pool mining): keep everything.
        assert_eq!(reachable_gates(testnet, None), vec![0, 108_000]);

        // Mainnet shape, H6 armed ahead of the tip: the H4 era is gone, H5 and H6 are kept.
        let mainnet = vec![54_766_000, 59_009_037, 70_000_000];
        assert_eq!(reachable_gates(mainnet.clone(), Some(66_000_000)), vec![59_009_037, 70_000_000]);
        // Once the tip passes the H6 gate the H5 model retires on its own — no code change needed.
        assert_eq!(reachable_gates(mainnet, Some(70_000_001)), vec![70_000_000]);

        // Exactly at a gate: that era has just begun, the previous one is over.
        assert_eq!(reachable_gates(vec![0, 108_000], Some(108_000)), vec![108_000]);
        // H6 unscheduled: staging collapses onto the H5 gate, leaving a single live era.
        assert_eq!(reachable_gates(vec![54_766_000, 59_009_037, 59_009_037], Some(66_000_000)), vec![59_009_037]);
    }

    /// The per-block tier table — mirror of the node's `POM_TIERS_H6` order. `u64::MAX` sits
    /// at/after every gate on any network, so this exercises the table without touching the
    /// global testnet switch.
    #[test]
    fn tier_table_mirrors_node() {
        let daa = u64::MAX;
        assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), Some(0));
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), Some(1));
        assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, daa), Some(3));
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), Some(4));

        // The hardware-tier -> model map agrees with the table, tier for tier.
        assert_eq!(pom_model_for_tier(daa, Tier::VeryLight).unwrap().model_id, QWEN3_5_9B_ABLITERATED.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Light).unwrap().model_id, GLM_4_9B_0414.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Default).unwrap().model_id, GEMMA_4_12B_ABLITERATED.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::High).unwrap().model_id, QWEN3_6_27B.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::VeryHigh).unwrap().model_id, KIMI_LINEAR_48B.model_id);

        // Every registry model is a mineable tier, and every tier's staged model is in the registry.
        for spec in REGISTRY {
            assert!(is_pom_model(&spec.model_id), "{} is not a PoM model", spec.name);
            assert!(pom_tier_index(&spec.model_id, daa).is_some(), "{} has no tier", spec.name);
        }

        // Below the gate the miner produces nothing rather than a block the node would reject.
        let gate = crate::pom::pom_v3_activation_daa();
        if gate > 0 {
            assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, gate - 1), None);
            assert!(pom_model_for_tier(gate - 1, Tier::Default).is_none());
        }
    }
}
