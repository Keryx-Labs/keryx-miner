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
///   --high        Qwen3.6-27B            Q4_K_M (Alibaba)  — 24 GB  (Qwen3.8-27B Q4_K from H14)
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
    /// GGUF quantized — DeepSeek V4 architecture (H14 network model and its shards). Served by
    /// a pipeline head only.
    GgufDeepSeekV4,
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
    /// KV cache + CUDA workspace. Used by the capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
    /// Smallest context window (tokens) the engine must hold to serve the model: the budget
    /// clients assume. A card that cannot allocate it does not declare the model.
    pub ctx_floor: u32,
    /// Largest context window worth allocating when the VRAM allows it.
    pub ctx_cap: u32,
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
    ctx_floor: 8_192,
    ctx_cap: 32_768,
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
    ctx_floor: 32_768,
    ctx_cap: 32_768,
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
    ctx_floor: 32_768,
    ctx_cap: 65_536,
};

/// Tier-3 model from the private-inference gate (H14) — Huihui-Qwen3.8-27B-abliterated Q4_K.
/// `model_id` MUST equal the node's `QWEN3_8_27B_MODEL_ID`.
pub const QWEN3_8_27B: ModelSpec = ModelSpec {
    name: "qwen3.8-27b",
    model_id: [
        0x73, 0x74, 0x0b, 0x44, 0x3b, 0xdc, 0x00, 0xaf,
        0xda, 0x5f, 0xa3, 0x4e, 0xb9, 0x99, 0x9d, 0x3f,
        0xea, 0x77, 0xdc, 0xc3, 0xf6, 0xde, 0x23, 0x8f,
        0xab, 0x70, 0x13, 0x94, 0xcd, 0xc9, 0x6f, 0xb3,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmW7LDz7ZTfw9vpAR9jMhFHWriLhxh728Kihp7oTSLgvyg"],
    dir_name: "Qwen3.8-27B",
    // ~16.8 GB Q4_K weights + KV/workspace → 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
    ctx_floor: 32_768,
    ctx_cap: 65_536,
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
    ctx_floor: 32_768,
    ctx_cap: 131_072,
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
    ctx_floor: 32_768,
    ctx_cap: 131_072,
};

// ── H18 lineup ──────────────────────────────────────────────────────────────────────────────────
// Tiers 0 and 1 from `crate::pom::model_split_activation_daa()`; tiers 2-4 are unchanged.
// model_id bytes MUST equal the node's `POM_TIERS_MODEL_SPLIT` constants.

/// H18 tier-0 model — NVIDIA-Nemotron-3.5-Bolt-E4B heretic Q6_K (Losew GGUF, arch gemma4).
/// `model_id` MUST equal the node's `BOLT_E4B_MODEL_ID`.
pub const BOLT_E4B: ModelSpec = ModelSpec {
    name: "nemotron-3.5-bolt-e4b",
    model_id: [
        0xf9, 0x10, 0x8e, 0xb1, 0xc6, 0x7b, 0x3e, 0xe0,
        0xd4, 0x4c, 0x96, 0xf2, 0x59, 0xe3, 0xfd, 0x93,
        0xee, 0x50, 0x75, 0x19, 0x4b, 0x54, 0x43, 0x61,
        0x65, 0xcf, 0x19, 0x87, 0xab, 0xaf, 0x5a, 0xc9,
    ],
    format: ModelFormat::GgufGemma4,
    tokenizer_cid: "",
    weight_cids: &["Qmf6tsQ2DDMLSrQKwuL2NfmJUBkeV9GGwZUu8Co9JfJJ4G"],
    dir_name: "Bolt-E4B",
    min_vram_mb: 8_000,
    ctx_floor: 32_768,
    ctx_cap: 32_768,
};

/// H18 tier-1 model — Qwen3.5-9B-abliterated Q8_0 (huihui-ai abliteration, mradermacher GGUF).
/// `model_id` MUST equal the node's `QWEN3_5_9B_Q8_MODEL_ID`.
pub const QWEN3_5_9B_Q8: ModelSpec = ModelSpec {
    name: "qwen3.5-9b-q8",
    model_id: [
        0x25, 0xdd, 0x00, 0x1c, 0xa8, 0x01, 0x70, 0x8f,
        0xa8, 0x2a, 0x9d, 0x2a, 0x77, 0xe7, 0xba, 0xec,
        0x4f, 0x27, 0x43, 0xae, 0xbe, 0xe9, 0x9d, 0xc6,
        0x04, 0xc3, 0xbf, 0xbc, 0xf6, 0x71, 0x64, 0x57,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmQtTFMQYvnoZ1ZxNsuth8zzLWoBVKnCDVoW2ViCn8H6na"],
    dir_name: "Qwen3.5-9B-Q8_0",
    min_vram_mb: 12_000,
    ctx_floor: 32_768,
    ctx_cap: 32_768,
};

/// Lineup the per-tier rules of tiers 0-4 follow — mirror of the node's `Lineup`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lineup {
    H14,
    H18,
}

/// Lineup of a lineup model: tiers 0 and 1 changed models at H18, tiers 2-4 did not, so the
/// model alone decides it.
pub fn lineup_of(model_id: &[u8; 32]) -> Lineup {
    if *model_id == BOLT_E4B.model_id || *model_id == QWEN3_5_9B_Q8.model_id { Lineup::H18 } else { Lineup::H14 }
}

// ── Network model ───────────────────────────────────────────────────────────────────────────────
// Active at `crate::pom::model_split_activation_daa()`. Tier 4 is retired: from the gate a 32 GB card
// mines one of the four ranges of DeepSeek-V4-Flash-0731 (Q2), the one the chain assigns to its
// escrow key under the template's pruning point (`assigned_range`). The whole model is the id
// AiRequests target; nobody mines it. model_id bytes MUST equal the node's `POM_TIERS_MODEL_SPLIT`.

/// The whole network model — DeepSeek-V4-Flash-0731 abliterated (huihui-ai, ds4, Q2). Request
/// target and pipeline-head capability only; never downloaded, walked or mined as such.
/// `model_id` MUST equal the node's `NETWORK_MODEL_V4_FLASH_MODEL_ID`.
pub const V4_FLASH: ModelSpec = ModelSpec {
    name: "v4-flash-0731",
    model_id: [
        0xf9, 0xe0, 0x9f, 0xc3, 0x78, 0x23, 0xa0, 0x3d,
        0x20, 0x9b, 0x1e, 0xe8, 0x4c, 0x49, 0x67, 0x28,
        0x54, 0x6c, 0xab, 0x62, 0x41, 0x87, 0x01, 0xa0,
        0xf0, 0xa3, 0xda, 0xa6, 0xd3, 0x2a, 0xb8, 0xdf,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmfA4tHEStL5poiBvHm1rid4b3cbQbhJ8v83RjE3Uytksg"],
    dir_name: "V4-Flash-0731",
    min_vram_mb: 0,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// V4-Flash-0731 range 0: layers 0-10, 32 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_0_MODEL_ID`.
pub const V4_FLASH_SHARD_0: ModelSpec = ModelSpec {
    name: "v4-flash-0731-range-0",
    model_id: [
        0x22, 0x7b, 0x7a, 0x33, 0xbf, 0x60, 0x86, 0xfe,
        0xca, 0xeb, 0xf6, 0xc2, 0x74, 0xf7, 0x42, 0xa9,
        0x9e, 0x75, 0x21, 0x70, 0x0e, 0x26, 0x25, 0x3a,
        0xe7, 0xf5, 0x77, 0x82, 0x20, 0x08, 0x54, 0xf8,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmQfFmvDm4SFxuFi2rMXEAWH29dfiTUaajdpFDpYHeLqy1"],
    dir_name: "V4-Flash-0731-range-0",
    min_vram_mb: 30000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// V4-Flash-0731 range 1: layers 11-21, 32 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_1_MODEL_ID`.
pub const V4_FLASH_SHARD_1: ModelSpec = ModelSpec {
    name: "v4-flash-0731-range-1",
    model_id: [
        0xa6, 0xab, 0x8d, 0x41, 0x50, 0x86, 0x04, 0x40,
        0xf3, 0xe0, 0x9a, 0x8a, 0x91, 0x70, 0x13, 0xc1,
        0x6f, 0x55, 0x67, 0x16, 0x11, 0x13, 0xa3, 0x87,
        0x2a, 0x7f, 0x58, 0x71, 0x02, 0x2d, 0x13, 0xfd,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmZZG7FgKAyobfSnQfLrEiXyewicskH41AoLrSB1oEgPAt"],
    dir_name: "V4-Flash-0731-range-1",
    min_vram_mb: 30000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// V4-Flash-0731 range 2: layers 22-32, 32 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_2_MODEL_ID`.
pub const V4_FLASH_SHARD_2: ModelSpec = ModelSpec {
    name: "v4-flash-0731-range-2",
    model_id: [
        0xc0, 0x9b, 0x96, 0xe1, 0xa7, 0xa4, 0x69, 0x15,
        0x0b, 0xf9, 0x9e, 0xaf, 0x6e, 0xe8, 0xcc, 0x43,
        0xeb, 0x97, 0xf6, 0x3a, 0xff, 0x57, 0xa8, 0x3b,
        0x29, 0x9b, 0x6e, 0xe8, 0x82, 0x82, 0x3e, 0xdb,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmbJWbGDBaS8RuYLkw74qqNaJseEf4hgKeNLk8rqBLMmjQ"],
    dir_name: "V4-Flash-0731-range-2",
    min_vram_mb: 30000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// V4-Flash-0731 range 3: layers 33-42, 32 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_3_MODEL_ID`.
pub const V4_FLASH_SHARD_3: ModelSpec = ModelSpec {
    name: "v4-flash-0731-range-3",
    model_id: [
        0xa4, 0x9c, 0xdb, 0xb0, 0x81, 0xe0, 0xeb, 0xe8,
        0x86, 0x63, 0xd8, 0x40, 0x4b, 0xa7, 0x7b, 0xe8,
        0x61, 0xdb, 0x2b, 0x9e, 0x85, 0x7b, 0xbd, 0xa2,
        0x3b, 0x5e, 0xba, 0xba, 0xc1, 0x9e, 0x36, 0x17,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmZREJF9HZyZQdvDUGiCdUDuAjABW8wmyYLGpbfcjSLi8v"],
    dir_name: "V4-Flash-0731-range-3",
    min_vram_mb: 30000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// The four mainnet ranges, layer order (`V4_FLASH_SHARDS[k]` is tier `5 + k`).
pub const V4_FLASH_SHARDS: [&ModelSpec; 4] = [&V4_FLASH_SHARD_0, &V4_FLASH_SHARD_1, &V4_FLASH_SHARD_2, &V4_FLASH_SHARD_3];

// Devnet bench network model: Qwen3.5-9B in two shards ─ mirror of the node's
// `POM_TIERS_H14_SPLIT9B`: the whole pipeline on one card.

pub const SPLIT9B_SHARD_0: ModelSpec = ModelSpec {
    name: "split9b-shard-0",
    model_id: [
        0x8d, 0xba, 0x34, 0xd0, 0x28, 0x5b, 0xe2, 0x87,
        0xf0, 0x22, 0xda, 0xd0, 0xf3, 0x3e, 0xa5, 0x88,
        0xc8, 0x36, 0x94, 0xd4, 0x83, 0x59, 0x60, 0x8c,
        0xc8, 0xd4, 0x74, 0x24, 0x26, 0x76, 0xf2, 0x79,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmXstsoY2jeNkQhVgWceiLLkBxHD4uooicnTPYUjDNCkq6"],
    dir_name: "Split9B-shard-0",
    min_vram_mb: 3_000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

pub const SPLIT9B_SHARD_1: ModelSpec = ModelSpec {
    name: "split9b-shard-1",
    model_id: [
        0xa1, 0x98, 0xfe, 0x36, 0x9a, 0x63, 0xde, 0x8d,
        0x75, 0xc5, 0xeb, 0x4a, 0x46, 0xa8, 0x24, 0xe0,
        0x20, 0x55, 0x4a, 0xc2, 0x65, 0x48, 0xad, 0x68,
        0x8c, 0x90, 0xa9, 0x60, 0xdb, 0x52, 0x49, 0x2e,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmZDTf3KtGYjR7qUgD2PHfK3divUhNtYNcHvuSsbCNZYU5"],
    dir_name: "Split9B-shard-1",
    min_vram_mb: 3_000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// Bench network model: it must not share a `model_id` with any lineup tier, or the node maps a
/// request to lineup tier 0 instead of the network model. Same directory as the 9B so the packed
/// head is found there; never mined, so never downloaded.
pub const SPLIT9B_WHOLE: ModelSpec = ModelSpec {
    name: "split9b-whole",
    model_id: [
        0x1f, 0x22, 0xb2, 0x13, 0x3c, 0x40, 0xb1, 0x6d,
        0x0b, 0xa1, 0x5d, 0xff, 0xb4, 0x82, 0x3f, 0x88,
        0x9b, 0x76, 0xeb, 0x83, 0x4f, 0xe6, 0xc9, 0x0d,
        0xed, 0xc6, 0xa3, 0x8d, 0x50, 0x4a, 0x46, 0xcb,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &[],
    dir_name: "Qwen3.5-9B-abliterated",
    min_vram_mb: 8_000,
    ctx_floor: 4_096,
    ctx_cap: 4_096,
};

/// The network model of one network: the whole model requests target, its ranges in tier
/// order, their layer spans, the layer count and the packed head.
pub struct NetworkModelSpec {
    pub whole: &'static ModelSpec,
    pub shards: &'static [&'static ModelSpec],
    pub layers: &'static [(u32, u32)],
    pub n_layer: u32,
    pub head_cid: &'static str,
    pub head_digest_hex: &'static str,
}

pub static NETWORK_MODEL_MAINNET: NetworkModelSpec = NetworkModelSpec {
    whole: &V4_FLASH,
    shards: &V4_FLASH_SHARDS,
    layers: &[(0, 10), (11, 21), (22, 32), (33, 42)],
    n_layer: 43,
    head_cid: "QmVEZUvJF9e2FRMf5aW6MxWGENgqmhbo2dr87em2hK4QFp",
    head_digest_hex: "667269bca5045390ff0ed3084a224c3d29666ee5fffa1a19e9e97fd6312ec56f",
};

pub static NETWORK_MODEL_SPLIT9B: NetworkModelSpec = NetworkModelSpec {
    whole: &SPLIT9B_WHOLE,
    shards: &[&SPLIT9B_SHARD_0, &SPLIT9B_SHARD_1],
    layers: &[(0, 15), (16, 31)],
    n_layer: 32,
    head_cid: "QmQSC2sjKaAqZH4E6HGbtkowWqGKBrHx4JvAWmXsTpNdDC",
    head_digest_hex: "1f22b2133c40b16d0ba15dffb4823f889b76eb834fe6c90dedc6a38d504a46cb",
};

/// The network model of the selected network: split V4-Flash-0731, or the two-range 9B bench
/// on devnet (`--devnet`).
pub fn network_model() -> &'static NetworkModelSpec {
    if crate::pom::is_devnet() {
        &NETWORK_MODEL_SPLIT9B
    } else {
        &NETWORK_MODEL_MAINNET
    }
}

/// First tier index of the network model — mirror of the node's `NETWORK_MODEL_TIER`.
pub const NETWORK_MODEL_TIER: u8 = 4;

/// Tier as shown to users: a network-model range reads `4 range k`.
pub fn tier_label(tier: u8) -> String {
    if tier > NETWORK_MODEL_TIER {
        format!("{} range {}", NETWORK_MODEL_TIER, tier - NETWORK_MODEL_TIER - 1)
    } else {
        tier.to_string()
    }
}

/// Range index assigned to an escrow key under a pruning point — mirror of the node's
/// `network_model_range`.
pub fn network_model_range(pruning_point: &[u8; 32], escrow_pubkey: &[u8; 32], n_shards: usize) -> u8 {
    let mut h = blake3::Hasher::new();
    h.update(b"KeryxRangeV1");
    h.update(pruning_point);
    h.update(escrow_pubkey);
    let word = u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap());
    (word % n_shards.max(1) as u64) as u8
}

/// Escrow key of the coinbase this miner builds; the range is keyed on it.
static RANGE_ESCROW_KEY: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();

pub fn set_range_escrow_key(key: [u8; 32]) {
    let _ = RANGE_ESCROW_KEY.set(key);
}

/// Range assigned under the latest template's pruning point; `NO_RANGE` until one is seen.
static ASSIGNED_RANGE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(NO_RANGE);
const NO_RANGE: u8 = u8::MAX;

/// Records the pruning point of a block template and returns the range it assigns, `None` without
/// an escrow key. Returns whether it changed through the second value.
pub fn note_template_pruning_point(pruning_point: &[u8; 32]) -> Option<(u8, bool)> {
    let key = RANGE_ESCROW_KEY.get()?;
    let range = network_model_range(pruning_point, key, network_model().shards.len());
    let prev = ASSIGNED_RANGE.swap(range, std::sync::atomic::Ordering::Relaxed);
    Some((range, prev != range))
}

/// Range a pruning point assigns this miner's escrow key, `None` without a key.
pub fn range_for(pruning_point: &[u8; 32]) -> Option<u8> {
    Some(network_model_range(pruning_point, RANGE_ESCROW_KEY.get()?, network_model().shards.len()))
}

/// Whether a block mined by `model_id` under `pruning_point` carries the range the node expects:
/// true for any lineup model, and for a range only when it is the assigned one.
pub fn range_matches(model_id: &[u8; 32], pruning_point: Option<&[u8; 32]>) -> bool {
    let Some(k) = shard_index(model_id) else { return true };
    pruning_point.and_then(range_for) == Some(k)
}

/// The range the latest template assigns this miner, if known.
pub fn assigned_range() -> Option<u8> {
    match ASSIGNED_RANGE.load(std::sync::atomic::Ordering::Relaxed) {
        NO_RANGE => None,
        k => Some(k),
    }
}

/// The range a 32 GB card mines past the gate: the assigned one, `None` until a template is seen.
pub fn assigned_range_spec() -> Option<&'static ModelSpec> {
    network_model().shards.get(assigned_range()? as usize).copied()
}

/// Shard index of a network-model shard id on the selected network.
pub fn shard_index(model_id: &[u8; 32]) -> Option<u8> {
    network_model().shards.iter().position(|s| s.model_id == *model_id).map(|k| k as u8)
}

/// Whether `model_id` is one of the Proof-of-Model tier models. DAA-independent — used at startup
/// to pick a mineable PoM model before any block DAA is known (the tier *index* is then computed
/// per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_5_9B_ABLITERATED.model_id
        || *model_id == GLM_4_9B_0414.model_id
        || *model_id == GEMMA_4_12B_ABLITERATED.model_id
        || *model_id == QWEN3_6_27B.model_id
        || *model_id == QWEN3_8_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
        || *model_id == BOLT_E4B.model_id
        || *model_id == QWEN3_5_9B_Q8.model_id
        || shard_index(model_id).is_some()
}

/// Mirror of the node's per-block tier table (`POM_TIERS_H6`, `POM_TIERS_H14` from the
/// private-inference gate, then the model-split table), recomputed from the block DAA. Below the
/// H6 gate this binary refuses to mine (None) — it never produces a pre-H6-era block. From the
/// model-split gate the shards add their tiers after the lineup; the whole network model is never
/// mined.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    let split = daa >= crate::pom::model_split_activation_daa();
    if split {
        if let Some(k) = shard_index(model_id) {
            return Some(NETWORK_MODEL_TIER + 1 + k);
        }
    }
    if *model_id == BOLT_E4B.model_id {
        split.then_some(0)
    } else if *model_id == QWEN3_5_9B_Q8.model_id {
        split.then_some(1)
    } else if split && (*model_id == QWEN3_5_9B_ABLITERATED.model_id || *model_id == GLM_4_9B_0414.model_id) {
        None
    } else if *model_id == QWEN3_5_9B_ABLITERATED.model_id {
        Some(0)
    } else if *model_id == GLM_4_9B_0414.model_id {
        Some(1)
    } else if *model_id == GEMMA_4_12B_ABLITERATED.model_id {
        Some(2)
    } else if *model_id == QWEN3_6_27B.model_id {
        (daa < crate::pom::private_inference_activation_daa()).then_some(3)
    } else if *model_id == QWEN3_8_27B.model_id {
        (daa >= crate::pom::private_inference_activation_daa()).then_some(3)
    } else if *model_id == KIMI_LINEAR_48B.model_id {
        (!split).then_some(4)
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

/// True once the H14 model split has a scheduled DAA.
pub fn model_split_staged() -> bool {
    crate::pom::model_split_activation_daa() != u64::MAX
}

/// DAA marking the latest scheduled era for startup staging (VRAM ladder + initial mining model).
/// The miner does NOT idle until the crossing — it stages against the latest scheduled lineup,
/// prefetches every scheduled era (`pom_models_all_eras`), and hot-swaps the resident model at
/// the crossing (`pom_gpu::advance_mining_tier_if_due`).
pub fn staging_daa() -> u64 {
    if model_split_staged() {
        crate::pom::model_split_activation_daa()
    } else {
        match crate::pom::private_inference_activation_daa() {
            u64::MAX => crate::pom::pom_v3_activation_daa(),
            h14 => h14.max(crate::pom::pom_v3_activation_daa()),
        }
    }
}

/// The tier of a `pom_tier_index` value.
pub fn tier_from_index(index: u8) -> Option<Tier> {
    Some(match index {
        0 => Tier::VeryLight,
        1 => Tier::Light,
        2 => Tier::Default,
        3 => Tier::High,
        4 => Tier::VeryHigh,
        _ => return None,
    })
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
    let split = daa >= crate::pom::model_split_activation_daa();
    if split && tier == Tier::VeryHigh {
        return assigned_range_spec();
    }
    Some(match tier {
        Tier::VeryLight if split => &BOLT_E4B,
        Tier::VeryLight => &QWEN3_5_9B_ABLITERATED,
        Tier::Light if split => &QWEN3_5_9B_Q8,
        Tier::Light => &GLM_4_9B_0414,
        Tier::Default => &GEMMA_4_12B_ABLITERATED,
        Tier::High if daa >= crate::pom::private_inference_activation_daa() => &QWEN3_8_27B,
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
/// (height unknown) keeps every scheduled era.
pub fn pom_models_all_eras(tier: Tier, chain_daa: Option<u64>) -> Vec<&'static ModelSpec> {
    let mut gates = vec![crate::pom::pom_v3_activation_daa(), staging_daa()];
    if crate::pom::private_inference_activation_daa() != u64::MAX {
        gates.push(crate::pom::private_inference_activation_daa());
    }
    if model_split_staged() {
        gates.push(crate::pom::model_split_activation_daa());
    }
    let mut out: Vec<&'static ModelSpec> = Vec::new();
    for gate in reachable_gates(gates, chain_daa) {
        // A 32 GB card keeps every range on disk: the chain may assign any of them.
        let specs: Vec<&'static ModelSpec> = if tier == Tier::VeryHigh && gate >= crate::pom::model_split_activation_daa() {
            network_model().shards.to_vec()
        } else {
            pom_model_for_tier(gate, tier).into_iter().collect()
        };
        for s in specs {
            if !out.iter().any(|x| x.model_id == s.model_id) {
                out.push(s);
            }
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
/// Chain DAA seen at startup, `u64::MAX` when unknown (node unreachable, pool mining).
static CURRENT_DAA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Records where the chain is, so the announced lineup is the era being mined rather than the
/// staged one: a shard announced before its gate cannot be loaded by the inference engine, and
/// the miner would withdraw it and park its tier.
pub fn set_current_daa(daa: u64) {
    CURRENT_DAA.store(daa, std::sync::atomic::Ordering::Relaxed);
}

pub fn spec_for_tier(tier: Tier) -> &'static ModelSpec {
    let daa = CURRENT_DAA.load(std::sync::atomic::Ordering::Relaxed);
    if daa != u64::MAX {
        // Below the first PoM gate the era carries no model at all; the lineup is what a miner
        // can serve there, never a shard.
        let era = daa.max(crate::pom::pom_v3_activation_daa());
        if let Some(spec) = pom_model_for_tier(era, tier) {
            return spec;
        }
    }
    // A 32 GB card past the gate learns its range from the first template; until then it stages
    // the first range, swapped before any load if the template assigns another.
    pom_model_for_tier(staging_daa(), tier)
        .or_else(|| (tier == Tier::VeryHigh).then(|| network_model().shards[0]))
        .expect("the staging era carries every tier")
}

/// Resolves a model name/id.
pub const REGISTRY: &[&ModelSpec] = &[
    &QWEN3_5_9B_ABLITERATED,
    &GLM_4_9B_0414,
    &GEMMA_4_12B_ABLITERATED,
    &QWEN3_6_27B,
    &QWEN3_8_27B,
    &KIMI_LINEAR_48B,
    &BOLT_E4B,
    &QWEN3_5_9B_Q8,
    &V4_FLASH,
    &V4_FLASH_SHARD_0,
    &V4_FLASH_SHARD_1,
    &V4_FLASH_SHARD_2,
    &V4_FLASH_SHARD_3,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

/// The lineup model a GGUF path belongs to, by its `models/<dir_name>/` component.
pub fn spec_for_gguf(path: &str) -> Option<&'static ModelSpec> {
    let normalized = path.replace('\\', "/");
    REGISTRY.iter().copied().find(|m| normalized.contains(&format!("/{}/", m.dir_name)))
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
        // The last lineup-era score: just under the model-split gate (or the top of the range while it is
        // unscheduled), at/after every other gate on any network.
        let daa = crate::pom::model_split_activation_daa().saturating_sub(1).max(crate::pom::pom_v3_activation_daa());
        assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), Some(0));
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), Some(1));
        assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_8_27B.model_id, daa), Some(3));
        assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, daa), None);
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), Some(4));
        assert_eq!(pom_tier_index(&BOLT_E4B.model_id, daa), None);
        assert_eq!(pom_tier_index(&QWEN3_5_9B_Q8.model_id, daa), None);

        // The hardware-tier -> model map agrees with the table, tier for tier.
        assert_eq!(pom_model_for_tier(daa, Tier::VeryLight).unwrap().model_id, QWEN3_5_9B_ABLITERATED.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Light).unwrap().model_id, GLM_4_9B_0414.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Default).unwrap().model_id, GEMMA_4_12B_ABLITERATED.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::High).unwrap().model_id, QWEN3_8_27B.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::VeryHigh).unwrap().model_id, KIMI_LINEAR_48B.model_id);

        // Just below the H14 gate tier 3 is still Qwen3.6-27B.
        let pre_h14 = crate::pom::private_inference_activation_daa().saturating_sub(1);
        if pre_h14 >= crate::pom::pom_v3_activation_daa() {
            assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, pre_h14), Some(3));
            assert_eq!(pom_tier_index(&QWEN3_8_27B.model_id, pre_h14), None);
            assert_eq!(pom_model_for_tier(pre_h14, Tier::High).unwrap().model_id, QWEN3_6_27B.model_id);
        }

        // Every lineup model is a mineable tier in some lineup era; the network model and its
        // shards are not.
        for spec in REGISTRY.iter().take(8) {
            assert!(is_pom_model(&spec.model_id), "{} is not a PoM model", spec.name);
            let tier = pom_tier_index(&spec.model_id, daa)
                .or_else(|| pom_tier_index(&spec.model_id, pre_h14))
                .or_else(|| pom_tier_index(&spec.model_id, u64::MAX));
            assert!(tier.is_some(), "{} has no tier", spec.name);
        }
        assert!(!is_pom_model(&V4_FLASH.model_id));
        for spec in network_model().shards {
            assert!(is_pom_model(&spec.model_id));
            assert_eq!(pom_tier_index(&spec.model_id, daa), None);
        }
        // the layout is internally consistent
        for nm in [&NETWORK_MODEL_MAINNET] {
            assert_eq!(nm.shards.len(), nm.layers.len());
            assert_eq!(nm.layers[0].0, 0);
            assert_eq!(nm.layers[nm.layers.len() - 1].1 + 1, nm.n_layer);
            for w in nm.layers.windows(2) {
                assert_eq!(w[0].1 + 1, w[1].0);
            }
        }

        // Below the gate the miner produces nothing rather than a block the node would reject.
        let gate = crate::pom::pom_v3_activation_daa();
        if gate > 0 {
            assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, gate - 1), None);
            assert!(pom_model_for_tier(gate - 1, Tier::Default).is_none());
        }
    }

    /// The model-split era — mirror of the node's `POM_TIERS_MODEL_SPLIT`: tiers 0-3 keep the H18
    /// lineup, tier 4 is the network model, ranges are tiers 5-8 and a 32 GB card mines the one assigned to
    /// its escrow key. `u64::MAX` is at/after the gate on any network.
    #[test]
    fn model_split_era_mines_the_assigned_range() {
        let daa = u64::MAX;
        assert!(daa >= crate::pom::model_split_activation_daa());
        for (k, spec) in V4_FLASH_SHARDS.iter().enumerate() {
            assert_eq!(pom_tier_index(&spec.model_id, daa), Some(NETWORK_MODEL_TIER + 1 + k as u8), "{}", spec.name);
            assert_eq!(shard_index(&spec.model_id), Some(k as u8));
            assert_eq!(spec.dir_name, format!("V4-Flash-0731-range-{}", k));
        }
        assert_eq!(pom_tier_index(&V4_FLASH.model_id, daa), None);
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), None);
        assert_eq!(pom_tier_index(&BOLT_E4B.model_id, daa), Some(0));
        assert_eq!(pom_tier_index(&QWEN3_5_9B_Q8.model_id, daa), Some(1));
        assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), None);
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), None);
        assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_8_27B.model_id, daa), Some(3));
        assert_eq!(lineup_of(&BOLT_E4B.model_id), Lineup::H18);
        assert_eq!(lineup_of(&QWEN3_5_9B_Q8.model_id), Lineup::H18);
        assert_eq!(lineup_of(&GEMMA_4_12B_ABLITERATED.model_id), Lineup::H14);
        assert_eq!(pom_model_for_tier(daa, Tier::VeryLight).unwrap().model_id, BOLT_E4B.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Light).unwrap().model_id, QWEN3_5_9B_Q8.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Default).unwrap().model_id, GEMMA_4_12B_ABLITERATED.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::High).unwrap().model_id, QWEN3_8_27B.model_id);
        // once the gate is scheduled, a 32 GB card keeps every range on disk
        if model_split_staged() {
            let all = pom_models_all_eras(Tier::VeryHigh, Some(daa));
            for spec in V4_FLASH_SHARDS {
                assert!(all.iter().any(|s| s.model_id == spec.model_id), "{}", spec.name);
            }
        }
        // every network-model CID decodes to its model_id (CIDv0[2..34])
        for spec in REGISTRY.iter().skip(6) {
            let raw = base58btc_decode(spec.weight_cids[0]);
            assert_eq!(&raw[..2], &[0x12, 0x20], "{}", spec.name);
            assert_eq!(&raw[2..34], &spec.model_id, "{}", spec.name);
        }
        let head = base58btc_decode(NETWORK_MODEL_MAINNET.head_cid);
        assert_eq!(hex::encode(&head[2..34]), NETWORK_MODEL_MAINNET.head_digest_hex);
    }

    /// Same keying as the node: pruning point and escrow key, four ranges, spread evenly.
    #[test]
    fn range_follows_the_pruning_point_and_the_escrow_key() {
        let (pp, key) = ([7u8; 32], [9u8; 32]);
        let r = network_model_range(&pp, &key, 4);
        assert!(r < 4);
        assert_eq!(network_model_range(&pp, &key, 4), r);
        assert_eq!(network_model_range(&pp, &key, 1), 0);
        let mut seen = [0u32; 4];
        for b in 0..=255u8 {
            seen[network_model_range(&pp, &[b; 32], 4) as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n > 40), "{:?}", seen);
        assert!((0..64u8).any(|b| network_model_range(&[b; 32], &key, 4) != r));
        // lineup models are never range-checked; a range needs the assigned one
        assert!(range_matches(&BOLT_E4B.model_id, None));
        assert!(!range_matches(&V4_FLASH_SHARD_0.model_id, None));
    }

    fn base58btc_decode(s: &str) -> Vec<u8> {
        const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        let mut out: Vec<u8> = Vec::new();
        for c in s.bytes() {
            let mut carry = ALPHABET.iter().position(|&a| a == c).unwrap() as u32;
            for b in out.iter_mut().rev() {
                let v = *b as u32 * 58 + carry;
                *b = (v & 0xff) as u8;
                carry = v >> 8;
            }
            while carry > 0 {
                out.insert(0, (carry & 0xff) as u8);
                carry >>= 8;
            }
        }
        out
    }
}
