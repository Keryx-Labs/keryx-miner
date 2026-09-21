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

// ── H14 network model ───────────────────────────────────────────────────────────────────────────
// Active at `crate::pom::h14_activation_daa()`. The lineup above stays in the registry for the
// blocks that mined it but is paused: from the gate a GPU walks and mines ONE shard of
// DeepSeek-V4-Flash (Q2), picked by its VRAM class. The whole model is the id AiRequests target;
// nobody mines it. model_id bytes MUST equal the node's `POM_TIERS_H14` / `NETWORK_MODEL_SHARDS`.

/// The whole network model — DeepSeek-V4-Flash abliterated (huihui-ai, ds4, Q2). Request target
/// and pipeline-head capability only; never downloaded, walked or mined as such.
/// `model_id` MUST equal the node's `NETWORK_MODEL_V4_FLASH_MODEL_ID`.
pub const V4_FLASH: ModelSpec = ModelSpec {
    name: "v4-flash",
    model_id: [
        0x91, 0x85, 0x70, 0xa8, 0x4e, 0x9e, 0x18, 0x32,
        0x21, 0x10, 0x17, 0x03, 0x32, 0xf8, 0x0a, 0x92,
        0x25, 0xe0, 0x78, 0xca, 0x7f, 0xc7, 0xb5, 0x4f,
        0xcd, 0xa3, 0xa7, 0xf4, 0x5a, 0xb6, 0x58, 0x10,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmY8hqif5NA9S7nWRpGHLNkDV6hue7QtSk7cKGSK8wpPq1"],
    dir_name: "V4-Flash",
    min_vram_mb: 0,
};

/// V4-Flash shard 0: layers 0-2, 8 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_0_MODEL_ID`.
pub const V4_FLASH_SHARD_0: ModelSpec = ModelSpec {
    name: "v4-flash-shard-0",
    model_id: [
        0xd1, 0x63, 0x3f, 0x54, 0xd8, 0x57, 0x4c, 0x86,
        0xff, 0xe6, 0x14, 0x99, 0x4c, 0xf2, 0xd6, 0x10,
        0x01, 0x99, 0x4e, 0x75, 0x18, 0x61, 0xe3, 0x59,
        0xb9, 0xc7, 0x04, 0x83, 0x40, 0xdb, 0x10, 0x80,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmcS1hAPeJvfE7VbHu5Kmb9rESEQFYuvE92daGVrC2H5QB"],
    dir_name: "V4-Flash-shard-0",
    min_vram_mb: 6_000,
};

/// V4-Flash shard 1: layers 3-7, 12 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_1_MODEL_ID`.
pub const V4_FLASH_SHARD_1: ModelSpec = ModelSpec {
    name: "v4-flash-shard-1",
    model_id: [
        0xad, 0x0b, 0x32, 0x63, 0x12, 0x42, 0xca, 0x0c,
        0x42, 0xde, 0xaa, 0x97, 0x79, 0x58, 0x50, 0xb1,
        0x4a, 0x73, 0x4e, 0x7e, 0xd3, 0x30, 0xfa, 0xc7,
        0x3c, 0xa4, 0xff, 0x04, 0x7a, 0xf9, 0xaa, 0xb5,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmZz99HvaeEAysHeX4ToJ7FAE4BdZqFXwWcCoiL5dww7A4"],
    dir_name: "V4-Flash-shard-1",
    min_vram_mb: 10_500,
};

/// V4-Flash shard 2: layers 8-12, 12 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_2_MODEL_ID`.
pub const V4_FLASH_SHARD_2: ModelSpec = ModelSpec {
    name: "v4-flash-shard-2",
    model_id: [
        0x28, 0x74, 0x21, 0x89, 0x10, 0x1a, 0x56, 0x75,
        0x94, 0xe7, 0x85, 0x58, 0xe4, 0xdb, 0xbd, 0x21,
        0xb1, 0xa8, 0x74, 0xb5, 0xfb, 0x3d, 0xcf, 0xac,
        0x77, 0xad, 0xf1, 0xac, 0x45, 0x54, 0x35, 0xff,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmR4ZixmaPkPd2nKKuKKRHZJsNnoPwzWbyjRr8bSrVHcJS"],
    dir_name: "V4-Flash-shard-2",
    min_vram_mb: 10_500,
};

/// V4-Flash shard 3: layers 13-19, 16 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_3_MODEL_ID`.
pub const V4_FLASH_SHARD_3: ModelSpec = ModelSpec {
    name: "v4-flash-shard-3",
    model_id: [
        0x8d, 0x64, 0x04, 0x5e, 0xf4, 0x40, 0xa8, 0x09,
        0xc7, 0x18, 0x9e, 0xc0, 0x6c, 0x5b, 0xe3, 0xff,
        0xc2, 0x8b, 0x06, 0x94, 0x67, 0xe7, 0x8e, 0x78,
        0x0a, 0xbf, 0xd1, 0xd8, 0x4f, 0xf7, 0x3d, 0x90,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmXraegdj7Co3YJAmoWM3wR8dwKCP548UaXGRHwDSRQXRq"],
    dir_name: "V4-Flash-shard-3",
    min_vram_mb: 14_500,
};

/// V4-Flash shard 4: layers 20-29, 24 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_4_MODEL_ID`.
pub const V4_FLASH_SHARD_4: ModelSpec = ModelSpec {
    name: "v4-flash-shard-4",
    model_id: [
        0x9e, 0xb2, 0x3e, 0x53, 0x97, 0x6d, 0xfb, 0x66,
        0x05, 0xaa, 0x65, 0x7d, 0xec, 0x94, 0x7c, 0xa7,
        0x87, 0x7a, 0x21, 0x35, 0xc5, 0x12, 0x27, 0x04,
        0x9c, 0xa4, 0xd1, 0x62, 0x5e, 0x84, 0xeb, 0x19,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmZ28mLQbMkf3vg4aYA3XDYvnhPL6HAodLWoU1HZ2rubeY"],
    dir_name: "V4-Flash-shard-4",
    min_vram_mb: 21_000,
};

/// V4-Flash shard 5: layers 30-42, 32 GB class. `model_id` MUST equal the node's
/// `V4_FLASH_SHARD_5_MODEL_ID`.
pub const V4_FLASH_SHARD_5: ModelSpec = ModelSpec {
    name: "v4-flash-shard-5",
    model_id: [
        0xc4, 0x81, 0x8f, 0x6b, 0x13, 0x76, 0x69, 0xb3,
        0xf0, 0x01, 0x78, 0x5b, 0xa1, 0x09, 0xf2, 0x55,
        0x92, 0x0a, 0x66, 0xfa, 0x21, 0x9f, 0xcf, 0x32,
        0x86, 0xb0, 0xbe, 0x0f, 0xde, 0x50, 0x97, 0x42,
    ],
    format: ModelFormat::GgufDeepSeekV4,
    tokenizer_cid: "",
    weight_cids: &["QmbZjCfrPdi3tvAYg9bZKqsyaxeBN98huSiq8V7xSSmsG5"],
    dir_name: "V4-Flash-shard-5",
    min_vram_mb: 30_000,
};

/// The six mainnet shards, layer order (`V4_FLASH_SHARDS[k]` is tier `6 + k`).
pub const V4_FLASH_SHARDS: [&ModelSpec; 6] =
    [&V4_FLASH_SHARD_0, &V4_FLASH_SHARD_1, &V4_FLASH_SHARD_2, &V4_FLASH_SHARD_3, &V4_FLASH_SHARD_4, &V4_FLASH_SHARD_5];

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
};

/// The network model of one network: the whole model requests target, its shards in tier
/// order, their layer ranges, the layer count and the packed head.
pub struct NetworkModelSpec {
    pub whole: &'static ModelSpec,
    pub shards: &'static [&'static ModelSpec],
    pub layers: &'static [(u32, u32)],
    pub n_layer: u32,
    pub head_cid: &'static str,
    pub head_digest_hex: &'static str,
    /// Hardware tier → shard index.
    pub shard_of_tier: fn(Tier) -> usize,
}

fn mainnet_shard_of_tier(tier: Tier) -> usize {
    match tier {
        Tier::VeryLight => 0,
        Tier::Light => light_shard() as usize,
        Tier::Default => 3,
        Tier::High => 4,
        Tier::VeryHigh => 5,
    }
}

fn devnet_shard_of_tier(tier: Tier) -> usize {
    match tier {
        Tier::VeryLight | Tier::Light | Tier::Default => 0,
        Tier::High | Tier::VeryHigh => 1,
    }
}

pub static NETWORK_MODEL_MAINNET: NetworkModelSpec = NetworkModelSpec {
    whole: &V4_FLASH,
    shards: &[&V4_FLASH_SHARD_0, &V4_FLASH_SHARD_1, &V4_FLASH_SHARD_2, &V4_FLASH_SHARD_3, &V4_FLASH_SHARD_4, &V4_FLASH_SHARD_5],
    layers: &[(0, 2), (3, 7), (8, 12), (13, 19), (20, 29), (30, 42)],
    n_layer: 43,
    head_cid: "QmZU3jvigYXpCVFSzfjJoJA3Yy27ryujyvp67gWhcq37iZ",
    head_digest_hex: "a555a9dcb5dac4d7c8c2609781d7013d8a8d8b6416429ae50254a093c1181df2",
    shard_of_tier: mainnet_shard_of_tier,
};

pub static NETWORK_MODEL_SPLIT9B: NetworkModelSpec = NetworkModelSpec {
    whole: &SPLIT9B_WHOLE,
    shards: &[&SPLIT9B_SHARD_0, &SPLIT9B_SHARD_1],
    layers: &[(0, 15), (16, 31)],
    n_layer: 32,
    head_cid: "QmQSC2sjKaAqZH4E6HGbtkowWqGKBrHx4JvAWmXsTpNdDC",
    head_digest_hex: "1f22b2133c40b16d0ba15dffb4823f889b76eb834fe6c90dedc6a38d504a46cb",
    shard_of_tier: devnet_shard_of_tier,
};

/// The network model of the selected network: split V4-Flash, or the two-shard 9B bench
/// on devnet (`--devnet`).
pub fn network_model() -> &'static NetworkModelSpec {
    if crate::pom::is_devnet() {
        &NETWORK_MODEL_SPLIT9B
    } else {
        &NETWORK_MODEL_MAINNET
    }
}

/// First tier index of the network model — mirror of the node's `NETWORK_MODEL_TIER`.
pub const NETWORK_MODEL_TIER: u8 = 5;


/// Which of the two 12 GB shards (1 or 2) a `Tier::Light` GPU mines. Spread by the mining
/// address at startup so the two are served without coordination; `--force-model shard-2`
/// pins it.
static LIGHT_SHARD: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);

pub fn set_light_shard(k: u8) {
    LIGHT_SHARD.store(if k == 2 { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

pub fn light_shard() -> u8 {
    LIGHT_SHARD.load(std::sync::atomic::Ordering::Relaxed)
}

/// The shard a hardware tier mines under H14 on the selected network.
pub fn shard_for_tier(tier: Tier) -> &'static ModelSpec {
    let nm = network_model();
    nm.shards[(nm.shard_of_tier)(tier)]
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
        || *model_id == KIMI_LINEAR_48B.model_id
        || shard_index(model_id).is_some()
}

/// Mirror of the node's per-block tier table (`POM_TIERS_H6`, then `POM_TIERS_H14`), recomputed
/// from the block DAA. Below the H6 gate this binary refuses to mine (None) — it never produces a
/// pre-H6-era block. From the H14 gate only a shard has a tier: the lineup is paused and the
/// whole network model is never mined.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    if daa >= crate::pom::h14_activation_daa() {
        return shard_index(model_id).map(|k| NETWORK_MODEL_TIER + 1 + k);
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

/// True once the H14 model split has a scheduled DAA.
pub fn h14_staged() -> bool {
    crate::pom::h14_activation_daa() != u64::MAX
}

/// DAA marking the latest scheduled era for startup staging (VRAM ladder + initial mining model).
/// The miner does NOT idle until the crossing — it stages against the latest scheduled lineup,
/// prefetches every scheduled era (`pom_models_all_eras`), and hot-swaps the resident model at
/// the crossing (`pom_gpu::advance_mining_tier_if_due`).
pub fn staging_daa() -> u64 {
    if h14_staged() {
        crate::pom::h14_activation_daa()
    } else {
        crate::pom::pom_v3_activation_daa()
    }
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
    if daa >= crate::pom::h14_activation_daa() {
        return Some(shard_for_tier(tier));
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
    let gates = vec![crate::pom::pom_v3_activation_daa(), crate::pom::h14_activation_daa(), staging_daa()];
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
    pom_model_for_tier(staging_daa(), tier).expect("the staging era carries every tier")
}

/// Resolves a model name/id.
pub const REGISTRY: &[&ModelSpec] = &[
    &QWEN3_5_9B_ABLITERATED,
    &GLM_4_9B_0414,
    &GEMMA_4_12B_ABLITERATED,
    &QWEN3_6_27B,
    &KIMI_LINEAR_48B,
    &V4_FLASH,
    &V4_FLASH_SHARD_0,
    &V4_FLASH_SHARD_1,
    &V4_FLASH_SHARD_2,
    &V4_FLASH_SHARD_3,
    &V4_FLASH_SHARD_4,
    &V4_FLASH_SHARD_5,
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
        // The last H6-era score: just under the H14 gate (or the top of the range while H14 is
        // unscheduled), at/after every other gate on any network.
        let daa = crate::pom::h14_activation_daa().saturating_sub(1).max(crate::pom::pom_v3_activation_daa());
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

        // Every lineup model is a mineable tier in the H6 era; the network model and its shards
        // are not.
        for spec in REGISTRY.iter().take(5) {
            assert!(is_pom_model(&spec.model_id), "{} is not a PoM model", spec.name);
            assert!(pom_tier_index(&spec.model_id, daa).is_some(), "{} has no tier", spec.name);
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
            for t in [Tier::VeryLight, Tier::Light, Tier::Default, Tier::High, Tier::VeryHigh] {
                assert!((nm.shard_of_tier)(t) < nm.shards.len());
            }
            assert_eq!((nm.shard_of_tier)(Tier::VeryHigh), nm.shards.len() - 1);
        }

        // Below the gate the miner produces nothing rather than a block the node would reject.
        let gate = crate::pom::pom_v3_activation_daa();
        if gate > 0 {
            assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, gate - 1), None);
            assert!(pom_model_for_tier(gate - 1, Tier::Default).is_none());
        }
    }

    /// The H14 era — mirror of the node's `POM_TIERS_H14`: shards are tiers 6-11, the lineup is
    /// paused, the whole model is never a mined tier. `u64::MAX` is at/after the gate on any
    /// network, scheduled or not.
    #[test]
    fn h14_era_mines_shards_only() {
        let daa = u64::MAX;
        for (k, spec) in V4_FLASH_SHARDS.iter().enumerate() {
            assert_eq!(pom_tier_index(&spec.model_id, daa), Some(NETWORK_MODEL_TIER + 1 + k as u8), "{}", spec.name);
            assert_eq!(shard_index(&spec.model_id), Some(k as u8));
            assert_eq!(spec.dir_name, format!("V4-Flash-shard-{}", k));
        }
        assert_eq!(pom_tier_index(&V4_FLASH.model_id, daa), None);
        for spec in REGISTRY.iter().take(5) {
            assert_eq!(pom_tier_index(&spec.model_id, daa), None, "{} must be paused", spec.name);
        }
        // one shard per card class; the 12 GB class follows the light-shard choice
        assert_eq!(pom_model_for_tier(daa, Tier::VeryLight).unwrap().model_id, V4_FLASH_SHARD_0.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::Default).unwrap().model_id, V4_FLASH_SHARD_3.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::High).unwrap().model_id, V4_FLASH_SHARD_4.model_id);
        assert_eq!(pom_model_for_tier(daa, Tier::VeryHigh).unwrap().model_id, V4_FLASH_SHARD_5.model_id);
        set_light_shard(2);
        assert_eq!(pom_model_for_tier(daa, Tier::Light).unwrap().model_id, V4_FLASH_SHARD_2.model_id);
        set_light_shard(1);
        assert_eq!(pom_model_for_tier(daa, Tier::Light).unwrap().model_id, V4_FLASH_SHARD_1.model_id);
        set_light_shard(7);
        assert_eq!(light_shard(), 1);
        // every network-model CID decodes to its model_id (CIDv0[2..34])
        for spec in REGISTRY.iter().skip(5) {
            let raw = base58btc_decode(spec.weight_cids[0]);
            assert_eq!(&raw[..2], &[0x12, 0x20], "{}", spec.name);
            assert_eq!(&raw[2..34], &spec.model_id, "{}", spec.name);
        }
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
