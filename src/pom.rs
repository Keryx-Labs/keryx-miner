//! Proof-of-Model — miner-side possession proof builder (build order §6).
//!
//! Byte-exact mirror of the node's verifier (`keryx-node-hardfork consensus/core/src/pom.rs`)
//! and the canonical reference (`pom-core`). The miner runs the memory-hard walk over its
//! resident weight blob; once a winning nonce is found, `build_proof` re-walks (recording the
//! trace), commits it, and opens the `t` Fiat-Shamir-selected steps with Merkle paths to the
//! tier root `R_T` and the trace root.
//!
//! The `PomProof`/`PomOpening` structs MUST keep the exact field order/types of the node's
//! (borsh wire format), and the primitives MUST stay bit-identical (the node re-derives the
//! same challenges and recomputes the same transitions). See POM_CONSENSUS_SPEC.md.

use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

pub(crate) fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::FileExt;
        return file.read_exact_at(buf, offset);
    }
    #[cfg(target_family = "windows")]
    {
        use std::os::windows::fs::FileExt;
        let mut pos = 0usize;
        while pos < buf.len() {
            let n = file.seek_read(&mut buf[pos..], offset + pos as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_exact_at: eof"));
            }
            pos += n;
        }
        return Ok(());
    }
}
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const CHUNK_WORDS: usize = 4; // 32 B chunk
const SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

/// Walk length / opening count — MUST match the node's `POM_WALK_STEPS` / `POM_OPENINGS`.
/// K=256 — chosen compromise (~25 MH/s on a 3090, solid possession).
pub const POM_WALK_STEPS: u32 = 256;
pub const POM_OPENINGS: usize = 32;

/// Merkle tree checkpoint interval: store every K-th level on disk (level 0 never stored —
/// recomputed from the GGUF on demand; root always stored).
const CHECKPOINT_INTERVAL: u32 = 6;
const CACHE_FORMAT_VERSION: u32 = 1;
const CANONICAL_LAYOUT_VERSION: u32 = 1;
const CACHE_METADATA_FILE: &str = "pom-tree.json";

// --- wire structs (field order == node's PomOpening/PomProof) ---

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomOpening {
    pub state_before: u64,
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
    pub trace_path_before: Vec<[u8; 32]>,
    pub trace_path_after: Vec<[u8; 32]>,
}

/// H4 recompute-from-chunks walk step — mirror of the node's `PomStep`. The chunk index is
/// NOT carried (the verifier derives `state % N` while re-walking).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomStep {
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
    /// H4 recompute-from-chunks walk record. `None` on every pre-H4 proof. MUST keep the exact
    /// field order/types of the node's `PomProof::steps_v2` (borsh wire format).
    pub steps_v2: Option<Vec<PomStep>>,
    /// H6 matrix-walk witness. When present the legacy fields above are canonical placeholders
    /// (`trace_root` zeroed, empty paths/openings, `steps_v2 = None`) except `tier` (mirrored),
    /// `final_state` (= `pom_v3::fold64(roots[K])`) and `pow_value` (era pow fold of it).
    /// Trailing field, same era-exact wire mechanism as `steps_v2` — mirror of the node's.
    pub v3: Option<crate::pom_v3::PomProofV3>,
}

/// Exact pre-H4 layout of `PomProof` (no `steps_v2`) — mirror of the node's `PomProofPreH4`.
/// A pre-H4 proof MUST serialize through this so the currently-running node (7-field decode)
/// keeps accepting it byte-for-byte. See `PomProof::to_wire_bytes`.
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofPreH4 {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
}

/// Exact pre-H6 layout of `PomProof` (no `v3`) — mirror of the node's `PomProofPreV3`. A proof
/// without the v3 extension MUST serialize through this so pre-H6 nodes keep accepting it
/// byte-for-byte. See `PomProof::to_wire_bytes`.
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofPreV3 {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
    pub steps_v2: Option<Vec<PomStep>>,
}

impl From<PomProofPreV3> for PomProof {
    fn from(p: PomProofPreV3) -> Self {
        Self {
            tier: p.tier,
            trace_root: p.trace_root,
            pow_value: p.pow_value,
            final_state: p.final_state,
            initial_trace_path: p.initial_trace_path,
            final_trace_path: p.final_trace_path,
            openings: p.openings,
            steps_v2: p.steps_v2,
            v3: None,
        }
    }
}

impl From<PomProofPreH4> for PomProof {
    fn from(p: PomProofPreH4) -> Self {
        Self {
            tier: p.tier,
            trace_root: p.trace_root,
            pow_value: p.pow_value,
            final_state: p.final_state,
            initial_trace_path: p.initial_trace_path,
            final_trace_path: p.final_trace_path,
            openings: p.openings,
            steps_v2: None,
            v3: None,
        }
    }
}

impl PomProof {
    /// Canonical wire (borsh) encoding, era-exact — mirror of the node's `to_wire_bytes`: a proof
    /// without the v3 extension encodes byte-identically to the pre-H6 layout, and without the v2
    /// extension to the pre-H4 layout. The submit path MUST use this, never `borsh::to_vec`.
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        if self.v3.is_some() {
            borsh::to_vec(self).expect("PomProof borsh serialize")
        } else if self.steps_v2.is_some() {
            borsh::to_vec(&PomProofPreV3 {
                tier: self.tier,
                trace_root: self.trace_root,
                pow_value: self.pow_value,
                final_state: self.final_state,
                initial_trace_path: self.initial_trace_path.clone(),
                final_trace_path: self.final_trace_path.clone(),
                openings: self.openings.clone(),
                steps_v2: self.steps_v2.clone(),
            })
            .expect("PomProof borsh serialize")
        } else {
            borsh::to_vec(&PomProofPreH4 {
                tier: self.tier,
                trace_root: self.trace_root,
                pow_value: self.pow_value,
                final_state: self.final_state,
                initial_trace_path: self.initial_trace_path.clone(),
                final_trace_path: self.final_trace_path.clone(),
                openings: self.openings.clone(),
            })
            .expect("PomProof borsh serialize")
        }
    }

    /// Decode the canonical wire encoding, any era — mirror of the node's `from_wire_bytes`.
    pub fn from_wire_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        borsh::from_slice::<PomProof>(bytes)
            .or_else(|_| borsh::from_slice::<PomProofPreV3>(bytes).map(PomProof::from))
            .or_else(|_| borsh::from_slice::<PomProofPreH4>(bytes).map(PomProof::from))
    }
}

// --- byte-exact primitives (mirror node) ---

#[inline]
pub fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
pub fn seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ SEED_SALT)
}

/// Pre-H5 possession transition (FROZEN — produces all blocks below `H5_ACTIVATION_DAA`). The 4
/// chunk words are XOR-folded into one accumulator before a single `mix64`, so only their XOR
/// (8 bytes) is load-bearing. Kept verbatim for historical parity with the node's `transition_v1`.
#[inline]
pub fn transition_v1(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

/// H5 possession transition (active at/after `H5_ACTIVATION_DAA`). `mix64` is chained through each
/// of the 4 chunk words, so all 32 bytes are load-bearing and order-dependent — the v1 fold
/// shortcut is closed. Byte-exact mirror of the node's `transition_v2`.
#[inline]
pub fn transition_v2(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h = mix64(h ^ w);
    }
    h
}

/// Selects the era transition by `walk_v2` (from `H5_ACTIVATION_DAA` on the block's daa_score).
#[inline]
pub fn transition(state: u64, chunk: &[u64; CHUNK_WORDS], walk_v2: bool) -> u64 {
    if walk_v2 { transition_v2(state, chunk) } else { transition_v1(state, chunk) }
}

#[inline]
pub fn chunk_to_words(c: &[u8; 32]) -> [u64; CHUNK_WORDS] {
    let mut w = [0u64; CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
pub fn words_to_bytes(w: &[u64; CHUNK_WORDS]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, wi) in w.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
    }
    b
}

#[inline]
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

pub fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

#[inline]
fn pph_words(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(pre_pow_hash[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// H3 domain salt applied to the pre_pow_hash words feeding both PoM folds at/after
/// `POM_LEVEL_ACTIVATION_DAA`. Forced-update mechanism: every walk trajectory and pow value
/// changes at the gate, so pre-H3 binaries produce proofs the node rejects. The CUDA kernel
/// is unchanged — the host salts the pph words before upload (`pom_gpu::mine`).
/// Derivation: sha256("keryx-h3-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H3_PPH_SALT`.
pub const POM_H3_PPH_SALT: [u64; 4] = [0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE];

/// pph words for the era selected by `h3` (raw pre-H3, salted at/after the H3 gate).
/// These feed the POW fold in every era — H5.1 does NOT change them.
#[inline]
pub fn pph_words_for_era(pre_pow_hash: &[u8; 32], h3: bool) -> [u64; 4] {
    let mut w = pph_words(pre_pow_hash);
    if h3 {
        for (wi, si) in w.iter_mut().zip(POM_H3_PPH_SALT.iter()) {
            *wi ^= si;
        }
    }
    w
}

/// H5.1 domain salt applied to the pph words feeding the WALK SEED fold only, at/after
/// `h5_1_activation_daa()`. Emergency relaunch 2026-07-24: every walk trajectory changes at the
/// gate so pre-H5.1 blocks fail node body validation; the pow fold keeps the H3 salt (header-only
/// pow and block levels are era-stable). The CUDA kernel receives seed and pow words separately.
/// Derivation: sha256("keryx-h5.1-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H5_1_PPH_SALT`.
pub const POM_H5_1_PPH_SALT: [u64; 4] = [0x0F86D1400D3F8664, 0xC296B67C7A7A6A5B, 0x5F89AD33D961FEAA, 0xAC6C9AFDFA053580];

/// H5.2 domain salt applied to the pph words feeding the WALK SEED fold only, at/after
/// `h5_2_activation_daa()`. Chain anchoring 2026-07-25: rotating the seed salt makes every
/// pre-gate fork point of the relaunched chain permanently uncompetitive. Seed fold only —
/// the pow fold keeps the H3 salt (header-only pow and block levels are era-stable).
/// Derivation: sha256("keryx-h5.2-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H5_2_PPH_SALT`.
pub const POM_H5_2_PPH_SALT: [u64; 4] = [0x584ADE0A598D896D, 0x8783631D81BC2695, 0x2917FCF883A0B862, 0x533CCCFAC88FD614];

/// pph words feeding the SEED fold for the era selected by (`h3`, `h5_1`, `h5_2`).
#[inline]
pub fn seed_pph_words_for_era(pre_pow_hash: &[u8; 32], h3: bool, h5_1: bool, h5_2: bool) -> [u64; 4] {
    if h5_2 {
        let mut w = pph_words(pre_pow_hash);
        for (wi, si) in w.iter_mut().zip(POM_H5_2_PPH_SALT.iter()) {
            *wi ^= si;
        }
        w
    } else if h5_1 {
        let mut w = pph_words(pre_pow_hash);
        for (wi, si) in w.iter_mut().zip(POM_H5_1_PPH_SALT.iter()) {
            *wi ^= si;
        }
        w
    } else {
        pph_words_for_era(pre_pow_hash, h3)
    }
}

#[inline]
fn pom_block_seed_from_words(p: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_seed_fold` and the node's `pom_block_seed`(`_h3`/`_h5_1`).
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool, h5_1: bool, h5_2: bool) -> u64 {
    pom_block_seed_from_words(&seed_pph_words_for_era(pre_pow_hash, h3, h5_1, h5_2), timestamp, nonce)
}

/// Canonical pow value (256-bit LE) = mix64-fold of (final_state, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_pow_fold` and the node's `pom_pow_value`(`_h3`).
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32], h3: bool) -> [u8; 32] {
    let p = pph_words_for_era(pre_pow_hash, h3);
    let o0 = mix64(final_state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&o0.to_le_bytes());
    out[8..16].copy_from_slice(&o1.to_le_bytes());
    out[16..24].copy_from_slice(&o2.to_le_bytes());
    out[24..32].copy_from_slice(&o3.to_le_bytes());
    out
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root: empty leaves");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
        let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
        path.push(sib);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        idx >>= 1;
        level = next;
    }
    path
}

pub(crate) fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.
pub fn challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// The hot search walk: K data-dependent reads, returns only `state[K]` (no trace recording).
/// This is the per-nonce work; on GPU (slice 3b) this becomes the kernel over VRAM weights.
pub fn walk_final<F: Fn(u64) -> [u64; CHUNK_WORDS]>(seed: u64, n_chunks: u64, k: u32, read_chunk: F, walk_v2: bool) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off), walk_v2);
        off = state % n_chunks;
    }
    state
}

/// CPU Proof-of-Model mining (slice 3a — functional, slow). Searches nonces in
/// `nonce_start..nonce_start+max_nonces`; on the first whose `pom_pow_value <= target`,
/// re-walks to build the full `PomProof`. GPU fast-path is slice 3b. Returns the winning
/// nonce + proof, or None if the range is exhausted.
#[allow(clippy::too_many_arguments)]
pub fn mine_pom(
    index: &WeightIndex,
    tier: u8,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target: &[u8; 32],
    k: u32,
    t: usize,
    nonce_start: u64,
    max_nonces: u64,
    h3: bool,
) -> Option<(u64, PomProof)> {
    for nonce in nonce_start..nonce_start.saturating_add(max_nonces) {
        // Legacy pre-H4 path: the H5.1/H5.2 eras can never reach it (both gate after H4), so false.
        let seed = pom_block_seed(pre_pow_hash, timestamp, nonce, h3, false, false);
        // mine_pom pairs with the v1 spot-check `build_proof`, so the search walk stays v1.
        let final_state = walk_final(seed, index.n_chunks, k, |o| index.read_chunk(o), false);
        if le_leq(&pom_pow_value(final_state, pre_pow_hash, h3), target) {
            let proof = build_proof(tier, pre_pow_hash, nonce, seed, index.n_chunks, k, t, |o| index.read_chunk(o), |o| index.merkle_path(o), h3);
            return Some((nonce, proof));
        }
    }
    None
}

/// PROVER. Re-walk the (already-won) nonce recording the trace, commit it, and open the
/// `t` FS-selected steps. `read_chunk(off)` reads the 32 B chunk at canonical chunk index
/// `off` from the resident weight blob; `weight_leaves` is the precomputed per-chunk leaf
/// set (`blake(chunk_bytes)`) over the canonical layout, used to produce weight Merkle paths.
#[allow(clippy::too_many_arguments)]
pub fn build_proof<F, WP>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    n_chunks: u64,
    k: u32,
    t: usize,
    read_chunk: F,
    weight_path: WP,
    h3: bool,
) -> PomProof
where
    F: Fn(u64) -> [u64; CHUNK_WORDS],
    WP: Fn(u64) -> Vec<[u8; 32]>,
{
    let mut trace = Vec::with_capacity(k as usize + 1);
    let mut state = seed;
    trace.push(state);
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition_v1(state, &read_chunk(off));
        trace.push(state);
        off = state % n_chunks;
    }
    let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
    let trace_root = merkle_root(&trace_leaves);
    let final_state = trace[k as usize];
    let pow_value = pom_pow_value(final_state, pre_pow_hash, h3);

    let chs = challenges(pre_pow_hash, nonce, &trace_root, &pow_value, t, k);
    let openings = chs
        .iter()
        .map(|&i| {
            let i = i as usize;
            let sb = trace[i];
            let off = sb % n_chunks;
            PomOpening {
                state_before: sb,
                chunk: words_to_bytes(&read_chunk(off)),
                weight_path: weight_path(off),
                trace_path_before: merkle_proof(&trace_leaves, i),
                trace_path_after: merkle_proof(&trace_leaves, i + 1),
            }
        })
        .collect();

    PomProof {
        tier,
        trace_root,
        pow_value,
        final_state,
        initial_trace_path: merkle_proof(&trace_leaves, 0),
        final_trace_path: merkle_proof(&trace_leaves, k as usize),
        openings,
        steps_v2: None,
        v3: None,
    }
}

/// H4 PROVER (recompute-from-chunks). Re-walk the (already-won) nonce recording, for each of the
/// K steps, the 32 B chunk read and its Merkle path under R_T. No trace tree, no Fiat-Shamir
/// openings: the node re-walks all K transitions itself and derives `final_state`, so nothing is
/// taken on the prover's word. Legacy trace-tree fields are canonically empty. Byte-exact mirror
/// of the node's `verify_pom_proof_v2` expectations.
#[allow(clippy::too_many_arguments)]
pub fn build_proof_v2<F, WP>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    seed: u64,
    n_chunks: u64,
    k: u32,
    read_chunk: F,
    weight_path: WP,
    h3: bool,
    walk_v2: bool,
) -> PomProof
where
    F: Fn(u64) -> [u64; CHUNK_WORDS],
    WP: Fn(u64) -> Vec<[u8; 32]>,
{
    let mut steps = Vec::with_capacity(k as usize);
    let mut state = seed;
    for _ in 0..k {
        let off = state % n_chunks;
        let chunk_words = read_chunk(off);
        steps.push(PomStep { chunk: words_to_bytes(&chunk_words), weight_path: weight_path(off) });
        state = transition(state, &chunk_words, walk_v2);
    }
    let final_state = state;
    let pow_value = pom_pow_value(final_state, pre_pow_hash, h3);

    PomProof {
        tier,
        trace_root: [0u8; 32],
        pow_value,
        final_state,
        initial_trace_path: vec![],
        final_trace_path: vec![],
        openings: vec![],
        steps_v2: Some(steps),
        v3: None,
    }
}

/// Self-check a built v2 proof before submit — same logic the node's `verify_pom_proof_v2` runs.
/// Cheap insurance against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof_v2(proof: &PomProof, pre_pow_hash: &[u8; 32], seed: u64, n_chunks: u64, k: u32, r_t: &[u8; 32], target: &[u8; 32], h3: bool, walk_v2: bool) -> bool {
    let steps = match &proof.steps_v2 {
        Some(s) if s.len() == k as usize => s,
        _ => return false,
    };
    if proof.trace_root != [0u8; 32]
        || !proof.initial_trace_path.is_empty()
        || !proof.final_trace_path.is_empty()
        || !proof.openings.is_empty()
    {
        return false;
    }
    let mut state = seed;
    for step in steps.iter() {
        let off = state % n_chunks;
        if !verify_merkle(blake(&step.chunk), off, &step.weight_path, r_t) {
            return false;
        }
        state = transition(state, &chunk_to_words(&step.chunk), walk_v2);
    }
    if state != proof.final_state {
        return false;
    }
    let pow_value = pom_pow_value(state, pre_pow_hash, h3);
    if pow_value != proof.pow_value {
        return false;
    }
    le_leq(&pow_value, target)
}

/// Self-check a built proof before submit (same logic the node runs). Cheap insurance
/// against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32], h3: bool) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash, h3) != proof.pow_value {
        return false;
    }
    if !le_leq(&proof.pow_value, target) {
        return false;
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return false;
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return false;
    }
    let chs = challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return false;
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return false;
        }
        let state_after = transition_v1(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return false;
        }
    }
    true
}

/// Source of the raw 32 B canonical chunks for `read_chunk`.
enum ChunkSource {
    /// In-RAM chunks for the synthetic test helper (`synth_index`), built without a GGUF.
    /// Test-only: production always uses `Gguf`, so it is compiled out of release builds.
    #[cfg(test)]
    Ram(Vec<u8>),
    /// Chunks read from the memory-mapped GGUF (OS page cache handles residency; no explicit host
    /// copy). `table[j] = (canonical chunk index of tensor j's first chunk, absolute file byte
    /// offset of that chunk)`, ascending by chunk index; `read_chunk` binary-searches it.
    Gguf { mmap: memmap2::Mmap, table: Vec<(u64, u64)> },
}

/// One checkpoint level stored on disk in the sparse Merkle tree file.
struct StoredLevel {
    level: u32,  // level index in the full tree (0 = leaves, root = total_levels - 1)
    offset: u64, // byte offset within the checkpoint file
    count: u64,  // node count at this level
}

/// Sidecar written next to `pom-tree.bin`: identifies the model the tree was built from and
/// authenticates the tree bytes, so a stale or truncated cache is rebuilt instead of trusted.
#[derive(Debug, Deserialize, Serialize)]
struct CacheMetadata {
    format_version: u32,
    layout_version: u32,
    model_id: [u8; 32],
    n_chunks: u64,
    chunk_size: u32,
    checkpoint_interval: u32,
    gguf_size: u64,
    tree_size: u64,
    tree_sha256: [u8; 32],
    root: [u8; 32],
}

fn cache_metadata_path(tree_path: &Path) -> PathBuf {
    tree_path.with_file_name(CACHE_METADATA_FILE)
}

fn validate_cache_metadata(
    metadata: &CacheMetadata,
    tree_path: &Path,
    expected_model_id: [u8; 32],
    n_chunks: u64,
    gguf_size: u64,
    root: [u8; 32],
) -> Result<()> {
    if metadata.format_version != CACHE_FORMAT_VERSION
        || metadata.layout_version != CANONICAL_LAYOUT_VERSION
        || metadata.model_id != expected_model_id
        || metadata.n_chunks != n_chunks
        || metadata.chunk_size != 32
        || metadata.checkpoint_interval != CHECKPOINT_INTERVAL
        || metadata.gguf_size != gguf_size
        || metadata.root != root
    {
        return Err(anyhow!("PoM: cached tree metadata does not match the verified model or cache format"));
    }

    let tree_size = std::fs::metadata(tree_path)?.len();
    if metadata.tree_size != tree_size {
        return Err(anyhow!("PoM: cached tree length does not match its metadata"));
    }
    let digest = crate::integrity::sha256_file(tree_path, |_, _| {})?;
    if metadata.tree_sha256 != digest {
        return Err(anyhow!("PoM: cached tree SHA-256 mismatch"));
    }
    Ok(())
}

fn write_cache_metadata(tree_path: &Path, metadata: &CacheMetadata) -> Result<()> {
    let path = cache_metadata_path(tree_path);
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = NamedTempFile::new_in(dir)?;
    serde_json::to_writer_pretty(temp.as_file_mut(), metadata)?;
    temp.as_file_mut().write_all(b"\n")?;
    temp.as_file_mut().sync_all()?;
    temp.persist(&path).map_err(|e| anyhow!("persist {}: {}", path.display(), e.error))?;
    Ok(())
}

/// Canonical weight index built once at startup from the resident model: the per-chunk
/// blake3 leaves (for Merkle paths), the recomputed tier root `R_T` (sanity-checked against
/// the consensus-pinned value), and a chunk reader. Canonical layout = name-sorted GGUF
/// tensors, `floor(len/32)` 32 B chunks — identical to `pom-rt-builder` and the node.
///
/// The sparse checkpoint Merkle tree lives on disk: only every K-th level is stored
/// (multiples of `CHECKPOINT_INTERVAL`, plus the root). Unstored intermediate levels are
/// recomputed from the GGUF on demand via `merkle_path`. This cuts tree storage from ~2N
/// nodes to ~N/(2^K - 1) nodes (~63× reduction for K=6).
pub struct WeightIndex {
    pub n_chunks: u64,
    pub r_t: [u8; 32],
    /// Raw 32 B chunk reader: GGUF-backed in production, RAM-backed in synthetic tests.
    chunks: ChunkSource,
    /// Sparse checkpoint file: only stored levels are persisted (pread).
    tree_file: File,
    #[allow(dead_code)]
    tree_path: PathBuf,
    /// Stored checkpoint levels (multiples of CHECKPOINT_INTERVAL + root).
    checkpoints: Vec<StoredLevel>,
    /// Full tree depth: levels 0..total_levels-1 where total_levels-1 is the root.
    total_levels: u32,
    /// Optional in-RAM dense tree (all levels). When present, `merkle_path` is a pure lookup
    /// instead of the sparse recompute.
    dense: Option<Vec<Vec<[u8; 32]>>>,
}

impl Drop for WeightIndex {
    fn drop(&mut self) {
        // Tree is intentionally persistent across restarts (GGUF is immutable).
    }
}

/// Compute checkpoint levels from leaf count alone — purely arithmetic, no I/O.
/// Returns (checkpoints, total_levels). Only stores multiples of CHECKPOINT_INTERVAL
/// plus the root; level 0 is never stored.
fn compute_checkpoint_offsets(n_chunks: u64) -> (Vec<StoredLevel>, u32) {
    let mut checkpoints = Vec::new();
    let mut count = n_chunks;
    let mut off: u64 = 0;
    let mut level: u32 = 0;

    loop {
        // Root (count=1) is always stored; other checkpoints at multiples of K, level > 0.
        let is_checkpoint = (level > 0 && level.is_multiple_of(CHECKPOINT_INTERVAL)) || count == 1;
        if is_checkpoint {
            checkpoints.push(StoredLevel { level, offset: off, count });
        }
        if count == 1 {
            break;
        }
        if is_checkpoint {
            off += count * 32;
        }
        count = count.div_ceil(2);
        level += 1;
    }
    // level is 0-indexed root index; total_levels = root index + 1
    (checkpoints, level + 1)
}

/// Open an existing checkpoint tree file and reconstruct the WeightIndex.
/// Detects legacy full-tree files (size mismatch) and returns an error so the caller can rebuild.
fn open_existing_tree(tree_path: &Path, gguf_path: &str, expected_model_id: [u8; 32]) -> Result<WeightIndex> {
    let mut file = File::open(gguf_path)?;
    let meta = crate::gguf::GgufMeta::read(&mut file)?;
    let names = meta.sorted_names();

    // Compute n_chunks (fast — header arithmetic only, no tensor data reads).
    let mut n_chunks: u64 = 0;
    let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
    for name in &names {
        let t = &meta.tensors[name];
        let file_off = meta.tensor_data_offset + t.offset;
        let full = t.nbytes / 32;
        if full > 0 {
            table.push((n_chunks, file_off));
        }
        n_chunks += full;
    }
    if n_chunks == 0 {
        return Err(anyhow!("PoM: model produced 0 chunks"));
    }
    drop(file);

    let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
    let expected_size = checkpoints.last().map(|cp| cp.offset + 32).unwrap_or(0);
    let actual_size = std::fs::metadata(tree_path)?.len();

    // Detect legacy full-tree file: it's ~2× the checkpoint size.
    if actual_size > expected_size + expected_size {
        log::info!(
            "PoM: legacy full-tree pom-tree.bin detected ({} bytes → {} MB); will rebuild as sparse checkpoint (~{} MB for ~{}× savings)",
            actual_size,
            actual_size / 1_048_576,
            expected_size / 1_048_576,
            actual_size / expected_size.max(1),
        );
        return Err(anyhow!(
            "PoM: legacy full-tree detected ({} bytes) — rebuild with sparse checkpoints (expect ~{} bytes)",
            actual_size, expected_size
        ));
    }
    if actual_size != expected_size {
        return Err(anyhow!(
            "PoM: cached tree size mismatch (expected {}, got {}) — delete pom-tree.bin to rebuild",
            expected_size, actual_size
        ));
    }

    let tree_file = OpenOptions::new().read(true).open(tree_path)?;

    let root_cp = checkpoints.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&tree_file, &mut r_t, root_cp.offset)?;

    let metadata_path = cache_metadata_path(tree_path);
    let metadata: CacheMetadata = serde_json::from_reader(File::open(&metadata_path)?)
        .map_err(|e| anyhow!("PoM: invalid cache metadata {}: {}", metadata_path.display(), e))?;
    validate_cache_metadata(
        &metadata,
        tree_path,
        expected_model_id,
        n_chunks,
        std::fs::metadata(gguf_path)?.len(),
        r_t,
    )?;

    let mmap = unsafe { memmap2::Mmap::map(&File::open(gguf_path)?)? };
    Ok(WeightIndex {
        n_chunks,
        r_t,
        chunks: ChunkSource::Gguf { mmap, table },
        tree_file,
        tree_path: tree_path.to_path_buf(),
        checkpoints,
        total_levels,
        dense: None,
    })
}

/// Available RAM in bytes from /proc/meminfo (Linux); None if unavailable.
pub fn available_ram_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse::<u64>().ok().map(|kb| kb * 1024);
        }
    }
    None
}

impl WeightIndex {
    /// Build from a GGUF on disk (pread of each tensor's raw bytes). The bytes are the GGUF's
    /// exact on-disk quantized bytes — the same the miner serves in VRAM and the builder pinned
    /// in `R_T`. The sparse checkpoint Merkle tree is persisted to `pom-tree.bin` next to the
    /// GGUF: only every K-th level is stored (~N/(2^K-1) nodes vs ~2N for a full tree). On
    /// subsequent restarts the existing tree is reused (GGUF is immutable), avoiding a rebuild.
    pub fn build_from_gguf(path: &str, model_id: [u8; 32]) -> Result<Self> {
        let dir = std::path::Path::new(path).parent().unwrap_or_else(|| std::path::Path::new("."));
        let tree_path = dir.join("pom-tree.bin");

        // Clean up old PID-named files left by previous versions.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("pom-tree-") && name_str.ends_with(".bin") && name_str != "pom-tree.bin" {
                    log::info!("PoM: removing legacy tree file {}", entry.path().display());
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }

        // Reuse existing checkpoint tree if valid.
        if tree_path.exists() {
            match open_existing_tree(&tree_path, path, model_id) {
                Ok(idx) => {
                    log::info!("PoM: reusing cached weight index — {} chunks", idx.n_chunks);
                    return Ok(idx);
                }
                Err(e) => {
                    log::warn!("PoM: cached tree invalid ({}), rebuilding…", e);
                    let _ = std::fs::remove_file(&tree_path);
                    let _ = std::fs::remove_file(cache_metadata_path(&tree_path));
                }
            }
        }

        let mut file = File::open(path)?;
        let meta = crate::gguf::GgufMeta::read(&mut file)?;
        let names = meta.sorted_names(); // canonical order

        // Phase 0: hash leaves from GGUF chunks → write first checkpoint level (level K) to disk.
        // Process in batches of 2^K leaves, building a mini-tree per batch and writing only
        // its root (the level-K node). Uses duplicate-last for the final partial batch.
        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k; // 64 for K=6

        let mut writer = BufWriter::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tree_path)?,
        );

        let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
        let mut n_chunks: u64 = 0;
        let mut batch_buf: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);

        // Stream each tensor's raw on-disk bytes in bounded slabs (the biggest tensors are
        // multi-GB — no full-tensor buffering needed to hash 32 B chunks).
        const SLAB_CHUNKS: u64 = 1 << 16; // 2 MiB per read
        let mut slab = vec![0u8; (SLAB_CHUNKS * 32) as usize];
        for name in &names {
            let t = &meta.tensors[name];
            let file_off = meta.tensor_data_offset + t.offset;
            let full = t.nbytes / 32;
            if full > 0 {
                table.push((n_chunks, file_off));
            }
            let mut done: u64 = 0;
            while done < full {
                let take = SLAB_CHUNKS.min(full - done);
                let buf = &mut slab[..(take * 32) as usize];
                read_exact_at(&file, buf, file_off + done * 32)?;
                for c in 0..take as usize {
                    let chunk = &buf[c * 32..c * 32 + 32];
                    batch_buf.push(blake(chunk));
                    n_chunks += 1;
                    if batch_buf.len() == batch_size as usize {
                        let level_k_node = fold_levels(&batch_buf, k);
                        writer.write_all(&level_k_node)?;
                        batch_buf.clear();
                    }
                }
                done += take;
            }
        }
        // Final partial batch: fold_levels carries the partial tail the full K levels (duplicate-last).
        // Do NOT pad to batch_size — padding at level 0 changes intermediate hashes.
        if !batch_buf.is_empty() {
            let level_k_node = fold_levels(&batch_buf, k);
            writer.write_all(&level_k_node)?;
        }

        if n_chunks == 0 {
            return Err(anyhow!("PoM: model produced 0 chunks"));
        }

        // Build higher checkpoint levels (2K, 3K, ..., root) from level-K nodes.
        writer.flush()?;
        drop(writer);
        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n_chunks)?;

        let tree_size = std::fs::metadata(&tree_path)?.len();
        let tree_sha256 = crate::integrity::sha256_file(&tree_path, |_, _| {})?;
        write_cache_metadata(
            &tree_path,
            &CacheMetadata {
                format_version: CACHE_FORMAT_VERSION,
                layout_version: CANONICAL_LAYOUT_VERSION,
                model_id,
                n_chunks,
                chunk_size: 32,
                checkpoint_interval: CHECKPOINT_INTERVAL,
                gguf_size: std::fs::metadata(path)?.len(),
                tree_size,
                tree_sha256,
                root: r_t,
            },
        )?;

        let mmap = unsafe { memmap2::Mmap::map(&File::open(path)?)? };
        let tree_file = File::open(&tree_path)?;
        Ok(WeightIndex {
            n_chunks,
            r_t,
            chunks: ChunkSource::Gguf { mmap, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
            dense: None,
        })
    }

    /// 32 B chunk at canonical index `off` (panics if out of range — `off < n_chunks`).
    pub fn read_chunk(&self, off: u64) -> [u64; CHUNK_WORDS] {
        chunk_to_words(&self.read_chunk_bytes(off))
    }

    /// Raw 32 B chunk bytes — used for leaf hashing in merkle_path and the llama walk byte-gate.
    pub(crate) fn read_chunk_bytes(&self, off: u64) -> [u8; 32] {
        let mut arr = [0u8; 32];
        match &self.chunks {
            #[cfg(test)]
            ChunkSource::Ram(data) => {
                let base = (off as usize) * 32;
                arr.copy_from_slice(&data[base..base + 32]);
            }
            ChunkSource::Gguf { mmap, table } => {
                let j = table.partition_point(|&(start, _)| start <= off) - 1;
                let (start, file_off) = table[j];
                let b = (file_off + (off - start) * 32) as usize;
                arr.copy_from_slice(&mmap[b..b + 32]);
            }
        }
        arr
    }

    /// Find the stored checkpoint at `level`, panics if not found.
    fn find_checkpoint(&self, level: u32) -> &StoredLevel {
        self.checkpoints.iter().find(|cp| cp.level == level).expect("PoM: checkpoint not found")
    }

    /// Number of nodes at `level` in the full tree (0-indexed, level 0 = leaves).
    fn count_at_level(&self, level: u32) -> u64 {
        let mut count = self.n_chunks;
        for _ in 0..level {
            count = count.div_ceil(2);
        }
        count
    }

    /// Compute the hash of the subtree whose root sits `log2(span)` levels above `src_level`, rooted
    /// at source-level index `start` and covering `span` source nodes. `src_level`: 0 = GGUF chunks,
    /// >0 = stored checkpoint level. `span` is always a power of two (= 2^(target_level - src_level)).
    ///
    /// Reads ONLY the in-range source nodes (a partial subtree exists only at the right edge) and
    /// folds them EXACTLY `log2(span)` levels with per-level duplicate-last (`fold_levels`). Padding
    /// the source by clamping the last valid index — the old approach — was WRONG: it injects extra
    /// duplicated leaves that fold into a different node than the dense tree's `hash(x, x)` carry of a
    /// lone INNER node, so reconstructed siblings (and thus proofs) mismatched at right-edge offsets.
    fn compute_subtree_hash(&self, start: u64, span: u64, src_level: u32) -> [u8; 32] {
        debug_assert!(span.is_power_of_two());
        let rounds = span.trailing_zeros();
        let source_count = if src_level == 0 { self.n_chunks } else { self.find_checkpoint(src_level).count };
        if start >= source_count {
            return [0u8; 32]; // guard: a real sibling subtree always starts in range
        }
        let end = (start + span).min(source_count);
        let nodes: Vec<[u8; 32]> = if src_level == 0 {
            // Source is GGUF: read the in-range chunks via pread and hash each into a leaf.
            (start..end).map(|i| blake(&self.read_chunk_bytes(i))).collect()
        } else {
            // Source is a stored checkpoint: read the in-range nodes from file.
            let cp = self.find_checkpoint(src_level);
            (start..end)
                .map(|i| {
                    let mut buf = [0u8; 32];
                    read_exact_at(&self.tree_file, &mut buf, cp.offset + i * 32).expect("PoM checkpoint read subtree");
                    buf
                })
                .collect()
        };
        fold_levels(&nodes, rounds)
    }

    /// Build the in-RAM dense tree; afterwards `merkle_path` is a pure lookup. Reads every chunk once.
    pub fn build_dense(&mut self) {
        if self.dense.is_some() {
            return;
        }
        let mut levels: Vec<Vec<[u8; 32]>> = vec![(0..self.n_chunks).map(|i| blake(&self.read_chunk_bytes(i))).collect()];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                let r = if i + 1 < cur.len() { cur[i + 1] } else { cur[i] };
                next.push(hash_pair(&cur[i], &r));
                i += 2;
            }
            levels.push(next);
        }
        self.dense = Some(levels);
    }

    /// Inclusion path for chunk index `off`, reading stored siblings from the checkpoint file
    /// and computing unstored intermediate levels on-the-fly from the GGUF.
    /// Byte-identical to the full-tree `merkle_path`: an out-of-range sibling is the node itself.
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
        if let Some(dense) = &self.dense {
            let mut path = Vec::with_capacity(dense.len().saturating_sub(1));
            let mut idx = off as usize;
            for level in &dense[..dense.len() - 1] {
                let sib = idx ^ 1;
                path.push(if sib < level.len() { level[sib] } else { level[idx] });
                idx >>= 1;
            }
            return path;
        }
        let total_levels = self.total_levels;
        let mut path = Vec::with_capacity(total_levels as usize);
        let mut idx: u64 = off;

        for level in 0..total_levels {
            if level == total_levels - 1 {
                break; // root has no sibling
            }

            let sib_idx = idx ^ 1;
            let is_stored = level > 0 && (level.is_multiple_of(CHECKPOINT_INTERVAL) || level == total_levels - 1);

            let node = if is_stored {
                // Read sibling directly from checkpoint file.
                let cp = self.find_checkpoint(level);
                let real_idx = if sib_idx < cp.count { sib_idx } else { idx };
                let mut buf = [0u8; 32];
                read_exact_at(&self.tree_file, &mut buf, cp.offset + real_idx * 32).expect("PoM checkpoint read");
                buf
            } else {
                // Compute sibling from nearest source below.
                // If sibling index is out of range, duplicate-last: use the node itself as sibling.
                let node_count = self.count_at_level(level);
                let real_sib_idx = if sib_idx < node_count { sib_idx } else { idx };
                let src_level = (level / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL;
                let span = 1u64 << (level - src_level);
                self.compute_subtree_hash(real_sib_idx * span, span, src_level)
            };

            path.push(node);
            idx >>= 1;
        }
        path
    }
}

/// Reduce a slice of leaves straight to the single canonical root (duplicate-last each level).
/// Applied to ALL leaves at once this is the dense reference root; it is NOT safe for batched
/// sub-folds (it stops at one node, dropping the remaining `hash(x,x)` carries — the e1811a0 bug),
/// so the build/path use `fold_levels` instead. Retained as the independent dense oracle in tests.
#[cfg(test)]
#[inline]
fn merkle_root_mini(leaves: &[[u8; 32]]) -> [u8; 32] {
    debug_assert!(!leaves.is_empty());
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Reduce `batch` by EXACTLY `rounds` canonical levels — duplicate-last each round, AND keep
/// carrying a lone node via `hash(x, x)` once the batch collapses to one node before `rounds` is
/// reached. For a full `2^rounds` batch this equals `merkle_root_mini`; for a short tail it carries
/// the remaining levels, matching the dense `merkle_root` the node pins in `POM_TIERS`.
///
/// This is the fix for the sparse-build `R_T` bug: `merkle_root_mini` stops at `len == 1`, so a
/// partial batch of `m ≤ 2^(rounds-1)` nodes lands fewer than `rounds` levels up and drops the
/// remaining `hash(x, x)` carries — yielding a wrong checkpoint node (hence wrong `R_T`) for every
/// non-power-of-two `N`. A batch fold must always land exactly `rounds` levels up.
#[inline]
fn fold_levels(batch: &[[u8; 32]], rounds: u32) -> [u8; 32] {
    debug_assert!(!batch.is_empty());
    let mut level = batch.to_vec();
    for _ in 0..rounds {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Build higher checkpoint levels from the already-written level-K nodes in the tree file.
/// Reads level-K from the file, writes each higher checkpoint level (2K, 3K, ..., root),
/// and returns the checkpoint layout + R_T.
fn finalize_checkpoint_upper(
    tree_path: &std::path::Path,
    n_chunks: u64,
) -> Result<(Vec<StoredLevel>, u32, [u8; 32])> {
    let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
    let mut file_for_read = File::open(tree_path)?;
    let mut prev_offset: u64 = checkpoints[0].offset;
    let mut prev_count = checkpoints[0].count;
    let mut prev_level = checkpoints[0].level;

    // Open for appending higher levels
    let mut writer = OpenOptions::new().read(true).write(true).open(tree_path)?;
    writer.seek(SeekFrom::End(0))?;
    let mut buf_writer = BufWriter::new(writer);

    for cp in &checkpoints[1..] {
        // Fold the previous stored level up to this checkpoint's level. A regular checkpoint sits
        // CHECKPOINT_INTERVAL levels above the previous; the final (root) fold may span fewer. Batch
        // the previous level by exactly 2^rounds and fold each batch EXACTLY `rounds` levels, so a
        // partial tail carries via hash(x,x) like the dense tree. Node count per level is
        // ceil(prev_count / 2^rounds) == cp.count (ceil(ceil(n/2)/2)…=ceil(n/2^rounds)), so offsets line up.
        let rounds = cp.level - prev_level;
        let batch_size = 1u64 << rounds;
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        let mut read_idx: u64 = 0;

        while read_idx < prev_count {
            let take = batch_size.min(prev_count - read_idx);
            batch.clear();
            for i in 0..take {
                let index = read_idx + i;
                let mut node = [0u8; 32];
                read_exact_at(&file_for_read, &mut node, prev_offset + index * 32)?;
                batch.push(node);
            }
            let parent_node = fold_levels(&batch, rounds);
            buf_writer.write_all(&parent_node)?;
            read_idx += take;
        }

        buf_writer.flush()?;
        file_for_read = File::open(tree_path)?;
        prev_offset = cp.offset;
        prev_count = cp.count;
        prev_level = cp.level;
    }

    // Read R_T from the last checkpoint (root)
    let root_cp = checkpoints.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&file_for_read, &mut r_t, root_cp.offset)?;

    Ok((checkpoints, total_levels, r_t))
}

/// Runtime network selector for every DAA activation gate below (plus the PoW salt gates in
/// `pow::heavy_hash`). Set once at startup from the `--testnet` CLI flag, before any mining
/// state is built. Runtime rather than a compile-time edit so ONE binary serves both networks —
/// the testnet gate set MUST mirror the node's `TESTNET_PARAMS` exactly, the same node↔miner
/// lockstep rule as the mainnet constants.
static TESTNET_GATES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Switches every activation gate (PoM + PoW salts) to its testnet value. Called once at
/// startup when `--testnet` is passed, before mining starts.
pub fn set_testnet(enabled: bool) {
    TESTNET_GATES.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// True when the miner runs with testnet gate values (`--testnet`).
#[inline(always)]
pub fn is_testnet() -> bool {
    TESTNET_GATES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Picks the gate value for the selected network (`--testnet`).
#[inline(always)]
fn gate(mainnet: u64, testnet: u64) -> u64 {
    if is_testnet() {
        testnet
    } else {
        mainnet
    }
}

/// PoM possession activation DAA score — MUST match the node's `pom_activation`.
/// `u64::MAX` = never (dormant): mining stays on legacy kHeavyHash, no proof produced.
///
/// Mainnet: 37_780_000 (2026-06-26 18:00 UTC) — MUST equal the node's
/// MAINNET_PARAMS.pom_activation = new(37_780_000).
/// Testnet: 0 (PoM from genesis) — node TESTNET_PARAMS.pom_activation = new(0).
#[inline(always)]
pub fn pom_activation_daa() -> u64 {
    gate(37_780_000, 0)
}

/// H3 (PoM block-level hardfork) activation DAA score. At/after this score the block header
/// commits to the winning walk's `final_state` (`pomFinalState`): the node hashes it into the
/// block hash, re-checks `pom_pow_value(final_state, pre_pow_hash) <= target` header-only, and
/// derives the block level from it again (bounded pruning proof, from-scratch IBD). The miner
/// MUST fill it on submit exactly like the nonce — a post-H3 block without it is rejected
/// (`InvalidPoW` / `PomFinalStateMismatch`). The pre-PoW hash is NOT affected (the walk seed
/// derives from it, so the field lives only in the final block hash).
///
/// H3 also salts the pph words feeding both PoM folds from this score (POM_H3_PPH_SALT).
///
/// Mainnet: 43_450_000 — picked 2026-07-05 08:49 UTC (tip 43,117,871) targeting activation
/// ≈ 2026-07-05 18:00 UTC. MUST equal the node's MAINNET_PARAMS.pom_level_activation
/// = new(43_450_000).
/// Testnet: 1 — node TESTNET_PARAMS.pom_level_activation = new(1).
#[inline(always)]
pub fn pom_level_activation_daa() -> u64 {
    gate(43_450_000, 1)
}

/// H4 (coin-age + PoM verifier v2) activation DAA score. At/after this score the miner builds the
/// recompute-from-chunks proof (`build_proof_v2`: all K chunks the walk read, each Merkle-proven
/// under R_T, no trace tree / no spot-check) instead of the 32/256-opening `build_proof`. The node
/// switches its verifier at the SAME score (`coin_age_verification_activation`) — node↔miner
/// lockstep, exactly like `pom_level_activation_daa`. Mainnet H4: 54_766_000 (2026-07-18 ~20:31
/// UTC). MUST equal the node's MAINNET_PARAMS.coin_age_verification_activation (=
/// H4_ACTIVATION_DAA).
/// Testnet: 0 — node TESTNET_PARAMS.coin_age_verification_activation = new(0).
#[inline(always)]
pub fn coin_age_verification_activation_daa() -> u64 {
    gate(54_766_000, 0)
}

/// H5 activation DAA score. At/after this score the possession walk switches from the frozen v1
/// XOR-fold (`transition_v1`) to the non-foldable mix64-chained `transition_v2`, both on the GPU
/// kernel (`pom_mine.cu`, `walk_v2` param) and the CPU walk/proof path — closing the pre-H5 fold
/// shortcut. MUST equal the node's `MAINNET_PARAMS.h5_activation` (= node `H5_ACTIVATION_DAA`),
/// node↔miner lockstep exactly like `coin_age_verification_activation_daa`. Set to the relaunch tip
/// DAA — MUST equal node `MAINNET_PARAMS.h5_activation` / `H5_ACTIVATION_DAA` = 59_009_037.
/// Testnet: 0 — node TESTNET_PARAMS.h5_activation = new(0).
#[inline(always)]
pub fn h5_activation_daa() -> u64 {
    gate(59_009_037, 0)
}

/// H5.1 (emergency relaunch 2026-07-24) activation DAA score. At/after this score the walk seed
/// derives from the H5.1-salted pph words (`POM_H5_1_PPH_SALT`) — seed fold only, the pow fold
/// keeps the H3 salt. Gate = virtual daa of the isolated relaunch base. MUST equal the node's
/// `MAINNET_PARAMS.h5_1_activation` / `H5_1_ACTIVATION_DAA` = 59_027_921.
/// Testnet: 0 — node TESTNET_PARAMS.h5_1_activation = new(0).
#[inline(always)]
pub fn h5_1_activation_daa() -> u64 {
    gate(59_027_921, 0)
}

/// H5.2 chain-anchoring gate. MUST equal the node's
/// `MAINNET_PARAMS.h5_2_activation` / `H5_2_ACTIVATION_DAA` = 59_170_000.
pub fn h5_2_activation_daa() -> u64 {
    gate(59_170_000, 0)
}

/// H6 matrix-walk gate. At/after this score (the TEMPLATE's daa_score, never wall clock or tip)
/// the miner grinds the v3 walk and builds `PomProofV3`. MUST equal the node's
/// `pom_v3_activation`: mainnet 76_316_623, testnet 1000.
pub fn pom_v3_activation_daa() -> u64 {
    gate(76_316_623, 1000)
}

/// Resident possession indices, built lazily when PoM activates, keyed by MODEL (era-stable).
/// A tier POSITION shifts across eras (a lineup insertion renumbers the models below it) while
/// the model's index bytes are identical — keying by position would strand a built index at a
/// crossing. A heterogeneous rig mines several models at once (one per GPU); each model's index
/// is built once and shared (`Arc`) across every GPU on it.
static POM_INDICES: OnceLock<Mutex<HashMap<[u8; 32], Arc<WeightIndex>>>> = OnceLock::new();

fn pom_indices() -> &'static Mutex<HashMap<[u8; 32], Arc<WeightIndex>>> {
    POM_INDICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install a model's possession index (built from that resident model). Idempotent per model.
pub fn set_index(model_id: [u8; 32], index: WeightIndex) {
    if let Ok(mut g) = pom_indices().lock() {
        g.insert(model_id, Arc::new(index));
    }
}

/// Drop a model's possession index (era crossing retires the model — frees the RAM).
pub fn clear_index(model_id: &[u8; 32]) {
    if let Ok(mut g) = pom_indices().lock() {
        g.remove(model_id);
    }
}

/// The possession index for a specific model, if built.
pub fn active_index_for_model(model_id: &[u8; 32]) -> Option<Arc<WeightIndex>> {
    pom_indices().lock().ok().and_then(|g| g.get(model_id).cloned())
}

/// Any built index (lowest model_id for determinism). Used by the fallback walk, which has no
/// per-device tier assignment, and by "is any index ready" checks.
pub fn any_active_index() -> Option<([u8; 32], Arc<WeightIndex>)> {
    pom_indices().lock().ok().and_then(|g| g.iter().min_by_key(|(m, _)| **m).map(|(m, i)| (*m, i.clone())))
}

/// Test-only WeightIndex over arbitrary RAM chunks (`data` = chunk-aligned canonical bytes) —
/// real checkpoint tree + merkle paths, no GGUF.
#[cfg(test)]
pub(crate) fn index_from_ram(data: Vec<u8>) -> WeightIndex {
    use std::sync::atomic::{AtomicU64, Ordering as O};
    static UNIQ: AtomicU64 = AtomicU64::new(0);
    let uid = UNIQ.fetch_add(1, O::Relaxed);
    let tree_path = std::env::temp_dir().join(format!("keryx-pom-synth-{}-{}.bin", std::process::id(), uid));
    let _ = std::fs::remove_file(&tree_path);

    let n = (data.len() / 32) as u64;
    let k = CHECKPOINT_INTERVAL;
    let batch_size = 1u64 << k; // 64 for K=6

    // Write level-K nodes from batches of chunk leaves.
    let mut writer = BufWriter::new(
        OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
    );
    let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
    for o in 0..n as usize {
        batch.push(blake(&data[o * 32..o * 32 + 32]));
        if batch.len() == batch_size as usize {
            let level_k_node = fold_levels(&batch, k);
            writer.write_all(&level_k_node).unwrap();
            batch.clear();
        }
    }
    // Final partial batch: fold_levels carries the partial tail the full K levels (duplicate-last).
    if !batch.is_empty() {
        writer.write_all(&fold_levels(&batch, k)).unwrap();
    }
    writer.flush().unwrap();
    drop(writer);

    let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
    let tree_file = File::open(&tree_path).unwrap();
    WeightIndex {
        n_chunks: n,
        r_t,
        chunks: ChunkSource::Ram(data),
        tree_file,
        tree_path,
        checkpoints,
        total_levels,
        dense: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(off: u64) -> [u64; CHUNK_WORDS] {
        let mut c = [0u64; CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    // Synthetic WeightIndex (no GGUF) — exercises the real read_chunk + O(log N) merkle_path
    // with the sparse checkpoint tree (same structure as production).
    fn synth_index(n: u64) -> WeightIndex {
        let mut data = Vec::with_capacity(n as usize * 32);
        for o in 0..n {
            data.extend_from_slice(&words_to_bytes(&synth_chunk(o)));
        }
        index_from_ram(data)
    }

    /// Regression for the sparse-checkpoint R_T bug (commit e1811a0): the checkpoint-built root MUST
    /// equal the dense canonical root for every N — including non-power-of-two sizes whose short leaf
    /// tail OR intermediate-fold tail used to drop the `hash(x, x)` carries (`merkle_root_mini` stopped
    /// at one node). The dense reference is `merkle_root_mini` over ALL leaves at once (it reduces
    /// straight to the true root, un-batched), which is exactly what `pom-rt-builder` pins in
    /// `POM_TIERS`. Includes the report's known-broken sizes (2000, 4968, 12345, 100000).
    #[test]
    fn dense_merkle_path_matches_sparse() {
        for n in [64u64, 65, 100, 1000, 2000, 4096, 4968, 12345, 65536, 100000, 131072] {
            let mut idx = synth_index(n);
            let step = (n as usize / 37).max(1);
            let offs: Vec<u64> = (0..n).step_by(step).collect();
            let sparse: Vec<Vec<[u8; 32]>> = offs.iter().map(|&o| idx.merkle_path(o)).collect();
            idx.build_dense();
            for (k, &o) in offs.iter().enumerate() {
                assert_eq!(idx.merkle_path(o), sparse[k], "path mismatch n={n} off={o}");
            }
            let dense = idx.dense.as_ref().unwrap();
            assert_eq!(dense.last().unwrap()[0], idx.r_t, "dense root != r_t, n={n}");
        }
    }

    #[test]
    fn sparse_build_root_matches_dense_root() {
        for n in [64u64, 65, 100, 1000, 2000, 4096, 4968, 12345, 65536, 100000, 131072] {
            let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
            let dense = merkle_root_mini(&leaves);
            let idx = synth_index(n);
            assert_eq!(idx.r_t, dense, "sparse-built R_T != dense root for N={n}");
            let _ = std::fs::remove_file(&idx.tree_path);
        }
    }

    /// End-to-end check against a node-pinned root: build the sparse index from a real GGUF and
    /// assert its R_T equals the value `pom-rt-builder` pinned in the node's `POM_TIERS`. This closes
    /// the loop the synthetic test can't (real chunking: name-sorted tensors, floor(len/32), the exact
    /// on-disk quantized bytes). `#[ignore]`d — needs the GGUF on disk; run with:
    ///   KERYX_POM_TEST_GGUF=/path/model.gguf KERYX_POM_TEST_ROOT=<hex> \
    ///     cargo test --release weight_index_matches_pinned_root -- --ignored --nocapture
    #[test]
    #[ignore]
    fn weight_index_matches_pinned_root() {
        let path = std::env::var("KERYX_POM_TEST_GGUF").expect("set KERYX_POM_TEST_GGUF=/path/model.gguf");
        let expected = std::env::var("KERYX_POM_TEST_ROOT").expect("set KERYX_POM_TEST_ROOT=<hex>").to_lowercase();
        // Force a fresh build (don't reuse a possibly-stale cached tree from an older binary).
        let dir = std::path::Path::new(&path).parent().unwrap();
        let _ = std::fs::remove_file(dir.join("pom-tree.bin"));
        let _ = std::fs::remove_file(dir.join(CACHE_METADATA_FILE));
        let idx = WeightIndex::build_from_gguf(&path, [0u8; 32]).unwrap();
        let got: String = idx.r_t.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(got, expected, "R_T mismatch vs pinned root for {path}");
    }

    /// GGUF-backed `read_chunk`: lay the canonical chunks across 3 "tensors" with header + inter-
    /// tensor padding (so file offset != off*32), build the per-tensor offset table, and assert
    /// `read_chunk` (pread) returns the exact canonical chunks AND that a proof verifies — same as
    /// the RAM path, with no host copy of the weights.
    #[test]
    fn gguf_chunk_source_reads_match_and_proof_verifies() {
        let n = 1000u64;
        let uid = std::process::id();
        let gguf_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-{uid}.bin"));
        let _ = std::fs::remove_file(&gguf_path);
        let mut f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&gguf_path).unwrap();

        // 3 tensors at chunk-start boundaries, with padding so file_off is not simply off*32.
        let splits = [0u64, 400, 750, n];
        let mut table: Vec<(u64, u64)> = Vec::new();
        let mut pos: u64 = 17; // header padding
        f.seek(SeekFrom::Start(pos)).unwrap();
        for w in splits.windows(2) {
            table.push((w[0], pos));
            for o in w[0]..w[1] {
                f.write_all(&words_to_bytes(&synth_chunk(o))).unwrap();
                pos += 32;
            }
            pos += 13; // inter-tensor padding gap
            f.seek(SeekFrom::Start(pos)).unwrap();
        }
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&File::open(&gguf_path).unwrap()).unwrap() };

        // Build the sparse checkpoint tree over the canonical synth chunks, with the GGUF chunk source.
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-tree-{uid}.bin"));
        let _ = std::fs::remove_file(&tree_path);

        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k;
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
        );
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        for o in 0..n {
            batch.push(blake(&words_to_bytes(&synth_chunk(o))));
            if batch.len() == batch_size as usize {
                writer.write_all(&fold_levels(&batch, k)).unwrap();
                batch.clear();
            }
        }
        if !batch.is_empty() {
            writer.write_all(&fold_levels(&batch, k)).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);

        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
        let tree_file = File::open(&tree_path).unwrap();
        let idx = WeightIndex {
            n_chunks: n,
            r_t,
            chunks: ChunkSource::Gguf { mmap, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
            dense: None,
        };

        // Every chunk read by pread matches the canonical chunk, across all segments + padding.
        for o in 0..n {
            assert_eq!(idx.read_chunk(o), synth_chunk(o), "chunk {o}");
        }
        // A proof built from the GGUF source verifies against R_T (target 0xff..ff = first nonce wins).
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [7u8; 32];
        let target = [0xffu8; 32];
        let (nonce, proof) = mine_pom(&idx, 2, &pph, 123, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 123, nonce, false, false, false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));

        let _ = std::fs::remove_file(&gguf_path);
    }

    /// Real-GGUF byte-identity: build the index from a downloaded model and prove that chunks
    /// read by `pread` (GGUF) verify against the model's own freshly-built `R_T`. Confirms the
    /// header arithmetic (`tensor_data_offset + offset`, per-dtype sizes) addresses the exact
    /// on-disk bytes for real quant types. Ignored (needs the GGUF); run:
    /// `cargo test -p keryx-miner -- --ignored gguf_real`.
    #[test]
    #[ignore]
    fn gguf_real_model_read_chunk_byte_identical() {
        // Override with KERYX_TEST_GGUF to point at any locally downloaded model.
        let path = std::env::var("KERYX_TEST_GGUF")
            .unwrap_or_else(|_| "target/release/models/Qwen3.5-9B-abliterated/model.gguf".to_string());
        if !std::path::Path::new(&path).exists() {
            eprintln!("skip: GGUF not found at {path}");
            return;
        }
        let idx = WeightIndex::build_from_gguf(&path, [0u8; 32]).expect("build index from real GGUF");
        eprintln!("real model index: N={} chunks", idx.n_chunks);
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [3u8; 32];
        let target = [0xffu8; 32]; // max → the first nonce wins, so 1 nonce suffices
        let (nonce, proof) = mine_pom(&idx, 0, &pph, 99, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 99, nonce, false, false, false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false),
            "GGUF-pread chunks must verify against the model's R_T (byte-identity broken otherwise)"
        );
    }

    #[test]
    fn weight_index_root_matches_standalone() {
        // The prebuilt-tree root equals the standalone merkle_root over the same leaves.
        let n = 1000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        assert_eq!(idx.r_t, merkle_root(&leaves));
    }

    #[test]
    fn cache_metadata_rejects_corruption_and_wrong_model() {
        let dir = tempfile::tempdir().unwrap();
        let tree = dir.path().join("pom-tree.bin");
        std::fs::write(&tree, b"authenticated sparse tree").unwrap();
        let model_id = [0x42; 32];
        let root = [0x24; 32];
        let digest = crate::integrity::sha256_file(&tree, |_, _| {}).unwrap();
        let metadata = CacheMetadata {
            format_version: CACHE_FORMAT_VERSION,
            layout_version: CANONICAL_LAYOUT_VERSION,
            model_id,
            n_chunks: 123,
            chunk_size: 32,
            checkpoint_interval: CHECKPOINT_INTERVAL,
            gguf_size: 456,
            tree_size: std::fs::metadata(&tree).unwrap().len(),
            tree_sha256: digest,
            root,
        };

        validate_cache_metadata(&metadata, &tree, model_id, 123, 456, root).unwrap();
        assert!(validate_cache_metadata(&metadata, &tree, [0x43; 32], 123, 456, root).is_err());
        assert!(validate_cache_metadata(&metadata, &tree, model_id, 124, 456, root).is_err());

        let mut bytes = std::fs::read(&tree).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&tree, bytes).unwrap();
        assert!(validate_cache_metadata(&metadata, &tree, model_id, 123, 456, root).is_err());
    }

    #[test]
    fn cache_metadata_round_trips_through_the_sidecar_file() {
        let dir = tempfile::tempdir().unwrap();
        let tree = dir.path().join("pom-tree.bin");
        std::fs::write(&tree, b"authenticated sparse tree").unwrap();
        let metadata = CacheMetadata {
            format_version: CACHE_FORMAT_VERSION,
            layout_version: CANONICAL_LAYOUT_VERSION,
            model_id: [0x42; 32],
            n_chunks: 123,
            chunk_size: 32,
            checkpoint_interval: CHECKPOINT_INTERVAL,
            gguf_size: 456,
            tree_size: std::fs::metadata(&tree).unwrap().len(),
            tree_sha256: crate::integrity::sha256_file(&tree, |_, _| {}).unwrap(),
            root: [0x24; 32],
        };
        write_cache_metadata(&tree, &metadata).unwrap();

        let path = cache_metadata_path(&tree);
        assert_eq!(path.file_name().unwrap(), CACHE_METADATA_FILE);
        let reloaded: CacheMetadata = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        validate_cache_metadata(&reloaded, &tree, [0x42; 32], 123, 456, [0x24; 32]).unwrap();
    }

    #[test]
    fn merkle_path_matches_in_memory_proof() {
        // The checkpoint merkle_path must be byte-identical to the in-memory merkle_proof.
        let n = 4096;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();

        for off in [0, 1, n / 2, n - 2, n - 1] {
            let checkpoint_path = idx.merkle_path(off);
            let memory_path = merkle_proof(&leaves, off as usize);
            assert_eq!(checkpoint_path.len(), memory_path.len(), "path length mismatch at off={off}");
            for (i, (cp, mp)) in checkpoint_path.iter().zip(memory_path.iter()).enumerate() {
                assert_eq!(cp, mp, "path mismatch at off={off}, level={i}");
            }
        }
    }

    /// Regression for the sparse-checkpoint PATH bug: every offset's reconstructed `merkle_path`
    /// must be byte-identical to the dense `merkle_proof` for non-power-of-two N. The old
    /// `compute_subtree_hash` clamped the source to fill the span and mismatched the dense
    /// duplicate-last carry at right-edge offsets. Exhaustive over the report's broken sizes; the
    /// pre-existing test only used n=4096 (pow2) and missed it entirely.
    #[test]
    fn merkle_path_matches_dense_proof_nonpow2() {
        for n in [65u64, 100, 1000, 2000, 4968, 12345] {
            let idx = synth_index(n);
            let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
            for off in 0..n {
                assert_eq!(idx.merkle_path(off), merkle_proof(&leaves, off as usize), "path mismatch N={n} off={off}");
            }
            let _ = std::fs::remove_file(&idx.tree_path);
        }
        // Larger N: strided sweep + dense right edge (where the duplicate-last carry bites hardest).
        let n = 100_000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        for off in (0..n).step_by(257).chain(n - 300..n) {
            assert_eq!(idx.merkle_path(off), merkle_proof(&leaves, off as usize), "path mismatch N={n} off={off}");
        }
        let _ = std::fs::remove_file(&idx.tree_path);
    }

    #[test]
    fn build_then_self_verify() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let seed = pom_block_seed(&pph, 111, nonce, false, false, false);

        let proof =
            build_proof(2, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn build_v2_then_self_verify() {
        let k = 256u32;
        let idx = synth_index(4096);
        let pph = blake(b"v2-pph");
        let seed = pom_block_seed(&pph, 111, 0xabc, true, false, false);

        let proof = build_proof_v2(3, &pph, seed, idx.n_chunks, k, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true, false);
        assert_eq!(proof.tier, 3);
        assert!(proof.steps_v2.as_ref().unwrap().len() == k as usize);
        assert!(proof.openings.is_empty() && proof.trace_root == [0u8; 32]);
        assert!(verify_proof_v2(&proof, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, false));

        // Wrong seed / wrong root / wrong target all fail the self-check.
        assert!(!verify_proof_v2(&proof, &pph, seed ^ 1, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, false));
        assert!(!verify_proof_v2(&proof, &pph, seed, idx.n_chunks, k, &blake(b"wrong"), &[0xff; 32], true, false));
        assert!(!verify_proof_v2(&proof, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0u8; 32], true, false));

        // Wire round-trip: a v2 proof encodes through the pre-H6 layout (era-exact) and
        // decodes back through the fallback chain.
        let bytes = proof.to_wire_bytes();
        let back = PomProof::from_wire_bytes(&bytes).unwrap();
        assert!(verify_proof_v2(&back, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, false));
    }

    /// H5 era-gating of the walk: the frozen v1 fold and the mix64-chained v2 walk produce different
    /// final states from the same weights/seed, and a proof built for one era is rejected when
    /// re-walked under the other. Mirrors the node's `v2_walk_era_gating`.
    #[test]
    fn v2_walk_era_gating() {
        let k = 256u32;
        let idx = synth_index(4096);
        let pph = blake(b"h5-era");
        let seed = pom_block_seed(&pph, 1, 42, true, false, false);

        let p_v1 = build_proof_v2(0, &pph, seed, idx.n_chunks, k, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true, false);
        let p_v2 = build_proof_v2(0, &pph, seed, idx.n_chunks, k, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true, true);

        // Same weights + seed, different walk -> different derived final_state.
        assert_ne!(p_v1.final_state, p_v2.final_state);

        // Each proof self-checks only under its own era.
        assert!(verify_proof_v2(&p_v1, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, false));
        assert!(verify_proof_v2(&p_v2, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, true));

        // Cross-era: re-walking with the wrong transition diverges -> rejected.
        assert!(!verify_proof_v2(&p_v2, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, false));
        assert!(!verify_proof_v2(&p_v1, &pph, seed, idx.n_chunks, k, &idx.r_t, &[0xff; 32], true, true));
    }

    /// A pre-H4 proof MUST wire-encode byte-identically to the 7-field `PomProofPreH4` layout —
    /// the invariant that keeps the currently-running (pre-H4) node accepting new-miner blocks.
    #[test]
    fn pre_h4_proof_wire_bytes_are_legacy_exact() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"legacy-pph");
        let seed = pom_block_seed(&pph, 1, 7, false, false, false);
        let proof = build_proof(1, &pph, 7, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        let legacy = borsh::to_vec(&PomProofPreH4 {
            tier: proof.tier,
            trace_root: proof.trace_root,
            pow_value: proof.pow_value,
            final_state: proof.final_state,
            initial_trace_path: proof.initial_trace_path.clone(),
            final_trace_path: proof.final_trace_path.clone(),
            openings: proof.openings.clone(),
        })
        .unwrap();
        assert_eq!(proof.to_wire_bytes(), legacy);
    }

    #[test]
    fn h3_salt_changes_seed_and_pow_and_roundtrips() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"h3-pph");
        let nonce = 0xdef;
        // The salted era must diverge from the raw era on both folds.
        let seed_pre = pom_block_seed(&pph, 42, nonce, false, false, false);
        let seed_h3 = pom_block_seed(&pph, 42, nonce, true, false, false);
        assert_ne!(seed_pre, seed_h3, "H3 salt must change the walk seed");
        assert_ne!(pom_pow_value(7, &pph, false), pom_pow_value(7, &pph, true), "H3 salt must change the pow value");
        // A proof built in the H3 era verifies in the H3 era and fails in the pre-H3 era.
        let proof =
            build_proof(1, &pph, nonce, seed_h3, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true);
        assert!(verify_proof(&pph, nonce, seed_h3, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], true));
        assert!(
            !verify_proof(&pph, nonce, seed_h3, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false),
            "an H3-era proof must not verify under the unsalted fold"
        );
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce, false, false, false);
        let proof =
            build_proof(0, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            !verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0u8; 32], false),
            "zero target must fail"
        );
        assert!(
            !verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &blake(b"wrong"), &[0xff; 32], false),
            "wrong R_T must fail"
        );
    }

    #[test]
    fn cpu_mine_finds_nonce_and_proof_verifies() {
        let (k, t) = (128u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"mine-pph");
        let ts = 555;
        // Target requiring pow_value MSB <= 0x10 (~6.6% of nonces) — found within a few tries.
        let mut target = [0xffu8; 32];
        target[31] = 0x10;
        let (nonce, proof) = mine_pom(&idx, 1, &pph, ts, &target, k, t, 0, 100_000, false).expect("mine a nonce");
        let seed = pom_block_seed(&pph, ts, nonce, false, false, false);
        // The proof verifies against the same target the node would use.
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));
        assert_eq!(proof.tier, 1);
    }

}
