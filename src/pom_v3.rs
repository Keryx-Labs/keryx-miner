//! PoM v3 (H6 matrix-state walk) — wire structs and constants.
//!
//! Byte-exact mirror of the node's `consensus/core/src/pom_v3.rs` and `POM_V3_SPEC.md`.
//! Field order and types are the borsh wire format; constants and salts MUST stay
//! bit-identical to the node's.

use borsh::{BorshDeserialize, BorshSerialize};

/// State dimension (state = D x D int8, row-major).
pub const POM_V3_D: usize = 256;
/// Walk steps per nonce.
pub const POM_V3_K: usize = 256;
/// Spot-check openings per proof.
pub const POM_V3_CHECKS: usize = 32;
/// Canonical R_T chunk size (bytes).
pub const POM_V3_CHUNK_BYTES: usize = 32;
/// Tile = 2048 consecutive canonical chunks (64 KB).
pub const POM_V3_TILE_BYTES: usize = 65536;
pub const POM_V3_TILE_CHUNKS: u64 = (POM_V3_TILE_BYTES / POM_V3_CHUNK_BYTES) as u64;
/// Offset-chain snippet = first canonical chunk of a tile.
pub const POM_V3_SNIPPET_BYTES: usize = 32;
/// Chunks per opened tile column (256 B) and its subtree depth (8 leaves).
pub const POM_V3_COL_CHUNKS: u64 = (POM_V3_D / POM_V3_CHUNK_BYTES) as u64;
pub const POM_V3_COL_SUBTREE_DEPTH: u32 = 3;

/// sha256("keryx-h6-s0-row-salt")
pub const POM_V3_S0_ROW_SALT: u64 = 0x6B61F28F3CC48744;
/// sha256("keryx-h6-offset-first-salt")
pub const POM_V3_OFFSET_FIRST_SALT: u64 = 0x3F1F886D659E316A;
/// sha256("keryx-h6-offset-step-salt")
pub const POM_V3_OFFSET_STEP_SALT: u64 = 0xD4C194F3ADB3B1C7;
/// Spot-check PRF domain prefix.
pub const POM_V3_CHECKS_DOMAIN: &[u8] = b"keryx-h6-checks-v3";

/// One spot-check opening — mirror of the node's `PomV3Opening` (borsh field order).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomV3Opening {
    pub row_before: Vec<u8>,
    pub path_before: Vec<[u8; 32]>,
    pub row_after: Vec<u8>,
    pub path_after: Vec<[u8; 32]>,
    pub tile_col: Vec<u8>,
    pub col_path: Vec<[u8; 32]>,
    pub snippet_path: Vec<[u8; 32]>,
}

/// H6 walk witness — mirror of the node's `PomProofV3` (borsh field order).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofV3 {
    pub tier: u8,
    /// `root(S_0) ..= root(S_K)` — K+1 entries.
    pub roots: Vec<[u8; 32]>,
    /// `snippets[t-1]` = first 32 B of the tile read at step t — K entries.
    pub snippets: Vec<[u8; 32]>,
    /// The `POM_V3_CHECKS` openings, in PRF challenge order.
    pub checks: Vec<PomV3Opening>,
}

/// 64-bit header fold of `roots[K]` — carried as `Header.pom_final_state`.
#[inline]
pub fn fold64(root: &[u8; 32]) -> u64 {
    u64::from_le_bytes(root[..8].try_into().unwrap())
}

// --- walk primitives (byte-exact mirrors of the node's; the GPU kernel is the walk, these
// --- serve the host proof-build and the pre-submit self-check) ---

use crate::pom::{blake, merkle_proof, merkle_root, mix64, verify_merkle, WeightIndex};
use anyhow::{anyhow, Result};

#[inline]
pub fn rho8(acc: i32, tweak: u32) -> u8 {
    let mut z = (acc as u32) ^ tweak;
    z = z.wrapping_mul(0x9E3779B9);
    z ^= z >> 16;
    z = z.wrapping_mul(0x85EBCA6B);
    z ^= z >> 13;
    (z & 0xff) as u8
}

#[inline]
pub fn rho_tweak(step: u32, row: u32, col: u32) -> u32 {
    step.wrapping_mul(0x9E3779B9).wrapping_add(row.wrapping_mul(0xC2B2AE35)).wrapping_add(col.wrapping_mul(0x85EBCA6B))
}

#[inline]
pub fn dot_i8(row: &[u8], col: &[u8]) -> i32 {
    row.iter().zip(col.iter()).map(|(&a, &b)| (a as i8 as i32) * (b as i8 as i32)).sum()
}

#[inline]
pub fn snippet_fold(snippet: &[u8; POM_V3_SNIPPET_BYTES]) -> u64 {
    let mut sf = 0u64;
    for w in 0..8 {
        let word = u32::from_le_bytes(snippet[w * 4..w * 4 + 4].try_into().unwrap()) as u64;
        sf = mix64(sf ^ word);
    }
    sf
}

#[inline]
pub fn v3_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V3_OFFSET_FIRST_SALT) % n_tiles
}

#[inline]
pub fn v3_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V3_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V3_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

/// S_0 from the block seed — mix64 keystream per row.
pub fn v3_initial_state(seed: u64) -> Vec<u8> {
    let mut s = vec![0u8; POM_V3_D * POM_V3_D];
    for r in 0..POM_V3_D {
        let mut h = mix64(seed ^ POM_V3_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V3_D / 4 {
            h = mix64(h);
            s[r * POM_V3_D + k4 * 4..r * POM_V3_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

/// blake3 row leaves of one D x D state.
fn state_leaves(state: &[u8]) -> Vec<[u8; 32]> {
    (0..POM_V3_D).map(|r| blake(&state[r * POM_V3_D..(r + 1) * POM_V3_D])).collect()
}

pub fn v3_state_root(state: &[u8]) -> [u8; 32] {
    merkle_root(&state_leaves(state))
}

/// The spot-check PRF (t, x, j) triples — mirror of the node's `v3_check_points`.
pub fn v3_check_points(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    roots: &[[u8; 32]],
    snippets: &[[u8; 32]],
) -> Vec<(u32, u32, u32)> {
    let mut acc = Vec::with_capacity(POM_V3_CHECKS_DOMAIN.len() + 32 + 8 + 64);
    acc.extend_from_slice(POM_V3_CHECKS_DOMAIN);
    acc.extend_from_slice(pre_pow_hash);
    acc.extend_from_slice(&nonce.to_le_bytes());
    let mut roots_buf = Vec::with_capacity(roots.len() * 32);
    roots.iter().for_each(|r| roots_buf.extend_from_slice(r));
    acc.extend_from_slice(&blake(&roots_buf));
    let mut snip_buf = Vec::with_capacity(snippets.len() * 32);
    snippets.iter().for_each(|s| snip_buf.extend_from_slice(s));
    acc.extend_from_slice(&blake(&snip_buf));
    let prf_seed = blake(&acc);

    (0..POM_V3_CHECKS as u64)
        .map(|i| {
            let mut buf = [0u8; 40];
            buf[..32].copy_from_slice(&prf_seed);
            buf[32..].copy_from_slice(&i.to_le_bytes());
            let d = blake(&buf);
            let t = 1 + (u64::from_le_bytes(d[..8].try_into().unwrap()) % POM_V3_K as u64) as u32;
            let x = (u64::from_le_bytes(d[8..16].try_into().unwrap()) % POM_V3_D as u64) as u32;
            let j = (u64::from_le_bytes(d[16..24].try_into().unwrap()) % POM_V3_D as u64) as u32;
            (t, x, j)
        })
        .collect()
}

/// Build the v3 witness from the GPU dump (`states` = S_0..=S_K concatenated, `snippets` =
/// K x 32 B) and the host possession index (chunk reads + R_T Merkle paths).
pub fn build_proof_v3(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    states: &[u8],
    snippets_bytes: &[u8],
    index: &WeightIndex,
) -> Result<PomProofV3> {
    let d2 = POM_V3_D * POM_V3_D;
    if states.len() != (POM_V3_K + 1) * d2 || snippets_bytes.len() != POM_V3_K * POM_V3_SNIPPET_BYTES {
        return Err(anyhow!("v3 dump has wrong shape"));
    }
    let n_tiles = index.n_chunks / POM_V3_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small for the v3 walk"));
    }
    let snippets: Vec<[u8; 32]> = snippets_bytes.chunks(32).map(|c| c.try_into().unwrap()).collect();

    // Per-state leaf levels (row hashes) — kept for the opening paths.
    let leaves: Vec<Vec<[u8; 32]>> = (0..=POM_V3_K).map(|t| state_leaves(&states[t * d2..(t + 1) * d2])).collect();
    let roots: Vec<[u8; 32]> = leaves.iter().map(|l| merkle_root(l)).collect();

    // Offset chain from (seed, snippet list) — identical to the verifier's derivation.
    let mut offsets = Vec::with_capacity(POM_V3_K);
    let mut off = v3_first_offset(seed, n_tiles);
    for step in 1..=POM_V3_K as u64 {
        offsets.push(off);
        if (step as usize) < POM_V3_K {
            off = v3_next_offset(seed, step, &snippets[step as usize - 1], n_tiles);
        }
    }

    let points = v3_check_points(pre_pow_hash, nonce, &roots, &snippets);
    let checks = points
        .iter()
        .map(|&(t, x, j)| {
            let (ti, xi, ji) = (t as usize, x as usize, j as u64);
            let off = offsets[ti - 1];
            let col_first_leaf = off * POM_V3_TILE_CHUNKS + ji * POM_V3_COL_CHUNKS;
            let mut tile_col = Vec::with_capacity(POM_V3_D);
            for c in 0..POM_V3_COL_CHUNKS {
                tile_col.extend_from_slice(&index.read_chunk_bytes(col_first_leaf + c));
            }
            // The chunk leaf path minus its lowest `depth` siblings IS the aligned-subtree
            // path (the column never straddles R_T's duplicate-last edge).
            let col_path = index.merkle_path(col_first_leaf)[POM_V3_COL_SUBTREE_DEPTH as usize..].to_vec();
            let snippet_path = index.merkle_path(off * POM_V3_TILE_CHUNKS);
            PomV3Opening {
                row_before: states[(ti - 1) * d2 + xi * POM_V3_D..(ti - 1) * d2 + (xi + 1) * POM_V3_D].to_vec(),
                path_before: merkle_proof(&leaves[ti - 1], xi),
                row_after: states[ti * d2 + xi * POM_V3_D..ti * d2 + (xi + 1) * POM_V3_D].to_vec(),
                path_after: merkle_proof(&leaves[ti], xi),
                tile_col,
                col_path,
                snippet_path,
            }
        })
        .collect();
    Ok(PomProofV3 { tier, roots, snippets, checks })
}

/// Fold 8 column-chunk leaves into their aligned depth-3 subtree root.
fn col_subtree_root(col: &[u8]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = col.chunks(POM_V3_CHUNK_BYTES).map(blake).collect();
    merkle_root(&leaves)
}

/// Pre-submit self-check — same logic as the node's `verify_pom_proof_v3`. Cheap insurance
/// against emitting a block the node will reject.
pub fn verify_proof_v3(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    proof: &PomProofV3,
    r_t: &[u8; 32],
    n_chunks: u64,
) -> bool {
    if proof.roots.len() != POM_V3_K + 1 || proof.snippets.len() != POM_V3_K || proof.checks.len() != POM_V3_CHECKS {
        return false;
    }
    let n_tiles = n_chunks / POM_V3_TILE_CHUNKS;
    if n_tiles == 0 {
        return false;
    }
    if v3_state_root(&v3_initial_state(seed)) != proof.roots[0] {
        return false;
    }
    let mut offsets = Vec::with_capacity(POM_V3_K);
    let mut off = v3_first_offset(seed, n_tiles);
    for step in 1..=POM_V3_K as u64 {
        offsets.push(off);
        if (step as usize) < POM_V3_K {
            off = v3_next_offset(seed, step, &proof.snippets[step as usize - 1], n_tiles);
        }
    }
    let points = v3_check_points(pre_pow_hash, nonce, &proof.roots, &proof.snippets);
    for (point, open) in points.iter().zip(proof.checks.iter()) {
        let &(t, x, j) = point;
        let (ti, xi, ji) = (t as usize, x as usize, j as u64);
        if open.row_before.len() != POM_V3_D || open.row_after.len() != POM_V3_D || open.tile_col.len() != POM_V3_D {
            return false;
        }
        if !verify_merkle(blake(&open.row_before), xi as u64, &open.path_before, &proof.roots[ti - 1]) {
            return false;
        }
        if !verify_merkle(blake(&open.row_after), xi as u64, &open.path_after, &proof.roots[ti]) {
            return false;
        }
        let off_t = offsets[ti - 1];
        let subtree_index = off_t * (POM_V3_TILE_CHUNKS >> POM_V3_COL_SUBTREE_DEPTH) + ji;
        if !verify_merkle(col_subtree_root(&open.tile_col), subtree_index, &open.col_path, r_t) {
            return false;
        }
        if !verify_merkle(blake(&proof.snippets[ti - 1]), off_t * POM_V3_TILE_CHUNKS, &open.snippet_path, r_t) {
            return false;
        }
        if rho8(dot_i8(&open.row_before, &open.tile_col), rho_tweak(t, x, j)) != open.row_after[ji as usize] {
            return false;
        }
    }
    true
}

/// Test-only reference walk (host): S_0..=S_K + snippets + offsets over an in-RAM blob.
/// Mirrors the node's `v3_walk`; production never host-walks (the GPU kernel is the walk).
#[cfg(test)]
pub(crate) fn ref_walk(seed: u64, blob: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u64>) {
    let d2 = POM_V3_D * POM_V3_D;
    let n_tiles = (blob.len() / POM_V3_CHUNK_BYTES) as u64 / POM_V3_TILE_CHUNKS;
    assert!(n_tiles > 0);
    let mut states = Vec::with_capacity((POM_V3_K + 1) * d2);
    states.extend_from_slice(&v3_initial_state(seed));
    let mut snippets = Vec::with_capacity(POM_V3_K * POM_V3_SNIPPET_BYTES);
    let mut offsets = Vec::with_capacity(POM_V3_K);
    let mut off = v3_first_offset(seed, n_tiles);
    for step in 1..=POM_V3_K as u32 {
        let tile = &blob[(off as usize) * POM_V3_TILE_BYTES..(off as usize + 1) * POM_V3_TILE_BYTES];
        let snippet: [u8; POM_V3_SNIPPET_BYTES] = tile[..POM_V3_SNIPPET_BYTES].try_into().unwrap();
        offsets.push(off);
        snippets.extend_from_slice(&snippet);
        let prev = &states[(step as usize - 1) * d2..(step as usize) * d2].to_vec();
        let mut next = vec![0u8; d2];
        for x in 0..POM_V3_D {
            let row = &prev[x * POM_V3_D..(x + 1) * POM_V3_D];
            for j in 0..POM_V3_D {
                let col = &tile[j * POM_V3_D..(j + 1) * POM_V3_D];
                next[x * POM_V3_D + j] = rho8(dot_i8(row, col), rho_tweak(step, x as u32, j as u32));
            }
        }
        states.extend_from_slice(&next);
        if (step as usize) < POM_V3_K {
            off = v3_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    (states, snippets, offsets)
}

/// Test-only lockstep blob — byte-identical to the node's `pom_v3::tests::test_blob(16)`.
#[cfg(test)]
pub(crate) fn lockstep_blob() -> Vec<u8> {
    let n_bytes = 16 * POM_V3_TILE_BYTES + 5 * POM_V3_CHUNK_BYTES;
    let mut blob = vec![0u8; n_bytes];
    let mut h = 0xDEADBEEFu64;
    for b in blob.iter_mut() {
        h = mix64(h);
        *b = h as u8;
    }
    blob
}

#[cfg(test)]
pub(crate) const LOCKSTEP_PPH: [u8; 32] = [7u8; 32];
#[cfg(test)]
pub(crate) const LOCKSTEP_NONCE: u64 = 0x1234_5678_9ABC_DEF0;
#[cfg(test)]
pub(crate) const LOCKSTEP_SEED: u64 = 0x0F1E_2D3C_4B5A_6978;

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Pinned outputs of the NODE's reference implementation (`consensus/core/src/pom_v3.rs`)
    /// on `test_blob(16)` with the same (pph, nonce, seed) — extracted 2026-08-09. Any
    /// mismatch here is a consensus fork between miner and node.
    #[test]
    fn mirror_matches_node_vector() {
        let blob = lockstep_blob();
        let index = crate::pom::index_from_ram(blob.clone());
        assert_eq!(index.n_chunks, 32773);
        assert_eq!(hex(&index.r_t), "957cc59cac21dc53875d87044154276ff9a89a23c931372e5f5ba01ba835a62e");

        let (states, snippets, offsets) = ref_walk(LOCKSTEP_SEED, &blob);
        assert_eq!(offsets[0], 14);
        assert_eq!(offsets[1], 9);
        assert_eq!(offsets[255], 7);
        assert_eq!(hex(&snippets[..32]), "7cb1aaab42c66cac4831cefc70d15dae94cf114aeb34d0f755c24fb0e6cd94f8");
        assert_eq!(hex(&snippets[255 * 32..]), "f23bb55dd3994d376e8205ae48686407b6d760022fd4c0c340bc2e1c53cfec47");

        let d2 = POM_V3_D * POM_V3_D;
        let roots: Vec<[u8; 32]> = (0..=POM_V3_K).map(|t| v3_state_root(&states[t * d2..(t + 1) * d2])).collect();
        assert_eq!(hex(&roots[0]), "b2750257e531cf82593f5444f762705c1987f4a7a84c68b5632db180b3763f82");
        assert_eq!(hex(&roots[1]), "5abe3c3e33139591f476928f360f7feaeeef216aadcf8f10b71e12dfc05ccccf");
        assert_eq!(hex(&roots[256]), "0b1385d0a317df0b3c87181d04f4c031e41786b66fcc0b5eee7ece9dabcb91e6");
        assert_eq!(fold64(&roots[256]), 0x0bdf17a3d085130b);

        let snippet_arr: Vec<[u8; 32]> = snippets.chunks(32).map(|c| c.try_into().unwrap()).collect();
        let points = v3_check_points(&LOCKSTEP_PPH, LOCKSTEP_NONCE, &roots, &snippet_arr);
        assert_eq!(points[0], (152, 117, 215));
        assert_eq!(points[31], (167, 171, 49));

        // Proof build over the index + self-check (same logic as the node's verifier).
        let proof =
            build_proof_v3(0, &LOCKSTEP_PPH, LOCKSTEP_NONCE, LOCKSTEP_SEED, &states, &snippets, &index).unwrap();
        assert_eq!(proof.roots, roots);
        assert!(verify_proof_v3(&LOCKSTEP_PPH, LOCKSTEP_NONCE, LOCKSTEP_SEED, &proof, &index.r_t, index.n_chunks));

        // A tampered opening must fail the self-check.
        let mut bad = proof.clone();
        bad.checks[0].tile_col[17] ^= 0xFF;
        assert!(!verify_proof_v3(&LOCKSTEP_PPH, LOCKSTEP_NONCE, LOCKSTEP_SEED, &bad, &index.r_t, index.n_chunks));
    }

    /// The v3-bearing container round-trips the era-exact wire, and legacy proofs stay
    /// byte-identical to their pre-H6 encoding.
    #[test]
    fn wire_era_exact() {
        let opening = PomV3Opening {
            row_before: vec![1; POM_V3_D],
            path_before: vec![[2; 32]; 8],
            row_after: vec![3; POM_V3_D],
            path_after: vec![[4; 32]; 8],
            tile_col: vec![5; POM_V3_D],
            col_path: vec![[6; 32]; 12],
            snippet_path: vec![[7; 32]; 15],
        };
        let v3 = PomProofV3 {
            tier: 2,
            roots: vec![[8; 32]; POM_V3_K + 1],
            snippets: vec![[9; 32]; POM_V3_K],
            checks: vec![opening; POM_V3_CHECKS],
        };
        let with_v3 = crate::pom::PomProof {
            tier: 2,
            trace_root: [0u8; 32],
            pow_value: [1u8; 32],
            final_state: 42,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: Some(v3),
        };
        let bytes = with_v3.to_wire_bytes();
        // A v3 proof decodes only through the full (v3-aware) layout.
        assert!(borsh::from_slice::<crate::pom::PomProof>(&bytes).is_ok());
        assert!(borsh::from_slice::<crate::pom::PomProofPreV3>(&bytes).is_err());

        // Without v3 the wire re-encodes through the pre-H6 layout byte-identically.
        let without_v3 = crate::pom::PomProof { v3: None, steps_v2: Some(vec![]), ..with_v3.clone() };
        let legacy = borsh::to_vec(&crate::pom::PomProofPreV3 {
            tier: without_v3.tier,
            trace_root: without_v3.trace_root,
            pow_value: without_v3.pow_value,
            final_state: without_v3.final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: Some(vec![]),
        })
        .unwrap();
        assert_eq!(without_v3.to_wire_bytes(), legacy);
    }
}
