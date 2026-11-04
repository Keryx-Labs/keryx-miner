//! PoM v5 — byte-exact mirror of the node's
//! `consensus/core/src/pom_v5.rs`. Salts, rounds and the tile space MUST stay bit-identical.

use crate::pom::{blake, mix64, verify_merkle, WeightIndex};
use crate::pom_v3::{fold64, snippet_fold};
use crate::pom_v4::{
    v4_first_offset, v4_initial_state, v4_state_root, v4_tile_subtree_root, v4_transition, PomProofV4, PomV4RangeProof, POM_V4_D, POM_V4_K,
    POM_V4_OFFSET_STEP_SALT, POM_V4_SNIPPET_BYTES, POM_V4_TILE_BYTES, POM_V4_TILE_CHUNKS, POM_V4_TILE_SUBTREE_DEPTH,
};
use anyhow::{anyhow, Result};

/// Bytes of one pad entry.
pub const POM_V5_P_ENTRY_BYTES: usize = 32;
/// Pad entries per pad tile.
pub const POM_V5_TILE_ENTRIES: u64 = (POM_V4_TILE_BYTES / POM_V5_P_ENTRY_BYTES) as u64;
/// ARX rounds per pad entry.
pub const POM_V5_P_ROUNDS: u32 = 1_920;
/// Context pad tiles of each H14 tier; 0 = no pad. Mirror of the node's `POM_V5_PAD_TILES_H14`.
pub const POM_V5_PAD_TILES_H14: [u64; 5] = [638_976, 3_213_312, 4_804_608, 6_471_680, 2_450_432];
/// Context pad tiles of each H14 tier from the node `pad_margin_activation`. Mirror of `POM_V5_PAD_TILES_H16`.
pub const POM_V5_PAD_TILES_H16: [u64; 5] = [434_176, 3_008_512, 4_599_808, 6_266_880, 2_245_632];

/// Context pad tiles of `tier` at `daa`; 0 = no pad.
pub fn pom_v5_pad_tiles(tier: u8, daa: u64) -> u64 {
    let table = if daa >= crate::pom::pad_margin_activation_daa() { &POM_V5_PAD_TILES_H16 } else { &POM_V5_PAD_TILES_H14 };
    table.get(tier as usize).copied().unwrap_or(0)
}

const PAD_DOMAIN: &[u8] = b"KERYX-POM-V5-PAD";
/// IV folded into the pad seed words.
pub const PAD_IV: [u64; 4] = [0x510e_527f_ade6_82d1, 0x9b05_688c_2b3e_6c1f, 0x1f83_d9ab_fb41_bd6b, 0x5be0_cd19_137e_2179];

/// Seed of the context pad of `tier`.
pub fn v5_pad_seed(tier: u8, model_id: &[u8; 32], r_t: &[u8; 32]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(PAD_DOMAIN.len() + 1 + 64);
    buf.extend_from_slice(PAD_DOMAIN);
    buf.push(tier);
    buf.extend_from_slice(model_id);
    buf.extend_from_slice(r_t);
    blake(&buf)
}

#[inline(always)]
fn round4(a: &mut u64, b: &mut u64, c: &mut u64, d: &mut u64) {
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(32);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(24);
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(16);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(63);
}

/// The four seed words the GPU fill kernel starts from, IVs already folded in.
pub fn v5_pad_seed_words(pad_seed: &[u8; 32]) -> [u64; 4] {
    let w = |i: usize| u64::from_le_bytes(pad_seed[i * 8..i * 8 + 8].try_into().unwrap());
    [w(0) ^ PAD_IV[0], w(1) ^ PAD_IV[1], w(2) ^ PAD_IV[2], w(3) ^ PAD_IV[3]]
}

fn pad_entry_into(words: &[u64; 4], index: u64, out: &mut [u8]) {
    let (mut a, mut b, mut c, mut d) = (words[0], words[1] ^ index, words[2], words[3]);
    for _ in 0..POM_V5_P_ROUNDS {
        round4(&mut a, &mut b, &mut c, &mut d);
    }
    out[0..8].copy_from_slice(&a.to_le_bytes());
    out[8..16].copy_from_slice(&b.to_le_bytes());
    out[16..24].copy_from_slice(&c.to_le_bytes());
    out[24..32].copy_from_slice(&d.to_le_bytes());
}

/// Pad entry `index` of the pad seeded by `pad_seed`.
pub fn v5_pad_entry(pad_seed: &[u8; 32], index: u64) -> [u8; POM_V5_P_ENTRY_BYTES] {
    let mut out = [0u8; POM_V5_P_ENTRY_BYTES];
    pad_entry_into(&v5_pad_seed_words(pad_seed), index, &mut out);
    out
}

/// Pad tile `tile`: its entries in order.
pub fn v5_pad_tile(pad_seed: &[u8; 32], tile: u64) -> Vec<u8> {
    let words = v5_pad_seed_words(pad_seed);
    let mut out = vec![0u8; POM_V4_TILE_BYTES];
    for (k, entry) in out.chunks_exact_mut(POM_V5_P_ENTRY_BYTES).enumerate() {
        pad_entry_into(&words, tile * POM_V5_TILE_ENTRIES + k as u64, entry);
    }
    out
}

/// Pad steps per walk.
pub fn v5_pad_steps(weight_tiles: u64, pad_tiles: u64) -> u64 {
    let k = POM_V4_K as u64;
    let n = weight_tiles + pad_tiles;
    ((k * pad_tiles + n / 2) / n).clamp(1, k - 1)
}

/// Whether step `step` (1-based) reads a pad tile.
pub fn v5_is_pad_step(step: u64, pad_steps: u64) -> bool {
    let k = POM_V4_K as u64;
    (step * pad_steps) / k != ((step - 1) * pad_steps) / k
}

/// Fold of a walk state.
pub fn v5_state_fold(state: &[u8]) -> u64 {
    let mut h = 0u64;
    for r in 0..POM_V4_D {
        let w = u64::from_le_bytes(state[r * POM_V4_D..r * POM_V4_D + 8].try_into().unwrap());
        h ^= mix64(w ^ r as u64);
    }
    h
}

/// Tile offset of step 1.
pub fn v5_first_offset(seed: u64, weight_tiles: u64) -> u64 {
    v4_first_offset(seed, weight_tiles)
}

/// Tile offset of step `step + 1`.
pub fn v5_next_offset(
    seed: u64,
    step: u64,
    snippet: &[u8; POM_V4_SNIPPET_BYTES],
    state_fold: u64,
    weight_tiles: u64,
    pad_tiles: u64,
    pad_steps: u64,
) -> u64 {
    let raw = mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet) ^ state_fold);
    if v5_is_pad_step(step + 1, pad_steps) {
        weight_tiles + raw % pad_tiles
    } else {
        raw % weight_tiles
    }
}

/// Runs the v5 walk; returns the derived `final_state`.
fn walk<F>(seed: u64, weight_tiles: u64, pad_tiles: u64, pad_seed: &[u8; 32], mut fetch: F) -> Result<u64>
where
    F: FnMut(usize, u64) -> Result<Option<Vec<u8>>>,
{
    if weight_tiles == 0 || pad_tiles == 0 {
        return Err(anyhow!("no tile space for the v5 walk"));
    }
    let pad_steps = v5_pad_steps(weight_tiles, pad_tiles);
    let mut state = v4_initial_state(seed);
    let mut off = v5_first_offset(seed, weight_tiles);
    for step in 1..=POM_V4_K {
        let tile = match fetch(step - 1, off)? {
            Some(t) => t,
            None => v5_pad_tile(pad_seed, off - weight_tiles),
        };
        if step < POM_V4_K {
            let snippet: [u8; POM_V4_SNIPPET_BYTES] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v5_next_offset(seed, step as u64, &snippet, v5_state_fold(&state), weight_tiles, pad_tiles, pad_steps);
        }
        state = v4_transition(&state, &tile, step as u32);
    }
    Ok(fold64(&v4_state_root(&state)))
}

/// Builds the v5 proof of `seed` and its derived `final_state`.
pub fn build_proof_v5(tier: u8, seed: u64, index: &WeightIndex, pad_seed: &[u8; 32], pad_tiles: u64) -> Result<(PomProofV4, u64)> {
    let weight_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
    let mut proof = PomProofV4 { tier, tiles: Vec::with_capacity(POM_V4_K), merkle: Vec::with_capacity(POM_V4_K) };
    let final_state = walk(seed, weight_tiles, pad_tiles, pad_seed, |_, off| {
        if off < weight_tiles {
            let mut t = Vec::with_capacity(POM_V4_TILE_BYTES);
            for c in 0..POM_V4_TILE_CHUNKS {
                t.extend_from_slice(&index.read_chunk_bytes(off * POM_V4_TILE_CHUNKS + c));
            }
            let path = index.merkle_path(off * POM_V4_TILE_CHUNKS)[POM_V4_TILE_SUBTREE_DEPTH as usize..].to_vec();
            proof.tiles.push(t.clone());
            proof.merkle.push(PomV4RangeProof { path });
            Ok(Some(t))
        } else {
            proof.tiles.push(Vec::new());
            proof.merkle.push(PomV4RangeProof { path: Vec::new() });
            Ok(None)
        }
    })?;
    Ok((proof, final_state))
}

/// Pre-submit self-check, as the node verifies: returns the derived `final_state`.
pub fn verify_proof_v5(seed: u64, proof: &PomProofV4, r_t: &[u8; 32], n_chunks: u64, pad_seed: &[u8; 32], pad_tiles: u64) -> Result<u64> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(anyhow!("v5 proof wrong shape"));
    }
    let weight_tiles = n_chunks / POM_V4_TILE_CHUNKS;
    walk(seed, weight_tiles, pad_tiles, pad_seed, |i, off| {
        let step = i + 1;
        if off < weight_tiles {
            if proof.tiles[i].len() != POM_V4_TILE_BYTES {
                return Err(anyhow!("v5 weight tile wrong shape at step {step}"));
            }
            if !verify_merkle(v4_tile_subtree_root(&proof.tiles[i]), off, &proof.merkle[i].path, r_t) {
                return Err(anyhow!("v5 tile fails range proof at step {step}"));
            }
            Ok(Some(proof.tiles[i].clone()))
        } else {
            if !proof.tiles[i].is_empty() || !proof.merkle[i].path.is_empty() {
                return Err(anyhow!("v5 pad position not empty at step {step}"));
            }
            Ok(None)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom_v4::POM_V4_CHUNK_BYTES;

    #[test]
    fn pad_matches_the_node_vectors() {
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        assert_eq!(hex::encode(seed), "cb678ed7a91fb53d40ad3b966d0219e92c75ead7761d6404bd533cc7ddf34f96");
        assert_eq!(hex::encode(v5_pad_entry(&seed, 5)), "0601327f5974ca61e9a78cd447afdda4cd64050bac6ddf552ce1a165f04727e0");
        assert_eq!(hex::encode(blake(&v5_pad_tile(&seed, 3))), "fa8d888e763b63542c23ad2a3e2ac715bf180f51f6b0c357a0399595593d55c2");
    }

    #[test]
    fn walk_matches_the_node_vector() {
        let mut blob = vec![0u8; 64 * POM_V4_TILE_BYTES + 3 * POM_V4_CHUNK_BYTES];
        let mut h = 0xC0FFEEu64;
        for b in blob.iter_mut() {
            h = crate::pom::mix64(h);
            *b = h as u8;
        }
        let index = crate::pom::index_from_ram(blob);
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        let (proof, fs) = build_proof_v5(1, 0x55AA, &index, &seed, 61).unwrap();
        assert_eq!(fs, 5_477_356_865_520_795_736);
        assert_eq!(verify_proof_v5(0x55AA, &proof, &index.r_t, index.n_chunks, &seed, 61).unwrap(), fs);
    }

    #[test]
    fn pad_tables_are_frozen() {
        assert_eq!(POM_V5_PAD_TILES_H14, [638_976, 3_213_312, 4_804_608, 6_471_680, 2_450_432]);
        assert_eq!(POM_V5_PAD_TILES_H16, [434_176, 3_008_512, 4_599_808, 6_266_880, 2_245_632]);
        for t in 0..POM_V5_PAD_TILES_H14.len() {
            assert_eq!(POM_V5_PAD_TILES_H16[t], POM_V5_PAD_TILES_H14[t] - 204_800);
            assert!(POM_V5_PAD_TILES_H16[t] > 0);
        }
    }

    #[test]
    fn pad_tiles_switch_at_the_gate() {
        let gate = crate::pom::pad_margin_activation_daa();
        assert!(gate > crate::pom::pom_v5_activation_daa());
        for t in 0..POM_V5_PAD_TILES_H14.len() {
            assert_eq!(pom_v5_pad_tiles(t as u8, gate - 1), POM_V5_PAD_TILES_H14[t]);
            assert_eq!(pom_v5_pad_tiles(t as u8, gate), POM_V5_PAD_TILES_H16[t]);
        }
        assert_eq!(pom_v5_pad_tiles(POM_V5_PAD_TILES_H14.len() as u8, gate), 0);
    }
}
