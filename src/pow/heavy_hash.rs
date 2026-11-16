use crate::pow::{hasher::HeavyHasher, xoshiro::XoShiRo256PlusPlus};
use crate::Hash;
use std::mem::MaybeUninit;

/// Domain-separation salts — must match `KERYX_MATRIX_SALT_V1/V2/V4` in
/// `consensus/pow/src/matrix.rs` exactly or the miner will derive a different
/// matrix than the node and every submitted block will be rejected.
/// v3 is intentionally skipped: it belongs to the abandoned diff-spiral chain.
const KERYX_MATRIX_SALT_V1: [u8; 32] = *b"KERYX:KeryxHash-v1:2026-04-12:xx";
const KERYX_MATRIX_SALT_V2: [u8; 32] = *b"KERYX:KeryxHash-v2:2026-05-29:xx";
const KERYX_MATRIX_SALT_V4: [u8; 32] = *b"KERYX:KeryxHash-v4:2026-06-07:xx";

/// DAA score at which the miner switches to SALT v2 — must match `pow_salt_v2_activation`
/// in network params. Miners compiled before this update will keep using v1 and their
/// blocks will be rejected after activation — that is the forced-update mechanism.
///
/// Mainnet: 17_275_000 (2026-05-30 ~15:00 UTC emergency activation) — matches the node's
/// MAINNET_PARAMS.pow_salt_v2_activation = new(17_275_000).
/// Testnet and devnet: 0 (active from genesis — no mid-chain salt transition there).
#[inline(always)]
pub fn pow_salt_v2_activation_daa() -> u64 {
    if keryx_miner::pom::is_testnet() || keryx_miner::pom::is_devnet() {
        0
    } else {
        17_275_000
    }
}

/// DAA score at which the miner switches to SALT v4 (chain relaunch on stock difficulty) —
/// must match `pow_salt_v4_activation` in network params. The matrix is generated host-side
/// here (the CUDA kernel receives the precomputed matrix), so no kernel/PTX change is needed.
///
/// Mainnet: 21_932_751 (same DAA as the old v3 gate; forks cleanly off the broken chain) —
/// matches the node's MAINNET_PARAMS.pow_salt_v4_activation = new(21_932_751).
/// Testnet and devnet: 0 (active from genesis) — node params pow_salt_v4_activation = new(0).
#[inline(always)]
pub fn pow_salt_v4_activation_daa() -> u64 {
    if keryx_miner::pom::is_testnet() || keryx_miner::pom::is_devnet() {
        0
    } else {
        21_932_751
    }
}

/// Returns the active matrix-salt version (1, 2 or 4) for a block at `daa_score`.
/// Must mirror `active_salt_version` in `consensus/pow/src/lib.rs` (compared with `>=`).
#[inline(always)]
pub fn active_salt_version(daa_score: u64) -> u8 {
    if daa_score >= pow_salt_v4_activation_daa() {
        4
    } else if daa_score >= pow_salt_v2_activation_daa() {
        2
    } else {
        1
    }
}

/// Round constants for wave_mix — same as `WAVE_MIX_KEYS` in matrix.rs.
const WAVE_MIX_KEYS: [u64; 4] = [
    0x9e3779b97f4a7c15,
    0x6c62272e07bb0142,
    0xb5ad4eceda1ce2a9,
    0x243f6a8885a308d3,
];

/// Rotation amounts — same as `WAVE_MIX_ROTATIONS` in matrix.rs.
const WAVE_MIX_ROTATIONS: [u32; 4] = [17, 31, 47, 13];

/// 4-round ARX post-processing — must be bit-for-bit identical to
/// `fn wave_mix()` in `consensus/pow/src/matrix.rs`.
#[inline(always)]
fn wave_mix(bytes: [u8; 32]) -> [u8; 32] {
    let mut w = [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ];
    for r in 0..4usize {
        w[0] = w[0].wrapping_add(w[1]).rotate_left(WAVE_MIX_ROTATIONS[0]) ^ WAVE_MIX_KEYS[r & 3];
        w[2] = w[2].wrapping_add(w[3]).rotate_left(WAVE_MIX_ROTATIONS[2]) ^ WAVE_MIX_KEYS[(r + 2) & 3];
        w[1] = w[1].wrapping_add(w[2]).rotate_left(WAVE_MIX_ROTATIONS[1]) ^ WAVE_MIX_KEYS[(r + 1) & 3];
        w[3] = w[3].wrapping_add(w[0]).rotate_left(WAVE_MIX_ROTATIONS[3]) ^ WAVE_MIX_KEYS[(r + 3) & 3];
    }
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&w[0].to_le_bytes());
    out[8..16].copy_from_slice(&w[1].to_le_bytes());
    out[16..24].copy_from_slice(&w[2].to_le_bytes());
    out[24..32].copy_from_slice(&w[3].to_le_bytes());
    out
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Matrix(pub [[u16; 64]; 64]);

impl Matrix {
    #[inline(always)]
    pub fn generate(hash: Hash, salt_version: u8) -> Self {
        // XOR the block-hash seed with the active Keryx domain salt before the PRNG.
        // Must match Matrix::generate() in consensus/pow/src/matrix.rs.
        let salt: &[u8; 32] = match salt_version {
            1 => &KERYX_MATRIX_SALT_V1,
            2 => &KERYX_MATRIX_SALT_V2,
            _ => &KERYX_MATRIX_SALT_V4,
        };
        let salted = {
            let mut bytes = hash.to_le_bytes();
            bytes.iter_mut().zip(salt.iter()).for_each(|(b, s)| *b ^= s);
            Hash::from_le_bytes(bytes)
        };
        let mut generator = XoShiRo256PlusPlus::new(salted);
        loop {
            let mat = Self::rand_matrix_no_rank_check(&mut generator);
            if mat.compute_rank() == 64 {
                return mat;
            }
        }
    }

    #[inline(always)]
    fn rand_matrix_no_rank_check(generator: &mut XoShiRo256PlusPlus) -> Self {
        Self(array_from_fn(|_| {
            let mut val = 0;
            array_from_fn(|j| {
                let shift = j % 16;
                if shift == 0 {
                    val = generator.u64();
                }
                (val >> (4 * shift) & 0x0F) as u16
            })
        }))
    }

    #[inline(always)]
    fn convert_to_float(&self) -> [[f64; 64]; 64] {
        // SAFETY: An uninitialized MaybrUninit is always safe.
        let mut out: [[MaybeUninit<f64>; 64]; 64] = unsafe { MaybeUninit::uninit().assume_init() };

        out.iter_mut().zip(self.0.iter()).for_each(|(out_row, mat_row)| {
            out_row.iter_mut().zip(mat_row).for_each(|(out_element, &element)| {
                out_element.write(f64::from(element));
            })
        });
        // SAFETY: The loop above wrote into all indexes.
        unsafe { std::mem::transmute(out) }
    }

    pub fn compute_rank(&self) -> usize {
        const EPS: f64 = 1e-9;
        let mut mat_float = self.convert_to_float();
        let mut rank = 0;
        let mut row_selected = [false; 64];
        for i in 0..64 {
            if i >= 64 {
                // Required for optimization, See https://github.com/rust-lang/rust/issues/90794
                unreachable!()
            }
            let mut j = 0;
            while j < 64 {
                if !row_selected[j] && mat_float[j][i].abs() > EPS {
                    break;
                }
                j += 1;
            }
            if j != 64 {
                rank += 1;
                row_selected[j] = true;
                for p in (i + 1)..64 {
                    mat_float[j][p] /= mat_float[j][i];
                }
                for k in 0..64 {
                    if k != j && mat_float[k][i].abs() > EPS {
                        for p in (i + 1)..64 {
                            mat_float[k][p] -= mat_float[j][p] * mat_float[k][i];
                        }
                    }
                }
            }
        }
        rank
    }

    pub fn heavy_hash(&self, hash: Hash) -> Hash {
        let hash = hash.to_le_bytes();
        // SAFETY: An uninitialized MaybrUninit is always safe.
        let mut vec: [MaybeUninit<u8>; 64] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..32 {
            vec[2 * i].write(hash[i] >> 4);
            vec[2 * i + 1].write(hash[i] & 0x0F);
        }
        // SAFETY: The loop above wrote into all indexes.
        let vec: [u8; 64] = unsafe { std::mem::transmute(vec) };

        // Matrix-vector multiplication, convert to 4 bits, and then combine back to 8 bits.
        let mut product: [u8; 32] = array_from_fn(|i| {
            let mut sum1 = 0;
            let mut sum2 = 0;
            for (j, &elem) in vec.iter().enumerate() {
                sum1 += self.0[2 * i][j] * (elem as u16);
                sum2 += self.0[2 * i + 1][j] * (elem as u16);
            }
            ((sum1 >> 10) << 4) as u8 | (sum2 >> 10) as u8
        });

        // Concatenate 4 LSBs back to 8 bit xor with sum1
        product.iter_mut().zip(hash).for_each(|(p, h)| *p ^= h);

        // Keryx wave-mix: ARX post-processing — must match the node's matrix.rs.
        let product = wave_mix(product);

        HeavyHasher::hash(Hash::from_le_bytes(product))
    }
}

pub fn array_from_fn<F, T, const N: usize>(mut cb: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut idx = 0;
    [(); N].map(|_| {
        let res = cb(idx);
        idx += 1;
        res
    })
}

#[cfg(test)]
mod tests {
    use crate::pow::heavy_hash::Matrix;
    use crate::pow::xoshiro::XoShiRo256PlusPlus;
    use crate::Hash;

    #[test]
    fn test_compute_rank() {
        let zero = Matrix([[0; 64]; 64]);
        assert_eq!(zero.compute_rank(), 0);
        let mut matrix = zero;
        let mut gen = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        matrix.0.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|val| {
                *val = gen.u64() as u16;
            })
        });
        assert_eq!(matrix.compute_rank(), 64);

        matrix.0[0] = matrix.0[1];
        assert_eq!(matrix.compute_rank(), 63);
    }

    #[test]
    fn test_heavy_hash() {
        let expected_hash = Hash::from_le_bytes([
            178, 101, 253, 94, 61, 63, 163, 211, 32, 243, 219, 71, 156, 142, 30, 187, 110, 120, 204, 88, 234, 105, 27,
            70, 148, 93, 226, 28, 116, 230, 27, 246,
        ]);
        let test_matrix = Matrix::generate(Hash::from_le_bytes([42; 32]), 1);
        let hash = Hash::from_le_bytes([
            82, 46, 212, 218, 28, 192, 143, 92, 213, 66, 86, 63, 245, 241, 155, 189, 73, 159, 229, 180, 202, 105, 159,
            166, 109, 172, 128, 136, 169, 195, 97, 41,
        ]);
        assert_eq!(test_matrix.heavy_hash(hash), expected_hash);
    }
    #[test]
    fn test_generate_matrix() {
        #[rustfmt::skip]
        let expected_matrix = Matrix([
            [1, 15, 8, 4, 9, 4, 1, 3, 9, 3, 0, 5, 8, 10, 4, 9, 4, 15, 14, 2, 15, 0, 1, 11, 9, 14, 4, 0, 11, 15, 11, 15, 11, 8, 2, 6, 0, 4, 1, 4, 5, 6, 9, 1, 13, 5, 6, 11, 12, 14, 7, 7, 4, 10, 13, 8, 12, 9, 13, 11, 1, 2, 3, 2],
            [2, 12, 12, 7, 10, 8, 15, 6, 5, 4, 8, 4, 4, 7, 12, 2, 15, 14, 3, 6, 7, 4, 0, 0, 6, 9, 6, 10, 15, 6, 13, 13, 11, 15, 8, 12, 13, 0, 4, 8, 3, 14, 1, 5, 7, 15, 1, 10, 1, 9, 14, 11, 12, 7, 7, 5, 9, 5, 0, 4, 1, 6, 15, 7],
            [2, 8, 14, 3, 4, 7, 6, 6, 6, 1, 1, 7, 14, 13, 15, 14, 11, 12, 11, 2, 15, 13, 11, 13, 12, 5, 1, 6, 14, 7, 15, 4, 7, 14, 9, 10, 4, 3, 5, 15, 12, 4, 8, 9, 5, 13, 1, 1, 10, 5, 12, 1, 1, 6, 12, 3, 2, 12, 4, 9, 4, 12, 8, 10],
            [8, 13, 13, 14, 12, 2, 10, 15, 0, 4, 5, 4, 0, 3, 15, 1, 12, 8, 9, 14, 11, 7, 15, 3, 4, 6, 4, 4, 5, 0, 8, 9, 9, 1, 11, 10, 8, 4, 5, 13, 4, 7, 4, 1, 5, 3, 7, 8, 1, 12, 7, 7, 1, 5, 13, 7, 9, 0, 11, 6, 9, 15, 9, 4],
            [8, 8, 7, 15, 4, 12, 8, 1, 14, 0, 6, 1, 8, 5, 12, 12, 13, 6, 9, 14, 14, 11, 0, 6, 10, 12, 3, 14, 12, 11, 7, 15, 8, 5, 0, 4, 1, 13, 0, 6, 9, 12, 15, 13, 11, 9, 0, 4, 7, 1, 9, 7, 14, 11, 4, 13, 2, 3, 12, 15, 12, 12, 4, 4],
            [3, 11, 3, 3, 2, 4, 6, 1, 1, 0, 14, 5, 15, 9, 3, 12, 5, 5, 2, 0, 9, 6, 6, 7, 5, 15, 15, 0, 5, 4, 6, 8, 2, 11, 6, 15, 9, 4, 9, 8, 1, 15, 4, 1, 14, 10, 6, 4, 15, 13, 12, 4, 14, 1, 4, 15, 4, 5, 9, 7, 12, 8, 0, 6],
            [5, 10, 1, 10, 5, 4, 2, 5, 13, 0, 12, 9, 7, 9, 15, 12, 10, 1, 1, 11, 0, 10, 9, 13, 15, 9, 3, 11, 4, 6, 8, 0, 8, 6, 14, 7, 3, 15, 8, 13, 6, 11, 0, 15, 5, 1, 3, 3, 10, 1, 15, 8, 10, 2, 5, 7, 0, 11, 11, 1, 12, 8, 4, 6],
            [7, 13, 7, 13, 7, 6, 15, 10, 12, 2, 12, 5, 4, 13, 8, 6, 13, 10, 12, 6, 8, 9, 10, 8, 7, 5, 13, 0, 8, 5, 10, 0, 0, 15, 13, 3, 3, 15, 15, 6, 15, 7, 14, 10, 12, 15, 4, 9, 9, 1, 11, 3, 0, 12, 14, 5, 5, 15, 10, 7, 6, 0, 14, 1],
            [13, 6, 13, 5, 1, 9, 13, 13, 8, 7, 2, 0, 15, 13, 2, 15, 10, 0, 13, 6, 14, 9, 8, 8, 12, 11, 14, 13, 11, 3, 15, 5, 13, 12, 15, 8, 6, 6, 7, 0, 8, 6, 8, 2, 3, 4, 5, 2, 11, 6, 10, 13, 6, 7, 10, 7, 12, 4, 1, 0, 11, 12, 3, 3],
            [3, 4, 5, 5, 10, 11, 2, 5, 7, 5, 4, 4, 0, 14, 4, 14, 3, 1, 6, 9, 8, 0, 1, 2, 0, 7, 1, 4, 12, 3, 10, 8, 13, 3, 13, 15, 0, 8, 10, 7, 11, 11, 4, 3, 13, 1, 10, 13, 6, 12, 4, 6, 9, 14, 0, 2, 9, 0, 4, 13, 1, 3, 6, 7],
            [3, 6, 8, 12, 3, 6, 10, 11, 12, 2, 7, 9, 6, 9, 12, 13, 2, 6, 7, 3, 2, 11, 6, 12, 1, 8, 6, 3, 13, 6, 5, 14, 0, 3, 4, 12, 3, 13, 2, 8, 10, 6, 3, 5, 4, 6, 6, 1, 6, 4, 5, 2, 0, 4, 1, 10, 15, 9, 3, 10, 14, 8, 13, 8],
            [0, 14, 10, 9, 0, 12, 2, 13, 11, 13, 12, 4, 14, 0, 5, 0, 1, 13, 3, 2, 8, 14, 14, 5, 7, 9, 15, 14, 4, 3, 3, 10, 2, 2, 12, 2, 4, 2, 11, 5, 5, 5, 5, 10, 7, 8, 11, 12, 15, 6, 12, 14, 11, 14, 11, 13, 8, 2, 1, 1, 2, 9, 11, 4],
            [4, 6, 1, 13, 8, 10, 5, 9, 7, 7, 5, 1, 10, 8, 0, 13, 13, 1, 15, 7, 14, 1, 15, 10, 6, 11, 8, 14, 15, 5, 13, 2, 4, 5, 3, 7, 10, 7, 9, 14, 0, 15, 15, 7, 5, 13, 1, 8, 8, 3, 14, 11, 1, 1, 0, 8, 7, 2, 5, 1, 11, 5, 15, 15],
            [11, 12, 6, 7, 10, 2, 11, 14, 10, 3, 2, 7, 14, 13, 10, 12, 1, 4, 3, 2, 10, 13, 11, 5, 14, 7, 3, 6, 12, 6, 4, 14, 6, 11, 12, 5, 1, 14, 8, 9, 13, 9, 15, 8, 11, 2, 0, 5, 13, 11, 12, 11, 1, 2, 2, 7, 8, 9, 12, 7, 15, 0, 0, 10],
            [2, 14, 5, 14, 8, 3, 10, 0, 13, 9, 13, 8, 15, 6, 0, 10, 11, 0, 9, 4, 8, 8, 4, 12, 1, 2, 7, 14, 10, 4, 5, 12, 5, 8, 5, 5, 8, 14, 6, 5, 11, 6, 2, 11, 11, 14, 11, 12, 6, 7, 8, 12, 11, 7, 15, 6, 4, 1, 4, 9, 9, 4, 11, 6],
            [10, 4, 9, 3, 1, 13, 6, 5, 13, 14, 15, 7, 6, 8, 8, 9, 8, 10, 15, 10, 14, 5, 4, 9, 5, 15, 3, 1, 12, 3, 2, 9, 4, 0, 7, 4, 12, 13, 14, 2, 10, 13, 14, 11, 2, 10, 4, 11, 6, 10, 15, 11, 1, 5, 12, 1, 14, 3, 10, 10, 4, 7, 0, 5],
            [3, 6, 10, 3, 5, 14, 6, 10, 15, 5, 4, 0, 2, 14, 11, 8, 9, 7, 6, 1, 1, 9, 8, 15, 11, 7, 0, 7, 11, 0, 3, 4, 8, 7, 14, 4, 3, 1, 10, 3, 7, 10, 10, 12, 2, 11, 2, 1, 2, 7, 2, 6, 6, 6, 4, 2, 8, 6, 5, 12, 6, 0, 9, 14],
            [6, 8, 1, 8, 12, 9, 1, 2, 5, 7, 11, 7, 2, 1, 10, 10, 6, 13, 7, 7, 7, 7, 8, 10, 3, 13, 0, 1, 14, 12, 10, 6, 10, 6, 10, 9, 5, 14, 15, 15, 6, 2, 14, 9, 0, 12, 12, 7, 10, 12, 10, 15, 2, 6, 4, 6, 10, 13, 4, 2, 12, 8, 12, 8],
            [11, 9, 13, 7, 5, 3, 12, 14, 15, 2, 6, 13, 8, 4, 11, 7, 3, 7, 11, 4, 1, 3, 2, 13, 5, 6, 5, 9, 7, 5, 14, 13, 14, 2, 6, 14, 12, 2, 5, 7, 10, 12, 13, 3, 2, 7, 1, 0, 9, 2, 3, 8, 0, 12, 12, 12, 12, 8, 9, 1, 3, 3, 7, 9],
            [3, 11, 12, 11, 1, 8, 4, 6, 11, 8, 4, 6, 5, 13, 15, 11, 7, 2, 10, 9, 9, 4, 4, 5, 7, 6, 8, 1, 7, 11, 1, 0, 8, 10, 7, 9, 6, 8, 13, 4, 9, 15, 12, 6, 14, 3, 11, 13, 10, 9, 8, 5, 2, 5, 2, 7, 15, 12, 15, 0, 10, 11, 9, 7],
            [15, 2, 5, 3, 1, 8, 10, 12, 10, 13, 14, 11, 7, 0, 10, 5, 14, 7, 12, 13, 0, 3, 9, 12, 13, 4, 15, 6, 3, 3, 4, 12, 1, 3, 3, 9, 11, 13, 10, 9, 11, 15, 0, 14, 4, 9, 15, 5, 4, 7, 11, 4, 4, 3, 2, 15, 3, 5, 11, 3, 2, 2, 5, 4],
            [7, 11, 3, 12, 14, 13, 2, 8, 10, 5, 9, 2, 5, 0, 7, 0, 15, 6, 12, 0, 3, 0, 11, 5, 7, 0, 4, 7, 3, 13, 5, 10, 15, 5, 6, 8, 2, 4, 6, 11, 11, 5, 0, 7, 7, 14, 2, 10, 15, 3, 5, 9, 14, 14, 3, 13, 2, 11, 2, 1, 2, 9, 15, 8],
            [15, 11, 2, 5, 12, 2, 15, 5, 2, 6, 0, 11, 3, 9, 13, 11, 14, 12, 8, 8, 11, 13, 1, 7, 12, 14, 5, 8, 5, 15, 0, 13, 12, 13, 12, 1, 15, 8, 1, 6, 13, 14, 2, 12, 2, 9, 10, 7, 13, 6, 9, 5, 1, 11, 10, 11, 15, 2, 5, 5, 15, 2, 4, 10],
            [4, 7, 15, 8, 12, 3, 2, 12, 10, 3, 13, 14, 4, 7, 15, 9, 2, 6, 5, 6, 12, 8, 14, 9, 12, 4, 10, 14, 6, 7, 0, 2, 7, 3, 13, 8, 8, 6, 0, 1, 3, 8, 12, 1, 12, 14, 8, 13, 4, 11, 5, 10, 3, 13, 14, 14, 3, 11, 0, 11, 10, 3, 13, 3],
            [6, 1, 4, 4, 6, 2, 12, 14, 9, 0, 11, 13, 5, 1, 8, 11, 14, 13, 5, 0, 5, 0, 13, 6, 9, 8, 1, 6, 8, 12, 11, 0, 13, 0, 8, 8, 2, 1, 11, 7, 5, 13, 5, 0, 10, 10, 12, 0, 10, 5, 4, 13, 2, 13, 15, 9, 5, 5, 13, 2, 11, 5, 0, 10],
            [10, 6, 13, 5, 7, 14, 0, 14, 7, 7, 3, 4, 6, 1, 7, 15, 13, 5, 11, 15, 10, 4, 9, 0, 9, 2, 9, 0, 3, 8, 10, 5, 7, 6, 11, 13, 9, 2, 5, 5, 8, 6, 6, 8, 12, 13, 5, 12, 14, 15, 5, 8, 0, 5, 6, 8, 2, 7, 2, 15, 8, 11, 11, 1],
            [14, 9, 3, 7, 12, 8, 7, 12, 14, 1, 9, 6, 10, 7, 7, 11, 14, 0, 10, 4, 1, 15, 9, 4, 3, 5, 14, 1, 13, 12, 10, 4, 7, 15, 8, 13, 6, 3, 7, 2, 10, 14, 1, 4, 7, 2, 9, 6, 4, 3, 10, 11, 1, 11, 2, 4, 6, 7, 0, 12, 15, 6, 8, 12],
            [2, 2, 9, 3, 12, 5, 13, 1, 14, 0, 14, 12, 2, 12, 13, 5, 6, 6, 2, 6, 8, 5, 9, 14, 14, 12, 1, 13, 4, 12, 15, 4, 10, 2, 2, 9, 10, 4, 0, 0, 4, 15, 5, 8, 5, 5, 4, 5, 6, 7, 12, 1, 6, 0, 14, 11, 1, 11, 9, 6, 9, 0, 3, 1],
            [1, 7, 3, 15, 2, 11, 13, 10, 15, 8, 12, 5, 14, 2, 14, 2, 5, 1, 7, 9, 4, 13, 4, 13, 9, 8, 9, 6, 10, 11, 1, 7, 12, 13, 10, 7, 6, 11, 7, 10, 13, 7, 1, 10, 15, 10, 14, 8, 2, 9, 1, 7, 8, 4, 9, 15, 12, 3, 6, 12, 12, 8, 4, 11],
            [11, 4, 15, 8, 0, 4, 6, 3, 5, 3, 11, 13, 0, 4, 2, 0, 3, 14, 1, 2, 5, 15, 8, 8, 3, 1, 7, 11, 0, 1, 5, 12, 7, 6, 0, 3, 7, 10, 15, 14, 2, 4, 1, 8, 11, 0, 5, 9, 1, 3, 14, 11, 7, 3, 6, 13, 13, 12, 10, 15, 12, 9, 0, 3],
            [2, 15, 7, 8, 11, 8, 12, 12, 15, 14, 8, 15, 15, 4, 6, 9, 9, 5, 5, 2, 15, 5, 15, 14, 4, 11, 8, 0, 5, 9, 15, 3, 14, 1, 12, 13, 0, 9, 0, 2, 3, 3, 15, 15, 3, 0, 13, 10, 6, 6, 7, 12, 8, 0, 12, 15, 15, 15, 9, 5, 11, 7, 4, 7],
            [14, 15, 10, 13, 13, 12, 12, 0, 11, 5, 9, 5, 4, 5, 14, 5, 9, 12, 2, 5, 2, 4, 9, 6, 8, 15, 10, 11, 14, 15, 13, 4, 5, 1, 13, 9, 4, 8, 4, 4, 7, 3, 14, 3, 6, 7, 6, 11, 13, 0, 6, 6, 8, 5, 3, 6, 5, 6, 10, 13, 0, 12, 4, 2],
            [13, 10, 6, 9, 12, 10, 0, 7, 0, 14, 5, 14, 5, 7, 14, 4, 11, 14, 6, 7, 8, 6, 12, 12, 5, 11, 6, 2, 7, 11, 5, 2, 2, 2, 11, 11, 5, 13, 0, 1, 7, 8, 3, 15, 7, 8, 4, 6, 10, 1, 10, 7, 4, 4, 2, 14, 10, 14, 7, 11, 3, 10, 6, 6],
            [12, 1, 12, 3, 5, 6, 11, 6, 14, 3, 13, 10, 9, 10, 15, 4, 4, 12, 11, 2, 9, 2, 14, 5, 7, 15, 12, 8, 9, 5, 14, 9, 2, 4, 15, 14, 15, 12, 14, 1, 13, 15, 9, 5, 7, 8, 2, 14, 4, 5, 8, 7, 3, 0, 5, 13, 6, 3, 11, 12, 15, 8, 2, 11],
            [9, 0, 3, 10, 9, 8, 12, 1, 5, 6, 15, 3, 6, 15, 10, 13, 11, 3, 1, 4, 0, 6, 4, 5, 11, 8, 6, 14, 6, 7, 14, 4, 10, 3, 5, 13, 8, 3, 11, 15, 8, 0, 2, 6, 14, 11, 14, 3, 15, 3, 5, 5, 0, 8, 10, 11, 9, 9, 5, 7, 4, 9, 7, 3],
            [10, 12, 11, 0, 12, 15, 8, 4, 8, 9, 12, 14, 13, 15, 15, 8, 11, 5, 0, 1, 0, 13, 6, 2, 13, 3, 9, 11, 10, 6, 13, 15, 12, 5, 11, 7, 3, 11, 11, 2, 4, 4, 13, 11, 6, 4, 4, 1, 7, 0, 10, 9, 15, 3, 11, 12, 10, 0, 7, 6, 0, 6, 8, 14],
            [2, 8, 6, 6, 4, 2, 9, 11, 8, 1, 0, 5, 8, 4, 12, 15, 5, 8, 8, 6, 10, 9, 12, 14, 8, 9, 5, 12, 13, 7, 3, 9, 7, 10, 10, 6, 14, 12, 6, 6, 0, 14, 1, 7, 4, 11, 5, 7, 3, 0, 6, 8, 7, 13, 6, 9, 5, 14, 0, 6, 9, 9, 4, 6],
            [11, 15, 14, 6, 14, 5, 10, 10, 7, 0, 14, 11, 15, 1, 1, 2, 1, 9, 13, 9, 10, 12, 2, 11, 7, 10, 2, 5, 1, 0, 5, 15, 15, 11, 2, 12, 4, 12, 10, 11, 9, 10, 15, 8, 14, 10, 1, 15, 12, 11, 4, 2, 5, 9, 7, 4, 3, 8, 5, 10, 7, 5, 5, 3],
            [12, 6, 6, 0, 1, 7, 7, 12, 7, 1, 3, 4, 5, 6, 3, 3, 5, 3, 9, 14, 8, 12, 15, 3, 6, 10, 8, 15, 12, 1, 9, 2, 5, 0, 2, 8, 9, 0, 9, 7, 5, 11, 8, 2, 0, 11, 0, 15, 13, 9, 13, 12, 7, 15, 8, 6, 1, 11, 11, 2, 11, 8, 2, 0],
            [14, 6, 8, 0, 5, 4, 2, 6, 0, 15, 4, 1, 12, 12, 7, 10, 8, 12, 12, 14, 8, 10, 3, 5, 13, 14, 15, 10, 11, 0, 15, 9, 11, 8, 15, 6, 7, 6, 6, 13, 5, 2, 7, 5, 1, 1, 9, 1, 6, 0, 12, 8, 13, 14, 5, 3, 0, 14, 8, 2, 12, 5, 0, 3],
            [4, 10, 2, 7, 11, 12, 0, 5, 10, 11, 11, 4, 2, 4, 7, 11, 5, 7, 1, 7, 13, 14, 13, 7, 15, 12, 9, 12, 13, 8, 4, 4, 13, 0, 12, 14, 9, 15, 15, 7, 3, 3, 14, 4, 7, 12, 3, 4, 3, 6, 4, 10, 10, 14, 10, 8, 2, 12, 7, 11, 12, 6, 4, 13],
            [7, 10, 9, 3, 12, 15, 3, 10, 11, 15, 2, 7, 7, 3, 15, 9, 6, 14, 11, 12, 11, 6, 10, 11, 13, 15, 7, 13, 6, 11, 4, 0, 0, 15, 8, 7, 8, 7, 4, 13, 3, 13, 3, 15, 12, 8, 15, 3, 12, 9, 8, 10, 0, 11, 5, 11, 12, 7, 11, 4, 5, 7, 5, 14],
            [2, 7, 12, 12, 9, 9, 6, 6, 11, 2, 1, 11, 8, 14, 6, 4, 9, 6, 12, 4, 11, 9, 1, 5, 5, 1, 2, 1, 12, 14, 9, 12, 10, 1, 12, 5, 2, 3, 3, 0, 14, 0, 15, 8, 14, 14, 13, 5, 0, 12, 8, 8, 2, 12, 14, 11, 13, 10, 9, 8, 14, 12, 0, 6],
            [10, 6, 9, 10, 9, 2, 15, 2, 13, 13, 0, 3, 5, 6, 0, 12, 7, 15, 4, 2, 2, 11, 7, 13, 8, 13, 6, 11, 0, 1, 8, 8, 1, 0, 15, 0, 15, 8, 14, 0, 1, 4, 15, 0, 12, 10, 7, 1, 6, 3, 13, 15, 13, 1, 0, 10, 11, 10, 3, 8, 10, 5, 0, 14],
            [3, 0, 11, 12, 8, 12, 6, 9, 9, 3, 7, 6, 3, 9, 7, 3, 8, 6, 6, 6, 11, 5, 13, 1, 14, 15, 11, 15, 11, 3, 12, 6, 7, 9, 14, 2, 5, 5, 5, 3, 7, 13, 2, 8, 8, 8, 15, 9, 4, 4, 8, 14, 6, 4, 4, 0, 12, 9, 13, 15, 12, 1, 6, 2],
            [13, 4, 6, 5, 12, 11, 13, 10, 4, 11, 15, 12, 5, 12, 14, 3, 4, 4, 6, 1, 6, 12, 9, 10, 12, 6, 9, 7, 0, 11, 7, 0, 0, 4, 1, 15, 6, 7, 9, 10, 3, 8, 2, 1, 8, 3, 1, 4, 1, 13, 13, 2, 14, 12, 1, 13, 12, 5, 1, 0, 8, 10, 8, 2],
            [6, 8, 8, 10, 2, 3, 11, 9, 7, 1, 5, 12, 3, 4, 11, 7, 4, 6, 8, 15, 1, 7, 10, 3, 9, 5, 3, 0, 9, 14, 4, 11, 7, 2, 2, 15, 6, 0, 1, 11, 14, 3, 1, 14, 10, 1, 4, 14, 9, 1, 4, 12, 5, 12, 8, 13, 3, 9, 15, 5, 11, 14, 10, 4],
            [2, 14, 13, 13, 15, 3, 12, 12, 5, 14, 9, 11, 12, 8, 12, 13, 15, 4, 14, 6, 15, 7, 9, 4, 4, 14, 4, 3, 2, 11, 4, 0, 11, 15, 5, 5, 15, 1, 10, 12, 0, 8, 15, 8, 13, 8, 9, 2, 5, 2, 7, 15, 3, 4, 0, 8, 7, 5, 1, 10, 7, 15, 12, 8],
            [2, 9, 7, 10, 4, 13, 2, 4, 2, 12, 2, 6, 4, 0, 7, 8, 6, 1, 12, 1, 4, 4, 15, 2, 5, 0, 11, 12, 13, 14, 4, 9, 13, 4, 0, 2, 6, 6, 3, 14, 7, 7, 7, 8, 13, 15, 9, 7, 15, 3, 15, 13, 15, 14, 2, 7, 14, 14, 8, 11, 10, 14, 12, 13],
            [5, 1, 7, 7, 10, 14, 15, 1, 5, 11, 11, 6, 1, 5, 4, 8, 2, 0, 12, 8, 15, 15, 15, 11, 1, 5, 15, 14, 4, 15, 9, 6, 9, 8, 3, 6, 4, 5, 7, 9, 2, 9, 5, 15, 9, 6, 7, 12, 11, 11, 7, 14, 3, 2, 13, 6, 7, 13, 8, 5, 4, 10, 15, 6],
            [10, 3, 5, 3, 8, 0, 15, 4, 7, 0, 3, 5, 15, 6, 4, 5, 1, 1, 10, 0, 15, 5, 11, 2, 3, 1, 14, 7, 15, 14, 1, 4, 8, 2, 5, 6, 11, 7, 9, 13, 2, 2, 14, 4, 9, 4, 12, 4, 14, 13, 13, 2, 0, 5, 1, 13, 11, 15, 0, 9, 10, 8, 5, 3],
            [4, 6, 13, 1, 10, 0, 15, 6, 11, 11, 3, 12, 14, 10, 3, 13, 12, 1, 6, 8, 5, 5, 11, 12, 10, 3, 13, 15, 12, 15, 8, 10, 15, 2, 4, 8, 9, 10, 14, 7, 3, 13, 8, 9, 8, 14, 7, 4, 15, 10, 0, 6, 4, 14, 11, 8, 3, 6, 10, 13, 0, 11, 13, 7],
            [13, 12, 3, 0, 15, 8, 15, 0, 11, 3, 13, 15, 3, 2, 2, 2, 9, 3, 11, 4, 4, 10, 10, 4, 7, 5, 13, 10, 5, 4, 0, 1, 12, 14, 9, 7, 3, 0, 8, 10, 6, 4, 0, 9, 5, 9, 11, 3, 14, 3, 13, 1, 6, 13, 0, 9, 9, 3, 2, 6, 2, 8, 2, 10],
            [0, 10, 0, 5, 7, 10, 10, 14, 7, 15, 0, 0, 10, 0, 15, 0, 7, 1, 8, 11, 2, 5, 2, 13, 4, 14, 0, 12, 15, 13, 12, 14, 15, 8, 5, 12, 9, 12, 0, 11, 6, 7, 10, 13, 0, 0, 0, 15, 12, 13, 14, 15, 15, 2, 7, 5, 6, 12, 1, 7, 15, 10, 0, 1],
            [0, 12, 11, 15, 14, 2, 11, 15, 8, 14, 15, 8, 3, 0, 7, 12, 0, 8, 0, 15, 10, 1, 1, 1, 9, 10, 12, 13, 3, 13, 3, 0, 10, 12, 10, 15, 11, 2, 11, 1, 10, 3, 9, 4, 14, 3, 8, 1, 10, 5, 1, 0, 1, 13, 13, 14, 6, 7, 4, 3, 10, 11, 10, 6],
            [15, 15, 2, 10, 3, 10, 0, 5, 8, 7, 3, 15, 9, 1, 12, 2, 8, 8, 7, 5, 1, 9, 11, 8, 11, 11, 10, 11, 6, 12, 3, 1, 12, 10, 7, 15, 3, 12, 2, 13, 8, 5, 14, 15, 8, 11, 13, 12, 11, 3, 15, 5, 2, 6, 14, 12, 6, 5, 3, 6, 4, 9, 15, 1],
            [3, 12, 5, 6, 15, 6, 9, 10, 4, 13, 14, 5, 12, 10, 5, 13, 14, 15, 4, 10, 9, 4, 10, 12, 0, 11, 9, 14, 15, 5, 11, 9, 4, 6, 13, 2, 0, 12, 9, 11, 7, 12, 0, 9, 14, 15, 1, 14, 10, 5, 4, 11, 6, 4, 2, 9, 7, 0, 8, 1, 5, 3, 2, 0],
            [5, 12, 8, 10, 14, 1, 7, 15, 4, 0, 15, 9, 11, 2, 10, 10, 8, 3, 8, 7, 3, 10, 10, 3, 9, 12, 15, 13, 3, 9, 12, 10, 6, 12, 8, 3, 12, 8, 10, 14, 1, 8, 7, 7, 8, 5, 8, 4, 13, 12, 13, 7, 6, 15, 6, 1, 0, 2, 1, 7, 8, 0, 8, 15],
            [15, 12, 9, 0, 1, 10, 8, 12, 4, 3, 12, 11, 2, 3, 13, 11, 15, 1, 3, 0, 7, 4, 10, 0, 3, 11, 14, 7, 7, 6, 1, 8, 1, 5, 6, 7, 8, 9, 6, 6, 11, 9, 13, 15, 3, 9, 5, 9, 10, 10, 13, 4, 3, 12, 13, 10, 1, 14, 0, 2, 7, 9, 5, 4],
            [2, 9, 14, 14, 4, 4, 9, 13, 0, 5, 3, 3, 13, 14, 15, 12, 6, 13, 5, 14, 11, 9, 8, 3, 12, 9, 13, 1, 12, 15, 1, 6, 2, 6, 9, 15, 8, 14, 1, 2, 8, 5, 3, 2, 10, 12, 3, 4, 3, 2, 7, 0, 1, 3, 15, 4, 0, 0, 14, 5, 4, 1, 8, 14],
            [5, 15, 0, 5, 13, 14, 10, 14, 15, 1, 3, 2, 1, 13, 6, 2, 13, 8, 4, 12, 8, 3, 5, 5, 3, 11, 3, 7, 9, 15, 4, 13, 11, 15, 3, 0, 8, 0, 5, 15, 10, 11, 6, 0, 5, 7, 10, 3, 8, 4, 0, 10, 6, 1, 7, 4, 0, 4, 4, 10, 13, 1, 15, 13],
            [4, 6, 4, 4, 8, 8, 2, 13, 8, 2, 4, 0, 0, 10, 6, 6, 14, 1, 8, 5, 12, 10, 9, 8, 15, 4, 5, 7, 10, 12, 3, 14, 0, 14, 15, 15, 8, 14, 5, 2, 14, 0, 2, 2, 15, 5, 13, 12, 9, 10, 3, 10, 2, 3, 8, 8, 15, 2, 1, 7, 4, 6, 6, 3],
            [8, 0, 12, 10, 0, 9, 12, 14, 4, 13, 8, 15, 2, 13, 4, 15, 4, 9, 15, 6, 11, 9, 0, 7, 4, 15, 2, 6, 0, 14, 8, 8, 15, 5, 1, 8, 15, 15, 9, 2, 2, 10, 14, 2, 15, 13, 13, 9, 2, 12, 15, 3, 14, 11, 1, 12, 7, 1, 7, 6, 9, 13, 3, 4],
            [15, 8, 8, 15, 10, 0, 2, 15, 10, 8, 13, 9, 0, 13, 0, 6, 2, 7, 11, 1, 13, 15, 7, 9, 5, 7, 7, 12, 0, 10, 14, 11, 3, 5, 13, 15, 11, 0, 12, 4, 7, 1, 10, 8, 10, 11, 0, 5, 3, 0, 7, 11, 8, 14, 9, 4, 2, 3, 0, 7, 14, 10, 5, 12]
        ]);
        let hash = Hash::from_le_bytes([42; 32]);
        let matrix = Matrix::generate(hash, 1);
        assert_eq!(matrix, expected_matrix);
    }
}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use super::{Matrix, XoShiRo256PlusPlus};
    use crate::Hash;
    use rand::{thread_rng, Rng};

    #[bench]
    pub fn bench_compute_rank(bh: &mut Bencher) {
        let mut generator = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        let mut matrix = Matrix::rand_matrix_no_rank_check(&mut generator);
        bh.iter(|| {
            for _ in 0..10 {
                black_box(&mut matrix);
                black_box(matrix.compute_rank());
            }
        });
    }

    #[bench]
    pub fn bench_heavy_hash(bh: &mut Bencher) {
        let mut generator = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        let mut input = Hash::new(thread_rng().gen());
        let mut matrix = Matrix::rand_matrix_no_rank_check(&mut generator);
        bh.iter(|| {
            for _ in 0..10 {
                black_box(&mut matrix);
                black_box(&mut input);
                black_box(matrix.heavy_hash(input));
            }
        });
    }
}
