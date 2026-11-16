//! Merkle proofs over model chunks and trace leaves.

use blake3::hazmat::{HasherExt, Mode, merge_subtrees_non_root, merge_subtrees_root};

pub type Hash = [u8; 32];

pub fn hash_pair(l: &Hash, r: &Hash) -> Hash {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(l);
    buf[32..].copy_from_slice(r);
    *blake3::hash(&buf).as_bytes()
}

/// Inclusion of the contiguous leaves starting at `first` in a tree of `n_leaves` leaves.
pub fn rt_verify_range(root: &Hash, n_leaves: u64, first: u64, leaves: &[Hash], proof: &[Hash]) -> bool {
    if leaves.is_empty() || first.checked_add(leaves.len() as u64).is_none_or(|e| e > n_leaves) || proof.len() > 128 {
        return false;
    }
    let mut level = leaves.to_vec();
    let (mut lo, mut n) = (first, n_leaves);
    let mut p = proof.iter();
    while n > 1 {
        let hi = lo + level.len() as u64 - 1;
        let mut cur = Vec::with_capacity(level.len() + 2);
        let start = if !lo.is_multiple_of(2) {
            match p.next() {
                Some(s) => cur.push(*s),
                None => return false,
            }
            lo - 1
        } else {
            lo
        };
        cur.extend_from_slice(&level);
        if hi.is_multiple_of(2) {
            if hi + 1 < n {
                match p.next() {
                    Some(s) => cur.push(*s),
                    None => return false,
                }
            } else {
                cur.push(*cur.last().unwrap());
            }
        }
        level = cur.chunks(2).map(|c| hash_pair(&c[0], &c[1])).collect();
        lo = start / 2;
        n = n.div_ceil(2);
    }
    p.next().is_none() && level.len() == 1 && &level[0] == root
}

/// Root and range proof of `count` leaves starting at `first`.
pub fn rt_range_proof(leaves: &[Hash], first: u64, count: u64) -> (Hash, Vec<Hash>) {
    let mut level = leaves.to_vec();
    let (mut lo, mut hi) = (first, first + count - 1);
    let mut proof = Vec::new();
    while level.len() > 1 {
        let n = level.len() as u64;
        if !lo.is_multiple_of(2) {
            proof.push(level[(lo - 1) as usize]);
        }
        if hi.is_multiple_of(2) && hi + 1 < n {
            proof.push(level[(hi + 1) as usize]);
        }
        level = level.chunks(2).map(|c| if c.len() == 2 { hash_pair(&c[0], &c[1]) } else { hash_pair(&c[0], &c[0]) }).collect();
        lo /= 2;
        hi /= 2;
    }
    (level[0], proof)
}

/// Inclusion of one leaf.
pub fn rt_verify_leaf(root: &Hash, n_leaves: u64, index: u64, leaf: &Hash, path: &[Hash]) -> bool {
    rt_verify_range(root, n_leaves, index, std::slice::from_ref(leaf), path)
}

fn b3_merge(l: &Hash, r: &Hash, root: bool) -> Hash {
    if root { *merge_subtrees_root(l, r, Mode::Hash).as_bytes() } else { merge_subtrees_non_root(l, r, Mode::Hash) }
}

/// Root of the trace tree over `leaves`.
pub fn b3_root(leaves: &[Hash]) -> Hash {
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let n = level.len();
        let mut next: Vec<Hash> = level.chunks_exact(2).map(|c| b3_merge(&c[0], &c[1], n == 2)).collect();
        if !n.is_multiple_of(2) {
            next.push(level[n - 1]);
        }
        level = next;
    }
    level[0]
}

/// Sibling path of leaf `index` in the trace tree.
pub fn b3_path(leaves: &[Hash], index: usize) -> Vec<Hash> {
    let mut level = leaves.to_vec();
    let mut i = index;
    let mut path = Vec::new();
    while level.len() > 1 {
        let n = level.len();
        if n.is_multiple_of(2) || i != n - 1 {
            path.push(level[i ^ 1]);
        }
        let mut next: Vec<Hash> = level.chunks_exact(2).map(|c| b3_merge(&c[0], &c[1], n == 2)).collect();
        if !n.is_multiple_of(2) {
            next.push(level[n - 1]);
        }
        level = next;
        i /= 2;
    }
    path
}

/// Trace tree root reached from `leaf` at `index`.
pub fn b3_fold(n_leaves: u64, index: u64, leaf: &Hash, path: &[Hash]) -> Option<Hash> {
    if index >= n_leaves || path.len() > 64 {
        return None;
    }
    let (mut acc, mut i, mut n) = (*leaf, index, n_leaves);
    let mut p = path.iter();
    while n > 1 {
        if n.is_multiple_of(2) || i != n - 1 {
            let s = p.next()?;
            acc = if i.is_multiple_of(2) { b3_merge(&acc, s, n == 2) } else { b3_merge(s, &acc, n == 2) };
        }
        i /= 2;
        n = n.div_ceil(2);
    }
    p.next().is_none().then_some(acc)
}

pub fn b3_verify(root: &Hash, n_leaves: u64, index: u64, leaf: &Hash, path: &[Hash]) -> bool {
    b3_fold(n_leaves, index, leaf, path).as_ref() == Some(root)
}

/// BLAKE3 hash of an input rebuilt from one of its chunks.
pub fn b3_leaf_from_chunk(total_len: u64, c: u64, chunk: &[u8], path: &[Hash]) -> Option<Hash> {
    let n_chunks = total_len.div_ceil(B3_CHUNK as u64).max(1);
    let expected = if c + 1 < n_chunks { B3_CHUNK as u64 } else { total_len.checked_sub(c * B3_CHUNK as u64)? };
    if c >= n_chunks || chunk.len() as u64 != expected {
        return None;
    }
    if n_chunks == 1 {
        return path.is_empty().then(|| *blake3::hash(chunk).as_bytes());
    }
    b3_fold(n_chunks, c, &b3_chunk_cv(chunk, c), path)
}

pub const B3_CHUNK: usize = 1024;

/// Range proof of the `count` leaves from `first` in the trace tree over `leaves`.
pub fn b3_range_proof(leaves: &[Hash], first: u64, count: u64) -> Vec<Hash> {
    let mut level = leaves.to_vec();
    let (mut lo, mut hi) = (first, first + count - 1);
    let mut proof = Vec::new();
    while level.len() > 1 {
        let n = level.len();
        if !lo.is_multiple_of(2) {
            proof.push(level[lo as usize - 1]);
        }
        if hi.is_multiple_of(2) && (hi as usize) + 1 < n {
            proof.push(level[hi as usize + 1]);
        }
        let mut next: Vec<Hash> = level.chunks_exact(2).map(|c| b3_merge(&c[0], &c[1], n == 2)).collect();
        if !n.is_multiple_of(2) {
            next.push(level[n - 1]);
        }
        level = next;
        lo /= 2;
        hi /= 2;
    }
    proof
}

/// Trace tree root reached from the contiguous `leaves` at `first` in a tree of `n_leaves` leaves.
pub fn b3_range_fold(n_leaves: u64, first: u64, leaves: &[Hash], proof: &[Hash]) -> Option<Hash> {
    if leaves.is_empty() || first.checked_add(leaves.len() as u64)? > n_leaves || proof.len() > 128 {
        return None;
    }
    let mut level = leaves.to_vec();
    let (mut lo, mut n) = (first, n_leaves);
    let mut p = proof.iter();
    while n > 1 {
        let hi = lo + level.len() as u64 - 1;
        let mut cur = Vec::with_capacity(level.len() + 2);
        let start = if lo.is_multiple_of(2) {
            lo
        } else {
            cur.push(*p.next()?);
            lo - 1
        };
        cur.extend_from_slice(&level);
        if hi.is_multiple_of(2) && hi + 1 < n {
            cur.push(*p.next()?);
        }
        level = cur.chunks(2).map(|c| if c.len() == 2 { b3_merge(&c[0], &c[1], n == 2) } else { c[0] }).collect();
        lo = start / 2;
        n = n.div_ceil(2);
    }
    (p.next().is_none() && level.len() == 1).then(|| level[0])
}

/// BLAKE3 hash of an input of `total_len` bytes rebuilt from its chunks starting at `first`,
/// given end to end in `bytes`.
pub fn b3_leaf_from_chunks(total_len: u64, first: u64, bytes: &[u8], proof: &[Hash]) -> Option<Hash> {
    let n = total_len.div_ceil(B3_CHUNK as u64).max(1);
    let count = (bytes.len() as u64).div_ceil(B3_CHUNK as u64);
    let end = first.checked_add(count)?;
    if count == 0 || end > n || bytes.len() as u64 != total_len.min(end * B3_CHUNK as u64) - first * B3_CHUNK as u64 {
        return None;
    }
    if n == 1 {
        return proof.is_empty().then(|| *blake3::hash(bytes).as_bytes());
    }
    let cvs: Vec<Hash> = bytes.chunks(B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, first + i as u64)).collect();
    b3_range_fold(n, first, &cvs, proof)
}

/// Range proof of the chunks `first..first + count` of `data`.
pub fn b3_chunks_proof(data: &[u8], first: u64, count: u64) -> Vec<Hash> {
    let cvs: Vec<Hash> = data.chunks(B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect();
    if cvs.len() <= 1 { Vec::new() } else { b3_range_proof(&cvs, first, count) }
}

/// Chaining value of chunk `c` of a multi-chunk input.
pub fn b3_chunk_cv(chunk: &[u8], c: u64) -> Hash {
    let mut h = blake3::Hasher::new();
    h.set_input_offset(c * B3_CHUNK as u64);
    h.update(chunk);
    h.finalize_non_root()
}

/// Inclusion of chunk `c` of an input whose BLAKE3 hash is `leaf`.
pub fn b3_verify_chunk(leaf: &Hash, total_len: u64, c: u64, chunk: &[u8], path: &[Hash]) -> bool {
    b3_leaf_from_chunk(total_len, c, chunk, path).as_ref() == Some(leaf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Hash> {
        (0..n as u32).map(|i| *blake3::hash(&i.to_le_bytes()).as_bytes()).collect()
    }

    #[test]
    fn rt_ranges() {
        for n in [1usize, 2, 3, 5, 8, 13, 33, 100] {
            let l = leaves(n);
            for first in 0..n {
                for count in 1..=(n - first).min(7) {
                    let (root, proof) = rt_range_proof(&l, first as u64, count as u64);
                    assert!(
                        rt_verify_range(&root, n as u64, first as u64, &l[first..first + count], &proof),
                        "n {n} first {first} count {count}"
                    );
                    let mut bad = l[first..first + count].to_vec();
                    bad[0][0] ^= 1;
                    assert!(!rt_verify_range(&root, n as u64, first as u64, &bad, &proof));
                }
            }
        }
    }

    #[test]
    fn b3_tree_matches_blake3_internal_tree() {
        for len in [1025usize, 2048, 3000, 5 * 1024, 7 * 1024 + 5, 19584, 65536] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let n = len.div_ceil(B3_CHUNK);
            let cvs: Vec<Hash> = (0..n).map(|c| b3_chunk_cv(&data[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(len)], c as u64)).collect();
            let leaf = *blake3::hash(&data).as_bytes();
            assert_eq!(b3_root(&cvs), leaf, "len {len}");
            for c in 0..n {
                let path = b3_path(&cvs, c);
                assert!(b3_verify_chunk(&leaf, len as u64, c as u64, &data[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(len)], &path));
            }
        }
    }

    #[test]
    fn b3_ranges() {
        for n in [1usize, 2, 3, 4, 5, 7, 11, 33, 38, 64, 100] {
            let l = leaves(n);
            let root = b3_root(&l);
            for first in 0..n {
                for count in 1..=n - first {
                    let proof = b3_range_proof(&l, first as u64, count as u64);
                    let span = &l[first..first + count];
                    assert_eq!(b3_range_fold(n as u64, first as u64, span, &proof), Some(root), "n {n} first {first} count {count}");
                    let mut bad = span.to_vec();
                    bad[count - 1][0] ^= 1;
                    assert_ne!(b3_range_fold(n as u64, first as u64, &bad, &proof), Some(root));
                    if first > 0 {
                        assert_ne!(b3_range_fold(n as u64, first as u64 - 1, span, &proof), Some(root));
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_ranges_rebuild_the_blake3_hash() {
        for len in [700usize, 1024, 1025, 3000, 7 * 1024 + 5, 19584] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13 + 1) as u8).collect();
            let leaf = *blake3::hash(&data).as_bytes();
            let n = len.div_ceil(B3_CHUNK);
            for first in 0..n {
                for count in 1..=n - first {
                    let bytes = &data[first * B3_CHUNK..((first + count) * B3_CHUNK).min(len)];
                    let proof = b3_chunks_proof(&data, first as u64, count as u64);
                    assert_eq!(b3_leaf_from_chunks(len as u64, first as u64, bytes, &proof), Some(leaf), "len {len} first {first} count {count}");
                    if first + count < n {
                        assert_ne!(b3_leaf_from_chunks(len as u64, first as u64 + 1, &data[(first + 1) * B3_CHUNK..((first + 1 + count) * B3_CHUNK).min(len)], &proof), Some(leaf));
                    }
                    if count < n && first + count == n && first > 0 {
                        assert_ne!(b3_leaf_from_chunks(len as u64, first as u64 - 1, bytes, &proof), Some(leaf));
                    }
                    if bytes.len() > 1 {
                        assert!(b3_leaf_from_chunks(len as u64, first as u64, &bytes[1..], &proof).is_none_or(|h| h != leaf));
                    }
                }
            }
        }
    }

    #[test]
    fn b3_paths() {
        for n in [1usize, 2, 3, 4, 5, 7, 11, 249, 498] {
            let l = leaves(n);
            let root = b3_root(&l);
            for i in 0..n {
                let path = b3_path(&l, i);
                assert!(b3_verify(&root, n as u64, i as u64, &l[i], &path));
                assert!(n == 1 || !b3_verify(&root, n as u64, i as u64, &l[(i + 1) % n], &path));
            }
        }
    }
}
