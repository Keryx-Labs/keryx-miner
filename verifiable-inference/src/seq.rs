//! Sequence leaves: one leaf per layer and head holding every position end to end, opened by
//! byte spans.

use crate::f16;
use crate::manifest::{RECUR_GDA, RECUR_KDA};
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_leaf_from_chunks, b3_root};

/// Trace root over the position tree root and the sequence tree root.
pub fn trace_root(pos_root: &Hash, seq_root: &Hash) -> Hash {
    b3_root(&[*pos_root, *seq_root])
}

/// Leaves of the sequence tree: key and value cache of each attention head, then records and
/// checkpoints of each recurrent head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqLayout {
    pub n_attn: u64,
    pub n_recur: u64,
    pub heads: u64,
}

impl SeqLayout {
    pub fn n_leaves(&self) -> u64 {
        2 * self.heads * (self.n_attn + self.n_recur) + self.heads * self.n_attn
    }

    /// Key cache leaf of head `h` of attention layer `slot`.
    pub fn kcache(&self, slot: u64, h: u64) -> Option<u64> {
        (slot < self.n_attn && h < self.heads).then(|| 2 * (slot * self.heads + h))
    }

    /// Value cache leaf of head `h` of attention layer `slot`.
    pub fn vcache(&self, slot: u64, h: u64) -> Option<u64> {
        self.kcache(slot, h).map(|i| i + 1)
    }

    /// Record leaf of head `h` of recurrent layer `slot`.
    pub fn record(&self, slot: u64, h: u64) -> Option<u64> {
        (slot < self.n_recur && h < self.heads).then(|| 2 * self.heads * self.n_attn + 2 * (slot * self.heads + h))
    }

    /// Checkpoint leaf of head `h` of recurrent layer `slot`.
    pub fn checkpoint(&self, slot: u64, h: u64) -> Option<u64> {
        self.record(slot, h).map(|i| i + 1)
    }

    /// Softmax mass leaf of key/value head `h` of attention layer `slot`: per position, the log
    /// mass of each query head of the group, f32.
    pub fn lse(&self, slot: u64, h: u64) -> Option<u64> {
        (slot < self.n_attn && h < self.heads).then(|| 2 * self.heads * (self.n_attn + self.n_recur) + slot * self.heads + h)
    }
}

/// A byte span of one sequence leaf: the whole chunks covering it, their range proof, and the
/// leaf's path in the sequence tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeqSpan {
    pub first_chunk: u32,
    pub bytes: Vec<u8>,
    pub chunk_proof: Vec<Hash>,
    pub leaf_path: Vec<Hash>,
}

/// Bytes `[lo, hi)` of sequence leaf `leaf` of `total_len` bytes, from `span`.
pub fn open_span<'a>(seq_root: &Hash, layout: &SeqLayout, leaf: u64, total_len: u64, lo: u64, hi: u64, span: &'a SeqSpan) -> Option<&'a [u8]> {
    let c = B3_CHUNK as u64;
    if lo >= hi || hi > total_len || span.first_chunk as u64 != lo / c || span.bytes.len() as u64 != total_len.min(hi.div_ceil(c) * c) - lo / c * c {
        return None;
    }
    let h = b3_leaf_from_chunks(total_len, lo / c, &span.bytes, &span.chunk_proof)?;
    if b3_fold(layout.n_leaves(), leaf, &h, &span.leaf_path)? != *seq_root {
        return None;
    }
    let base = lo / c * c;
    span.bytes.get((lo - base) as usize..(hi - base) as usize)
}

/// Bytes of the record of one recurrent head at one position.
pub fn record_len(kind: u32, hd: usize) -> usize {
    4 * 2 * hd + 4 + if kind == RECUR_KDA { 2 * hd } else { 4 }
}

/// Inputs and output of the delta rule of one head at one position.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Output before the gated norm.
    pub o: Vec<f32>,
    pub beta: f32,
    /// Log-decay: one value (GDA) or one per key dimension (KDA).
    pub g: Vec<f32>,
}

impl Record {
    /// Record bytes: q, k, v, o as f16, beta as f32, then g as f32 (GDA) or f16 (KDA).
    pub fn encode(&self, kind: u32) -> Vec<u8> {
        let mut b = Vec::with_capacity(record_len(kind, self.q.len()));
        for x in [&self.q, &self.k, &self.v, &self.o].into_iter().flatten() {
            b.extend_from_slice(&f16::from_f32(*x).to_le_bytes());
        }
        b.extend_from_slice(&self.beta.to_le_bytes());
        if kind == RECUR_KDA {
            for x in &self.g {
                b.extend_from_slice(&f16::from_f32(*x).to_le_bytes());
            }
        } else {
            b.extend_from_slice(&self.g[0].to_le_bytes());
        }
        b
    }

    pub fn decode(kind: u32, hd: usize, b: &[u8]) -> Option<Record> {
        if !matches!(kind, RECUR_GDA | RECUR_KDA) || b.len() != record_len(kind, hd) {
            return None;
        }
        let halves = |o: usize, n: usize| (0..n).map(|i| f16::read(b, o + 2 * i)).collect::<Vec<f32>>();
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let beta = f32_at(8 * hd);
        let g = if kind == RECUR_KDA { halves(8 * hd + 4, hd) } else { vec![f32_at(8 * hd + 4)] };
        let r = Record { q: halves(0, hd), k: halves(2 * hd, hd), v: halves(4 * hd, hd), o: halves(6 * hd, hd), beta, g };
        let finite = [&r.q, &r.k, &r.v, &r.o, &r.g].into_iter().flatten().all(|x| x.is_finite()) && beta.is_finite();
        finite.then_some(r)
    }
}

/// Bytes of a checkpoint leaf: the state after every `interval` positions, the last partial
/// interval excluded.
pub fn checkpoint_len(hd: u64, n_pos: u64, interval: u64) -> u64 {
    n_pos / interval * hd * hd * 4
}

/// Byte span of state rows `r0..r0 + n` (one row per value dimension, `hd` f32 over the key
/// dimensions) of checkpoint `m >= 1`, the state after position `m * interval - 1`.
pub fn checkpoint_rows(hd: u64, m: u64, r0: u64, n: u64) -> (u64, u64) {
    let base = (m - 1) * hd * hd * 4;
    (base + r0 * hd * 4, base + (r0 + n) * hd * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::{b3_chunks_proof, b3_path};

    fn leaf_bytes(i: u64, n: usize) -> Vec<u8> {
        (0..n).map(|j| (j as u64 * 31 + i * 7) as u8).collect()
    }

    #[test]
    fn sequence_leaves_do_not_overlap() {
        let l = SeqLayout { n_attn: 3, n_recur: 5, heads: 4 };
        let mut seen = std::collections::HashSet::new();
        for s in 0..3 {
            for h in 0..4 {
                assert!(seen.insert(l.kcache(s, h).unwrap()) && seen.insert(l.vcache(s, h).unwrap()) && seen.insert(l.lse(s, h).unwrap()));
            }
        }
        for s in 0..5 {
            for h in 0..4 {
                assert!(seen.insert(l.record(s, h).unwrap()) && seen.insert(l.checkpoint(s, h).unwrap()));
            }
        }
        assert_eq!(seen.len() as u64, l.n_leaves());
        assert!(seen.iter().all(|&i| i < l.n_leaves()));
        assert!(l.kcache(3, 0).is_none() && l.lse(3, 0).is_none() && l.lse(0, 4).is_none() && l.record(0, 4).is_none() && l.record(5, 0).is_none());
    }

    #[test]
    fn spans_open_against_the_sequence_root() {
        let layout = SeqLayout { n_attn: 1, n_recur: 1, heads: 2 };
        let leaves: Vec<Vec<u8>> = (0..layout.n_leaves()).map(|i| leaf_bytes(i, 700 + 1500 * i as usize)).collect();
        let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let root = b3_root(&hashes);
        let span = |i: usize, lo: usize, hi: usize| {
            let data = &leaves[i];
            let (c0, c1) = (lo / B3_CHUNK, (hi - 1) / B3_CHUNK);
            SeqSpan {
                first_chunk: c0 as u32,
                bytes: data[c0 * B3_CHUNK..((c1 + 1) * B3_CHUNK).min(data.len())].to_vec(),
                chunk_proof: b3_chunks_proof(data, c0 as u64, (c1 - c0 + 1) as u64),
                leaf_path: b3_path(&hashes, i),
            }
        };
        for (i, leaf) in leaves.iter().enumerate() {
            let n = leaf.len();
            for (lo, hi) in [(0, 1), (0, n), (n / 3, n / 2), (n - 5, n), (1000, 1100.min(n))] {
                if lo >= hi {
                    continue;
                }
                let sp = span(i, lo, hi);
                let got = open_span(&root, &layout, i as u64, n as u64, lo as u64, hi as u64, &sp).unwrap_or_else(|| panic!("leaf {i} {lo}..{hi}"));
                assert_eq!(got, &leaf[lo..hi]);
                assert!(open_span(&root, &layout, (i as u64 + 1) % layout.n_leaves(), n as u64, lo as u64, hi as u64, &sp).is_none());
                let mut bad = sp.clone();
                bad.bytes[0] ^= 1;
                assert!(open_span(&root, &layout, i as u64, n as u64, lo as u64, hi as u64, &bad).is_none());
                if hi < n {
                    assert!(open_span(&root, &layout, i as u64, n as u64, lo as u64, hi as u64 + B3_CHUNK as u64, &sp).is_none());
                }
            }
        }
    }

    #[test]
    fn records_round_trip_through_f16() {
        let hd = 128;
        let v = |s: f32| (0..hd).map(|i| s * ((i as f32 * 0.37).sin())).collect::<Vec<f32>>();
        for kind in [RECUR_GDA, RECUR_KDA] {
            let g = if kind == RECUR_KDA { v(-0.2) } else { vec![-0.731] };
            let r = Record { q: v(0.3), k: v(0.09), v: v(1.7), o: v(0.5), beta: 0.62, g };
            let b = r.encode(kind);
            assert_eq!(b.len(), record_len(kind, hd));
            let back = Record::decode(kind, hd, &b).unwrap();
            let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-3 * x.abs().max(1e-2));
            assert!(close(&r.q, &back.q) && close(&r.v, &back.v) && close(&r.o, &back.o) && close(&r.g, &back.g));
            assert_eq!(back.beta, r.beta);
            assert!(Record::decode(kind, hd, &b[1..]).is_none());
            assert!(Record::decode(7, hd, &b).is_none());
        }
        assert_eq!(record_len(RECUR_GDA, 128), 1032);
        assert_eq!(record_len(RECUR_KDA, 128), 1284);
    }

    #[test]
    fn checkpoint_rows_stay_inside_the_leaf() {
        let (hd, n_pos, c) = (128, 200, 48);
        let len = checkpoint_len(hd, n_pos, c);
        assert_eq!(len, 4 * 128 * 128 * 4);
        let (lo, hi) = checkpoint_rows(hd, 4, 112, 16);
        assert_eq!((lo, hi), (3 * 65536 + 112 * 512, 4 * 65536));
        assert!(hi <= len);
        assert_eq!(checkpoint_rows(hd, 1, 0, 1), (0, 512));
    }
}
