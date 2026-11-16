//! Attention openings by blocks of positions: softmax partials in a tree, one level per phase.

use crate::attn::{ARITHS, ATTN_THRESHOLD, Arith, Partial, partial};
use crate::merkle::{Hash, b3_path, b3_root, b3_verify};

/// Bytes one phase of an attention opening may spend on cache rows or on partials.
pub const BLOCK_BUDGET: usize = 96 * 1024;
/// Largest difference of the log-sum-exp of two partials of the same positions.
pub const LSE_TOL: f64 = 1.0 / 256.0;
/// Largest difference of the outputs of two partials of the same positions, in Q8 steps.
pub const PARTIAL_STEPS: f64 = 0.25;
/// Difference of one output value left out of the steps of a leaf: one f16 rounding of the
/// smallest values.
const LEAF_FLOOR: f64 = 1.0 / (1u64 << 24) as f64;
/// The same for a node, whose children and parent are rounded each.
const NODE_FLOOR: f64 = 2.0 * LEAF_FLOOR;
/// Largest difference between the log-sum-exp of the opened children and the committed one, at a
/// largest score of zero.
pub const LSE_ROOT_TOL: f64 = 1.0 / 64.0;

/// [`LSE_ROOT_TOL`] widened with the largest score `m`, up to `|m| = 176`.
pub fn root_lse_tol(m: f64) -> f64 {
    LSE_ROOT_TOL + m.abs().min(176.0) / 4096.0
}

#[cfg(test)]
thread_local! {
    static TEST_BUDGET: std::cell::Cell<usize> = const { std::cell::Cell::new(BLOCK_BUDGET) };
}

/// Sets the phase budget of the calling test thread.
#[cfg(test)]
pub(crate) fn set_test_budget(b: usize) {
    TEST_BUDGET.with(|c| c.set(b));
}

fn budget() -> usize {
    #[cfg(test)]
    return TEST_BUDGET.with(|c| c.get());
    #[cfg(not(test))]
    BLOCK_BUDGET
}

/// Positions per leaf and children per node for cache rows of `row_bytes` per position and
/// partials of `partial_bytes`; `None` when a phase cannot hold two partials or one row.
pub fn block_params(row_bytes: usize, partial_bytes: usize) -> Option<(u64, u64)> {
    let budget = budget();
    let fit = |unit: usize| {
        let mut n = 1usize;
        while 2 * n * unit <= budget {
            n *= 2;
        }
        (unit > 0 && n * unit <= budget).then_some(n as u64)
    };
    let (c, b) = (fit(row_bytes)?, fit(partial_bytes)?);
    (b >= 2).then_some((c, b))
}

/// Encoded length of a partial of `d` values.
pub const fn partial_len(d: usize) -> usize {
    2 * d + 8
}

/// `p` as f16 values, then `m` and `l` as f32.
pub fn encode_partial(p: &Partial) -> Vec<u8> {
    let mut out = Vec::with_capacity(partial_len(p.o.len()));
    for &x in &p.o {
        out.extend_from_slice(&crate::f16::from_f32(x as f32).to_le_bytes());
    }
    out.extend_from_slice(&(p.m as f32).to_le_bytes());
    out.extend_from_slice(&(p.l as f32).to_le_bytes());
    out
}

/// Partials of `d` values end to end; `None` on a length mismatch, a non-finite value or a sum
/// below one (the largest score contributes one).
pub fn decode_partials(b: &[u8], d: usize) -> Option<Vec<Partial>> {
    let n = partial_len(d);
    if d == 0 || b.is_empty() || !b.len().is_multiple_of(n) {
        return None;
    }
    b.chunks_exact(n)
        .map(|c| {
            let o: Vec<f64> = (0..d).map(|i| crate::f16::read(c, 2 * i) as f64).collect();
            let f = |off: usize| f32::from_le_bytes(c[off..off + 4].try_into().unwrap()) as f64;
            let (m, l) = (f(2 * d), f(2 * d + 4));
            (o.iter().all(|x| x.is_finite()) && m.is_finite() && l.is_finite() && l >= 1.0).then_some(Partial { o, m, l })
        })
        .collect()
}

/// Leaves of `c` positions aligned on multiples of `c`, over `[lo, hi)`, under nodes of `b` children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockTree {
    pub lo: u64,
    pub hi: u64,
    pub c: u64,
    pub b: u64,
}

impl BlockTree {
    pub fn new(lo: u64, hi: u64, c: u64, b: u64) -> Option<Self> {
        (lo < hi && c >= 1 && b >= 2).then_some(Self { lo, hi, c, b })
    }

    pub fn n_leaves(&self) -> u64 {
        (self.hi - 1) / self.c - self.lo / self.c + 1
    }

    /// Levels of partials above the leaves: zero for a single leaf.
    pub fn depth(&self) -> u32 {
        let (n, mut d, mut span) = (self.n_leaves(), 0u32, 1u64);
        while span < n {
            span = span.saturating_mul(self.b);
            d += 1;
        }
        d
    }

    fn leaves_per_node(&self, level: u32) -> u64 {
        self.b.saturating_pow(self.depth() - level)
    }

    fn positions(&self, a: u64, z: u64) -> (u64, u64) {
        let first = self.lo / self.c;
        (self.lo.max((first + a) * self.c), self.hi.min((first + z) * self.c))
    }

    fn leaf_range(&self, path: &[u64]) -> Option<(u64, u64)> {
        if path.len() > self.depth() as usize {
            return None;
        }
        let (mut a, mut z) = (0, self.n_leaves());
        for (level, &i) in path.iter().enumerate() {
            let s = self.leaves_per_node(level as u32 + 1);
            if i >= (z - a).div_ceil(s) {
                return None;
            }
            a += i * s;
            z = z.min(a + s);
        }
        Some((a, z))
    }

    /// Positions of the node reached from the root by `path`, one child index per level.
    pub fn node(&self, path: &[u64]) -> Option<(u64, u64)> {
        self.leaf_range(path).map(|(a, z)| self.positions(a, z))
    }

    /// Positions of each child of the node reached by `path`; `None` for a leaf.
    pub fn children(&self, path: &[u64]) -> Option<Vec<(u64, u64)>> {
        if path.len() >= self.depth() as usize {
            return None;
        }
        let (a, z) = self.leaf_range(path)?;
        let s = self.leaves_per_node(path.len() as u32 + 1);
        Some((a..z).step_by(s as usize).map(|x| self.positions(x, z.min(x + s))).collect())
    }
}

/// Partial of the union of the runs of `parts`.
pub fn combine(parts: &[Partial]) -> Option<Partial> {
    let d = parts.first()?.o.len();
    if parts.iter().any(|p| p.o.len() != d) {
        return None;
    }
    let m = parts.iter().map(|p| p.m).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = parts.iter().map(|p| p.l * libm::exp(p.m - m)).collect();
    let l: f64 = w.iter().sum();
    let mut o = vec![0f64; d];
    for (wi, p) in w.iter().zip(parts) {
        for (a, &x) in o.iter_mut().zip(&p.o) {
            *a += wi * x;
        }
    }
    o.iter_mut().for_each(|a| *a /= l);
    (m.is_finite() && l.is_finite() && o.iter().all(|a| a.is_finite())).then_some(Partial { o, m, l })
}

/// Largest difference between `a` and `b` beyond `floor` per value, in Q8 steps of each block of
/// 32 values of `scale`.
fn steps_on(a: &[f64], b: &[f64], scale: &[f64], floor: f64) -> Option<f64> {
    if a.len() != b.len() || a.len() != scale.len() || !b.len().is_multiple_of(32) {
        return None;
    }
    let mut dev = 0f64;
    for ((x, y), z) in a.chunks_exact(32).zip(b.chunks_exact(32)).zip(scale.chunks_exact(32)) {
        let step = (z.iter().fold(0f64, |m, v| m.max(v.abs())) / 127.0).max(f64::MIN_POSITIVE);
        dev = x.iter().zip(y).fold(dev, |m, (p, q)| m.max(((p - q).abs() - floor).max(0.0) / step));
    }
    Some(dev)
}

/// Log of the softmax mass of a partial.
pub fn lse(p: &Partial) -> f64 {
    p.m + libm::log(p.l)
}

/// Whether `a` and `b`, two partials of the same positions, agree.
pub fn same_partial(a: &Partial, b: &Partial) -> bool {
    (lse(a) - lse(b)).abs() <= LSE_TOL && steps_on(&a.o, &b.o, &b.o, LEAF_FLOOR).is_some_and(|d| d <= PARTIAL_STEPS)
}

/// Whether `children` combine into `parent`.
pub fn check_node(parent: &Partial, children: &[Partial]) -> bool {
    let Some(c) = combine(children) else { return false };
    let w: Vec<f64> = children.iter().map(|p| p.l * libm::exp(p.m - c.m) / c.l).collect();
    let scale: Vec<f64> = (0..c.o.len())
        .map(|k| children.iter().zip(&w).map(|(p, wi)| wi * p.o[k].abs()).sum::<f64>().max(parent.o.get(k).map_or(0.0, |x| x.abs())))
        .collect();
    (lse(&c) - lse(parent)).abs() <= LSE_TOL && steps_on(&c.o, &parent.o, &scale, NODE_FLOOR).is_some_and(|d| d <= PARTIAL_STEPS)
}

/// Whether the committed attention-output input `x` (`n_out` values, Q8_1 in `layout`) carries
/// head `head` as `children` combined, scaled by the sigmoid of `gate` when gated.
pub fn check_root(x: &[u8], layout: u8, n_out: usize, head: usize, children: &[Partial], gate: Option<&[f32]>) -> bool {
    let Some(mut o) = combine(children).map(|p| p.o) else { return false };
    let d = o.len();
    if !d.is_multiple_of(32) || (head + 1) * d > n_out {
        return false;
    }
    if let Some(g) = gate
        && crate::attn::apply_gate(&mut o, g).is_none()
    {
        return false;
    }
    root_within(x, layout, n_out, head * d / 32, &o)
}

/// Largest deviation, in quantization steps, of a value of the attention-output input.
pub const ATTN_OUT_MAX: f64 = 3.0;
/// Values of one block allowed past `ATTN_THRESHOLD`, up to `ATTN_OUT_MAX`.
pub const ATTN_OUT_SPARE: usize = 4;

/// Whether `o` matches the Q8 blocks of the attention-output input `x` from `first_block`.
pub fn root_within(x: &[u8], layout: u8, n_out: usize, first_block: usize, o: &[f64]) -> bool {
    let dev = crate::chain::block_outliers(x, layout, n_out, first_block, o, ATTN_THRESHOLD);
    dev.is_some_and(|(dv, spare)| dv <= ATTN_OUT_MAX && spare <= ATTN_OUT_SPARE)
}

/// [`check_root`] with the softmax mass of `children` matching the committed `lse` of the head.
pub fn check_root_lse(x: &[u8], layout: u8, n_out: usize, head: usize, children: &[Partial], gate: Option<&[f32]>, committed: f32) -> bool {
    committed.is_finite()
        && combine(children).is_some_and(|c| (lse(&c) - committed as f64).abs() <= root_lse_tol(c.m))
        && check_root(x, layout, n_out, head, children, gate)
}

/// Share of the uniform draw among children, out of 8.
const UNIFORM_EIGHTHS: u8 = 4;

/// Draw weights of `parts`: each child's mass times its largest output value, floored at
/// `2^-10` of the mass-weighted mean of those values, quantized to `u16` with every weight at
/// least one; uniform when every output is zero.
pub fn child_weights(parts: &[Partial]) -> Option<Vec<u16>> {
    let m = parts.iter().map(|p| p.m).fold(f64::NEG_INFINITY, f64::max);
    if parts.is_empty() || !m.is_finite() {
        return None;
    }
    let mass: Vec<f64> = parts.iter().map(|p| p.l * libm::exp(p.m - m)).collect();
    let peak: Vec<f64> = parts.iter().map(|p| p.o.iter().fold(0f64, |a, x| a.max(x.abs()))).collect();
    let total_mass: f64 = mass.iter().sum();
    let tau = mass.iter().zip(&peak).map(|(w, o)| w * o).sum::<f64>() / total_mass / 1024.0;
    let c: Vec<f64> = mass.iter().zip(&peak).map(|(w, &o)| w * (o + tau)).collect();
    let total: f64 = c.iter().sum();
    if !total_mass.is_finite() || total_mass <= 0.0 || !total.is_finite() {
        return None;
    }
    if total <= 0.0 {
        return Some(vec![1; parts.len()]);
    }
    Some(c.iter().map(|&x| ((x / total * 65535.0) as u64).clamp(1, 65535) as u16).collect())
}

/// Child drawn by the chain randomness `rand` among children of draw `weights`: uniformly with
/// probability 1/2, else in proportion to the weights.
pub fn draw_weighted_child(rand: &Hash, weights: &[u16]) -> u64 {
    let h = blake3::hash(&[rand.as_slice(), b"attn-child-w"].concat());
    let b = h.as_bytes();
    let n = weights.len().max(1) as u64;
    let r = u64::from_le_bytes(b[8..16].try_into().unwrap());
    if b[0] & 7 < UNIFORM_EIGHTHS {
        return r % n;
    }
    let total: u64 = weights.iter().map(|&w| w as u64).sum();
    let mut at = r % total.max(1);
    for (i, &w) in weights.iter().enumerate() {
        if at < w as u64 {
            return i as u64;
        }
        at -= w as u64;
    }
    n - 1
}

/// Whether `claimed` is the partial of query `q` (unscaled) over the rows of a leaf under one of
/// the kernel roundings.
pub fn check_leaf(q: &[f64], scale: f64, keys: &[Vec<f32>], values: &[Vec<f32>], claimed: &Partial) -> bool {
    ARITHS.iter().any(|&a| partial(a, q, scale, keys, values).is_some_and(|p| same_partial(&p, claimed)))
}

/// Partials end to end.
pub fn encode_partials(parts: &[Partial]) -> Vec<u8> {
    parts.iter().flat_map(encode_partial).collect()
}

fn partial_leaves(bytes: &[u8], d: usize) -> Option<Vec<Hash>> {
    let n = partial_len(d);
    (d > 0 && !bytes.is_empty() && bytes.len().is_multiple_of(n)).then(|| bytes.chunks_exact(n).map(|c| *blake3::hash(c).as_bytes()).collect())
}

/// Root over the encoded partials `bytes` of `d` values, one leaf per partial.
pub fn partials_root(bytes: &[u8], d: usize) -> Option<Hash> {
    partial_leaves(bytes, d).map(|l| b3_root(&l))
}

/// Encoded partial `i` of `bytes` and its path under `partials_root`.
pub fn partial_opening(bytes: &[u8], d: usize, i: usize) -> Option<(Vec<u8>, Vec<Hash>)> {
    let leaves = partial_leaves(bytes, d)?;
    let n = partial_len(d);
    (i < leaves.len()).then(|| (bytes[i * n..(i + 1) * n].to_vec(), b3_path(&leaves, i)))
}

/// Partial `i` of the `n` committed under `root`, from its encoding and path.
pub fn open_partial(root: &Hash, n: u64, i: u64, bytes: &[u8], path: &[Hash], d: usize) -> Option<Partial> {
    if bytes.len() != partial_len(d) || !b3_verify(root, n, i, blake3::hash(bytes).as_bytes(), path) {
        return None;
    }
    decode_partials(bytes, d)?.pop()
}

/// Child drawn by the chain randomness `rand` among `n`.
pub fn draw_child(rand: &Hash, n: u64) -> u64 {
    let h = blake3::hash(&[rand.as_slice(), b"attn-child"].concat());
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap()) % n.max(1)
}

/// Partials under `arith` of the children of the node at `path`, from query `q` (unscaled) and
/// the cache rows of positions `lo..`.
#[allow(clippy::too_many_arguments)]
pub fn children_partials(arith: Arith, q: &[f64], scale: f64, keys: &[Vec<f32>], values: &[Vec<f32>], lo: u64, tree: &BlockTree, path: &[u64]) -> Option<Vec<Partial>> {
    tree.children(path)?
        .into_iter()
        .map(|(p0, p1)| {
            let r = p0.checked_sub(lo)? as usize..p1.checked_sub(lo)? as usize;
            partial(arith, q, scale, keys.get(r.clone())?, values.get(r)?)
        })
        .collect()
}

/// Encoded partials of the children of `path` under the first rounding whose decoded partials
/// pass `check`.
#[allow(clippy::too_many_arguments)]
pub fn pick_partials(q: &[f64], scale: f64, keys: &[Vec<f32>], values: &[Vec<f32>], lo: u64, tree: &BlockTree, path: &[u64], check: impl Fn(&[Partial]) -> bool) -> Option<Vec<u8>> {
    let d = values.first()?.len();
    ARITHS.iter().find_map(|&a| {
        let enc = encode_partials(&children_partials(a, q, scale, keys, values, lo, tree, path)?);
        decode_partials(&enc, d).is_some_and(|p| check(&p)).then_some(enc)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q8;
    use crate::testkit::q8_1_row;

    const D: usize = 64;
    const SCALE: f64 = 0.125;

    fn vecf(seed: u64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (((i as u64).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2001) as f64 / 1000.0 - 1.0).collect()
    }

    fn rows(seed: u64, n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|p| vecf(seed + p as u64, D).iter().map(|&x| x as f32).collect()).collect()
    }

    struct Case {
        q: Vec<f64>,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        tree: BlockTree,
    }

    fn case(lo: u64, hi: u64, c: u64, b: u64) -> Case {
        Case { q: vecf(3, D).iter().map(|x| 2.0 * x).collect(), keys: rows(10, hi as usize), values: rows(5000, hi as usize), tree: BlockTree::new(lo, hi, c, b).unwrap() }
    }

    impl Case {
        fn run(&self, (p0, p1): (u64, u64), arith: Arith) -> Partial {
            let r = p0 as usize..p1 as usize;
            partial(arith, &self.q, SCALE, &self.keys[r.clone()], &self.values[r]).unwrap()
        }

        fn children(&self, path: &[u64], arith: Arith) -> Vec<Partial> {
            self.tree.children(path).unwrap().into_iter().map(|r| self.run(r, arith)).collect()
        }

        fn decoded(&self, path: &[u64], arith: Arith) -> Vec<Partial> {
            let bytes: Vec<u8> = self.children(path, arith).iter().flat_map(encode_partial).collect();
            decode_partials(&bytes, D).unwrap()
        }

        fn x(&self, arith: Arith) -> Vec<u8> {
            let mut all = vec![0.0; D];
            all.extend(self.run((self.tree.lo, self.tree.hi), arith).o);
            q8_1_row(&all)
        }

        fn leaf_rows(&self, path: &[u64]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let (p0, p1) = self.tree.node(path).unwrap();
            (self.keys[p0 as usize..p1 as usize].to_vec(), self.values[p0 as usize..p1 as usize].to_vec())
        }
    }

    #[test]
    fn block_params_follow_the_tier_shapes() {
        let q8 = |n: usize| n / 32 * 34;
        assert_eq!(block_params(2 * q8(128), partial_len(128)), Some((256, 256)));
        assert_eq!(block_params(2 * q8(256), partial_len(256)), Some((128, 128)));
        assert_eq!(block_params(2 * q8(512), partial_len(512)), Some((64, 64)));
        assert_eq!(block_params(q8(576), partial_len(512)), Some((128, 64)));
        assert_eq!(block_params(BLOCK_BUDGET + 1, 8), None);
        assert_eq!(block_params(8, BLOCK_BUDGET), None);
    }

    #[test]
    fn the_tree_covers_every_position_once_on_aligned_leaves() {
        for (lo, hi, c, b) in [(0, 1, 4, 2), (0, 300, 64, 4), (5, 300, 64, 4), (70, 1100, 16, 3), (0, 32768, 64, 64), (31744, 32768, 128, 128)] {
            let t = BlockTree::new(lo, hi, c, b).unwrap();
            let d = t.depth();
            assert!(b.pow(d) >= t.n_leaves() && (d == 0 || b.pow(d - 1) < t.n_leaves()));
            assert_eq!(t.node(&[]), Some((lo, hi)));
            let mut stack = vec![Vec::<u64>::new()];
            let mut leaves = Vec::new();
            while let Some(path) = stack.pop() {
                match t.children(&path) {
                    Some(ch) => {
                        let (p0, p1) = t.node(&path).unwrap();
                        assert_eq!((ch[0].0, ch.last().unwrap().1), (p0, p1));
                        assert!(ch.windows(2).all(|w| w[0].1 == w[1].0) && ch.len() as u64 <= b);
                        assert_eq!(t.node(&[path.clone(), vec![ch.len() as u64]].concat()), None);
                        stack.extend((0..ch.len() as u64).map(|i| [path.clone(), vec![i]].concat()));
                    }
                    None => {
                        assert_eq!(path.len(), d as usize);
                        leaves.push(t.node(&path).unwrap());
                    }
                }
            }
            leaves.sort();
            assert_eq!(leaves.len() as u64, t.n_leaves());
            assert_eq!((leaves[0].0, leaves.last().unwrap().1), (lo, hi));
            assert!(leaves.windows(2).all(|w| w[0].1 == w[1].0 && w[1].0 % c == 0));
            assert!(leaves.iter().all(|&(p0, p1)| p1 - p0 <= c && p0 / c == (p1 - 1) / c));
        }
        assert_eq!(BlockTree::new(0, 32768, 256, 256).unwrap().depth(), 1);
        assert_eq!(BlockTree::new(0, 32768, 64, 64).unwrap().depth(), 2);
        assert_eq!(BlockTree::new(3, 3, 4, 2), None);
    }

    #[test]
    fn partials_round_trip_and_reject_bad_encodings() {
        let k = case(0, 40, 16, 2);
        let p = k.run((0, 40), Arith::Exact);
        let e = encode_partial(&p);
        assert_eq!(e.len(), partial_len(D));
        let back = &decode_partials(&e, D).unwrap()[0];
        assert!(same_partial(back, &p));
        assert_eq!(decode_partials(&e[1..], D), None);
        let mut low = e.clone();
        low[2 * D + 4..].copy_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(decode_partials(&low, D), None);
        let mut nan = e.clone();
        nan[2 * D..2 * D + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(decode_partials(&nan, D), None);
    }

    #[test]
    fn combined_leaves_give_the_whole_run() {
        let k = case(5, 300, 16, 4);
        for a in ARITHS {
            let whole = k.run((5, 300), a);
            let leaves: Vec<Partial> = (0..k.tree.n_leaves()).map(|i| {
                let first = 5 / 16;
                k.run((5u64.max((first + i) * 16), 300u64.min((first + i + 1) * 16)), a)
            }).collect();
            let c = combine(&leaves).unwrap();
            assert!(same_partial(&c, &whole), "{a:?}");
            if a == Arith::Exact {
                assert!(c.o.iter().zip(&whole.o).all(|(x, y)| (x - y).abs() < 1e-12));
            }
        }
    }

    #[test]
    fn an_honest_opening_passes_every_phase_on_every_path() {
        let k = case(5, 300, 16, 5);
        assert_eq!(k.tree.depth(), 2);
        for a in ARITHS {
            let x = k.x(a);
            assert!(check_root(&x, q8::MMVQ, 2 * D, 1, &k.decoded(&[], a), None), "{a:?}");
            for i in 0..k.tree.children(&[]).unwrap().len() as u64 {
                let parent = &k.decoded(&[], a)[i as usize];
                assert!(check_node(parent, &k.decoded(&[i], a)), "{a:?} {i}");
                for j in 0..k.tree.children(&[i]).unwrap().len() as u64 {
                    let claimed = &k.decoded(&[i], a)[j as usize];
                    let (kr, vr) = k.leaf_rows(&[i, j]);
                    assert!(check_leaf(&k.q, SCALE, &kr, &vr, claimed), "{a:?} {i} {j}");
                }
            }
        }
    }

    #[test]
    fn an_inconsistent_root_fails_at_the_node_or_the_leaf() {
        let k = case(0, 256, 16, 4);
        let honest = k.decoded(&[], Arith::Exact);
        let mut forged_x = vec![0.0; D];
        forged_x.extend(vecf(77, D).iter().map(|v| v * 0.3));
        let forged_x = q8_1_row(&forged_x);
        assert!(!check_root(&forged_x, q8::MMVQ, 2 * D, 1, &honest, None));
        let mut lie = honest.clone();
        let o: Vec<f64> = vecf(77, D).iter().map(|v| v * 0.3).collect();
        lie[2] = decode_partials(&encode_partial(&Partial { o, m: honest[2].m + 60.0, l: 1.0 }), D).unwrap().remove(0);
        assert!(check_root(&forged_x, q8::MMVQ, 2 * D, 1, &lie, None));
        let mass = lse(&k.run((0, 256), Arith::Exact)) as f32;
        assert!(!check_root_lse(&forged_x, q8::MMVQ, 2 * D, 1, &lie, None, mass));
        assert!(!check_node(&lie[2], &k.decoded(&[2], Arith::Exact)));
        for j in 0..4u64 {
            let (kr, vr) = k.leaf_rows(&[2, j]);
            assert!(check_leaf(&k.q, SCALE, &kr, &vr, &k.decoded(&[2], Arith::Exact)[j as usize]));
        }
    }

    #[test]
    fn the_root_tolerance_widens_with_the_largest_score_up_to_a_cap() {
        assert_eq!(root_lse_tol(0.0), LSE_ROOT_TOL);
        assert_eq!(root_lse_tol(-147.0), root_lse_tol(147.0));
        assert!(root_lse_tol(147.0) > 0.0070 * 4.0);
        assert_eq!(root_lse_tol(1e9), root_lse_tol(176.0));
        assert_eq!(root_lse_tol(176.0), LSE_ROOT_TOL + 176.0 / 4096.0);
    }

    #[test]
    fn the_committed_mass_binds_the_root() {
        let k = case(0, 256, 16, 4);
        for a in ARITHS {
            let honest = k.decoded(&[], a);
            let x = k.x(a);
            let mass = lse(&k.run((0, 256), a)) as f32;
            assert!(check_root_lse(&x, q8::MMVQ, 2 * D, 1, &honest, None, mass), "{a:?}");
            assert!(!check_root_lse(&x, q8::MMVQ, 2 * D, 1, &honest, None, mass + (2.0 * root_lse_tol(176.0)) as f32), "{a:?}");
            assert!(!check_root_lse(&x, q8::MMVQ, 2 * D, 1, &honest, None, f32::NAN), "{a:?}");
        }
        let honest = k.decoded(&[], Arith::Exact);
        let top = (0..honest.len()).max_by(|&i, &j| lse(&honest[i]).total_cmp(&lse(&honest[j]))).unwrap();
        let mut suppressed = honest.clone();
        suppressed[top].m -= 8.0;
        let c = combine(&suppressed).unwrap();
        let mut all = vec![0.0; D];
        all.extend(c.o.iter().copied());
        let forged_x = q8_1_row(&all);
        assert!(check_root(&forged_x, q8::MMVQ, 2 * D, 1, &suppressed, None));
        let mass = lse(&k.run((0, 256), Arith::Exact)) as f32;
        assert!(!check_root_lse(&forged_x, q8::MMVQ, 2 * D, 1, &suppressed, None, mass));
    }

    #[test]
    fn weighted_draws_follow_the_contribution_with_a_uniform_floor() {
        let n = 10_000u32;
        let count = |w: &[u16], i: u64| (0..n).filter(|s| draw_weighted_child(blake3::hash(&s.to_le_bytes()).as_bytes(), w) == i).count() as f64 / n as f64;
        let heavy = [1u16, 1, 65533];
        let u = UNIFORM_EIGHTHS as f64 / 8.0;
        assert!((count(&heavy, 2) - (1.0 - u + u / 3.0)).abs() < 0.02);
        assert!((count(&heavy, 0) - u / 3.0).abs() < 0.015, "the uniform share reaches a near-zero child");
        let even = [100u16; 4];
        for i in 0..4 {
            assert!((count(&even, i) - 0.25).abs() < 0.02);
        }
        assert!((0..n).all(|s| draw_weighted_child(blake3::hash(&s.to_le_bytes()).as_bytes(), &[7, 9, 11]) < 3));
        assert_eq!(draw_weighted_child(&[0; 32], &[]), 0);
    }

    #[test]
    fn a_dominating_child_draws_the_bulk_of_the_weight() {
        let k = case(0, 256, 16, 4);
        let honest = k.decoded(&[], Arith::Exact);
        let w = child_weights(&honest).unwrap();
        assert_eq!(w.len(), honest.len());
        assert!(w.iter().all(|&x| x >= 1));
        let mut lie = honest.clone();
        lie[2].m += 60.0;
        let w = child_weights(&lie).unwrap();
        assert!(w[2] as f64 / w.iter().map(|&x| x as f64).sum::<f64>() > 0.99);
        assert_eq!(child_weights(&[]), None);
        let mut nan = honest.clone();
        nan[0].m = f64::NAN;
        assert_eq!(child_weights(&nan), None);
        let mut zero = honest.clone();
        zero.iter_mut().for_each(|p| p.o.iter_mut().for_each(|x| *x = 0.0));
        assert_eq!(child_weights(&zero), Some(vec![1; honest.len()]));
    }

    #[test]
    fn a_negligible_child_cannot_raise_the_floor() {
        let k = case(0, 256, 16, 4);
        let honest = k.decoded(&[], Arith::Exact);
        let low = honest.iter().map(|p| p.m).fold(f64::INFINITY, f64::min);
        let mut quiet = honest.clone();
        quiet[0] = Partial { o: vec![0.0; honest[0].o.len()], m: low - 40.0, l: 1.0 };
        let mut loud = quiet.clone();
        loud[0].o.iter_mut().for_each(|x| *x = 65504.0);
        assert_eq!(child_weights(&loud), child_weights(&quiet));
    }

    #[test]
    fn inconsistent_partials_fail_their_checks() {
        let k = case(0, 256, 16, 4);
        let parent = &k.decoded(&[], Arith::Exact)[1];
        let children = k.decoded(&[1], Arith::Exact);
        let (kr, vr) = k.leaf_rows(&[1, 3]);
        let leaf = &children[3];
        let mut bad_m = leaf.clone();
        bad_m.m += 0.05;
        let mut bad_l = leaf.clone();
        bad_l.l *= 1.05;
        let mut bad_o = leaf.clone();
        bad_o.o[7] += 0.1;
        for bad in [&bad_m, &bad_l, &bad_o] {
            assert!(!check_leaf(&k.q, SCALE, &kr, &vr, bad), "{bad:?}");
        }
        let (kr2, vr2) = k.leaf_rows(&[1, 2]);
        assert!(!check_leaf(&k.q, SCALE, &kr2, &vr2, leaf));
        assert!(!check_node(parent, &children[..3]));
        let mut swapped = children.clone();
        swapped.swap(0, 3);
        assert!(check_node(parent, &swapped));
    }

    #[test]
    fn a_leaf_off_by_less_than_a_step_is_refused() {
        let k = case(0, 256, 16, 4);
        let (kr, vr) = k.leaf_rows(&[1, 3]);
        let leaf = &k.decoded(&[1], Arith::Exact)[3];
        assert!(check_leaf(&k.q, SCALE, &kr, &vr, leaf));
        let step = leaf.o[..32].iter().fold(0f64, |m, v| m.max(v.abs())) / 127.0;
        let i = (0..32).min_by(|&a, &b| leaf.o[a].abs().total_cmp(&leaf.o[b].abs())).unwrap();
        let mut off = leaf.clone();
        off.o[i] += 0.4 * step;
        let off = decode_partials(&encode_partial(&off), D).unwrap().remove(0);
        assert!(!check_leaf(&k.q, SCALE, &kr, &vr, &off));
    }

    #[test]
    fn honest_children_that_cancel_still_combine_into_their_parent() {
        let small = |seed| vecf(seed, D).iter().map(|v| v * 0.5).collect::<Vec<f64>>();
        let big: Vec<f64> = vecf(5, D).iter().map(|v| v * 40.0).collect();
        let a = Partial { o: big.iter().zip(small(6)).map(|(x, y)| x + y).collect(), m: 3.0, l: 2.0 };
        let b = Partial { o: big.iter().zip(small(7)).map(|(x, y)| y - x).collect(), m: 3.0, l: 2.0 };
        let round = |p: &Partial| decode_partials(&encode_partial(p), D).unwrap().remove(0);
        let parent = round(&combine(&[a.clone(), b.clone()]).unwrap());
        let children = [round(&a), round(&b)];
        assert!(steps_on(&combine(&children).unwrap().o, &parent.o, &parent.o, 0.0).unwrap() > PARTIAL_STEPS, "the parent scale alone would refuse it");
        assert!(check_node(&parent, &children));
        let mut shifted = children.clone();
        shifted[0].o.iter_mut().for_each(|x| *x += 1.0);
        assert!(!check_node(&parent, &shifted));
    }

    #[test]
    fn honest_partials_of_tiny_values_still_pass() {
        let round = |p: &Partial| decode_partials(&encode_partial(p), D).unwrap().remove(0);
        for scale in [1e-5, 3e-6, 1e-7, 1e-9] {
            let k = case(0, 256, 16, 4);
            let tiny = |p: &Partial| Partial { o: p.o.iter().map(|x| x * scale).collect(), ..p.clone() };
            let children: Vec<Partial> = k.children(&[], Arith::Exact).iter().map(tiny).collect();
            let parent = round(&combine(&children).unwrap());
            let rounded: Vec<Partial> = children.iter().map(round).collect();
            assert!(rounded.iter().zip(&children).all(|(r, c)| same_partial(c, r)), "leaves at {scale}");
            assert!(check_node(&parent, &rounded), "node at {scale}");
        }
    }

    #[test]
    fn committed_partials_open_only_at_their_index() {
        let k = case(0, 256, 16, 4);
        let bytes = encode_partials(&k.children(&[], Arith::Exact));
        let root = partials_root(&bytes, D).unwrap();
        for i in 0..4 {
            let (b, path) = partial_opening(&bytes, D, i).unwrap();
            let p = open_partial(&root, 4, i as u64, &b, &path, D).unwrap();
            assert_eq!(p, decode_partials(&b, D).unwrap()[0]);
            assert_eq!(open_partial(&root, 4, (i as u64 + 1) % 4, &b, &path, D), None);
            let mut flipped = b.clone();
            flipped[5] ^= 1;
            assert_eq!(open_partial(&root, 4, i as u64, &flipped, &path, D), None);
            assert_eq!(open_partial(&root, 4, i as u64, &b[1..], &path, D), None);
        }
        assert_eq!(partial_opening(&bytes, D, 4), None);
        assert_eq!(partials_root(&bytes[1..], D), None);
        assert_eq!(partials_root(&[], D), None);
    }

    #[test]
    fn drawn_children_stay_in_range_and_spread() {
        for n in [1u64, 2, 5, 64, 512] {
            let mut seen = std::collections::HashSet::new();
            for s in 0..10_000u32 {
                let i = draw_child(blake3::hash(&s.to_le_bytes()).as_bytes(), n);
                assert!(i < n);
                seen.insert(i);
            }
            assert_eq!(seen.len() as u64, n.min(512));
        }
    }

    #[test]
    fn picked_partials_pass_the_check_they_are_picked_for() {
        let k = case(5, 300, 16, 5);
        let x = k.x(Arith::F16);
        let picked = pick_partials(&k.q, SCALE, &k.keys[5..], &k.values[5..], 5, &k.tree, &[], |p| check_root(&x, q8::MMVQ, 2 * D, 1, p, None)).unwrap();
        assert!(check_root(&x, q8::MMVQ, 2 * D, 1, &decode_partials(&picked, D).unwrap(), None));
        assert_eq!(pick_partials(&k.q, SCALE, &k.keys[5..], &k.values[5..], 5, &k.tree, &[], |_| false), None);
        assert_eq!(pick_partials(&k.q, SCALE, &k.keys[6..], &k.values[6..], 6, &k.tree, &[], |_| true), None);
    }

    #[test]
    fn the_gate_scales_the_combined_head() {
        let k = case(0, 64, 16, 4);
        let gate: Vec<f32> = vecf(9, D).iter().map(|&x| 3.0 * x as f32).collect();
        let mut o = k.run((0, 64), Arith::Exact).o;
        crate::attn::apply_gate(&mut o, &gate).unwrap();
        let mut all = vec![0.0; D];
        all.extend(o);
        let x = q8_1_row(&all);
        let ch = k.decoded(&[], Arith::Exact);
        assert!(check_root(&x, q8::MMVQ, 2 * D, 1, &ch, Some(&gate)));
        assert!(!check_root(&x, q8::MMVQ, 2 * D, 1, &ch, None));
        assert!(!check_root(&x, q8::MMVQ, 2 * D, 0, &ch, Some(&gate)));
    }
}
