//! Recomputation of one attention head against the committed key/value cache.

/// Rotary embedding kind of adjacent pairs `(2i, 2i + 1)`.
pub const ROPE_NORM: u32 = 0;
/// Rotary embedding kind of pairs `(i, i + n_rot / 2)`.
pub const ROPE_NEOX: u32 = 1;

fn theta(pos: u64, i: usize, n_rot: usize, base: f64, ff: Option<&[f32]>) -> f64 {
    pos as f64 * libm::pow(base, -2.0 * i as f64 / n_rot as f64) / ff.map_or(1.0, |f| f[i] as f64)
}

/// Rotary embedding of adjacent pairs of `x` at position `pos` over its first `n_rot` values;
/// `ff` divides the angle of each pair.
pub fn rope_norm(x: &mut [f64], pos: u64, n_rot: usize, base: f64, ff: Option<&[f32]>) {
    for i in 0..n_rot / 2 {
        let theta = theta(pos, i, n_rot, base, ff);
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// NEOX rotary embedding of `x` at position `pos` over its first `n_rot` values; `ff` divides
/// the angle of each pair.
pub fn rope_neox(x: &mut [f64], pos: u64, n_rot: usize, base: f64, ff: Option<&[f32]>) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = theta(pos, i, n_rot, base, ff);
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

/// RMS normalization of `x`, scaled by `w` when given.
pub fn rms_norm(x: &mut [f64], w: Option<&[f32]>, eps: f64) {
    let ms = x.iter().map(|a| a * a).sum::<f64>() / x.len() as f64;
    let r = 1.0 / libm::sqrt(ms + eps);
    for (i, a) in x.iter_mut().enumerate() {
        *a *= r * w.map_or(1.0, |w| w[i] as f64);
    }
}

/// Output of query `q` over `keys` and `values` (one per attended position), scores scaled by
/// `scale`; `None` on a shape mismatch, an empty context or a non-finite value.
pub fn head_output(q: &[f64], keys: &[Vec<f32>], values: &[Vec<f32>], scale: f64) -> Option<Vec<f64>> {
    partial(Arith::Exact, q, scale, keys, values).map(|p| p.o)
}

/// Softmax partial of one query over a run of positions: `o` the weighted mean of the values,
/// `m` the largest scaled score, `l` the sum of `exp(score - m)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    pub o: Vec<f64>,
    pub m: f64,
    pub l: f64,
}

/// Partial of query `q` (unscaled) over `keys` and `values` under the rounding `arith`; `None` on
/// a shape mismatch, an empty run or a non-finite value.
pub fn partial(arith: Arith, q: &[f64], scale: f64, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Option<Partial> {
    if keys.is_empty() || keys.len() != values.len() || keys.iter().any(|k| k.len() != q.len()) {
        return None;
    }
    let d = values[0].len();
    if values.iter().any(|v| v.len() != d) {
        return None;
    }
    let h = |x: f32| match arith {
        Arith::F16 => crate::f16::round(x) as f64,
        _ => x as f64,
    };
    let (qv, sc) = match arith {
        Arith::Exact => (q.to_vec(), scale),
        Arith::Q8Query => (q8_1_query(q, scale)?, scale),
        Arith::F16 => {
            let sh = f16_round(scale);
            (q.iter().map(|&x| f16_round(f16_round(x) * sh)).collect(), 1.0)
        }
    };
    let scores: Vec<f64> = keys.iter().map(|k| qv.iter().zip(k).map(|(&a, &b)| a * h(b)).sum::<f64>() * sc).collect();
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return None;
    }
    let w: Vec<f64> = scores.iter().map(|&x| libm::exp(x - m)).collect();
    let l: f64 = w.iter().sum();
    let mut o = vec![0f64; d];
    for (&wi, v) in w.iter().zip(values) {
        let wi = if arith == Arith::F16 { f16_round(wi) } else { wi };
        for (a, &x) in o.iter_mut().zip(v) {
            *a += wi * h(x);
        }
    }
    o.iter_mut().for_each(|a| *a /= l);
    o.iter().all(|a| a.is_finite()).then_some(Partial { o, m, l })
}

/// Shape of the attention of a layer.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_kind: u32,
    pub n_rot: usize,
    pub rope_base: f64,
    pub kq_scale: f64,
}

impl AttnShape {
    /// Query heads per key/value head, or `None` for an inconsistent shape.
    pub fn group(&self) -> Option<usize> {
        let ok = self.n_head_kv > 0
            && self.n_head.is_multiple_of(self.n_head_kv)
            && self.n_rot <= self.head_dim
            && self.n_rot.is_multiple_of(2)
            && matches!(self.rope_kind, ROPE_NORM | ROPE_NEOX);
        ok.then(|| self.n_head / self.n_head_kv)
    }

    /// Rotary embedding of one head vector at position `pos`, angles divided by `ff`.
    pub fn rope(&self, x: &mut [f64], pos: u64, ff: Option<&[f32]>) {
        match self.rope_kind {
            ROPE_NORM => rope_norm(x, pos, self.n_rot, self.rope_base, ff),
            _ => rope_neox(x, pos, self.n_rot, self.rope_base, ff),
        }
    }
}

/// Per-head transforms from the projections to the attended query, key and value.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prep<'a> {
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// Unweighted RMS normalization of the value.
    pub v_norm: bool,
    pub freqs: Option<&'a [f32]>,
    pub eps: f64,
}

impl Prep<'_> {
    /// Whether the weights fit `shape`.
    pub fn fits(&self, shape: &AttnShape) -> bool {
        let hd = shape.head_dim;
        self.q_norm.is_none_or(|w| w.len() == hd)
            && self.k_norm.is_none_or(|w| w.len() == hd)
            && self.freqs.is_none_or(|f| f.len() == shape.n_rot / 2 && f.iter().all(|&x| x.is_finite() && x > 0.0))
    }

    fn rotated(&self, shape: &AttnShape, raw: &[f32], norm: Option<&[f32]>, pos: u64) -> Vec<f64> {
        let mut x: Vec<f64> = raw.iter().map(|&a| a as f64).collect();
        if norm.is_some() {
            rms_norm(&mut x, norm, self.eps);
        }
        shape.rope(&mut x, pos, self.freqs);
        x
    }

    /// Attended query of one head at position `pos`.
    pub fn query(&self, shape: &AttnShape, raw: &[f32], pos: u64) -> Vec<f64> {
        self.rotated(shape, raw, self.q_norm, pos)
    }

    /// Cached key of one head at position `pos`.
    pub fn key(&self, shape: &AttnShape, raw: &[f32], pos: u64) -> Vec<f64> {
        self.rotated(shape, raw, self.k_norm, pos)
    }

    /// Cached value of one head.
    pub fn value(&self, raw: &[f32]) -> Vec<f64> {
        let mut x: Vec<f64> = raw.iter().map(|&a| a as f64).collect();
        if self.v_norm {
            rms_norm(&mut x, None, self.eps);
        }
        x
    }
}

/// Outputs, end to end, of the query heads sharing key/value head `kv` at position `t`, from the
/// query projection `q_proj` (all heads) and the group's cached `keys`/`values`.
pub fn group_output(shape: &AttnShape, prep: &Prep, q_proj: &[f32], kv: usize, t: u64, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Option<Vec<f64>> {
    group_output_in(Arith::Exact, shape, prep, q_proj, kv, t, keys, values)
}

/// Rounding of an attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    Exact,
    /// Scaled query rounded through Q8_1.
    Q8Query,
    /// Scaled query, keys, values and weights rounded to f16.
    F16,
}

pub const ARITHS: [Arith; 3] = [Arith::Exact, Arith::Q8Query, Arith::F16];

/// `group_output` under the rounding `arith`.
#[allow(clippy::too_many_arguments)]
pub fn group_output_in(arith: Arith, shape: &AttnShape, prep: &Prep, q_proj: &[f32], kv: usize, t: u64, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Option<Vec<f64>> {
    let g = shape.group()?;
    let hd = shape.head_dim;
    if kv >= shape.n_head_kv || q_proj.len() != shape.n_head * hd || !prep.fits(shape) {
        return None;
    }
    let mut out = Vec::with_capacity(g * hd);
    for qi in kv * g..(kv + 1) * g {
        let q = prep.query(shape, &q_proj[qi * hd..(qi + 1) * hd], t);
        out.extend(partial(arith, &q, shape.kq_scale, keys, values)?.o);
    }
    Some(out)
}

/// `x` rounded to the nearest f16 (ties to even); infinite beyond the f16 range.
fn f16_round(x: f64) -> f64 {
    crate::f16::round(x as f32) as f64
}

/// `q` after `q * scale` is rounded through Q8_1 (f32 blocks of 32), divided back by `scale`.
fn q8_1_query(q: &[f64], scale: f64) -> Option<Vec<f64>> {
    if !q.len().is_multiple_of(32) || !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let mut out = Vec::with_capacity(q.len());
    for b in q.chunks_exact(32) {
        let xs: Vec<f32> = b.iter().map(|&x| (x * scale) as f32).collect();
        let d = xs.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
        out.extend(xs.iter().map(|&x| if d == 0.0 { 0.0 } else { (x / d).round() as f64 * d as f64 / scale }));
    }
    Some(out)
}

/// Scales each head of `out` by the sigmoid of its gate.
pub fn apply_gate(out: &mut [f64], gate: &[f32]) -> Option<()> {
    if out.len() != gate.len() {
        return None;
    }
    for (o, &g) in out.iter_mut().zip(gate) {
        *o /= 1.0 + libm::exp(-(g as f64));
    }
    out.iter().all(|o| o.is_finite()).then_some(())
}

/// Largest deviation, in Q8_1 steps, of a recomputed attention output.
pub const ATTN_THRESHOLD: f64 = 2.0;

/// Whether the committed attention-output input `x` (all heads, Q8_1 in `layout`) carries
/// `group`, the recomputed outputs of the query heads sharing key/value head `kv`.
pub fn check_group(shape: &AttnShape, x: &[u8], layout: u8, kv: usize, group: &[f64]) -> bool {
    let (Some(g), hd) = (shape.group(), shape.head_dim) else { return false };
    if group.len() != g * hd || hd % 32 != 0 {
        return false;
    }
    let first = kv * g * hd / 32;
    crate::chain::block_deviation(x, layout, shape.n_head * hd, first, group).is_some_and(|d| d <= ATTN_THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::block_deviation;
    use crate::q8;
    use crate::testkit::q8_1_row;

    const DIM: usize = 128;

    #[test]
    fn q8_1_query_rounds_each_block_to_half_a_step() {
        let q: Vec<f64> = (0..64).map(|i| ((i * 37 % 61) as f64 - 30.0) / 7.0).collect();
        let s = 0.125;
        let r = q8_1_query(&q, s).unwrap();
        for (b, rb) in q.chunks(32).zip(r.chunks(32)) {
            let step = b.iter().fold(0f64, |m, x| m.max((x * s).abs())) / 127.0;
            assert!(b.iter().zip(rb).all(|(x, y)| ((x - y) * s).abs() <= step / 2.0 + 1e-6));
        }
        assert!(q8_1_query(&q[..40], s).is_none());
        assert!(q8_1_query(&q, 0.0).is_none());
    }

    fn vecf(seed: u64, n: usize, a: f64) -> Vec<f64> {
        (0..n).map(|i| a * (((i as u64).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2001) as f64 / 1000.0 - a).collect()
    }

    fn kv(t: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let keys = (0..=t).map(|p| vecf(10 + p as u64, DIM, 1.0).iter().map(|&x| x as f32).collect()).collect();
        let values = (0..=t).map(|p| vecf(50 + p as u64, DIM, 1.0).iter().map(|&x| x as f32).collect()).collect();
        (keys, values)
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let x = vecf(1, DIM, 1.0);
        let mut y = x.clone();
        rope_neox(&mut y, 0, DIM, 10_000.0, None);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn norm_rope_turns_adjacent_pairs_and_keeps_the_rest() {
        let x = vecf(5, DIM, 1.0);
        let mut y = x.clone();
        rope_norm(&mut y, 0, DIM / 2, 10_000.0, None);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-12));
        rope_norm(&mut y, 9, DIM / 2, 10_000.0, None);
        assert!(x[DIM / 2..].iter().zip(&y[DIM / 2..]).all(|(a, b)| a == b));
        let n = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
        assert!((n(&x) - n(&y)).abs() < 1e-9);
        let mut z = x.clone();
        rope_neox(&mut z, 9, DIM / 2, 10_000.0, None);
        assert!(y.iter().zip(&z).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn rope_keeps_the_norm() {
        let x = vecf(2, DIM, 1.0);
        let mut y = x.clone();
        rope_neox(&mut y, 37, DIM, 10_000.0, None);
        let n = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
        assert!((n(&x) - n(&y)).abs() < 1e-9);
    }

    #[test]
    fn an_honest_head_output_passes() {
        let (keys, values) = kv(20);
        let q = vecf(3, DIM, 1.0);
        let out = head_output(&q, &keys, &values, 1.0 / (DIM as f64).sqrt()).unwrap();
        let x = q8_1_row(&out);
        assert!(block_deviation(&x, q8::MMVQ, DIM, 0, &out).is_some_and(|d| d <= 1.0));
    }

    #[test]
    fn a_zeroed_head_output_fails() {
        let (keys, values) = kv(20);
        let q = vecf(3, DIM, 1.0);
        let out = head_output(&q, &keys, &values, 1.0 / (DIM as f64).sqrt()).unwrap();
        let x = q8_1_row(&vec![0.0; DIM]);
        assert!(!block_deviation(&x, q8::MMVQ, DIM, 0, &out).is_some_and(|d| d <= 1.0));
    }

    #[test]
    fn a_changed_key_changes_the_output() {
        let (keys, values) = kv(20);
        let q = vecf(3, DIM, 1.0);
        let scale = 1.0 / (DIM as f64).sqrt();
        let honest = head_output(&q, &keys, &values, scale).unwrap();
        let mut bad = keys.clone();
        bad[7] = vecf(999, DIM, 1.0).iter().map(|&x| x as f32).collect();
        let forged = head_output(&q, &bad, &values, scale).unwrap();
        let x = q8_1_row(&forged);
        assert!(!block_deviation(&x, q8::MMVQ, DIM, 0, &honest).is_some_and(|d| d <= 1.0));
    }

    const SHAPE: AttnShape = AttnShape { n_head: 8, n_head_kv: 2, head_dim: 64, rope_kind: ROPE_NEOX, n_rot: 64, rope_base: 10_000.0, kq_scale: 0.125 };

    fn rotated(seed: u64, p: usize) -> Vec<f32> {
        let mut k = vecf(seed + p as u64, SHAPE.head_dim, 1.0);
        SHAPE.rope(&mut k, p as u64, None);
        k.iter().map(|&x| x as f32).collect()
    }

    fn group_kv(kv: usize, t: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let keys = (0..=t).map(|p| rotated(100 * (kv as u64 + 1), p)).collect();
        let values = (0..=t).map(|p| vecf(700 * (kv as u64 + 1) + p as u64, SHAPE.head_dim, 1.0).iter().map(|&x| x as f32).collect()).collect();
        (keys, values)
    }

    fn q_proj() -> Vec<f32> {
        vecf(4, SHAPE.n_head * SHAPE.head_dim, 1.0).iter().map(|&x| x as f32).collect()
    }

    /// Attention-output input with every head honest.
    fn honest_x(t: usize) -> Vec<u8> {
        let q = q_proj();
        let mut all = Vec::new();
        for kv in 0..SHAPE.n_head_kv {
            let (k, v) = group_kv(kv, t);
            all.extend(group_output(&SHAPE, &Prep::default(), &q, kv, t as u64, &k, &v).unwrap());
        }
        q8_1_row(&all)
    }

    #[test]
    fn an_honest_gqa_group_passes() {
        let t = 15;
        let x = honest_x(t);
        for kv in 0..SHAPE.n_head_kv {
            let (k, v) = group_kv(kv, t);
            let g = group_output(&SHAPE, &Prep::default(), &q_proj(), kv, t as u64, &k, &v).unwrap();
            assert!(check_group(&SHAPE, &x, q8::MMVQ, kv, &g), "kv {kv}");
        }
    }

    #[test]
    fn a_group_against_the_wrong_key_head_fails() {
        let t = 15;
        let x = honest_x(t);
        let (k, v) = group_kv(1, t);
        let g = group_output(&SHAPE, &Prep::default(), &q_proj(), 0, t as u64, &k, &v).unwrap();
        assert!(!check_group(&SHAPE, &x, q8::MMVQ, 0, &g));
    }

    #[test]
    fn a_zeroed_head_in_the_group_fails() {
        let t = 15;
        let x = honest_x(t);
        let mut forged = honest_x(t);
        let blocks_per_head = SHAPE.head_dim / 32;
        for b in 0..blocks_per_head {
            let o = b * 36;
            forged[o..o + 36].copy_from_slice(&q8_1_row(&vec![0.0; 32]));
        }
        let (k, v) = group_kv(0, t);
        let honest = group_output(&SHAPE, &Prep::default(), &q_proj(), 0, t as u64, &k, &v).unwrap();
        assert!(check_group(&SHAPE, &x, q8::MMVQ, 0, &honest));
        assert!(!check_group(&SHAPE, &forged, q8::MMVQ, 0, &honest));
    }

    #[test]
    fn f16_round_matches_half_precision() {
        assert_eq!(f16_round(1.0), 1.0);
        assert_eq!(f16_round(1.0 + libm::exp2(-11.0)), 1.0);
        assert_eq!(f16_round(1.0 + 3.0 * libm::exp2(-11.0)), 1.0 + libm::exp2(-9.0));
        assert_eq!(f16_round(-0.1), -0.0999755859375);
        assert_eq!(f16_round(libm::exp2(-24.0) * 0.75), libm::exp2(-24.0));
        assert_eq!(f16_round(65519.0), 65504.0);
        assert_eq!(f16_round(65520.0), f64::INFINITY);
    }

    #[test]
    fn every_kernel_rounding_accepts_the_group_only() {
        let t = 15;
        let x = honest_x(t);
        let mut forged = honest_x(t);
        forged[..36].copy_from_slice(&q8_1_row(&vec![0.0; 32]));
        let (k, v) = group_kv(0, t);
        let (k1, v1) = group_kv(1, t);
        for ar in ARITHS {
            let g = group_output_in(ar, &SHAPE, &Prep::default(), &q_proj(), 0, t as u64, &k, &v).unwrap();
            assert!(check_group(&SHAPE, &x, q8::MMVQ, 0, &g), "{ar:?}");
            assert!(!check_group(&SHAPE, &forged, q8::MMVQ, 0, &g), "{ar:?}");
            let wrong = group_output_in(ar, &SHAPE, &Prep::default(), &q_proj(), 0, t as u64, &k1, &v1).unwrap();
            assert!(!check_group(&SHAPE, &x, q8::MMVQ, 0, &wrong), "{ar:?}");
        }
    }

    #[test]
    fn a_query_rotated_at_another_position_fails() {
        let t = 15;
        let x = honest_x(t);
        let (k, v) = group_kv(0, t);
        let g = group_output(&SHAPE, &Prep::default(), &q_proj(), 0, t as u64 + 1, &k, &v).unwrap();
        assert!(!check_group(&SHAPE, &x, q8::MMVQ, 0, &g));
    }

    #[test]
    fn an_inconsistent_shape_is_rejected() {
        let bad = AttnShape { n_head: 7, ..SHAPE };
        assert!(bad.group().is_none());
        let (k, v) = group_kv(0, 3);
        assert!(group_output(&bad, &Prep::default(), &q_proj(), 0, 3, &k, &v).is_none());
        assert!(group_output(&SHAPE, &Prep::default(), &q_proj(), SHAPE.n_head_kv, 3, &k, &v).is_none());
    }

    #[test]
    fn frequency_divisors_scale_the_angle() {
        let x = vecf(6, DIM, 1.0);
        let (mut a, mut b, mut c) = (x.clone(), x.clone(), x.clone());
        rope_neox(&mut a, 11, DIM, 10_000.0, None);
        rope_neox(&mut b, 11, DIM, 10_000.0, Some(&[1.0; DIM / 2]));
        rope_neox(&mut c, 11, DIM, 10_000.0, Some(&[1e30; DIM / 2]));
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-12));
        assert!(x.iter().zip(&c).all(|(p, q)| (p - q).abs() < 1e-12));
    }

    #[test]
    fn rms_norm_gives_unit_rms_before_the_weight() {
        let mut x = vecf(8, DIM, 3.0);
        rms_norm(&mut x, None, 0.0);
        assert!((x.iter().map(|a| a * a).sum::<f64>() / DIM as f64 - 1.0).abs() < 1e-12);
        let w = [2.0f32; DIM];
        let mut y = vecf(8, DIM, 3.0);
        rms_norm(&mut y, Some(&w), 0.0);
        assert!(x.iter().zip(&y).all(|(a, b)| (2.0 * a - b).abs() < 1e-12));
    }

    #[test]
    fn a_shape_mismatch_is_rejected() {
        let (keys, mut values) = kv(5);
        values.pop();
        assert!(head_output(&vecf(3, DIM, 1.0), &keys, &values, 1.0).is_none());
    }
}
