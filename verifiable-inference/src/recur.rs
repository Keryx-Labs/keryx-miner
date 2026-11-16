//! Recomputation of a recurrent layer's delta rule over one segment of one head.

use crate::chain::{ChainCtx, block_outliers, chunk_leaf, f32_weights, f32s, shape_any};
use crate::draw;
use crate::manifest::{self, Chain, RECUR_GDA, RECUR_KDA};
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_verify};
use crate::opening::{LeafChunk, RecurOpening};
use crate::prefix::THRESHOLD;
use crate::seq::{self, Record, SeqLayout, SeqSpan};
use crate::verify::{Fail, PosShape, ResponseCtx, TierCtx, open_entry};

pub const RECUR_POINT: u32 = crate::chain::ATTN_POINT + 1;
/// State rows recomputed per point.
pub const RECUR_ROWS: u64 = 16;
/// Tolerance of a recomputed value against its record, relative to the value plus the scale of
/// its vector.
pub const RECUR_TOL: f64 = 1.0 / 64.0;
/// Largest `|v|` of a record.
pub const RECUR_V_MAX: f64 = 1024.0;
/// Largest deviation, in quantization steps, of a value of the output projection's input.
pub const RECUR_OUT_MAX: f64 = 2.0;
/// Values of one block allowed past `THRESHOLD`, up to `RECUR_OUT_MAX`.
pub const RECUR_OUT_SPARE: usize = 4;

/// Smallest step of a recorded f16 value.
const F16_MIN_STEP: f64 = 1.0 / 16_777_216.0;

fn rms(v: &[f32]) -> f64 {
    libm::sqrt(v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / v.len().max(1) as f64)
}

fn close(x: f64, want: f64, scale: f64) -> bool {
    x.is_finite() && (x - want).abs() <= RECUR_TOL * (want.abs() + scale) + F16_MIN_STEP
}

fn all_close(x: &[f64], want: &[f32]) -> bool {
    let s = rms(want);
    x.len() == want.len() && x.iter().zip(want).all(|(&a, &b)| close(a, b as f64, s))
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-x))
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { libm::log1p(libm::exp(x)) }
}

fn l2_norm(x: &mut [f64], eps: f64) {
    let r = 1.0 / libm::sqrt((x.iter().map(|a| a * a).sum::<f64>()).max(eps * eps));
    x.iter_mut().for_each(|a| *a *= r);
}

/// Values `[off, off + n)` of the f32 leaf `leaf` of `total` bytes, held in one opened chunk, and
/// the position root the chunk reaches.
fn chunk_values(shape: &PosShape, leaf: u64, c: &LeafChunk, total: u64, off: usize, n: usize) -> Result<(Hash, Vec<f32>), Fail> {
    let per = B3_CHUNK / 4;
    if off % per + n > per {
        return Err(Fail::Manifest);
    }
    let h = chunk_leaf(c, total, (off / per) as u64)?;
    let root = b3_fold(shape.n_leaves, leaf, &h, &c.leaf_path).ok_or(Fail::Trace)?;
    let v = f32s(&c.chunk);
    Ok((root, v.get(off % per..off % per + n).ok_or(Fail::Shape)?.to_vec()))
}

/// Recurrent point of a response whose trace commits sequence leaves.
pub fn verify_recur_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, op: &RecurOpening) -> Result<(), Fail> {
    let (nl, n_pos, n_recur) = (ctx.n_layers as u64, resp.n_pos as u64, ctx.n_recur as u64);
    if n_pos == 0 || n_recur == 0 || ctx.seq_heads == 0 {
        return Err(Fail::Unsupported);
    }
    if seq::trace_root(&op.pos_root, &op.seq_root) != resp.trace_root {
        return Err(Fail::Trace);
    }
    let d = draw::point(seed, RECUR_POINT);
    let slot = d[1] % n_recur;
    let layer = (op.layer.index as u64).checked_sub(ctx.layer_base as u64).filter(|&l| l < nl).ok_or(Fail::Manifest)?;
    let le = open_entry(tier, &op.layer, ctx.layer_base as u64 + layer)?;
    let (ll, r) = match le.chain {
        Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => (ll, ll.recur.ok_or(Fail::Shape)?),
        _ => return Err(Fail::Manifest),
    };
    let (hd, nv, nk, cl, iv) = (r.head_dim as usize, r.n_v_heads as u64, r.n_k_heads as u64, r.conv_len as usize, r.interval as u64);
    if r.slot as u64 != slot {
        return Err(Fail::Manifest);
    }
    if !matches!(r.kind, RECUR_GDA | RECUR_KDA) || !(1..=8).contains(&cl) || iv == 0 {
        return Err(Fail::Unsupported);
    }
    if hd == 0 || !hd.is_multiple_of(32) || !(hd as u64).is_multiple_of(RECUR_ROWS) || nk == 0 || !nv.is_multiple_of(nk) || nv > ctx.seq_heads as u64 || (r.kind == RECUR_KDA && nk != nv) {
        return Err(Fail::Manifest);
    }
    let kda = r.kind == RECUR_KDA;
    let h = d[2] % nv;
    let hk = h % nk;
    let s = d[0] % n_pos.div_ceil(iv);
    let (c, e) = (s * iv, ((s + 1) * iv).min(n_pos));
    let r0 = d[3] % (hd as u64 / RECUR_ROWS) * RECUR_ROWS;
    let p = c + (d[3] >> 32) % (e - c);

    let layout = SeqLayout { n_attn: ctx.n_attn as u64, n_recur, heads: ctx.seq_heads as u64 };
    let rl = seq::record_len(r.kind, hd) as u64;
    let rec_leaf = layout.record(slot, h).ok_or(Fail::Shape)?;
    let bytes = seq::open_span(&op.seq_root, &layout, rec_leaf, n_pos * rl, c * rl, e * rl, &op.records).ok_or(Fail::Trace)?;
    let recs = bytes.chunks_exact(rl as usize).map(|b| Record::decode(r.kind, hd, b)).collect::<Option<Vec<_>>>().ok_or(Fail::Shape)?;

    let ck_leaf = layout.checkpoint(slot, h).ok_or(Fail::Shape)?;
    let ck_len = seq::checkpoint_len(hd as u64, n_pos, iv);
    let rows_at = |m: u64, span: &SeqSpan| -> Result<Vec<Vec<f64>>, Fail> {
        let (lo, hi) = seq::checkpoint_rows(hd as u64, m, r0, RECUR_ROWS);
        let b = seq::open_span(&op.seq_root, &layout, ck_leaf, ck_len, lo, hi, span).ok_or(Fail::Trace)?;
        let rows: Vec<Vec<f64>> = f32s(b).chunks(hd).map(|row| row.iter().map(|&x| x as f64).collect()).collect();
        let row_max = RECUR_V_MAX * libm::sqrt(hd as f64);
        if rows.iter().any(|row| row.iter().any(|x| !x.is_finite()) || libm::sqrt(row.iter().map(|x| x * x).sum::<f64>()) > row_max) {
            return Err(Fail::Mismatch);
        }
        Ok(rows)
    };
    let mut state = if s == 0 {
        if op.start_rows != SeqSpan::default() {
            return Err(Fail::Shape);
        }
        vec![vec![0f64; hd]; RECUR_ROWS as usize]
    } else {
        rows_at(s, &op.start_rows)?
    };
    let out_scale = 1.0 / libm::sqrt(hd as f64);
    for rec in &recs {
        let g: Vec<f64> = if kda { rec.g.iter().map(|&x| libm::exp(x as f64)).collect() } else { vec![libm::exp(rec.g[0] as f64); hd] };
        let q_norm = libm::sqrt(rec.q.iter().map(|&x| x as f64 * x as f64).sum::<f64>());
        let k_norm = libm::sqrt(rec.k.iter().map(|&x| x as f64 * x as f64).sum::<f64>());
        let beta = rec.beta as f64;
        if q_norm > 1.0 + RECUR_TOL || k_norm > 1.0 + RECUR_TOL || !(0.0..=1.0 + RECUR_TOL).contains(&beta)
            || rec.g.iter().any(|&x| !(x <= 0.0))
            || rec.v.iter().any(|&x| !(x.abs() as f64 <= RECUR_V_MAX))
        {
            return Err(Fail::Mismatch);
        }
        for (ri, row) in state.iter_mut().enumerate() {
            let col = r0 as usize + ri;
            let kv: f64 = (0..hd).map(|i| g[i] * row[i] * rec.k[i] as f64).sum();
            let delta = (rec.v[col] as f64 - kv) * rec.beta as f64;
            for i in 0..hd {
                row[i] = g[i] * row[i] + rec.k[i] as f64 * delta;
            }
            let o = (0..hd).map(|i| row[i] * rec.q[i] as f64).sum::<f64>() * out_scale;
            let bound = out_scale * q_norm * libm::sqrt(row.iter().map(|x| x * x).sum::<f64>());
            if !close(o, rec.o[col] as f64, bound) {
                return Err(Fail::Mismatch);
            }
        }
    }
    if e - c == iv {
        let want = rows_at(s + 1, &op.end_rows)?;
        for (row, w) in state.iter().zip(&want) {
            let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
            if !all_close(row, &w32) {
                return Err(Fail::Mismatch);
            }
        }
    } else if op.end_rows != SeqSpan::default() {
        return Err(Fail::Shape);
    }

    // Anchor: the record at `p` against the committed projections and the layer output.
    let pshape = shape_any(tier, ctx);
    let n_taps = (p as usize + 1).min(cl);
    if op.taps.len() != n_taps || op.weights.len() != 6 {
        return Err(Fail::Shape);
    }
    let (hd64, nv64, nk64) = (hd as u64, nv, nk);
    let (offs, in_total) = if kda {
        ([h * hd64, h * hd64, h * hd64], 4 * nv64 * hd64)
    } else {
        ([hk * hd64, (nk64 + hk) * hd64, 2 * nk64 * hd64 + h * hd64], 4 * (2 * nk64 + nv64) * hd64)
    };
    let srcs = [r.q, r.k, r.v];
    let mut win = vec![vec![vec![0f64; hd]; cl]; 3];
    let mut root_p = None;
    for (j, tap) in op.taps.iter().enumerate() {
        let pos = p + 1 + j as u64 - n_taps as u64;
        let mut root = None;
        for (x, lc) in [&tap.q, &tap.k, &tap.v].into_iter().enumerate() {
            let (rt, vals) = chunk_values(&pshape, 2 * srcs[x] as u64 + 1, lc, in_total, offs[x] as usize, hd)?;
            if root.is_some_and(|r0: Hash| r0 != rt) {
                return Err(Fail::Trace);
            }
            root = Some(rt);
            win[x][cl - n_taps + j] = vals.iter().map(|&v| v as f64).collect();
        }
        let root = root.ok_or(Fail::Shape)?;
        if !b3_verify(&op.pos_root, n_pos, pos, &root, &tap.pos_path) {
            return Err(Fail::Trace);
        }
        if pos == p {
            root_p = Some(root);
        }
    }
    let root_p = root_p.ok_or(Fail::Shape)?;
    let conv_total = cl as u64 * if kda { nv64 * hd64 } else { (2 * nk64 + nv64) * hd64 };
    let convs = [r.conv_q, r.conv_k, r.conv_v];
    let mut conv_out = Vec::with_capacity(3);
    for x in 0..3 {
        let w = f32_weights(tier, &op.weights[x], convs[x], conv_total, offs[x] * cl as u64, hd64 * cl as u64)?;
        let y: Vec<f64> = (0..hd)
            .map(|i| {
                let a: f64 = (0..cl).map(|j| w[i * cl + j] as f64 * win[x][j][i]).sum();
                a * sigmoid(a)
            })
            .collect();
        conv_out.push(y);
    }
    let rec = &recs[(p - c) as usize];
    let eps = ctx.eps;
    let (mut q, mut k, v) = (conv_out[0].clone(), conv_out[1].clone(), conv_out[2].clone());
    l2_norm(&mut q, eps);
    l2_norm(&mut k, eps);
    if !all_close(&q, &rec.q) || !all_close(&k, &rec.k) || !all_close(&v, &rec.v) {
        return Err(Fail::Mismatch);
    }
    let same_root = |rt: Hash| if rt == root_p { Ok(()) } else { Err(Fail::Trace) };
    let (rt, beta) = chunk_values(&pshape, 2 * r.beta as u64 + 1, &op.beta, 4 * nv64, h as usize, 1)?;
    same_root(rt)?;
    if !close(sigmoid(beta[0] as f64), rec.beta as f64, 0.0) {
        return Err(Fail::Mismatch);
    }
    let gw = if kda { hd } else { 1 };
    let de = open_entry(tier, &op.decay_entry, r.decay as u64)?;
    if de.role != manifest::ROLE_PRODUCT {
        return Err(Fail::Manifest);
    }
    let dt_in = de.bias == manifest::NONE;
    let (rt, dy) = chunk_values(&pshape, 2 * r.decay as u64 + 1, &op.decay, 4 * nv64 * gw as u64, h as usize * gw, gw)?;
    same_root(rt)?;
    let dt = f32_weights(tier, &op.weights[3], r.dt, nv64 * gw as u64, h * gw as u64, gw as u64)?;
    let a = f32_weights(tier, &op.weights[4], r.a, nv64, h, 1)?;
    let g: Vec<f64> = (0..gw).map(|i| a[0] as f64 * softplus(dy[i] as f64 + if dt_in { dt[i] as f64 } else { 0.0 })).collect();
    if !all_close(&g, &rec.g) {
        return Err(Fail::Mismatch);
    }
    let (rt, gate) = chunk_values(&pshape, 2 * r.gate as u64 + 1, &op.gate, 4 * nv64 * hd64, h as usize * hd, hd)?;
    same_root(rt)?;
    let norm = f32_weights(tier, &op.weights[5], r.norm, hd64, 0, hd64)?;
    let oe = open_entry(tier, &op.out_entry, ll.proj as u64)?;
    if oe.k != nv64 * hd64 {
        return Err(Fail::Manifest);
    }
    let lay = crate::verify::layout_for(crate::verify::expected_ubatch(p, resp.n_prompt), oe.wtype);
    if op.out.bytes.len() as u64 != crate::verify::x_len(oe.k, lay) {
        return Err(Fail::Shape);
    }
    if !b3_verify(&root_p, pshape.n_leaves, 2 * ll.proj as u64, blake3::hash(&op.out.bytes).as_bytes(), &op.out.leaf_path) {
        return Err(Fail::Trace);
    }
    let ms = rec.o.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / hd as f64;
    let rn = 1.0 / libm::sqrt(ms + eps);
    let y: Vec<f64> = (0..hd)
        .map(|i| {
            let z = gate[i] as f64;
            let act = if kda { sigmoid(z) } else { z * sigmoid(z) };
            rec.o[i] as f64 * rn * norm[i] as f64 * act
        })
        .collect();
    let dev = block_outliers(&op.out.bytes, lay, (nv64 * hd64) as usize, h as usize * hd / 32, &y, THRESHOLD);
    let ok = dev.is_some_and(|(dv, spare)| dv <= RECUR_OUT_MAX && spare <= RECUR_OUT_SPARE);
    ok.then_some(()).ok_or(Fail::Mismatch)
}
