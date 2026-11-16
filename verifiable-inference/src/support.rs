//! Structural support of a manifest by the attention and recurrent verifiers: the checks that
//! depend only on the manifest and the draw, never on an opening.

use crate::attn::Prep;
use crate::attn_blocks::{BlockTree, block_params, partial_len};
use crate::chain::{ChainCtx, gqa_shape, seq_layout};
use crate::manifest::{self, AttnLinks, Chain, Entry, LayerLinks, RECUR_GDA, RECUR_KDA, RecurLinks};
use crate::merkle::B3_CHUNK;
use crate::recur::RECUR_ROWS;
use crate::seq::SeqLayout;

/// Whether every attention and recurrent point drawable on `entries`, up to `n_pos_max`
/// positions, has a shape the verifiers accept; the first layer that has none otherwise.
pub fn check_manifest(entries: &[Entry], ctx: &ChainCtx, n_pos_max: u64) -> Result<(), String> {
    let entry = |i: u32| entries.get(i as usize).ok_or_else(|| format!("entry {i} missing"));
    let sl = seq_layout(ctx);
    let (mut n_attn, mut n_recur) = (0, 0);
    for l in 0..ctx.n_layers {
        let le = entry(ctx.layer_base + l)?;
        let ll = match le.chain {
            Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer == l => ll,
            _ => return Err(format!("layer {l}: no layer entry")),
        };
        if let Some(a) = ll.attn {
            n_attn += 1;
            attn_layer(&entry, ctx, &sl, &ll, &a, n_pos_max).map_err(|e| format!("layer {l} attention: {e}"))?;
        }
        if let Some(r) = ll.recur {
            n_recur += 1;
            recur_layer(&entry, ctx, &sl, &ll, &r).map_err(|e| format!("layer {l} recurrence: {e}"))?;
        }
    }
    if n_attn != ctx.n_attn || n_recur != ctx.n_recur {
        return Err(format!("{n_attn} attention and {n_recur} recurrent layers for {} and {}", ctx.n_attn, ctx.n_recur));
    }
    Ok(())
}

fn norm_entry<'a>(entry: &impl Fn(u32) -> Result<&'a Entry, String>, i: u32, k: u64) -> Result<Option<Vec<f32>>, String> {
    if i == manifest::NONE {
        return Ok(None);
    }
    let e = entry(i)?;
    if e.role != manifest::ROLE_NORM || e.wtype != 0 || e.k != k {
        return Err(format!("entry {i} is not a {k}-wide norm"));
    }
    Ok(Some(vec![1.0; k as usize]))
}

fn attn_layer<'a>(
    entry: &impl Fn(u32) -> Result<&'a Entry, String>,
    ctx: &ChainCtx,
    sl: &SeqLayout,
    ll: &LayerLinks,
    a: &AttnLinks,
    n_pos_max: u64,
) -> Result<(), String> {
    if ctx.n_attn == 0 || ctx.n_attn > ctx.n_layers || ctx.seq_heads == 0 || a.slot >= ctx.n_attn {
        return Err("context or slot".into());
    }
    if a.softcap != 0 || a.v_norm > 1 || a.gated > 1 {
        return Err("softcap, value norm or gate".into());
    }
    let hd = a.head_dim as usize;
    if hd == 0 || !hd.is_multiple_of(32) {
        return Err("head dim".into());
    }
    let mla = a.mla_rank != 0;
    let (row, dv) = if mla { (hd / 32 * 34, a.mla_rank as usize) } else { (2 * (hd / 32 * 34), hd) };
    let (c, b) = block_params(row, partial_len(dv)).ok_or("block parameters")?;
    for t in [0, 1, n_pos_max / 2, n_pos_max - 1] {
        let start = if !mla && a.swa_window > 0 { (t + 1).saturating_sub(a.swa_window as u64) } else { 0 };
        BlockTree::new(start, t + 1, c, b).ok_or(format!("block tree at {t}"))?;
    }
    let none = manifest::NONE;
    let qe = entry(a.q)?;
    if mla {
        let (nh, r) = (a.n_head as usize, a.mla_rank as usize);
        let rope = hd.checked_sub(r).filter(|_| r.is_multiple_of(32) && nh > 0).ok_or("latent rank")?;
        if a.n_head_kv != 1 || a.n_rot != 0 || a.swa_window != 0 || a.gated != 0 || a.v_norm != 0 || a.q_norm != none || a.rope_freqs != none {
            return Err("latent attention shape".into());
        }
        if a.v != none || a.k_norm == none || a.absorb == none {
            return Err("latent attention links".into());
        }
        if !qe.m.is_multiple_of(nh as u64) || (qe.m / nh as u64).checked_sub(rope as u64).is_none_or(|n| n == 0) {
            return Err("query width".into());
        }
        let oe = entry(if a.out == none { ll.proj } else { a.out })?;
        if oe.k as usize != r || oe.n_exp as usize != nh || oe.role != manifest::ROLE_HEADS {
            return Err("output heads".into());
        }
        sl.kcache(a.slot as u64, 0).zip(sl.lse(a.slot as u64, 0)).ok_or("key cache or mass leaf")?;
    } else {
        if a.absorb != none || a.out != none {
            return Err("grouped-query links".into());
        }
        let shape = gqa_shape(a);
        shape.group().ok_or("head grouping")?;
        if qe.m as usize != shape.n_head * hd * (1 + a.gated as usize) {
            return Err("query width".into());
        }
        if entry(ll.proj)?.k as usize != shape.n_head * hd {
            return Err("output width".into());
        }
        let (qn, kn, ff) = (norm_entry(entry, a.q_norm, hd as u64)?, norm_entry(entry, a.k_norm, hd as u64)?, norm_entry(entry, a.rope_freqs, (a.n_rot / 2) as u64)?);
        let prep = Prep { q_norm: qn.as_deref(), k_norm: kn.as_deref(), v_norm: a.v_norm == 1, freqs: ff.as_deref(), eps: ctx.eps };
        if !prep.fits(&shape) {
            return Err("norm widths".into());
        }
        for kv in 0..a.n_head_kv as u64 {
            sl.kcache(a.slot as u64, kv).zip(sl.lse(a.slot as u64, kv)).ok_or("cache or mass leaf")?;
        }
    }
    Ok(())
}

fn recur_layer<'a>(entry: &impl Fn(u32) -> Result<&'a Entry, String>, ctx: &ChainCtx, sl: &SeqLayout, ll: &LayerLinks, r: &RecurLinks) -> Result<(), String> {
    if ctx.n_recur == 0 || ctx.seq_heads == 0 || r.slot >= ctx.n_recur {
        return Err("context or slot".into());
    }
    let (hd, nv, nk, cl) = (r.head_dim as usize, r.n_v_heads as usize, r.n_k_heads as usize, r.conv_len as u64);
    if !matches!(r.kind, RECUR_GDA | RECUR_KDA) || !(1..=8).contains(&cl) || r.interval == 0 {
        return Err("kind, convolution or interval".into());
    }
    let kda = r.kind == RECUR_KDA;
    if hd == 0 || !hd.is_multiple_of(32) || !(hd as u64).is_multiple_of(RECUR_ROWS) || nk == 0 || !nv.is_multiple_of(nk) || nv > ctx.seq_heads as usize || (kda && nk != nv) {
        return Err("heads".into());
    }
    if entry(r.decay)?.role != manifest::ROLE_PRODUCT {
        return Err("decay product".into());
    }
    if entry(ll.proj)?.k != (nv * hd) as u64 {
        return Err("output width".into());
    }
    let gw = if kda { hd } else { 1 };
    let conv = cl * if kda { nv * hd } else { (2 * nk + nv) * hd } as u64;
    for (i, k) in [(r.conv_q, conv), (r.conv_k, conv), (r.conv_v, conv), (r.dt, (nv * gw) as u64), (r.a, nv as u64), (r.norm, hd as u64)] {
        norm_entry(entry, i, k)?;
    }
    let per = B3_CHUNK / 4;
    let fits = |off: usize, n: usize| off % per + n <= per;
    for h in 0..nv {
        let hk = h % nk;
        let taps = if kda { [h * hd; 3] } else { [hk * hd, (nk + hk) * hd, 2 * nk * hd + h * hd] };
        if !taps.iter().all(|&o| fits(o, hd)) || !fits(h, 1) || !fits(h * gw, gw) || !fits(h * hd, hd) {
            return Err(format!("head {h} straddles a chunk"));
        }
        sl.record(r.slot as u64, h as u64).zip(sl.checkpoint(r.slot as u64, h as u64)).ok_or(format!("head {h} sequence leaves"))?;
    }
    Ok(())
}
