//! Verification of opened points.

use crate::dot::{self, Recomputed};
use crate::draw;
use crate::manifest::{self, Entry};
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_leaf_from_chunk, b3_verify, rt_verify_range};
use crate::opening::{ChunkRange, EntryOpening, ExpertIds, Point};
use crate::{q8, quant};

pub const EPS: f64 = 1e-5;
pub const EPS_GEGLU: f64 = 3e-5;
const MMVQ_MAX_BATCH: u32 = 8;
/// Prompt positions per batch of the traced run.
pub const UBATCH: u32 = 512;
const MATRIX_ROW_PADDING: u64 = 512;

/// Consensus data of a tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierCtx {
    pub r_t: Hash,
    pub n_chunks: u64,
    pub m_t: Hash,
    pub n_entries: u64,
    pub n_products: u64,
}

/// Committed data of the response.
pub struct ResponseCtx {
    pub trace_root: Hash,
    pub n_prompt: u32,
    pub n_pos: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fail {
    Shape,
    Manifest,
    Weights,
    Trace,
    Unsupported,
    Mismatch,
}

/// Leaf layout of a traced position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PosShape {
    pub n_leaves: u64,
    pub meta_index: u64,
    /// Token in the meta leaf (stage 2).
    pub tok: Option<u32>,
    /// Active experts per position of an expert product; 0 = none.
    pub n_used: u32,
}

impl PosShape {
    pub fn stage1(n_products: u64) -> Self {
        Self { n_leaves: 2 * n_products + 1, meta_index: 2 * n_products, tok: None, n_used: 0 }
    }

    pub(crate) fn meta_hash(&self, t: u64, n_ubatch: u32) -> Hash {
        match self.tok {
            Some(tok) => *blake3::hash(&crate::chain::meta(t as u32, n_ubatch, tok)).as_bytes(),
            None => {
                let mut m = [0u8; 8];
                m[..4].copy_from_slice(&(t as u32).to_le_bytes());
                m[4..].copy_from_slice(&n_ubatch.to_le_bytes());
                *blake3::hash(&m).as_bytes()
            }
        }
    }
}

/// What a verified point leaves for further checks at its position.
#[derive(Debug, Clone)]
pub(crate) struct Opened {
    pub pos_root: Hash,
    pub input: Entry,
    pub layout: u8,
    pub slot: u64,
    pub expert: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Report {
    pub t: u64,
    pub rank: u64,
    pub row: u64,
    pub y: f64,
    pub y_hat: f64,
    pub tol: f64,
}

pub(crate) fn open_entry(tier: &TierCtx, e: &EntryOpening, expected_index: u64) -> Result<Entry, Fail> {
    if e.index as u64 != expected_index || !manifest::verify_entry(&tier.m_t, tier.n_entries, expected_index, &e.bytes, &e.path) {
        return Err(Fail::Manifest);
    }
    let (entry, len) = Entry::parse(&e.bytes).ok_or(Fail::Manifest)?;
    (len == e.bytes.len()).then_some(entry).ok_or(Fail::Manifest)
}

/// Tensor bytes proven against the model root.
pub(crate) fn open_bytes(tier: &TierCtx, entry: &Entry, r: &ChunkRange, offset: u64, len: u64) -> Result<Vec<u8>, Fail> {
    let end = offset + len;
    if len == 0 || end > entry.n_bytes / 32 * 32 {
        return Err(Fail::Shape);
    }
    let first = entry.chunk_base + offset / 32;
    let last = entry.chunk_base + (end - 1) / 32;
    if r.first_chunk != first || r.chunks.len() as u64 != (last - first + 1) * 32 {
        return Err(Fail::Shape);
    }
    let leaves: Vec<Hash> = r.chunks.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    if !rt_verify_range(&tier.r_t, tier.n_chunks, first, &leaves, &r.proof) {
        return Err(Fail::Weights);
    }
    let o = (offset % 32) as usize;
    Ok(r.chunks[o..o + len as usize].to_vec())
}

pub(crate) fn open_expert_row(tier: &TierCtx, entry: &Entry, r: &ChunkRange, exp: u64, row: u64) -> Result<Vec<u8>, Fail> {
    if row >= entry.m || exp >= entry.n_exp as u64 {
        return Err(Fail::Shape);
    }
    let rb = entry.row_bytes as u64;
    open_bytes(tier, entry, r, (exp * entry.m + row) * rb, rb)
}

pub(crate) fn open_row(tier: &TierCtx, entry: &Entry, r: &ChunkRange, row: u64) -> Result<Vec<u8>, Fail> {
    if row >= entry.m || entry.n_exp != 1 {
        return Err(Fail::Shape);
    }
    let rb = entry.row_bytes as u64;
    open_bytes(tier, entry, r, row * rb, rb)
}

/// Batch size the traced run computes position `t` with.
pub fn expected_ubatch(t: u64, n_prompt: u32) -> u32 {
    if t >= n_prompt as u64 {
        return 1;
    }
    let start = t / UBATCH as u64 * UBATCH as u64;
    UBATCH.min(n_prompt - start as u32)
}

/// Q8_1 layout of the input of a product of `role` and weight type `wtype` in a batch of `n_ubatch`.
pub fn input_layout(role: u8, n_ubatch: u32, wtype: u32) -> u8 {
    if role == manifest::ROLE_OUTPUT { q8::MMVQ } else { layout_for(n_ubatch, wtype) }
}

pub(crate) fn layout_for(n_ubatch: u32, wtype: u32) -> u8 {
    if n_ubatch <= MMVQ_MAX_BATCH {
        q8::MMVQ
    } else if matches!(wtype, 2 | 3 | 7 | quant::Q4_K | quant::Q5_K) {
        q8::MMQ_DS4
    } else {
        q8::MMQ_D4
    }
}

pub fn x_len(k: u64, layout: u8) -> u64 {
    let kp = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if layout == q8::MMVQ { kp / 32 * 36 } else { kp / 128 * 144 }
}

fn glu(op: u8, g: f64) -> Option<(f64, f64)> {
    match op {
        manifest::GLU_SWIGLU => {
            let s = 1.0 / (1.0 + libm::exp(-g));
            Some((g * s, s * (1.0 + g * (1.0 - s))))
        }
        manifest::GLU_GEGLU => {
            let a = (2.0 / std::f64::consts::PI).sqrt();
            let t = libm::tanh(a * (g + 0.044715 * g * g * g));
            Some((0.5 * g * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * a * (1.0 + 3.0 * 0.044715 * g * g)))
        }
        _ => None,
    }
}

/// Verify point `i` of an opening.
pub fn verify_point(tier: &TierCtx, resp: &ResponseCtx, seed: &Hash, i: u32, p: &Point) -> Result<Report, Fail> {
    verify_point_at(tier, resp, seed, i, p, &PosShape::stage1(tier.n_products), None).map(|(r, _)| r)
}

pub(crate) fn verify_point_at(
    tier: &TierCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    i: u32,
    p: &Point,
    shape: &PosShape,
    ids: Option<&ExpertIds>,
) -> Result<(Report, Opened), Fail> {
    let n_traced = resp.n_pos as u64;
    if n_traced == 0 || resp.n_prompt == 0 || resp.n_prompt > resp.n_pos {
        return Err(Fail::Shape);
    }
    if tier.n_products == 0 || tier.n_products > tier.n_entries {
        return Err(Fail::Shape);
    }
    let d = draw::point(seed, i);
    let rank = d[1] % tier.n_products;

    let first = open_entry(tier, &p.entry, rank)?;
    let output_only = first.role == manifest::ROLE_OUTPUT;
    let heads = first.role == manifest::ROLE_HEADS;
    if first.role != manifest::ROLE_PRODUCT && !output_only && !heads {
        return Err(Fail::Manifest);
    }
    let t = if output_only {
        let first_pos = (resp.n_prompt as u64).checked_sub(1).ok_or(Fail::Shape)?;
        first_pos + d[0] % n_traced.checked_sub(first_pos).filter(|&n| n > 0).ok_or(Fail::Shape)?
    } else {
        d[0] % n_traced
    };
    let (gate, up) = match first.glu_form {
        manifest::FORM_NONE | manifest::FORM_SPLIT => {
            if p.partner.is_some() {
                return Err(Fail::Shape);
            }
            ((first.clone(), rank), None)
        }
        manifest::FORM_PAIR_GATE | manifest::FORM_PAIR_UP => {
            let po = p.partner.as_ref().ok_or(Fail::Shape)?;
            let other = open_entry(tier, po, first.partner as u64)?;
            if other.partner as u64 != rank || other.k != first.k || other.m != first.m {
                return Err(Fail::Manifest);
            }
            if first.glu_form == manifest::FORM_PAIR_GATE {
                ((first.clone(), rank), Some(other))
            } else {
                ((other, first.partner as u64), Some(first.clone()))
            }
        }
        _ => return Err(Fail::Unsupported),
    };
    let (g_entry, g_rank) = gate;
    let experts = g_entry.n_exp > 1 && !heads;
    if experts != ids.is_some() || (experts && shape.n_used == 0) || (heads && (g_entry.n_exp < 2 || up.is_some())) {
        return Err(Fail::Shape);
    }
    if up.as_ref().is_some_and(|u| u.n_exp != g_entry.n_exp) {
        return Err(Fail::Manifest);
    }
    let n_used = if experts {
        shape.n_used as u64
    } else if heads {
        g_entry.n_exp as u64
    } else {
        1
    };
    let slot = if n_used > 1 { (d[3] >> 32) % n_used } else { 0 };
    let split = g_entry.glu_form == manifest::FORM_SPLIT;
    let m_out = if split { g_entry.m / 2 } else { g_entry.m };
    if m_out == 0 {
        return Err(Fail::Shape);
    }
    let row = d[2] % m_out;

    let n_pos_leaves = shape.n_leaves;
    let layout = input_layout(first.role, p.n_ubatch, g_entry.wtype);
    let x_one = x_len(g_entry.k, layout);
    let x = match p.x.len() as u64 {
        n if n == x_one => &p.x[..],
        n if n_used > 1 && n == n_used * x_one => &p.x[(slot * x_one) as usize..((slot + 1) * x_one) as usize],
        _ => return Err(Fail::Shape),
    };
    let pos_root = b3_fold(n_pos_leaves, 2 * g_rank, blake3::hash(&p.x).as_bytes(), &p.x_path).ok_or(Fail::Trace)?;
    if !b3_verify(&resp.trace_root, n_traced, t, &pos_root, &p.pos_path) {
        return Err(Fail::Trace);
    }
    if p.n_ubatch != expected_ubatch(t, resp.n_prompt) {
        return Err(Fail::Mismatch);
    }
    if !b3_verify(&pos_root, n_pos_leaves, shape.meta_index, &shape.meta_hash(t, p.n_ubatch), &p.meta_path) {
        return Err(Fail::Trace);
    }
    let body = n_used * m_out * 4;
    if experts && body % B3_CHUNK as u64 != 0 {
        return Err(Fail::Shape);
    }
    let y_off = (slot * m_out + row) * 4;
    let y_total = body + if experts { 4 * n_used } else { 0 };
    if p.y_total_len as u64 != y_total || p.y_chunk_index as u64 != y_off / B3_CHUNK as u64 {
        return Err(Fail::Shape);
    }
    let y_leaf = b3_leaf_from_chunk(p.y_total_len as u64, p.y_chunk_index as u64, &p.y_chunk, &p.y_chunk_path).ok_or(Fail::Trace)?;
    if !b3_verify(&pos_root, n_pos_leaves, 2 * g_rank + 1, &y_leaf, &p.y_path) {
        return Err(Fail::Trace);
    }
    let o = (y_off % B3_CHUNK as u64) as usize;
    let y = f32::from_le_bytes(p.y_chunk[o..o + 4].try_into().unwrap()) as f64;
    let exp = match ids {
        Some(e) => {
            let last = body / B3_CHUNK as u64;
            if b3_leaf_from_chunk(y_total, last, &e.chunk, &e.chunk_path) != Some(y_leaf) {
                return Err(Fail::Trace);
            }
            let s = 4 * slot as usize;
            let id = u32::from_le_bytes(e.chunk.get(s..s + 4).ok_or(Fail::Shape)?.try_into().unwrap()) as u64;
            if id >= g_entry.n_exp as u64 {
                return Err(Fail::Mismatch);
            }
            id
        }
        None => slot,
    };

    let k = g_entry.k as usize;
    if !q8::sums_consistent(x, layout, k) {
        return Err(Fail::Mismatch);
    }
    let dot_row = |entry: &Entry, r: &ChunkRange, rr: u64| -> Result<Recomputed, Fail> {
        let bytes = open_expert_row(tier, entry, r, exp, rr)?;
        dot::canonical_dot(entry.wtype, &bytes, k, x, layout).ok_or(Fail::Unsupported)
    };

    let (y_hat, tol) = if g_entry.glu_form == manifest::FORM_NONE {
        if p.rows.len() != 1 {
            return Err(Fail::Shape);
        }
        let r = dot_row(&g_entry, &p.rows[0], row)?;
        let b = match (g_entry.bias, &p.bias) {
            (manifest::NONE, None) => 0.0,
            (bi, Some((be, br))) if bi != manifest::NONE => {
                let bias_entry = open_entry(tier, be, bi as u64)?;
                if bias_entry.role != manifest::ROLE_BIAS || bias_entry.wtype != 0 || bias_entry.k != g_entry.m {
                    return Err(Fail::Manifest);
                }
                let v = open_bytes(tier, &bias_entry, br, row * 4, 4)?;
                f32::from_le_bytes(v.try_into().unwrap()) as f64
            }
            _ => return Err(Fail::Shape),
        };
        (r.y_hat + b, EPS * (r.scale + b.abs()))
    } else {
        if p.rows.len() != 2 || p.bias.is_some() {
            return Err(Fail::Shape);
        }
        let (rg, ru) = if split {
            (dot_row(&g_entry, &p.rows[0], row)?, dot_row(&g_entry, &p.rows[1], row + m_out)?)
        } else {
            (dot_row(&g_entry, &p.rows[0], row)?, dot_row(up.as_ref().unwrap(), &p.rows[1], row)?)
        };
        let (f, df) = glu(g_entry.glu_op, rg.y_hat).ok_or(Fail::Unsupported)?;
        let h = f * ru.y_hat;
        let eps_glu = if g_entry.glu_op == manifest::GLU_GEGLU { EPS_GEGLU } else { EPS };
        (h, EPS * ((df * ru.y_hat).abs() * rg.scale + f.abs() * ru.scale) + eps_glu * (rg.y_hat * ru.y_hat).abs())
    };
    if !(y.is_finite() && y_hat.is_finite() && tol.is_finite()) || !(y - y_hat).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
        return Err(Fail::Mismatch);
    }
    Ok((Report { t, rank, row, y, y_hat, tol }, Opened { pos_root, input: g_entry, layout, slot, expert: exp }))
}
