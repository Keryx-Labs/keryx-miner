//! Openings of the deepseek41 pieces.

use crate::llama_engine::SeqCapture;
use crate::pom::WeightIndex;
use crate::trace::{bytes_range, leaf_bytes, Captures, SeqTrace, TierManifest, TraceRecord};
use crate::trace_seq::{find_layer, span, SpanPlan};
use keryx_verifiable_inference::chain;
use keryx_verifiable_inference::draw;
use keryx_verifiable_inference::dsv41::COMP_POINT;
use keryx_verifiable_inference::manifest::{self, CompLinks};
use keryx_verifiable_inference::merkle::Hash;
use keryx_verifiable_inference::opening::{CompOpeningPoint, SelMember};

/// What the compressed row point of a seed opens: the source layer, and the block when the
/// response completes one.
#[derive(Clone, Copy, Debug)]
pub struct CompDraw {
    pub layer: u64,
    pub c: CompLinks,
    pub k: Option<u64>,
    pub n_blocks: u64,
}

/// The compressed row point of `seed`, drawn as the node draws it.
pub fn draw_comp(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<CompDraw> {
    let ctx = m.chain?;
    if ctx.n_comp == 0 {
        return None;
    }
    let d = draw::point(seed, COMP_POINT);
    let slot = d[1] % ctx.n_comp as u64;
    let (layer, _, c) = find_layer(m, |ll| ll.comp.filter(|c| c.slot as u64 == slot && c.src == ll.layer && c.ratio > 0))?;
    let n_blocks = rec.n_pos as u64 / c.ratio as u64;
    Some(CompDraw { layer, c, k: (n_blocks > 0).then(|| d[0] % n_blocks), n_blocks })
}

impl CompDraw {
    fn members(&self) -> impl Iterator<Item = u64> + '_ {
        let r = self.c.ratio as u64;
        self.k.into_iter().flat_map(move |k| k * r..(k + 1) * r)
    }

    /// Captures of the block's members: the compressor value and gate outputs.
    pub fn captures(&self) -> Vec<(u32, u32)> {
        let c = &self.c;
        self.members()
            .flat_map(|p| {
                let gate = (c.gate != manifest::NONE).then_some((p as u32, 2 * c.gate + 1));
                std::iter::once((p as u32, 2 * c.kv + 1)).chain(gate)
            })
            .collect()
    }

    fn row_bytes(&self) -> u64 {
        self.c.row_len as u64 / 32 * 34
    }

    /// The compressed row span the point opens.
    pub fn spans(&self, m: &TierManifest) -> Option<Vec<SpanPlan>> {
        let Some(k) = self.k else { return Some(Vec::new()) };
        let layout = chain::seq_layout(&m.chain?);
        let rb = self.row_bytes();
        Some(vec![SpanPlan::new(layout.comp(self.c.slot as u64)?, self.n_blocks * rb, k * rb, (k + 1) * rb)?])
    }
}

/// The compressed row opening of `d` from its captures and span.
#[allow(clippy::too_many_arguments)]
pub fn comp_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    index: &WeightIndex,
    d: &CompDraw,
    caps: &Captures,
    spans: &[SeqCapture],
    response_txid: Hash,
) -> Result<CompOpeningPoint, String> {
    let base = m.chain.ok_or("no chain context")?.layer_base as u64;
    let root = CompOpeningPoint { response_txid, pos_root: rec.trace_root, seq_root: seq.root, layer: m.entry_opening(base + d.layer), ..Default::default() };
    if d.k.is_none() {
        return Ok(root);
    }
    let c = &d.c;
    let gated = c.gate != manifest::NONE;
    let cap = |pos: u64, leaf: u32| caps.get(&(pos as u32, leaf)).ok_or_else(|| format!("no capture of leaf {leaf} at {pos}"));
    let dim = m.entry(c.kv as u64).m;
    let whole = |e: u32, len: u64| bytes_range(m, index, e as u64, 0, len).ok_or_else(|| format!("no weights of entry {e}"));
    let mut weights = vec![whole(c.norm, 4 * dim)?];
    if c.ik != manifest::NONE {
        let me = m.entry(c.ik as u64);
        weights.push(whole(c.ik, me.n_bytes)?);
        weights.push(whole(c.k_norm, 4 * me.m)?);
    }
    let members = d
        .members()
        .map(|p| {
            let g = if gated { leaf_bytes(rec, p, 2 * c.gate + 1, cap(p, 2 * c.gate + 1)?) } else { Default::default() };
            Ok(SelMember { pos_path: rec.pos_path(p), k: leaf_bytes(rec, p, 2 * c.kv + 1, cap(p, 2 * c.kv + 1)?), g })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let plans = d.spans(m).ok_or("no compressed row span")?;
    let row = span(&plans[0], spans.first().ok_or("no compressed row capture")?, seq)?;
    let mut entries = vec![m.entry_opening(c.kv as u64)];
    if gated {
        entries.push(m.entry_opening(c.gate as u64));
    }
    Ok(CompOpeningPoint { entries, weights, row, members, ..root })
}

/// What the selection point of a seed opens: the layer keeping (or picking) the drawn selection
/// leaf, and the position when the response is long enough for something to be left out.
#[derive(Clone, Copy, Debug)]
pub struct CompSelDraw {
    pub layer: u64,
    pub c: CompLinks,
    pub picks: bool,
    pub t: Option<u64>,
    pub sel_leaf: u32,
    pub cand_leaf: Option<u32>,
}

/// The two-level selection point of `seed`, drawn as the node draws it.
pub fn draw_comp_sel(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<CompSelDraw> {
    use keryx_verifiable_inference::dsa::SEL_POINT;
    let ctx = m.chain?;
    if ctx.n_sel == 0 || ctx.n_comp == 0 {
        return None;
    }
    let d = draw::point(seed, SEL_POINT);
    let s = d[1] % ctx.n_sel as u64;
    let (layer, _, (c, picks)) = find_layer(m, |ll| {
        let c = ll.comp?;
        let picks = c.pick == 1 && c.cand as u64 == s;
        let keeps = c.sel as u64 == s && c.topk_src == ll.layer;
        (picks != keeps && c.ratio > 0 && c.iq != manifest::NONE).then_some((c, picks))
    })?;
    let (ratio, n_pos) = (c.ratio as u64, rec.n_pos as u64);
    let first = if picks { (c.cand_n as u64 * c.cand_bs as u64 + 1) * ratio - 1 } else { (c.n_top as u64 + 1) * ratio - 1 };
    let t = (first < n_pos).then(|| first + d[0] % (n_pos - first));
    let leaf = |i: u32| chain::sel_leaf(m.n_products as u64, ctx.n_layers as u64, chain::n_router(&ctx), i as u64) as u32;
    let cand_leaf = (!picks && c.cand != manifest::NONE && c.pick != 1).then(|| leaf(c.cand));
    Some(CompSelDraw { layer, c, picks, t, sel_leaf: leaf(s as u32), cand_leaf })
}

impl CompSelDraw {
    /// Captures at the drawn position: the selection, the indexer query and head weights, and the
    /// candidate blocks the selection is restricted to.
    pub fn captures(&self) -> Vec<(u32, u32)> {
        let Some(t) = self.t else { return Vec::new() };
        let t = t as u32;
        let c = &self.c;
        let mut v = vec![(t, self.sel_leaf), (t, 2 * c.iq + 1), (t, 2 * c.iw + 1)];
        v.extend(self.cand_leaf.map(|l| (t, l)));
        v
    }

    /// The compressed rows (or blocks of rows) `seed` draws from the selection `sel`.
    pub fn spans(&self, m: &TierManifest, n_pos: u64, seed: &Hash, sel: &[u8]) -> Option<Vec<SpanPlan>> {
        use keryx_verifiable_inference::dsa::drawn_pools;
        use keryx_verifiable_inference::dsv41::parse_kept;
        let t = self.t?;
        let c = &self.c;
        let (ratio, bs) = (c.ratio as u64, c.cand_bs as u64);
        let n_vis = (t + 1) / ratio;
        let rb = c.row_len as u64 / 32 * 34;
        let total = n_pos / ratio * rb;
        let leaf = chain::seq_layout(&m.chain?).comp(c.slot as u64)?;
        if self.picks {
            let nbv = n_vis.div_ceil(bs);
            let (_, blocks) = parse_kept(sel, c.cand_n as usize, nbv)?;
            drawn_pools(seed, &blocks, nbv).iter().map(|&b| SpanPlan::new(leaf, total, b * bs * rb, ((b + 1) * bs).min(n_vis) * rb)).collect()
        } else {
            let (_, ids) = parse_kept(sel, c.n_top as usize, n_vis)?;
            drawn_pools(seed, &ids, n_vis).iter().map(|&j| SpanPlan::new(leaf, total, j * rb, (j + 1) * rb)).collect()
        }
    }
}

/// The two-level selection opening of `d` from its captures and spans.
#[allow(clippy::too_many_arguments)]
pub fn comp_sel_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    d: &CompSelDraw,
    seed: &Hash,
    caps: &Captures,
    spans: &[SeqCapture],
    response_txid: Hash,
) -> Result<keryx_verifiable_inference::opening::CompSelOpening, String> {
    use keryx_verifiable_inference::opening::CompSelOpening;
    let base = m.chain.ok_or("no chain context")?.layer_base as u64;
    let root = CompSelOpening { response_txid, pos_root: rec.trace_root, seq_root: seq.root, layer: m.entry_opening(base + d.layer), ..Default::default() };
    let Some(t) = d.t else { return Ok(root) };
    let c = &d.c;
    let cap = |leaf: u32| caps.get(&(t as u32, leaf)).ok_or_else(|| format!("no capture of leaf {leaf} at {t}"));
    let sel = cap(d.sel_leaf)?;
    let plans = d.spans(m, rec.n_pos as u64, seed, sel).ok_or("selection out of shape")?;
    if plans.len() != spans.len() {
        return Err(format!("{} row spans captured for {}", spans.len(), plans.len()));
    }
    let rows = plans.iter().zip(spans).map(|(p, s)| span(p, s, seq)).collect::<Result<Vec<_>, String>>()?;
    Ok(CompSelOpening {
        entries: vec![m.entry_opening(c.iq as u64), m.entry_opening(c.iw as u64)],
        pos_path: rec.pos_path(t),
        sel: leaf_bytes(rec, t, d.sel_leaf, sel),
        iq: leaf_bytes(rec, t, 2 * c.iq + 1, cap(2 * c.iq + 1)?),
        iw: leaf_bytes(rec, t, 2 * c.iw + 1, cap(2 * c.iw + 1)?),
        cand: match d.cand_leaf {
            Some(l) => leaf_bytes(rec, t, l, cap(l)?),
            None => Default::default(),
        },
        rows,
        ..root
    })
}

/// Token maps of the n-gram lookups, by root: little-endian u32 per token, zero-padded to 32 B.
static TOKEN_MAPS: std::sync::Mutex<Vec<(Hash, std::sync::Arc<Vec<u8>>)>> = std::sync::Mutex::new(Vec::new());

/// Keeps the token map `map` for the openings and returns its root.
pub fn register_token_map(map: &[i32]) -> Hash {
    let mut bytes: Vec<u8> = map.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect();
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let leaves: Vec<Hash> = bytes.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    let root = keryx_verifiable_inference::merkle::rt_range_proof(&leaves, 0, 1).0;
    let mut t = TOKEN_MAPS.lock().unwrap_or_else(|e| e.into_inner());
    if !t.iter().any(|(r, _)| *r == root) {
        t.push((root, std::sync::Arc::new(bytes)));
    }
    root
}

fn token_map(root: &Hash) -> Option<std::sync::Arc<Vec<u8>>> {
    TOKEN_MAPS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(r, _)| r == root).map(|(_, b)| b.clone())
}

/// Makes the n-gram tables of a model openable: the weight index of `table_gguf` (tree kept as
/// `engram-tree.bin`) and the token map stored in `body_gguf`. Returns the table root.
pub fn register_engram(body_gguf: &str, table_gguf: &str, model_id: Hash) -> Result<Hash, String> {
    let mut f = std::fs::File::open(body_gguf).map_err(|e| format!("{body_gguf}: {e}"))?;
    let arch = "deepseek41";
    let map = crate::gguf::read_i32_array(&mut f, &format!("{arch}.engram.token_map")).map_err(|e| e.to_string())?.ok_or("no token map in the model")?;
    register_token_map(&map);
    let index = crate::pom::WeightIndex::build_from_gguf_named(table_gguf, model_id, "engram-tree.bin").map_err(|e| e.to_string())?;
    let root = index.r_t;
    crate::pom::register_table_index(std::sync::Arc::new(index));
    Ok(root)
}

/// The input link of the n-gram value product at position `t`, whose drawn input block is `block`:
/// the tokens back from `t` bound to their positions, their map entries and the table row the
/// block reads.
pub fn engram_link(m: &TierManifest, rec: &TraceRecord, tokens: &[i32], layer: u64, t: u64, block: u64) -> Option<keryx_verifiable_inference::opening::LinkOpening> {
    use keryx_verifiable_inference::merkle::{b3_path, rt_range_proof};
    use keryx_verifiable_inference::opening::{ChunkRange, LinkOpening, PrevToken};
    let ctx = m.chain?;
    let le = ctx.layer_base as u64 + layer;
    let e = match m.entry(le).chain {
        manifest::Chain::Layer(ll) => ll.engram?,
        _ => return None,
    };
    let tok = |q: u64| tokens.get(q as usize).map(|&v| v as u32);
    let shifts = t.min(e.n_gram as u64 - 1);
    let nl = m.n_leaves();
    let prev: Vec<PrevToken> =
        (0..shifts).map(|s| t - 1 - s).map(|q| Some(PrevToken { tok: tok(q)?, meta_path: b3_path(rec.leaves_at(q), nl - 1), pos_path: rec.pos_path(q) })).collect::<Option<_>>()?;
    let toks: Vec<u32> = std::iter::once(tok(t)).chain(prev.iter().map(|p| Some(p.tok))).collect::<Option<_>>()?;
    let map = token_map(&e.map_root)?;
    let leaves: Vec<Hash> = map.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    let mut ids = Vec::with_capacity(e.n_gram as usize);
    let mut ranges = Vec::with_capacity(toks.len());
    for &tk in &toks {
        if tk >= e.map_len {
            ids.push(None);
            ranges.push(ChunkRange::default());
            continue;
        }
        let c = tk as u64 * 4 / 32;
        let o = (tk as usize * 4) % 32;
        let chunk = &map[c as usize * 32..c as usize * 32 + 32];
        ids.push(Some(u32::from_le_bytes(chunk[o..o + 4].try_into().ok()?)));
        ranges.push(ChunkRange { first_chunk: c, chunks: chunk.to_vec(), proof: rt_range_proof(&leaves, c, 1).1 });
    }
    ids.resize(e.n_gram as usize, None);
    let rows = keryx_verifiable_inference::engram::engram_rows(&e, &ids)?;
    let d = e.head_dim as u64;
    let row = *rows.get((block * 32 / d) as usize)? as u64;
    let rb = keryx_verifiable_inference::quant::row_bytes(e.table_type, d as usize)? as u64;
    let table = crate::pom::table_index(&e.table_root)?;
    let row = crate::trace::range_proof(&table, row * rb / 32, ((row + 1) * rb - 1) / 32)?;
    Some(LinkOpening::Engram { layer: m.entry_opening(le), prev, map: ranges, row })
}

/// What the n-gram residual point of a seed opens: the module's layer, the position, the stream
/// and the chunk of it.
#[derive(Clone, Copy, Debug)]
pub struct EngramDraw {
    pub layer: u64,
    pub e: manifest::EngramLinks,
    pub t: u64,
    pub stream: u64,
    pub chunk: u64,
    pub n_hc: u64,
    pub dim: u64,
}

/// The n-gram residual point of `seed`, drawn as the node draws it.
pub fn draw_engram(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<EngramDraw> {
    use keryx_verifiable_inference::engram::ENGRAM_POINT;
    let ctx = m.chain?;
    if ctx.n_engram == 0 || rec.n_pos == 0 {
        return None;
    }
    let d = draw::point(seed, ENGRAM_POINT);
    let module = d[1] % ctx.n_engram as u64;
    let (layer, ll, e) = find_layer(m, |ll| ll.engram.filter(|e| e.module as u64 == module))?;
    let n_hc = ll.hc?.n_hc as u64;
    let dim = m.entry(e.kv as u64).m.checked_div(n_hc + 1)?;
    let per = 4 * dim / keryx_verifiable_inference::merkle::B3_CHUNK as u64;
    if layer == 0 || per == 0 {
        return None;
    }
    Some(EngramDraw { layer, e, t: d[0] % rec.n_pos as u64, stream: d[2] % n_hc, chunk: (d[2] / n_hc) % per, n_hc, dim })
}

impl EngramDraw {
    fn per(&self) -> u64 {
        4 * self.dim / keryx_verifiable_inference::merkle::B3_CHUNK as u64
    }

    fn src_leaf(&self, np: u64) -> u32 {
        chain::resid_leaf(np, self.layer - 1, chain::Resid::Out) as u32
    }

    /// Captures at the drawn position: the streams after and before the lookup, the value product.
    pub fn captures(&self, np: u64) -> Vec<(u32, u32)> {
        let t = self.t as u32;
        vec![(t, self.e.out_leaf), (t, self.src_leaf(np)), (t, 2 * self.e.kv + 1)]
    }
}

/// The n-gram residual opening of `d` from its captures.
pub fn engram_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    index: &WeightIndex,
    d: &EngramDraw,
    caps: &Captures,
    response_txid: Hash,
) -> Result<keryx_verifiable_inference::opening::EngramOpening, String> {
    use crate::trace::leaf_chunk;
    use keryx_verifiable_inference::opening::EngramOpening;
    let ctx = m.chain.ok_or("no chain context")?;
    let (np, t, e) = (m.n_products as u64, d.t, &d.e);
    let cap = |leaf: u32| caps.get(&(t as u32, leaf)).ok_or_else(|| format!("no capture of leaf {leaf} at {t}"));
    let (out, src, y) = (cap(e.out_leaf)?, cap(d.src_leaf(np))?, cap(2 * e.kv + 1)?);
    let per = d.per();
    let base = d.stream * per;
    let chunk = |leaf: u32, b: &[u8], c: u64| leaf_chunk(rec, t, leaf, b, c).ok_or_else(|| format!("no chunk {c} of leaf {leaf} at {t}"));
    let (x, key) = (0..per)
        .map(|c| Ok((chunk(d.src_leaf(np), src, base + c)?, chunk(2 * e.kv + 1, y, base + c)?)))
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .unzip();
    Ok(EngramOpening {
        response_txid,
        pos_root: rec.trace_root,
        seq_root: seq.root,
        layer: m.entry_opening(ctx.layer_base as u64 + d.layer),
        kv: m.entry_opening(e.kv as u64),
        gate: bytes_range(m, index, e.gate as u64, 4 * d.stream * d.dim, 4 * d.dim).ok_or("no gate weights")?,
        pos_path: rec.pos_path(t),
        dst: chunk(e.out_leaf, out, base + d.chunk)?,
        x,
        key,
        value: chunk(2 * e.kv + 1, y, d.n_hc * per + d.chunk)?,
    })
}

/// What the attention point of a seed opens on a compressed attention layer: the position, the
/// query head, the window, and the selected compressed rows once the selection is read.
#[derive(Clone, Debug)]
pub struct CompAttnDraw {
    pub layer: u64,
    pub a: manifest::AttnLinks,
    pub c: CompLinks,
    pub t: u64,
    pub qh: u64,
    pub w0: u64,
    pub n_w: u64,
    /// Selection leaf of the layer (none on a window-only layer) and the rows it keeps.
    pub sel_leaf: Option<u32>,
    pub ids: Option<Vec<u64>>,
    anchor_draw: u64,
}

/// The attention point of `seed` when it lands on a compressed attention layer, as the node
/// draws it; the selected rows come with [`CompAttnDraw::with_sel`].
pub fn draw_comp_attn(m: &TierManifest, rec: &TraceRecord, seed: &Hash) -> Option<CompAttnDraw> {
    let ctx = m.chain?;
    if ctx.n_attn == 0 {
        return None;
    }
    let d = draw::point(seed, chain::ATTN_POINT);
    let slot = d[1] % ctx.n_attn as u64;
    let (layer, ll, a) = find_layer(m, |ll| ll.attn.filter(|a| a.slot as u64 == slot))?;
    let c = ll.comp?;
    let t = chain::resid_position(&ctx, &rec.response_ctx(), layer, d[0]).ok()?;
    let w0 = (t + 1).saturating_sub(a.swa_window as u64);
    let sel_leaf = (c.ratio > 0).then(|| chain::sel_leaf(m.n_products as u64, ctx.n_layers as u64, chain::n_router(&ctx), c.sel as u64) as u32);
    let ids = if c.ratio == 0 { Some(Vec::new()) } else { None };
    Some(CompAttnDraw { layer, a, c, t, qh: d[2] % a.n_head as u64, w0, n_w: t + 1 - w0, sel_leaf, ids, anchor_draw: d[3] })
}

impl CompAttnDraw {
    /// The rows the selection leaf `sel` keeps at the drawn position.
    pub fn with_sel(mut self, sel: &[u8]) -> Option<Self> {
        let n_top = self.c.n_top as usize;
        if n_top == 0 || sel.len() != 4 + 4 * n_top {
            return None;
        }
        let n_vis = (self.t + 1) / self.c.ratio as u64;
        let slots: Vec<u64> = sel[4..].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as u64).collect();
        self.ids = Some(slots[..(n_vis as usize).min(n_top)].to_vec());
        Some(self)
    }

    pub fn tree(&self) -> Option<keryx_verifiable_inference::attn_blocks::BlockTree> {
        use keryx_verifiable_inference::attn_blocks::{block_params, partial_len, BlockTree};
        let (cl, b) = block_params(keryx_verifiable_inference::dsv41::COMP_ROW_COST, partial_len(self.a.head_dim as usize))?;
        BlockTree::new(0, self.n_w + self.ids.as_ref()?.len() as u64, cl, b)
    }

    pub fn leaf(&self, path: &[u64]) -> bool {
        self.tree().is_some_and(|t| path.len() as u32 == t.depth())
    }

    /// Window positions `[ws, we)` and selected-row indexes `[cs, ce)` under the node of `path`.
    fn node(&self, path: &[u64]) -> Option<((u64, u64), (usize, usize))> {
        let (v0, v1) = self.tree()?.node(path)?;
        let n_w = self.n_w;
        Some(((v0.min(n_w), v1.min(n_w)), ((v0.max(n_w) - n_w) as usize, (v1.max(n_w) - n_w) as usize)))
    }

    fn anchor(&self, path: &[u64]) -> Option<Option<u64>> {
        let ((ws, we), _) = self.node(path)?;
        Some((ws < we).then(|| self.w0 + ws + self.anchor_draw % (we - ws)))
    }

    /// Rotary base and scaling of the layer.
    pub fn rope(&self) -> (f64, keryx_verifiable_inference::attn::Yarn) {
        if self.c.ratio > 0 { (f32::from_bits(self.c.rope_base) as f64, self.c.yarn()) } else { (f32::from_bits(self.a.rope_base) as f64, keryx_verifiable_inference::attn::Yarn::NONE) }
    }

    /// Captures the phase of `path` needs: the query, the output and selection at the first
    /// phase, the anchor of a leaf.
    pub fn captures(&self, path: &[u64]) -> Option<Vec<(u32, u32)>> {
        let (a, t) = (&self.a, self.t as u32);
        let mut out = vec![(t, 2 * a.q + 1), (t, 2 * a.out)];
        out.extend(self.sel_leaf.map(|l| (t, l)));
        if self.leaf(path) {
            if let Some(ps) = self.anchor(path)? {
                out.push((ps as u32, 2 * a.k + 1));
            }
        }
        Some(out)
    }

    /// Spans of the phase of `path`: the window rows and selected rows of its node, then the
    /// head's mass at the first phase.
    pub fn spans(&self, m: &TierManifest, n_pos: u64, path: &[u64]) -> Option<Vec<SpanPlan>> {
        let layout = chain::seq_layout(&m.chain?);
        let (a, c) = (&self.a, &self.c);
        let rb = (a.head_dim / 32 * 34) as u64;
        let ((ws, we), (cs, ce)) = self.node(path)?;
        let mut out = Vec::new();
        if ws < we {
            out.push(SpanPlan::new(layout.kcache(a.slot as u64, 0)?, n_pos * rb, (self.w0 + ws) * rb, (self.w0 + we) * rb)?);
        }
        if ce > cs {
            let crb = c.row_len as u64 / 32 * 34;
            let total = n_pos / c.ratio as u64 * crb;
            let leaf = layout.comp(c.slot as u64)?;
            for &id in &self.ids.as_ref()?[cs..ce] {
                out.push(SpanPlan::new(leaf, total, id * crb, id * crb + rb)?);
            }
        }
        if path.is_empty() {
            let nh = a.n_head as u64;
            let lo = (self.t * nh + self.qh) * 4;
            out.push(SpanPlan::new(layout.lse(a.slot as u64, 0)?, n_pos * nh * 4, lo, lo + 4)?);
        }
        Some(out)
    }

    /// The rotated query head and the keys of the node of `path` (window rows, then the latent
    /// of the selected rows), from the opening's spans.
    pub fn query_and_keys(&self, caps: &Captures, op: &keryx_verifiable_inference::opening::CompAttnOpening, path: &[u64]) -> Option<(Vec<f64>, Vec<Vec<f32>>)> {
        use keryx_verifiable_inference::merkle::B3_CHUNK;
        use keryx_verifiable_inference::quant::{dequant_row, Q8_0};
        let hd = self.a.head_dim as usize;
        let qb = caps.get(&(self.t as u32, 2 * self.a.q + 1))?;
        let lo = self.qh as usize * hd * 4;
        let mut q: Vec<f64> = qb.get(lo..lo + 4 * hd)?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect();
        let (base, yarn) = self.rope();
        keryx_verifiable_inference::dsv41::rope_tail(&mut q, self.t, self.a.n_rot as usize, base, &yarn);
        let rb = hd / 32 * 34;
        let ((ws, we), _) = self.node(path)?;
        let mut keys = Vec::new();
        if ws < we {
            let s = &op.window;
            let off = ((self.w0 + ws) as usize * rb).checked_sub(s.first_chunk as usize * B3_CHUNK)?;
            for r in s.bytes.get(off..off + (we - ws) as usize * rb)?.chunks_exact(rb) {
                keys.push(dequant_row(Q8_0, r, hd)?);
            }
        }
        let crb = self.c.row_len as usize / 32 * 34;
        let ((_, _), (cs, _)) = self.node(path)?;
        for (j, s) in op.comp.iter().enumerate() {
            let id = self.ids.as_ref()?[cs + j] as usize;
            let off = (id * crb).checked_sub(s.first_chunk as usize * B3_CHUNK)?;
            keys.push(dequant_row(Q8_0, s.bytes.get(off..off + rb)?, hd)?);
        }
        Some((q, keys))
    }
}

/// Every field the phase of `path` of `d` carries, partials left empty; window and selected rows
/// are kept on inner phases too, for the partials, and reduced by [`comp_attn_phase`].
#[allow(clippy::too_many_arguments)]
pub fn comp_attn_opening(
    m: &TierManifest,
    rec: &TraceRecord,
    seq: &SeqTrace,
    index: &WeightIndex,
    d: &CompAttnDraw,
    caps: &Captures,
    spans: &[SeqCapture],
    response_txid: Hash,
    path: &[u64],
    prev: Option<&[u8]>,
) -> Result<keryx_verifiable_inference::opening::CompAttnOpening, String> {
    use crate::trace_seq::value_chunks as value_chunks_pub;
    use keryx_verifiable_inference::attn_blocks::partial_opening;
    use keryx_verifiable_inference::opening::{CompAttnOpening, SelOpening};
    let (a, c, t) = (&d.a, &d.c, d.t);
    let hd = a.head_dim as u64;
    let cap = |pos: u64, leaf: u32| caps.get(&(pos as u32, leaf)).ok_or_else(|| format!("no capture of leaf {leaf} at {pos}"));
    let plans = d.spans(m, rec.n_pos as u64, path).ok_or("no attention spans")?;
    if plans.len() != spans.len() {
        return Err(format!("{} spans captured for {}", spans.len(), plans.len()));
    }
    let mut opened = plans.iter().zip(spans).map(|(p, s)| span(p, s, seq)).collect::<Result<Vec<_>, String>>()?;
    let lse = if path.is_empty() { opened.pop().ok_or("no mass span")? } else { Default::default() };
    let ((ws, we), _) = d.node(path).ok_or("no node")?;
    let window = if ws < we { opened.remove(0) } else { Default::default() };
    let (parent, parent_path) = match (path.last(), prev) {
        (Some(&i), Some(b)) => partial_opening(b, hd as usize, i as usize).ok_or("no parent partial")?,
        (None, None) => Default::default(),
        _ => return Err("the parent partials do not match the path".into()),
    };
    let base = m.chain.ok_or("no chain context")?.layer_base as u64;
    let root = path.is_empty();
    let last = d.leaf(path);
    let sel = match d.sel_leaf {
        Some(l) => {
            let lb = leaf_bytes(rec, t, l, cap(t, l)?);
            Some(SelOpening { bytes: lb.bytes, leaf_path: lb.leaf_path, pos_path: rec.pos_path(t) })
        }
        None => None,
    };
    let mut op = CompAttnOpening { response_txid, pos_root: rec.trace_root, seq_root: seq.root, layer: m.entry_opening(base + d.layer), sel, path: path.to_vec(), parent, parent_path, window, comp: opened, ..Default::default() };
    if root || last {
        op.q_entry = m.entry_opening(a.q as u64);
        op.q = value_chunks_pub(rec, t, 2 * a.q + 1, cap(t, 2 * a.q + 1)?, d.qh * hd, hd).ok_or("query chunks")?;
        op.pos_path = rec.pos_path(t);
    }
    if root {
        op.o_entry = m.entry_opening(a.out as u64);
        op.out = leaf_bytes(rec, t, 2 * a.out, cap(t, 2 * a.out)?);
        op.sinks = bytes_range(m, index, c.sinks as u64, 4 * d.qh, 4).ok_or("no sink weights")?;
        op.lse = lse;
    }
    if last {
        if let Some(ps) = d.anchor(path).ok_or("no anchor")? {
            op.anchor_pos_path = rec.pos_path(ps);
            op.anchor_k = value_chunks_pub(rec, ps, 2 * a.k + 1, cap(ps, 2 * a.k + 1)?, 0, hd).ok_or("anchor chunks")?;
            op.norms = vec![bytes_range(m, index, a.k_norm as u64, 0, 4 * hd).ok_or("key norm weights")?];
        }
    }
    Ok(op)
}

/// `op` reduced to the fields its phase carries, with `partials` on an inner phase.
pub fn comp_attn_phase(d: &CompAttnDraw, op: &keryx_verifiable_inference::opening::CompAttnOpening, path: &[u64], partials: Vec<u8>) -> keryx_verifiable_inference::opening::CompAttnOpening {
    if d.leaf(path) {
        return op.clone();
    }
    keryx_verifiable_inference::opening::CompAttnOpening { window: Default::default(), comp: Vec::new(), anchor_pos_path: Vec::new(), anchor_k: Vec::new(), norms: Vec::new(), partials, ..op.clone() }
}

/// The phase of `path` of `d` from its full opening `full`: the partials of an inner phase picked
/// so the node's check of the phase passes.
#[allow(clippy::too_many_arguments)]
pub fn comp_attn_step(
    tier: &keryx_verifiable_inference::verify::TierCtx,
    ctx: &keryx_verifiable_inference::chain::ChainCtx,
    resp: &keryx_verifiable_inference::verify::ResponseCtx,
    seed: &Hash,
    d: &CompAttnDraw,
    full: &keryx_verifiable_inference::opening::CompAttnOpening,
    caps: &Captures,
    path: &[u64],
    parent_root: Option<&Hash>,
) -> Result<keryx_verifiable_inference::opening::CompAttnOpening, String> {
    use keryx_verifiable_inference::attn_blocks::{encode_partials, pick_partials};
    use keryx_verifiable_inference::dsv41::verify_comp_attn_phase;
    let verify = |op: &keryx_verifiable_inference::opening::CompAttnOpening| verify_comp_attn_phase(tier, ctx, resp, seed, path, parent_root, op);
    let op = if d.leaf(path) {
        full.clone()
    } else {
        let tree = d.tree().ok_or("no attention tree")?;
        let (q, keys) = d.query_and_keys(caps, full, path).ok_or("no query or keys")?;
        let p0 = tree.node(path).ok_or("no node")?.0;
        let scale = f32::from_bits(d.a.kq_scale) as f64;
        let check = |p: &[keryx_verifiable_inference::attn::Partial]| verify(&comp_attn_phase(d, full, path, encode_partials(p))).is_ok();
        let partials = pick_partials(&q, scale, &keys, &keys, p0, &tree, path, check).ok_or_else(|| format!("no partials verify at layer {}, position {}, path {path:?}", d.layer, d.t))?;
        comp_attn_phase(d, full, path, partials)
    };
    verify(&op).map_err(|e| format!("compressed attention phase {e:?} at layer {}, position {}, path {path:?}", d.layer, d.t))?;
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_verifiable_inference::dsv41::verify_comp_point;
    use keryx_verifiable_inference::dsv41kit::{build, CompKit, N};
    use keryx_verifiable_inference::merkle::b3_chunk_cv;
    use keryx_verifiable_inference::verify::Fail;

    struct Fixture {
        m: TierManifest,
        index: WeightIndex,
        rec: TraceRecord,
        seq: SeqTrace,
        caps: Captures,
        kit: CompKit,
    }

    fn fixture(swap: bool) -> Fixture {
        let kit = build(swap);
        let mut file = b"KXMANIF2".to_vec();
        file.extend((kit.entries.len() as u32).to_le_bytes());
        file.extend((kit.tier.n_products as u32).to_le_bytes());
        for b in &kit.entries {
            file.extend_from_slice(b);
        }
        let mut m = TierManifest::parse(&file).expect("kit manifest");
        m.chain = Some(kit.ctx);
        let index = crate::pom::index_from_ram(kit.weights.clone());
        assert_eq!(index.r_t, kit.tier.r_t);
        let leaves: Vec<Hash> = kit.pos_leaves.iter().flat_map(|p| p.iter().map(|b| *blake3::hash(b).as_bytes())).collect();
        let rec = TraceRecord::new_chain(&m, N as u32, vec![N as u32; N], leaves, &vec![7; N]).expect("kit record");
        assert_eq!(rec.trace_root, kit.pos_root);
        let seq = SeqTrace::new(kit.seq_leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect()).unwrap();
        assert_eq!(seq.root, kit.seq_root);
        let caps: Captures = kit.pos_leaves.iter().enumerate().flat_map(|(t, p)| p.iter().enumerate().map(move |(l, b)| ((t as u32, l as u32), b.clone()))).collect();
        Fixture { m, index, rec, seq, caps, kit }
    }

    /// What the engine would return for `plans`: the span bytes and the chunk values of the leaf.
    fn replay(kit: &CompKit, plans: &[SpanPlan]) -> Vec<SeqCapture> {
        plans
            .iter()
            .map(|p| {
                let data = &kit.seq_leaves[p.leaf as usize];
                let cvs = data.chunks(keryx_verifiable_inference::merkle::B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect();
                SeqCapture { bytes: data[p.lo as usize..p.hi as usize].to_vec(), cvs }
            })
            .collect()
    }

    fn open(f: &Fixture, seed: &Hash) -> CompOpeningPoint {
        let d = draw_comp(&f.m, &f.rec, seed).expect("draw");
        let spans = replay(&f.kit, &d.spans(&f.m).unwrap());
        comp_opening(&f.m, &f.rec, &f.seq, &f.index, &d, &f.caps, &spans, [1; 32]).unwrap()
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..24u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    fn sel_fixture(fault: keryx_verifiable_inference::dsv41kit::SelFault) -> (TierManifest, TraceRecord, SeqTrace, Captures, keryx_verifiable_inference::dsv41kit::SelKit) {
        use keryx_verifiable_inference::dsv41kit::{build_sel, S_N};
        let kit = build_sel(fault);
        let mut file = b"KXMANIF2".to_vec();
        file.extend((kit.entries.len() as u32).to_le_bytes());
        file.extend((kit.tier.n_products as u32).to_le_bytes());
        for b in &kit.entries {
            file.extend_from_slice(b);
        }
        let mut m = TierManifest::parse(&file).expect("kit manifest");
        m.chain = Some(kit.ctx);
        let leaves: Vec<Hash> = kit.pos.iter().flat_map(|p| p.1.clone()).collect();
        let rec = TraceRecord::new_chain(&m, S_N as u32, vec![1; S_N], leaves, &vec![7; S_N]).expect("kit record");
        assert_eq!(rec.trace_root, kit.pos_root);
        let seq = SeqTrace::new(kit.seq_hashes.clone()).unwrap();
        let caps: Captures = kit.pos.iter().enumerate().flat_map(|(t, p)| p.0.iter().enumerate().map(move |(l, b)| ((t as u32, l as u32), b.clone()))).collect();
        (m, rec, seq, caps, kit)
    }

    fn sel_verdicts(fault: keryx_verifiable_inference::dsv41kit::SelFault) -> Vec<Result<(), Fail>> {
        let (m, rec, seq, caps, kit) = sel_fixture(fault);
        let resp = crate::trace_seq::seq_response_ctx(&rec, &seq);
        (0..96u32)
            .map(|s| {
                let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                let d = draw_comp_sel(&m, &rec, &seed).expect("draw");
                let plans = d.t.and_then(|t| caps.get(&(t as u32, d.sel_leaf))).map(|sel| d.spans(&m, rec.n_pos as u64, &seed, sel).expect("spans")).unwrap_or_default();
                let spans: Vec<SeqCapture> = plans
                    .iter()
                    .map(|p| {
                        let data = &kit.seq_leaves[p.leaf as usize];
                        let cvs = data.chunks(keryx_verifiable_inference::merkle::B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect();
                        SeqCapture { bytes: data[p.lo as usize..p.hi as usize].to_vec(), cvs }
                    })
                    .collect();
                let op = comp_sel_opening(&m, &rec, &seq, &d, &seed, &caps, &spans, [1; 32]).unwrap();
                keryx_verifiable_inference::dsv41::verify_comp_sel_point(&kit.tier, &kit.ctx, &resp, &seed, &op)
            })
            .collect()
    }

    #[test]
    fn two_level_selection_openings_verify() {
        use keryx_verifiable_inference::dsv41kit::SelFault;
        let v = sel_verdicts(SelFault::None);
        assert!(v.iter().all(|r| r.is_ok()), "{v:?}");
        for f in [SelFault::LowRows, SelFault::LowBlocks, SelFault::OutsideCandidates] {
            let v = sel_verdicts(f);
            assert!(v.iter().all(|r| r.is_ok() || *r == Err(Fail::Mismatch)), "{v:?}");
            assert!(v.iter().any(|r| *r == Err(Fail::Mismatch)));
        }
    }

    #[test]
    fn n_gram_links_match_the_node_and_verify() {
        use keryx_verifiable_inference::engramkit::{build_link, K, N};
        use keryx_verifiable_inference::testkit::layout_at;
        let kit = build_link(false);
        let mut file = b"KXMANIF2".to_vec();
        file.extend((kit.entries.len() as u32).to_le_bytes());
        file.extend((kit.tier.n_products as u32).to_le_bytes());
        for b in &kit.entries {
            file.extend_from_slice(b);
        }
        let mut m = TierManifest::parse(&file).expect("kit manifest");
        m.chain = Some(kit.ctx);
        let tokens: Vec<i32> = kit.tokens.iter().map(|&t| t as i32).collect();
        let ubatch: Vec<u32> = (0..N).map(|t| layout_at(t as u64, N as u32).0).collect();
        let leaves: Vec<Hash> = kit.pos.iter().flat_map(|p| p.1.clone()).collect();
        let rec = TraceRecord::new_chain(&m, N as u32, ubatch, leaves, &tokens).expect("kit record");
        let table = std::sync::Arc::new(crate::pom::index_from_ram(kit.table.clone()));
        assert_eq!(table.r_t, kit.e.table_root);
        crate::pom::register_table_index(table);
        let map: Vec<i32> = kit.map.iter().map(|&v| v as i32).collect();
        assert_eq!(register_token_map(&map), kit.e.map_root);
        let resp = keryx_verifiable_inference::verify::ResponseCtx { trace_root: rec.trace_root, n_prompt: N as u32, n_pos: N as u32 };
        let mut at_start = 0;
        for s in 0..32u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let (p, tok, want) = kit.open(&seed, false);
            let t = draw::point(&seed, 0)[0] % N as u64;
            at_start += (t < 3) as u32;
            let block = chain::link_block(&seed, 0, &p.x, layout_at(t, N as u32).1, K as u64).unwrap();
            let got = engram_link(&m, &rec, &tokens, 0, t, block).expect("link");
            assert_eq!(got, want, "seed {s}");
            chain::verify_chain_point(&kit.tier, &kit.ctx, &resp, &seed, 0, &p, tok, &got, None).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
        }
        assert!(at_start > 0, "the first positions, with fewer tokens behind them, are drawn");
    }

    #[test]
    fn n_gram_residual_openings_match_the_node_and_verify() {
        use keryx_verifiable_inference::engramkit::{build_resid, ResidFault, R_N};
        for fault in [ResidFault::None, ResidFault::PlainGate, ResidFault::Ungated, ResidFault::SwappedGates] {
            let kit = build_resid(fault);
            let mut file = b"KXMANIF2".to_vec();
            file.extend((kit.entries.len() as u32).to_le_bytes());
            file.extend((kit.tier.n_products as u32).to_le_bytes());
            for b in &kit.entries {
                file.extend_from_slice(b);
            }
            let mut m = TierManifest::parse(&file).expect("kit manifest");
            m.chain = Some(kit.ctx);
            let index = crate::pom::index_from_ram(kit.weights.clone());
            let leaves: Vec<Hash> = kit.pos.iter().flat_map(|p| p.1.clone()).collect();
            let rec = TraceRecord::new_chain(&m, R_N as u32, vec![1; R_N], leaves, &vec![7; R_N]).expect("kit record");
            let seq = SeqTrace { leaves: Vec::new(), root: [5; 32] };
            let caps: Captures = kit.pos.iter().enumerate().flat_map(|(t, p)| p.0.iter().enumerate().map(move |(l, b)| ((t as u32, l as u32), b.clone()))).collect();
            for s in 0..32u32 {
                let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
                let d = draw_engram(&m, &rec, &seed).expect("draw");
                assert!(d.captures(m.n_products as u64).iter().all(|c| caps.contains_key(c)));
                let got = engram_opening(&m, &rec, &seq, &index, &d, &caps, [1; 32]).unwrap();
                assert_eq!(got, kit.open(&seed), "seed {s}");
                let r = keryx_verifiable_inference::engram::verify_engram_point(&kit.tier, &kit.ctx, &kit.resp(), &seed, &got);
                assert_eq!(r, if fault == ResidFault::None { Ok(()) } else { Err(Fail::Mismatch) }, "seed {s}");
            }
        }
    }

    /// Every phase of every path of the compressed attention point of `seed`, opened by the miner
    /// and checked by the node; the deepest path reached, or the first refusal.
    fn walk_attn(kit: &keryx_verifiable_inference::dsv41kit::AttnKit, seed: &Hash) -> Result<usize, (Vec<u64>, String)> {
        use keryx_verifiable_inference::chain::AttnStep;
        use keryx_verifiable_inference::dsv41::verify_comp_attn_phase;
        use keryx_verifiable_inference::dsv41kit::A_N;
        use keryx_verifiable_inference::testkit::layout_at;
        let mut file = b"KXMANIF2".to_vec();
        file.extend((kit.entries.len() as u32).to_le_bytes());
        file.extend((kit.tier.n_products as u32).to_le_bytes());
        for b in &kit.entries {
            file.extend_from_slice(b);
        }
        let mut m = TierManifest::parse(&file).expect("kit manifest");
        m.chain = Some(kit.ctx);
        let index = crate::pom::index_from_ram(kit.weights.clone());
        let ubatch: Vec<u32> = (0..A_N).map(|t| layout_at(t as u64, A_N as u32).0).collect();
        let leaves: Vec<Hash> = kit.pos.iter().flat_map(|p| p.1.clone()).collect();
        let rec = TraceRecord::new_chain(&m, A_N as u32, ubatch, leaves, &vec![7; A_N]).expect("kit record");
        let seq = SeqTrace::new(kit.seq_hashes.clone()).unwrap();
        let caps: Captures = kit.pos.iter().enumerate().flat_map(|(t, p)| p.0.iter().enumerate().map(move |(l, b)| ((t as u32, l as u32), b.clone()))).collect();
        let resp = kit.resp();
        let mut d = draw_comp_attn(&m, &rec, seed).expect("draw");
        if let Some(l) = d.sel_leaf {
            let sel = caps[&(d.t as u32, l)].clone();
            d = d.with_sel(&sel).expect("selection");
        }
        let replay = |plans: &[SpanPlan]| -> Vec<SeqCapture> {
            plans
                .iter()
                .map(|p| {
                    let data = &kit.seq_leaves[p.leaf as usize];
                    let cvs = data.chunks(keryx_verifiable_inference::merkle::B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect();
                    SeqCapture { bytes: data[p.lo as usize..p.hi as usize].to_vec(), cvs }
                })
                .collect()
        };
        let mut stack: Vec<(Vec<u64>, Option<Vec<u8>>, Option<Hash>)> = vec![(Vec::new(), None, None)];
        let mut deepest = 0;
        while let Some((path, prev, root)) = stack.pop() {
            let fail = |e: String| (path.clone(), e);
            assert!(d.captures(&path).unwrap().iter().all(|c| caps.contains_key(c)));
            let spans = replay(&d.spans(&m, rec.n_pos as u64, &path).unwrap());
            let full = comp_attn_opening(&m, &rec, &seq, &index, &d, &caps, &spans, [1; 32], &path, prev.as_deref()).map_err(fail)?;
            let op = comp_attn_step(&kit.tier, &kit.ctx, &resp, seed, &d, &full, &caps, &path, root.as_ref()).map_err(fail)?;
            match verify_comp_attn_phase(&kit.tier, &kit.ctx, &resp, seed, &path, root.as_ref(), &op).map_err(|e| fail(format!("{e:?}")))? {
                AttnStep::Done => deepest = deepest.max(path.len()),
                AttnStep::Next { root, n, .. } => stack.extend((0..n).map(|i| ([path.clone(), vec![i]].concat(), Some(op.partials.clone()), Some(root)))),
            }
        }
        Ok(deepest)
    }

    #[test]
    fn compressed_attention_openings_verify_at_every_phase() {
        use keryx_verifiable_inference::attn_blocks::set_test_budget;
        use keryx_verifiable_inference::dsv41kit::{build_attn, AttnFault, A_BUDGET, A_N, A_NTOP};
        set_test_budget(A_BUDGET);
        let seeds: Vec<Hash> = (0..40u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes()).collect();
        let kit = build_attn(AttnFault::None);
        let deep = seeds.iter().map(|s| walk_attn(&kit, s).unwrap_or_else(|e| panic!("{e:?}"))).max();
        assert!(deep >= Some(1), "{deep:?}");
        let t_of = |s: &Hash| (draw::point(s, chain::ATTN_POINT)[0] % A_N as u64) as usize;
        for fault in [AttnFault::NoSink, AttnFault::Dense, AttnFault::ForgedWindow] {
            let kit = build_attn(fault);
            let mut refused = 0;
            for s in &seeds {
                match walk_attn(&kit, s) {
                    Ok(_) => assert!(fault == AttnFault::Dense && (t_of(s) + 1) / 2 <= A_NTOP || fault == AttnFault::ForgedWindow, "{s:?}"),
                    Err((path, _)) => {
                        refused += 1;
                        if fault != AttnFault::ForgedWindow {
                            assert!(path.is_empty(), "refused at {path:?}");
                        }
                    }
                }
            }
            assert!(refused > 0, "fault never refused");
        }
    }

    /// The n-gram tables of a local DeepSeek-V4.1 directory (`KERYX_DSV41_DIR`) register under
    /// their pinned root, beside an untouched model tree.
    #[test]
    #[ignore]
    fn local_n_gram_tables_register_under_their_pinned_root() {
        let dir = std::path::PathBuf::from(std::env::var("KERYX_DSV41_DIR").unwrap());
        let spec = crate::models::DEEPSEEK_V4_1_FLASH;
        let c = crate::models::companion(&spec).unwrap();
        let (body, table) = (dir.join("model.gguf"), dir.join(c.file));
        let root = register_engram(body.to_str().unwrap(), table.to_str().unwrap(), spec.model_id).unwrap();
        assert_eq!(root, c.root);
        assert!(crate::pom::table_index(&c.root).is_some());
        let map = crate::gguf::read_i32_array(&mut std::fs::File::open(&body).unwrap(), "deepseek41.engram.token_map").unwrap().unwrap();
        assert!(token_map(&register_token_map(&map)).is_some());
    }

    #[test]
    fn compressed_row_openings_verify() {
        let f = fixture(false);
        let resp = crate::trace_seq::seq_response_ctx(&f.rec, &f.seq);
        for s in seeds() {
            let d = draw_comp(&f.m, &f.rec, &s).unwrap();
            assert_eq!(d.captures().len(), 4);
            verify_comp_point(&f.m.tier_ctx(&f.index), &f.kit.ctx, &resp, &s, &open(&f, &s)).unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    #[test]
    fn a_trace_of_misweighted_rows_cannot_be_opened_into_a_passing_point() {
        let f = fixture(true);
        let resp = crate::trace_seq::seq_response_ctx(&f.rec, &f.seq);
        for s in seeds() {
            assert_eq!(verify_comp_point(&f.m.tier_ctx(&f.index), &f.kit.ctx, &resp, &s, &open(&f, &s)), Err(Fail::Mismatch));
        }
    }
}
