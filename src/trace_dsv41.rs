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
