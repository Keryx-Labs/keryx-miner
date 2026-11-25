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
