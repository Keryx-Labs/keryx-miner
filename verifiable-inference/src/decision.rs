//! Decision requests: typed questions over a state, answered by a pointer head without generation.

use crate::chain::{ChainCtx, shape};
use crate::draw;
use crate::manifest;
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_leaf_from_chunk, b3_verify};
use crate::opening::{DecisionLeaf, DecisionOpening};
use crate::verify::{Fail, ResponseCtx, TierCtx, expected_ubatch, open_entry};

/// Closes the plaintext of a decision request.
pub const REQUEST_MAGIC: &[u8; 4] = b"KXDQ";
/// Opens the plaintext of a decision answer.
pub const ANSWER_MAGIC: &[u8; 4] = b"KXDA";
pub const REQUEST_VERSION: u8 = 1;
pub const MAX_OPTIONS: usize = 255;
/// Draw index of the decision point.
pub const DECISION_POINT: u32 = crate::recur::RECUR_POINT + 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionOption {
    pub key: String,
    pub description: String,
}

/// Plaintext of a sealed decision request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionRequest {
    pub kind: QuestionType,
    pub state: String,
    pub instructions: String,
    pub options: Vec<DecisionOption>,
}

fn put_str(out: &mut Vec<u8>, s: &str, wide: bool) {
    if wide {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    } else {
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    }
    out.extend_from_slice(s.as_bytes());
}

fn take<'a>(b: &'a [u8], o: &mut usize, n: usize) -> Option<&'a [u8]> {
    let s = b.get(*o..o.checked_add(n)?)?;
    *o += n;
    Some(s)
}

fn get_str(b: &[u8], o: &mut usize, wide: bool) -> Option<String> {
    let n = if wide {
        u32::from_le_bytes(take(b, o, 4)?.try_into().ok()?) as usize
    } else {
        u16::from_le_bytes(take(b, o, 2)?.try_into().ok()?) as usize
    };
    String::from_utf8(take(b, o, n)?.to_vec()).ok()
}

impl DecisionRequest {
    /// `"KXDQ" [version] [type] [state: u32 len] [instructions: u32 len] [n] n × ([key: u16 len] [description: u16 len])`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(REQUEST_MAGIC);
        out.push(REQUEST_VERSION);
        out.push(self.kind as u8);
        put_str(&mut out, &self.state, true);
        put_str(&mut out, &self.instructions, true);
        out.push(self.options.len() as u8);
        for o in &self.options {
            put_str(&mut out, &o.key, false);
            put_str(&mut out, &o.description, false);
        }
        out
    }

    /// A well-formed request: known type, 1 to 255 options, a yes/no question with exactly the
    /// options `false` then `true`, a score question with a description per level.
    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.get(..4)? != REQUEST_MAGIC || *b.get(4)? != REQUEST_VERSION {
            return None;
        }
        let kind = match *b.get(5)? {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            2 => QuestionType::Noul,
            _ => return None,
        };
        let mut o = 6;
        let state = get_str(b, &mut o, true)?;
        let instructions = get_str(b, &mut o, true)?;
        let n = *take(b, &mut o, 1)?.first()? as usize;
        let mut options = Vec::with_capacity(n);
        for _ in 0..n {
            options.push(DecisionOption { key: get_str(b, &mut o, false)?, description: get_str(b, &mut o, false)? });
        }
        let ok = o == b.len()
            && !options.is_empty()
            && match kind {
                QuestionType::Noul => options.len() == 2 && options[0].key == "false" && options[1].key == "true",
                QuestionType::Score => options.iter().all(|x| !x.description.is_empty()),
                QuestionType::Choice => options.iter().all(|x| !x.key.is_empty()),
            };
        ok.then_some(Self { kind, state, instructions, options })
    }
}

/// One piece of a rendered decision prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    /// A special token, by its text.
    Special(&'static str),
    /// Plain text, tokenized without special tokens.
    Text(String),
    /// The request-bound prefix tokens.
    Prefix,
}

pub const KEV_STATE: &str = "<|fim_prefix|>";
pub const KEV_QUESTION: &str = "<|fim_middle|>";
pub const KEV_OPTION_OPEN: &str = "<|box_start|>";
/// Token an option is read at.
pub const KEV_MARKER: &str = "<|box_end|>";
/// Token the question is read at: the last prompt token.
pub const KEV_POINTER: &str = "<|fim_suffix|>";

/// Text of an option in the Kev prompt.
pub fn kev_option_text(kind: QuestionType, o: &DecisionOption) -> String {
    let desc = |s: String| if o.description.is_empty() { s } else { format!("{s}: {}", o.description) };
    match kind {
        QuestionType::Score => o.description.clone(),
        QuestionType::Choice => desc(o.key.clone()),
        QuestionType::Noul => desc(if o.key == "true" { "yes".into() } else { "no".into() }),
    }
}

/// The Kev prompt of a request: state, request prefix, question, one marked span per option, pointer.
pub fn kev_segments(r: &DecisionRequest) -> Vec<Segment> {
    let mut s = vec![Segment::Special(KEV_STATE), Segment::Prefix, Segment::Text("\n".into()), Segment::Text(r.state.clone()), Segment::Special(KEV_QUESTION), Segment::Text(r.instructions.clone())];
    for o in &r.options {
        s.push(Segment::Special(KEV_OPTION_OPEN));
        s.push(Segment::Text(kev_option_text(r.kind, o)));
        s.push(Segment::Special(KEV_MARKER));
    }
    s.push(Segment::Special(KEV_POINTER));
    s
}

/// Plaintext of a sealed decision answer: the prompt position of each option's marker and its score.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAnswer {
    pub positions: Vec<u32>,
    pub scores: Vec<f32>,
}

impl DecisionAnswer {
    /// Plaintext length of an answer over `n` options.
    pub fn len_for(n: usize) -> usize {
        4 + 1 + 8 * n
    }

    /// `"KXDA" [n] n × [position: u32] n × [score: f32]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::len_for(self.positions.len()));
        out.extend_from_slice(ANSWER_MAGIC);
        out.push(self.positions.len() as u8);
        for p in &self.positions {
            out.extend_from_slice(&p.to_le_bytes());
        }
        for s in &self.scores {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.get(..4)? != ANSWER_MAGIC {
            return None;
        }
        let n = *b.get(4)? as usize;
        if n == 0 || b.len() != Self::len_for(n) {
            return None;
        }
        let word = |i: usize| <[u8; 4]>::try_from(&b[5 + 4 * i..9 + 4 * i]).unwrap();
        let positions = (0..n).map(|i| u32::from_le_bytes(word(i))).collect();
        let scores = (0..n).map(|i| f32::from_le_bytes(word(n + i))).collect();
        Some(Self { positions, scores })
    }

    /// Commitment carried by the response in place of the generated-token root.
    pub fn commitment(plain: &[u8]) -> Hash {
        *blake3::hash(plain).as_bytes()
    }
}

/// Softmax of the scores at temperature `t`: the probabilities a client shows.
pub fn probabilities(scores: &[f32], t: f64) -> Vec<f64> {
    let m = scores.iter().fold(f64::NEG_INFINITY, |a, &s| a.max(s as f64));
    let e: Vec<f64> = scores.iter().map(|&s| libm::exp((s as f64 - m) / t)).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

/// Shape rule of a decision response: prefill only, no generation budget, one answer per option.
pub fn shape_ok(prompt_len: u32, answer_len: u32, max_tokens: u32, n_prompt: u32, n_pos: u32) -> bool {
    let n = (answer_len as usize).checked_sub(5).filter(|r| r % 8 == 0).map(|r| r / 8);
    max_tokens == 0
        && n_prompt > 0
        && n_pos == n_prompt
        && n_pos <= crate::shape::MAX_TRACE_POSITIONS
        && n_prompt >= crate::shape::min_prompt_positions(prompt_len)
        && n.is_some_and(|n| (1..=MAX_OPTIONS).contains(&n) && (n as u32) < n_prompt)
}

/// Consensus data of a decision tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionCtx {
    /// Manifest rank of the pointer head product (`m` outputs, the first half read at the pointer).
    pub head_rank: u32,
    pub marker_tok: u32,
    pub pointer_tok: u32,
}

/// Relative tolerance of a recomputed score.
pub const EPS_SCORE: f64 = 1e-6;

/// Score of an option from the pointer head outputs: the first half at the pointer against the
/// second half at the option's marker, over the square root of the half width.
pub fn score(q: &[f32], o: &[f32]) -> (f64, f64) {
    let scale = 1.0 / libm::sqrt(q.len() as f64);
    let (mut s, mut mag) = (0f64, 0f64);
    for (&a, &b) in q.iter().zip(o) {
        let p = a as f64 * b as f64;
        s += p;
        mag += p.abs();
    }
    (s * scale, mag * scale)
}

fn head_half(tier: &TierCtx, ctx: &ChainCtx, dctx: &DecisionCtx, resp: &ResponseCtx, leaf: &DecisionLeaf, t: u64, tok: u32, half: u64, m: u64) -> Result<Vec<f32>, Fail> {
    let total = 4 * m;
    let c = &leaf.y;
    if c.total_len as u64 != total || c.index as u64 != half {
        return Err(Fail::Shape);
    }
    let shape = shape(tier, ctx, tok);
    let y_leaf = b3_leaf_from_chunk(total, c.index as u64, &c.chunk, &c.chunk_path).ok_or(Fail::Trace)?;
    let pos_root = b3_fold(shape.n_leaves, 2 * dctx.head_rank as u64 + 1, &y_leaf, &c.leaf_path).ok_or(Fail::Trace)?;
    if leaf.n_ubatch != expected_ubatch(t, resp.n_prompt)
        || !b3_verify(&resp.trace_root, resp.n_pos as u64, t, &pos_root, &leaf.pos_path)
        || !b3_verify(&pos_root, shape.n_leaves, shape.meta_index, &shape.meta_hash(t, leaf.n_ubatch), &leaf.meta_path)
    {
        return Err(Fail::Trace);
    }
    crate::chain::finite_f32s(&c.chunk)
}

/// Verify the decision opening of a response whose generated-token root is `answer_root`: the
/// answer it carries, then the drawn option's score against the pointer head outputs at its
/// marker and at the pointer.
pub fn verify_decision_opening(
    tier: &TierCtx,
    ctx: &ChainCtx,
    dctx: &DecisionCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    answer_root: &Hash,
    op: &DecisionOpening,
) -> Result<(), Fail> {
    if DecisionAnswer::commitment(&op.answer) != *answer_root {
        return Err(Fail::Mismatch);
    }
    let a = DecisionAnswer::decode(&op.answer).ok_or(Fail::Shape)?;
    let n_prompt = resp.n_prompt as u64;
    if resp.n_pos != resp.n_prompt || n_prompt < 2 {
        return Err(Fail::Shape);
    }
    let pointer = n_prompt - 1;
    if !a.positions.windows(2).all(|w| w[0] < w[1]) || a.positions.last().is_none_or(|&p| p as u64 >= pointer) || a.scores.iter().any(|s| !s.is_finite()) {
        return Err(Fail::Mismatch);
    }
    let head = open_entry(tier, &op.head, dctx.head_rank as u64)?;
    let m = head.m;
    if head.role != manifest::ROLE_PRODUCT || head.n_exp != 1 || head.glu_form != manifest::FORM_NONE || 2 * m != B3_CHUNK as u64 {
        return Err(Fail::Manifest);
    }
    let i = (draw::point(seed, DECISION_POINT)[0] % a.positions.len() as u64) as usize;
    let t = a.positions[i] as u64;
    let o = head_half(tier, ctx, dctx, resp, &op.marker, t, dctx.marker_tok, 1, m)?;
    let q = head_half(tier, ctx, dctx, resp, &op.pointer, pointer, dctx.pointer_tok, 0, m)?;
    if q.len() as u64 != m / 2 || o.len() != q.len() {
        return Err(Fail::Shape);
    }
    let (s, mag) = score(&q, &o);
    let got = a.scores[i] as f64;
    let tol = EPS_SCORE * mag + f32::EPSILON as f64 * s.abs() + f32::MIN_POSITIVE as f64;
    if !(got - s).abs().partial_cmp(&tol).is_some_and(std::cmp::Ordering::is_le) {
        return Err(Fail::Mismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::ChainCtx;
    use crate::merkle::{b3_chunks_proof, b3_path, b3_root, rt_range_proof};
    use crate::opening::{EntryOpening, LeafChunk};
    use crate::verify::PosShape;

    const M: u64 = 512;
    const HEAD: u32 = 3;
    const N_PRODUCTS: u64 = 5;
    const MARKER: u32 = 900;
    const POINTER: u32 = 901;

    fn entry_bytes(m: u64, role: u8) -> Vec<u8> {
        let mut b = Vec::new();
        let name = b"cls.output.weight";
        b.extend_from_slice(&(name.len() as u16).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2560u64.to_le_bytes());
        b.extend_from_slice(&m.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 20]);
        b.extend_from_slice(&[role, 0, manifest::FORM_NONE]);
        b.extend_from_slice(&manifest::NONE.to_le_bytes());
        b.extend_from_slice(&manifest::NONE.to_le_bytes());
        b
    }

    struct Fixture {
        tier: TierCtx,
        ctx: ChainCtx,
        dctx: DecisionCtx,
        entries: Vec<Vec<u8>>,
        toks: Vec<u32>,
        ys: Vec<Vec<f32>>,
        pos_leaves: Vec<Vec<Hash>>,
        trace_root: Hash,
    }

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A prompt of `n` positions with markers at `markers` and the pointer last.
    fn fixture(n: usize, markers: &[usize], head_m: u64) -> Fixture {
        let mut entries: Vec<Vec<u8>> = (0..N_PRODUCTS + 2).map(|i| entry_bytes(64 + i, manifest::ROLE_PRODUCT)).collect();
        entries[HEAD as usize] = entry_bytes(head_m, manifest::ROLE_PRODUCT);
        let leaves: Vec<Hash> = entries.iter().map(|b| manifest::Entry::leaf(b)).collect();
        let m_t = rt_range_proof(&leaves, 0, 1).0;
        let tier = TierCtx { r_t: [0; 32], n_chunks: 1, m_t, n_entries: entries.len() as u64, n_products: N_PRODUCTS };
        let ctx = ChainCtx { n_layers: 1, layer_base: N_PRODUCTS as u32, embd_entry: 0, eps: 1e-6, embd_scale: 1.0, trim_last: false, n_used: 0, n_attn: 0, n_recur: 0, seq_heads: 0 };
        let dctx = DecisionCtx { head_rank: HEAD, marker_tok: MARKER, pointer_tok: POINTER };
        let mut toks: Vec<u32> = (0..n as u32).map(|i| 10 + i).collect();
        for &p in markers {
            toks[p] = MARKER;
        }
        toks[n - 1] = POINTER;
        let mut seed = 7u64;
        let mut ys = Vec::new();
        let mut pos_leaves = Vec::new();
        for (t, &tok) in toks.iter().enumerate() {
            let shape = shape(&tier, &ctx, tok);
            let y: Vec<f32> = (0..head_m).map(|_| lcg(&mut seed)).collect();
            let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut l: Vec<Hash> = (0..shape.n_leaves).map(|i| *blake3::hash(&[t as u8, i as u8, 99]).as_bytes()).collect();
            l[2 * HEAD as usize + 1] = *blake3::hash(&bytes).as_bytes();
            l[shape.meta_index as usize] = shape.meta_hash(t as u64, expected_ubatch(t as u64, n as u32));
            ys.push(y);
            pos_leaves.push(l);
        }
        let roots: Vec<Hash> = pos_leaves.iter().map(|l| b3_root(l)).collect();
        Fixture { tier, ctx, dctx, entries, toks, ys, trace_root: b3_root(&roots), pos_leaves }
    }

    impl Fixture {
        fn resp(&self) -> ResponseCtx {
            ResponseCtx { trace_root: self.trace_root, n_prompt: self.toks.len() as u32, n_pos: self.toks.len() as u32 }
        }

        fn leaf(&self, t: usize, half: u32) -> DecisionLeaf {
            let shape = shape(&self.tier, &self.ctx, self.toks[t]);
            let bytes: Vec<u8> = self.ys[t].iter().flat_map(|v| v.to_le_bytes()).collect();
            let roots: Vec<Hash> = self.pos_leaves.iter().map(|l| b3_root(l)).collect();
            let c = half as usize * B3_CHUNK;
            DecisionLeaf {
                n_ubatch: expected_ubatch(t as u64, self.toks.len() as u32),
                pos_path: b3_path(&roots, t),
                meta_path: b3_path(&self.pos_leaves[t], shape.meta_index as usize),
                y: LeafChunk {
                    total_len: bytes.len() as u32,
                    index: half,
                    chunk: bytes[c..(c + B3_CHUNK).min(bytes.len())].to_vec(),
                    chunk_path: b3_chunks_proof(&bytes, half as u64, 1),
                    leaf_path: b3_path(&self.pos_leaves[t], 2 * HEAD as usize + 1),
                },
            }
        }

        fn answer(&self, markers: &[usize]) -> DecisionAnswer {
            let p = self.toks.len() - 1;
            let h = (M / 2) as usize;
            let scores = markers.iter().map(|&m| score(&self.ys[p][..h], &self.ys[m][h..]).0 as f32).collect();
            DecisionAnswer { positions: markers.iter().map(|&m| m as u32).collect(), scores }
        }

        fn opening(&self, a: &DecisionAnswer, seed: &Hash) -> (DecisionOpening, Hash) {
            let plain = a.encode();
            let i = (draw::point(seed, DECISION_POINT)[0] % a.positions.len() as u64) as usize;
            let head = EntryOpening { index: HEAD, bytes: self.entries[HEAD as usize].clone(), path: rt_range_proof(&self.entries.iter().map(|b| manifest::Entry::leaf(b)).collect::<Vec<_>>(), HEAD as u64, 1).1 };
            let op = DecisionOpening {
                response_txid: [1; 32],
                roots: None,
                head,
                answer: plain.clone(),
                marker: self.leaf(a.positions[i] as usize, 1),
                pointer: self.leaf(self.toks.len() - 1, 0),
            };
            (op, DecisionAnswer::commitment(&plain))
        }

        fn verify(&self, op: &DecisionOpening, root: &Hash, seed: &Hash) -> Result<(), Fail> {
            verify_decision_opening(&self.tier, &self.ctx, &self.dctx, &self.resp(), seed, root, op)
        }
    }

    const MARKERS: [usize; 4] = [5, 9, 14, 20];

    #[test]
    fn honest_decision_openings_verify_on_every_option() {
        let f = fixture(24, &MARKERS, M);
        let a = f.answer(&MARKERS);
        let mut hit = [false; 4];
        for s in 0..64u8 {
            let seed = [s; 32];
            let (op, root) = f.opening(&a, &seed);
            assert_eq!(f.verify(&op, &root, &seed), Ok(()));
            hit[(draw::point(&seed, DECISION_POINT)[0] % 4) as usize] = true;
            assert_eq!(DecisionOpening::decode(&op.encode()), Some(op));
        }
        assert!(hit.iter().all(|&h| h));
    }

    #[test]
    fn a_wrong_score_is_rejected() {
        let f = fixture(24, &MARKERS, M);
        for s in 0..16u8 {
            let seed = [s; 32];
            let mut a = f.answer(&MARKERS);
            let i = (draw::point(&seed, DECISION_POINT)[0] % 4) as usize;
            a.scores[i] += 0.01 * a.scores[i].abs().max(1.0);
            let (op, root) = f.opening(&a, &seed);
            assert_eq!(f.verify(&op, &root, &seed), Err(Fail::Mismatch));
        }
    }

    #[test]
    fn the_answer_must_match_the_response_commitment() {
        let f = fixture(24, &MARKERS, M);
        let seed = [3; 32];
        let (op, _) = f.opening(&f.answer(&MARKERS), &seed);
        let mut other = f.answer(&MARKERS);
        other.scores[0] -= 1.0;
        assert_eq!(f.verify(&op, &DecisionAnswer::commitment(&other.encode()), &seed), Err(Fail::Mismatch));
    }

    #[test]
    fn a_position_that_is_not_a_marker_is_rejected() {
        let f = fixture(24, &MARKERS, M);
        let fake = [5, 9, 15, 20];
        for s in 0..32u8 {
            let seed = [s; 32];
            let i = (draw::point(&seed, DECISION_POINT)[0] % 4) as usize;
            let (op, root) = f.opening(&f.answer(&fake), &seed);
            let want = if fake[i] == MARKERS[i] { Ok(()) } else { Err(Fail::Trace) };
            assert_eq!(f.verify(&op, &root, &seed), want);
        }
    }

    #[test]
    fn positions_must_increase_and_stop_before_the_pointer() {
        let f = fixture(24, &MARKERS, M);
        let seed = [1; 32];
        for bad in [[9usize, 5, 14, 20], [5, 9, 9, 20], [5, 9, 14, 23]] {
            let (op, root) = f.opening(&f.answer(&bad), &seed);
            assert_eq!(f.verify(&op, &root, &seed), Err(Fail::Mismatch));
        }
    }

    #[test]
    fn a_head_of_another_width_is_refused() {
        let f = fixture(24, &MARKERS, 2 * M);
        let seed = [2; 32];
        let (op, root) = f.opening(&f.answer(&MARKERS), &seed);
        assert_eq!(f.verify(&op, &root, &seed), Err(Fail::Manifest));
    }

    #[test]
    fn a_generated_position_is_refused() {
        let f = fixture(24, &MARKERS, M);
        let seed = [4; 32];
        let (op, root) = f.opening(&f.answer(&MARKERS), &seed);
        let resp = ResponseCtx { trace_root: f.trace_root, n_prompt: 23, n_pos: 24 };
        assert_eq!(verify_decision_opening(&f.tier, &f.ctx, &f.dctx, &resp, &seed, &root, &op), Err(Fail::Shape));
    }

    #[test]
    fn formats_round_trip() {
        let r = DecisionRequest {
            kind: QuestionType::Choice,
            state: "ticket: charged twice".into(),
            instructions: "Which team?".into(),
            options: vec![DecisionOption { key: "billing".into(), description: "refunds".into() }, DecisionOption { key: "tech".into(), description: String::new() }],
        };
        assert_eq!(DecisionRequest::decode(&r.encode()), Some(r.clone()));
        let noul = DecisionRequest { kind: QuestionType::Noul, options: vec![DecisionOption { key: "true".into(), description: String::new() }, DecisionOption { key: "false".into(), description: String::new() }], ..r.clone() };
        assert_eq!(DecisionRequest::decode(&noul.encode()), None);
        let mut trailing = r.encode();
        trailing.push(0);
        assert_eq!(DecisionRequest::decode(&trailing), None);
        let a = DecisionAnswer { positions: vec![3, 8], scores: vec![1.5, -2.0] };
        assert_eq!(a.encode().len(), DecisionAnswer::len_for(2));
        assert_eq!(DecisionAnswer::decode(&a.encode()), Some(a));
        let p = probabilities(&[1.0, 1.0], 2.4);
        assert!((p[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_kev_prompt_marks_each_option_and_ends_on_the_pointer() {
        let r = DecisionRequest {
            kind: QuestionType::Noul,
            state: "s".into(),
            instructions: "q".into(),
            options: vec![DecisionOption { key: "false".into(), description: String::new() }, DecisionOption { key: "true".into(), description: "it is".into() }],
        };
        let s = kev_segments(&r);
        assert_eq!(&s[..3], &[Segment::Special(KEV_STATE), Segment::Prefix, Segment::Text("\n".into())]);
        assert_eq!(s.iter().filter(|x| **x == Segment::Special(KEV_MARKER)).count(), 2);
        assert!(s.contains(&Segment::Text("no".into())) && s.contains(&Segment::Text("yes: it is".into())));
        assert_eq!(s.last(), Some(&Segment::Special(KEV_POINTER)));
    }

    #[test]
    fn decision_shapes() {
        let len = |n| DecisionAnswer::len_for(n) as u32;
        assert!(shape_ok(1_000, len(4), 0, 300, 300));
        assert!(!shape_ok(1_000, len(4), 16, 300, 300));
        assert!(!shape_ok(1_000, len(4), 0, 300, 301));
        assert!(!shape_ok(1_000, len(4) + 1, 0, 300, 300));
        assert!(!shape_ok(1_000, 5, 0, 300, 300));
        assert!(!shape_ok(16_000, len(4), 0, 300, 300));
        assert!(!shape_ok(1_000, len(4), 0, 4, 4));
        let _ = PosShape::stage1(1);
    }
}
