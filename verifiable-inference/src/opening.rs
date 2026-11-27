//! Opening payload format.

use crate::merkle::Hash;

pub const VERSION: u8 = 1;
/// Version 1 plus a prefix point after the points.
pub const VERSION_PREFIX: u8 = 2;
/// Stage-2 opening: each point carries its token and input link, then the prefix and residual points.
pub const VERSION_CHAIN: u8 = 3;
/// Envelope naming the seed block an opening was built for: `[4] [response id] [seed block]`, then
/// the inner body's version and the rest of that body after its response id.
pub const VERSION_SEEDED: u8 = 4;
/// Attention opening of a response, sent apart from its trace opening.
pub const VERSION_ATTN: u8 = 5;
/// Recurrent opening of a response, sent apart from its trace opening.
pub const VERSION_RECUR: u8 = 6;
/// Stage-2 opening of a trace that commits sequence leaves: the response id is followed by the
/// position and sequence roots.
pub const VERSION_ROOTS: u8 = 7;
/// Residual opening of a response, sent apart from its trace opening.
pub const VERSION_RESID: u8 = 8;
/// Decision opening of a response, sent apart from its trace opening.
pub const VERSION_DECISION: u8 = 9;
/// Selection opening of a response, sent apart from its trace opening.
pub const VERSION_SEL: u8 = 10;
/// Compressed row opening of a response, sent apart from its trace opening.
pub const VERSION_COMP: u8 = 11;
/// Attention opening of a compressed attention layer, sent as the attention piece.
pub const VERSION_ATTN_COMP: u8 = 12;
/// Selection opening of a compressed attention tier, sent as the selection piece.
pub const VERSION_COMP_SEL: u8 = 13;
/// N-gram lookup residual opening of a response, sent apart from its trace opening.
pub const VERSION_ENGRAM: u8 = 14;
/// Part of a stage-2 trace opening: some drawn points with their draw index, and the prefix point
/// in one of the parts.
pub const VERSION_CHAIN_PART: u8 = 15;
/// Half of a selection opening: the drawn pools of one side only, after the response id a part byte.
pub const VERSION_SEL_PART: u8 = 16;
/// Selection part carrying the pools drawn among the selected ones.
pub const SEL_PART_SELECTED: u8 = 1;
/// Selection part carrying the pools drawn among the others.
pub const SEL_PART_OUTSIDE: u8 = 2;
/// Largest opening body.
pub const MAX_BODY_LEN: usize = 112 * 1024;
/// `[escrow_pubkey: 32] [signature: 64]` after the body.
pub const SIGNATURE_LEN: usize = 32 + 64;
/// Smallest opening payload: version, response id, point count and the signature.
pub const MIN_PAYLOAD_LEN: usize = 1 + 32 + 1 + SIGNATURE_LEN;
pub const MAX_PAYLOAD_LEN: usize = MAX_BODY_LEN + SIGNATURE_LEN;

/// Signed opening payload split into its body, signer key and signature.
pub struct SignedPayload<'a> {
    pub body: &'a [u8],
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

impl<'a> SignedPayload<'a> {
    pub fn split(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < MIN_PAYLOAD_LEN || payload.len() > MAX_PAYLOAD_LEN {
            return None;
        }
        let (body, tail) = payload.split_at(payload.len() - SIGNATURE_LEN);
        Some(Self { body, escrow_pubkey: tail[..32].try_into().ok()?, signature: tail[32..].try_into().ok()? })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntryOpening {
    pub index: u32,
    pub bytes: Vec<u8>,
    pub path: Vec<Hash>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkRange {
    pub first_chunk: u64,
    pub chunks: Vec<u8>,
    pub proof: Vec<Hash>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Point {
    pub n_ubatch: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub entry: EntryOpening,
    pub partner: Option<EntryOpening>,
    pub bias: Option<(EntryOpening, ChunkRange)>,
    pub rows: Vec<ChunkRange>,
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub y_total_len: u32,
    pub y_chunk_index: u32,
    pub y_chunk: Vec<u8>,
    pub y_chunk_path: Vec<Hash>,
    pub y_path: Vec<Hash>,
}

/// The layer-0 check of one prefix position: the input of the first product against the
/// embedding of the token the request id puts there.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefixPoint {
    pub n_ubatch: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub entry: EntryOpening,
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub embd: (EntryOpening, ChunkRange),
    pub norm: (EntryOpening, ChunkRange),
}

/// Last chunk of an expert product's `Y` leaf, which ends with the slot expert ids.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertIds {
    pub chunk: Vec<u8>,
    pub chunk_path: Vec<Hash>,
}

/// One 1 KiB chunk of a committed leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafChunk {
    pub total_len: u32,
    pub index: u32,
    pub chunk: Vec<u8>,
    pub chunk_path: Vec<Hash>,
    pub leaf_path: Vec<Hash>,
}

/// A whole committed leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafBytes {
    pub bytes: Vec<u8>,
    pub leaf_path: Vec<Hash>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinkOpening {
    None,
    Norm { src: LeafBytes, norm: (EntryOpening, ChunkRange) },
    Token { embd: (EntryOpening, ChunkRange), norm: (EntryOpening, ChunkRange) },
    Source { src: EntryOpening, y: LeafChunk, ids: Option<ExpertIds> },
    /// The whole residual streams a mix product reads, or the token embedding at layer 0.
    HcIn { layer: EntryOpening, src: HcSource },
    /// The streams, the mix output and the mix weights the first product of a half reads.
    HcPre { layer: EntryOpening, src: HcSource, mixes: LeafBytes, base: (EntryOpening, ChunkRange), norm: (EntryOpening, ChunkRange) },
    /// The layer of an n-gram lookup, the tokens before the position (their meta leaves), the
    /// token map entries of the n-gram (the position's own first), and the table row read.
    Engram { layer: EntryOpening, prev: Vec<PrevToken>, map: Vec<ChunkRange>, row: ChunkRange },
}

/// A committed token of an earlier position: its meta leaf's path and the position's path.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrevToken {
    pub tok: u32,
    pub meta_path: Vec<Hash>,
    pub pos_path: Vec<Hash>,
}

/// Residual streams of a hyper-connection link: committed, or the token embedding repeated.
#[derive(Debug, Clone, PartialEq)]
pub enum HcSource {
    Streams(LeafBytes),
    Token((EntryOpening, ChunkRange)),
}

/// One chunk of each source stream and the mix a hyper-connection residual is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct HcAdd {
    /// The chunk of the half's output at the same offset as the residual chunk.
    pub o: LeafChunk,
    /// The same chunk of every source stream; empty when the source is the token embedding.
    pub streams: Vec<LeafChunk>,
    pub mixes: LeafBytes,
    pub base: (EntryOpening, ChunkRange),
    /// The routed experts of a MoE half, whose output adds to `o`.
    pub moe: Option<Box<HcMoe>>,
}

/// The routed experts of a hyper-connection MoE half, read from the router input as committed by
/// the gate product of the experts.
#[derive(Debug, Clone, PartialEq)]
pub struct HcMoe {
    pub scores: LeafBytes,
    pub bias: (EntryOpening, ChunkRange),
    pub ids: ExpertIds,
    pub slots: Vec<LeafChunk>,
    pub router_row: (EntryOpening, ChunkRange),
    /// The gate product of the experts and its whole input leaf.
    pub gate: EntryOpening,
    pub x: LeafBytes,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResidSrc {
    Chunk(LeafChunk),
    Token(EntryOpening, ChunkRange),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResidAdd {
    Chunk(LeafChunk),
    Normed { y: LeafBytes, norm: (EntryOpening, ChunkRange) },
    Moe(Box<MoeAdd>),
    Hc(Box<HcAdd>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeAdd {
    pub scores: LeafBytes,
    pub bias: (EntryOpening, ChunkRange),
    pub ids: ExpertIds,
    pub slots: Vec<LeafChunk>,
    pub shexp: LeafChunk,
    pub router_row: (EntryOpening, ChunkRange),
    /// The whole attention residual of the layer.
    pub resid: LeafBytes,
    /// The whole FFN norm weight of the layer.
    pub norm: (EntryOpening, ChunkRange),
    /// The whole gate row of a gated shared expert.
    pub shexp_gate: Option<(EntryOpening, ChunkRange)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidPoint {
    pub n_ubatch: u32,
    pub tok: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub layer: EntryOpening,
    pub product: EntryOpening,
    pub dst: LeafChunk,
    pub src: ResidSrc,
    pub add: ResidAdd,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Opening {
    pub response_txid: Hash,
    pub points: Vec<Point>,
    pub prefix: Option<PrefixPoint>,
    /// Stage 2: `(token, link)` of each point.
    pub links: Vec<(u32, LinkOpening)>,
    /// Stage 2: expert ids of each point drawn on an expert product.
    pub experts: Vec<Option<ExpertIds>>,
    pub resid: Option<ResidPoint>,
    /// Position and sequence roots of a trace that commits sequence leaves.
    pub roots: Option<(Hash, Hash)>,
}

/// The residual point of a response, sent apart from its trace opening.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidOpening {
    pub response_txid: Hash,
    /// Position and sequence roots of a trace that commits sequence leaves.
    pub roots: Option<(Hash, Hash)>,
    pub resid: ResidPoint,
    /// One row of the product whose output the residual adds.
    pub feed: FeedRow,
    /// One row of a routed expert, when the residual adds routed experts.
    pub expert: Option<ExpertRow>,
}

/// Input and one weight row of the product feeding a residual.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeedRow {
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub row: ChunkRange,
    pub bias: Option<(EntryOpening, ChunkRange)>,
}

/// Input and one weight row of a routed expert feeding a residual.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertRow {
    pub entry: EntryOpening,
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub row: ChunkRange,
}

/// Attention point of a response at one phase of the block tree of its attended positions: the
/// drawn query position, the cache of the positions a phase opens as spans of the head's sequence
/// leaves, and softmax partials.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AttnOpening {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub q_entry: EntryOpening,
    pub o_entry: EntryOpening,
    /// Weights of the query norm, key norm and RoPE frequencies the layer declares, in that order.
    pub norms: Vec<(EntryOpening, ChunkRange)>,
    pub pos_path: Vec<Hash>,
    pub q: Vec<LeafChunk>,
    pub out: LeafBytes,
    pub kcache: crate::seq::SeqSpan,
    /// Empty for a latent-cache layer.
    pub vcache: crate::seq::SeqSpan,
    /// Path of the anchor position, and chunks of the key and value projection outputs there.
    pub anchor_pos_path: Vec<Hash>,
    pub anchor_k: Vec<LeafChunk>,
    pub anchor_v: Vec<LeafChunk>,
    /// Chunks of the latent query of the drawn head (latent-cache layers).
    pub absorb: Vec<LeafChunk>,
    /// Child drawn after each earlier phase: empty opens the output, a full path the cache of one
    /// leaf.
    pub path: Vec<u64>,
    /// Partial of the opened node as the previous phase committed it, and its path there.
    pub parent: Vec<u8>,
    pub parent_path: Vec<Hash>,
    /// Partials of the children of the opened node.
    pub partials: Vec<u8>,
    /// Committed log mass of the drawn query head at the drawn position (output phase).
    pub lse: crate::seq::SeqSpan,
    /// The selection the drawn position attends, on a sparse attention layer (every phase).
    pub sel: Option<SelOpening>,
}

/// A selection leaf of a position: the chosen pools, and its paths to the position root.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelOpening {
    pub bytes: Vec<u8>,
    pub leaf_path: Vec<Hash>,
    pub pos_path: Vec<Hash>,
}

/// The projection outputs read by the convolutions at one position of the anchor window.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecurTap {
    pub pos_path: Vec<Hash>,
    pub q: LeafChunk,
    pub k: LeafChunk,
    pub v: LeafChunk,
}

/// Recurrent point of a response: one segment of one head between two state checkpoints.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecurOpening {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub out_entry: EntryOpening,
    pub decay_entry: EntryOpening,
    /// Convolution weights of the head's query, key and value channels, then its decay bias,
    /// decay scale and the output norm.
    pub weights: Vec<(EntryOpening, ChunkRange)>,
    pub records: crate::seq::SeqSpan,
    /// Drawn state rows at the segment start (none for the first segment) and end (none for a
    /// partial last segment).
    pub start_rows: crate::seq::SeqSpan,
    pub end_rows: crate::seq::SeqSpan,
    /// Anchor window, oldest first, ending at the anchor position.
    pub taps: Vec<RecurTap>,
    pub beta: LeafChunk,
    pub decay: LeafChunk,
    pub gate: LeafChunk,
    pub out: LeafBytes,
}

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes32(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.0.extend_from_slice(b);
    }
    fn hashes(&mut self, h: &[Hash]) {
        self.u8(h.len() as u8);
        for x in h {
            self.0.extend_from_slice(x);
        }
    }
    fn entry(&mut self, e: &EntryOpening) {
        self.u32(e.index);
        self.u16(e.bytes.len() as u16);
        self.0.extend_from_slice(&e.bytes);
        self.hashes(&e.path);
    }
    fn range(&mut self, r: &ChunkRange) {
        self.u64(r.first_chunk);
        self.u32((r.chunks.len() / 32) as u32);
        self.0.extend_from_slice(&r.chunks);
        self.hashes(&r.proof);
    }
    fn weights(&mut self, w: &(EntryOpening, ChunkRange)) {
        self.entry(&w.0);
        self.range(&w.1);
    }
    fn leaf_chunk(&mut self, c: &LeafChunk) {
        self.u32(c.total_len);
        self.u32(c.index);
        self.bytes32(&c.chunk);
        self.hashes(&c.chunk_path);
        self.hashes(&c.leaf_path);
    }
    fn leaf_bytes(&mut self, l: &LeafBytes) {
        self.bytes32(&l.bytes);
        self.hashes(&l.leaf_path);
    }
    fn seq_span(&mut self, s: &crate::seq::SeqSpan) {
        self.u32(s.first_chunk);
        self.bytes32(&s.bytes);
        self.hashes(&s.chunk_proof);
        self.hashes(&s.leaf_path);
    }
    fn link(&mut self, l: &LinkOpening) {
        match l {
            LinkOpening::None => self.u8(0),
            LinkOpening::Norm { src, norm } => {
                self.u8(1);
                self.leaf_bytes(src);
                self.weights(norm);
            }
            LinkOpening::Token { embd, norm } => {
                self.u8(2);
                self.weights(embd);
                self.weights(norm);
            }
            LinkOpening::Source { src, y, ids } => {
                self.u8(3);
                self.entry(src);
                self.leaf_chunk(y);
                match ids {
                    Some(e) => {
                        self.u8(1);
                        self.bytes32(&e.chunk);
                        self.hashes(&e.chunk_path);
                    }
                    None => self.u8(0),
                }
            }
            LinkOpening::HcIn { layer, src } => {
                self.u8(4);
                self.entry(layer);
                self.hc_source(src);
            }
            LinkOpening::HcPre { layer, src, mixes, base, norm } => {
                self.u8(5);
                self.entry(layer);
                self.hc_source(src);
                self.leaf_bytes(mixes);
                self.weights(base);
                self.weights(norm);
            }
            LinkOpening::Engram { layer, prev, map, row } => {
                self.u8(6);
                self.entry(layer);
                self.u8(prev.len() as u8);
                for p in prev {
                    self.u32(p.tok);
                    self.hashes(&p.meta_path);
                    self.hashes(&p.pos_path);
                }
                self.u8(map.len() as u8);
                for m in map {
                    self.range(m);
                }
                self.range(row);
            }
        }
    }
    fn hc_source(&mut self, s: &HcSource) {
        match s {
            HcSource::Streams(b) => {
                self.u8(0);
                self.leaf_bytes(b);
            }
            HcSource::Token(w) => {
                self.u8(1);
                self.weights(w);
            }
        }
    }
    fn resid(&mut self, p: &ResidPoint) {
        self.u32(p.n_ubatch);
        self.u32(p.tok);
        self.hashes(&p.pos_path);
        self.hashes(&p.meta_path);
        self.entry(&p.layer);
        self.entry(&p.product);
        self.leaf_chunk(&p.dst);
        match &p.src {
            ResidSrc::Chunk(c) => {
                self.u8(0);
                self.leaf_chunk(c);
            }
            ResidSrc::Token(e, r) => {
                self.u8(1);
                self.entry(e);
                self.range(r);
            }
        }
        match &p.add {
            ResidAdd::Chunk(c) => {
                self.u8(0);
                self.leaf_chunk(c);
            }
            ResidAdd::Normed { y, norm } => {
                self.u8(1);
                self.leaf_bytes(y);
                self.weights(norm);
            }
            ResidAdd::Moe(m) => {
                self.u8(if m.shexp_gate.is_some() { 3 } else { 2 });
                self.leaf_bytes(&m.scores);
                self.weights(&m.bias);
                self.bytes32(&m.ids.chunk);
                self.hashes(&m.ids.chunk_path);
                self.u8(m.slots.len() as u8);
                for c in &m.slots {
                    self.leaf_chunk(c);
                }
                self.leaf_chunk(&m.shexp);
                self.weights(&m.router_row);
                self.leaf_bytes(&m.resid);
                self.weights(&m.norm);
                if let Some(g) = &m.shexp_gate {
                    self.weights(g);
                }
            }
            ResidAdd::Hc(h) => {
                self.u8(4);
                self.leaf_chunk(&h.o);
                self.u8(h.streams.len() as u8);
                for c in &h.streams {
                    self.leaf_chunk(c);
                }
                self.leaf_bytes(&h.mixes);
                self.weights(&h.base);
                match &h.moe {
                    Some(m) => {
                        self.u8(1);
                        self.leaf_bytes(&m.scores);
                        self.weights(&m.bias);
                        self.bytes32(&m.ids.chunk);
                        self.hashes(&m.ids.chunk_path);
                        self.u8(m.slots.len() as u8);
                        for c in &m.slots {
                            self.leaf_chunk(c);
                        }
                        self.weights(&m.router_row);
                        self.entry(&m.gate);
                        self.leaf_bytes(&m.x);
                    }
                    None => self.u8(0),
                }
            }
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    o: usize,
}

impl R<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.o..self.o.checked_add(n)?)?;
        self.o += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes32(&mut self) -> Option<Vec<u8>> {
        let n = self.u32()? as usize;
        Some(self.take(n)?.to_vec())
    }
    fn hashes(&mut self) -> Option<Vec<Hash>> {
        let n = self.u8()? as usize;
        (0..n).map(|_| self.take(32).map(|s| s.try_into().unwrap())).collect()
    }
    fn entry(&mut self) -> Option<EntryOpening> {
        let index = self.u32()?;
        let n = self.u16()? as usize;
        let bytes = self.take(n)?.to_vec();
        Some(EntryOpening { index, bytes, path: self.hashes()? })
    }
    fn range(&mut self) -> Option<ChunkRange> {
        let first_chunk = self.u64()?;
        let n = self.u32()? as usize;
        let chunks = self.take(n.checked_mul(32)?)?.to_vec();
        Some(ChunkRange { first_chunk, chunks, proof: self.hashes()? })
    }
    fn weights(&mut self) -> Option<(EntryOpening, ChunkRange)> {
        Some((self.entry()?, self.range()?))
    }
    fn leaf_chunk(&mut self) -> Option<LeafChunk> {
        Some(LeafChunk {
            total_len: self.u32()?,
            index: self.u32()?,
            chunk: self.bytes32()?,
            chunk_path: self.hashes()?,
            leaf_path: self.hashes()?,
        })
    }
    fn leaf_bytes(&mut self) -> Option<LeafBytes> {
        Some(LeafBytes { bytes: self.bytes32()?, leaf_path: self.hashes()? })
    }
    fn seq_span(&mut self) -> Option<crate::seq::SeqSpan> {
        Some(crate::seq::SeqSpan { first_chunk: self.u32()?, bytes: self.bytes32()?, chunk_proof: self.hashes()?, leaf_path: self.hashes()? })
    }
    fn link(&mut self) -> Option<LinkOpening> {
        Some(match self.u8()? {
            0 => LinkOpening::None,
            1 => LinkOpening::Norm { src: self.leaf_bytes()?, norm: self.weights()? },
            2 => LinkOpening::Token { embd: self.weights()?, norm: self.weights()? },
            3 => LinkOpening::Source {
                src: self.entry()?,
                y: self.leaf_chunk()?,
                ids: match self.u8()? {
                    0 => None,
                    1 => Some(ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? }),
                    _ => return None,
                },
            },
            4 => LinkOpening::HcIn { layer: self.entry()?, src: self.hc_source()? },
            5 => LinkOpening::HcPre { layer: self.entry()?, src: self.hc_source()?, mixes: self.leaf_bytes()?, base: self.weights()?, norm: self.weights()? },
            6 => {
                let layer = self.entry()?;
                let np = self.u8()? as usize;
                let prev = (0..np).map(|_| Some(PrevToken { tok: self.u32()?, meta_path: self.hashes()?, pos_path: self.hashes()? })).collect::<Option<Vec<_>>>()?;
                let nm = self.u8()? as usize;
                let map = (0..nm).map(|_| self.range()).collect::<Option<Vec<_>>>()?;
                LinkOpening::Engram { layer, prev, map, row: self.range()? }
            }
            _ => return None,
        })
    }
    fn hc_source(&mut self) -> Option<HcSource> {
        Some(match self.u8()? {
            0 => HcSource::Streams(self.leaf_bytes()?),
            1 => HcSource::Token(self.weights()?),
            _ => return None,
        })
    }
    fn resid(&mut self) -> Option<ResidPoint> {
        let n_ubatch = self.u32()?;
        let tok = self.u32()?;
        let pos_path = self.hashes()?;
        let meta_path = self.hashes()?;
        let layer = self.entry()?;
        let product = self.entry()?;
        let dst = self.leaf_chunk()?;
        let src = match self.u8()? {
            0 => ResidSrc::Chunk(self.leaf_chunk()?),
            1 => ResidSrc::Token(self.entry()?, self.range()?),
            _ => return None,
        };
        let add = match self.u8()? {
            0 => ResidAdd::Chunk(self.leaf_chunk()?),
            1 => ResidAdd::Normed { y: self.leaf_bytes()?, norm: self.weights()? },
            tag @ (2 | 3) => {
                let scores = self.leaf_bytes()?;
                let bias = self.weights()?;
                let ids = ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? };
                let n = self.u8()? as usize;
                let slots = (0..n).map(|_| self.leaf_chunk()).collect::<Option<Vec<_>>>()?;
                ResidAdd::Moe(Box::new(MoeAdd {
                    scores,
                    bias,
                    ids,
                    slots,
                    shexp: self.leaf_chunk()?,
                    router_row: self.weights()?,
                    resid: self.leaf_bytes()?,
                    norm: self.weights()?,
                    shexp_gate: if tag == 3 { Some(self.weights()?) } else { None },
                }))
            }
            4 => {
                let o = self.leaf_chunk()?;
                let n = self.u8()? as usize;
                let streams = (0..n).map(|_| self.leaf_chunk()).collect::<Option<Vec<_>>>()?;
                let (mixes, base) = (self.leaf_bytes()?, self.weights()?);
                let moe = match self.u8()? {
                    0 => None,
                    1 => {
                        let (scores, bias) = (self.leaf_bytes()?, self.weights()?);
                        let ids = ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? };
                        let n = self.u8()? as usize;
                        let slots = (0..n).map(|_| self.leaf_chunk()).collect::<Option<Vec<_>>>()?;
                        Some(Box::new(HcMoe { scores, bias, ids, slots, router_row: self.weights()?, gate: self.entry()?, x: self.leaf_bytes()? }))
                    }
                    _ => return None,
                };
                ResidAdd::Hc(Box::new(HcAdd { o, streams, mixes, base, moe }))
            }
            _ => return None,
        };
        Some(ResidPoint { n_ubatch, tok, pos_path, meta_path, layer, product, dst, src, add })
    }
}

impl Opening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        let chain = !self.links.is_empty() || self.resid.is_some();
        w.u8(if chain && self.roots.is_some() {
            VERSION_ROOTS
        } else if chain {
            VERSION_CHAIN
        } else if self.prefix.is_some() {
            VERSION_PREFIX
        } else {
            VERSION
        });
        w.0.extend_from_slice(&self.response_txid);
        if let (true, Some((p, q))) = (chain, &self.roots) {
            w.0.extend_from_slice(p);
            w.0.extend_from_slice(q);
        }
        w.u8(self.points.len() as u8);
        for p in &self.points {
            w.point(p);
        }
        if chain {
            for (i, (tok, link)) in self.links.iter().enumerate() {
                w.u32(*tok);
                w.link(link);
                w.expert_ids(self.experts.get(i).and_then(Option::as_ref));
            }
        }
        if let Some(p) = &self.prefix {
            w.prefix_point(p);
        }
        if let Some(p) = &self.resid {
            w.resid(p);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<Opening> {
        if b.first() == Some(&VERSION_SEEDED) {
            let inner = *b.get(65)?;
            if inner == VERSION_SEEDED {
                return None;
            }
            let mut v = Vec::with_capacity(b.len() - 32);
            v.push(inner);
            v.extend_from_slice(&b[1..33]);
            v.extend_from_slice(&b[66..]);
            return Self::decode(&v);
        }
        let mut r = R { b, o: 0 };
        let version = r.u8()?;
        if !matches!(version, VERSION | VERSION_PREFIX | VERSION_CHAIN | VERSION_ROOTS) {
            return None;
        }
        let chain = matches!(version, VERSION_CHAIN | VERSION_ROOTS);
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let roots = if version == VERSION_ROOTS { Some((r.take(32)?.try_into().ok()?, r.take(32)?.try_into().ok()?)) } else { None };
        let n = r.u8()? as usize;
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            points.push(r.point()?);
        }
        let mut experts = Vec::new();
        let links = if chain {
            let mut links = Vec::with_capacity(n);
            for _ in 0..n {
                links.push((r.u32()?, r.link()?));
                experts.push(r.expert_ids()?);
            }
            links
        } else {
            Vec::new()
        };
        let prefix = if version >= VERSION_PREFIX { Some(r.prefix_point()?) } else { None };
        let resid = if chain && r.o < b.len() { Some(r.resid()?) } else { None };
        (r.o == b.len()).then_some(Opening { response_txid, points, prefix, links, experts, resid, roots })
    }
}

impl ResidOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_RESID);
        w.0.extend_from_slice(&self.response_txid);
        match &self.roots {
            Some((p, q)) => {
                w.u8(1);
                w.0.extend_from_slice(p);
                w.0.extend_from_slice(q);
            }
            None => w.u8(0),
        }
        w.resid(&self.resid);
        w.bytes32(&self.feed.x);
        w.hashes(&self.feed.x_path);
        w.range(&self.feed.row);
        match &self.feed.bias {
            Some(b) => {
                w.u8(1);
                w.weights(b);
            }
            None => w.u8(0),
        }
        match &self.expert {
            Some(e) => {
                w.u8(1);
                w.entry(&e.entry);
                w.bytes32(&e.x);
                w.hashes(&e.x_path);
                w.range(&e.row);
            }
            None => w.u8(0),
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<ResidOpening> {
        if b.first() != Some(&VERSION_RESID) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let roots = match r.u8()? {
            0 => None,
            1 => Some((r.take(32)?.try_into().ok()?, r.take(32)?.try_into().ok()?)),
            _ => return None,
        };
        let resid = r.resid()?;
        let feed = FeedRow {
            x: r.bytes32()?,
            x_path: r.hashes()?,
            row: r.range()?,
            bias: match r.u8()? {
                0 => None,
                1 => Some(r.weights()?),
                _ => return None,
            },
        };
        let expert = match r.u8()? {
            0 => None,
            1 => Some(ExpertRow { entry: r.entry()?, x: r.bytes32()?, x_path: r.hashes()?, row: r.range()? }),
            _ => return None,
        };
        (r.o == b.len()).then_some(ResidOpening { response_txid, roots, resid, feed, expert })
    }
}

/// The seed block a seeded opening body names; `None` for an unseeded body.
pub fn seed_block(body: &[u8]) -> Option<Hash> {
    if body.first() != Some(&VERSION_SEEDED) {
        return None;
    }
    body.get(33..65)?.try_into().ok()
}

/// Whether `body`, seeded or not, decodes as an audit opening of any piece.
pub fn is_audit_body(body: &[u8]) -> bool {
    Opening::decode(body).is_some()
        || inner_body(body).is_some_and(|b| {
            AttnOpening::decode(&b).is_some()
                || RecurOpening::decode(&b).is_some()
                || ResidOpening::decode(&b).is_some()
                || ChainPart::decode(&b).is_some()
                || DecisionOpening::decode(&b).is_some()
                || SelOpeningPoint::decode(&b).is_some()
                || CompOpeningPoint::decode(&b).is_some()
                || CompAttnOpening::decode(&b).is_some()
                || CompSelOpening::decode(&b).is_some()
                || EngramOpening::decode(&b).is_some()
        })
}

/// The body inside a seeded envelope, or `body` itself when unseeded.
pub fn inner_body(body: &[u8]) -> Option<Vec<u8>> {
    if body.first() != Some(&VERSION_SEEDED) {
        return Some(body.to_vec());
    }
    let inner = *body.get(65)?;
    if inner == VERSION_SEEDED {
        return None;
    }
    let mut v = Vec::with_capacity(body.len() - 32);
    v.push(inner);
    v.extend_from_slice(&body[1..33]);
    v.extend_from_slice(&body[66..]);
    Some(v)
}

/// Path of an attention opening body, seeded or not; `None` for another body.
pub fn attn_path(body: &[u8]) -> Option<Vec<u64>> {
    let b = inner_body(body)?;
    AttnOpening::decode(&b).map(|o| o.path).or_else(|| CompAttnOpening::decode(&b).map(|o| o.path))
}

/// Wraps an unseeded opening body into the seeded envelope naming `seed_block`.
pub fn seeded(body: &[u8], seed_block: &Hash) -> Option<Vec<u8>> {
    let version = *body.first()?;
    if !matches!(version, VERSION | VERSION_PREFIX | VERSION_CHAIN | VERSION_ATTN | VERSION_RECUR | VERSION_ROOTS | VERSION_RESID | VERSION_DECISION | VERSION_SEL | VERSION_COMP | VERSION_ATTN_COMP | VERSION_COMP_SEL | VERSION_ENGRAM | VERSION_CHAIN_PART | VERSION_SEL_PART) || body.len() < 33 {
        return None;
    }
    let mut v = Vec::with_capacity(body.len() + 33);
    v.push(VERSION_SEEDED);
    v.extend_from_slice(&body[1..33]);
    v.extend_from_slice(seed_block);
    v.push(version);
    v.extend_from_slice(&body[33..]);
    Some(v)
}

impl AttnOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_ATTN);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.entry(&self.q_entry);
        w.entry(&self.o_entry);
        w.u8(self.norms.len() as u8);
        for n in &self.norms {
            w.weights(n);
        }
        w.hashes(&self.pos_path);
        w.u8(self.q.len() as u8);
        for c in &self.q {
            w.leaf_chunk(c);
        }
        w.leaf_bytes(&self.out);
        w.seq_span(&self.kcache);
        w.seq_span(&self.vcache);
        w.hashes(&self.anchor_pos_path);
        for a in [&self.anchor_k, &self.anchor_v, &self.absorb] {
            w.u8(a.len() as u8);
            for c in a {
                w.leaf_chunk(c);
            }
        }
        w.u8(self.path.len() as u8);
        for c in &self.path {
            w.u64(*c);
        }
        w.bytes32(&self.parent);
        w.hashes(&self.parent_path);
        w.bytes32(&self.partials);
        w.seq_span(&self.lse);
        if let Some(s) = &self.sel {
            w.u8(1);
            w.bytes32(&s.bytes);
            w.hashes(&s.leaf_path);
            w.hashes(&s.pos_path);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<AttnOpening> {
        if b.first() != Some(&VERSION_ATTN) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let q_entry = r.entry()?;
        let o_entry = r.entry()?;
        let nn = r.u8()? as usize;
        let norms = (0..nn).map(|_| r.weights()).collect::<Option<Vec<_>>>()?;
        let pos_path = r.hashes()?;
        let nq = r.u8()? as usize;
        let q = (0..nq).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        let out = r.leaf_bytes()?;
        let kcache = r.seq_span()?;
        let vcache = r.seq_span()?;
        let anchor_pos_path = r.hashes()?;
        let mut lists = [Vec::new(), Vec::new(), Vec::new()];
        for a in lists.iter_mut() {
            let n = r.u8()? as usize;
            *a = (0..n).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        }
        let [anchor_k, anchor_v, absorb] = lists;
        let np = r.u8()? as usize;
        let path = (0..np).map(|_| r.u64()).collect::<Option<Vec<_>>>()?;
        let parent = r.bytes32()?;
        let parent_path = r.hashes()?;
        let partials = r.bytes32()?;
        let lse = r.seq_span()?;
        let sel = match r.o == b.len() {
            true => None,
            false if r.u8()? == 1 => Some(SelOpening { bytes: r.bytes32()?, leaf_path: r.hashes()?, pos_path: r.hashes()? }),
            false => return None,
        };
        let o = AttnOpening {
            response_txid,
            pos_root,
            seq_root,
            layer,
            q_entry,
            o_entry,
            norms,
            pos_path,
            q,
            out,
            kcache,
            vcache,
            anchor_pos_path,
            anchor_k,
            anchor_v,
            absorb,
            path,
            parent,
            parent_path,
            partials,
            lse,
            sel,
        };
        (r.o == b.len()).then_some(o)
    }
}

impl RecurOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_RECUR);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.entry(&self.out_entry);
        w.entry(&self.decay_entry);
        w.u8(self.weights.len() as u8);
        for x in &self.weights {
            w.weights(x);
        }
        for s in [&self.records, &self.start_rows, &self.end_rows] {
            w.seq_span(s);
        }
        w.u8(self.taps.len() as u8);
        for t in &self.taps {
            w.hashes(&t.pos_path);
            w.leaf_chunk(&t.q);
            w.leaf_chunk(&t.k);
            w.leaf_chunk(&t.v);
        }
        for c in [&self.beta, &self.decay, &self.gate] {
            w.leaf_chunk(c);
        }
        w.leaf_bytes(&self.out);
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<RecurOpening> {
        if b.first() != Some(&VERSION_RECUR) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let out_entry = r.entry()?;
        let decay_entry = r.entry()?;
        let nw = r.u8()? as usize;
        let weights = (0..nw).map(|_| r.weights()).collect::<Option<Vec<_>>>()?;
        let records = r.seq_span()?;
        let start_rows = r.seq_span()?;
        let end_rows = r.seq_span()?;
        let nt = r.u8()? as usize;
        let taps = (0..nt)
            .map(|_| Some(RecurTap { pos_path: r.hashes()?, q: r.leaf_chunk()?, k: r.leaf_chunk()?, v: r.leaf_chunk()? }))
            .collect::<Option<Vec<_>>>()?;
        let (beta, decay, gate) = (r.leaf_chunk()?, r.leaf_chunk()?, r.leaf_chunk()?);
        let out = r.leaf_bytes()?;
        let o = RecurOpening { response_txid, pos_root, seq_root, layer, out_entry, decay_entry, weights, records, start_rows, end_rows, taps, beta, decay, gate, out };
        (r.o == b.len()).then_some(o)
    }
}

/// A whole drawn point: its weight rows and the link of its input.
pub const HALF_WHOLE: u8 = 0;
/// The weight rows of a drawn point alone, its link empty.
pub const HALF_PRODUCT: u8 = 1;
/// The link of a drawn point's input alone, the point without its weight rows.
pub const HALF_LINK: u8 = 2;

/// One drawn point of a stage-2 trace opening with its draw index, or one half of it.
#[derive(Debug, Clone, PartialEq)]
pub struct PartPoint {
    pub index: u8,
    pub half: u8,
    pub point: Point,
    pub tok: u32,
    pub link: LinkOpening,
    pub experts: Option<ExpertIds>,
}

/// Part of a stage-2 trace opening.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChainPart {
    pub response_txid: Hash,
    pub roots: Option<(Hash, Hash)>,
    pub points: Vec<PartPoint>,
    pub prefix: Option<PrefixPoint>,
}

impl ChainPart {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_CHAIN_PART);
        w.0.extend_from_slice(&self.response_txid);
        match &self.roots {
            Some((p, q)) => {
                w.u8(1);
                w.0.extend_from_slice(p);
                w.0.extend_from_slice(q);
            }
            None => w.u8(0),
        }
        w.u8(self.points.len() as u8);
        for p in &self.points {
            w.u8(p.index);
            w.u8(p.half);
            w.point(&p.point);
            w.u32(p.tok);
            w.link(&p.link);
            w.expert_ids(p.experts.as_ref());
        }
        match &self.prefix {
            Some(p) => {
                w.u8(1);
                w.prefix_point(p);
            }
            None => w.u8(0),
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<ChainPart> {
        if b.first() != Some(&VERSION_CHAIN_PART) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let roots = match r.u8()? {
            0 => None,
            1 => Some((r.take(32)?.try_into().ok()?, r.take(32)?.try_into().ok()?)),
            _ => return None,
        };
        let n = r.u8()? as usize;
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let index = r.u8()?;
            let half = r.u8()?;
            if half > HALF_LINK {
                return None;
            }
            points.push(PartPoint { index, half, point: r.point()?, tok: r.u32()?, link: r.link()?, experts: r.expert_ids()? });
        }
        let prefix = match r.u8()? {
            0 => None,
            1 => Some(r.prefix_point()?),
            _ => return None,
        };
        (r.o == b.len()).then_some(ChainPart { response_txid, roots, points, prefix })
    }

    /// Splits a stage-2 trace opening into parts of at most `max_len` encoded bytes; `None` when
    /// `o` is not a stage-2 trace opening or one of its points does not fit alone.
    pub fn split(o: &Opening, max_len: usize) -> Option<Vec<ChainPart>> {
        if o.resid.is_some() || o.links.len() != o.points.len() || o.experts.len() != o.points.len() || o.points.len() > u8::MAX as usize {
            return None;
        }
        let empty = ChainPart { response_txid: o.response_txid, roots: o.roots, points: Vec::new(), prefix: None };
        let base = empty.encode().len();
        let mut items: Vec<(usize, Option<PartPoint>)> = Vec::with_capacity(o.points.len() + 1);
        let size = |pp: &PartPoint| ChainPart { points: vec![pp.clone()], ..empty.clone() }.encode().len() - base;
        for (i, (p, (tok, link))) in o.points.iter().zip(&o.links).enumerate() {
            let whole = PartPoint { index: i as u8, half: HALF_WHOLE, point: p.clone(), tok: *tok, link: link.clone(), experts: o.experts[i].clone() };
            let one = size(&whole);
            if base + one <= max_len {
                items.push((one, Some(whole)));
                continue;
            }
            let product = PartPoint { half: HALF_PRODUCT, link: LinkOpening::None, ..whole.clone() };
            let input = Point { rows: Vec::new(), bias: None, ..p.clone() };
            let linked = PartPoint { half: HALF_LINK, point: input, ..whole };
            items.push((size(&product), Some(product)));
            items.push((size(&linked), Some(linked)));
        }
        let prefix = o.prefix.as_ref()?;
        let size = ChainPart { prefix: Some(prefix.clone()), ..empty.clone() }.encode().len() - base;
        items.push((size, None));
        items.sort_by(|a, b| b.0.cmp(&a.0));
        let mut parts: Vec<(usize, ChainPart)> = Vec::new();
        for (size, item) in items {
            if base + size > max_len {
                return None;
            }
            let slot = match parts.iter().position(|(len, _)| len + size <= max_len) {
                Some(k) => k,
                None => {
                    parts.push((base, empty.clone()));
                    parts.len() - 1
                }
            };
            let (len, part) = &mut parts[slot];
            *len += size;
            match item {
                Some(pp) => part.points.push(pp),
                None => part.prefix = Some(prefix.clone()),
            }
        }
        let mut out: Vec<ChainPart> = parts.into_iter().map(|(_, mut p)| {
            p.points.sort_by_key(|q| (q.index, q.half));
            p
        }).collect();
        out.sort_by_key(|p| p.points.first().map_or(u8::MAX, |q| q.index));
        Some(out)
    }
}

impl W {
    fn point(&mut self, p: &Point) {
        self.u32(p.n_ubatch);
        self.hashes(&p.pos_path);
        self.hashes(&p.meta_path);
        self.entry(&p.entry);
        match &p.partner {
            Some(e) => {
                self.u8(1);
                self.entry(e);
            }
            None => self.u8(0),
        }
        match &p.bias {
            Some((e, r)) => {
                self.u8(1);
                self.entry(e);
                self.range(r);
            }
            None => self.u8(0),
        }
        self.u8(p.rows.len() as u8);
        for r in &p.rows {
            self.range(r);
        }
        self.bytes32(&p.x);
        self.hashes(&p.x_path);
        self.u32(p.y_total_len);
        self.u32(p.y_chunk_index);
        self.bytes32(&p.y_chunk);
        self.hashes(&p.y_chunk_path);
        self.hashes(&p.y_path);
    }

    fn prefix_point(&mut self, p: &PrefixPoint) {
        self.u32(p.n_ubatch);
        self.hashes(&p.pos_path);
        self.hashes(&p.meta_path);
        self.entry(&p.entry);
        self.bytes32(&p.x);
        self.hashes(&p.x_path);
        self.entry(&p.embd.0);
        self.range(&p.embd.1);
        self.entry(&p.norm.0);
        self.range(&p.norm.1);
    }

    fn expert_ids(&mut self, e: Option<&ExpertIds>) {
        match e {
            Some(e) => {
                self.u8(1);
                self.bytes32(&e.chunk);
                self.hashes(&e.chunk_path);
            }
            None => self.u8(0),
        }
    }
}

impl R<'_> {
    fn point(&mut self) -> Option<Point> {
        let n_ubatch = self.u32()?;
        let pos_path = self.hashes()?;
        let meta_path = self.hashes()?;
        let entry = self.entry()?;
        let partner = if self.u8()? == 1 { Some(self.entry()?) } else { None };
        let bias = if self.u8()? == 1 { Some((self.entry()?, self.range()?)) } else { None };
        let nr = self.u8()? as usize;
        let rows = (0..nr).map(|_| self.range()).collect::<Option<Vec<_>>>()?;
        Some(Point {
            n_ubatch,
            pos_path,
            meta_path,
            entry,
            partner,
            bias,
            rows,
            x: self.bytes32()?,
            x_path: self.hashes()?,
            y_total_len: self.u32()?,
            y_chunk_index: self.u32()?,
            y_chunk: self.bytes32()?,
            y_chunk_path: self.hashes()?,
            y_path: self.hashes()?,
        })
    }

    fn prefix_point(&mut self) -> Option<PrefixPoint> {
        Some(PrefixPoint {
            n_ubatch: self.u32()?,
            pos_path: self.hashes()?,
            meta_path: self.hashes()?,
            entry: self.entry()?,
            x: self.bytes32()?,
            x_path: self.hashes()?,
            embd: (self.entry()?, self.range()?),
            norm: (self.entry()?, self.range()?),
        })
    }

    /// `None` on a malformed field, `Some(None)` when the point carries no expert ids.
    fn expert_ids(&mut self) -> Option<Option<ExpertIds>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? })),
            _ => None,
        }
    }
}

#[cfg(test)]
mod chain_part_tests {
    use super::*;

    fn point(seed: u8, x: usize) -> Point {
        Point { n_ubatch: 512, pos_path: vec![[seed; 32]; 3], x: vec![seed; x], y_chunk: vec![seed; 64], ..Default::default() }
    }

    fn opening(xs: &[usize]) -> Opening {
        let points: Vec<Point> = xs.iter().enumerate().map(|(i, &x)| point(i as u8 + 1, x)).collect();
        let links = (0..xs.len())
            .map(|i| (i as u32 + 7, LinkOpening::HcIn { layer: EntryOpening::default(), src: HcSource::Streams(LeafBytes { bytes: vec![i as u8; 4096], leaf_path: vec![[1; 32]; 2] }) }))
            .collect();
        let experts = (0..xs.len()).map(|i| (i == 1).then(|| ExpertIds { chunk: vec![3; 32], chunk_path: vec![[2; 32]] })).collect();
        let prefix = Some(PrefixPoint { n_ubatch: 512, x: vec![9; 4608], ..Default::default() });
        Opening { response_txid: [5; 32], points, prefix, links, experts, resid: None, roots: Some(([6; 32], [7; 32])) }
    }

    #[test]
    fn a_part_round_trips_and_is_an_audit_body() {
        let parts = ChainPart::split(&opening(&[1000, 2000]), MAX_BODY_LEN).unwrap();
        assert_eq!(parts.len(), 1);
        let b = parts[0].encode();
        assert_eq!(ChainPart::decode(&b).as_ref(), Some(&parts[0]));
        assert!(is_audit_body(&b));
        let s = seeded(&b, &[8; 32]).unwrap();
        assert!(is_audit_body(&s));
        assert_eq!(inner_body(&s).as_deref(), Some(&b[..]));
        assert!(ChainPart::decode(&b[..b.len() - 1]).is_none());
        assert!(Opening::decode(&b).is_none());
    }

    #[test]
    fn a_split_covers_every_point_and_the_prefix_under_the_cap() {
        let o = opening(&[60_000, 50_000]);
        assert!(o.encode().len() > 100_000);
        let parts = ChainPart::split(&o, 70_000).unwrap();
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.encode().len() <= 70_000));
        let mut idx: Vec<u8> = parts.iter().flat_map(|p| p.points.iter().map(|q| q.index)).collect();
        idx.sort();
        assert_eq!(idx, vec![0, 1]);
        assert_eq!(parts.iter().filter(|p| p.prefix.is_some()).count(), 1);
        for p in &parts {
            for q in &p.points {
                let i = q.index as usize;
                assert_eq!((&q.point, q.tok, &q.link, &q.experts), (&o.points[i], o.links[i].0, &o.links[i].1, &o.experts[i]));
            }
        }
    }

    #[test]
    fn a_point_over_the_cap_is_split_into_its_rows_and_its_link() {
        let mut o = opening(&[20_000, 1000]);
        o.points[0].rows = vec![ChunkRange { first_chunk: 0, chunks: vec![4; 20_000], proof: vec![[3; 32]; 30] }];
        o.links[0].1 = LinkOpening::HcIn { layer: EntryOpening::default(), src: HcSource::Streams(LeafBytes { bytes: vec![1; 30_000], leaf_path: vec![] }) };
        let parts = ChainPart::split(&o, 60_000).unwrap();
        assert!(parts.iter().all(|p| p.encode().len() <= 60_000));
        let halves: Vec<&PartPoint> = parts.iter().flat_map(|p| p.points.iter()).filter(|q| q.index == 0).collect();
        assert_eq!(halves.len(), 2);
        let product = halves.iter().find(|q| q.half == HALF_PRODUCT).unwrap();
        let link = halves.iter().find(|q| q.half == HALF_LINK).unwrap();
        assert_eq!((&product.point, &product.link), (&o.points[0], &LinkOpening::None));
        assert_eq!(&link.link, &o.links[0].1);
        assert!(link.point.rows.is_empty() && link.point.x == o.points[0].x);
        assert!(parts.iter().flat_map(|p| p.points.iter()).any(|q| q.index == 1 && q.half == HALF_WHOLE));
        for p in &parts {
            assert_eq!(ChainPart::decode(&p.encode()).as_ref(), Some(p));
        }
    }

    #[test]
    fn a_point_over_the_cap_cannot_be_split() {
        assert!(ChainPart::split(&opening(&[80_000, 1000]), 70_000).is_none());
        let mut o = opening(&[1000, 1000]);
        o.prefix = None;
        assert!(ChainPart::split(&o, MAX_BODY_LEN).is_none());
    }

    #[test]
    fn the_whole_opening_encoding_is_unchanged() {
        let o = opening(&[3000, 4000]);
        assert_eq!(Opening::decode(&o.encode()), Some(o));
    }
}

#[cfg(test)]
mod audit_body_tests {
    use super::*;

    fn seeded(body: &[u8]) -> Vec<u8> {
        let mut v = vec![VERSION_SEEDED];
        v.extend_from_slice(&body[1..33]);
        v.extend_from_slice(&[9u8; 32]);
        v.push(body[0]);
        v.extend_from_slice(&body[33..]);
        v
    }

    #[test]
    fn attention_and_recurrent_bodies_are_audit_bodies() {
        let attn = AttnOpening { response_txid: [4; 32], path: vec![1], partials: vec![2; 4], ..Default::default() }.encode();
        let recur = RecurOpening { response_txid: [4; 32], ..Default::default() }.encode();
        for b in [&attn, &recur] {
            assert!(is_audit_body(b));
            assert!(is_audit_body(&seeded(b)));
            assert_eq!(inner_body(&seeded(b)).as_deref(), Some(&b[..]));
            assert!(!is_audit_body(&b[..b.len() - 1]));
        }
        assert!(!is_audit_body(&[]));
        assert!(!is_audit_body(&[0xEE; 64]));
    }
}

#[cfg(test)]
mod recur_opening_tests {
    use super::*;
    use crate::seq::SeqSpan;

    fn chunk(seed: u8, n: usize) -> LeafChunk {
        LeafChunk { total_len: 9000, index: seed as u32, chunk: vec![seed; n], chunk_path: vec![[seed; 32]; 3], leaf_path: vec![[seed ^ 2; 32]; 4] }
    }

    fn span(seed: u8, n: usize) -> SeqSpan {
        SeqSpan { first_chunk: seed as u32, bytes: vec![seed; n], chunk_proof: vec![[seed; 32]; 5], leaf_path: vec![[seed ^ 1; 32]; 7] }
    }

    fn sample() -> RecurOpening {
        let e = |i: u32| EntryOpening { index: i, bytes: vec![i as u8; 40], path: vec![[i as u8; 32]; 4] };
        RecurOpening {
            response_txid: [1; 32],
            pos_root: [2; 32],
            seq_root: [3; 32],
            layer: e(9),
            out_entry: e(4),
            decay_entry: e(5),
            weights: (0..6).map(|i| (e(20 + i), ChunkRange { first_chunk: i as u64, chunks: vec![i as u8; 64], proof: vec![[i as u8; 32]; 2] })).collect(),
            records: span(5, 50_000),
            start_rows: span(6, 8192),
            end_rows: SeqSpan::default(),
            taps: (0..4u8).map(|t| RecurTap { pos_path: vec![[t; 32]; 6], q: chunk(t, 1024), k: chunk(t + 10, 1024), v: chunk(t + 20, 512) }).collect(),
            beta: chunk(30, 128),
            decay: chunk(31, 1024),
            gate: chunk(32, 1024),
            out: LeafBytes { bytes: vec![7; 4608], leaf_path: vec![[8; 32]; 6] },
        }
    }

    #[test]
    fn recurrent_opening_round_trips() {
        let o = sample();
        let b = o.encode();
        assert_eq!(b[0], VERSION_RECUR);
        assert_eq!(RecurOpening::decode(&b), Some(o));
        let mut c = b.clone();
        c.push(0);
        assert!(RecurOpening::decode(&c).is_none());
        assert!(AttnOpening::decode(&b).is_none() && Opening::decode(&b).is_none());
        assert!(seeded(&b, &[9; 32]).is_some());
    }
}

#[cfg(test)]
mod resid_opening_tests {
    use super::*;

    fn chunk(seed: u8, n: usize) -> LeafChunk {
        LeafChunk { total_len: 9000, index: seed as u32, chunk: vec![seed; n], chunk_path: vec![[seed; 32]; 3], leaf_path: vec![[seed ^ 2; 32]; 4] }
    }

    fn point() -> ResidPoint {
        let e = |i: u32| EntryOpening { index: i, bytes: vec![i as u8; 40], path: vec![[i as u8; 32]; 4] };
        ResidPoint {
            n_ubatch: 1,
            tok: 7,
            pos_path: vec![[1; 32]; 6],
            meta_path: vec![[2; 32]; 9],
            layer: e(3),
            product: e(4),
            dst: chunk(5, 1024),
            src: ResidSrc::Chunk(chunk(6, 1024)),
            add: ResidAdd::Normed { y: LeafBytes { bytes: vec![7; 4096], leaf_path: vec![[8; 32]; 9] }, norm: (e(9), ChunkRange { first_chunk: 3, chunks: vec![1; 1024], proof: vec![[4; 32]; 20] }) },
        }
    }

    #[test]
    fn residual_opening_round_trips() {
        let e = EntryOpening { index: 11, bytes: vec![11; 40], path: vec![[11; 32]; 4] };
        let range = |s: u8| ChunkRange { first_chunk: s as u64, chunks: vec![s; 4352], proof: vec![[s; 32]; 30] };
        let feed = FeedRow { x: vec![3; 4608], x_path: vec![[3; 32]; 9], row: range(4), bias: Some((e.clone(), range(5))) };
        let expert = ExpertRow { entry: e, x: vec![6; 9216], x_path: vec![[6; 32]; 9], row: range(7) };
        for (roots, expert) in [(None, None), (Some(([5; 32], [6; 32])), Some(expert))] {
            let o = ResidOpening { response_txid: [1; 32], roots, resid: point(), feed: feed.clone(), expert };
            let b = o.encode();
            assert_eq!(b[0], VERSION_RESID);
            assert_eq!(ResidOpening::decode(&b), Some(o));
            assert!(is_audit_body(&b));
            let s = seeded(&b, &[9; 32]).unwrap();
            assert!(is_audit_body(&s));
            assert_eq!(inner_body(&s), Some(b.clone()));
            let mut c = b.clone();
            c.push(0);
            assert!(ResidOpening::decode(&c).is_none() && !is_audit_body(&b[..b.len() - 1]));
            assert!(Opening::decode(&b).is_none() && AttnOpening::decode(&b).is_none() && RecurOpening::decode(&b).is_none());
        }
        let w = |s: u8| (EntryOpening { index: s as u32, bytes: vec![s; 40], path: vec![[s; 32]; 4] }, range(s));
        for gate in [None, Some(w(12))] {
            let mut p = point();
            p.add = ResidAdd::Moe(Box::new(MoeAdd {
                scores: LeafBytes { bytes: vec![1; 1024], leaf_path: vec![[1; 32]; 9] },
                bias: w(2),
                ids: ExpertIds { chunk: vec![3; 32], chunk_path: vec![[3; 32]; 2] },
                slots: vec![chunk(4, 1024), chunk(5, 1024)],
                shexp: chunk(6, 1024),
                router_row: w(7),
                resid: LeafBytes { bytes: vec![8; 4096], leaf_path: vec![[8; 32]; 9] },
                norm: w(9),
                shexp_gate: gate,
            }));
            let o = ResidOpening { response_txid: [1; 32], roots: None, resid: p, feed: feed.clone(), expert: None };
            assert_eq!(ResidOpening::decode(&o.encode()), Some(o));
        }
    }

    #[test]
    fn a_trace_opening_may_leave_its_residual_out() {
        let p = Point { x: vec![1; 36], ..Default::default() };
        let base = Opening {
            response_txid: [1; 32],
            points: vec![p.clone(), p],
            prefix: Some(PrefixPoint::default()),
            links: vec![(3, LinkOpening::None), (4, LinkOpening::None)],
            experts: vec![None, None],
            resid: None,
            roots: Some(([5; 32], [6; 32])),
        };
        for o in [base.clone(), Opening { resid: Some(point()), ..base }] {
            let b = o.encode();
            assert_eq!(b[0], VERSION_ROOTS);
            assert_eq!(Opening::decode(&b), Some(o));
        }
    }
}

#[cfg(test)]
mod attn_opening_tests {
    use super::*;

    fn chunk(seed: u8, n: usize) -> LeafChunk {
        LeafChunk { total_len: 4096, index: seed as u32, chunk: vec![seed; n], chunk_path: vec![[seed; 32]; 2], leaf_path: vec![[seed ^ 1; 32]; 3] }
    }

    fn sample() -> AttnOpening {
        let e = |i: u32| EntryOpening { index: i, bytes: vec![i as u8; 40], path: vec![[i as u8; 32]; 4] };
        let span = |seed: u8, n: usize| crate::seq::SeqSpan { first_chunk: seed as u32, bytes: vec![seed; n], chunk_proof: vec![[seed; 32]; 5], leaf_path: vec![[seed ^ 1; 32]; 9] };
        AttnOpening {
            response_txid: [7; 32],
            pos_root: [8; 32],
            seq_root: [9; 32],
            layer: e(9),
            q_entry: e(1),
            o_entry: e(2),
            norms: vec![(e(3), ChunkRange { first_chunk: 4, chunks: vec![8; 64], proof: vec![[2; 32]; 3] }), (e(5), ChunkRange::default())],
            pos_path: vec![[3; 32]; 5],
            q: vec![chunk(1, 1024), chunk(2, 1024)],
            out: LeafBytes { bytes: vec![5; 4608], leaf_path: vec![[6; 32]; 6] },
            kcache: span(20, 54_400),
            vcache: span(21, 54_400),
            anchor_pos_path: vec![[4; 32]; 8],
            anchor_k: vec![chunk(9, 1024)],
            anchor_v: vec![chunk(10, 1024)],
            absorb: vec![chunk(11, 1024), chunk(12, 1024)],
            path: vec![3, 1],
            parent: vec![13; 264],
            parent_path: vec![[14; 32]; 8],
            partials: vec![15; 264 * 3],
            lse: span(22, 1024),
            sel: None,
        }
    }

    #[test]
    fn attention_opening_round_trips() {
        let o = sample();
        let b = o.encode();
        assert_eq!(b[0], VERSION_ATTN);
        assert_eq!(AttnOpening::decode(&b), Some(o));
        let with_sel = AttnOpening { sel: Some(SelOpening { bytes: vec![1, 0, 0, 0, 5, 0, 0, 0], leaf_path: vec![[3; 32]; 4], pos_path: vec![[4; 32]; 3] }), ..sample() };
        assert_eq!(AttnOpening::decode(&with_sel.encode()), Some(with_sel.clone()));
        let empty = AttnOpening { sel: Some(SelOpening { bytes: Vec::new(), leaf_path: vec![[3; 32]; 4], pos_path: vec![[4; 32]; 3] }), ..sample() };
        assert_eq!(AttnOpening::decode(&empty.encode()), Some(empty));
        let mut w = W(sample().encode());
        w.u8(0);
        assert!(AttnOpening::decode(&w.0).is_none());
    }

    #[test]
    fn trailing_bytes_and_other_versions_are_rejected() {
        let mut b = sample().encode();
        b.push(0);
        assert!(AttnOpening::decode(&b).is_none());
        let mut c = sample().encode();
        c[0] = VERSION_CHAIN;
        assert!(AttnOpening::decode(&c).is_none());
        assert!(Opening::decode(&sample().encode()).is_none());
    }
}

/// The pointer head output at one position: the meta leaf and one chunk of the head's `Y` leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionLeaf {
    pub n_ubatch: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub y: LeafChunk,
}

/// The answer of a decision response and the head outputs that give its drawn score.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionOpening {
    pub response_txid: Hash,
    /// Position and sequence roots of a trace that commits sequence leaves.
    pub roots: Option<(Hash, Hash)>,
    pub head: EntryOpening,
    /// The answer plaintext.
    pub answer: Vec<u8>,
    pub marker: DecisionLeaf,
    pub pointer: DecisionLeaf,
}

impl W {
    fn decision_leaf(&mut self, l: &DecisionLeaf) {
        self.u32(l.n_ubatch);
        self.hashes(&l.pos_path);
        self.hashes(&l.meta_path);
        self.leaf_chunk(&l.y);
    }
}

impl R<'_> {
    fn decision_leaf(&mut self) -> Option<DecisionLeaf> {
        Some(DecisionLeaf { n_ubatch: self.u32()?, pos_path: self.hashes()?, meta_path: self.hashes()?, y: self.leaf_chunk()? })
    }
}

impl DecisionOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_DECISION);
        w.0.extend_from_slice(&self.response_txid);
        match &self.roots {
            Some((p, q)) => {
                w.u8(1);
                w.0.extend_from_slice(p);
                w.0.extend_from_slice(q);
            }
            None => w.u8(0),
        }
        w.entry(&self.head);
        w.bytes32(&self.answer);
        w.decision_leaf(&self.marker);
        w.decision_leaf(&self.pointer);
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<DecisionOpening> {
        if b.first() != Some(&VERSION_DECISION) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let roots = match r.u8()? {
            0 => None,
            1 => Some((r.take(32)?.try_into().ok()?, r.take(32)?.try_into().ok()?)),
            _ => return None,
        };
        let head = r.entry()?;
        let answer = r.bytes32()?;
        let marker = r.decision_leaf()?;
        let pointer = r.decision_leaf()?;
        (r.o == b.len()).then_some(DecisionOpening { response_txid, roots, head, answer, marker, pointer })
    }
}

/// One member position of a drawn pool: its indexer key and pool gate outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelMember {
    pub pos_path: Vec<Hash>,
    pub k: LeafBytes,
    pub g: LeafBytes,
}

/// Selection point of a response: the selection leaf of the drawn position and layer, the
/// indexer query and head weights there, and the members of each drawn pool in draw order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelOpeningPoint {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    /// Indexer query, key, gate and head weight products, in that order.
    pub entries: Vec<EntryOpening>,
    /// Key norm weight, key norm bias and pool position bias, in that order.
    pub norms: Vec<(EntryOpening, ChunkRange)>,
    pub pos_path: Vec<Hash>,
    pub sel: LeafBytes,
    pub iq: LeafBytes,
    pub w: LeafBytes,
    pub members: Vec<SelMember>,
    /// 0 for every drawn pool, else [`SEL_PART_SELECTED`] or [`SEL_PART_OUTSIDE`].
    pub part: u8,
}

impl SelOpeningPoint {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(if self.part == 0 { VERSION_SEL } else { VERSION_SEL_PART });
        w.0.extend_from_slice(&self.response_txid);
        if self.part != 0 {
            w.u8(self.part);
        }
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.u8(self.entries.len() as u8);
        for e in &self.entries {
            w.entry(e);
        }
        w.u8(self.norms.len() as u8);
        for n in &self.norms {
            w.weights(n);
        }
        w.hashes(&self.pos_path);
        for l in [&self.sel, &self.iq, &self.w] {
            w.leaf_bytes(l);
        }
        w.u8(self.members.len() as u8);
        for m in &self.members {
            w.hashes(&m.pos_path);
            w.leaf_bytes(&m.k);
            w.leaf_bytes(&m.g);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<SelOpeningPoint> {
        let split = match b.first() {
            Some(&VERSION_SEL) => false,
            Some(&VERSION_SEL_PART) => true,
            _ => return None,
        };
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let part = if split { r.u8()? } else { 0 };
        if split && part != SEL_PART_SELECTED && part != SEL_PART_OUTSIDE {
            return None;
        }
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let ne = r.u8()? as usize;
        let entries = (0..ne).map(|_| r.entry()).collect::<Option<Vec<_>>>()?;
        let nn = r.u8()? as usize;
        let norms = (0..nn).map(|_| r.weights()).collect::<Option<Vec<_>>>()?;
        let pos_path = r.hashes()?;
        let (sel, iq, w) = (r.leaf_bytes()?, r.leaf_bytes()?, r.leaf_bytes()?);
        let nm = r.u8()? as usize;
        let members = (0..nm).map(|_| Some(SelMember { pos_path: r.hashes()?, k: r.leaf_bytes()?, g: r.leaf_bytes()? })).collect::<Option<Vec<_>>>()?;
        (r.o == b.len()).then_some(SelOpeningPoint { response_txid, pos_root, seq_root, layer, entries, norms, pos_path, sel, iq, w, members, part })
    }

    /// The opening as sent: whole when it fits `max_len` encoded bytes, else its selected and
    /// outside halves; `None` when a half does not fit or the members do not split evenly.
    pub fn split(&self, max_len: usize) -> Option<Vec<SelOpeningPoint>> {
        if self.part != 0 {
            return None;
        }
        if self.encode().len() <= max_len {
            return Some(vec![self.clone()]);
        }
        if self.members.is_empty() || self.members.len() % 2 != 0 {
            return None;
        }
        let half = self.members.len() / 2;
        let parts = [(SEL_PART_SELECTED, &self.members[..half]), (SEL_PART_OUTSIDE, &self.members[half..])]
            .map(|(part, members)| SelOpeningPoint { part, members: members.to_vec(), ..self.clone() });
        parts.iter().all(|p| p.encode().len() <= max_len).then(|| parts.to_vec())
    }
}

/// Compressed row point of a response: the drawn row of the compressed cache of a source layer,
/// the compressor outputs of the block's members (`k` value, `g` gate, empty without a gate), and
/// the norm, indexer key matrix and key norm the row is built with, in that order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompOpeningPoint {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub entries: Vec<EntryOpening>,
    pub weights: Vec<(EntryOpening, ChunkRange)>,
    pub row: crate::seq::SeqSpan,
    pub members: Vec<SelMember>,
}

impl CompOpeningPoint {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_COMP);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.u8(self.entries.len() as u8);
        for e in &self.entries {
            w.entry(e);
        }
        w.u8(self.weights.len() as u8);
        for n in &self.weights {
            w.weights(n);
        }
        w.seq_span(&self.row);
        w.u8(self.members.len() as u8);
        for m in &self.members {
            w.hashes(&m.pos_path);
            w.leaf_bytes(&m.k);
            w.leaf_bytes(&m.g);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<CompOpeningPoint> {
        if b.first() != Some(&VERSION_COMP) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let ne = r.u8()? as usize;
        let entries = (0..ne).map(|_| r.entry()).collect::<Option<Vec<_>>>()?;
        let nw = r.u8()? as usize;
        let weights = (0..nw).map(|_| r.weights()).collect::<Option<Vec<_>>>()?;
        let row = r.seq_span()?;
        let nm = r.u8()? as usize;
        let members = (0..nm).map(|_| Some(SelMember { pos_path: r.hashes()?, k: r.leaf_bytes()?, g: r.leaf_bytes()? })).collect::<Option<Vec<_>>>()?;
        (r.o == b.len()).then_some(CompOpeningPoint { response_txid, pos_root, seq_root, layer, entries, weights, row, members })
    }
}

/// Phase of the attention point of a compressed attention layer: keys are the window of the
/// layer's own cache, then the compressed rows the selection keeps. Query fields at the root and
/// leaf phases, output, sink and mass at the root, rows and an anchor of the window at a leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompAttnOpening {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub q_entry: EntryOpening,
    pub q: Vec<LeafChunk>,
    pub pos_path: Vec<Hash>,
    pub o_entry: EntryOpening,
    pub out: LeafBytes,
    pub sinks: (EntryOpening, ChunkRange),
    pub lse: crate::seq::SeqSpan,
    pub sel: Option<SelOpening>,
    pub path: Vec<u64>,
    pub parent: Vec<u8>,
    pub parent_path: Vec<Hash>,
    pub partials: Vec<u8>,
    pub window: crate::seq::SeqSpan,
    pub comp: Vec<crate::seq::SeqSpan>,
    pub anchor_pos_path: Vec<Hash>,
    pub anchor_k: Vec<LeafChunk>,
    pub norms: Vec<(EntryOpening, ChunkRange)>,
}

impl CompAttnOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_ATTN_COMP);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.entry(&self.q_entry);
        w.u8(self.q.len() as u8);
        for c in &self.q {
            w.leaf_chunk(c);
        }
        w.hashes(&self.pos_path);
        w.entry(&self.o_entry);
        w.leaf_bytes(&self.out);
        w.weights(&self.sinks);
        w.seq_span(&self.lse);
        match &self.sel {
            Some(s) => {
                w.u8(1);
                w.bytes32(&s.bytes);
                w.hashes(&s.leaf_path);
                w.hashes(&s.pos_path);
            }
            None => w.u8(0),
        }
        w.u8(self.path.len() as u8);
        for c in &self.path {
            w.u64(*c);
        }
        w.bytes32(&self.parent);
        w.hashes(&self.parent_path);
        w.bytes32(&self.partials);
        w.seq_span(&self.window);
        w.u16(self.comp.len() as u16);
        for s in &self.comp {
            w.seq_span(s);
        }
        w.hashes(&self.anchor_pos_path);
        w.u8(self.anchor_k.len() as u8);
        for c in &self.anchor_k {
            w.leaf_chunk(c);
        }
        w.u8(self.norms.len() as u8);
        for n in &self.norms {
            w.weights(n);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<CompAttnOpening> {
        if b.first() != Some(&VERSION_ATTN_COMP) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let q_entry = r.entry()?;
        let nq = r.u8()? as usize;
        let q = (0..nq).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        let pos_path = r.hashes()?;
        let o_entry = r.entry()?;
        let out = r.leaf_bytes()?;
        let sinks = r.weights()?;
        let lse = r.seq_span()?;
        let sel = match r.u8()? {
            0 => None,
            1 => Some(SelOpening { bytes: r.bytes32()?, leaf_path: r.hashes()?, pos_path: r.hashes()? }),
            _ => return None,
        };
        let np = r.u8()? as usize;
        let path = (0..np).map(|_| r.u64()).collect::<Option<Vec<_>>>()?;
        let parent = r.bytes32()?;
        let parent_path = r.hashes()?;
        let partials = r.bytes32()?;
        let window = r.seq_span()?;
        let nc = r.u16()? as usize;
        let comp = (0..nc).map(|_| r.seq_span()).collect::<Option<Vec<_>>>()?;
        let anchor_pos_path = r.hashes()?;
        let na = r.u8()? as usize;
        let anchor_k = (0..na).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        let nn = r.u8()? as usize;
        let norms = (0..nn).map(|_| r.weights()).collect::<Option<Vec<_>>>()?;
        (r.o == b.len()).then_some(CompAttnOpening {
            response_txid, pos_root, seq_root, layer, q_entry, q, pos_path, o_entry, out, sinks, lse, sel, path, parent, parent_path, partials,
            window, comp, anchor_pos_path, anchor_k, norms,
        })
    }
}

/// Selection point of a compressed attention tier: the drawn selection leaf (the rows a layer
/// keeps, or the candidate blocks it picks), the indexer query and head weights there, the
/// candidate leaf the rows are restricted to, and the compressed rows of the drawn rows or blocks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompSelOpening {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub entries: Vec<EntryOpening>,
    pub pos_path: Vec<Hash>,
    pub sel: LeafBytes,
    pub iq: LeafBytes,
    pub iw: LeafBytes,
    pub cand: LeafBytes,
    pub rows: Vec<crate::seq::SeqSpan>,
}

impl CompSelOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_COMP_SEL);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.u8(self.entries.len() as u8);
        for e in &self.entries {
            w.entry(e);
        }
        w.hashes(&self.pos_path);
        for l in [&self.sel, &self.iq, &self.iw, &self.cand] {
            w.leaf_bytes(l);
        }
        w.u8(self.rows.len() as u8);
        for r in &self.rows {
            w.seq_span(r);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<CompSelOpening> {
        if b.first() != Some(&VERSION_COMP_SEL) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let layer = r.entry()?;
        let ne = r.u8()? as usize;
        let entries = (0..ne).map(|_| r.entry()).collect::<Option<Vec<_>>>()?;
        let pos_path = r.hashes()?;
        let (sel, iq, iw, cand) = (r.leaf_bytes()?, r.leaf_bytes()?, r.leaf_bytes()?, r.leaf_bytes()?);
        let nr = r.u8()? as usize;
        let rows = (0..nr).map(|_| r.seq_span()).collect::<Option<Vec<_>>>()?;
        (r.o == b.len()).then_some(CompSelOpening { response_txid, pos_root, seq_root, layer, entries, pos_path, sel, iq, iw, cand, rows })
    }
}

/// N-gram lookup point of a response: one chunk of one stream after the lookup, the whole stream
/// before it, the stream's key and gate weights, and the value chunk, at the drawn position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngramOpening {
    pub response_txid: Hash,
    pub pos_root: Hash,
    pub seq_root: Hash,
    pub layer: EntryOpening,
    pub kv: EntryOpening,
    pub gate: (EntryOpening, ChunkRange),
    pub pos_path: Vec<Hash>,
    pub dst: LeafChunk,
    pub x: Vec<LeafChunk>,
    pub key: Vec<LeafChunk>,
    pub value: LeafChunk,
}

impl EngramOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION_ENGRAM);
        w.0.extend_from_slice(&self.response_txid);
        w.0.extend_from_slice(&self.pos_root);
        w.0.extend_from_slice(&self.seq_root);
        w.entry(&self.layer);
        w.entry(&self.kv);
        w.weights(&self.gate);
        w.hashes(&self.pos_path);
        w.leaf_chunk(&self.dst);
        for list in [&self.x, &self.key] {
            w.u8(list.len() as u8);
            for c in list {
                w.leaf_chunk(c);
            }
        }
        w.leaf_chunk(&self.value);
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<EngramOpening> {
        if b.first() != Some(&VERSION_ENGRAM) {
            return None;
        }
        let mut r = R { b, o: 1 };
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let pos_root: Hash = r.take(32)?.try_into().ok()?;
        let seq_root: Hash = r.take(32)?.try_into().ok()?;
        let (layer, kv, gate, pos_path, dst) = (r.entry()?, r.entry()?, r.weights()?, r.hashes()?, r.leaf_chunk()?);
        let nx = r.u8()? as usize;
        let x = (0..nx).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        let nk = r.u8()? as usize;
        let key = (0..nk).map(|_| r.leaf_chunk()).collect::<Option<Vec<_>>>()?;
        let value = r.leaf_chunk()?;
        (r.o == b.len()).then_some(EngramOpening { response_txid, pos_root, seq_root, layer, kv, gate, pos_path, dst, x, key, value })
    }
}
