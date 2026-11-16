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
                self.u8(2);
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
            2 => {
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
                }))
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
            w.u32(p.n_ubatch);
            w.hashes(&p.pos_path);
            w.hashes(&p.meta_path);
            w.entry(&p.entry);
            match &p.partner {
                Some(e) => {
                    w.u8(1);
                    w.entry(e);
                }
                None => w.u8(0),
            }
            match &p.bias {
                Some((e, r)) => {
                    w.u8(1);
                    w.entry(e);
                    w.range(r);
                }
                None => w.u8(0),
            }
            w.u8(p.rows.len() as u8);
            for r in &p.rows {
                w.range(r);
            }
            w.bytes32(&p.x);
            w.hashes(&p.x_path);
            w.u32(p.y_total_len);
            w.u32(p.y_chunk_index);
            w.bytes32(&p.y_chunk);
            w.hashes(&p.y_chunk_path);
            w.hashes(&p.y_path);
        }
        if chain {
            for (i, (tok, link)) in self.links.iter().enumerate() {
                w.u32(*tok);
                w.link(link);
                match self.experts.get(i).cloned().flatten() {
                    Some(e) => {
                        w.u8(1);
                        w.bytes32(&e.chunk);
                        w.hashes(&e.chunk_path);
                    }
                    None => w.u8(0),
                }
            }
        }
        if let Some(p) = &self.prefix {
            w.u32(p.n_ubatch);
            w.hashes(&p.pos_path);
            w.hashes(&p.meta_path);
            w.entry(&p.entry);
            w.bytes32(&p.x);
            w.hashes(&p.x_path);
            w.entry(&p.embd.0);
            w.range(&p.embd.1);
            w.entry(&p.norm.0);
            w.range(&p.norm.1);
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
            let n_ubatch = r.u32()?;
            let pos_path = r.hashes()?;
            let meta_path = r.hashes()?;
            let entry = r.entry()?;
            let partner = if r.u8()? == 1 { Some(r.entry()?) } else { None };
            let bias = if r.u8()? == 1 { Some((r.entry()?, r.range()?)) } else { None };
            let nr = r.u8()? as usize;
            let rows = (0..nr).map(|_| r.range()).collect::<Option<Vec<_>>>()?;
            points.push(Point {
                n_ubatch,
                pos_path,
                meta_path,
                entry,
                partner,
                bias,
                rows,
                x: r.bytes32()?,
                x_path: r.hashes()?,
                y_total_len: r.u32()?,
                y_chunk_index: r.u32()?,
                y_chunk: r.bytes32()?,
                y_chunk_path: r.hashes()?,
                y_path: r.hashes()?,
            });
        }
        let mut experts = Vec::new();
        let links = if chain {
            let mut links = Vec::with_capacity(n);
            for _ in 0..n {
                links.push((r.u32()?, r.link()?));
                experts.push(match r.u8()? {
                    0 => None,
                    1 => Some(ExpertIds { chunk: r.bytes32()?, chunk_path: r.hashes()? }),
                    _ => return None,
                });
            }
            links
        } else {
            Vec::new()
        };
        let prefix = if version >= VERSION_PREFIX {
            Some(PrefixPoint {
                n_ubatch: r.u32()?,
                pos_path: r.hashes()?,
                meta_path: r.hashes()?,
                entry: r.entry()?,
                x: r.bytes32()?,
                x_path: r.hashes()?,
                embd: (r.entry()?, r.range()?),
                norm: (r.entry()?, r.range()?),
            })
        } else {
            None
        };
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

/// Whether `body` decodes as a trace, attention, recurrent or residual opening.
pub fn is_audit_body(body: &[u8]) -> bool {
    Opening::decode(body).is_some()
        || inner_body(body).is_some_and(|b| AttnOpening::decode(&b).is_some() || RecurOpening::decode(&b).is_some() || ResidOpening::decode(&b).is_some())
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
    inner_body(body).as_deref().and_then(AttnOpening::decode).map(|o| o.path)
}

/// Wraps an unseeded opening body into the seeded envelope naming `seed_block`.
pub fn seeded(body: &[u8], seed_block: &Hash) -> Option<Vec<u8>> {
    let version = *body.first()?;
    if !matches!(version, VERSION | VERSION_PREFIX | VERSION_CHAIN | VERSION_ATTN | VERSION_RECUR | VERSION_ROOTS | VERSION_RESID) || body.len() < 33 {
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
        }
    }

    #[test]
    fn attention_opening_round_trips() {
        let o = sample();
        let b = o.encode();
        assert_eq!(b[0], VERSION_ATTN);
        assert_eq!(AttnOpening::decode(&b), Some(o));
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
