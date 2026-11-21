//! Gateway of a network-model link (H14), and the head side of it.
//!
//! A link miner listens on one public TCP port. A pipeline head connects, both sides prove their
//! escrow key over an ephemeral secp256k1 ECDH handshake, and everything after runs in
//! ChaCha20-Poly1305 frames. The link only ever runs its own layers: the head sends token ids or
//! a residual of fixed shape, the link answers with its residual. No graph or code crosses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use secp256k1::{ecdh::SharedSecret, All, Keypair, Message, PublicKey, Secp256k1, SecretKey, XOnlyPublicKey};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

const MAGIC: &[u8; 4] = b"KRXG";
const VERSION: u8 = 3;
const CLIENT_DOMAIN: &[u8] = b"KeryxGatewayClientV1";
const SERVER_DOMAIN: &[u8] = b"KeryxGatewayServerV1";
const RESPONDER_DOMAIN: &[u8] = b"KeryxServiceResponderV1";
/// Must equal the node's `LINK_SIG_DOMAIN`.
const LINK_DOMAIN: &[u8] = b"KeryxPipelineLinkV1";
const CHALLENGE_LEN: usize = 4 + 1 + 32;
const HELLO_LEN: usize = 4 + 1 + 32 + 33 + 32 + 32 + 64;
const REPLY_LEN: usize = 1 + 33 + 32 + 1 + 64;
/// Concurrent sessions a gateway holds, and per head.
const MAX_SESSIONS: usize = 16;
const MAX_SESSIONS_PER_HEAD: usize = 4;
/// How often a live session re-checks that its head is still assigned.
static REAUTH_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(30_000);

fn reauth_interval() -> Duration {
    Duration::from_millis(REAUTH_MS.load(std::sync::atomic::Ordering::Relaxed))
}
/// A link signs only for a head that computed through it this recently.
const SIGN_AFTER_DATA: Duration = Duration::from_secs(3600);
const MAX_FRAME: usize = 16 << 20;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const CONTROL_KEEPALIVE: Duration = Duration::from_secs(30);

// server-side idle limit of a session (a head that vanished must not keep the GPU paused)
static SESSION_IDLE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(120_000);

fn session_idle_timeout() -> Duration {
    Duration::from_millis(SESSION_IDLE_MS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Holds the GPU for the head from its first step to the end of its sequence: PoW and PoM stay
/// paused.
struct ShardSessionGuard;

impl ShardSessionGuard {
    fn new() -> Self {
        crate::pom_gpu::shard_session_begin();
        ShardSessionGuard
    }
}

impl Drop for ShardSessionGuard {
    fn drop(&mut self) {
        crate::pom_gpu::shard_session_end();
    }
}
/// Recent-coinbase memory of the peer directory, in DAA (the node's eligibility window).
const PEER_MEMORY_DAA: u64 = 6_000;

const FRAME_PING: u8 = 1;
const FRAME_PONG: u8 = 2;
const FRAME_SIGN: u8 = 3;
const FRAME_SIGNED: u8 = 4;
const FRAME_ERROR: u8 = 5;
const FRAME_OPEN: u8 = 6;
const FRAME_OPENED: u8 = 7;
const FRAME_STEP: u8 = 8;
const FRAME_RESULT: u8 = 9;
const FRAME_CLOSE: u8 = 10;

const STATUS_OK: u8 = 0;
const STATUS_NO_SHARD: u8 = 1;
const STATUS_NOT_ASSIGNED: u8 = 2;
const STATUS_BUSY: u8 = 3;

/// The escrow key as a gateway identity: proves the peer and signs responses as a link.
pub struct GatewayIdentity {
    secp: Secp256k1<All>,
    secret: SecretKey,
    pub pubkey: [u8; 32],
}

impl GatewayIdentity {
    pub fn from_privkey_hex(privkey_hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(privkey_hex.trim()).map_err(|e| format!("escrow key hex: {}", e))?;
        let secp = Secp256k1::new();
        let secret = SecretKey::from_slice(&bytes).map_err(|e| format!("escrow key: {}", e))?;
        let (xonly, _) = Keypair::from_secret_key(&secp, &secret).x_only_public_key();
        Ok(Self { secp, secret, pubkey: xonly.serialize() })
    }

    fn sign_digest(&self, digest: &[u8; 32]) -> [u8; 64] {
        let msg = Message::from_digest_slice(digest).unwrap();
        let keypair = Keypair::from_secret_key(&self.secp, &self.secret);
        *self.secp.sign_schnorr_no_aux_rand(&msg, &keypair).as_ref()
    }

}

/// Digest a link signs: its served `tier`, the head's key, then the response's v1 and extension
/// bytes (`AiResponsePayload::link_signed_bytes`).
fn link_digest(tier: u8, head_pubkey: &[u8; 32], message: &[u8]) -> [u8; 32] {
    domain_hash(LINK_DOMAIN, &[&[tier], head_pubkey, message])
}

fn domain_hash(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake2b_simd::Params::new().hash_length(32).to_state();
    h.update(domain);
    for p in parts {
        h.update(p);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}

fn verify_schnorr(pubkey: &[u8; 32], sig: &[u8; 64], digest: &[u8; 32]) -> bool {
    let Ok(pk) = XOnlyPublicKey::from_slice(pubkey) else { return false };
    let Ok(sig) = secp256k1::schnorr::Signature::from_slice(sig) else { return false };
    let Ok(msg) = Message::from_digest_slice(digest) else { return false };
    secp256k1::SECP256K1.verify_schnorr(&sig, &msg, &pk).is_ok()
}

/// Verifies a link signature the way the node does.
pub fn verify_response_signature(pubkey: &[u8; 32], sig: &[u8; 64], signed_bytes: &[u8]) -> bool {
    verify_schnorr(pubkey, sig, &domain_hash(RESPONDER_DOMAIN, &[signed_bytes]))
}

// ── Framed cipher ──────────────────────────────────────────────────────────────────────────────

struct Sealer {
    aead: ChaCha20Poly1305,
    counter: u64,
}

impl Sealer {
    fn new(key: &[u8; 32]) -> Self {
        Self { aead: ChaCha20Poly1305::new(Key::from_slice(key)), counter: 0 }
    }

    fn nonce(&mut self) -> Nonce {
        let mut n = [0u8; 12];
        n[..8].copy_from_slice(&self.counter.to_le_bytes());
        self.counter += 1;
        *Nonce::from_slice(&n)
    }

    fn seal(&mut self, kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut plain = Vec::with_capacity(1 + payload.len());
        plain.push(kind);
        plain.extend_from_slice(payload);
        let nonce = self.nonce();
        self.aead.encrypt(&nonce, plain.as_ref()).expect("chacha20poly1305 encrypt")
    }

    fn open(&mut self, cipher: &[u8]) -> Option<(u8, Vec<u8>)> {
        let nonce = self.nonce();
        let plain = self.aead.decrypt(&nonce, cipher).ok()?;
        let (kind, payload) = plain.split_first()?;
        Some((*kind, payload.to_vec()))
    }
}

async fn write_frame<W: AsyncWriteExt + Unpin>(w: &mut W, sealer: &mut Sealer, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    let sealed = sealer.seal(kind, payload);
    let mut out = Vec::with_capacity(4 + sealed.len());
    out.extend_from_slice(&(sealed.len() as u32).to_le_bytes());
    out.extend_from_slice(&sealed);
    w.write_all(&out).await?;
    w.flush().await
}

async fn read_frame<R: AsyncReadExt + Unpin>(r: &mut R, sealer: &mut Sealer) -> std::io::Result<(u8, Vec<u8>)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    if len < 17 || len > MAX_FRAME {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "frame length"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).await?;
    sealer.open(&buf).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "frame authentication"))
}

/// A frame that cannot be written within this long means the peer stopped reading.
const WRITE_DEADLINE: Duration = Duration::from_secs(30);
/// Longest a shard session lives, whatever its traffic.
const MAX_SESSION: Duration = Duration::from_secs(90 * 60);
/// How long a handshake waits for the head's assignment to reach this miner's template.
const ASSIGNMENT_GRACE: Duration = Duration::from_secs(6);

/// Reads frames in a task of its own: a cancelled read would lose a half-read frame and desync
/// the stream. The channel closes on the first error.
fn spawn_frame_reader(
    mut rd: tokio::net::tcp::OwnedReadHalf,
    mut sealer: Sealer,
) -> (mpsc::Receiver<std::io::Result<(u8, Vec<u8>)>>, tokio::task::JoinHandle<()>) {
    let (tx, rx) = mpsc::channel(64);
    let task = tokio::spawn(async move {
        loop {
            let frame = read_frame(&mut rd, &mut sealer).await;
            let failed = frame.is_err();
            if tx.send(frame).await.is_err() || failed {
                break;
            }
        }
    });
    (rx, task)
}

async fn write_frame_by<W: AsyncWriteExt + Unpin>(w: &mut W, sealer: &mut Sealer, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    match tokio::time::timeout(WRITE_DEADLINE, write_frame(w, sealer, kind, payload)).await {
        Ok(r) => r,
        Err(_) => Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "peer stopped reading")),
    }
}

/// Session keys: one per direction, both derived from the ECDH secret and the transcript.
struct SessionKeys {
    c2s: [u8; 32],
    s2c: [u8; 32],
}

fn session_keys(shared: &SharedSecret, transcript: &[u8; 32]) -> SessionKeys {
    let mut material = Vec::with_capacity(64);
    material.extend_from_slice(shared.as_ref());
    material.extend_from_slice(transcript);
    SessionKeys {
        c2s: blake3::derive_key("Keryx shard gateway client-to-server v1", &material),
        s2c: blake3::derive_key("Keryx shard gateway server-to-client v1", &material),
    }
}

fn transcript(
    model_id: &[u8; 32],
    client_eph: &[u8; 33],
    server_eph: &[u8; 33],
    client_pk: &[u8; 32],
    server_pk: &[u8; 32],
    nonce: &[u8; 32],
    tier: u8,
) -> [u8; 32] {
    domain_hash(b"KeryxGatewayTranscriptV2", &[model_id, client_eph, server_eph, client_pk, server_pk, nonce, &[tier]])
}

fn ephemeral() -> (SecretKey, PublicKey) {
    loop {
        let bytes: [u8; 32] = rand::random();
        if let Ok(secret) = SecretKey::from_slice(&bytes) {
            let public = PublicKey::from_secret_key(secp256k1::SECP256K1, &secret);
            return (secret, public);
        }
    }
}

// ── Advertised endpoint + peer directory ──────────────────────────────────────────────────────

static ADVERTISED: OnceLock<String> = OnceLock::new();

/// The public `host:port` written to the coinbase as `/ai:ep:`.
pub fn set_advertised(endpoint: &str) {
    let _ = ADVERTISED.set(endpoint.to_string());
}

pub fn advertised() -> Option<&'static str> {
    ADVERTISED.get().map(|s| s.as_str())
}

#[derive(Clone, Debug)]
pub struct PeerEntry {
    pub endpoint: String,
    pub caps: Vec<[u8; 32]>,
    pub daa: u64,
}

fn peers() -> &'static Mutex<HashMap<[u8; 32], PeerEntry>> {
    static P: OnceLock<Mutex<HashMap<[u8; 32], PeerEntry>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

fn field_after<'a>(payload: &'a [u8], marker: &[u8]) -> Option<&'a [u8]> {
    let start = payload.windows(marker.len()).position(|w| w == marker)? + marker.len();
    let rest = &payload[start..];
    let end = rest.iter().position(|&b| b == b'/').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// An advertised endpoint is an `ip:port`; off the test networks the address must be public.
fn endpoint_acceptable(endpoint: &str) -> bool {
    let Ok(addr) = endpoint.parse::<std::net::SocketAddr>() else { return false };
    if addr.port() == 0 {
        return false;
    }
    if crate::pom::is_devnet() || crate::pom::is_testnet() {
        return true;
    }
    match addr.ip() {
        std::net::IpAddr::V4(ip) => {
            !(ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast())
        }
        std::net::IpAddr::V6(ip) => {
            let seg = ip.segments()[0];
            !(ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() || (seg & 0xfe00) == 0xfc00 || (seg & 0xffc0) == 0xfe80)
        }
    }
}

/// Records a coinbase payload in the peer directory: escrow key → advertised endpoint and caps.
pub fn note_coinbase(payload: &[u8], daa: u64) {
    let Some(escrow) = field_after(payload, b"/escrow:") else { return };
    let Ok(pk) = hex::decode(escrow) else { return };
    let Ok(pk): Result<[u8; 32], _> = pk.try_into() else { return };
    let Some(ep) = field_after(payload, b"/ai:ep:") else { return };
    let Ok(endpoint) = std::str::from_utf8(ep) else { return };
    if !endpoint_acceptable(endpoint) {
        return;
    }
    let caps: Vec<[u8; 32]> = field_after(payload, b"/ai:cap:")
        .map(|c| c.split(|&b| b == b',').filter_map(|h| hex::decode(h).ok()?.try_into().ok()).collect())
        .unwrap_or_default();
    let mut g = peers().lock().unwrap_or_else(|p| p.into_inner());
    g.retain(|_, e| e.daa + PEER_MEMORY_DAA > daa);
    g.insert(pk, PeerEntry { endpoint: endpoint.to_string(), caps, daa });
}

/// Peers that advertised `model_id` recently: (escrow key, endpoint), newest first.
pub fn peers_for(model_id: &[u8; 32]) -> Vec<([u8; 32], String)> {
    let g = peers().lock().unwrap_or_else(|p| p.into_inner());
    let mut v: Vec<(u64, [u8; 32], String)> =
        g.iter().filter(|(_, e)| e.caps.contains(model_id)).map(|(pk, e)| (e.daa, *pk, e.endpoint.clone())).collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, pk, ep)| (pk, ep)).collect()
}

/// Endpoint a known peer (by escrow key) advertised, if any.
pub fn peer_endpoint(escrow_pubkey: &[u8; 32]) -> Option<String> {
    peers().lock().ok()?.get(escrow_pubkey).map(|e| e.endpoint.clone())
}

// ── Server ────────────────────────────────────────────────────────────────────────────────────

/// Shape of a served layer range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeDims {
    /// Residual floats per token, in and out.
    pub n_hc: usize,
    pub n_vocab: usize,
    /// Tokens one sequence holds.
    pub n_ctx: u32,
    /// The range takes token ids instead of a residual.
    pub from_tokens: bool,
}

/// The layers a link computes for its heads.
pub trait RangeCompute: Send + Sync {
    fn dims(&self) -> RangeDims;
    /// Reserves a sequence.
    fn open(&self) -> Result<u32, String>;
    /// Runs `n` tokens of `seq` from `pos0`: token ids or `n × n_hc` residual floats in, `n × n_hc` floats out.
    fn step(&self, seq: u32, pos0: u32, tokens: Option<&[i32]>, input: Option<&[f32]>) -> Result<Vec<f32>, String>;
    /// Forgets and releases `seq`.
    fn close(&self, seq: u32);
}

/// Brings up the range of a served shard (blocking, may load).
pub type RangeProvider = Arc<dyn Fn() -> Result<Arc<dyn RangeCompute>, String> + Send + Sync>;

/// Resolves a shard model_id to its tier and range — `None` unless this miner mines and serves it.
pub type ShardResolver = Arc<dyn Fn(&[u8; 32]) -> Option<(u8, RangeProvider)> + Send + Sync>;

/// Whether `head` may use this miner's shard of `tier`: it is the declared head of a live
/// assignment where `own` declared that tier, for `request` when given.
pub type HeadAuthorizer = Arc<dyn Fn(&[u8; 32], &[u8; 32], u8, Option<&[u8; 32]>) -> bool + Send + Sync>;

/// The pipeline assignments of the chain tip decide.
pub fn assignment_authorizer() -> HeadAuthorizer {
    Arc::new(|head: &[u8; 32], own: &[u8; 32], tier: u8, request: Option<&[u8; 32]>| crate::pipeline::authorizes(head, own, tier, request))
}

/// Tokens one `STEP` carries at most.
pub const MAX_STEP_TOKENS: usize = 128;
/// How long a head waits for one step of a link.
const STEP_TIMEOUT: Duration = Duration::from_secs(120);

fn open_sessions() -> &'static Mutex<HashMap<[u8; 32], usize>> {
    static S: OnceLock<Mutex<HashMap<[u8; 32], usize>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A slot among the gateway's sessions, released on drop.
struct SessionSlot([u8; 32]);

impl SessionSlot {
    fn take(head: [u8; 32]) -> Option<Self> {
        let mut g = open_sessions().lock().unwrap_or_else(|p| p.into_inner());
        let total: usize = g.values().sum();
        let mine = g.get(&head).copied().unwrap_or(0);
        if total >= MAX_SESSIONS || mine >= MAX_SESSIONS_PER_HEAD {
            return None;
        }
        *g.entry(head).or_insert(0) += 1;
        Some(SessionSlot(head))
    }
}

impl Drop for SessionSlot {
    fn drop(&mut self) {
        let mut g = open_sessions().lock().unwrap_or_else(|p| p.into_inner());
        if let Some(n) = g.get_mut(&self.0) {
            *n -= 1;
            if *n == 0 {
                g.remove(&self.0);
            }
        }
    }
}

/// Last time each (head, request) computed through this gateway.
fn computed() -> &'static Mutex<HashMap<([u8; 32], [u8; 32]), std::time::Instant>> {
    static D: OnceLock<Mutex<HashMap<([u8; 32], [u8; 32]), std::time::Instant>>> = OnceLock::new();
    D.get_or_init(|| Mutex::new(HashMap::new()))
}

fn note_computed(head: &[u8; 32], request: &[u8; 32]) {
    let mut g = computed().lock().unwrap_or_else(|p| p.into_inner());
    g.retain(|_, t| t.elapsed() < SIGN_AFTER_DATA);
    g.insert((*head, *request), std::time::Instant::now());
}

fn computed_for(head: &[u8; 32], request: &[u8; 32]) -> bool {
    computed().lock().map(|g| g.get(&(*head, *request)).is_some_and(|t| t.elapsed() < SIGN_AFTER_DATA)).unwrap_or(false)
}

/// The one message signed per (request, head): a retry of the same message is signed again,
/// another one is refused.
fn signed_once() -> &'static Mutex<HashMap<([u8; 32], [u8; 32]), [u8; 32]>> {
    static S: OnceLock<Mutex<HashMap<([u8; 32], [u8; 32]), [u8; 32]>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

fn may_sign(request: [u8; 32], head: [u8; 32], digest: [u8; 32]) -> bool {
    let mut g = signed_once().lock().unwrap_or_else(|p| p.into_inner());
    if g.len() >= 10_000 {
        g.clear();
    }
    *g.entry((request, head)).or_insert(digest) == digest
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Input of one step.
#[derive(Debug, PartialEq)]
pub enum StepInput {
    Tokens(Vec<i32>),
    Residual(Vec<f32>),
}

impl StepInput {
    fn len(&self, n_hc: usize) -> usize {
        match self {
            StepInput::Tokens(t) => t.len(),
            StepInput::Residual(r) => r.len() / n_hc.max(1),
        }
    }

    fn encode(&self, pos0: u32, n_hc: usize) -> Vec<u8> {
        let n = self.len(n_hc) as u32;
        let mut out = Vec::with_capacity(8 + n as usize * 4 * n_hc.max(1));
        out.extend_from_slice(&pos0.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        match self {
            StepInput::Tokens(t) => t.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            StepInput::Residual(r) => out.extend_from_slice(&f32s_to_bytes(r)),
        }
        out
    }
}

/// Decodes and checks a `STEP` against the open sequence: holds `len` tokens of `max_len`.
fn parse_step(payload: &[u8], dims: &RangeDims, len: u32, max_len: u32) -> Result<(u32, StepInput), &'static str> {
    if payload.len() < 8 {
        return Err("step: short frame");
    }
    let pos0 = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let n = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    if n == 0 || n > MAX_STEP_TOKENS {
        return Err("step: token count");
    }
    if pos0 != len {
        return Err("step: position");
    }
    if pos0 as u64 + n as u64 > max_len as u64 {
        return Err("step: beyond the session length");
    }
    let data = &payload[8..];
    let per_token = if dims.from_tokens { 4 } else { dims.n_hc * 4 };
    if per_token == 0 || data.len() != n * per_token {
        return Err("step: size");
    }
    if dims.from_tokens {
        let ids: Vec<i32> = data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if ids.iter().any(|&t| t < 0 || t as usize >= dims.n_vocab) {
            return Err("step: token id");
        }
        Ok((pos0, StepInput::Tokens(ids)))
    } else {
        let x = bytes_to_f32s(data);
        if x.iter().any(|v| !v.is_finite()) {
            return Err("step: non-finite residual");
        }
        Ok((pos0, StepInput::Residual(x)))
    }
}

/// Runs the gateway listener until the process ends.
pub async fn serve(listen: String, identity: Arc<GatewayIdentity>, resolve: ShardResolver, authorize: HeadAuthorizer) -> Result<(), String> {
    let listener = TcpListener::bind(&listen).await.map_err(|e| format!("gateway bind {}: {}", listen, e))?;
    log::info!("shard gateway: listening on {}", listen);
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(x) => {
                let _ = x.0.set_nodelay(true);
                x
            }
            Err(e) => {
                log::warn!("shard gateway: accept failed: {}", e);
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
        };
        let identity = identity.clone();
        let resolve = resolve.clone();
        let authorize = authorize.clone();
        tokio::spawn(async move {
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, server_handshake(stream, &identity, &resolve, &authorize)).await {
                Ok(Ok((stream, keys, provider, client_pk, tier, slot))) => {
                    log::debug!("shard gateway: session from {} (head {})", peer, hex::encode(client_pk));
                    if let Err(e) = server_session(stream, keys, provider, identity, tier, client_pk, authorize).await {
                        log::debug!("shard gateway: session {} ended: {}", peer, e);
                    }
                    drop(slot);
                }
                Ok(Err(e)) => log::debug!("shard gateway: handshake from {} rejected: {}", peer, e),
                Err(_) => log::debug!("shard gateway: handshake from {} timed out", peer),
            }
        });
    }
}

async fn server_handshake(
    mut stream: TcpStream,
    identity: &GatewayIdentity,
    resolve: &ShardResolver,
    authorize: &HeadAuthorizer,
) -> Result<(TcpStream, SessionKeys, RangeProvider, [u8; 32], u8, SessionSlot), String> {
    let nonce: [u8; 32] = rand::random();
    let mut challenge = Vec::with_capacity(CHALLENGE_LEN);
    challenge.extend_from_slice(MAGIC);
    challenge.push(VERSION);
    challenge.extend_from_slice(&nonce);
    stream.write_all(&challenge).await.map_err(|e| e.to_string())?;
    let mut hello = [0u8; HELLO_LEN];
    stream.read_exact(&mut hello).await.map_err(|e| e.to_string())?;
    if &hello[..4] != MAGIC || hello[4] != VERSION {
        return Err("bad magic or version".into());
    }
    let model_id: [u8; 32] = hello[5..37].try_into().unwrap();
    let client_eph: [u8; 33] = hello[37..70].try_into().unwrap();
    let client_pk: [u8; 32] = hello[70..102].try_into().unwrap();
    let expected_pk: [u8; 32] = hello[102..134].try_into().unwrap();
    let client_sig: [u8; 64] = hello[134..198].try_into().unwrap();
    if expected_pk != identity.pubkey {
        return Err("hello addressed to another key".into());
    }
    if !verify_schnorr(&client_pk, &client_sig, &domain_hash(CLIENT_DOMAIN, &[&model_id, &client_eph, &nonce, &expected_pk])) {
        return Err("client signature".into());
    }
    let client_eph_pk = PublicKey::from_slice(&client_eph).map_err(|_| "client ephemeral key".to_string())?;
    let refuse = |status: u8| {
        let mut reply = vec![status];
        reply.resize(REPLY_LEN, 0);
        reply
    };
    let Some((tier, provider)) = resolve(&model_id) else {
        let _ = stream.write_all(&refuse(STATUS_NO_SHARD)).await;
        return Err(format!("no shard {} served here", hex::encode(model_id)));
    };
    let grace = tokio::time::Instant::now() + ASSIGNMENT_GRACE;
    while !authorize(&client_pk, &identity.pubkey, tier, None) && tokio::time::Instant::now() < grace {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    if !authorize(&client_pk, &identity.pubkey, tier, None) {
        let _ = stream.write_all(&refuse(STATUS_NOT_ASSIGNED)).await;
        return Err(format!("head {} is not assigned to this shard", hex::encode(client_pk)));
    }
    let Some(slot) = SessionSlot::take(client_pk) else {
        let _ = stream.write_all(&refuse(STATUS_BUSY)).await;
        return Err("session limit".into());
    };
    let (eph_secret, eph_public) = ephemeral();
    let server_eph = eph_public.serialize();
    let sig = identity.sign_digest(&domain_hash(SERVER_DOMAIN, &[&model_id, &client_eph, &server_eph, &client_pk, &nonce, &[tier]]));
    let mut reply = Vec::with_capacity(REPLY_LEN);
    reply.push(STATUS_OK);
    reply.extend_from_slice(&server_eph);
    reply.extend_from_slice(&identity.pubkey);
    reply.push(tier);
    reply.extend_from_slice(&sig);
    stream.write_all(&reply).await.map_err(|e| e.to_string())?;
    let shared = SharedSecret::new(&client_eph_pk, &eph_secret);
    let keys = session_keys(&shared, &transcript(&model_id, &client_eph, &server_eph, &client_pk, &identity.pubkey, &nonce, tier));
    Ok((stream, keys, provider, client_pk, tier, slot))
}

/// The sequence a head holds on this link.
struct OpenRange {
    request: [u8; 32],
    max_len: u32,
    len: u32,
    seq: u32,
    compute: Arc<dyn RangeCompute>,
    dims: RangeDims,
}

async fn server_session(
    stream: TcpStream,
    keys: SessionKeys,
    provider: RangeProvider,
    identity: Arc<GatewayIdentity>,
    tier: u8,
    head_pubkey: [u8; 32],
    authorize: HeadAuthorizer,
) -> std::io::Result<()> {
    let (rd, mut wr) = stream.into_split();
    let (mut frames, reader) = spawn_frame_reader(rd, Sealer::new(&keys.c2s));
    let mut tx = Sealer::new(&keys.s2c);
    let (out_tx, mut out_rx) = mpsc::channel::<(u8, Vec<u8>)>(64);
    // single writer of the encrypted side; it stops when the head stops reading
    let mut writer = tokio::spawn(async move {
        while let Some((kind, payload)) = out_rx.recv().await {
            if write_frame_by(&mut wr, &mut tx, kind, &payload).await.is_err() {
                break;
            }
        }
    });
    let mut open: Option<OpenRange> = None;
    let mut pause: Option<ShardSessionGuard> = None;
    let mut reauth = tokio::time::interval(reauth_interval());
    reauth.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    reauth.tick().await;
    let end_of_life = tokio::time::sleep(MAX_SESSION);
    tokio::pin!(end_of_life);
    let mut idle_until = tokio::time::Instant::now() + session_idle_timeout();
    let refused = |msg: &'static str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    let result: std::io::Result<()> = loop {
        let frame = tokio::select! {
            f = frames.recv() => f,
            _ = reauth.tick() => {
                if !authorize(&head_pubkey, &identity.pubkey, tier, None) {
                    break Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "head no longer assigned"));
                }
                continue;
            }
            _ = tokio::time::sleep_until(idle_until) => {
                break Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "head silent, session dropped"));
            }
            _ = &mut end_of_life => {
                break Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "session lifetime reached"));
            }
            _ = &mut writer => {
                break Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "head stopped reading"));
            }
        };
        let (kind, payload) = match frame {
            Some(Ok(f)) => f,
            Some(Err(e)) => break Err(e),
            None => break Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "head closed")),
        };
        idle_until = tokio::time::Instant::now() + session_idle_timeout();
        match kind {
            FRAME_OPEN => {
                let msg: Option<&'static str> = if open.is_some() {
                    Some("open: a sequence is already open")
                } else if payload.len() != 36 {
                    Some("open: bad length")
                } else {
                    None
                };
                if let Some(msg) = msg {
                    let _ = out_tx.send((FRAME_ERROR, msg.as_bytes().to_vec())).await;
                    break Err(refused(msg));
                }
                let request: [u8; 32] = payload[..32].try_into().unwrap();
                let max_len = u32::from_le_bytes(payload[32..36].try_into().unwrap());
                if !authorize(&head_pubkey, &identity.pubkey, tier, Some(&request)) {
                    let _ = out_tx.send((FRAME_ERROR, b"open: request not assigned".to_vec())).await;
                    break Err(refused("open: request not assigned"));
                }
                let p = provider.clone();
                let compute = match tokio::task::spawn_blocking(move || p()).await {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => {
                        log::warn!("shard gateway: range unavailable: {}", e);
                        let _ = out_tx.send((FRAME_ERROR, b"open: range unavailable".to_vec())).await;
                        break Err(refused("open: range unavailable"));
                    }
                    Err(_) => {
                        let _ = out_tx.send((FRAME_ERROR, b"open: range unavailable".to_vec())).await;
                        break Err(refused("open: range unavailable"));
                    }
                };
                let dims = compute.dims();
                if max_len == 0 || max_len > dims.n_ctx {
                    let _ = out_tx.send((FRAME_ERROR, b"open: length".to_vec())).await;
                    break Err(refused("open: length"));
                }
                let seq = match compute.open() {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = out_tx.send((FRAME_ERROR, b"open: busy".to_vec())).await;
                        break Err(refused("open: busy"));
                    }
                };
                let mut reply = Vec::with_capacity(9);
                reply.extend_from_slice(&(dims.n_hc as u32).to_le_bytes());
                reply.extend_from_slice(&(dims.n_vocab as u32).to_le_bytes());
                reply.push(dims.from_tokens as u8);
                open = Some(OpenRange { request, max_len, len: 0, seq, compute, dims });
                if out_tx.send((FRAME_OPENED, reply)).await.is_err() {
                    break Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "head stopped reading"));
                }
            }
            FRAME_STEP => {
                let Some(o) = open.as_mut() else {
                    let _ = out_tx.send((FRAME_ERROR, b"step: no open sequence".to_vec())).await;
                    break Err(refused("step: no open sequence"));
                };
                let (pos0, input) = match parse_step(&payload, &o.dims, o.len, o.max_len) {
                    Ok(x) => x,
                    Err(msg) => {
                        let _ = out_tx.send((FRAME_ERROR, msg.as_bytes().to_vec())).await;
                        break Err(refused(msg));
                    }
                };
                if pause.is_none() {
                    pause = Some(ShardSessionGuard::new());
                }
                let n = input.len(o.dims.n_hc) as u32;
                let (compute, seq) = (o.compute.clone(), o.seq);
                let out = tokio::task::spawn_blocking(move || match &input {
                    StepInput::Tokens(t) => compute.step(seq, pos0, Some(t), None),
                    StepInput::Residual(r) => compute.step(seq, pos0, None, Some(r)),
                })
                .await;
                let out = match out {
                    Ok(Ok(v)) => v,
                    Ok(Err(e)) => {
                        log::warn!("shard gateway: step failed: {}", e);
                        let _ = out_tx.send((FRAME_ERROR, b"step: compute failed".to_vec())).await;
                        break Err(refused("step: compute failed"));
                    }
                    Err(_) => {
                        let _ = out_tx.send((FRAME_ERROR, b"step: compute failed".to_vec())).await;
                        break Err(refused("step: compute failed"));
                    }
                };
                o.len += n;
                note_computed(&head_pubkey, &o.request);
                if out_tx.send((FRAME_RESULT, f32s_to_bytes(&out))).await.is_err() {
                    break Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "head stopped reading"));
                }
            }
            FRAME_CLOSE => {
                if let Some(o) = open.take() {
                    o.compute.close(o.seq);
                }
                pause = None;
            }
            FRAME_PING => {
                let _ = out_tx.send((FRAME_PONG, payload)).await;
            }
            FRAME_SIGN => {
                // The served tier and the authenticated head are bound here, not taken from the head;
                // the request must be assigned to both, and the head must have computed it through us.
                if payload.len() < keryx_inference::AI_RESPONSE_PAYLOAD_LEN || payload.len() > keryx_inference::MAX_AI_RESPONSE_PAYLOAD_LEN {
                    let _ = out_tx.send((FRAME_ERROR, b"sign: bad length".to_vec())).await;
                    continue;
                }
                let request: [u8; 32] = payload[..32].try_into().unwrap();
                let digest = link_digest(tier, &head_pubkey, &payload);
                let refusal: Option<&[u8]> = if !authorize(&head_pubkey, &identity.pubkey, tier, Some(&request)) {
                    Some(b"sign: request not assigned")
                } else if !computed_for(&head_pubkey, &request) {
                    Some(b"sign: request not computed through this link")
                } else if !may_sign(request, head_pubkey, digest) {
                    Some(b"sign: another response already signed")
                } else {
                    None
                };
                if let Some(msg) = refusal {
                    let _ = out_tx.send((FRAME_ERROR, msg.to_vec())).await;
                    continue;
                }
                let sig = identity.sign_digest(&digest);
                let mut reply = Vec::with_capacity(32 + 64);
                reply.extend_from_slice(&identity.pubkey);
                reply.extend_from_slice(&sig);
                let _ = out_tx.send((FRAME_SIGNED, reply)).await;
            }
            _ => {
                let _ = out_tx.send((FRAME_ERROR, b"unknown frame".to_vec())).await;
                break Err(refused("unknown frame"));
            }
        }
    };
    reader.abort();
    if let Some(o) = open.take() {
        o.compute.close(o.seq);
    }
    drop(pause);
    drop(out_tx);
    if !writer.is_finished() {
        let _ = tokio::time::timeout(WRITE_DEADLINE, &mut writer).await;
        writer.abort();
    }
    result
}

// ── Client (pipeline head) ─────────────────────────────────────────────────────────────────────

type Reply = oneshot::Sender<Result<(u8, Vec<u8>), String>>;

enum Ctl {
    /// A frame answered by exactly one frame.
    Call(u8, Vec<u8>, Reply),
    /// A frame with no answer.
    Send(u8, Vec<u8>),
    Ping(oneshot::Sender<Result<Duration, String>>),
}

enum Pending {
    Ping(std::time::Instant, Option<oneshot::Sender<Result<Duration, String>>>),
    Reply(Reply),
}

/// What a link reports when its sequence opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkDims {
    pub n_hc: usize,
    pub n_vocab: usize,
    pub from_tokens: bool,
}

/// A shard reached through its gateway: one encrypted connection carrying its sequence's steps,
/// pings and the link signature.
pub struct LinkSession {
    pub remote: String,
    pub model_id: [u8; 32],
    pub tier: u8,
    pub peer_pubkey: [u8; 32],
    /// This head's escrow key, which the link binds into its signature.
    pub head_pubkey: [u8; 32],
    ctl: mpsc::Sender<Ctl>,
    dims: Mutex<Option<LinkDims>>,
}

struct Tunnel {
    stream: TcpStream,
    keys: SessionKeys,
}

async fn client_handshake(
    remote: &str,
    model_id: &[u8; 32],
    identity: &GatewayIdentity,
    expected_pk: &[u8; 32],
    expected_tier: u8,
) -> Result<Tunnel, String> {
    let mut stream = tokio::time::timeout(HANDSHAKE_TIMEOUT, TcpStream::connect(remote))
        .await
        .map_err(|_| format!("connect {}: timeout", remote))?
        .map_err(|e| format!("connect {}: {}", remote, e))?;
    let _ = stream.set_nodelay(true);
    let mut challenge = [0u8; CHALLENGE_LEN];
    tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.read_exact(&mut challenge))
        .await
        .map_err(|_| "handshake challenge timeout".to_string())?
        .map_err(|e| e.to_string())?;
    if &challenge[..4] != MAGIC || challenge[4] != VERSION {
        return Err(format!("gateway {}: bad magic or version", remote));
    }
    let nonce: [u8; 32] = challenge[5..37].try_into().unwrap();
    let (eph_secret, eph_public) = ephemeral();
    let client_eph = eph_public.serialize();
    let sig = identity.sign_digest(&domain_hash(CLIENT_DOMAIN, &[model_id, &client_eph, &nonce, expected_pk]));
    let mut hello = Vec::with_capacity(HELLO_LEN);
    hello.extend_from_slice(MAGIC);
    hello.push(VERSION);
    hello.extend_from_slice(model_id);
    hello.extend_from_slice(&client_eph);
    hello.extend_from_slice(&identity.pubkey);
    hello.extend_from_slice(expected_pk);
    hello.extend_from_slice(&sig);
    stream.write_all(&hello).await.map_err(|e| e.to_string())?;
    let mut reply = [0u8; REPLY_LEN];
    // the link may wait for the assignment to reach its own template before answering
    tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.read_exact(&mut reply))
        .await
        .map_err(|_| "handshake reply timeout".to_string())?
        .map_err(|e| e.to_string())?;
    if reply[0] != STATUS_OK {
        return Err(format!("gateway {} refused: status {}", remote, reply[0]));
    }
    let server_eph: [u8; 33] = reply[1..34].try_into().unwrap();
    let server_pk: [u8; 32] = reply[34..66].try_into().unwrap();
    let tier = reply[66];
    let server_sig: [u8; 64] = reply[67..131].try_into().unwrap();
    if server_pk != *expected_pk || tier != expected_tier {
        return Err(format!("gateway {} is not the expected link", remote));
    }
    if !verify_schnorr(&server_pk, &server_sig, &domain_hash(SERVER_DOMAIN, &[model_id, &client_eph, &server_eph, &identity.pubkey, &nonce, &[tier]])) {
        return Err("gateway signature".into());
    }
    let server_eph_pk = PublicKey::from_slice(&server_eph).map_err(|_| "gateway ephemeral key".to_string())?;
    let shared = SharedSecret::new(&server_eph_pk, &eph_secret);
    let keys = session_keys(&shared, &transcript(model_id, &client_eph, &server_eph, &identity.pubkey, &server_pk, &nonce, tier));
    Ok(Tunnel { stream, keys })
}

/// Connects to `remote` for `model_id`, which must be served by `peer_pubkey` as `tier`.
pub async fn connect_link(remote: &str, model_id: [u8; 32], identity: Arc<GatewayIdentity>, peer_pubkey: [u8; 32], tier: u8) -> Result<LinkSession, String> {
    let t = client_handshake(remote, &model_id, &identity, &peer_pubkey, tier).await?;
    let (ctl_tx, ctl_rx) = mpsc::channel::<Ctl>(16);
    tokio::spawn(control_loop(t, ctl_rx));
    Ok(LinkSession {
        remote: remote.to_string(),
        model_id,
        tier,
        peer_pubkey,
        head_pubkey: identity.pubkey,
        ctl: ctl_tx,
        dims: Mutex::new(None),
    })
}

/// Requests go out in order and the link answers them in order: one queue matches the replies.
async fn control_loop(t: Tunnel, mut ctl: mpsc::Receiver<Ctl>) {
    let (rd, mut wr) = t.stream.into_split();
    let mut tx = Sealer::new(&t.keys.c2s);
    let (mut frames, reader) = spawn_frame_reader(rd, Sealer::new(&t.keys.s2c));
    let mut pending: std::collections::VecDeque<Pending> = Default::default();
    let mut keepalive = tokio::time::interval(CONTROL_KEEPALIVE);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    keepalive.tick().await;
    loop {
        tokio::select! {
            _ = keepalive.tick() => {
                if write_frame_by(&mut wr, &mut tx, FRAME_PING, &[]).await.is_err() {
                    break;
                }
                pending.push_back(Pending::Ping(std::time::Instant::now(), None));
            }
            req = ctl.recv() => {
                let Some(req) = req else { break };
                match req {
                    Ctl::Call(kind, bytes, reply) => {
                        if write_frame_by(&mut wr, &mut tx, kind, &bytes).await.is_err() {
                            let _ = reply.send(Err("link connection closed".into()));
                            break;
                        }
                        pending.push_back(Pending::Reply(reply));
                    }
                    Ctl::Send(kind, bytes) => {
                        if write_frame_by(&mut wr, &mut tx, kind, &bytes).await.is_err() {
                            break;
                        }
                    }
                    Ctl::Ping(reply) => {
                        if write_frame_by(&mut wr, &mut tx, FRAME_PING, &[]).await.is_err() {
                            let _ = reply.send(Err("link connection closed".into()));
                            break;
                        }
                        pending.push_back(Pending::Ping(std::time::Instant::now(), Some(reply)));
                    }
                }
            }
            frame = frames.recv() => {
                let Some(Ok((kind, payload))) = frame else { break };
                match (pending.pop_front(), kind) {
                    (Some(Pending::Ping(t0, reply)), FRAME_PONG) => {
                        if let Some(reply) = reply {
                            let _ = reply.send(Ok(t0.elapsed()));
                        }
                    }
                    (Some(Pending::Reply(reply)), FRAME_ERROR) => {
                        let _ = reply.send(Err(String::from_utf8_lossy(&payload).into_owned()));
                    }
                    (Some(Pending::Reply(reply)), _) if kind != FRAME_PONG => {
                        let _ = reply.send(Ok((kind, payload)));
                    }
                    _ => break,
                }
            }
        }
    }
    reader.abort();
    for p in pending.drain(..) {
        match p {
            Pending::Reply(r) => {
                let _ = r.send(Err("link connection closed".into()));
            }
            Pending::Ping(_, Some(r)) => {
                let _ = r.send(Err("link connection closed".into()));
            }
            Pending::Ping(_, None) => {}
        }
    }
}

impl LinkSession {
    async fn call(&self, kind: u8, payload: Vec<u8>, timeout: Duration) -> Result<(u8, Vec<u8>), String> {
        let (tx, rx) = oneshot::channel();
        self.ctl.send(Ctl::Call(kind, payload, tx)).await.map_err(|_| "link connection closed".to_string())?;
        tokio::time::timeout(timeout, rx).await.map_err(|_| "link timeout".to_string())?.map_err(|_| "link connection closed".to_string())?
    }

    /// Opens this head's sequence of `max_len` tokens for `request` on the link.
    pub async fn open(&self, request: &[u8; 32], max_len: u32) -> Result<LinkDims, String> {
        let mut payload = request.to_vec();
        payload.extend_from_slice(&max_len.to_le_bytes());
        let (kind, reply) = self.call(FRAME_OPEN, payload, STEP_TIMEOUT).await?;
        if kind != FRAME_OPENED || reply.len() != 9 {
            return Err("bad open reply".into());
        }
        let dims = LinkDims {
            n_hc: u32::from_le_bytes(reply[0..4].try_into().unwrap()) as usize,
            n_vocab: u32::from_le_bytes(reply[4..8].try_into().unwrap()) as usize,
            from_tokens: reply[8] != 0,
        };
        if dims.n_hc == 0 {
            return Err("bad open reply".into());
        }
        *self.dims.lock().unwrap_or_else(|p| p.into_inner()) = Some(dims);
        Ok(dims)
    }

    /// Runs `input` at `pos0` through the link's layers: its `n × n_hc` residual floats.
    pub async fn step(&self, pos0: u32, input: &StepInput) -> Result<Vec<f32>, String> {
        let dims = (*self.dims.lock().unwrap_or_else(|p| p.into_inner())).ok_or("link sequence not open")?;
        let n = input.len(dims.n_hc);
        let (kind, reply) = self.call(FRAME_STEP, input.encode(pos0, dims.n_hc), STEP_TIMEOUT).await?;
        if kind != FRAME_RESULT || reply.len() != n * dims.n_hc * 4 {
            return Err("bad step reply".into());
        }
        Ok(bytes_to_f32s(&reply))
    }

    /// Releases the link's sequence; the connection stays up for the signature.
    pub async fn close(&self) {
        let _ = self.ctl.send(Ctl::Send(FRAME_CLOSE, Vec::new())).await;
    }

    /// Has the link sign `message` (the response's v1 and extension bytes) for its tier under
    /// this head.
    pub async fn sign(&self, message: &[u8]) -> Result<keryx_inference::AiResponseLink, String> {
        let (kind, payload) = self.call(FRAME_SIGN, message.to_vec(), Duration::from_secs(30)).await?;
        if kind != FRAME_SIGNED || payload.len() != 96 {
            return Err("bad signature frame".into());
        }
        let link = keryx_inference::AiResponseLink {
            tier: self.tier,
            escrow_pubkey: payload[..32].try_into().unwrap(),
            signature: payload[32..96].try_into().unwrap(),
        };
        if link.escrow_pubkey != self.peer_pubkey || !verify_schnorr(&link.escrow_pubkey, &link.signature, &link_digest(self.tier, &self.head_pubkey, message)) {
            return Err("link signature does not verify".into());
        }
        Ok(link)
    }

    /// Round-trip time of the link connection.
    pub async fn ping(&self) -> Result<Duration, String> {
        let (tx, rx) = oneshot::channel();
        self.ctl.send(Ctl::Ping(tx)).await.map_err(|_| "link connection closed".to_string())?;
        tokio::time::timeout(Duration::from_secs(10), rx).await.map_err(|_| "ping timeout".to_string())?.map_err(|_| "link connection closed".to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(seed: u8) -> Arc<GatewayIdentity> {
        Arc::new(GatewayIdentity::from_privkey_hex(&hex::encode([seed; 32])).unwrap())
    }

    #[test]
    fn frames_roundtrip_and_reject_tampering() {
        let key = [7u8; 32];
        let mut a = Sealer::new(&key);
        let mut b = Sealer::new(&key);
        let sealed = a.seal(FRAME_STEP, b"hello");
        assert_eq!(b.open(&sealed), Some((FRAME_STEP, b"hello".to_vec())));
        let mut tampered = a.seal(FRAME_PING, b"x");
        tampered[3] ^= 1;
        assert!(b.open(&tampered).is_none());
    }

    #[test]
    fn coinbase_directory_parses_endpoint_and_caps() {
        let pk = [0x11u8; 32];
        let cap = [0x22u8; 32];
        let payload = format!(
            "\x00\x01binaryhead/escrow:{}/esig:abcd/1234/ai:v1:ffff/ai:cap:{},{}/ai:ep:203.0.113.5:60052",
            hex::encode(pk),
            hex::encode(cap),
            hex::encode([0x33u8; 32])
        );
        note_coinbase(payload.as_bytes(), 100);
        assert_eq!(peer_endpoint(&pk).as_deref(), Some("203.0.113.5:60052"));
        assert_eq!(peers_for(&cap), vec![(pk, "203.0.113.5:60052".to_string())]);
        // a coinbase without an endpoint is not a peer
        note_coinbase(format!("/escrow:{}/ai:cap:{}", hex::encode([0x44u8; 32]), hex::encode(cap)).as_bytes(), 101);
        assert_eq!(peers_for(&cap).len(), 1);
        // hostnames and non-public addresses are not endpoints
        for ep in ["host:60052", "127.0.0.1:60052", "10.1.2.3:60052", "192.168.1.5:1", "169.254.0.1:1", "[fe80::1]:1", "203.0.113.5:0"] {
            assert!(!endpoint_acceptable(ep), "{ep}");
        }
        note_coinbase(format!("/escrow:{}/ai:ep:10.0.0.7:60052", hex::encode([0x66u8; 32])).as_bytes(), 102);
        assert!(peer_endpoint(&[0x66u8; 32]).is_none());
        // old entries expire
        note_coinbase(format!("/escrow:{}/ai:ep:198.51.100.7:1", hex::encode([0x55u8; 32])).as_bytes(), 100 + PEER_MEMORY_DAA + 1);
        assert!(peer_endpoint(&pk).is_none());
    }

    const HC: usize = 4;

    fn dims(from_tokens: bool) -> RangeDims {
        RangeDims { n_hc: HC, n_vocab: 100, n_ctx: 64, from_tokens }
    }

    #[test]
    fn steps_are_checked_before_any_compute() {
        let d = dims(true);
        let ok = StepInput::Tokens(vec![1, 2, 3]).encode(5, HC);
        assert_eq!(parse_step(&ok, &d, 5, 64), Ok((5, StepInput::Tokens(vec![1, 2, 3]))));
        assert_eq!(parse_step(&ok, &d, 4, 64), Err("step: position"));
        assert_eq!(parse_step(&ok, &d, 5, 7), Err("step: beyond the session length"));
        assert_eq!(parse_step(&ok[..ok.len() - 1], &d, 5, 64), Err("step: size"));
        assert_eq!(parse_step(&ok[..6], &d, 5, 64), Err("step: short frame"));
        assert_eq!(parse_step(&StepInput::Tokens(vec![100]).encode(0, HC), &d, 0, 64), Err("step: token id"));
        assert_eq!(parse_step(&StepInput::Tokens(vec![-1]).encode(0, HC), &d, 0, 64), Err("step: token id"));
        assert_eq!(parse_step(&StepInput::Tokens(vec![]).encode(0, HC), &d, 0, 64), Err("step: token count"));
        let many = StepInput::Tokens(vec![1; MAX_STEP_TOKENS + 1]).encode(0, HC);
        assert_eq!(parse_step(&many, &dims(true), 0, 4096), Err("step: token count"));
        // a residual range takes exactly n × n_hc finite floats
        let r = dims(false);
        let x = StepInput::Residual(vec![0.5; 2 * HC]).encode(0, HC);
        assert_eq!(parse_step(&x, &r, 0, 64), Ok((0, StepInput::Residual(vec![0.5; 2 * HC]))));
        let mut nan = vec![0.5; 2 * HC];
        nan[3] = f32::NAN;
        assert_eq!(parse_step(&StepInput::Residual(nan).encode(0, HC), &r, 0, 64), Err("step: non-finite residual"));
        // tokens sent to a residual range do not have the residual's size
        assert_eq!(parse_step(&ok, &r, 5, 64), Err("step: size"));
    }

    /// Adds `bias` to every residual float (or turns ids into floats); two sequences.
    struct FakeRange {
        from_tokens: bool,
        bias: f32,
        n_ctx: u32,
        seqs: Mutex<[bool; 2]>,
        steps: std::sync::atomic::AtomicUsize,
    }

    impl FakeRange {
        fn new(from_tokens: bool, bias: f32) -> Arc<Self> {
            Self::with_ctx(from_tokens, bias, 64)
        }
        fn with_ctx(from_tokens: bool, bias: f32, n_ctx: u32) -> Arc<Self> {
            Arc::new(Self { from_tokens, bias, n_ctx, seqs: Mutex::new([false; 2]), steps: Default::default() })
        }
        fn held(&self) -> usize {
            self.seqs.lock().unwrap().iter().filter(|s| **s).count()
        }
    }

    impl RangeCompute for FakeRange {
        fn dims(&self) -> RangeDims {
            RangeDims { n_ctx: self.n_ctx, ..dims(self.from_tokens) }
        }
        fn open(&self) -> Result<u32, String> {
            let mut g = self.seqs.lock().unwrap();
            let i = g.iter().position(|s| !*s).ok_or("busy")?;
            g[i] = true;
            Ok(i as u32)
        }
        fn step(&self, _seq: u32, _pos0: u32, tokens: Option<&[i32]>, input: Option<&[f32]>) -> Result<Vec<f32>, String> {
            self.steps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(match (tokens, input) {
                (Some(t), None) => t.iter().flat_map(|&id| std::iter::repeat(id as f32 + self.bias).take(HC)).collect(),
                (None, Some(x)) => x.iter().map(|v| v + self.bias).collect(),
                _ => return Err("bad input".into()),
            })
        }
        fn close(&self, seq: u32) {
            self.seqs.lock().unwrap()[seq as usize] = false;
        }
    }

    /// Tests that compute share the process-wide pause counter: one at a time.
    fn session_tests() -> &'static tokio::sync::Mutex<()> {
        static L: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
        L.get_or_init(|| tokio::sync::Mutex::new(()))
    }

    fn allow_all() -> HeadAuthorizer {
        Arc::new(|_: &[u8; 32], _: &[u8; 32], _: u8, _: Option<&[u8; 32]>| true)
    }

    async fn gateway(shard: Arc<GatewayIdentity>, model: [u8; 32], range: Arc<FakeRange>, authorize: HeadAuthorizer) -> String {
        let gw = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gw_addr = gw.local_addr().unwrap().to_string();
        drop(gw);
        let resolve: ShardResolver = Arc::new(move |id: &[u8; 32]| {
            let r = range.clone();
            let provider: RangeProvider = Arc::new(move || Ok(r.clone() as Arc<dyn RangeCompute>));
            (*id == model).then_some((8u8, provider))
        });
        let listen = gw_addr.clone();
        tokio::spawn(async move { serve(listen, shard, resolve, authorize).await.unwrap() });
        tokio::time::sleep(Duration::from_millis(100)).await;
        gw_addr
    }

    #[tokio::test]
    async fn silent_head_releases_the_shard_session() {
        let _serial = session_tests().lock().await;
        SESSION_IDLE_MS.store(300, std::sync::atomic::Ordering::Relaxed);
        let shard = identity(3);
        let head = identity(4);
        let model = [0xCDu8; 32];
        let range = FakeRange::new(true, 0.0);
        let gw_addr = gateway(shard.clone(), model, range.clone(), allow_all()).await;
        let before = crate::pom_gpu::shard_sessions();
        let link = connect_link(&gw_addr, model, head, shard.pubkey, 8).await.unwrap();
        // opening alone does not pause the GPU
        link.open(&[1u8; 32], 32).await.unwrap();
        assert_eq!(range.held(), 1);
        assert_eq!(crate::pom_gpu::shard_sessions(), before);
        // the first step does, until the head goes silent
        link.step(0, &StepInput::Tokens(vec![3])).await.unwrap();
        assert_eq!(crate::pom_gpu::shard_sessions(), before + 1);
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert_eq!(crate::pom_gpu::shard_sessions(), before);
        assert_eq!(range.held(), 0);
        SESSION_IDLE_MS.store(120_000, std::sync::atomic::Ordering::Relaxed);
    }

    #[tokio::test]
    async fn frequent_reauthorization_keeps_a_streaming_session_intact() {
        let _serial = session_tests().lock().await;
        REAUTH_MS.store(5, std::sync::atomic::Ordering::Relaxed);
        let shard = identity(8);
        let head = identity(9);
        let model = [0xEFu8; 32];
        let range = FakeRange::with_ctx(false, 1.0, 4096);
        let gw_addr = gateway(shard.clone(), model, range, allow_all()).await;
        let link = connect_link(&gw_addr, model, head, shard.pubkey, 8).await.unwrap();
        link.open(&[2u8; 32], 4096).await.unwrap();
        // many large steps while the session re-checks its head every few ms
        let mut pos = 0u32;
        for i in 0..20 {
            let x: Vec<f32> = (0..MAX_STEP_TOKENS * HC).map(|j| (i * j) as f32).collect();
            let out = link.step(pos, &StepInput::Residual(x.clone())).await.unwrap();
            assert_eq!(out, x.iter().map(|v| v + 1.0).collect::<Vec<_>>());
            pos += MAX_STEP_TOKENS as u32;
        }
        REAUTH_MS.store(30_000, std::sync::atomic::Ordering::Relaxed);
    }

    #[tokio::test]
    async fn only_an_assigned_head_reaches_the_shard() {
        let shard = identity(5);
        let head = identity(6);
        let stranger = identity(7);
        let model = [0xEEu8; 32];
        let head_pk = head.pubkey;
        let authorize: HeadAuthorizer = Arc::new(move |h: &[u8; 32], _: &[u8; 32], _: u8, _: Option<&[u8; 32]>| *h == head_pk);
        let gw_addr = gateway(shard.clone(), model, FakeRange::new(true, 0.0), authorize).await;
        assert!(connect_link(&gw_addr, model, stranger, shard.pubkey, 8).await.is_err());
        // a hello addressed to another key, or expecting another tier, is refused
        assert!(connect_link(&gw_addr, model, head.clone(), [9u8; 32], 8).await.is_err());
        assert!(connect_link(&gw_addr, model, head.clone(), shard.pubkey, 7).await.is_err());
        assert!(connect_link(&gw_addr, model, head, shard.pubkey, 8).await.is_ok());
    }

    #[tokio::test]
    async fn a_malformed_step_ends_the_session_without_computing() {
        let _serial = session_tests().lock().await;
        let shard = identity(10);
        let head = identity(11);
        let model = [0xA1u8; 32];
        let range = FakeRange::new(true, 0.0);
        let gw_addr = gateway(shard.clone(), model, range.clone(), allow_all()).await;
        let link = connect_link(&gw_addr, model, head.clone(), shard.pubkey, 8).await.unwrap();
        // a step before any open is refused
        assert!(link.step(0, &StepInput::Tokens(vec![1])).await.is_err());
        let link = connect_link(&gw_addr, model, head.clone(), shard.pubkey, 8).await.unwrap();
        assert!(link.open(&[3u8; 32], 65).await.is_err(), "longer than the link's context");
        let link = connect_link(&gw_addr, model, head, shard.pubkey, 8).await.unwrap();
        link.open(&[3u8; 32], 16).await.unwrap();
        assert!(link.step(1, &StepInput::Tokens(vec![1])).await.is_err(), "position gap");
        assert!(link.ping().await.is_err(), "the session is closed after a refused step");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(range.steps.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(range.held(), 0);
    }

    #[tokio::test]
    async fn steps_and_sign_end_to_end() {
        let _serial = session_tests().lock().await;
        let shard = identity(1);
        let head = identity(2);
        let model = [0xABu8; 32];
        let assigned = [7u8; 32];
        let authorize: HeadAuthorizer =
            Arc::new(move |_: &[u8; 32], _: &[u8; 32], _: u8, r: Option<&[u8; 32]>| r.is_none_or(|r| *r == assigned));
        let range = FakeRange::new(true, 0.5);
        let gw_addr = gateway(shard.clone(), model, range.clone(), authorize).await;

        // unknown shard is refused
        assert!(connect_link(&gw_addr, [0u8; 32], head.clone(), shard.pubkey, 8).await.is_err());
        // a request not assigned cannot be opened
        let stray_link = connect_link(&gw_addr, model, head.clone(), shard.pubkey, 8).await.unwrap();
        assert!(stray_link.open(&[8u8; 32], 16).await.is_err());

        let link = connect_link(&gw_addr, model, head.clone(), shard.pubkey, 8).await.unwrap();
        assert_eq!(link.tier, 8);
        assert!(link.ping().await.unwrap() < Duration::from_secs(1));
        let payload = keryx_inference::AiResponsePayload::new(assigned, 9, [1u8; 34], 3).with_private_body(vec![4u8; 50]);
        let mut node_bytes = payload.link_signed_bytes(8, &head.pubkey);
        // no signature before the head computed the request through this link
        assert!(link.sign(&node_bytes[33..]).await.is_err());

        let d = link.open(&assigned, 16).await.unwrap();
        assert_eq!(d, LinkDims { n_hc: HC, n_vocab: 100, from_tokens: true });
        let out = link.step(0, &StepInput::Tokens(vec![1, 2])).await.unwrap();
        assert_eq!(out, vec![1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5]);
        let out = link.step(2, &StepInput::Tokens(vec![9])).await.unwrap();
        assert_eq!(out, vec![9.5; HC]);
        link.close().await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(range.held(), 0);

        // the node's message layout: tier, head key, then the payload bytes
        let signed = link.sign(&node_bytes[33..]).await.unwrap();
        assert_eq!(signed.tier, 8);
        assert!(verify_schnorr(&signed.escrow_pubkey, &signed.signature, &domain_hash(LINK_DOMAIN, &[&node_bytes])));
        // bound to this head and tier, and not a responder signature
        assert!(!verify_schnorr(&signed.escrow_pubkey, &signed.signature, &link_digest(7, &head.pubkey, &node_bytes[33..])));
        assert!(!verify_schnorr(&signed.escrow_pubkey, &signed.signature, &link_digest(8, &shard.pubkey, &node_bytes[33..])));
        assert!(!verify_response_signature(&signed.escrow_pubkey, &signed.signature, &node_bytes[33..]));
        // the same message again is fine, another one for the same request is not
        assert!(link.sign(&node_bytes[33..]).await.is_ok());
        let other = keryx_inference::AiResponsePayload::new(assigned, 9, [2u8; 34], 3).link_signed_bytes(8, &head.pubkey);
        assert!(link.sign(&other[33..]).await.is_err());
        // a request not assigned is refused
        let stray = keryx_inference::AiResponsePayload::new([8u8; 32], 9, [1u8; 34], 3).link_signed_bytes(8, &head.pubkey);
        assert!(link.sign(&stray[33..]).await.is_err());
        node_bytes[0] = 7;
        assert!(!verify_schnorr(&signed.escrow_pubkey, &signed.signature, &domain_hash(LINK_DOMAIN, &[&node_bytes])));
        // shorter than a v1 payload is refused by the link
        assert!(link.sign(&[5u8; 10]).await.is_err());
    }
}
