//! Gateway in front of the in-process shard rpc servers (H14), and the head-side link proxy.
//!
//! A shard miner listens on one public TCP port. A pipeline head connects, both sides prove
//! their escrow key over an ephemeral secp256k1 ECDH handshake, and everything after runs in
//! ChaCha20-Poly1305 frames: tunnelled ggml-rpc bytes to the loopback shard server, plus
//! control frames (ping, and the link signature of a response). The ggml-rpc server itself is
//! never exposed: it has no authentication of its own.
//!
//! On the head, `connect_link` opens a control tunnel to a shard and a loopback listener whose
//! every accepted connection is tunnelled to that shard, so the ggml-rpc client in
//! `libkeryx-llama.so` only ever talks to 127.0.0.1.

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
const VERSION: u8 = 1;
const CLIENT_DOMAIN: &[u8] = b"KeryxGatewayClientV1";
const SERVER_DOMAIN: &[u8] = b"KeryxGatewayServerV1";
const RESPONDER_DOMAIN: &[u8] = b"KeryxServiceResponderV1";
const HELLO_LEN: usize = 4 + 1 + 32 + 33 + 32 + 64;
const REPLY_LEN: usize = 1 + 33 + 32 + 1 + 64;
const MAX_FRAME: usize = 16 << 20;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Recent-coinbase memory of the peer directory, in DAA (the node's eligibility window).
const PEER_MEMORY_DAA: u64 = 6_000;

const FRAME_DATA: u8 = 0;
const FRAME_PING: u8 = 1;
const FRAME_PONG: u8 = 2;
const FRAME_SIGN: u8 = 3;
const FRAME_SIGNED: u8 = 4;
const FRAME_ERROR: u8 = 5;

const STATUS_OK: u8 = 0;
const STATUS_NO_SHARD: u8 = 1;

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

    /// Link signature over the 78 v1 response bytes — same domain as the node's
    /// `verify_responder_signature`.
    pub fn sign_response(&self, signed_bytes: &[u8]) -> [u8; 64] {
        self.sign_digest(&domain_hash(RESPONDER_DOMAIN, &[signed_bytes]))
    }
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

fn transcript(model_id: &[u8; 32], client_eph: &[u8; 33], server_eph: &[u8; 33], client_pk: &[u8; 32], server_pk: &[u8; 32]) -> [u8; 32] {
    domain_hash(b"KeryxGatewayTranscriptV1", &[model_id, client_eph, server_eph, client_pk, server_pk])
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

/// Records a coinbase payload in the peer directory: escrow key → advertised endpoint and caps.
pub fn note_coinbase(payload: &[u8], daa: u64) {
    let Some(escrow) = field_after(payload, b"/escrow:") else { return };
    let Ok(pk) = hex::decode(escrow) else { return };
    let Ok(pk): Result<[u8; 32], _> = pk.try_into() else { return };
    let Some(ep) = field_after(payload, b"/ai:ep:") else { return };
    let Ok(endpoint) = std::str::from_utf8(ep) else { return };
    if endpoint.is_empty() || endpoint.len() > 64 {
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

/// Resolves a shard model_id to its local rpc endpoint and tier — set by the miner once a shard
/// is served. Absent = not served here.
pub type ShardResolver = Arc<dyn Fn(&[u8; 32]) -> Option<(String, u8)> + Send + Sync>;

/// Runs the gateway listener until the process ends.
pub async fn serve(listen: String, identity: Arc<GatewayIdentity>, resolve: ShardResolver) -> Result<(), String> {
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
        tokio::spawn(async move {
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, server_handshake(stream, &identity, &resolve)).await {
                Ok(Ok((stream, keys, local, client_pk))) => {
                    log::info!("shard gateway: session from {} (head {})", peer, hex::encode(client_pk));
                    if let Err(e) = server_session(stream, keys, local, identity).await {
                        log::debug!("shard gateway: session {} ended: {}", peer, e);
                    }
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
) -> Result<(TcpStream, SessionKeys, String, [u8; 32]), String> {
    let mut hello = [0u8; HELLO_LEN];
    stream.read_exact(&mut hello).await.map_err(|e| e.to_string())?;
    if &hello[..4] != MAGIC || hello[4] != VERSION {
        return Err("bad magic or version".into());
    }
    let model_id: [u8; 32] = hello[5..37].try_into().unwrap();
    let client_eph: [u8; 33] = hello[37..70].try_into().unwrap();
    let client_pk: [u8; 32] = hello[70..102].try_into().unwrap();
    let client_sig: [u8; 64] = hello[102..166].try_into().unwrap();
    if !verify_schnorr(&client_pk, &client_sig, &domain_hash(CLIENT_DOMAIN, &[&model_id, &client_eph])) {
        return Err("client signature".into());
    }
    let client_eph_pk = PublicKey::from_slice(&client_eph).map_err(|_| "client ephemeral key".to_string())?;
    let (local, tier) = match resolve(&model_id) {
        Some(x) => x,
        None => {
            let mut reply = vec![STATUS_NO_SHARD];
            reply.resize(REPLY_LEN, 0);
            let _ = stream.write_all(&reply).await;
            return Err(format!("no shard {} served here", hex::encode(model_id)));
        }
    };
    let (eph_secret, eph_public) = ephemeral();
    let server_eph = eph_public.serialize();
    let sig = identity.sign_digest(&domain_hash(SERVER_DOMAIN, &[&model_id, &client_eph, &server_eph, &client_pk]));
    let mut reply = Vec::with_capacity(REPLY_LEN);
    reply.push(STATUS_OK);
    reply.extend_from_slice(&server_eph);
    reply.extend_from_slice(&identity.pubkey);
    reply.push(tier);
    reply.extend_from_slice(&sig);
    stream.write_all(&reply).await.map_err(|e| e.to_string())?;
    let shared = SharedSecret::new(&client_eph_pk, &eph_secret);
    let keys = session_keys(&shared, &transcript(&model_id, &client_eph, &server_eph, &client_pk, &identity.pubkey));
    Ok((stream, keys, local, client_pk))
}

async fn server_session(stream: TcpStream, keys: SessionKeys, local: String, identity: Arc<GatewayIdentity>) -> std::io::Result<()> {
    let (mut rd, mut wr) = stream.into_split();
    let mut rx = Sealer::new(&keys.c2s);
    let mut tx = Sealer::new(&keys.s2c);
    let (out_tx, mut out_rx) = mpsc::channel::<(u8, Vec<u8>)>(64);
    // single writer of the encrypted side
    let writer = tokio::spawn(async move {
        while let Some((kind, payload)) = out_rx.recv().await {
            if write_frame(&mut wr, &mut tx, kind, &payload).await.is_err() {
                break;
            }
        }
    });
    let mut local_wr: Option<tokio::net::tcp::OwnedWriteHalf> = None;
    let mut local_reader: Option<tokio::task::JoinHandle<()>> = None;
    let result: std::io::Result<()> = loop {
        let (kind, payload) = match read_frame(&mut rd, &mut rx).await {
            Ok(f) => f,
            Err(e) => break Err(e),
        };
        match kind {
            FRAME_DATA => {
                if local_wr.is_none() {
                    // first data frame: open the loopback rpc connection lazily
                    let sock = match TcpStream::connect(&local).await {
                        Ok(s) => {
                            let _ = s.set_nodelay(true);
                            s
                        }
                        Err(e) => {
                            let _ = out_tx.send((FRAME_ERROR, format!("shard server unreachable: {}", e).into_bytes())).await;
                            break Err(e);
                        }
                    };
                    let (mut lrd, lwr) = sock.into_split();
                    let out = out_tx.clone();
                    local_reader = Some(tokio::spawn(async move {
                        let mut buf = vec![0u8; 256 * 1024];
                        loop {
                            match lrd.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => {
                                    if out.send((FRAME_DATA, buf[..n].to_vec())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }));
                    local_wr = Some(lwr);
                }
                if let Some(w) = local_wr.as_mut() {
                    if let Err(e) = w.write_all(&payload).await {
                        break Err(e);
                    }
                }
            }
            FRAME_PING => {
                let _ = out_tx.send((FRAME_PONG, payload)).await;
            }
            FRAME_SIGN => {
                if payload.len() != keryx_inference::AI_RESPONSE_PAYLOAD_LEN {
                    let _ = out_tx.send((FRAME_ERROR, b"sign: bad length".to_vec())).await;
                    continue;
                }
                let sig = identity.sign_response(&payload);
                let mut reply = Vec::with_capacity(32 + 64);
                reply.extend_from_slice(&identity.pubkey);
                reply.extend_from_slice(&sig);
                let _ = out_tx.send((FRAME_SIGNED, reply)).await;
            }
            _ => {}
        }
    };
    if let Some(h) = local_reader {
        h.abort();
    }
    drop(out_tx);
    let _ = writer.await;
    result
}

// ── Client (pipeline head) ─────────────────────────────────────────────────────────────────────

enum Ctl {
    Sign([u8; 78], oneshot::Sender<Result<keryx_inference::AiResponseLink, String>>),
    Ping(oneshot::Sender<Result<Duration, String>>),
}

/// A shard reached through its gateway: a loopback port the ggml-rpc client connects to, and a
/// control tunnel for pings and the link signature.
pub struct LinkSession {
    pub remote: String,
    pub model_id: [u8; 32],
    pub tier: u8,
    pub peer_pubkey: [u8; 32],
    pub local_port: u16,
    ctl: mpsc::Sender<Ctl>,
}

struct Tunnel {
    stream: TcpStream,
    keys: SessionKeys,
    tier: u8,
    peer_pubkey: [u8; 32],
}

async fn client_handshake(remote: &str, model_id: &[u8; 32], identity: &GatewayIdentity) -> Result<Tunnel, String> {
    let mut stream = tokio::time::timeout(HANDSHAKE_TIMEOUT, TcpStream::connect(remote))
        .await
        .map_err(|_| format!("connect {}: timeout", remote))?
        .map_err(|e| format!("connect {}: {}", remote, e))?;
    let _ = stream.set_nodelay(true);
    let (eph_secret, eph_public) = ephemeral();
    let client_eph = eph_public.serialize();
    let sig = identity.sign_digest(&domain_hash(CLIENT_DOMAIN, &[model_id, &client_eph]));
    let mut hello = Vec::with_capacity(HELLO_LEN);
    hello.extend_from_slice(MAGIC);
    hello.push(VERSION);
    hello.extend_from_slice(model_id);
    hello.extend_from_slice(&client_eph);
    hello.extend_from_slice(&identity.pubkey);
    hello.extend_from_slice(&sig);
    stream.write_all(&hello).await.map_err(|e| e.to_string())?;
    let mut reply = [0u8; REPLY_LEN];
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
    if !verify_schnorr(&server_pk, &server_sig, &domain_hash(SERVER_DOMAIN, &[model_id, &client_eph, &server_eph, &identity.pubkey])) {
        return Err("gateway signature".into());
    }
    let server_eph_pk = PublicKey::from_slice(&server_eph).map_err(|_| "gateway ephemeral key".to_string())?;
    let shared = SharedSecret::new(&server_eph_pk, &eph_secret);
    let keys = session_keys(&shared, &transcript(model_id, &client_eph, &server_eph, &identity.pubkey, &server_pk));
    Ok(Tunnel { stream, keys, tier, peer_pubkey: server_pk })
}

/// Opens the control tunnel to `remote` for `model_id` and a loopback listener whose
/// connections are each tunnelled to the same shard.
pub async fn connect_link(remote: &str, model_id: [u8; 32], identity: Arc<GatewayIdentity>, local_port: u16) -> Result<LinkSession, String> {
    let control = client_handshake(remote, &model_id, &identity).await?;
    let tier = control.tier;
    let peer_pubkey = control.peer_pubkey;
    let listener = TcpListener::bind(("127.0.0.1", local_port)).await.map_err(|e| format!("loopback bind: {}", e))?;
    let local_port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let (ctl_tx, ctl_rx) = mpsc::channel::<Ctl>(16);
    tokio::spawn(control_loop(control, ctl_rx));
    let remote_s = remote.to_string();
    let id = identity.clone();
    tokio::spawn(async move {
        loop {
            let Ok((local, _)) = listener.accept().await else { break };
            let _ = local.set_nodelay(true);
            let remote = remote_s.clone();
            let id = id.clone();
            tokio::spawn(async move {
                match client_handshake(&remote, &model_id, &id).await {
                    Ok(t) => {
                        if let Err(e) = data_tunnel(local, t).await {
                            log::debug!("link proxy {}: tunnel ended: {}", remote, e);
                        }
                    }
                    Err(e) => log::warn!("link proxy {}: {}", remote, e),
                }
            });
        }
    });
    Ok(LinkSession { remote: remote.to_string(), model_id, tier, peer_pubkey, local_port, ctl: ctl_tx })
}

async fn control_loop(t: Tunnel, mut ctl: mpsc::Receiver<Ctl>) {
    let (mut rd, mut wr) = t.stream.into_split();
    let mut tx = Sealer::new(&t.keys.c2s);
    let mut rx = Sealer::new(&t.keys.s2c);
    let tier = t.tier;
    let mut pending_sign: std::collections::VecDeque<oneshot::Sender<Result<keryx_inference::AiResponseLink, String>>> = Default::default();
    let mut pending_ping: std::collections::VecDeque<(std::time::Instant, oneshot::Sender<Result<Duration, String>>)> = Default::default();
    loop {
        tokio::select! {
            req = ctl.recv() => {
                let Some(req) = req else { break };
                match req {
                    Ctl::Sign(bytes, reply) => {
                        if write_frame(&mut wr, &mut tx, FRAME_SIGN, &bytes).await.is_err() {
                            let _ = reply.send(Err("control tunnel closed".into()));
                            break;
                        }
                        pending_sign.push_back(reply);
                    }
                    Ctl::Ping(reply) => {
                        if write_frame(&mut wr, &mut tx, FRAME_PING, &[]).await.is_err() {
                            let _ = reply.send(Err("control tunnel closed".into()));
                            break;
                        }
                        pending_ping.push_back((std::time::Instant::now(), reply));
                    }
                }
            }
            frame = read_frame(&mut rd, &mut rx) => {
                let Ok((kind, payload)) = frame else { break };
                match kind {
                    FRAME_PONG => {
                        if let Some((t0, reply)) = pending_ping.pop_front() {
                            let _ = reply.send(Ok(t0.elapsed()));
                        }
                    }
                    FRAME_SIGNED => {
                        if let Some(reply) = pending_sign.pop_front() {
                            if payload.len() != 96 {
                                let _ = reply.send(Err("bad signature frame".into()));
                            } else {
                                let link = keryx_inference::AiResponseLink {
                                    tier,
                                    escrow_pubkey: payload[..32].try_into().unwrap(),
                                    signature: payload[32..96].try_into().unwrap(),
                                };
                                let _ = reply.send(Ok(link));
                            }
                        }
                    }
                    FRAME_ERROR => {
                        let msg = String::from_utf8_lossy(&payload).into_owned();
                        if let Some(reply) = pending_sign.pop_front() {
                            let _ = reply.send(Err(msg));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for reply in pending_sign.drain(..) {
        let _ = reply.send(Err("control tunnel closed".into()));
    }
}

async fn data_tunnel(local: TcpStream, t: Tunnel) -> std::io::Result<()> {
    let (mut lrd, mut lwr) = local.into_split();
    let (mut rrd, mut rwr) = t.stream.into_split();
    let mut tx = Sealer::new(&t.keys.c2s);
    let mut rx = Sealer::new(&t.keys.s2c);
    let up = tokio::spawn(async move {
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            match lrd.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if write_frame(&mut rwr, &mut tx, FRAME_DATA, &buf[..n]).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
    let result = loop {
        match read_frame(&mut rrd, &mut rx).await {
            Ok((FRAME_DATA, payload)) => {
                if let Err(e) = lwr.write_all(&payload).await {
                    break Err(e);
                }
            }
            Ok((FRAME_ERROR, payload)) => {
                break Err(std::io::Error::new(std::io::ErrorKind::Other, String::from_utf8_lossy(&payload).into_owned()));
            }
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    up.abort();
    result
}

impl LinkSession {
    /// Asks the shard to sign the 78 v1 bytes of a response it served.
    pub async fn sign(&self, signed_bytes: [u8; 78]) -> Result<keryx_inference::AiResponseLink, String> {
        let (tx, rx) = oneshot::channel();
        self.ctl.send(Ctl::Sign(signed_bytes, tx)).await.map_err(|_| "control tunnel closed".to_string())?;
        let link = tokio::time::timeout(Duration::from_secs(30), rx).await.map_err(|_| "sign timeout".to_string())?.map_err(|_| "control tunnel closed".to_string())??;
        if !verify_response_signature(&link.escrow_pubkey, &link.signature, &signed_bytes) || link.escrow_pubkey != self.peer_pubkey {
            return Err("link signature does not verify".into());
        }
        Ok(link)
    }

    /// Round-trip time of the control tunnel.
    pub async fn ping(&self) -> Result<Duration, String> {
        let (tx, rx) = oneshot::channel();
        self.ctl.send(Ctl::Ping(tx)).await.map_err(|_| "control tunnel closed".to_string())?;
        tokio::time::timeout(Duration::from_secs(10), rx).await.map_err(|_| "ping timeout".to_string())?.map_err(|_| "control tunnel closed".to_string())?
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
        let sealed = a.seal(FRAME_DATA, b"hello");
        assert_eq!(b.open(&sealed), Some((FRAME_DATA, b"hello".to_vec())));
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
        // old entries expire
        note_coinbase(format!("/escrow:{}/ai:ep:h:1", hex::encode([0x55u8; 32])).as_bytes(), 100 + PEER_MEMORY_DAA + 1);
        assert!(peer_endpoint(&pk).is_none());
    }

    #[tokio::test]
    async fn handshake_tunnel_sign_and_ping_end_to_end() {
        // a fake shard server: echoes what it receives
        let echo = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let echo_port = echo.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = echo.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    while let Ok(n) = s.read(&mut buf).await {
                        if n == 0 || s.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });
        let shard = identity(1);
        let head = identity(2);
        let model = [0xABu8; 32];
        let gw = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gw_addr = gw.local_addr().unwrap().to_string();
        drop(gw);
        let resolve: ShardResolver = Arc::new(move |id: &[u8; 32]| (*id == model).then(|| (format!("127.0.0.1:{}", echo_port), 9u8)));
        let shard_id = shard.clone();
        let listen = gw_addr.clone();
        tokio::spawn(async move { serve(listen, shard_id, resolve).await.unwrap() });
        tokio::time::sleep(Duration::from_millis(100)).await;

        // unknown shard is refused
        assert!(connect_link(&gw_addr, [0u8; 32], head.clone(), 0).await.is_err());

        let link = connect_link(&gw_addr, model, head.clone(), 0).await.unwrap();
        assert_eq!(link.tier, 9);
        assert_eq!(link.peer_pubkey, shard.pubkey);
        assert!(link.ping().await.unwrap() < Duration::from_secs(1));
        let signed = link.sign([5u8; 78]).await.unwrap();
        assert_eq!(signed.tier, 9);
        assert!(verify_response_signature(&signed.escrow_pubkey, &signed.signature, &[5u8; 78]));

        // data tunnel: bytes reach the shard server and come back
        let mut c = TcpStream::connect(("127.0.0.1", link.local_port)).await.unwrap();
        c.write_all(b"ping through the tunnel").await.unwrap();
        let mut buf = [0u8; 64];
        let n = c.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"ping through the tunnel");
    }
}
