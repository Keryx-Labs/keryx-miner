//! Private (end-to-end encrypted) inference.
//!
//! A requester encrypts its prompt to the whole cohort of the requested tier — miners identified
//! by the x-only escrow pubkey they announce in their coinbases (`/escrow:<hex>`) and sign V2
//! `AiResponse`s with — and a responder encrypts its answer back to the requester. Only the
//! requester and the recipients can read the prompt, and only the requester and the responder
//! that wrote it can read an answer; every other node sees ciphertext in the transaction payloads.
//!
//! Everything the chain needs stays public and is bound into the ciphertext as associated data:
//! the request header (`model_id`, `max_tokens`, `inference_reward`, `priority_fee`), the
//! envelope header (ephemeral key, nonce, recipient list) and, for the answer, the request hash
//! and the responder key. A relayer cannot re-parameterize a prompt, re-address an answer or
//! swap ciphertexts between requests without breaking authentication.
//!
//! ## Wire formats (all integers little-endian)
//!
//! Request envelope — the whole `prompt` field of an [`AiRequestPayload`]:
//!
//! ```text
//! [magic: 4 = 00 'K' 'X' 'P'] [version: 1 = 0x01] [ephemeral_pubkey: 33 compressed secp256k1]
//! [nonce: 12] [n_recipients: 1, 1..=128]
//! n × { [escrow_pubkey: 32 x-only, strictly ascending] [wrapped_root_key: 48] }
//! [ciphertext: ChaCha20-Poly1305(k_prompt, nonce, prompt) — at least the 16-byte tag]
//! ```
//!
//! Response envelope — the `private_body` of an `AiResponsePayload` (or an IPFS blob):
//!
//! ```text
//! [magic: 4] [version: 1] [nonce: 12] [ciphertext: ChaCha20-Poly1305(k_response, nonce, answer)]
//! ```
//!
//! The leading NUL byte of the magic never starts a UTF-8 prompt, so a plaintext request is
//! never mistaken for an envelope.
//!
//! ## Keys (key schedule 2, what [`seal_request`] produces)
//!
//! * `seed` — 32 random bytes drawn by the requester per request; the one secret it keeps
//!   ([`PrivateRequestSecret`]). `root_key = HKDF-SHA256(salt = "KeryxPrivateInferenceV2",
//!   ikm = seed, info = "root")` and the ephemeral secret `e` is the first valid scalar of
//!   `HKDF(salt, ikm = seed, info = "ephemeral" || ctr)`, `ctr = 0, 1, …`, so the requester can
//!   recompute `e` when the answers arrive. The recipients learn `root_key`, never `seed` or `e`.
//! * Per recipient `i`: `ss_i = ECDH(e, lift_x(R_i))` (libsecp default: SHA-256 of the compressed
//!   shared point), `kek_i = HKDF(salt, ikm = ss_i, info = "kek" || E || R_i)`, and
//!   `wrapped_root_key_i = ChaCha20-Poly1305(kek_i, nonce, root_key, aad = request header)`.
//!   `lift_x` is the BIP-340 even-Y lift, so a responder whose full key has odd Y negates its
//!   secret before the ECDH.
//! * `k_prompt = HKDF(salt, ikm = root_key, info = "prompt")`.
//! * `k_response_i = HKDF(salt, ikm = ss_i || root_key, info = "response" || R_i)`. Every
//!   recipient unwraps the same `root_key`, but only responder `i` and the requester know `ss_i`:
//!   no recipient can read another responder's answer and re-seal it as its own. A fresh random
//!   nonce per envelope keeps a re-published answer safe too.
//!
//! Key schedule 1, sealed by clients that predate it, drew `root_key` and `e` at random, used the
//! salt `"KeryxPrivateInferenceV1"` and `k_response = HKDF(salt, ikm = root_key,
//! info = "response" || R_i)`, a key every recipient can derive. Schedule-1 requests are still
//! opened and answered so they stay servable; [`seal_request_v1`] seals them for cohorts that
//! may not open schedule 2 yet.
//!
//! The key schedule is not on the wire: consensus parses request envelopes and accepts version
//! 1 only, so a new version byte would be a hard fork. A responder learns the schedule from the
//! key wrap (exactly one schedule's KEK authenticates its wrapped root key), the requester from
//! its secret.
//!
//! Correctness of the answer is not verified on-chain today (see `utxo_validation.rs`); this
//! module only adds confidentiality and integrity between the two parties.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hmac::{Hmac, Mac};
use rand::{CryptoRng, RngCore};
use secp256k1::ecdh::SharedSecret;
use secp256k1::{Parity, PublicKey, SECP256K1, SecretKey, XOnlyPublicKey};
use sha2::Sha256;

use crate::ai_payload::{AI_REQUEST_HEADER_LEN, AiRequestPayload, MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN};

/// Envelope marker: a leading NUL so no UTF-8 prompt can collide with it.
pub const PRIVATE_MAGIC: [u8; 4] = [0x00, b'K', b'X', b'P'];
/// Envelope format version. The key schedule is not part of the format (see the module docs).
pub const PRIVATE_VERSION: u8 = 1;
/// Responders a single request may name: a whole tier cohort, with headroom.
pub const MAX_PRIVATE_RECIPIENTS: usize = 128;
/// ChaCha20-Poly1305 nonce length.
pub const PRIVATE_NONCE_LEN: usize = 12;
/// Poly1305 tag length.
pub const PRIVATE_TAG_LEN: usize = 16;
/// Root key length.
pub const PRIVATE_ROOT_KEY_LEN: usize = 32;
/// A wrapped root key: the key plus its tag.
pub const PRIVATE_WRAPPED_KEY_LEN: usize = PRIVATE_ROOT_KEY_LEN + PRIVATE_TAG_LEN;
/// One recipient entry: escrow pubkey plus wrapped root key.
pub const PRIVATE_RECIPIENT_ENTRY_LEN: usize = 32 + PRIVATE_WRAPPED_KEY_LEN;
/// Fixed request-envelope header: magic, version, ephemeral key, nonce, recipient count.
pub const PRIVATE_REQUEST_HEADER_LEN: usize = PRIVATE_MAGIC.len() + 1 + 33 + PRIVATE_NONCE_LEN + 1;
/// Fixed response-envelope header: magic, version, nonce.
pub const PRIVATE_RESPONSE_HEADER_LEN: usize = PRIVATE_MAGIC.len() + 1 + PRIVATE_NONCE_LEN;

/// Leading byte of the hex form of a key-schedule-2 [`PrivateRequestSecret`].
const SECRET_V2_TAG: u8 = 0x02;

const KDF_SALT_V1: &[u8] = b"KeryxPrivateInferenceV1";
const KDF_SALT_V2: &[u8] = b"KeryxPrivateInferenceV2";
const INFO_ROOT: &[u8] = b"root";
const INFO_EPHEMERAL: &[u8] = b"ephemeral";
const INFO_KEK: &[u8] = b"kek";
const INFO_PROMPT: &[u8] = b"prompt";
const INFO_RESPONSE: &[u8] = b"response";

/// How a request derives its keys (see the module docs). Not on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeySchedule {
    /// Random root key; every recipient can derive every responder's key.
    V1,
    /// Seed-derived root key and ephemeral secret; a responder's key needs its ECDH secret.
    V2,
}

impl KeySchedule {
    fn salt(self) -> &'static [u8] {
        match self {
            Self::V1 => KDF_SALT_V1,
            Self::V2 => KDF_SALT_V2,
        }
    }
}

/// Errors of the private-inference envelopes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PrivateError {
    #[error("not a private-inference envelope")]
    NotPrivate,
    #[error("malformed private-inference envelope: {0}")]
    Malformed(&'static str),
    #[error("too many recipients: {0} (maximum {MAX_PRIVATE_RECIPIENTS})")]
    TooManyRecipients(usize),
    #[error("prompt of {0} bytes does not fit the request payload (maximum {1} for this recipient count)")]
    PromptTooLarge(usize, usize),
    #[error("invalid secp256k1 key")]
    InvalidKey,
    #[error("this escrow key is not among the envelope recipients")]
    NotARecipient,
    #[error("authentication failed: wrong key or tampered data")]
    Authentication,
}

/// One named responder of a request: its escrow pubkey and the root key wrapped to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivateRecipient {
    pub escrow_pubkey: [u8; 32],
    pub wrapped_root_key: [u8; PRIVATE_WRAPPED_KEY_LEN],
}

/// A parsed request envelope (the `prompt` field of a private `AiRequestPayload`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivateRequestEnvelope {
    pub ephemeral_pubkey: [u8; 33],
    pub nonce: [u8; PRIVATE_NONCE_LEN],
    /// Strictly ascending by escrow pubkey — the canonical order, enforced by [`Self::parse`].
    pub recipients: Vec<PrivateRecipient>,
    pub ciphertext: Vec<u8>,
}

impl PrivateRequestEnvelope {
    /// Whether a prompt field carries the envelope marker. Cheap; says nothing about validity.
    pub fn is_private(prompt: &[u8]) -> bool {
        prompt.len() >= PRIVATE_MAGIC.len() && prompt[..PRIVATE_MAGIC.len()] == PRIVATE_MAGIC
    }

    /// Strict parse: exact layout, supported version, valid ephemeral point, 1..=128 valid x-only
    /// recipient keys in strictly ascending order, and room for the ciphertext tag. Consensus
    /// classifies a request as private iff this succeeds, so it must stay a pure function of
    /// the bytes.
    pub fn parse(prompt: &[u8]) -> Result<Self, PrivateError> {
        if !Self::is_private(prompt) {
            return Err(PrivateError::NotPrivate);
        }
        if prompt.len() < PRIVATE_REQUEST_HEADER_LEN {
            return Err(PrivateError::Malformed("truncated header"));
        }
        if prompt[4] != PRIVATE_VERSION {
            return Err(PrivateError::Malformed("unsupported version"));
        }
        let ephemeral_pubkey: [u8; 33] = prompt[5..38].try_into().unwrap();
        PublicKey::from_slice(&ephemeral_pubkey).map_err(|_| PrivateError::InvalidKey)?;
        let nonce: [u8; PRIVATE_NONCE_LEN] = prompt[38..50].try_into().unwrap();
        let n = prompt[50] as usize;
        if n == 0 {
            return Err(PrivateError::Malformed("no recipients"));
        }
        if n > MAX_PRIVATE_RECIPIENTS {
            return Err(PrivateError::TooManyRecipients(n));
        }
        let entries_end = PRIVATE_REQUEST_HEADER_LEN + n * PRIVATE_RECIPIENT_ENTRY_LEN;
        if prompt.len() < entries_end + PRIVATE_TAG_LEN {
            return Err(PrivateError::Malformed("truncated recipients or ciphertext"));
        }
        let mut recipients = Vec::with_capacity(n);
        let mut prev: Option<[u8; 32]> = None;
        for i in 0..n {
            let at = PRIVATE_REQUEST_HEADER_LEN + i * PRIVATE_RECIPIENT_ENTRY_LEN;
            let escrow_pubkey: [u8; 32] = prompt[at..at + 32].try_into().unwrap();
            XOnlyPublicKey::from_slice(&escrow_pubkey).map_err(|_| PrivateError::InvalidKey)?;
            if prev.is_some_and(|p| p >= escrow_pubkey) {
                return Err(PrivateError::Malformed("recipients not strictly ascending"));
            }
            prev = Some(escrow_pubkey);
            let wrapped_root_key: [u8; PRIVATE_WRAPPED_KEY_LEN] =
                prompt[at + 32..at + PRIVATE_RECIPIENT_ENTRY_LEN].try_into().unwrap();
            recipients.push(PrivateRecipient { escrow_pubkey, wrapped_root_key });
        }
        Ok(Self { ephemeral_pubkey, nonce, recipients, ciphertext: prompt[entries_end..].to_vec() })
    }

    /// Canonical bytes; `parse(serialize(e)) == e`.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.header_len() + self.ciphertext.len());
        out.extend_from_slice(&PRIVATE_MAGIC);
        out.push(PRIVATE_VERSION);
        out.extend_from_slice(&self.ephemeral_pubkey);
        out.extend_from_slice(&self.nonce);
        out.push(self.recipients.len() as u8);
        for r in &self.recipients {
            out.extend_from_slice(&r.escrow_pubkey);
            out.extend_from_slice(&r.wrapped_root_key);
        }
        out.extend_from_slice(&self.ciphertext);
        out
    }

    /// Bytes before the ciphertext — the part of the prompt field bound as associated data.
    pub fn header_len(&self) -> usize {
        PRIVATE_REQUEST_HEADER_LEN + self.recipients.len() * PRIVATE_RECIPIENT_ENTRY_LEN
    }

    /// The named responders' escrow pubkeys, ascending.
    pub fn recipient_keys(&self) -> impl Iterator<Item = &[u8; 32]> {
        self.recipients.iter().map(|r| &r.escrow_pubkey)
    }

    /// Position of `escrow_pubkey` in the recipient list.
    pub fn recipient_index(&self, escrow_pubkey: &[u8; 32]) -> Option<usize> {
        self.recipients.iter().position(|r| r.escrow_pubkey == *escrow_pubkey)
    }
}

/// A parsed response envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivateResponseEnvelope {
    pub nonce: [u8; PRIVATE_NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

impl PrivateResponseEnvelope {
    /// Whether a response body carries the envelope marker.
    pub fn is_private(body: &[u8]) -> bool {
        PrivateRequestEnvelope::is_private(body)
    }

    pub fn parse(body: &[u8]) -> Result<Self, PrivateError> {
        if !Self::is_private(body) {
            return Err(PrivateError::NotPrivate);
        }
        if body.len() < PRIVATE_RESPONSE_HEADER_LEN + PRIVATE_TAG_LEN {
            return Err(PrivateError::Malformed("truncated response envelope"));
        }
        if body[4] != PRIVATE_VERSION {
            return Err(PrivateError::Malformed("unsupported version"));
        }
        let nonce: [u8; PRIVATE_NONCE_LEN] = body[5..PRIVATE_RESPONSE_HEADER_LEN].try_into().unwrap();
        Ok(Self { nonce, ciphertext: body[PRIVATE_RESPONSE_HEADER_LEN..].to_vec() })
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PRIVATE_RESPONSE_HEADER_LEN + self.ciphertext.len());
        out.extend_from_slice(&PRIVATE_MAGIC);
        out.push(PRIVATE_VERSION);
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.ciphertext);
        out
    }
}

/// The requester's secret for one request, the only way to read its answers. Keep it until the
/// answer is decrypted; it is never recoverable from the chain.
#[derive(Clone, PartialEq, Eq)]
pub struct PrivateRequestSecret {
    schedule: KeySchedule,
    /// Key schedule 2: the seed every key of the request derives from. Schedule 1: the root key.
    bytes: [u8; PRIVATE_ROOT_KEY_LEN],
}

impl PrivateRequestSecret {
    /// `02 || seed` (66 hex digits); a key-schedule-1 secret is its bare root key (64 digits), as
    /// older clients printed it.
    pub fn to_hex(&self) -> String {
        match self.schedule {
            KeySchedule::V1 => hex::encode(self.bytes),
            KeySchedule::V2 => format!("{SECRET_V2_TAG:02x}{}", hex::encode(self.bytes)),
        }
    }

    /// Parses [`Self::to_hex`], including the bare root key of a key-schedule-1 request.
    pub fn from_hex(s: &str) -> Result<Self, PrivateError> {
        let bytes = hex::decode(s.trim()).map_err(|_| PrivateError::Malformed("request secret is not hex"))?;
        let (schedule, rest) = match bytes.as_slice() {
            [SECRET_V2_TAG, seed @ ..] if seed.len() == PRIVATE_ROOT_KEY_LEN => (KeySchedule::V2, seed),
            root_key if root_key.len() == PRIVATE_ROOT_KEY_LEN => (KeySchedule::V1, root_key),
            _ => return Err(PrivateError::Malformed("request secret must be 02 + a 32-byte seed, or a 32-byte root key")),
        };
        Ok(Self { schedule, bytes: rest.try_into().unwrap() })
    }

    /// The root key the recipients unwrap.
    fn root_key(&self) -> [u8; PRIVATE_ROOT_KEY_LEN] {
        match self.schedule {
            KeySchedule::V1 => self.bytes,
            KeySchedule::V2 => hkdf(KDF_SALT_V2, &self.bytes, &[INFO_ROOT]),
        }
    }

    /// The key `responder` sealed its answer with.
    fn response_key(&self, responder: &[u8; 32]) -> Result<[u8; 32], PrivateError> {
        let root_key = self.root_key();
        match self.schedule {
            KeySchedule::V1 => Ok(response_key_v1(&root_key, responder)),
            KeySchedule::V2 => {
                let shared = SharedSecret::new(&lift_x(responder)?, &ephemeral_secret_v2(&self.bytes));
                Ok(response_key_v2(&shared.secret_bytes(), &root_key, responder))
            }
        }
    }
}

impl std::fmt::Debug for PrivateRequestSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrivateRequestSecret(..)")
    }
}

impl Drop for PrivateRequestSecret {
    fn drop(&mut self) {
        self.bytes.fill(0);
    }
}

/// The key one responder seals its answer to one request with ([`seal_response`]). Under key
/// schedule 2 no other recipient of the request can derive it.
#[derive(Clone, PartialEq, Eq)]
pub struct ResponseKey {
    key: [u8; 32],
    responder: [u8; 32],
}

impl ResponseKey {
    /// The escrow pubkey of the responder the key belongs to.
    pub fn responder(&self) -> &[u8; 32] {
        &self.responder
    }
}

impl std::fmt::Debug for ResponseKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseKey").field("responder", &hex::encode(self.responder)).finish_non_exhaustive()
    }
}

impl Drop for ResponseKey {
    fn drop(&mut self) {
        self.key.fill(0);
    }
}

/// A request a responder has opened: the plaintext prompt and the key it seals its answer with
/// ([`seal_response`]).
#[derive(Clone, PartialEq, Eq)]
pub struct OpenedRequest {
    pub prompt: Vec<u8>,
    pub response_key: ResponseKey,
    /// Index of the responder in the envelope's recipient list.
    pub recipient_index: usize,
}

impl std::fmt::Debug for OpenedRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenedRequest")
            .field("prompt_len", &self.prompt.len())
            .field("recipient_index", &self.recipient_index)
            .finish()
    }
}

impl Drop for OpenedRequest {
    fn drop(&mut self) {
        self.prompt.fill(0);
    }
}

/// Largest plaintext prompt a request naming `n_recipients` responders can carry, given the
/// private AiRequest payload cap.
pub fn max_private_prompt_len(n_recipients: usize) -> usize {
    (MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN - AI_REQUEST_HEADER_LEN)
        .saturating_sub(PRIVATE_REQUEST_HEADER_LEN + n_recipients * PRIVATE_RECIPIENT_ENTRY_LEN + PRIVATE_TAG_LEN)
}

/// The x-only escrow pubkey of a secret key — what a miner announces as `/escrow:`.
pub fn escrow_pubkey_of(secret: &[u8; 32]) -> Result<[u8; 32], PrivateError> {
    let sk = SecretKey::from_slice(secret).map_err(|_| PrivateError::InvalidKey)?;
    Ok(sk.x_only_public_key(SECP256K1).0.serialize())
}

/// Requester side: builds a private `AiRequestPayload` — the prompt sealed to `recipients`
/// (escrow pubkeys, any order, duplicates ignored) under key schedule 2 — and returns the secret
/// to keep.
pub fn seal_request(
    model_id: [u8; 32],
    max_tokens: u32,
    inference_reward: u64,
    priority_fee: u64,
    prompt: &[u8],
    recipients: &[[u8; 32]],
) -> Result<(AiRequestPayload, PrivateRequestSecret), PrivateError> {
    seal_request_with_rng(model_id, max_tokens, inference_reward, priority_fee, prompt, recipients, &mut rand::rngs::OsRng)
}

/// [`seal_request`] with an explicit randomness source.
pub fn seal_request_with_rng<R: RngCore + CryptoRng>(
    model_id: [u8; 32],
    max_tokens: u32,
    inference_reward: u64,
    priority_fee: u64,
    prompt: &[u8],
    recipients: &[[u8; 32]],
    rng: &mut R,
) -> Result<(AiRequestPayload, PrivateRequestSecret), PrivateError> {
    let mut secret = PrivateRequestSecret { schedule: KeySchedule::V2, bytes: [0u8; PRIVATE_ROOT_KEY_LEN] };
    rng.fill_bytes(&mut secret.bytes);
    let mut nonce = [0u8; PRIVATE_NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let ephemeral_secret = ephemeral_secret_v2(&secret.bytes);
    let request = AiRequestPayload::new(model_id, max_tokens, inference_reward, priority_fee, Vec::new());
    let payload = seal_payload(&secret, &ephemeral_secret, nonce, request, prompt, recipients)?;
    Ok((payload, secret))
}

/// [`seal_request`] under key schedule 1, for cohorts that may not open schedule 2 yet.
pub fn seal_request_v1(
    model_id: [u8; 32],
    max_tokens: u32,
    inference_reward: u64,
    priority_fee: u64,
    prompt: &[u8],
    recipients: &[[u8; 32]],
) -> Result<(AiRequestPayload, PrivateRequestSecret), PrivateError> {
    seal_request_v1_with_rng(model_id, max_tokens, inference_reward, priority_fee, prompt, recipients, &mut rand::rngs::OsRng)
}

/// [`seal_request_v1`] with an explicit randomness source.
pub fn seal_request_v1_with_rng<R: RngCore + CryptoRng>(
    model_id: [u8; 32],
    max_tokens: u32,
    inference_reward: u64,
    priority_fee: u64,
    prompt: &[u8],
    recipients: &[[u8; 32]],
    rng: &mut R,
) -> Result<(AiRequestPayload, PrivateRequestSecret), PrivateError> {
    let mut secret = PrivateRequestSecret { schedule: KeySchedule::V1, bytes: [0u8; PRIVATE_ROOT_KEY_LEN] };
    rng.fill_bytes(&mut secret.bytes);
    let mut nonce = [0u8; PRIVATE_NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let ephemeral_secret = SecretKey::new(rng);
    let request = AiRequestPayload::new(model_id, max_tokens, inference_reward, priority_fee, Vec::new());
    let payload = seal_payload(&secret, &ephemeral_secret, nonce, request, prompt, recipients)?;
    Ok((payload, secret))
}

/// Seals `prompt` into `request` (its prompt field still empty) to `recipients`, under the key
/// schedule and root key of `secret`.
fn seal_payload(
    secret: &PrivateRequestSecret,
    ephemeral_secret: &SecretKey,
    nonce: [u8; PRIVATE_NONCE_LEN],
    mut request: AiRequestPayload,
    prompt: &[u8],
    recipients: &[[u8; 32]],
) -> Result<AiRequestPayload, PrivateError> {
    let mut keys: Vec<[u8; 32]> = recipients.to_vec();
    keys.sort_unstable();
    keys.dedup();
    if keys.is_empty() {
        return Err(PrivateError::Malformed("no recipients"));
    }
    if keys.len() > MAX_PRIVATE_RECIPIENTS {
        return Err(PrivateError::TooManyRecipients(keys.len()));
    }
    let points = keys.iter().map(lift_x).collect::<Result<Vec<PublicKey>, _>>()?;
    let max_prompt = max_private_prompt_len(keys.len());
    if prompt.len() > max_prompt {
        return Err(PrivateError::PromptTooLarge(prompt.len(), max_prompt));
    }

    let salt = secret.schedule.salt();
    let root_key = secret.root_key();
    let ephemeral_pubkey = PublicKey::from_secret_key(SECP256K1, ephemeral_secret).serialize();
    let header = request.header_bytes();

    let mut envelope =
        PrivateRequestEnvelope { ephemeral_pubkey, nonce, recipients: Vec::with_capacity(keys.len()), ciphertext: Vec::new() };
    for (key, point) in keys.iter().zip(points.iter()) {
        let shared = SharedSecret::new(point, ephemeral_secret);
        let kek = hkdf(salt, &shared.secret_bytes(), &[INFO_KEK, &ephemeral_pubkey, key]);
        let wrapped = aead_encrypt(&kek, &nonce, &root_key, &header)?;
        envelope.recipients.push(PrivateRecipient {
            escrow_pubkey: *key,
            wrapped_root_key: wrapped.try_into().map_err(|_| PrivateError::Malformed("bad wrapped key length"))?,
        });
    }
    // Associated data: the request header followed by the envelope header (the ciphertext is
    // still empty, so `serialize` yields exactly the header).
    let mut aad = header.to_vec();
    aad.extend_from_slice(&envelope.serialize());
    let k_prompt = hkdf(salt, &root_key, &[INFO_PROMPT]);
    envelope.ciphertext = aead_encrypt(&k_prompt, &nonce, prompt, &aad)?;

    request.prompt = envelope.serialize();
    Ok(request)
}

/// Responder side: opens a private request with the escrow secret key (the one that signs V2
/// responses). Fails with [`PrivateError::NotARecipient`] when the request names other miners.
/// Opens requests of both key schedules.
pub fn open_request(payload: &AiRequestPayload, escrow_secret: &[u8; 32]) -> Result<OpenedRequest, PrivateError> {
    let envelope = PrivateRequestEnvelope::parse(&payload.prompt)?;
    let sk = SecretKey::from_slice(escrow_secret).map_err(|_| PrivateError::InvalidKey)?;
    let (xonly, parity) = sk.x_only_public_key(SECP256K1);
    let my_key = xonly.serialize();
    let index = envelope.recipient_index(&my_key).ok_or(PrivateError::NotARecipient)?;
    // The requester lifted the x-only key to its even-Y point; match it.
    let sk_even = if parity == Parity::Odd { sk.negate() } else { sk };
    let ephemeral_point = PublicKey::from_slice(&envelope.ephemeral_pubkey).map_err(|_| PrivateError::InvalidKey)?;
    let shared = SharedSecret::new(&ephemeral_point, &sk_even).secret_bytes();
    let header = payload.header_bytes();
    // The request's key schedule is the one whose KEK authenticates the wrapped root key.
    let (schedule, root_key) = [KeySchedule::V2, KeySchedule::V1]
        .into_iter()
        .find_map(|schedule| {
            let kek = hkdf(schedule.salt(), &shared, &[INFO_KEK, &envelope.ephemeral_pubkey, &my_key]);
            let root_key = aead_decrypt(&kek, &envelope.nonce, &envelope.recipients[index].wrapped_root_key, &header).ok()?;
            Some((schedule, <[u8; PRIVATE_ROOT_KEY_LEN]>::try_from(root_key).ok()?))
        })
        .ok_or(PrivateError::Authentication)?;

    let mut aad = header.to_vec();
    aad.extend_from_slice(&payload.prompt[..envelope.header_len()]);
    let k_prompt = hkdf(schedule.salt(), &root_key, &[INFO_PROMPT]);
    let prompt = aead_decrypt(&k_prompt, &envelope.nonce, &envelope.ciphertext, &aad)?;
    let key = match schedule {
        KeySchedule::V1 => response_key_v1(&root_key, &my_key),
        KeySchedule::V2 => response_key_v2(&shared, &root_key, &my_key),
    };
    Ok(OpenedRequest { prompt, response_key: ResponseKey { key, responder: my_key }, recipient_index: index })
}

/// Responder side: seals `answer` for the requester under the key [`open_request`] returned.
/// `request_hash` is the value the response payload carries.
pub fn seal_response(key: &ResponseKey, request_hash: &[u8; 32], answer: &[u8]) -> Vec<u8> {
    seal_response_with_rng(key, request_hash, answer, &mut rand::rngs::OsRng)
}

/// [`seal_response`] with an explicit randomness source.
pub fn seal_response_with_rng<R: RngCore + CryptoRng>(
    key: &ResponseKey,
    request_hash: &[u8; 32],
    answer: &[u8],
    rng: &mut R,
) -> Vec<u8> {
    let mut nonce = [0u8; PRIVATE_NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let aad = response_aad(request_hash, &key.responder);
    let ciphertext = aead_encrypt(&key.key, &nonce, answer, &aad).expect("ChaCha20-Poly1305 seals any in-memory input");
    PrivateResponseEnvelope { nonce, ciphertext }.serialize()
}

/// Requester side: opens a response body sealed by `responder_escrow_pubkey` for `request_hash`.
pub fn open_response(
    secret: &PrivateRequestSecret,
    request_hash: &[u8; 32],
    responder_escrow_pubkey: &[u8; 32],
    body: &[u8],
) -> Result<Vec<u8>, PrivateError> {
    let envelope = PrivateResponseEnvelope::parse(body)?;
    let key = secret.response_key(responder_escrow_pubkey)?;
    let aad = response_aad(request_hash, responder_escrow_pubkey);
    aead_decrypt(&key, &envelope.nonce, &envelope.ciphertext, &aad)
}

fn response_aad(request_hash: &[u8; 32], responder_escrow_pubkey: &[u8; 32]) -> [u8; 64] {
    let mut aad = [0u8; 64];
    aad[..32].copy_from_slice(request_hash);
    aad[32..].copy_from_slice(responder_escrow_pubkey);
    aad
}

/// Key schedule 1's `k_response`: derivable by every recipient of the request.
fn response_key_v1(root_key: &[u8; 32], responder: &[u8; 32]) -> [u8; 32] {
    hkdf(KDF_SALT_V1, root_key, &[INFO_RESPONSE, responder])
}

/// Key schedule 2's `k_response`: `shared` is the requester–responder ECDH secret `ss_i`.
fn response_key_v2(shared: &[u8; 32], root_key: &[u8; 32], responder: &[u8; 32]) -> [u8; 32] {
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(shared);
    ikm[32..].copy_from_slice(root_key);
    hkdf(KDF_SALT_V2, &ikm, &[INFO_RESPONSE, responder])
}

/// Key schedule 2's ephemeral secret: the first valid scalar of `HKDF(seed, "ephemeral" || ctr)`.
fn ephemeral_secret_v2(seed: &[u8; 32]) -> SecretKey {
    (0..=u8::MAX)
        .find_map(|ctr| SecretKey::from_slice(&hkdf(KDF_SALT_V2, seed, &[INFO_EPHEMERAL, &[ctr]])).ok())
        .expect("an HKDF output is a valid scalar except with probability 2^-128")
}

/// The BIP-340 even-Y point of an x-only escrow pubkey.
fn lift_x(key: &[u8; 32]) -> Result<PublicKey, PrivateError> {
    XOnlyPublicKey::from_slice(key).map(|x| x.public_key(Parity::Even)).map_err(|_| PrivateError::InvalidKey)
}

/// HKDF-SHA256 (RFC 5869) restricted to a single 32-byte output block.
fn hkdf(salt: &[u8], ikm: &[u8], info: &[&[u8]]) -> [u8; 32] {
    let mut extract = <Hmac<Sha256> as Mac>::new_from_slice(salt).expect("HMAC accepts any key length");
    extract.update(ikm);
    let prk = extract.finalize().into_bytes();
    let mut expand = <Hmac<Sha256> as Mac>::new_from_slice(&prk).expect("HMAC accepts any key length");
    for part in info {
        expand.update(part);
    }
    expand.update(&[1u8]);
    let okm = expand.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&okm);
    out
}

fn aead_encrypt(key: &[u8; 32], nonce: &[u8; PRIVATE_NONCE_LEN], msg: &[u8], aad: &[u8]) -> Result<Vec<u8>, PrivateError> {
    ChaCha20Poly1305::new(Key::from_slice(key))
        .encrypt(Nonce::from_slice(nonce), Payload { msg, aad })
        .map_err(|_| PrivateError::Malformed("encryption failed"))
}

fn aead_decrypt(key: &[u8; 32], nonce: &[u8; PRIVATE_NONCE_LEN], msg: &[u8], aad: &[u8]) -> Result<Vec<u8>, PrivateError> {
    ChaCha20Poly1305::new(Key::from_slice(key))
        .decrypt(Nonce::from_slice(nonce), Payload { msg, aad })
        .map_err(|_| PrivateError::Authentication)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const MODEL: [u8; 32] = [0xAB; 32];

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// A (secret, x-only pubkey) escrow identity, optionally forced to a given key parity.
    fn escrow_key(rng: &mut StdRng, want_odd: Option<bool>) -> ([u8; 32], [u8; 32]) {
        loop {
            let sk = SecretKey::new(rng);
            let (xonly, parity) = sk.x_only_public_key(SECP256K1);
            if want_odd.is_none_or(|odd| odd == (parity == Parity::Odd)) {
                return (sk.secret_bytes(), xonly.serialize());
            }
        }
    }

    fn seal(prompt: &[u8], recipients: &[[u8; 32]], rng: &mut StdRng) -> (AiRequestPayload, PrivateRequestSecret) {
        seal_request_with_rng(MODEL, 256, 100_000_000, 30_000_000, prompt, recipients, rng).unwrap()
    }

    fn seal_v1(prompt: &[u8], recipients: &[[u8; 32]], rng: &mut StdRng) -> (AiRequestPayload, PrivateRequestSecret) {
        seal_request_v1_with_rng(MODEL, 256, 100_000_000, 30_000_000, prompt, recipients, rng).unwrap()
    }

    #[test]
    fn plaintext_prompts_are_not_private() {
        assert!(!PrivateRequestEnvelope::is_private(b"What is the capital of France?"));
        assert!(!PrivateRequestEnvelope::is_private(b""));
        assert!(!PrivateRequestEnvelope::is_private(b"KXP"));
        assert_eq!(PrivateRequestEnvelope::parse(b"hello"), Err(PrivateError::NotPrivate));
    }

    #[test]
    fn single_recipient_roundtrip_both_directions() {
        let mut r = rng(1);
        let (sk, pk) = escrow_key(&mut r, None);
        let prompt = b"Summarize the plot of Hamlet in three sentences.";
        let (payload, secret) = seal(prompt, &[pk], &mut r);

        assert!(payload.is_private());
        assert!(payload.serialize().len() <= MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN);
        let envelope = PrivateRequestEnvelope::parse(&payload.prompt).unwrap();
        assert_eq!(envelope.recipients.len(), 1);
        assert_eq!(envelope.serialize(), payload.prompt);
        // The ciphertext is opaque: the prompt never appears in the payload.
        assert!(!payload.serialize().windows(prompt.len()).any(|w| w == prompt));

        // The requester recomputes its ephemeral key from the secret it keeps.
        let ephemeral = PublicKey::from_secret_key(SECP256K1, &ephemeral_secret_v2(&secret.bytes)).serialize();
        assert_eq!(envelope.ephemeral_pubkey, ephemeral);

        let opened = open_request(&payload, &sk).unwrap();
        assert_eq!(opened.prompt, prompt);
        assert_eq!(opened.recipient_index, 0);
        assert_eq!(opened.response_key.responder(), &pk);

        let request_hash = [0x77u8; 32];
        let answer = b"Hamlet's uncle murders the king. Hamlet feigns madness and seeks revenge. Nearly everyone dies.";
        let body = seal_response_with_rng(&opened.response_key, &request_hash, answer, &mut r);
        assert!(PrivateResponseEnvelope::is_private(&body));
        assert!(!body.windows(answer.len()).any(|w| w == answer));
        assert_eq!(open_response(&secret, &request_hash, &pk, &body).unwrap(), answer);
    }

    #[test]
    fn odd_parity_escrow_keys_open_the_request() {
        let mut r = rng(2);
        let (sk_odd, pk_odd) = escrow_key(&mut r, Some(true));
        let (sk_even, pk_even) = escrow_key(&mut r, Some(false));
        let (payload, _) = seal(b"parity test", &[pk_odd, pk_even], &mut r);
        assert_eq!(open_request(&payload, &sk_odd).unwrap().prompt, b"parity test");
        assert_eq!(open_request(&payload, &sk_even).unwrap().prompt, b"parity test");
    }

    #[test]
    fn every_named_recipient_opens_and_nobody_else_does() {
        let mut r = rng(3);
        let ids: Vec<_> = (0..4).map(|_| escrow_key(&mut r, None)).collect();
        let named: Vec<[u8; 32]> = ids[..3].iter().map(|(_, pk)| *pk).collect();
        // Any order and duplicates on input; canonical ascending order on the wire.
        let mut input = named.clone();
        input.reverse();
        input.push(named[0]);
        let (payload, secret) = seal(b"multi", &input, &mut r);
        let envelope = PrivateRequestEnvelope::parse(&payload.prompt).unwrap();
        let mut expected = named.clone();
        expected.sort_unstable();
        assert_eq!(envelope.recipient_keys().copied().collect::<Vec<_>>(), expected);

        let opened: Vec<OpenedRequest> = ids[..3].iter().map(|(sk, _)| open_request(&payload, sk).unwrap()).collect();
        for o in &opened {
            assert_eq!(o.prompt, b"multi");
        }
        assert_eq!(open_request(&payload, &ids[3].0), Err(PrivateError::NotARecipient));

        // Each responder seals under its own key: same answer, different bodies, and a body
        // opened against the wrong responder key fails.
        let rh = [1u8; 32];
        let body_a = seal_response_with_rng(&opened[0].response_key, &rh, b"same answer", &mut r);
        let body_b = seal_response_with_rng(&opened[1].response_key, &rh, b"same answer", &mut r);
        assert_ne!(body_a, body_b);
        assert_eq!(open_response(&secret, &rh, &named[0], &body_a).unwrap(), b"same answer");
        assert_eq!(open_response(&secret, &rh, &named[1], &body_b).unwrap(), b"same answer");
        assert_eq!(open_response(&secret, &rh, &named[0], &body_b), Err(PrivateError::Authentication));
    }

    #[test]
    fn no_recipient_can_read_another_responders_answer() {
        let mut r = rng(10);
        let (sk_a, pk_a) = escrow_key(&mut r, None);
        let (sk_b, pk_b) = escrow_key(&mut r, Some(true));
        let (payload, secret) = seal(b"one prompt, two responders", &[pk_a, pk_b], &mut r);
        let a = open_request(&payload, &sk_a).unwrap();
        let b = open_request(&payload, &sk_b).unwrap();
        let rh = [4u8; 32];
        let body = seal_response_with_rng(&a.response_key, &rh, b"A ran the model", &mut r);
        assert_eq!(open_response(&secret, &rh, &pk_a, &body).unwrap(), b"A ran the model");

        // B unwrapped the same root key as A. Under key schedule 1 that was enough to derive A's
        // response key; now it is not.
        let root_key = secret.root_key();
        assert_ne!(a.response_key.key, response_key_v1(&root_key, &pk_a));
        let root_only = PrivateRequestSecret { schedule: KeySchedule::V1, bytes: root_key };
        assert_eq!(open_response(&root_only, &rh, &pk_a, &body), Err(PrivateError::Authentication));

        // B's own key does not open A's body, and A's body re-signed by B does not open as B's.
        let envelope = PrivateResponseEnvelope::parse(&body).unwrap();
        assert!(aead_decrypt(&b.response_key.key, &envelope.nonce, &envelope.ciphertext, &response_aad(&rh, &pk_a)).is_err());
        assert_eq!(open_response(&secret, &rh, &pk_b, &body), Err(PrivateError::Authentication));
    }

    #[test]
    fn schedule_1_requests_still_open_and_answer() {
        let mut r = rng(11);
        let (sk_a, pk_a) = escrow_key(&mut r, None);
        let (sk_b, pk_b) = escrow_key(&mut r, Some(true));
        let (payload, secret) = seal_v1(b"sealed by an older client", &[pk_a, pk_b], &mut r);
        let a = open_request(&payload, &sk_a).unwrap();
        assert_eq!(a.prompt, b"sealed by an older client");
        assert_eq!(open_request(&payload, &sk_b).unwrap().prompt, b"sealed by an older client");

        // The older client printed the bare root key; it still opens the answer.
        let rh = [5u8; 32];
        let body = seal_response_with_rng(&a.response_key, &rh, b"answer", &mut r);
        assert_eq!(secret.to_hex(), hex::encode(secret.root_key()));
        let printed = PrivateRequestSecret::from_hex(&secret.to_hex()).unwrap();
        assert_eq!(open_response(&printed, &rh, &pk_a, &body).unwrap(), b"answer");
        // Schedule 1's response key is the one every recipient can derive from the root key.
        assert_eq!(a.response_key.key, response_key_v1(&secret.root_key(), &pk_a));
    }

    #[test]
    fn request_tampering_is_detected() {
        let mut r = rng(4);
        let (sk, pk) = escrow_key(&mut r, None);
        let (payload, _) = seal(b"tamper me", &[pk], &mut r);

        // Ciphertext byte flip.
        let mut t = payload.clone();
        let last = t.prompt.len() - 1;
        t.prompt[last] ^= 0x01;
        assert_eq!(open_request(&t, &sk), Err(PrivateError::Authentication));

        // Re-parameterized header: the wrapped key and the prompt are both bound to it.
        let mut t = payload.clone();
        t.max_tokens = 4096;
        assert_eq!(open_request(&t, &sk), Err(PrivateError::Authentication));
        let mut t = payload.clone();
        t.model_id = [0xCD; 32];
        assert_eq!(open_request(&t, &sk), Err(PrivateError::Authentication));

        // A wrapped key transplanted from another request to the same recipient.
        let (other, _) = seal(b"another request", &[pk], &mut r);
        let mut t = payload.clone();
        let at = PRIVATE_REQUEST_HEADER_LEN + 32;
        t.prompt[at..at + PRIVATE_WRAPPED_KEY_LEN].copy_from_slice(&other.prompt[at..at + PRIVATE_WRAPPED_KEY_LEN]);
        assert_eq!(open_request(&t, &sk), Err(PrivateError::Authentication));
    }

    #[test]
    fn response_tampering_is_detected() {
        let mut r = rng(5);
        let (sk, pk) = escrow_key(&mut r, None);
        let (payload, secret) = seal(b"q", &[pk], &mut r);
        let (_, other_secret) = seal(b"q", &[pk], &mut r);
        let key = open_request(&payload, &sk).unwrap().response_key.clone();
        let rh = [2u8; 32];
        let body = seal_response_with_rng(&key, &rh, b"answer", &mut r);
        assert_eq!(open_response(&secret, &rh, &pk, &body).unwrap(), b"answer");

        let mut t = body.clone();
        let last = t.len() - 1;
        t[last] ^= 0x80;
        assert_eq!(open_response(&secret, &rh, &pk, &t), Err(PrivateError::Authentication));
        assert_eq!(open_response(&secret, &[3u8; 32], &pk, &body), Err(PrivateError::Authentication));
        assert_eq!(open_response(&other_secret, &rh, &pk, &body), Err(PrivateError::Authentication));
        assert_eq!(open_response(&secret, &rh, &pk, b"plain text error"), Err(PrivateError::NotPrivate));
        assert_eq!(open_response(&secret, &rh, &pk, &body[..20]), Err(PrivateError::Malformed("truncated response envelope")));
        assert_eq!(open_response(&secret, &rh, &[0xFF; 32], &body), Err(PrivateError::InvalidKey));
    }

    #[test]
    fn parse_rejects_malformed_envelopes() {
        let mut r = rng(6);
        let (_, pk) = escrow_key(&mut r, None);
        let (payload, _) = seal(b"x", &[pk], &mut r);
        let good = payload.prompt.clone();

        let mut bad = good.clone();
        bad[4] = 2;
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::Malformed("unsupported version")));

        let mut bad = good.clone();
        bad[50] = 0;
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::Malformed("no recipients")));

        let mut bad = good.clone();
        bad[50] = (MAX_PRIVATE_RECIPIENTS + 1) as u8;
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::TooManyRecipients(MAX_PRIVATE_RECIPIENTS + 1)));

        let mut bad = good.clone();
        bad[50] = 2;
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::Malformed("truncated recipients or ciphertext")));

        let mut bad = good.clone();
        bad[5] = 0x05; // not a valid compressed-point prefix
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::InvalidKey));

        let mut bad = good.clone();
        bad[PRIVATE_REQUEST_HEADER_LEN..PRIVATE_REQUEST_HEADER_LEN + 32].copy_from_slice(&[0xFF; 32]); // x >= p
        assert_eq!(PrivateRequestEnvelope::parse(&bad), Err(PrivateError::InvalidKey));

        assert_eq!(PrivateRequestEnvelope::parse(&good[..40]), Err(PrivateError::Malformed("truncated header")));

        // Recipients must be strictly ascending: swap two entries of a two-recipient envelope.
        let (_, pk2) = escrow_key(&mut r, None);
        let (two, _) = seal(b"x", &[pk, pk2], &mut r);
        let e = PrivateRequestEnvelope::parse(&two.prompt).unwrap();
        let mut swapped = e.clone();
        swapped.recipients.swap(0, 1);
        assert_eq!(
            PrivateRequestEnvelope::parse(&swapped.serialize()),
            Err(PrivateError::Malformed("recipients not strictly ascending"))
        );
        let mut dup = e.clone();
        dup.recipients[1] = dup.recipients[0];
        assert_eq!(PrivateRequestEnvelope::parse(&dup.serialize()), Err(PrivateError::Malformed("recipients not strictly ascending")));
    }

    #[test]
    fn seal_validates_recipients_and_size() {
        let mut r = rng(7);
        let (_, pk) = escrow_key(&mut r, None);
        assert_eq!(seal_request_with_rng(MODEL, 1, 1, 1, b"p", &[], &mut r).unwrap_err(), PrivateError::Malformed("no recipients"));
        assert_eq!(seal_request_with_rng(MODEL, 1, 1, 1, b"p", &[[0xFF; 32]], &mut r).unwrap_err(), PrivateError::InvalidKey);
        let many: Vec<[u8; 32]> = (0..MAX_PRIVATE_RECIPIENTS + 1).map(|_| escrow_key(&mut r, None).1).collect();
        assert_eq!(
            seal_request_with_rng(MODEL, 1, 1, 1, b"p", &many, &mut r).unwrap_err(),
            PrivateError::TooManyRecipients(MAX_PRIVATE_RECIPIENTS + 1)
        );

        let max = max_private_prompt_len(1);
        let (payload, _) = seal(&vec![b'a'; max], &[pk], &mut r);
        assert_eq!(payload.serialize().len(), MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN);
        assert!(AiRequestPayload::deserialize(&payload.serialize()).is_some());
        assert_eq!(
            seal_request_with_rng(MODEL, 1, 1, 1, &vec![b'a'; max + 1], &[pk], &mut r).unwrap_err(),
            PrivateError::PromptTooLarge(max + 1, max)
        );
        // A full cohort still fits a useful prompt, and parses back.
        let full: Vec<[u8; 32]> = many[..MAX_PRIVATE_RECIPIENTS].to_vec();
        assert!(max_private_prompt_len(MAX_PRIVATE_RECIPIENTS) >= 4_000);
        let (payload, _) = seal(&vec![b'b'; max_private_prompt_len(MAX_PRIVATE_RECIPIENTS)], &full, &mut r);
        assert_eq!(payload.serialize().len(), MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN);
        assert_eq!(PrivateRequestEnvelope::parse(&payload.prompt).unwrap().recipients.len(), MAX_PRIVATE_RECIPIENTS);
    }

    #[test]
    fn sealing_is_randomized() {
        let mut r = rng(8);
        let (_, pk) = escrow_key(&mut r, None);
        let (a, sa) = seal(b"same", &[pk], &mut r);
        let (b, sb) = seal(b"same", &[pk], &mut r);
        assert_ne!(a.prompt, b.prompt);
        assert_ne!(sa, sb);
    }

    #[test]
    fn secret_hex_roundtrip_and_pubkey_derivation() {
        let mut r = rng(9);
        let (sk, pk) = escrow_key(&mut r, None);
        assert_eq!(escrow_pubkey_of(&sk).unwrap(), pk);

        let secret = PrivateRequestSecret { schedule: KeySchedule::V2, bytes: [0x42; 32] };
        assert_eq!(secret.to_hex(), format!("02{}", "42".repeat(32)));
        assert_eq!(PrivateRequestSecret::from_hex(&secret.to_hex()).unwrap(), secret);
        // A bare 32-byte key is the root key an older client printed, even when it starts with 02.
        let legacy = PrivateRequestSecret::from_hex(&"02".repeat(32)).unwrap();
        assert_eq!(legacy.schedule, KeySchedule::V1);
        assert_eq!(legacy.to_hex(), "02".repeat(32));

        assert!(PrivateRequestSecret::from_hex("zz").is_err());
        assert!(PrivateRequestSecret::from_hex("00").is_err());
        assert!(PrivateRequestSecret::from_hex(&format!("03{}", "42".repeat(32))).is_err());
        assert!(PrivateRequestSecret::from_hex(&format!("02{}", "42".repeat(33))).is_err());
    }
}
