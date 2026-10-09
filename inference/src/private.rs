//! Private (end-to-end encrypted) inference.
//!
//! A requester encrypts its prompt to the whole cohort of the requested tier — miners identified
//! by the x-only escrow pubkey they announce in their coinbases (`/escrow:<hex>`) and sign V2
//! `AiResponse`s with — and a responder encrypts its answer back to the requester. Only the
//! requester and the recipients can read either direction; every other node sees ciphertext in
//! the transaction payloads.
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
//! ## Keys
//!
//! * `root_key` — 32 random bytes drawn by the requester per request; the one secret it keeps.
//! * Per recipient `i`: `ss_i = ECDH(e, lift_x(R_i))` (libsecp default: SHA-256 of the compressed
//!   shared point), `kek_i = HKDF-SHA256(salt = "KeryxPrivateInferenceV1", ikm = ss_i,
//!   info = "kek" || E || R_i)`, and `wrapped_root_key_i = ChaCha20-Poly1305(kek_i, nonce,
//!   root_key, aad = request header)`. `lift_x` is the BIP-340 even-Y lift, so a responder whose
//!   full key has odd Y negates its secret before the ECDH.
//! * `k_prompt = HKDF(salt, ikm = root_key, info = "prompt")`.
//! * `k_response = HKDF(salt, ikm = root_key, info = "response" || responder_escrow_pubkey)` —
//!   distinct per responder, so several named responders answering the same request never share
//!   a key, and a fresh random nonce per envelope keeps a re-published answer safe too.
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
/// Envelope format version.
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

const KDF_SALT: &[u8] = b"KeryxPrivateInferenceV1";
const INFO_KEK: &[u8] = b"kek";
const INFO_PROMPT: &[u8] = b"prompt";
const INFO_RESPONSE: &[u8] = b"response";

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

/// The requester's secret for one request: the root key every direction derives from. Keep it
/// until the answer is decrypted; it is never recoverable from the chain.
#[derive(Clone, PartialEq, Eq)]
pub struct PrivateRequestSecret {
    pub root_key: [u8; PRIVATE_ROOT_KEY_LEN],
}

impl PrivateRequestSecret {
    pub fn to_hex(&self) -> String {
        hex::encode(self.root_key)
    }

    pub fn from_hex(s: &str) -> Result<Self, PrivateError> {
        let bytes = hex::decode(s.trim()).map_err(|_| PrivateError::Malformed("root key is not hex"))?;
        let root_key: [u8; 32] = bytes.try_into().map_err(|_| PrivateError::Malformed("root key must be 32 bytes"))?;
        Ok(Self { root_key })
    }
}

impl std::fmt::Debug for PrivateRequestSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrivateRequestSecret(..)")
    }
}

impl Drop for PrivateRequestSecret {
    fn drop(&mut self) {
        self.root_key.fill(0);
    }
}

/// A request a responder has opened: the plaintext prompt and the root key it needs to seal its
/// answer with [`seal_response`].
#[derive(Clone, PartialEq, Eq)]
pub struct OpenedRequest {
    pub prompt: Vec<u8>,
    pub root_key: [u8; PRIVATE_ROOT_KEY_LEN],
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
        self.root_key.fill(0);
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
/// (escrow pubkeys, any order, duplicates ignored) — and returns the secret to keep.
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
    let mut keys: Vec<[u8; 32]> = recipients.to_vec();
    keys.sort_unstable();
    keys.dedup();
    if keys.is_empty() {
        return Err(PrivateError::Malformed("no recipients"));
    }
    if keys.len() > MAX_PRIVATE_RECIPIENTS {
        return Err(PrivateError::TooManyRecipients(keys.len()));
    }
    let points = keys
        .iter()
        .map(|k| XOnlyPublicKey::from_slice(k).map(|x| x.public_key(Parity::Even)).map_err(|_| PrivateError::InvalidKey))
        .collect::<Result<Vec<PublicKey>, _>>()?;
    let max_prompt = max_private_prompt_len(keys.len());
    if prompt.len() > max_prompt {
        return Err(PrivateError::PromptTooLarge(prompt.len(), max_prompt));
    }

    let mut root_key = [0u8; PRIVATE_ROOT_KEY_LEN];
    rng.fill_bytes(&mut root_key);
    let mut nonce = [0u8; PRIVATE_NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let ephemeral_secret = SecretKey::new(rng);
    let ephemeral_pubkey = PublicKey::from_secret_key(SECP256K1, &ephemeral_secret).serialize();
    let header = AiRequestPayload::header_bytes_of(&model_id, max_tokens, inference_reward, priority_fee);

    let mut envelope =
        PrivateRequestEnvelope { ephemeral_pubkey, nonce, recipients: Vec::with_capacity(keys.len()), ciphertext: Vec::new() };
    for (key, point) in keys.iter().zip(points.iter()) {
        let shared = SharedSecret::new(point, &ephemeral_secret);
        let kek = hkdf(KDF_SALT, &shared.secret_bytes(), &[INFO_KEK, &ephemeral_pubkey, key]);
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
    let k_prompt = hkdf(KDF_SALT, &root_key, &[INFO_PROMPT]);
    envelope.ciphertext = aead_encrypt(&k_prompt, &nonce, prompt, &aad)?;

    let payload = AiRequestPayload::new(model_id, max_tokens, inference_reward, priority_fee, envelope.serialize());
    Ok((payload, PrivateRequestSecret { root_key }))
}

/// Responder side: opens a private request with the escrow secret key (the one that signs V2
/// responses). Fails with [`PrivateError::NotARecipient`] when the request names other miners.
pub fn open_request(payload: &AiRequestPayload, escrow_secret: &[u8; 32]) -> Result<OpenedRequest, PrivateError> {
    let envelope = PrivateRequestEnvelope::parse(&payload.prompt)?;
    let sk = SecretKey::from_slice(escrow_secret).map_err(|_| PrivateError::InvalidKey)?;
    let (xonly, parity) = sk.x_only_public_key(SECP256K1);
    let my_key = xonly.serialize();
    let index = envelope.recipient_index(&my_key).ok_or(PrivateError::NotARecipient)?;
    // The requester lifted the x-only key to its even-Y point; match it.
    let sk_even = if parity == Parity::Odd { sk.negate() } else { sk };
    let ephemeral_point = PublicKey::from_slice(&envelope.ephemeral_pubkey).map_err(|_| PrivateError::InvalidKey)?;
    let shared = SharedSecret::new(&ephemeral_point, &sk_even);
    let kek = hkdf(KDF_SALT, &shared.secret_bytes(), &[INFO_KEK, &envelope.ephemeral_pubkey, &my_key]);
    let header = payload.header_bytes();
    let root_key: [u8; PRIVATE_ROOT_KEY_LEN] =
        aead_decrypt(&kek, &envelope.nonce, &envelope.recipients[index].wrapped_root_key, &header)?
            .try_into()
            .map_err(|_| PrivateError::Authentication)?;

    let mut aad = header.to_vec();
    aad.extend_from_slice(&payload.prompt[..envelope.header_len()]);
    let k_prompt = hkdf(KDF_SALT, &root_key, &[INFO_PROMPT]);
    let prompt = aead_decrypt(&k_prompt, &envelope.nonce, &envelope.ciphertext, &aad)?;
    Ok(OpenedRequest { prompt, root_key, recipient_index: index })
}

/// Responder side: seals `answer` for the requester. `request_hash` is the value the response
/// payload carries; `responder_escrow_pubkey` the key that signs it.
pub fn seal_response(root_key: &[u8; 32], request_hash: &[u8; 32], responder_escrow_pubkey: &[u8; 32], answer: &[u8]) -> Vec<u8> {
    seal_response_with_rng(root_key, request_hash, responder_escrow_pubkey, answer, &mut rand::rngs::OsRng)
}

/// [`seal_response`] with an explicit randomness source.
pub fn seal_response_with_rng<R: RngCore + CryptoRng>(
    root_key: &[u8; 32],
    request_hash: &[u8; 32],
    responder_escrow_pubkey: &[u8; 32],
    answer: &[u8],
    rng: &mut R,
) -> Vec<u8> {
    let mut nonce = [0u8; PRIVATE_NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let key = hkdf(KDF_SALT, root_key, &[INFO_RESPONSE, responder_escrow_pubkey]);
    let aad = response_aad(request_hash, responder_escrow_pubkey);
    let ciphertext = aead_encrypt(&key, &nonce, answer, &aad).expect("ChaCha20-Poly1305 seals any in-memory input");
    PrivateResponseEnvelope { nonce, ciphertext }.serialize()
}

/// Requester side: opens a response body sealed by `responder_escrow_pubkey` for `request_hash`.
pub fn open_response(
    root_key: &[u8; 32],
    request_hash: &[u8; 32],
    responder_escrow_pubkey: &[u8; 32],
    body: &[u8],
) -> Result<Vec<u8>, PrivateError> {
    let envelope = PrivateResponseEnvelope::parse(body)?;
    let key = hkdf(KDF_SALT, root_key, &[INFO_RESPONSE, responder_escrow_pubkey]);
    let aad = response_aad(request_hash, responder_escrow_pubkey);
    aead_decrypt(&key, &envelope.nonce, &envelope.ciphertext, &aad)
}

fn response_aad(request_hash: &[u8; 32], responder_escrow_pubkey: &[u8; 32]) -> [u8; 64] {
    let mut aad = [0u8; 64];
    aad[..32].copy_from_slice(request_hash);
    aad[32..].copy_from_slice(responder_escrow_pubkey);
    aad
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

        let opened = open_request(&payload, &sk).unwrap();
        assert_eq!(opened.prompt, prompt);
        assert_eq!(opened.root_key, secret.root_key);
        assert_eq!(opened.recipient_index, 0);

        let request_hash = [0x77u8; 32];
        let answer = b"Hamlet's uncle murders the king. Hamlet feigns madness and seeks revenge. Nearly everyone dies.";
        let body = seal_response_with_rng(&opened.root_key, &request_hash, &pk, answer, &mut r);
        assert!(PrivateResponseEnvelope::is_private(&body));
        assert!(!body.windows(answer.len()).any(|w| w == answer));
        assert_eq!(open_response(&secret.root_key, &request_hash, &pk, &body).unwrap(), answer);
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

        for (sk, _) in &ids[..3] {
            let opened = open_request(&payload, sk).unwrap();
            assert_eq!(opened.prompt, b"multi");
            assert_eq!(opened.root_key, secret.root_key);
        }
        assert_eq!(open_request(&payload, &ids[3].0), Err(PrivateError::NotARecipient));

        // Each responder seals under its own derived key: same answer, different bodies, and a
        // body opened against the wrong responder key fails.
        let rh = [1u8; 32];
        let body_a = seal_response_with_rng(&secret.root_key, &rh, &named[0], b"same answer", &mut r);
        let body_b = seal_response_with_rng(&secret.root_key, &rh, &named[1], b"same answer", &mut r);
        assert_ne!(body_a, body_b);
        assert_eq!(open_response(&secret.root_key, &rh, &named[0], &body_a).unwrap(), b"same answer");
        assert_eq!(open_response(&secret.root_key, &rh, &named[0], &body_b), Err(PrivateError::Authentication));
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
        let (_, pk) = escrow_key(&mut r, None);
        let root = [9u8; 32];
        let rh = [2u8; 32];
        let body = seal_response_with_rng(&root, &rh, &pk, b"answer", &mut r);
        assert_eq!(open_response(&root, &rh, &pk, &body).unwrap(), b"answer");

        let mut t = body.clone();
        let last = t.len() - 1;
        t[last] ^= 0x80;
        assert_eq!(open_response(&root, &rh, &pk, &t), Err(PrivateError::Authentication));
        assert_eq!(open_response(&root, &[3u8; 32], &pk, &body), Err(PrivateError::Authentication));
        assert_eq!(open_response(&[8u8; 32], &rh, &pk, &body), Err(PrivateError::Authentication));
        assert_eq!(open_response(&root, &rh, &pk, b"plain text error"), Err(PrivateError::NotPrivate));
        assert_eq!(open_response(&root, &rh, &pk, &body[..20]), Err(PrivateError::Malformed("truncated response envelope")));
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
        assert_ne!(sa.root_key, sb.root_key);
    }

    #[test]
    fn secret_hex_roundtrip_and_pubkey_derivation() {
        let mut r = rng(9);
        let (sk, pk) = escrow_key(&mut r, None);
        assert_eq!(escrow_pubkey_of(&sk).unwrap(), pk);
        let secret = PrivateRequestSecret { root_key: [0x42; 32] };
        assert_eq!(PrivateRequestSecret::from_hex(&secret.to_hex()).unwrap(), secret);
        assert!(PrivateRequestSecret::from_hex("zz").is_err());
        assert!(PrivateRequestSecret::from_hex("00").is_err());
    }
}
