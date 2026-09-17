/// On-chain AI request/response payload types.
///
/// Shared between the consensus validator (node) and the miner so both
/// sides always agree on the binary layout.

/// Binary payload layout for `SUBNETWORK_ID_AI_REQUEST` transactions:
/// `[model_id: 32] [max_tokens: 4 LE] [inference_reward: 8 LE] [priority_fee: 8 LE] [prompt…]`
pub const MIN_AI_REQUEST_PAYLOAD_LEN: usize = 52;

/// Minimum priority_fee (sompi) for an AiRequest — matches the network flat minimum tx fee (0.3 KRX).
/// Requesters may set a higher value to get their request processed faster.
pub const MIN_AI_REQUEST_PRIORITY_FEE: u64 = 30_000_000;
pub const MAX_AI_REQUEST_PAYLOAD_LEN: usize = 4_096;

/// Binary payload layout for `SUBNETWORK_ID_AI_RESPONSE` transactions:
/// `[request_hash: 32] [challenge_window_end: 8 LE] [response_ipfs_cid: 34] [response_length: 4 LE]`
/// Fixed 78 bytes — result is stored off-chain on IPFS, CID pinned by the miner.
pub const AI_RESPONSE_PAYLOAD_LEN: usize = 78;
/// V2 (service-bond era) appends `[responder_escrow_pubkey: 32] [schnorr_signature: 64]`,
/// the signature covering the 78 v1 bytes. Only the two exact lengths are valid.
pub const AI_RESPONSE_PAYLOAD_V2_LEN: usize = AI_RESPONSE_PAYLOAD_LEN + 32 + 64;
/// V3 (model-split era) appends `[n_links: 1] n × [tier: 1] [link_escrow_pubkey: 32] [link_signature: 64]`
/// to the V2 bytes: the pipeline links that served the response, each signing the 78 v1 bytes.
/// `n` is 1..=MAX_AI_RESPONSE_LINKS; a V3 payload with no links is not a V3 payload.
pub const AI_RESPONSE_LINK_LEN: usize = 1 + 32 + 64;
pub const MAX_AI_RESPONSE_LINKS: usize = 15;
pub const MIN_AI_RESPONSE_PAYLOAD_LEN: usize = AI_RESPONSE_PAYLOAD_LEN;
pub const MAX_AI_RESPONSE_PAYLOAD_LEN: usize = AI_RESPONSE_PAYLOAD_V2_LEN + 1 + MAX_AI_RESPONSE_LINKS * AI_RESPONSE_LINK_LEN;

/// Binary payload layout for `SUBNETWORK_ID_AI_AVAIL` transactions:
/// `[request_hash: 32] [tier: 1] [escrow_pubkey: 32] [schnorr_signature: 64]`
/// Fixed 129 bytes — a shard holder declaring itself available for a network-model request.
/// The signature covers the 33 `[request_hash] [tier]` bytes.
pub const AI_AVAIL_PAYLOAD_LEN: usize = 32 + 1 + 32 + 64;
pub const AI_AVAIL_SIGNED_LEN: usize = 33;

/// Binary payload layout for `SUBNETWORK_ID_AI_CHALLENGE` transactions:
/// `[response_hash: 32] [challenger_deposit: 8 LE] [challenger_spk_version: 2 LE] [challenger_spk: 32] [proof_data…]`
/// `proof_data` is empty for Phase 3 A2b stubs; 32 bytes (request_hash) for Phase 3 C re-execution.
/// `challenger_spk_version` + `challenger_spk` identify where the slashed escrow goes after CSV expiry.
pub const MIN_AI_CHALLENGE_PAYLOAD_LEN: usize = 74;
pub const MAX_AI_CHALLENGE_PAYLOAD_LEN: usize = 32_768;

/// Hex-encoded subnetwork IDs as returned by the keryxd gRPC API.
/// Used by the miner to filter transactions from block templates.
pub const SUBNETWORK_ID_AI_REQUEST_HEX: &str = "0300000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_RESPONSE_HEX: &str = "0400000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_CHALLENGE_HEX: &str = "0500000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_AVAIL_HEX: &str = "0600000000000000000000000000000000000000";

/// Payload of a `SUBNETWORK_ID_AI_REQUEST` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiRequestPayload {
    /// 32-byte identifier for the target model (e.g. hash of model weights).
    pub model_id: [u8; 32],
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sompi paid to the miner who fulfils this request (redirected from fee burn to miner payout).
    pub inference_reward: u64,
    /// Sompi burned as a network fee (minimum MIN_AI_REQUEST_PRIORITY_FEE; set higher for priority).
    pub priority_fee: u64,
    /// Raw prompt bytes (UTF-8 recommended, not enforced at this layer).
    pub prompt: Vec<u8>,
}

impl AiRequestPayload {
    pub fn new(model_id: [u8; 32], max_tokens: u32, inference_reward: u64, priority_fee: u64, prompt: Vec<u8>) -> Self {
        Self { model_id, max_tokens, inference_reward, priority_fee, prompt }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIN_AI_REQUEST_PAYLOAD_LEN + self.prompt.len());
        out.extend_from_slice(&self.model_id);
        out.extend_from_slice(&self.max_tokens.to_le_bytes());
        out.extend_from_slice(&self.inference_reward.to_le_bytes());
        out.extend_from_slice(&self.priority_fee.to_le_bytes());
        out.extend_from_slice(&self.prompt);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_REQUEST_PAYLOAD_LEN || data.len() > MAX_AI_REQUEST_PAYLOAD_LEN {
            return None;
        }
        let model_id: [u8; 32] = data[0..32].try_into().ok()?;
        let max_tokens = u32::from_le_bytes(data[32..36].try_into().ok()?);
        let inference_reward = u64::from_le_bytes(data[36..44].try_into().ok()?);
        let priority_fee = u64::from_le_bytes(data[44..52].try_into().ok()?);
        let prompt = data[52..].to_vec();
        Some(Self { model_id, max_tokens, inference_reward, priority_fee, prompt })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

/// Responder identity of a V2 AiResponse: the miner's escrow pubkey and its schnorr
/// signature over the 78 v1 payload bytes — mirror of the node's `AiResponder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiResponder {
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

/// One pipeline link of a V3 AiResponse: the shard tier it served, its escrow pubkey and its
/// schnorr signature over the 78 v1 payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiResponseLink {
    pub tier: u8,
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

/// Payload of a `SUBNETWORK_ID_AI_RESPONSE` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiResponsePayload {
    /// blake2b-256 hash of the AiRequest payload this response answers.
    pub request_hash: [u8; 32],
    /// Blue score at which the challenge window closes (miner's escrow is locked until then).
    pub challenge_window_end: u64,
    /// Raw 34-byte IPFS multihash (sha2-256): [0x12, 0x20, <32 bytes digest>].
    /// The full inference result is stored off-chain; fetch via IPFS gateway using this CID.
    pub response_ipfs_cid: [u8; 34],
    /// Number of tokens generated by the miner.
    pub response_length: u32,
    /// V2 responder identity; `None` for a v1 payload.
    pub responder: Option<AiResponder>,
    /// V3 pipeline links; empty for a v1 or v2 payload.
    pub links: Vec<AiResponseLink>,
}

impl AiResponsePayload {
    pub fn new(request_hash: [u8; 32], challenge_window_end: u64, response_ipfs_cid: [u8; 34], response_length: u32) -> Self {
        Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder: None, links: Vec::new() }
    }

    pub fn new_v2(
        request_hash: [u8; 32],
        challenge_window_end: u64,
        response_ipfs_cid: [u8; 34],
        response_length: u32,
        responder: AiResponder,
    ) -> Self {
        Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder: Some(responder), links: Vec::new() }
    }

    pub fn new_v3(
        request_hash: [u8; 32],
        challenge_window_end: u64,
        response_ipfs_cid: [u8; 34],
        response_length: u32,
        responder: AiResponder,
        links: Vec<AiResponseLink>,
    ) -> Self {
        Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder: Some(responder), links }
    }

    /// A V3 payload: signed responder plus at least one signed link.
    pub fn is_v3(&self) -> bool {
        self.responder.is_some() && !self.links.is_empty()
    }

    /// The 78 v1 bytes — also the message covered by the V2 responder signature.
    pub fn signed_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(AI_RESPONSE_PAYLOAD_LEN);
        out.extend_from_slice(&self.request_hash);
        out.extend_from_slice(&self.challenge_window_end.to_le_bytes());
        out.extend_from_slice(&self.response_ipfs_cid);
        out.extend_from_slice(&self.response_length.to_le_bytes());
        out
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = self.signed_bytes();
        if let Some(r) = &self.responder {
            out.reserve(32 + 64 + 1 + self.links.len() * AI_RESPONSE_LINK_LEN);
            out.extend_from_slice(&r.escrow_pubkey);
            out.extend_from_slice(&r.signature);
            if !self.links.is_empty() {
                out.push(self.links.len() as u8);
                for l in &self.links {
                    out.push(l.tier);
                    out.extend_from_slice(&l.escrow_pubkey);
                    out.extend_from_slice(&l.signature);
                }
            }
        }
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        let len = data.len();
        if len != AI_RESPONSE_PAYLOAD_LEN && len < AI_RESPONSE_PAYLOAD_V2_LEN {
            return None;
        }
        let request_hash: [u8; 32] = data[0..32].try_into().ok()?;
        let challenge_window_end = u64::from_le_bytes(data[32..40].try_into().ok()?);
        let response_ipfs_cid: [u8; 34] = data[40..74].try_into().ok()?;
        let response_length = u32::from_le_bytes(data[74..78].try_into().ok()?);
        let mut links = Vec::new();
        let responder = if len >= AI_RESPONSE_PAYLOAD_V2_LEN {
            if len > AI_RESPONSE_PAYLOAD_V2_LEN {
                let n = data[AI_RESPONSE_PAYLOAD_V2_LEN] as usize;
                if n == 0 || n > MAX_AI_RESPONSE_LINKS || len != AI_RESPONSE_PAYLOAD_V2_LEN + 1 + n * AI_RESPONSE_LINK_LEN {
                    return None;
                }
                let mut pos = AI_RESPONSE_PAYLOAD_V2_LEN + 1;
                for _ in 0..n {
                    links.push(AiResponseLink {
                        tier: data[pos],
                        escrow_pubkey: data[pos + 1..pos + 33].try_into().ok()?,
                        signature: data[pos + 33..pos + 97].try_into().ok()?,
                    });
                    pos += AI_RESPONSE_LINK_LEN;
                }
            }
            Some(AiResponder {
                escrow_pubkey: data[78..110].try_into().ok()?,
                signature: data[110..174].try_into().ok()?,
            })
        } else {
            None
        };
        Some(Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder, links })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }

    /// Encode the raw multihash as a base58btc CIDv0 string (e.g. "Qm...").
    pub fn cid_v0(&self) -> String {
        base58btc_encode(&self.response_ipfs_cid)
    }
}

/// Payload of a `SUBNETWORK_ID_AI_AVAIL` transaction: a shard holder's availability
/// declaration for one network-model request, signed by its escrow key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiAvailPayload {
    pub request_hash: [u8; 32],
    pub tier: u8,
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

impl AiAvailPayload {
    pub fn new(request_hash: [u8; 32], tier: u8, escrow_pubkey: [u8; 32], signature: [u8; 64]) -> Self {
        Self { request_hash, tier, escrow_pubkey, signature }
    }

    /// The 33 bytes the declaration signature covers.
    pub fn signed_bytes(&self) -> [u8; AI_AVAIL_SIGNED_LEN] {
        avail_signed_bytes(&self.request_hash, self.tier)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(AI_AVAIL_PAYLOAD_LEN);
        out.extend_from_slice(&self.signed_bytes());
        out.extend_from_slice(&self.escrow_pubkey);
        out.extend_from_slice(&self.signature);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() != AI_AVAIL_PAYLOAD_LEN {
            return None;
        }
        Some(Self {
            request_hash: data[0..32].try_into().ok()?,
            tier: data[32],
            escrow_pubkey: data[33..65].try_into().ok()?,
            signature: data[65..129].try_into().ok()?,
        })
    }
}

/// The bytes an availability declaration signs: `[request_hash] [tier]`.
pub fn avail_signed_bytes(request_hash: &[u8; 32], tier: u8) -> [u8; AI_AVAIL_SIGNED_LEN] {
    let mut out = [0u8; AI_AVAIL_SIGNED_LEN];
    out[..32].copy_from_slice(request_hash);
    out[32] = tier;
    out
}

/// Payload of a `SUBNETWORK_ID_AI_CHALLENGE` transaction.
///
/// Submitted by anyone who believes a miner published a fraudulent AiResponse.
/// The `challenger_deposit` is burned if the challenge is invalid.
///
/// If the challenge is accepted (proof validates via re-execution), the miner's escrow outpoint
/// is recorded as slashed with the challenger's `challenger_spk`.  After the CSV lock (36,000
/// blocks) expires, the miner can spend the escrow but only if an output goes to `challenger_spk`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiChallengePayload {
    /// Transaction payload hash of the disputed `AiResponse` (blake2b-256).
    pub response_hash: [u8; 32],
    /// Sompi deposited by the challenger (burned if challenge fails).
    pub challenger_deposit: u64,
    /// SPK version for the challenger's receiving address (standard = 0).
    pub challenger_spk_version: u16,
    /// 32-byte script identifying the challenger's receiving address.
    /// After the slash is confirmed and CSV expires, any spend of the escrow must
    /// include an output with this exact script_public_key.
    pub challenger_spk: [u8; 32],
    /// Re-execution fraud proof: `request_hash` (32 bytes) in Phase 3 C, empty in Phase A.
    pub proof_data: Vec<u8>,
}

impl AiChallengePayload {
    pub fn new(
        response_hash: [u8; 32],
        challenger_deposit: u64,
        challenger_spk_version: u16,
        challenger_spk: [u8; 32],
        proof_data: Vec<u8>,
    ) -> Self {
        Self { response_hash, challenger_deposit, challenger_spk_version, challenger_spk, proof_data }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIN_AI_CHALLENGE_PAYLOAD_LEN + self.proof_data.len());
        out.extend_from_slice(&self.response_hash);
        out.extend_from_slice(&self.challenger_deposit.to_le_bytes());
        out.extend_from_slice(&self.challenger_spk_version.to_le_bytes());
        out.extend_from_slice(&self.challenger_spk);
        out.extend_from_slice(&self.proof_data);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_CHALLENGE_PAYLOAD_LEN || data.len() > MAX_AI_CHALLENGE_PAYLOAD_LEN {
            return None;
        }
        let response_hash: [u8; 32] = data[0..32].try_into().ok()?;
        let challenger_deposit = u64::from_le_bytes(data[32..40].try_into().ok()?);
        let challenger_spk_version = u16::from_le_bytes(data[40..42].try_into().ok()?);
        let challenger_spk: [u8; 32] = data[42..74].try_into().ok()?;
        let proof_data = data[74..].to_vec();
        Some(Self { response_hash, challenger_deposit, challenger_spk_version, challenger_spk, proof_data })
    }

    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

/// Base58btc encoding (Bitcoin/IPFS alphabet) for CIDv0 strings.
fn base58btc_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let mut digits: Vec<u8> = vec![0];
    for &byte in input {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let leading_zeros = input.iter().take_while(|&&b| b == 0).count();
    let mut out = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros { out.push('1'); }
    for d in digits.iter().rev() { out.push(ALPHABET[*d as usize] as char); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_request_roundtrip() {
        let req = AiRequestPayload::new(
            [42u8; 32],
            256,
            1_000_000,
            30_000_000,
            b"What is the capital of France?".to_vec(),
        );
        let bytes = req.serialize();
        let parsed = AiRequestPayload::deserialize(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn ai_response_roundtrip() {
        let cid = [0x12, 0x20, 0xAAu8, 0xBB, 0xCC, 0xDD,
                   1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                   17,18,19,20,21,22,23,24,25,26,27,28];
        let resp = AiResponsePayload::new([7u8; 32], 900_000, cid, 128);
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_LEN);
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn ai_response_rejects_wrong_size() {
        assert!(AiResponsePayload::deserialize(&[0u8; 40]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 77]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 79]).is_none());
    }

    #[test]
    fn ai_request_rejects_too_short() {
        assert!(AiRequestPayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_response_rejects_too_short() {
        assert!(AiResponsePayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_request_rejects_oversized() {
        let huge = vec![0u8; MAX_AI_REQUEST_PAYLOAD_LEN + 1];
        assert!(AiRequestPayload::deserialize(&huge).is_none());
    }

    #[test]
    fn ai_challenge_roundtrip() {
        let ch = AiChallengePayload::new([0xABu8; 32], 500_000, 0, [0xCDu8; 32], b"stub_proof".to_vec());
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(ch, parsed);
    }

    #[test]
    fn ai_challenge_empty_proof_roundtrip() {
        let ch = AiChallengePayload::new([1u8; 32], 1_000, 0, [2u8; 32], vec![]);
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(ch, parsed);
    }

    #[test]
    fn ai_challenge_rejects_too_short() {
        assert!(AiChallengePayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_challenge_spk_roundtrip() {
        let spk = [0x42u8; 32];
        let ch = AiChallengePayload::new([0u8; 32], 0, 1, spk, [0xAAu8; 32].to_vec());
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(parsed.challenger_spk_version, 1);
        assert_eq!(parsed.challenger_spk, spk);
        assert_eq!(parsed.proof_data, [0xAAu8; 32]);
    }

    #[test]
    fn ai_response_v2_roundtrip() {
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        let resp = AiResponsePayload::new_v2([1u8; 32], 42, [2u8; 34], 7, responder);
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_V2_LEN);
        // The signed message is exactly the v1 prefix.
        assert_eq!(&bytes[..AI_RESPONSE_PAYLOAD_LEN], resp.signed_bytes().as_slice());
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(parsed.responder, Some(responder));
        assert_eq!(parsed.request_hash, [1u8; 32]);

        // v1 stays v1 on the wire (78 bytes, responder None).
        let v1 = AiResponsePayload::new([1u8; 32], 42, [2u8; 34], 7);
        let v1_bytes = v1.serialize();
        assert_eq!(v1_bytes.len(), AI_RESPONSE_PAYLOAD_LEN);
        assert_eq!(AiResponsePayload::deserialize(&v1_bytes).unwrap().responder, None);
        // Any other length is invalid.
        assert!(AiResponsePayload::deserialize(&bytes[..100]).is_none());
    }

    #[test]
    fn ai_avail_roundtrip_is_exact_length() {
        let avail = AiAvailPayload::new([7u8; 32], 9, [0x33u8; 32], [0x44u8; 64]);
        let bytes = avail.serialize();
        assert_eq!(bytes.len(), AI_AVAIL_PAYLOAD_LEN);
        assert_eq!(AiAvailPayload::deserialize(&bytes), Some(avail));
        assert_eq!(&bytes[..AI_AVAIL_SIGNED_LEN], avail.signed_bytes().as_slice());
        let mut short = bytes.clone();
        short.pop();
        assert!(AiAvailPayload::deserialize(&short).is_none());
    }

    #[test]
    fn ai_response_v3_roundtrip() {
        let cid = [0x12u8; 34];
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        let links = vec![
            AiResponseLink { tier: 6, escrow_pubkey: [0x61u8; 32], signature: [0x62u8; 64] },
            AiResponseLink { tier: 9, escrow_pubkey: [0x91u8; 32], signature: [0x92u8; 64] },
        ];
        let resp = AiResponsePayload::new_v3([7u8; 32], 900_000, cid, 128, responder, links);
        assert!(resp.is_v3());
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_V2_LEN + 1 + 2 * AI_RESPONSE_LINK_LEN);
        assert_eq!(AiResponsePayload::deserialize(&bytes).unwrap(), resp);
        let v2 = AiResponsePayload::new_v2([7u8; 32], 900_000, cid, 128, responder);
        assert!(!v2.is_v3());
        assert_eq!(v2.serialize().len(), AI_RESPONSE_PAYLOAD_V2_LEN);
    }

    #[test]
    fn ai_response_v3_rejects_malformed_link_sections() {
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        let link = AiResponseLink { tier: 6, escrow_pubkey: [0x61u8; 32], signature: [0x62u8; 64] };
        let good = AiResponsePayload::new_v3([7u8; 32], 1, [0x12u8; 34], 1, responder, vec![link]).serialize();
        let mut zero = good.clone();
        zero[AI_RESPONSE_PAYLOAD_V2_LEN] = 0;
        assert!(AiResponsePayload::deserialize(&zero).is_none());
        let mut short = good.clone();
        short.pop();
        assert!(AiResponsePayload::deserialize(&short).is_none());
        let mut bare = good[..AI_RESPONSE_PAYLOAD_V2_LEN].to_vec();
        bare.push(1);
        assert!(AiResponsePayload::deserialize(&bare).is_none());
    }
}
