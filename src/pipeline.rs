//! Pipeline head (H14): serves a request addressed to the network model by assembling its
//! shards over the gateway links, generating on the head GGUF with every shard layer bound to
//! a remote resident copy, and collecting the links' signatures for the V3 response.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::models::{NETWORK_MODEL_TIER, V4_FLASH, V4_FLASH_N_LAYER, V4_FLASH_SHARDS, V4_FLASH_SHARD_LAYERS};
use crate::shard_gateway::{self, GatewayIdentity, LinkSession};

/// An armed network-model audit as the node publishes it in the block template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub request_hash: [u8; 32],
    pub accepted_daa: u64,
    pub window_end_daa: u64,
    /// `(shard tier, escrow pubkey)` drawn per tier.
    pub links: Vec<(u8, [u8; 32])>,
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    hex::decode(s).ok()?.try_into().ok()
}

/// Parses one `hash:accepted:window_end:tier-escrowhex,...` entry.
pub fn parse_assignment(s: &str) -> Option<Assignment> {
    let mut parts = s.split(':');
    let request_hash = unhex32(parts.next()?)?;
    let accepted_daa: u64 = parts.next()?.parse().ok()?;
    let window_end_daa: u64 = parts.next()?.parse().ok()?;
    let mut links = Vec::new();
    for item in parts.next()?.split(',').filter(|i| !i.is_empty()) {
        let (t, k) = item.split_once('-')?;
        links.push((t.parse().ok()?, unhex32(k)?));
    }
    Some(Assignment { request_hash, accepted_daa, window_end_daa, links })
}

fn assignments() -> &'static Mutex<HashMap<[u8; 32], Assignment>> {
    static A: OnceLock<Mutex<HashMap<[u8; 32], Assignment>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Replaces the known assignments with the template's `;`-separated list.
pub fn note_assignments(encoded: &str) {
    let parsed: HashMap<[u8; 32], Assignment> =
        encoded.split(';').filter(|s| !s.is_empty()).filter_map(parse_assignment).map(|a| (a.request_hash, a)).collect();
    if let Ok(mut g) = assignments().lock() {
        *g = parsed;
    }
}

pub fn assignment_for(request_hash: &[u8; 32]) -> Option<Assignment> {
    assignments().lock().ok()?.get(request_hash).cloned()
}

static IDENTITY: OnceLock<Arc<GatewayIdentity>> = OnceLock::new();

/// Installs the escrow identity the head connects to links with (and is drawn as).
pub fn set_identity(identity: Arc<GatewayIdentity>) {
    let _ = IDENTITY.set(identity);
}

pub fn identity() -> Option<Arc<GatewayIdentity>> {
    IDENTITY.get().cloned()
}

/// Tier of the head: the last shard.
pub fn head_tier() -> u8 {
    NETWORK_MODEL_TIER + V4_FLASH_SHARDS.len() as u8
}

/// This miner can head a pipeline: it serves the head shard, holds the head GGUF and has an
/// escrow identity.
pub fn head_ready() -> bool {
    identity().is_some()
        && crate::pom_gpu::shard_local_endpoint(&V4_FLASH_SHARDS[V4_FLASH_SHARDS.len() - 1].model_id).is_some()
        && crate::slm::head_gguf_path().exists()
}

/// `KERYX_LAYER_MAP` for a head whose rpc devices are the shards in tier order and whose local
/// GPU is the last device: every shard layer range on its rpc device, the output layer local.
pub fn layer_map() -> String {
    let mut items: Vec<String> = V4_FLASH_SHARD_LAYERS.iter().enumerate().map(|(k, (a, b))| format!("{}-{}:{}", a, b, k)).collect();
    items.push(format!("{}-{}:{}", V4_FLASH_N_LAYER, V4_FLASH_N_LAYER, V4_FLASH_SHARDS.len()));
    items.join(",")
}

/// What a served pipeline request yields: the text, and the live link sessions that sign it.
pub struct HeadResult {
    pub text: String,
    pub links: Vec<LinkSession>,
}

const ASSIGNMENT_WAIT: Duration = Duration::from_secs(90);

/// Serves `request_hash` as the pipeline head. Fails when this miner was not drawn as head,
/// when a shard has no reachable link, or when generation fails.
pub async fn run_head(request_hash: [u8; 32], prompt: String, max_tokens: usize) -> Result<HeadResult, String> {
    let identity = identity().ok_or("no escrow identity")?;
    // The audit arms one block after acceptance; the template carries it shortly after.
    let deadline = std::time::Instant::now() + ASSIGNMENT_WAIT;
    let assignment = loop {
        if let Some(a) = assignment_for(&request_hash) {
            break a;
        }
        if std::time::Instant::now() > deadline {
            return Err("no assignment published for this request".into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let head_tier = head_tier();
    if !assignment.links.iter().any(|(t, k)| *t == head_tier && *k == identity.pubkey) {
        return Err("not drawn as head for this request".into());
    }
    let mut links: Vec<LinkSession> = Vec::new();
    let mut endpoints: Vec<String> = Vec::new();
    for (k, spec) in V4_FLASH_SHARDS.iter().enumerate() {
        let tier = NETWORK_MODEL_TIER + 1 + k as u8;
        if tier == head_tier {
            let (own, _) = crate::pom_gpu::shard_local_endpoint(&spec.model_id).ok_or("own shard not served")?;
            endpoints.push(own);
            continue;
        }
        // the drawn keys first, then any recent advertiser of the shard as substitute
        let mut candidates: Vec<([u8; 32], String)> = assignment
            .links
            .iter()
            .filter(|(t, _)| *t == tier)
            .filter_map(|(_, k)| shard_gateway::peer_endpoint(k).map(|ep| (*k, ep)))
            .collect();
        for (k, ep) in shard_gateway::peers_for(&spec.model_id) {
            if !candidates.iter().any(|(c, _)| *c == k) {
                candidates.push((k, ep));
            }
        }
        let mut session = None;
        for (key, endpoint) in candidates {
            match shard_gateway::connect_link(&endpoint, spec.model_id, identity.clone(), 0).await {
                Ok(s) if s.peer_pubkey == key && s.tier == tier => {
                    session = Some(s);
                    break;
                }
                Ok(s) => log::warn!("pipeline: {} answered as tier {} key {} — skipped", endpoint, s.tier, hex::encode(s.peer_pubkey)),
                Err(e) => log::warn!("pipeline: shard {} at {}: {}", k, endpoint, e),
            }
        }
        let session = session.ok_or_else(|| format!("no reachable link for shard {} (tier {})", k, tier))?;
        endpoints.push(format!("127.0.0.1:{}", session.local_port));
        links.push(session);
    }
    let gpu = crate::pom_gpu::device_for_model(&V4_FLASH_SHARDS[V4_FLASH_SHARDS.len() - 1].model_id).unwrap_or(0) as usize;
    let head = crate::slm::head_gguf_path().to_string_lossy().into_owned();
    let rpc = endpoints.join(",");
    let ts: Vec<&str> = std::iter::repeat("1").take(endpoints.len() + 1).collect();
    let ts = ts.join(",");
    let map = layer_map();
    let templated = crate::slm::format_prompt_for(V4_FLASH.name, &prompt);
    log::info!("pipeline: heading request {} over {} links", hex::encode(&request_hash[..8]), links.len());
    let text = tokio::task::spawn_blocking(move || crate::llama_engine::head_generate(&head, gpu, &rpc, &ts, &map, &templated, max_tokens))
        .await
        .map_err(|e| format!("head task: {}", e))??;
    if text.trim().is_empty() {
        return Err("empty generation".into());
    }
    Ok(HeadResult { text, links })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignments_parse_and_index_by_request() {
        let rh = [7u8; 32];
        let k = [9u8; 32];
        let enc = format!("{}:100:6500:6-{},11-{}", hex::encode(rh), hex::encode(k), hex::encode([1u8; 32]));
        let a = parse_assignment(&enc).unwrap();
        assert_eq!(a.request_hash, rh);
        assert_eq!(a.accepted_daa, 100);
        assert_eq!(a.window_end_daa, 6500);
        assert_eq!(a.links, vec![(6, k), (11, [1u8; 32])]);
        note_assignments(&format!("{};bad;", enc));
        assert_eq!(assignment_for(&rh), Some(a));
        note_assignments("");
        assert!(assignment_for(&rh).is_none());
    }

    #[test]
    fn layer_map_covers_every_layer_and_the_output() {
        assert_eq!(layer_map(), "0-2:0,3-7:1,8-12:2,13-19:3,20-29:4,30-42:5,43-43:6");
        assert_eq!(head_tier(), 11);
    }
}
