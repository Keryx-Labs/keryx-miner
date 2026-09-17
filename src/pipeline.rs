//! Pipeline head (H14): serves a request addressed to the network model by assembling its
//! shards over the gateway links, generating on the head GGUF with every shard layer bound to
//! a remote resident copy, and collecting the links' signatures for the V3 response.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::models::{network_model, NETWORK_MODEL_TIER};
use crate::shard_gateway::{self, GatewayIdentity, LinkSession};

/// An armed network-model audit as the node publishes it in the block template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub request_hash: [u8; 32],
    pub accepted_daa: u64,
    pub window_end_daa: u64,
    /// `(shard tier, escrow pubkey)` of every availability declaration, chain order. The
    /// first key of a tier is its chosen one, the following ones its substitutes.
    pub links: Vec<(u8, [u8; 32])>,
}

impl Assignment {
    /// The declared keys of `tier`, chain order, each once.
    pub fn declared_keys(&self, tier: u8) -> Vec<[u8; 32]> {
        let mut out: Vec<[u8; 32]> = Vec::new();
        for (t, k) in self.links.iter() {
            if *t == tier && !out.contains(k) {
                out.push(*k);
            }
        }
        out
    }

    /// The first declared key of `tier`.
    pub fn chosen_key(&self, tier: u8) -> Option<[u8; 32]> {
        self.links.iter().find(|(t, _)| *t == tier).map(|(_, k)| *k)
    }

    /// Every shard tier has at least one declaration.
    pub fn complete(&self) -> bool {
        let nm = network_model();
        (0..nm.shards.len()).all(|k| self.chosen_key(NETWORK_MODEL_TIER + 1 + k as u8).is_some())
    }
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

fn responses_seen() -> &'static Mutex<HashSet<[u8; 32]>> {
    static R: OnceLock<Mutex<HashSet<[u8; 32]>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashSet::new()))
}

const MAX_RESPONSES_SEEN: usize = 10_000;

/// Records an AiResponse observed in a block: a substitute head stands down for it.
pub fn note_response_seen(request_hash: [u8; 32]) {
    if let Ok(mut g) = responses_seen().lock() {
        if g.len() >= MAX_RESPONSES_SEEN {
            g.clear();
        }
        g.insert(request_hash);
    }
}

pub fn response_seen(request_hash: &[u8; 32]) -> bool {
    responses_seen().lock().map(|g| g.contains(request_hash)).unwrap_or(false)
}

/// The shard tiers this miner serves right now.
pub fn served_tiers() -> Vec<u8> {
    network_model().shards.iter().filter_map(|spec| crate::pom_gpu::shard_local_endpoint(&spec.model_id).map(|(_, t)| t)).collect()
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
    NETWORK_MODEL_TIER + network_model().shards.len() as u8
}

/// This miner can head a pipeline: it serves the head shard, holds the head GGUF and has an
/// escrow identity.
pub fn head_ready() -> bool {
    let nm = network_model();
    identity().is_some()
        && crate::pom_gpu::shard_local_endpoint(&nm.shards[nm.shards.len() - 1].model_id).is_some()
        && crate::slm::head_gguf_path().exists()
}

/// `KERYX_LAYER_MAP` for a head whose rpc devices are the shards in tier order and whose local
/// GPU is the last device: every shard layer range on its rpc device, the output layer local.
pub fn layer_map() -> String {
    let nm = network_model();
    let mut items: Vec<String> = nm.layers.iter().enumerate().map(|(k, (a, b))| format!("{}-{}:{}", a, b, k)).collect();
    items.push(format!("{}-{}:{}", nm.n_layer, nm.n_layer, nm.shards.len()));
    items.join(",")
}

/// What a served pipeline request yields: the text, and the live link sessions that sign it.
pub struct HeadResult {
    pub text: String,
    pub links: Vec<LinkSession>,
}

/// How long a head waits for its declaration and the other tiers' to reach the audit.
const ASSIGNMENT_WAIT: Duration = Duration::from_secs(90);

/// How long a substitute head leaves the chosen one alone before serving itself.
const SUBSTITUTE_DELAY: Duration = Duration::from_secs(150);

/// What a head does with a network-model request first seen at `since`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadDecision {
    Serve,
    Wait,
    Drop(&'static str),
}

/// Serve right away as the chosen head once every tier has declared, as a substitute after
/// [`SUBSTITUTE_DELAY`] without a response on chain; wait meanwhile, mining on.
pub fn head_decision(request_hash: &[u8; 32], since: std::time::Instant, now_daa: u64) -> HeadDecision {
    let Some(identity) = identity() else { return HeadDecision::Drop("no escrow identity") };
    if response_seen(request_hash) {
        return HeadDecision::Drop("already served");
    }
    let waited = since.elapsed();
    let Some(a) = assignment_for(request_hash) else {
        return if waited > ASSIGNMENT_WAIT { HeadDecision::Drop("no assignment published") } else { HeadDecision::Wait };
    };
    if now_daa >= a.window_end_daa {
        return HeadDecision::Drop("service window closed");
    }
    let head_tier = head_tier();
    if !a.declared_keys(head_tier).contains(&identity.pubkey) {
        return if waited > ASSIGNMENT_WAIT { HeadDecision::Drop("own declaration not in the audit") } else { HeadDecision::Wait };
    }
    if a.chosen_key(head_tier) == Some(identity.pubkey) {
        if a.complete() || waited > ASSIGNMENT_WAIT {
            HeadDecision::Serve
        } else {
            HeadDecision::Wait
        }
    } else if waited > SUBSTITUTE_DELAY {
        HeadDecision::Serve
    } else {
        HeadDecision::Wait
    }
}

/// Serves `request_hash` as the pipeline head, after [`head_decision`] said so. Fails when a
/// shard has no reachable link or when generation fails.
pub async fn run_head(request_hash: [u8; 32], prompt: String, max_tokens: usize) -> Result<HeadResult, String> {
    let identity = identity().ok_or("no escrow identity")?;
    let head_tier = head_tier();
    let assignment = assignment_for(&request_hash).ok_or("no assignment published for this request")?;
    if assignment.chosen_key(head_tier) != Some(identity.pubkey) {
        log::info!("pipeline: chosen head silent for request {} — serving as substitute", hex::encode(&request_hash[..8]));
    }
    let mut links: Vec<LinkSession> = Vec::new();
    let mut endpoints: Vec<String> = Vec::new();
    let nm = network_model();
    for (k, spec) in nm.shards.iter().enumerate() {
        let tier = NETWORK_MODEL_TIER + 1 + k as u8;
        if tier == head_tier {
            let (own, _) = crate::pom_gpu::shard_local_endpoint(&spec.model_id).ok_or("own shard not served")?;
            endpoints.push(own);
            continue;
        }
        // the declared keys in order (chosen first), then any recent advertiser of the shard
        let mut candidates: Vec<([u8; 32], String)> = assignment
            .declared_keys(tier)
            .into_iter()
            .filter_map(|k| shard_gateway::peer_endpoint(&k).map(|ep| (k, ep)))
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
    let gpu = crate::pom_gpu::device_for_model(&nm.shards[nm.shards.len() - 1].model_id).unwrap_or(0) as usize;
    let head = crate::slm::head_gguf_path().to_string_lossy().into_owned();
    let rpc = endpoints.join(",");
    let ts: Vec<&str> = std::iter::repeat("1").take(endpoints.len() + 1).collect();
    let ts = ts.join(",");
    let map = layer_map();
    let templated = crate::slm::format_prompt_for(nm.whole.name, &prompt);
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
    fn declarations_are_ordered_and_the_first_is_chosen() {
        let a = Assignment {
            request_hash: [7u8; 32],
            accepted_daa: 1,
            window_end_daa: 2,
            links: vec![(6, [1u8; 32]), (11, [9u8; 32]), (6, [2u8; 32]), (6, [1u8; 32]), (11, [8u8; 32])],
        };
        assert_eq!(a.declared_keys(6), vec![[1u8; 32], [2u8; 32]]);
        assert_eq!(a.chosen_key(6), Some([1u8; 32]));
        assert_eq!(a.chosen_key(11), Some([9u8; 32]));
        assert_eq!(a.chosen_key(7), None);
        assert!(!a.complete());
        let full = Assignment {
            links: (0..network_model().shards.len()).map(|k| (NETWORK_MODEL_TIER + 1 + k as u8, [k as u8; 32])).collect(),
            ..a.clone()
        };
        assert!(full.complete());
        note_response_seen([5u8; 32]);
        assert!(response_seen(&[5u8; 32]));
        assert!(!response_seen(&[6u8; 32]));
    }

    #[test]
    fn head_decision_waits_serves_and_drops() {
        let rh = [0x42u8; 32];
        let now = std::time::Instant::now();
        let long_ago = now.checked_sub(SUBSTITUTE_DELAY + Duration::from_secs(1)).unwrap_or(now);
        set_identity(Arc::new(GatewayIdentity::from_privkey_hex(&"11".repeat(32)).unwrap()));
        let me = identity().unwrap().pubkey;
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Wait);
        let ht = head_tier();
        let other = [0xEEu8; 32];
        let mut links: Vec<(u8, [u8; 32])> = (0..network_model().shards.len() - 1).map(|k| (NETWORK_MODEL_TIER + 1 + k as u8, [k as u8; 32])).collect();
        links.push((ht, other));
        links.push((ht, me));
        let enc = format!("{}:1:1000:{}", hex::encode(rh), links.iter().map(|(t, k)| format!("{}-{}", t, hex::encode(k))).collect::<Vec<_>>().join(","));
        note_assignments(&enc);
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Wait);
        assert_eq!(head_decision(&rh, long_ago, 0), HeadDecision::Serve);
        assert_eq!(head_decision(&rh, now, 1000), HeadDecision::Drop("service window closed"));
        note_assignments("");
    }

    #[test]
    fn layer_map_covers_every_layer_and_the_output() {
        assert_eq!(layer_map(), "0-2:0,3-7:1,8-12:2,13-19:3,20-29:4,30-42:5,43-43:6");
        assert_eq!(head_tier(), 11);
    }
}
