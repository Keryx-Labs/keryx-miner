//! Network-model pipeline (H14). A link runs its own range on what a declared head sends it; the
//! head runs the last range, samples, and collects the links' signatures for the V3 response.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::models::{network_model, NETWORK_MODEL_TIER};
use crate::shard_gateway::{self, GatewayIdentity, LinkSession, RangeCompute, RangeDims, StepInput};

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

/// Shard tiers with no declaration yet in the published assignment (all of them when none is
/// published): what a head is still waiting for.
pub fn missing_shard_tiers(request_hash: &[u8; 32]) -> Vec<u8> {
    let nm = network_model();
    let a = assignment_for(request_hash);
    (0..nm.shards.len())
        .map(|k| NETWORK_MODEL_TIER + 1 + k as u8)
        .filter(|t| a.as_ref().map_or(true, |a| a.chosen_key(*t).is_none()))
        .collect()
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

/// Whether `head` heads a live assignment (`request` when given) in which `own` declared `tier`.
pub fn authorizes(head: &[u8; 32], own: &[u8; 32], tier: u8, request: Option<&[u8; 32]>) -> bool {
    let Ok(g) = assignments().lock() else { return false };
    let head_tier = head_tier();
    g.values()
        .filter(|a| request.is_none_or(|r| a.request_hash == *r))
        .any(|a| a.declared_keys(head_tier).contains(head) && a.declared_keys(tier).contains(own))
}

/// The shard tiers this miner serves right now.
pub fn served_tiers() -> Vec<u8> {
    network_model().shards.iter().filter_map(|spec| crate::pom_gpu::served_shard(&spec.model_id).map(|(_, t, _)| t)).collect()
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
    identity().is_some() && crate::pom_gpu::served_shard(&nm.shards[nm.shards.len() - 1].model_id).is_some() && crate::llm::head_gguf_path().exists()
}

/// Sequences a link holds at once.
const LINK_SEQS: u32 = 2;

/// Tokens of one sequence, prompt and answer.
fn range_ctx() -> u32 {
    network_model().whole.ctx_cap
}

/// The range of a link, its layers bound to the resident shard; one sequence per open session.
struct LinkRange {
    host: crate::llama_engine::SharedRange,
    dims: RangeDims,
    seqs: Mutex<Vec<bool>>,
}

impl RangeCompute for LinkRange {
    fn dims(&self) -> RangeDims {
        self.dims
    }

    fn open(&self) -> Result<u32, String> {
        let mut g = self.seqs.lock().unwrap_or_else(|p| p.into_inner());
        let i = g.iter().position(|used| !*used).ok_or("every sequence is in use")?;
        g[i] = true;
        Ok(i as u32)
    }

    fn step(&self, seq: u32, pos0: u32, tokens: Option<&[i32]>, input: Option<&[f32]>) -> Result<Vec<f32>, String> {
        let g = self.host.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.as_ref().ok_or("range unloaded")?;
        r.step(seq, pos0, tokens, input).map_err(|rc| format!("range step failed ({})", rc))
    }

    fn close(&self, seq: u32) {
        if let Some(r) = self.host.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            r.clear(seq);
        }
        if let Some(used) = self.seqs.lock().unwrap_or_else(|p| p.into_inner()).get_mut(seq as usize) {
            *used = false;
        }
    }
}

/// The resident range of shard `k` bound into the head GGUF (loaded on first use).
fn bound_range(k: usize, n_seq: u32) -> Result<(u32, crate::llama_engine::SharedRange), String> {
    let nm = network_model();
    let spec = nm.shards.get(k).ok_or("no such shard")?;
    let (gpu, _, gguf) = crate::pom_gpu::served_shard(&spec.model_id).ok_or("shard not resident")?;
    let head = crate::llm::head_gguf_path();
    if !head.exists() {
        return Err("head GGUF not ready".into());
    }
    let (first, last) = nm.layers[k];
    let host = crate::llama_engine::shard_range(gpu as usize, &gguf, &head.to_string_lossy(), first, last, range_ctx(), n_seq)?;
    Ok((gpu, host))
}

/// The link range of shard `k`, shared by every session on it while the shard stays resident.
fn link_range(k: usize) -> Result<Arc<dyn RangeCompute>, String> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<LinkRange>>>> = OnceLock::new();
    let (_, host) = bound_range(k, LINK_SEQS)?;
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(c) = cache.get(&k).filter(|c| Arc::ptr_eq(&c.host, &host)) {
        return Ok(c.clone());
    }
    let dims = {
        let g = host.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.as_ref().ok_or("range unloaded")?;
        RangeDims { n_hc: r.n_hc, n_vocab: r.n_vocab, n_ctx: range_ctx(), from_tokens: r.first == 0 }
    };
    let c = Arc::new(LinkRange { host, dims, seqs: Mutex::new(vec![false; LINK_SEQS as usize]) });
    cache.insert(k, c.clone());
    Ok(c)
}

/// The gateway's view of this miner: a shard it holds as a link (never the head shard).
pub fn link_resolver() -> shard_gateway::ShardResolver {
    Arc::new(|model_id: &[u8; 32]| {
        let k = crate::models::shard_index(model_id)? as usize;
        let (_, tier, _) = crate::pom_gpu::served_shard(model_id)?;
        if tier == head_tier() {
            return None;
        }
        let provider: shard_gateway::RangeProvider = Arc::new(move || link_range(k));
        Some((tier, provider))
    })
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
/// [`SUBSTITUTE_DELAY`] without a response on chain; wait meanwhile, mining on. Only declared
/// signers count on chain, so an incomplete pipeline is never served.
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
    if !a.complete() {
        return HeadDecision::Wait;
    }
    if a.chosen_key(head_tier) == Some(identity.pubkey) || waited > SUBSTITUTE_DELAY {
        HeadDecision::Serve
    } else {
        HeadDecision::Wait
    }
}

/// Most bytes a head answer holds.
const MAX_ANSWER_BYTES: usize = 64 * 1024 - 1;

/// One link of a running pipeline, with what it was fed so a substitute can be brought level.
struct HeadLink {
    k: usize,
    tier: u8,
    model_id: [u8; 32],
    session: LinkSession,
    /// Declared keys not tried yet, chain order.
    spare: std::collections::VecDeque<[u8; 32]>,
    tokens: Vec<i32>,
    residual: Vec<f32>,
    n_hc: usize,
}

impl HeadLink {
    fn fed(&self) -> usize {
        if self.n_hc == 0 {
            0
        } else if self.k == 0 {
            self.tokens.len()
        } else {
            self.residual.len() / self.n_hc
        }
    }

    fn record(&mut self, input: &StepInput) {
        match input {
            StepInput::Tokens(t) => self.tokens.extend_from_slice(t),
            StepInput::Residual(r) => self.residual.extend_from_slice(r),
        }
    }
}

/// Connects and opens the first reachable declared key in `spare`, which loses the keys it tried.
async fn open_next(
    request: &[u8; 32],
    max_len: u32,
    model_id: [u8; 32],
    tier: u8,
    k: usize,
    spare: &mut std::collections::VecDeque<[u8; 32]>,
    identity: &Arc<GatewayIdentity>,
) -> Result<(LinkSession, usize), String> {
    while let Some(key) = spare.pop_front() {
        let Some(endpoint) = shard_gateway::peer_endpoint(&key) else { continue };
        let t = std::time::Instant::now();
        let session = match shard_gateway::connect_link(&endpoint, model_id, identity.clone(), key, tier).await {
            Ok(s) => s,
            Err(e) => {
                log::warn!("pipeline: shard {} at {}: {}", k, endpoint, e);
                continue;
            }
        };
        match session.open(request, max_len).await {
            Ok(d) if d.from_tokens == (k == 0) => {
                log::info!("pipeline head: link shard {} (tier {}) {} open in {} ms", k, tier, endpoint, t.elapsed().as_millis());
                return Ok((session, d.n_hc));
            }
            Ok(_) => log::warn!("pipeline: shard {} at {}: wrong range input", k, endpoint),
            Err(e) => log::warn!("pipeline: shard {} at {}: {}", k, endpoint, e),
        }
    }
    Err(format!("no reachable link for shard {} (tier {})", k, tier))
}

/// Runs `input` at `pos0` through link `l`; a failed link is replaced by the next declared key,
/// which is first fed everything the failed one had computed.
async fn step_link(l: &mut HeadLink, request: &[u8; 32], max_len: u32, identity: &Arc<GatewayIdentity>, pos0: u32, input: StepInput) -> Result<Vec<f32>, String> {
    loop {
        match l.session.step(pos0, &input).await {
            Ok(out) => {
                l.record(&input);
                return Ok(out);
            }
            Err(e) => {
                log::warn!("pipeline: link shard {} failed at position {}: {} — replacing it", l.k, pos0, e);
                let (session, n_hc) = open_next(request, max_len, l.model_id, l.tier, l.k, &mut l.spare, identity).await?;
                if n_hc != l.n_hc {
                    return Err(format!("substitute for shard {} has another residual width", l.k));
                }
                let mut pos = 0usize;
                let total = l.fed();
                while pos < total {
                    let n = (total - pos).min(shard_gateway::MAX_STEP_TOKENS);
                    let chunk = if l.k == 0 {
                        StepInput::Tokens(l.tokens[pos..pos + n].to_vec())
                    } else {
                        StepInput::Residual(l.residual[pos * n_hc..(pos + n) * n_hc].to_vec())
                    };
                    session.step(pos as u32, &chunk).await.map_err(|e| format!("substitute for shard {} failed its replay: {}", l.k, e))?;
                    pos += n;
                }
                l.session = session;
            }
        }
    }
}

/// The head's own range: runs `x` at `pos0` and, when `sample`, draws the next token.
async fn head_step(host: &crate::llama_engine::SharedRange, pos0: u32, x: Vec<f32>, sample: bool) -> Result<Option<i32>, String> {
    let host = host.clone();
    tokio::task::spawn_blocking(move || {
        let g = host.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.as_ref().ok_or("head range unloaded")?;
        r.step(0, pos0, None, Some(&x)).map_err(|rc| format!("head range step failed ({})", rc))?;
        Ok(if sample { Some(r.sample(0).ok_or("no logits to sample")?) } else { None })
    })
    .await
    .map_err(|e| format!("head task: {}", e))?
}

/// One head serves one request at a time.
static HEADING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Serves `request_hash` as the pipeline head, after [`head_decision`] said so: each link runs its
/// own layers on what it receives, this miner runs the last range and samples.
pub async fn run_head(request_hash: [u8; 32], prompt: String, max_tokens: usize) -> Result<HeadResult, String> {
    let _heading = HEADING.lock().await;
    let identity = identity().ok_or("no escrow identity")?;
    let head_tier = head_tier();
    let assignment = assignment_for(&request_hash).ok_or("no assignment published for this request")?;
    if assignment.chosen_key(head_tier) != Some(identity.pubkey) {
        log::info!("pipeline: chosen head silent for request {} — serving as substitute", hex::encode(&request_hash[..8]));
    }
    let nm = network_model();
    let n_links = nm.shards.len() - 1;
    let t0 = std::time::Instant::now();
    let (_, host) = tokio::task::spawn_blocking(move || bound_range(n_links, 1)).await.map_err(|e| format!("head task: {}", e))??;
    let templated = crate::llm::format_prompt_for(nm.whole.name, &prompt);
    let prompt_ids = {
        let g = host.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.as_ref().ok_or("head range unloaded")?;
        r.clear(0);
        r.sampler_reset();
        r.tokenize(&templated)?
    };
    let ctx = range_ctx() as usize;
    if prompt_ids.len() + 16 > ctx {
        return Err(format!("{} prompt tokens do not fit the {}-token context", prompt_ids.len(), ctx));
    }
    let max_len = (prompt_ids.len() + max_tokens).min(ctx) as u32;

    let mut links: Vec<HeadLink> = Vec::with_capacity(n_links);
    for (k, spec) in nm.shards.iter().enumerate().take(n_links) {
        let tier = NETWORK_MODEL_TIER + 1 + k as u8;
        // the declared keys in order, chosen first: only a declared link counts on chain
        let mut spare: std::collections::VecDeque<[u8; 32]> = assignment.declared_keys(tier).into();
        let (session, n_hc) = open_next(&request_hash, max_len, spec.model_id, tier, k, &mut spare, &identity).await?;
        links.push(HeadLink { k, tier, model_id: spec.model_id, session, spare, tokens: Vec::new(), residual: Vec::new(), n_hc });
    }
    log::info!("pipeline: heading request {} over {} links, {} prompt tokens", hex::encode(&request_hash[..8]), links.len(), prompt_ids.len());

    let deadline = t0 + crate::llama_engine::HEAD_GENERATION_DEADLINE;
    let mut answer: Vec<u8> = Vec::new();
    let mut pos = 0u32;
    let mut feed: Vec<i32> = prompt_ids;
    let mut produced = 0usize;
    let result: Result<(), String> = async {
        loop {
            let mut next = None;
            for (c, chunk) in feed.chunks(shard_gateway::MAX_STEP_TOKENS).enumerate() {
                let last = (c + 1) * shard_gateway::MAX_STEP_TOKENS >= feed.len();
                let mut x = step_link(&mut links[0], &request_hash, max_len, &identity, pos, StepInput::Tokens(chunk.to_vec())).await?;
                for l in links.iter_mut().skip(1) {
                    x = step_link(l, &request_hash, max_len, &identity, pos, StepInput::Residual(x)).await?;
                }
                next = head_step(&host, pos, x, last).await?;
                pos += chunk.len() as u32;
            }
            let tok = next.ok_or("no token sampled")?;
            let piece = {
                let g = host.lock().unwrap_or_else(|p| p.into_inner());
                g.as_ref().ok_or("head range unloaded")?.piece(tok)?
            };
            let Some(piece) = piece else { return Ok(()) };
            if answer.len() + piece.len() > MAX_ANSWER_BYTES {
                return Ok(());
            }
            answer.extend_from_slice(&piece);
            produced += 1;
            if produced >= max_tokens || pos >= max_len {
                return Ok(());
            }
            if std::time::Instant::now() > deadline {
                return Err(format!("generation deadline exceeded after {} tokens", produced));
            }
            feed = vec![tok];
        }
    }
    .await;
    if let Some(r) = host.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        r.clear(0);
    }
    for l in &links {
        l.session.close().await;
    }
    result?;
    log::info!("pipeline head: {} tokens, {} bytes in {:.1} s", produced, answer.len(), t0.elapsed().as_secs_f64());
    let text = String::from_utf8(answer).map_err(|e| format!("head output is not UTF-8: {}", e))?;
    if text.trim().is_empty() {
        return Err("empty generation".into());
    }
    Ok(HeadResult { text, links: links.into_iter().map(|l| l.session).collect() })
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
    fn the_head_tier_is_the_last_range() {
        assert_eq!(head_tier(), 8);
    }
}
