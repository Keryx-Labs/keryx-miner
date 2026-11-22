//! Network-model pipeline (H14). A link runs its own range on what a declared head sends it; the
//! head runs the last range, samples, and collects the links' signatures for the V3 response.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::models::{network_model, NETWORK_MODEL_TIER};
use crate::shard_gateway::{self, GatewayIdentity, LinkSession, RangeCompute, RangeDims, StepInput};

/// Place of a declared key once the node drew the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Unranked,
    Pipeline(u32),
    Surplus,
}

/// An armed network-model audit as the node publishes it in the block template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub request_hash: [u8; 32],
    pub accepted_daa: u64,
    pub window_end_daa: u64,
    pub declare_end_daa: u64,
    /// `(shard tier, escrow pubkey, slot)` of every availability declaration, chain order.
    pub links: Vec<(u8, [u8; 32], Slot)>,
    /// Forced steps: `(head key, link tier, link key, position, answer deadline, answered)`.
    pub disputes: Vec<([u8; 32], u8, [u8; 32], u32, u64, bool)>,
}

impl Assignment {
    /// Whether the node drew the ranks.
    pub fn ranked(&self) -> bool {
        self.links.iter().any(|(_, _, s)| *s != Slot::Unranked)
    }

    pub fn declared(&self, tier: u8, key: &[u8; 32]) -> bool {
        self.links.iter().any(|(t, k, _)| *t == tier && k == key)
    }

    /// Slot of `key` declared for `tier`.
    pub fn slot(&self, tier: u8, key: &[u8; 32]) -> Option<Slot> {
        self.links.iter().find(|(t, k, _)| *t == tier && k == key).map(|(_, _, s)| *s)
    }

    /// The key of pipeline `i` for `tier`.
    pub fn pipeline_key(&self, tier: u8, i: u32) -> Option<[u8; 32]> {
        self.links.iter().find(|(t, _, s)| *t == tier && *s == Slot::Pipeline(i)).map(|(_, k, _)| *k)
    }

    pub fn pipeline_count(&self) -> usize {
        self.links.iter().filter_map(|(_, _, s)| if let Slot::Pipeline(i) = s { Some(*i as usize + 1) } else { None }).max().unwrap_or(0)
    }

    /// Keys a head of pipeline `i` may use for `tier`, in the order to try them: its own
    /// pipeline's, then the surplus keys.
    pub fn link_candidates(&self, tier: u8, i: u32) -> Vec<[u8; 32]> {
        let spares = self.links.iter().filter(|(t, _, s)| *t == tier && *s == Slot::Surplus).map(|(_, k, _)| *k);
        self.pipeline_key(tier, i).into_iter().chain(spares).collect()
    }
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    hex::decode(s).ok()?.try_into().ok()
}

/// Parses one `hash:accepted:window_end:declare_end:tier-escrowhex-slot,...:headhex-tier-linkhex-pos-deadline-answered,...` entry.
pub fn parse_assignment(s: &str) -> Option<Assignment> {
    let mut parts = s.split(':');
    let request_hash = unhex32(parts.next()?)?;
    let accepted_daa: u64 = parts.next()?.parse().ok()?;
    let window_end_daa: u64 = parts.next()?.parse().ok()?;
    let declare_end_daa: u64 = parts.next()?.parse().ok()?;
    let mut links = Vec::new();
    for item in parts.next()?.split(',').filter(|i| !i.is_empty()) {
        let mut f = item.split('-');
        let t = f.next()?.parse().ok()?;
        let k = unhex32(f.next()?)?;
        let slot = match f.next()? {
            "u" => Slot::Unranked,
            "s" => Slot::Surplus,
            i => Slot::Pipeline(i.parse().ok()?),
        };
        if f.next().is_some() {
            return None;
        }
        links.push((t, k, slot));
    }
    let mut disputes = Vec::new();
    for item in parts.next()?.split(',').filter(|i| !i.is_empty()) {
        let f: Vec<&str> = item.split('-').collect();
        let [h, t, l, pos, end, answered] = f.as_slice() else { return None };
        let answered = match *answered {
            "0" => false,
            "1" => true,
            _ => return None,
        };
        disputes.push((unhex32(h)?, t.parse().ok()?, unhex32(l)?, pos.parse().ok()?, end.parse().ok()?, answered));
    }
    if parts.next().is_some() {
        return None;
    }
    Some(Assignment { request_hash, accepted_daa, window_end_daa, declare_end_daa, links, disputes })
}

/// Zone codes, in the order of their declared byte.
const ZONES: [&str; keryx_inference::NETWORK_ZONES as usize] = ["eu", "na", "sa", "as", "oc", "af"];

pub fn parse_zone(s: &str) -> Option<u8> {
    ZONES.iter().position(|z| z.eq_ignore_ascii_case(s.trim())).map(|i| i as u8)
}

static ZONE: OnceLock<u8> = OnceLock::new();

pub fn set_zone(zone: u8) {
    let _ = ZONE.set(zone);
}

/// The zone this miner declares.
pub fn zone() -> u8 {
    ZONE.get().copied().unwrap_or(0)
}

pub fn zone_name(zone: u8) -> &'static str {
    ZONES.get(zone as usize).copied().unwrap_or("?")
}

fn assignments() -> &'static Mutex<HashMap<[u8; 32], Assignment>> {
    static A: OnceLock<Mutex<HashMap<[u8; 32], Assignment>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Replaces the known assignments with the template's `;`-separated list, and releases the
/// sequences kept for requests no longer audited.
pub fn note_assignments(encoded: &str) {
    let parsed: HashMap<[u8; 32], Assignment> =
        encoded.split(';').filter(|s| !s.is_empty()).filter_map(parse_assignment).map(|a| (a.request_hash, a)).collect();
    let live: HashSet<[u8; 32]> = parsed.keys().copied().collect();
    if let Ok(mut g) = assignments().lock() {
        *g = parsed;
    }
    shard_gateway::release_retained(&live);
}

/// bf16 bytes of `v`, rounded to nearest even.
pub fn f32s_to_bf16(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    out
}

pub fn bf16_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

const DEMAND_DOMAIN: &[u8] = b"KeryxPipelineDemandV1";
const ANSWER_DOMAIN: &[u8] = b"KeryxPipelineAnswerV1";

/// AiAvail payloads this miner built (forced steps, answers), waiting for submission.
fn outgoing() -> &'static Mutex<Vec<Vec<u8>>> {
    static O: OnceLock<Mutex<Vec<Vec<u8>>>> = OnceLock::new();
    O.get_or_init(Default::default)
}

/// The AiAvail payloads to submit now.
pub fn take_outgoing() -> Vec<Vec<u8>> {
    outgoing().lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default()
}

fn push_outgoing(payload: Vec<u8>) {
    if let Ok(mut g) = outgoing().lock() {
        if g.len() < 256 {
            g.push(payload);
        }
    }
}

/// Publishes a forced step: the link `(tier, key)` must answer position `pos` of `input` on chain.
fn force_step(identity: &GatewayIdentity, request: &[u8; 32], tier: u8, key: [u8; 32], pos: u32, receipt: Option<keryx_inference::AiStepReceipt>, input: Vec<u8>) {
    static FORCED: OnceLock<Mutex<std::collections::HashSet<([u8; 32], u8, [u8; 32])>>> = OnceLock::new();
    if let Ok(mut g) = FORCED.get_or_init(Default::default).lock() {
        if g.len() >= 4096 {
            g.clear();
        }
        if !g.insert((*request, tier, key)) {
            return;
        }
    }
    let receipt = receipt.filter(|r| r.end() == Some(pos));
    let pos = if receipt.is_some() { pos } else { 0 };
    let mut d = keryx_inference::AiDemandPayload {
        request_hash: *request,
        link_tier: tier,
        link_escrow_pubkey: key,
        head_escrow_pubkey: identity.pubkey,
        pos,
        receipt,
        input,
        signature: [0u8; 64],
    };
    d.signature = identity.sign_domain(DEMAND_DOMAIN, &d.signed_bytes());
    log::info!("pipeline: link of tier {} ({}) failed request {} — its position {} is forced on chain", tier, hex::encode(&key[..8]), hex::encode(&request[..8]), pos);
    push_outgoing(d.serialize());
}

/// Forced steps on this miner's keys seen in blocks or templates, by `(request, head, tier, pos)`:
/// their input, and whether an answer was built.
fn demands() -> &'static Mutex<HashMap<([u8; 32], [u8; 32], u8, u32), (Vec<u8>, bool)>> {
    static D: OnceLock<Mutex<HashMap<([u8; 32], [u8; 32], u8, u32), (Vec<u8>, bool)>>> = OnceLock::new();
    D.get_or_init(Default::default)
}

/// Records an AiAvail payload that forces a step on this miner's key.
pub fn note_demand(payload: &[u8]) {
    let Some(d) = keryx_inference::AiDemandPayload::deserialize(payload) else { return };
    if identity().is_none_or(|i| i.pubkey != d.link_escrow_pubkey) {
        return;
    }
    if let Ok(mut g) = demands().lock() {
        if g.len() < 256 {
            g.entry((d.request_hash, d.head_escrow_pubkey, d.link_tier, d.pos)).or_insert((d.input, false));
        }
    }
}

/// Answers the forced steps on this miner's keys the chain now awaits, each once.
pub fn answer_forced_steps(now_daa: u64) {
    let Some(identity) = identity() else { return };
    let due: Vec<(([u8; 32], [u8; 32], u8, u32), Vec<u8>)> = {
        let Ok(mut g) = demands().lock() else { return };
        let mut out = Vec::new();
        g.retain(|(rh, head, tier, pos), (input, started)| {
            let Some(a) = assignment_for(rh) else { return false };
            let open = a.disputes.iter().find(|(h, t, l, p, _, _)| h == head && t == tier && *l == identity.pubkey && p == pos);
            match open {
                Some((_, _, _, _, deadline, false)) if now_daa <= *deadline => {
                    if !*started {
                        *started = true;
                        out.push(((*rh, *head, *tier, *pos), input.clone()));
                    }
                    true
                }
                // not folded yet: keep until the audit closes
                None => true,
                _ => false,
            }
        });
        out
    };
    for ((rh, head, tier, pos), input) in due {
        let identity = identity.clone();
        tokio::task::spawn_blocking(move || match forced_output(&rh, &head, tier, pos, &input) {
            Ok(out) => {
                let mut a = keryx_inference::AiAnswerPayload {
                    request_hash: rh,
                    link_tier: tier,
                    link_escrow_pubkey: identity.pubkey,
                    head_escrow_pubkey: head,
                    pos,
                    trace_root: [0u8; 32],
                    output: f32s_to_bf16(&out),
                    signature: [0u8; 64],
                };
                a.signature = identity.sign_domain(ANSWER_DOMAIN, &a.signed_bytes());
                log::info!("pipeline: answering the step forced on tier {} for request {} at position {}", tier, hex::encode(&rh[..8]), pos);
                push_outgoing(a.serialize());
            }
            Err(e) => log::warn!("pipeline: cannot answer the step forced on tier {} for request {}: {}", tier, hex::encode(&rh[..8]), e),
        });
    }
}

/// This link's output for a forced position: from the sequence kept for that head, or from a
/// fresh one at position 0.
fn forced_output(request: &[u8; 32], head: &[u8; 32], tier: u8, pos: u32, input: &[u8]) -> Result<Vec<f32>, String> {
    let k = (tier - NETWORK_MODEL_TIER - 1) as usize;
    let input = if k == 0 {
        StepInput::Tokens(vec![i32::from_le_bytes(input.try_into().map_err(|_| "bad token input")?)])
    } else {
        StepInput::Residual(bf16_to_f32s(input))
    };
    if let Some(out) = shard_gateway::step_retained(request, head, pos, &input) {
        return out;
    }
    if pos != 0 {
        return Err(format!("no sequence kept at position {}", pos));
    }
    let range = link_range(k)?;
    let seq = range.open()?;
    let out = match &input {
        StepInput::Tokens(t) => range.step(seq, 0, Some(t), None),
        StepInput::Residual(r) => range.step(seq, 0, None, Some(r)),
    };
    range.close(seq);
    out
}

pub fn all_assignments() -> Vec<Assignment> {
    assignments().lock().map(|g| g.values().cloned().collect()).unwrap_or_default()
}

/// Failed links of this head's pipelines, `request -> (tier, key)`.
fn link_failures() -> &'static Mutex<HashMap<[u8; 32], (u8, [u8; 32])>> {
    static F: OnceLock<Mutex<HashMap<[u8; 32], (u8, [u8; 32])>>> = OnceLock::new();
    F.get_or_init(Default::default)
}

/// Reported requests: the node takes one report per head.
fn reported() -> &'static Mutex<HashSet<[u8; 32]>> {
    static R: OnceLock<Mutex<HashSet<[u8; 32]>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// Records that the link `(tier, key)` of this head's pipeline failed `request_hash`.
pub fn note_link_failure(request_hash: [u8; 32], tier: u8, key: [u8; 32]) {
    if reported().lock().is_ok_and(|r| r.contains(&request_hash)) {
        return;
    }
    if let Ok(mut g) = link_failures().lock() {
        g.entry(request_hash).or_insert((tier, key));
    }
}

/// Failure reports to publish, each handed out once.
pub fn take_failure_reports() -> Vec<([u8; 32], u8, [u8; 32])> {
    let Ok(mut g) = link_failures().lock() else { return Vec::new() };
    let Ok(mut done) = reported().lock() else { return Vec::new() };
    let Some(me) = identity().map(|i| i.pubkey) else { return Vec::new() };
    if done.len() >= MAX_RESPONSES_SEEN {
        done.clear();
    }
    let mut out = Vec::new();
    g.retain(|rh, (tier, key)| {
        let Some(a) = assignment_for(rh) else { return false };
        if response_seen(rh, &me) {
            return false;
        }
        if a.disputes.iter().any(|(h, t, l, _, _, answered)| *h == me && t == tier && l == key && *answered) {
            if done.insert(*rh) {
                out.push((*rh, *tier, *key));
            }
            return false;
        }
        true
    });
    out
}

pub fn assignment_for(request_hash: &[u8; 32]) -> Option<Assignment> {
    assignments().lock().ok()?.get(request_hash).cloned()
}

fn responses_seen() -> &'static Mutex<HashSet<([u8; 32], [u8; 32])>> {
    static R: OnceLock<Mutex<HashSet<([u8; 32], [u8; 32])>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashSet::new()))
}

const MAX_RESPONSES_SEEN: usize = 10_000;

/// Records an AiResponse observed in a block, by request and responder key.
pub fn note_response_seen(request_hash: [u8; 32], responder: [u8; 32]) {
    if let Ok(mut g) = responses_seen().lock() {
        if g.len() >= MAX_RESPONSES_SEEN {
            g.clear();
        }
        g.insert((request_hash, responder));
    }
}

/// Whether `responder` already answered `request_hash` on chain.
pub fn response_seen(request_hash: &[u8; 32], responder: &[u8; 32]) -> bool {
    responses_seen().lock().map(|g| g.contains(&(*request_hash, *responder))).unwrap_or(false)
}

/// Heads that opened a surplus key, by `(request, tier, key)`: a spare serves one head.
fn spare_claims() -> &'static Mutex<HashMap<([u8; 32], u8, [u8; 32]), [u8; 32]>> {
    static C: OnceLock<Mutex<HashMap<([u8; 32], u8, [u8; 32]), [u8; 32]>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Whether `head` heads a pipeline of a live assignment (`request` when given) in which `own`
/// is that pipeline's key for `tier`, or a surplus key no other head claimed yet.
pub fn authorizes(head: &[u8; 32], own: &[u8; 32], tier: u8, request: Option<&[u8; 32]>) -> bool {
    let Ok(g) = assignments().lock() else { return false };
    let head_tier = head_tier();
    let Some((rh, pipeline_of_own)) = g.values().filter(|a| request.is_none_or(|r| a.request_hash == *r)).find_map(|a| {
        let Some(Slot::Pipeline(h)) = a.slot(head_tier, head) else { return None };
        match a.slot(tier, own)? {
            Slot::Pipeline(l) if l == h => Some((a.request_hash, true)),
            Slot::Surplus => Some((a.request_hash, false)),
            _ => None,
        }
    }) else {
        return false;
    };
    if pipeline_of_own {
        return true;
    }
    let Ok(mut claims) = spare_claims().lock() else { return false };
    let live: HashSet<[u8; 32]> = g.keys().copied().collect();
    claims.retain(|(r, _, _), _| live.contains(r));
    *claims.entry((rh, tier, *own)).or_insert(*head) == *head
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

/// What a head does with a network-model request first seen at `since`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadDecision {
    Serve,
    Wait,
    Drop(&'static str),
}

/// Whether this head serves now.
pub fn head_decision(request_hash: &[u8; 32], since: std::time::Instant, now_daa: u64) -> HeadDecision {
    let Some(identity) = identity() else { return HeadDecision::Drop("no escrow identity") };
    if response_seen(request_hash, &identity.pubkey) {
        return HeadDecision::Drop("already served");
    }
    let waited = since.elapsed();
    let Some(a) = assignment_for(request_hash) else {
        return if waited > ASSIGNMENT_WAIT { HeadDecision::Drop("no assignment published") } else { HeadDecision::Wait };
    };
    if now_daa >= a.window_end_daa {
        return HeadDecision::Drop("service window closed");
    }
    match a.slot(head_tier(), &identity.pubkey) {
        Some(Slot::Pipeline(_)) => HeadDecision::Serve,
        Some(Slot::Surplus) => HeadDecision::Drop("no pipeline for this head"),
        Some(Slot::Unranked) => HeadDecision::Wait,
        None if a.ranked() => HeadDecision::Drop("own declaration not in the audit"),
        None => HeadDecision::Wait,
    }
}

/// What a head waits for, for the log: the shard tiers with no declaration yet, or the rank draw.
pub fn waiting_for(request_hash: &[u8; 32]) -> String {
    let nm = network_model();
    let a = assignment_for(request_hash);
    let missing: Vec<u8> = (0..nm.shards.len())
        .map(|k| NETWORK_MODEL_TIER + 1 + k as u8)
        .filter(|t| a.as_ref().is_none_or(|a| !a.links.iter().any(|(lt, _, _)| lt == t)))
        .collect();
    match a {
        Some(a) if missing.is_empty() => format!("the rank draw past DAA {}", a.declare_end_daa),
        _ => format!("shard tier(s) {:?} to declare", missing),
    }
}

/// A pipeline the head could not run, with the link of its own pipeline to report when that
/// link's range found no reachable key.
#[derive(Debug)]
pub struct HeadFailure {
    pub reason: String,
    pub failed_link: Option<(u8, [u8; 32])>,
}

impl From<String> for HeadFailure {
    fn from(reason: String) -> Self {
        Self { reason, failed_link: None }
    }
}

impl From<&str> for HeadFailure {
    fn from(reason: &str) -> Self {
        Self { reason: reason.to_string(), failed_link: None }
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
    /// Key of the session, and the last receipt it signed.
    key: [u8; 32],
    receipt: Option<keryx_inference::AiStepReceipt>,
    /// This pipeline's key for the tier, while no step was forced on it yet.
    own: Option<[u8; 32]>,
    /// The own key failed before any step.
    force_at_zero: bool,
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
) -> Result<(LinkSession, usize, [u8; 32]), String> {
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
                return Ok((session, d.n_hc, key));
            }
            Ok(_) => log::warn!("pipeline: shard {} at {}: wrong range input", k, endpoint),
            Err(e) => log::warn!("pipeline: shard {} at {}: {}", k, endpoint, e),
        }
    }
    Err(format!("no reachable link for shard {} (tier {})", k, tier))
}

/// The first position of `input`, as a forced step carries it: a token id, or a bf16 residual.
fn first_position(input: &StepInput, n_hc: usize) -> Vec<u8> {
    match input {
        StepInput::Tokens(t) => t.first().map(|x| x.to_le_bytes().to_vec()).unwrap_or_default(),
        StepInput::Residual(r) => f32s_to_bf16(&r[..n_hc.min(r.len())]),
    }
}

/// Runs `input` at `pos0` through link `l`; a failed link is replaced by the next declared key,
/// which is first fed everything the failed one had computed.
async fn step_link(l: &mut HeadLink, request: &[u8; 32], max_len: u32, identity: &Arc<GatewayIdentity>, pos0: u32, input: StepInput) -> Result<Vec<f32>, String> {
    loop {
        match l.session.step(pos0, &input).await {
            Ok((out, receipt)) => {
                if l.force_at_zero && pos0 == 0 {
                    l.force_at_zero = false;
                    if let Some(own) = l.own.take() {
                        force_step(identity, request, l.tier, own, 0, None, first_position(&input, l.n_hc));
                    }
                }
                l.record(&input);
                l.receipt = Some(receipt);
                return Ok(out);
            }
            Err(e) => {
                log::warn!("pipeline: link shard {} failed at position {}: {} — replacing it", l.k, pos0, e);
                if l.own == Some(l.key) {
                    l.own = None;
                    force_step(identity, request, l.tier, l.key, pos0, l.receipt.take(), first_position(&input, l.n_hc));
                }
                let (session, n_hc, key) = open_next(request, max_len, l.model_id, l.tier, l.k, &mut l.spare, identity).await?;
                if n_hc != l.n_hc {
                    return Err(format!("substitute for shard {} has another residual width", l.k));
                }
                let mut pos = 0usize;
                let total = l.fed();
                let mut receipt = None;
                while pos < total {
                    let n = (total - pos).min(shard_gateway::MAX_STEP_TOKENS);
                    let chunk = if l.k == 0 {
                        StepInput::Tokens(l.tokens[pos..pos + n].to_vec())
                    } else {
                        StepInput::Residual(l.residual[pos * n_hc..(pos + n) * n_hc].to_vec())
                    };
                    let (_, r) = session.step(pos as u32, &chunk).await.map_err(|e| format!("substitute for shard {} failed its replay: {}", l.k, e))?;
                    receipt = Some(r);
                    pos += n;
                }
                l.session = session;
                l.key = key;
                l.receipt = receipt;
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
pub async fn run_head(request_hash: [u8; 32], prompt: String, max_tokens: usize) -> Result<HeadResult, HeadFailure> {
    let _heading = HEADING.lock().await;
    let identity = identity().ok_or("no escrow identity")?;
    let head_tier = head_tier();
    let assignment = assignment_for(&request_hash).ok_or("no assignment published for this request")?;
    let Some(Slot::Pipeline(rank)) = assignment.slot(head_tier, &identity.pubkey) else {
        return Err("this head has no pipeline".into());
    };
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
        return Err(format!("{} prompt tokens do not fit the {}-token context", prompt_ids.len(), ctx).into());
    }
    let max_len = (prompt_ids.len() + max_tokens).min(ctx) as u32;

    let mut links: Vec<HeadLink> = Vec::with_capacity(n_links);
    for (k, spec) in nm.shards.iter().enumerate().take(n_links) {
        let tier = NETWORK_MODEL_TIER + 1 + k as u8;
        let own = assignment.pipeline_key(tier, rank);
        let mut spare: std::collections::VecDeque<[u8; 32]> = assignment.link_candidates(tier, rank).into();
        let opened = open_next(&request_hash, max_len, spec.model_id, tier, k, &mut spare, &identity).await;
        let (session, n_hc, key) = match opened {
            Ok(o) => o,
            Err(reason) => {
                if let Some(own) = own {
                    let mut input = StepInput::Tokens(vec![prompt_ids[0]]);
                    let mut known = true;
                    for l in links.iter_mut() {
                        match l.session.step(0, &input).await {
                            Ok((x, _)) => input = StepInput::Residual(x),
                            Err(_) => {
                                known = false;
                                break;
                            }
                        }
                    }
                    if known {
                        let n_hc = links.last().map_or(0, |l| l.n_hc);
                        force_step(&identity, &request_hash, tier, own, 0, None, first_position(&input, n_hc));
                    }
                }
                for l in &links {
                    l.session.close().await;
                }
                return Err(HeadFailure { reason, failed_link: own.map(|key| (tier, key)) });
            }
        };
        let force_at_zero = own.is_some_and(|o| o != key);
        links.push(HeadLink {
            k,
            tier,
            model_id: spec.model_id,
            session,
            spare,
            tokens: Vec::new(),
            residual: Vec::new(),
            n_hc,
            key,
            receipt: None,
            own: own.filter(|o| *o == key || force_at_zero),
            force_at_zero,
        });
    }
    log::info!(
        "pipeline: heading request {} as pipeline {} of {}, {} prompt tokens",
        hex::encode(&request_hash[..8]),
        rank + 1,
        assignment.pipeline_count(),
        prompt_ids.len()
    );

    let deadline = t0 + crate::llama_engine::HEAD_GENERATION_DEADLINE;
    let mut answer: Vec<u8> = Vec::new();
    let mut pos = 0u32;
    let mut feed: Vec<i32> = prompt_ids;
    let mut produced = 0usize;
    let failed = |reason: String, tier: u8| HeadFailure { reason, failed_link: assignment.pipeline_key(tier, rank).map(|key| (tier, key)) };
    let result: Result<(), HeadFailure> = async {
        loop {
            let mut next = None;
            for (c, chunk) in feed.chunks(shard_gateway::MAX_STEP_TOKENS).enumerate() {
                let last = (c + 1) * shard_gateway::MAX_STEP_TOKENS >= feed.len();
                let t0 = links[0].tier;
                let mut x = step_link(&mut links[0], &request_hash, max_len, &identity, pos, StepInput::Tokens(chunk.to_vec()))
                    .await
                    .map_err(|e| failed(e, t0))?;
                for l in links.iter_mut().skip(1) {
                    let t = l.tier;
                    x = step_link(l, &request_hash, max_len, &identity, pos, StepInput::Residual(x)).await.map_err(|e| failed(e, t))?;
                }
                next = head_step(&host, pos, x, last).await?;
                pos += chunk.len() as u32;
            }
            let tok = next.ok_or(HeadFailure::from("no token sampled"))?;
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
                return Err(format!("generation deadline exceeded after {} tokens", produced).into());
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
    // byte tokens can leave a sequence that is not UTF-8, or one cut by the last token
    let text = String::from_utf8_lossy(&answer).into_owned();
    if text.trim().is_empty() {
        return Err("empty generation".into());
    }
    Ok(HeadResult { text, links: links.into_iter().map(|l| l.session).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that write the process-wide assignments run one at a time.
    fn assignments_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: Mutex<()> = Mutex::new(());
        L.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn slot_code(s: Slot) -> String {
        match s {
            Slot::Unranked => "u".into(),
            Slot::Surplus => "s".into(),
            Slot::Pipeline(i) => i.to_string(),
        }
    }

    fn encode(rh: [u8; 32], links: &[(u8, [u8; 32], Slot)]) -> String {
        encode_with(rh, links, &[])
    }

    fn encode_with(rh: [u8; 32], links: &[(u8, [u8; 32], Slot)], disputes: &[([u8; 32], u8, [u8; 32], u32, u64, bool)]) -> String {
        let l: Vec<String> = links.iter().map(|(t, k, s)| format!("{}-{}-{}", t, hex::encode(k), slot_code(*s))).collect();
        let d: Vec<String> =
            disputes.iter().map(|(h, t, k, p, e, a)| format!("{}-{}-{}-{}-{}-{}", hex::encode(h), t, hex::encode(k), p, e, *a as u8)).collect();
        format!("{}:100:6500:400:{}:{}", hex::encode(rh), l.join(","), d.join(","))
    }

    fn me() -> [u8; 32] {
        set_identity(Arc::new(GatewayIdentity::from_privkey_hex(&"11".repeat(32)).unwrap()));
        identity().unwrap().pubkey
    }

    /// Two pipelines over every shard tier (key `16 * k + i`), a spare on the first range.
    fn two_pipelines(rh: [u8; 32], head1: [u8; 32]) -> String {
        let n = network_model().shards.len();
        let mut links = Vec::new();
        for k in 0..n {
            let t = NETWORK_MODEL_TIER + 1 + k as u8;
            for i in 0..2u8 {
                let key = if t == head_tier() && i == 1 { head1 } else { [16 * k as u8 + i; 32] };
                links.push((t, key, Slot::Pipeline(i as u32)));
            }
        }
        links.push((NETWORK_MODEL_TIER + 1, [2u8; 32], Slot::Surplus));
        encode(rh, &links)
    }

    #[test]
    fn assignments_parse_and_index_by_request() {
        let _serial = assignments_lock();
        let rh = [7u8; 32];
        let enc = encode(rh, &[(6, [9u8; 32], Slot::Pipeline(0)), (11, [1u8; 32], Slot::Surplus), (6, [3u8; 32], Slot::Unranked)]);
        let a = parse_assignment(&enc).unwrap();
        assert_eq!((a.request_hash, a.accepted_daa, a.window_end_daa, a.declare_end_daa), (rh, 100, 6500, 400));
        assert_eq!(a.links, vec![(6, [9u8; 32], Slot::Pipeline(0)), (11, [1u8; 32], Slot::Surplus), (6, [3u8; 32], Slot::Unranked)]);
        assert!(parse_assignment(&format!("{}:1:2:", hex::encode(rh))).is_none());
        assert!(parse_assignment(&format!("{}:1:2:3:6-{}", hex::encode(rh), hex::encode([1u8; 32]))).is_none());
        note_assignments(&format!("{};bad;", enc));
        assert_eq!(assignment_for(&rh), Some(a));
        note_assignments("");
        assert!(assignment_for(&rh).is_none());
    }

    #[test]
    fn zones_parse_by_code() {
        assert_eq!(parse_zone("eu"), Some(0));
        assert_eq!(parse_zone(" AS "), Some(3));
        assert_eq!(parse_zone("af"), Some(keryx_inference::NETWORK_ZONES - 1));
        assert_eq!(parse_zone("mars"), None);
        assert_eq!(zone_name(1), "na");
    }

    #[test]
    fn heads_take_their_pipeline_then_spares() {
        let rh = [0x43u8; 32];
        let a = parse_assignment(&two_pipelines(rh, [0xEEu8; 32])).unwrap();
        assert!(a.ranked());
        assert_eq!(a.pipeline_count(), 2);
        let t5 = NETWORK_MODEL_TIER + 1;
        assert_eq!(a.link_candidates(t5, 0), vec![[0u8; 32], [2u8; 32]], "own key, then the spare");
        assert_eq!(a.link_candidates(t5, 1), vec![[1u8; 32], [2u8; 32]]);
        assert_eq!(a.link_candidates(NETWORK_MODEL_TIER + 2, 1), vec![[17u8; 32]]);
        let unranked = Assignment { links: a.links.iter().map(|(t, k, _)| (*t, *k, Slot::Unranked)).collect(), ..a.clone() };
        assert!(!unranked.ranked());
        assert_eq!(unranked.pipeline_count(), 0);
    }

    #[test]
    fn links_serve_their_own_head_and_a_spare_serves_one() {
        let _serial = assignments_lock();
        let rh = [0x44u8; 32];
        note_assignments(&two_pipelines(rh, [0xEEu8; 32]));
        let t5 = NETWORK_MODEL_TIER + 1;
        let ht = head_tier();
        let n = network_model().shards.len() as u8 - 1;
        let head0 = [16 * n; 32];
        let head1 = [0xEEu8; 32];
        assert!(authorizes(&head0, &[0u8; 32], t5, Some(&rh)));
        assert!(!authorizes(&head1, &[0u8; 32], t5, Some(&rh)), "another pipeline's link");
        assert!(authorizes(&head1, &[1u8; 32], t5, Some(&rh)));
        assert!(!authorizes(&[0u8; 32], &[1u8; 32], t5, Some(&rh)), "not a head");
        assert!(!authorizes(&head0, &[0u8; 32], t5, Some(&[0u8; 32])));
        // the spare goes to the first head that opens it
        assert!(authorizes(&head1, &[2u8; 32], t5, Some(&rh)));
        assert!(authorizes(&head1, &[2u8; 32], t5, None));
        assert!(!authorizes(&head0, &[2u8; 32], t5, Some(&rh)));
        assert_eq!(ht, 8);
        note_assignments("");
    }

    #[test]
    fn head_decision_waits_for_the_rank_draw() {
        let _serial = assignments_lock();
        let rh = [0x42u8; 32];
        let now = std::time::Instant::now();
        set_identity(Arc::new(GatewayIdentity::from_privkey_hex(&"11".repeat(32)).unwrap()));
        let me = identity().unwrap().pubkey;
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Wait);
        let ranked = two_pipelines(rh, me);
        let a = parse_assignment(&ranked).unwrap();
        let unranked: Vec<(u8, [u8; 32], Slot)> = a.links.iter().map(|(t, k, _)| (*t, *k, Slot::Unranked)).collect();
        note_assignments(&encode(rh, &unranked));
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Wait);
        assert!(waiting_for(&rh).contains("rank draw"));
        note_assignments(&ranked);
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Serve);
        assert_eq!(head_decision(&rh, now, 6500), HeadDecision::Drop("service window closed"));
        // a surplus head has nothing to serve
        let spare: Vec<(u8, [u8; 32], Slot)> =
            a.links.iter().map(|(t, k, s)| (*t, *k, if *k == me { Slot::Surplus } else { *s })).collect();
        note_assignments(&encode(rh, &spare));
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Drop("no pipeline for this head"));
        // ranked without this head's declaration
        let without: Vec<(u8, [u8; 32], Slot)> = a.links.iter().filter(|(_, k, _)| *k != me).copied().collect();
        note_assignments(&encode(rh, &without));
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Drop("own declaration not in the audit"));
        note_assignments(&ranked);
        note_response_seen(rh, [0xEEu8; 32]);
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Serve);
        note_response_seen(rh, me);
        assert_eq!(head_decision(&rh, now, 0), HeadDecision::Drop("already served"));
        note_assignments("");
    }

    #[test]
    fn a_failed_link_is_reported_once_its_forced_step_is_answered() {
        let _serial = assignments_lock();
        let me = me();
        let rh = [0x45u8; 32];
        let links = parse_assignment(&two_pipelines(rh, me)).unwrap().links;
        note_assignments(&encode(rh, &links));
        note_link_failure(rh, 6, [1u8; 32]);
        note_link_failure([0x46u8; 32], 6, [1u8; 32]);
        assert!(take_failure_reports().is_empty());
        note_assignments(&encode_with(rh, &links, &[(me, 6, [1u8; 32], 0, 900, false)]));
        assert!(take_failure_reports().is_empty());
        note_assignments(&encode_with(rh, &links, &[(me, 6, [1u8; 32], 0, 900, true)]));
        assert_eq!(take_failure_reports(), vec![(rh, 6, [1u8; 32])]);
        note_link_failure(rh, 6, [1u8; 32]);
        assert!(take_failure_reports().is_empty(), "once per request");
        note_assignments("");
    }

    #[test]
    fn disputes_parse_from_the_assignment() {
        let rh = [0x47u8; 32];
        let enc = encode_with(rh, &[(6, [1u8; 32], Slot::Pipeline(0))], &[([2u8; 32], 6, [1u8; 32], 12, 950, true)]);
        assert_eq!(parse_assignment(&enc).unwrap().disputes, vec![([2u8; 32], 6, [1u8; 32], 12, 950, true)]);
        assert!(parse_assignment(&format!("{}x-1-2-3-4-5", enc)).is_none());
        assert!(parse_assignment(&enc.replace("-950-1", "-950-2")).is_none());
    }

    #[test]
    fn bf16_rounds_to_nearest_and_back() {
        let v = [1.0f32, -2.5, 0.0, 3.140625, 1.0e30];
        let b = f32s_to_bf16(&v);
        assert_eq!(b.len(), 10);
        assert_eq!(&b[..2], &[0x80, 0x3F]);
        assert_eq!(bf16_to_f32s(&b)[..4], [1.0, -2.5, 0.0, 3.140625]);
        assert!(keryx_inference::bf16_all_finite(&b), "large values keep their scale");
        // 1 + 2^-8 sits halfway between two bf16 values: ties go to even
        assert_eq!(bf16_to_f32s(&f32s_to_bf16(&[1.0 + 1.0 / 256.0])), vec![1.0]);
    }

    #[test]
    fn a_forced_step_is_signed_by_the_head() {
        let _serial = assignments_lock();
        let me = me();
        let id = identity().unwrap();
        let _ = take_outgoing();
        let receipt = keryx_inference::AiStepReceipt {
            request_hash: [9u8; 32],
            pos0: 0,
            n: 4,
            input_hash: [0; 32],
            output_hash: [0; 32],
            trace_root: [0; 32],
            previous: [0; 32],
            signature: [0; 64],
        };
        force_step(&id, &[9u8; 32], 7, [3u8; 32], 4, Some(receipt), f32s_to_bf16(&[0.5; 8]));
        force_step(&id, &[9u8; 32], 7, [4u8; 32], 9, Some(receipt), vec![1, 0, 0, 0]);
        force_step(&id, &[9u8; 32], 7, [4u8; 32], 0, None, vec![1, 0, 0, 0]);
        let out = take_outgoing();
        assert_eq!(out.len(), 2);
        let d = keryx_inference::AiDemandPayload::deserialize(&out[0]).unwrap();
        assert_eq!((d.head_escrow_pubkey, d.link_escrow_pubkey, d.link_tier, d.pos, d.receipt), (me, [3u8; 32], 7, 4, Some(receipt)));
        let digest = {
            let mut h = blake2b_simd::Params::new().hash_length(32).to_state();
            h.update(DEMAND_DOMAIN);
            h.update(&d.signed_bytes());
            h.finalize()
        };
        let msg = secp256k1::Message::from_digest_slice(digest.as_bytes()).unwrap();
        let sig = secp256k1::schnorr::Signature::from_slice(&d.signature).unwrap();
        let pk = secp256k1::XOnlyPublicKey::from_slice(&me).unwrap();
        assert!(secp256k1::SECP256K1.verify_schnorr(&sig, &msg, &pk).is_ok());
        let d2 = keryx_inference::AiDemandPayload::deserialize(&out[1]).unwrap();
        assert_eq!((d2.pos, d2.receipt), (0, None));
        assert!(take_outgoing().is_empty());
    }

    #[test]
    fn only_steps_forced_on_this_miner_are_kept() {
        let _serial = assignments_lock();
        let me = me();
        let d = |link: [u8; 32]| keryx_inference::AiDemandPayload {
            request_hash: [0x48u8; 32],
            link_tier: 6,
            link_escrow_pubkey: link,
            head_escrow_pubkey: [5u8; 32],
            pos: 0,
            receipt: None,
            input: vec![1, 0, 0, 0],
            signature: [0; 64],
        };
        note_demand(&d([6u8; 32]).serialize());
        note_demand(&d(me).serialize());
        note_demand(&d(me).serialize());
        let g = demands().lock().unwrap();
        assert_eq!(g.keys().filter(|(rh, ..)| *rh == [0x48u8; 32]).count(), 1);
    }

    #[test]
    fn the_head_tier_is_the_last_range() {
        assert_eq!(head_tier(), 8);
    }
}
