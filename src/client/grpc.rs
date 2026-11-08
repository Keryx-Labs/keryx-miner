use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetAwaitedOpeningsRequestMessage, GetBlockDagInfoRequestMessage, GetBlockRequestMessage,
    GetBlockTemplateRequestMessage, GetInfoRequestMessage, GetServiceProvidersRequestMessage,
    GetServiceStrikesRequestMessage, KaspadMessage, NotifyBlockAddedRequestMessage,
    NotifyNewBlockTemplateRequestMessage, NotifyVirtualSelectedParentChainChangedRequestMessage,
};
use crate::{miner::MinerManager, Error};

/// Asks the node for its virtual DAA score, before the mining client exists. Used to skip
/// downloading the models of eras the chain has already left. `None` on any failure — the caller
/// then keeps every scheduled era, which only costs bandwidth.
pub async fn query_virtual_daa(address: String) -> Option<u64> {
    let mut client = RpcClient::connect(address).await.ok()?;
    let (send, recv) = mpsc::channel(2);
    send.send(GetBlockDagInfoRequestMessage {}.into()).await.ok()?;
    let mut stream = client.message_stream(ReceiverStream::new(recv)).await.ok()?.into_inner();
    while let Ok(Some(msg)) = stream.message().await {
        if let Some(Payload::GetBlockDagInfoResponse(resp)) = msg.payload {
            return resp.error.is_none().then_some(resp.virtual_daa_score);
        }
    }
    None
}

/// Selected-parent chain entries seen while resolving block task seeds: `(daa_score, selected
/// parent, blue_work)` by block hash.
type ChainCache = std::collections::HashMap<String, (u64, String, String)>;

async fn chain_entry(
    tx: &mpsc::Sender<KaspadMessage>,
    stream: &mut Streaming<KaspadMessage>,
    cache: &mut ChainCache,
    hash: &str,
) -> Option<(u64, String, String)> {
    if let Some(e) = cache.get(hash) {
        return Some(e.clone());
    }
    tx.send(GetBlockRequestMessage { hash: hash.to_string(), include_transactions: false }.into()).await.ok()?;
    while let Ok(Some(msg)) = stream.message().await {
        if let Some(Payload::GetBlockResponse(resp)) = msg.payload {
            if resp.error.is_some() {
                return None;
            }
            let block = resp.block?;
            let (header, verbose) = (block.header?, block.verbose_data?);
            let e = (header.daa_score, verbose.selected_parent_hash, header.blue_work);
            cache.insert(hash.to_string(), e.clone());
            return Some(e);
        }
    }
    None
}

fn blue_work_key(w: &str) -> (usize, String) {
    let w = w.trim_start_matches('0').to_ascii_lowercase();
    (w.len(), w)
}

/// Seed block of the task window starting at `start` for a block over `parents`: its selected
/// parent's chain walked back to the first block whose DAA score is at most `start`.
async fn task_window_seed(client: &RpcClient<TonicChannel>, parents: &[String], start: u64, cache: &mut ChainCache) -> Option<[u8; 32]> {
    let mut client = client.clone();
    let (tx, rx) = mpsc::channel(4);
    let mut stream = client.message_stream(ReceiverStream::new(rx)).await.ok()?.into_inner();
    let mut best: Option<((usize, String), String)> = None;
    for p in parents {
        let (_, _, work) = chain_entry(&tx, &mut stream, cache, p).await?;
        let cand = (blue_work_key(&work), p.to_ascii_lowercase());
        if best.as_ref().is_none_or(|b| cand > *b) {
            best = Some(cand);
        }
    }
    let mut cur = best?.1;
    for _ in 0..4096 {
        let (daa, selected_parent, _) = chain_entry(&tx, &mut stream, cache, &cur).await?;
        if daa <= start {
            let mut seed = [0u8; 32];
            hex::decode_to_slice(&cur, &mut seed).ok()?;
            return Some(seed);
        }
        if selected_parent.is_empty() {
            return None;
        }
        cur = selected_parent;
    }
    None
}

/// Max AiRequest queue size — drop oldest when full to prevent unbounded memory growth.
const MAX_AI_QUEUE_SIZE: usize = 64;
/// Max boot-time escrow-validation GetBlock requests in flight at once — each answer
/// sends the next queued one, so thousands of state entries never overwhelm the
/// HTTP/2 flow-control window or delay the mining stream.
const VALIDATION_WINDOW: usize = 64;
/// Max unique stable-ids tracked for deduplication — evict when full.
const MAX_AI_SEEN_IDS: usize = 10_000;

/// DAA lookback of the startup backfill scan: deep enough to cover any service window still
/// open when the miner process comes up.
const AI_BACKFILL_WINDOW_DAA: u64 = 6_000;

/// Backfill GetBlock requests kept in flight.
const AI_BACKFILL_INFLIGHT: usize = 32;

/// Seconds between AiResponse resubmissions while the mempool has not accepted it.
const AI_RESPONSE_RETRY_SECS: u64 = 3;

/// Chain blocks fetched at once while rebuilding the chain after a restart.
const CHAIN_RESTORE_INFLIGHT: usize = 8;

/// Submission attempts before giving up on an AiResponse.
const AI_RESPONSE_MAX_ATTEMPTS: u32 = 100;

/// Opening resubmissions at the response pace before slowing to [`OPENING_SLOW_RETRY_SECS`].
const OPENING_FAST_RETRIES: u32 = 10;
const OPENING_SLOW_RETRY_SECS: u64 = 60;

/// Delay before a response accepted ahead of its request is sent again: past the mempool's
/// memory of accepted transactions.
const EARLY_RESPONSE_RESEND_DELAY: std::time::Duration = std::time::Duration::from_secs(125);
/// Sends of such a response before it is left alone.
const EARLY_RESPONSE_RESENDS: u32 = 10;

/// DAA an unaudited trace is kept past its seed block.
const TRACE_UNAUDITED_GRACE_DAA: u64 = 600;

/// Chain-log entries kept beyond the ones the pending traces still need.
const CHAIN_LOG_TAIL: usize = 512;


/// Built opening bodies kept per response, one per seed block.
const BUILT_OPENINGS_KEPT: usize = 4;

/// Served request hashes kept for strike attribution.
const SERVED_REQUESTS_KEPT: usize = 1024;

/// Reported strike burns remembered so each miss is logged once.
const STRIKE_BURNS_REPORTED_KEPT: usize = 4096;

/// Who a missed request belonged to, from this instance's point of view.
#[derive(Debug, PartialEq, Eq)]
enum StrikeCause {
    /// This instance answered the request (response txid).
    AnsweredHere(String),
    /// This instance queued the request but never answered it.
    DroppedHere,
    /// This instance saw the request but had no ready model to serve it.
    ModelUnreadyHere,
    /// The request never reached this instance.
    Elsewhere,
}

/// Attributes a pending burn to this instance; `None` when the burn carries no request hash.
fn attribute_strike(
    served_requests: &std::collections::HashMap<String, String>,
    seen_prefixes: &std::collections::HashSet<String>,
    skipped_unready: &std::collections::HashSet<String>,
    request_hash: &str,
) -> Option<StrikeCause> {
    let hash = request_hash.to_ascii_lowercase();
    if hash.is_empty() {
        return None;
    }
    if let Some(txid) = served_requests.get(&hash) {
        return Some(StrikeCause::AnsweredHere(txid.clone()));
    }
    let sid: String = hash.chars().take(16).collect();
    if seen_prefixes.contains(&sid) {
        return Some(StrikeCause::DroppedHere);
    }
    if skipped_unready.contains(&sid) {
        return Some(StrikeCause::ModelUnreadyHere);
    }
    Some(StrikeCause::Elsewhere)
}

/// Records that this instance answered `request_hash` with `response_txid` (FIFO-bounded).
fn record_served_request(
    served: &mut std::collections::HashMap<String, String>,
    order: &mut VecDeque<String>,
    request_hash: [u8; 32],
    response_txid: &str,
) {
    let key = hex::encode(request_hash);
    if served.contains_key(&key) {
        return;
    }
    served.insert(key.clone(), response_txid.to_string());
    order.push_back(key);
    while order.len() > SERVED_REQUESTS_KEPT {
        if let Some(old) = order.pop_front() {
            served.remove(&old);
        }
    }
}

/// Node silence after which a template request probes the connection.
const NODE_SILENCE_PROBE: std::time::Duration = std::time::Duration::from_secs(30);
/// Node silence after which the connection is dropped and reopened.
const NODE_SILENCE_LIMIT: std::time::Duration = std::time::Duration::from_secs(90);

/// Margin of an audit replay over its estimated cost before it takes the GPU from requests.
const REPLAY_URGENT_BASE_DAA: u64 = 1_500;
/// A chain rebuild answer of at least this many blocks is continued from its last block.
const CHAIN_RESTORE_PAGE_MIN: usize = 100;
/// Longest a response waits for its acceptance and draw: past the node's ledger horizon its
/// request is no longer pending.
const TRACE_UNDRAWN_LIMIT_DAA: u64 = 72_000;
/// Tries of a request whose inference fails for a passing reason, and the first retry delay.
const MAX_REQUEST_RETRIES: u8 = 3;
const REQUEST_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(20);
/// Token budget cap of a `KRX:AI:1` request, the node's request cap.
const MAX_KRX_AI_TOKENS: u64 = 4_096;
/// First delay before a failed chain rebuild is retried, doubled at each failure up to the max.
const CHAIN_RESTORE_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(5);
const CHAIN_RESTORE_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(60);

/// First delay before a failed replay is retried, doubled at each failure up to the max.
const REPLAY_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(30);
const REPLAY_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(600);

/// Delay before the replay retry following `failures` failed replays.
fn replay_backoff(failures: u32) -> std::time::Instant {
    std::time::Instant::now() + REPLAY_RETRY_BASE.saturating_mul(1 << failures.min(5)).min(REPLAY_RETRY_MAX)
}

use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));

/// A finished inference waiting for its CID to be confirmed readable.
struct PendingPublish {
    request_hash: [u8; 32],
    result: String,
    rx: oneshot::Receiver<Option<[u8; 34]>>,
}

/// An inference running for a queued request.
struct InferenceJob {
    stable_id: String,
    request_hash: [u8; 32],
    private: Option<PrivateRequest>,
    model_id: [u8; 32],
    tier: Option<u8>,
    max_tokens: u32,
    /// The result (`None`: the inference failed, no AiResponse is submitted), the duration of
    /// the inference in seconds, and whether a failure was transient.
    rx: oneshot::Receiver<(Option<keryx_miner::llm::Generated>, f64, bool)>,
    /// The queued request, put back in the queue after a transient failure.
    retry: Option<QueuedRequest>,
}

/// The trace of a finished inference, on its way into the AiResponse.
struct TraceContext {
    model_id: [u8; 32],
    tier: u8,
    max_tokens: u32,
    trace: keryx_miner::trace::ResponseTrace,
    /// Measured duration of the generation, in seconds.
    gen_secs: f64,
}

/// An AI request read from a transaction: raw payload, model id, prompt, plaintext prompt length,
/// max tokens, inference reward, private state.
type ExtractedRequest = (Vec<u8>, [u8; 32], String, u32, usize, u64, Option<PrivateRequest>);

/// A traced response, kept until its draw is settled and its opening window closes.
struct PendingTrace {
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    trace: Arc<keryx_miner::trace::ResponseTrace>,
    submitted_daa: u64,
    /// Chain block that accepted the response.
    accepted_hash: Option<String>,
    /// Last chain block known when the response was submitted.
    anchor_hash: Option<String>,
    /// Measured duration of its generation, in seconds; unknown once restored.
    gen_secs: Option<f64>,
    draw: Option<TraceDraw>,
    /// Opening already built and submitted for this response.
    opening: Option<keryx_miner::trace_store::StoredOpening>,
    /// Unsigned opening bodies already built, by seed block.
    built: Vec<(String, Vec<u8>)>,
    /// Attention phase body built while its draw was not known: seed block, path, body.
    built_phase: Option<(String, Vec<u64>, Vec<u8>)>,
    /// Chain reorgs that reset its acceptance or draw.
    reorgs: u32,
    /// Whether the node listed it at its last awaited-openings poll.
    node_awaits: Option<bool>,
    /// Attention phases after the first, once the first is submitted.
    attn: Option<AttnPhases>,
    /// The response transaction, while this run still holds it.
    response_tx: Option<crate::proto::RpcTransaction>,
}

/// The attention phases of an audited response, phase order.
struct AttnPhases {
    /// Unsigned body, opening transaction, signed payload and accepting chain block of each
    /// submitted phase (unknown for phases restored before the last).
    bodies: Vec<Vec<u8>>,
    txids: Vec<String>,
    payloads: Vec<String>,
    accepted: Vec<Option<String>>,
    /// Child drawn after each accepted phase, and the chain block that drew it.
    path: Vec<u64>,
    child_blocks: Vec<Option<String>>,
    /// Deadline of the last submitted or the next phase.
    deadline_daa: u64,
    /// A replay of the next phase was started for the current path.
    opened: bool,
    failures: u32,
    /// Earliest retry of a failed replay of the next phase.
    retry_at: Option<std::time::Instant>,
    /// Phase transactions undone by a reorg, and the chain block that accepted one afterwards: a
    /// phase rebuilt for the same child has the same transaction.
    dropped: std::collections::HashSet<String>,
    late: std::collections::HashMap<String, String>,
}

impl AttnPhases {
    fn new(body: Vec<u8>, txid: String, payload: String, deadline_daa: u64) -> Self {
        Self {
            bodies: vec![body],
            txids: vec![txid],
            payloads: vec![payload],
            accepted: vec![None],
            path: Vec::new(),
            child_blocks: Vec::new(),
            deadline_daa,
            opened: false,
            failures: 0,
            retry_at: None,
            dropped: Default::default(),
            late: Default::default(),
        }
    }

    fn restored(s: keryx_miner::trace_store::StoredAttn, txid: String) -> Option<Self> {
        let last = keryx_verifiable_inference::opening::AttnOpening::decode(s.bodies.last()?)?;
        let n = s.bodies.len();
        if last.path.len() + 1 != n {
            return None;
        }
        let mut accepted = vec![Some(String::new()); n - 1];
        accepted.push(s.accepted_hash);
        let mut txids = vec![String::new(); n - 1];
        txids.push(txid);
        let mut payloads = vec![String::new(); n - 1];
        payloads.push(s.payload);
        Some(Self {
            bodies: s.bodies,
            txids,
            payloads,
            accepted,
            child_blocks: vec![None; last.path.len()],
            path: last.path,
            deadline_daa: s.deadline_daa,
            opened: false,
            failures: 0,
            retry_at: None,
            dropped: Default::default(),
            late: Default::default(),
        })
    }

    fn stored(&self) -> keryx_miner::trace_store::StoredAttn {
        keryx_miner::trace_store::StoredAttn {
            bodies: self.bodies.clone(),
            payload: self.payloads.last().cloned().unwrap_or_default(),
            deadline_daa: self.deadline_daa,
            accepted_hash: self.accepted.last().cloned().flatten(),
        }
    }

    /// Partials the last submitted phase committed; empty after the last phase.
    fn last_partials(&self) -> Vec<u8> {
        self.bodies.last().and_then(|b| keryx_verifiable_inference::opening::AttnOpening::decode(b)).map(|o| o.partials).unwrap_or_default()
    }

    /// Chain block that accepted the last submitted phase, when its child is still to draw.
    fn awaiting_draw(&self) -> Option<&str> {
        if self.path.len() + 1 != self.bodies.len() || self.last_partials().is_empty() {
            return None;
        }
        self.accepted.last()?.as_deref().filter(|h| !h.is_empty())
    }

    /// Whether the next phase can be built: its child is drawn and no replay was started.
    /// Whether the next phase is drawn and not submitted yet.
    fn next_phase_owed(&self) -> bool {
        self.path.len() == self.bodies.len()
    }

    fn ready(&self) -> bool {
        self.path.len() == self.bodies.len() && !self.opened && self.retry_at.is_none_or(|t| t <= std::time::Instant::now())
    }

    /// Undoes the phases and children that depended on `removed` chain blocks; returns the
    /// opening transactions dropped, or `None` when nothing changed.
    fn on_removed(&mut self, removed: &std::collections::HashSet<&str>) -> Option<Vec<String>> {
        let gone = |h: &Option<String>| h.as_deref().is_some_and(|h| removed.contains(h));
        let k = self.accepted.iter().position(gone);
        let i = self.child_blocks.iter().position(gone);
        let cut = match (k, i) {
            (Some(k), Some(i)) => k.min(i),
            (Some(c), None) | (None, Some(c)) => c,
            (None, None) => return None,
        };
        let dropped = self.txids.split_off(cut + 1);
        self.dropped.extend(dropped.iter().cloned());
        self.bodies.truncate(cut + 1);
        self.payloads.truncate(cut + 1);
        self.accepted.truncate(cut + 1);
        self.path.truncate(cut);
        self.child_blocks.truncate(cut);
        if k == Some(cut) {
            self.accepted[cut] = None;
        }
        self.opened = false;
        self.failures = 0;
        Some(dropped)
    }
}

/// The draw of an accepted response once its seed block is on the chain.
struct TraceDraw {
    seed_hash: String,
    seed_daa: u64,
    /// DAA score of the chain block that accepted the response.
    accepted_daa: u64,
    seed: [u8; 32],
    audited: bool,
    deadline_daa: u64,
    /// A replay was started for this seed.
    opened: bool,
    /// Replays of this seed that built no opening.
    replay_failures: u32,
    /// Earliest retry of a failed replay.
    retry_at: Option<std::time::Instant>,
}

impl TraceDraw {
    /// Whether the first opening still needs a replay that may start now.
    fn replay_ready(&self) -> bool {
        self.audited && !self.opened && self.retry_at.is_none_or(|t| t <= std::time::Instant::now())
    }
}

/// A chain block in chain order; its DAA score arrives with the GetBlock answer.
struct ChainEntry {
    hash: String,
    daa: Option<u64>,
}

/// The replay of an audited response, running on the model's GPU.
struct ReplayJob {
    response_txid: String,
    seed_hash: String,
    /// Path of the attention phase it builds; `None` for the first opening.
    path: Option<Vec<u64>>,
    started: std::time::Instant,
    rx: oneshot::Receiver<Option<Vec<u8>>>,
}

/// An opening the mempool has not accepted yet.
struct OpeningInflight {
    tx: crate::proto::RpcTransaction,
    response_txid: String,
    deadline_daa: u64,
    attempts: u32,
    last: std::time::Instant,
}

fn hash32(hex_str: &str) -> Option<[u8; 32]> {
    hex::decode(hex_str).ok()?.try_into().ok()
}

/// Node version that answers `GetAwaitedOpenings`.
const AWAITED_OPENINGS_SINCE: (u32, u32, u32) = (1, 6, 6);
/// Latest opening deadline after a response's acceptance on the node; matches the node.
const AUDIT_SPAN_DAA: u64 = 270_000;
/// Attention phases the node's span test reserves; matches the node.
const AUDIT_PHASES_RESERVED: u64 = 3;
/// Seconds between two `GetAwaitedOpenings` polls.
const AWAITED_POLL_SECS: u64 = 10;
/// Response txids per `GetAwaitedOpenings` poll; the node accepts up to 1 024.
const AWAITED_POLL_MAX: usize = 1_024;
/// Seconds between two `GetServiceProviders` polls.
const PROVIDERS_POLL_SECS: u64 = 60;
/// Seconds added to the estimated generation time per sibling rank.
const SIBLING_MARGIN_SECS: u64 = 15;
/// Seconds between polls while a queued request has no node check yet.
const UNCHECKED_POLL_SECS: u64 = 2;
/// Longest wait for a node check before starting a request anyway.
const REQUEST_CHECK_WAIT_SECS: u64 = 15;
/// DAA kept between the end of an estimated generation and the answer window end.
const WINDOW_SAFETY_DAA: u64 = 600;

/// Rank of `own` among the escrow keys of its identity for `request_hash`: the rigs of one
/// address agree on it without talking to each other.
fn sibling_rank(request_hash: &[u8; 32], own: &[u8; 32], siblings: &[[u8; 32]]) -> usize {
    let key = |e: &[u8; 32]| {
        let mut b = [0u8; 64];
        b[..32].copy_from_slice(request_hash);
        b[32..].copy_from_slice(e);
        *blake2b_simd::blake2b(&b).as_bytes().first_chunk::<32>().unwrap()
    };
    let mine = key(own);
    siblings.iter().filter(|e| *e != own && key(e) < mine).count()
}

/// Whether the leading `major.minor.patch` of `version` is at least `min`.
fn version_at_least(version: &str, min: (u32, u32, u32)) -> bool {
    let mut parts = version.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse::<u32>().unwrap_or(0));
    let v = (parts.next().unwrap_or(0), parts.next().unwrap_or(0), parts.next().unwrap_or(0));
    v >= min
}

/// The seed block of a response accepted by chain block `accepted`: the first later chain block
/// at `TRACE_SEED_DELAY_DAA` or more, with its DAA score and the accepting block's. `None` while
/// a block on the way has no DAA score yet.
fn seed_block_after<'a>(log: &'a VecDeque<ChainEntry>, accepted: &str) -> Option<(&'a str, u64, u64)> {
    let start = log.iter().position(|e| e.hash == accepted)?;
    let accepted_daa = log[start].daa?;
    for e in log.iter().skip(start + 1) {
        let daa = e.daa?;
        if daa >= accepted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA {
            return Some((e.hash.as_str(), daa, accepted_daa));
        }
    }
    None
}

/// Puts `added`, the chain after `start` in chain order, right behind `start` in `log` (at its end
/// when `start` is not logged), moving the blocks already logged elsewhere and keeping their DAA
/// scores. Returns the placed blocks whose DAA score is unknown.
fn place_chain_after(log: &mut VecDeque<ChainEntry>, start: &str, added: Vec<String>) -> Vec<String> {
    let pos = log.iter().position(|e| e.hash == start).unwrap_or(usize::MAX);
    let added_set: std::collections::HashSet<&str> = added.iter().map(String::as_str).collect();
    let mut known: std::collections::HashMap<String, Option<u64>> = std::collections::HashMap::new();
    let mut kept: VecDeque<ChainEntry> = VecDeque::with_capacity(log.len() + added.len());
    let mut at = 0;
    for (i, e) in std::mem::take(log).into_iter().enumerate() {
        if added_set.contains(e.hash.as_str()) {
            known.insert(e.hash, e.daa);
            continue;
        }
        kept.push_back(e);
        if i == pos {
            at = kept.len();
        }
    }
    if pos == usize::MAX {
        at = kept.len();
    }
    let mut unknown = Vec::new();
    for (k, hash) in added.into_iter().enumerate() {
        let daa = known.remove(&hash).flatten();
        if daa.is_none() {
            unknown.push(hash.clone());
        }
        kept.insert(at + k, ChainEntry { hash, daa });
    }
    *log = kept;
    unknown
}

/// Private-inference state of a queued request; `None` for a plaintext request.
#[derive(Clone)]
enum PrivateRequest {
    /// Opened with the escrow key; the answer is sealed under this key.
    Opened(keryx_inference::ResponseKey),
    /// Names this miner but cannot be opened; answered with a plaintext error body.
    Unreadable,
}

/// Body of the answer to a private request this miner could not open.
const UNREADABLE_PRIVATE_REQUEST_ANSWER: &str = "keryx: private request could not be opened by this responder";
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    task_chain: ChainCache,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,

    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Queue of AiRequests waiting for inference.
    /// Each entry: (stable_id_hex16, request_hash, model_id, prompt, max_tokens, private state).
    /// Fed by both BlockAdded scans and block template scans.
    ai_request_queue: VecDeque<QueuedRequest>,

    /// Block hashes queued for boot-time escrow-state validation, drained in slices of
    /// VALIDATION_WINDOW so thousands of GetBlock requests never saturate the HTTP/2
    /// flow-control window (each consumed answer sends the next queued request).
    validation_queue: VecDeque<String>,

    /// Stable IDs already queued or in-flight — used for deduplication.
    ai_seen_prefixes: std::collections::HashSet<String>,

    /// Stable IDs of requests this escrow key has already answered on-chain; such a request is
    /// never served again (the node rejects the duplicate, the inference would be wasted).
    ai_answered_by_me: std::collections::HashSet<String>,

    /// Maps stable_id → (txid, inference_reward_sompi) for confirmed AiRequest TXs.
    /// Used by poll_inference to register the escrow outpoint after a successful AiResponse.
    ai_request_txids: std::collections::HashMap<String, (String, u64)>,

    /// In-flight AiResponse submissions not yet accepted by the mempool, keyed by txid.
    /// Value: (tx, submit attempts, last submit time). Resubmitted until accepted or expired —
    /// a transiently rejected response must keep trying while the service window is open.
    ai_response_inflight: std::collections::HashMap<String, (crate::proto::RpcTransaction, u32, std::time::Instant)>,

    /// Startup backfill: recent ancestors are walked parent-by-parent and scanned, so an
    /// AiRequest accepted before this process started is still served while its window is
    /// open. `None` cutoff = not seeded yet.
    backfill_cutoff_daa: Option<u64>,
    backfill_queue: VecDeque<String>,
    backfill_pending: std::collections::HashSet<String>,
    backfill_visited: std::collections::HashSet<String>,

    /// In-flight LLM inference for a queued request.
    inference_rx: Option<InferenceJob>,

    /// Traced responses by response txid, from submission to the end of their opening window.
    pending_traces: std::collections::HashMap<String, PendingTrace>,

    /// Recent chain blocks in chain order, back to the oldest one a pending trace needs.
    chain_log: VecDeque<ChainEntry>,

    /// On-disk copy of the pending traces.
    trace_store: keryx_miner::trace_store::TraceStore,
    /// Chain blocks to fetch after a restart, from the accepting block of each restored trace.
    chain_restore_queue: VecDeque<String>,
    chain_restore_pending: std::collections::HashSet<String>,
    /// Accepting blocks whose chain is asked for after a restart, one request at a time; their
    /// traces are not drawn before the answer.
    chain_restore_starts: VecDeque<String>,
    /// Failed chain rebuilds, by accepting block.
    chain_restore_failures: std::collections::HashMap<String, u32>,
    /// Failed chain rebuilds waiting for their retry.
    chain_restore_retry: Vec<(std::time::Instant, String)>,
    /// Chain-log blocks whose GetBlock failed: retry time, hash, failures.
    chain_block_retry: Vec<(std::time::Instant, String, u32)>,
    /// Traced responses waiting for their trace to reach the disk before they are sent.
    persisting: Vec<(std::sync::mpsc::Receiver<()>, String, crate::proto::RpcTransaction)>,
    /// Responses accepted before their request, sent again once it is accepted: next send,
    /// txid, sends so far.
    response_resend: Vec<(std::time::Instant, String, u32)>,
    /// Chain rebuilds continuing a page that ended before the tip.
    chain_restore_continuations: std::collections::HashSet<String>,

    /// In-flight replay building an opening.
    replay_rx: Option<ReplayJob>,

    /// Opening submissions not yet accepted by a chain block, keyed by txid.
    opening_inflight: std::collections::HashMap<String, OpeningInflight>,

    /// Finished inference being uploaded and confirmed on the public gateways; its AiResponse
    /// is built once the task hands back the CID (None: dropped).
    publish_rx: Option<PendingPublish>,

    /// In-flight inference for a node-issued challenge.
    /// Tuple: (challenge_string, result_receiver) where challenge_string = "model_id_hex:nonce_hex".
    /// When the result arrives, it is sent back via inference_result in the next GetBlockTemplateRequest.
    challenge_inference_rx: Option<(String, oneshot::Receiver<Option<String>>)>,

    /// Shared flag with MinerManager — suppresses GPU stall warnings during inference.
    opoi_challenge_active: Option<Arc<AtomicBool>>,

    /// Tracks pending submit-block requests so rejections can be attributed to the worker
    /// that originated the submission even though the submit response carries no device id.
    pending_block_submissions: Arc<Mutex<VecDeque<(String, String)>>>,

    /// Last DAA score seen in a block template — used to compute challenge_window_end.
    last_known_daa: u64,

    /// IPFS Kubo API URL for uploading inference results.
    ipfs_url: String,

    /// 64-char hex Schnorr pubkey embedded in coinbase extra_data as `/escrow:<pubkey>`.
    /// The node routes 20% of the block reward to the corresponding CSV-locked escrow output.
    escrow_pubkey: Option<String>,

    /// Auto-claim module: present when an escrow private key is available.
    escrow_watcher: Option<crate::escrow::EscrowWatcher>,

    /// 128-char hex delegation cert embedded as `/esig:<cert>`, binding the escrow key above to
    /// the payout address. Mandatory from H6 — a block without it is invalid.
    escrow_cert: Option<String>,

    /// Service-ledger identity of the payout address (the node's `miner_key`), the key strikes,
    /// burns and suspensions are reported against.
    service_identity: Option<String>,

    /// Last service-bond strike poll instant.
    last_strike_poll: std::time::Instant,
    /// The node answers `GetAwaitedOpenings`.
    node_awaited_openings: bool,
    /// A GetAwaitedOpenings answer was applied on this connection.
    awaited_answered: bool,
    last_awaited_poll: std::time::Instant,
    /// Responses asked about at the last awaited-openings poll.
    awaited_polled: Vec<String>,
    /// Queued requests asked about at the last awaited-openings poll (stable id, hash).
    awaited_requests_polled: Vec<(String, [u8; 32])>,
    /// An awaited-openings poll is unanswered, and when it was sent.
    awaited_poll_pending: bool,
    awaited_poll_sent: std::time::Instant,
    /// Poll the node at the next tick (a request was just queued).
    force_awaited_poll: bool,
    /// The node checks queued requests against this rig's identity.
    node_checks_requests: bool,
    /// Last node check per queued request: still awaited, window end, when.
    request_checks: std::collections::HashMap<String, (bool, u64, std::time::Instant)>,
    /// When each request was queued.
    request_queued_at: std::collections::HashMap<String, std::time::Instant>,
    /// Requests put back after a transient failure: tries so far and when the next may start.
    request_retries: std::collections::HashMap<String, (u8, std::time::Instant)>,
    /// Escrow keys of this rig's identity per tier, own key included.
    sibling_escrows: std::collections::HashMap<u32, Vec<[u8; 32]>>,
    last_providers_poll: std::time::Instant,
    /// Last message received from the node, and last probe sent while it is silent.
    last_message: std::time::Instant,
    last_probe: std::time::Instant,

    /// Last rendered service-bond status — logged only on change.
    strike_status: Option<String>,

    /// Status bar sink, so the standing is visible without reading the log.
    stats: Option<Arc<crate::stats::MinerStats>>,

    /// DAA of every miss seen since this miner started, for the lifetime tally. Counting the
    /// active strike counter would miss the worst ones: the third strike resets it to zero, and a
    /// served response clears it too. Each miss carries its own daa, so a set of those is exact
    /// for as long as the process runs.
    misses_seen: std::collections::HashSet<u64>,

    /// Request hash (hex) → response txid of the AiResponses this instance submitted or restored.
    served_requests: std::collections::HashMap<String, String>,
    /// FIFO eviction order of `served_requests`.
    served_requests_order: VecDeque<String>,
    /// Request hashes of pending burns already attributed in the log.
    strike_burns_reported: std::collections::HashSet<String>,

    /// Stable IDs of requests skipped because no ready model could serve them.
    ai_skipped_unready: std::collections::HashSet<String>,
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    async fn register(&mut self) -> Result<(), Error> {
        // We actually register in connect
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        self.opoi_challenge_active = Some(miner.opoi_challenge_flag());
        self.stats = Some(miner.stats_handle());
        // Harvest in-flight inference on a timer, independently of node notifications.
        // On a sole-producer node, pausing mining for inference stops block production,
        // so the node stops sending NewBlockTemplate notifications — without this timer
        // the finished inference would never be collected and mining would deadlock.
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(200));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let maybe_msg = tokio::select! {
                msg = self.stream.message() => Some(msg?),
                _ = tick.tick() => None,
            };
            if maybe_msg.is_some() {
                self.last_message = std::time::Instant::now();
            }
            match maybe_msg {
                Some(Some(m)) => match m.payload {
                    Some(payload) => self.handle_message(payload, miner).await?,
                    None => warn!("keryxd message payload is empty"),
                },
                Some(None) => break, // stream closed by node
                None => {
                    let silent = self.last_message.elapsed();
                    if silent >= NODE_SILENCE_LIMIT {
                        return Err(format!("no message from keryxd for {} s — reconnecting", silent.as_secs()).into());
                    }
                    if silent >= NODE_SILENCE_PROBE && self.last_probe.elapsed() >= NODE_SILENCE_PROBE {
                        self.last_probe = std::time::Instant::now();
                        warn!("No message from keryxd for {} s — probing", silent.as_secs());
                        self.client_get_block_template().await?;
                    }
                    // Timer tick: if a regular inference just finished, get a fresh template.
                    if (self.inference_rx.is_some() || self.publish_rx.is_some()) && self.poll_inference().await {
                        self.client_get_block_template().await?;
                    // If a challenge is in flight, keep pinging the node so the result is
                    // delivered as soon as the inference task completes. This is critical on
                    // sole-producer nodes where mining suspension stops NewBlockTemplate
                    // notifications and the response would otherwise never be sent.
                    } else if self.challenge_inference_rx.is_some() || (self.replay_rx.is_some() && self.poll_replay().await) {
                        self.client_get_block_template().await?;
                    }
                    if self.escrow_pubkey.is_some() && self.last_strike_poll.elapsed().as_secs() >= 60 {
                        self.last_strike_poll = std::time::Instant::now();
                        self.client_send(GetServiceStrikesRequestMessage {}).await?;
                    }
                    if self.node_awaited_openings && self.escrow_pubkey.is_some() && self.last_providers_poll.elapsed().as_secs() >= PROVIDERS_POLL_SECS {
                        self.last_providers_poll = std::time::Instant::now();
                        self.client_send(GetServiceProvidersRequestMessage { window_daa: 0 }).await?;
                    }
                    let unchecked = self.node_checks_requests
                        && self.ai_request_queue.iter().any(|(sid, ..)| {
                            !self.request_checks.contains_key(sid)
                                && self
                                    .request_queued_at
                                    .get(sid)
                                    .is_some_and(|q| q.elapsed().as_secs() < REQUEST_CHECK_WAIT_SECS)
                        });
                    let poll_due = self.last_awaited_poll.elapsed().as_secs()
                        >= if unchecked { UNCHECKED_POLL_SECS } else { AWAITED_POLL_SECS };
                    let poll_free =
                        !self.awaited_poll_pending || self.awaited_poll_sent.elapsed().as_secs() >= AWAITED_POLL_SECS;
                    if self.node_awaited_openings && poll_free && (poll_due || self.force_awaited_poll) {
                        self.force_awaited_poll = false;
                        self.last_awaited_poll = std::time::Instant::now();
                        self.poll_awaited_openings().await?;
                    }
                    self.release_persisted_responses().await?;
                    self.resend_early_responses().await?;
                    self.retry_pending_ai_responses().await?;
                    self.prune_traces();
                    self.retry_pending_openings().await?;
                    self.retry_chain_restores().await?;
                    self.retry_chain_blocks().await?;
                    self.try_start_replay(miner).await?;
                }
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }

    fn flush_escrow_state(&mut self) -> Result<(), Error> {
        self.trace_store.flush();
        self.escrow_watcher.as_mut().map_or(Ok(()), |watcher| watcher.flush_state().map_err(Into::into))
    }
}

impl KeryxdHandler {
    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        escrow_privkey: Option<String>,
        escrow_state_file: String,
        escrow_cert: Option<String>,
        chain_daa: Option<u64>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        let trace_store = keryx_miner::trace_store::TraceStore::new(&escrow_state_file);
        let (restored_traces, restored_openings) = Self::restore_traces(&trace_store);
        // Build EscrowWatcher from the resolved escrow privkey (derived or loaded from file).
        // The watcher also provides the pubkey to embed in coinbase extra_data.
        let (escrow_pubkey, escrow_watcher) = match escrow_privkey {
            Some(ref privkey) => {
                match crate::escrow::EscrowWatcher::new(privkey, &miner_address, escrow_state_file.into()) {
                    Ok(watcher) => {
                        let pk = watcher.pubkey_hex();
                        info!("Escrow active: pubkey={}", pk);
                        (Some(pk), Some(watcher))
                    }
                    Err(e) => {
                        log::error!("Failed to initialise EscrowWatcher: {} — escrow disabled", e);
                        (None, None)
                    }
                }
            }
            None => (None, None),
        };

        let service_identity = match crate::escrow::service_identity_hex(&miner_address) {
            Ok(id) => Some(id),
            Err(e) => {
                log::warn!("Cannot derive the service identity of the payout address: {}", e);
                None
            }
        };

        let endpoint: tonic::transport::Endpoint = address.try_into().map_err(Into::into)?;
        let channel = endpoint
            .connect_timeout(std::time::Duration::from_secs(10))
            .http2_keep_alive_interval(std::time::Duration::from_secs(15))
            .keep_alive_timeout(std::time::Duration::from_secs(20))
            .keep_alive_while_idle(true)
            .connect()
            .await?;
        let mut client = RpcClient::new(channel);
        // Outbound message channel to the node. ALL client->node messages share this:
        // mining (submit_block, GetBlockTemplate) AND inference traffic (per-block GetBlock,
        // escrow submit_transaction). With a capacity of 2 the inference traffic could fill the
        // buffer and block GetBlockTemplate, stalling template delivery → the GPU sits idle
        // between blocks. A large buffer keeps the mining requests from queuing behind inference traffic.
        let (send_channel, recv) = mpsc::channel(1024);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let pending_block_submissions = Arc::new(Mutex::new(VecDeque::new()));
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone(), Arc::clone(&pending_block_submissions));
        // Restored traces count as served by this instance.
        let mut served_requests = std::collections::HashMap::new();
        let mut served_requests_order = VecDeque::new();
        for (txid, trace) in &restored_traces {
            record_served_request(&mut served_requests, &mut served_requests_order, trace.request_hash, txid);
        }
        let mut handler = Box::new(Self {
            client,
            task_chain: ChainCache::new(),
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            block_channel,
            block_handle,
            ai_request_queue: VecDeque::new(),
            validation_queue: VecDeque::new(),
            ai_seen_prefixes: std::collections::HashSet::new(),
            ai_answered_by_me: std::collections::HashSet::new(),
            ai_request_txids: std::collections::HashMap::new(),
            ai_response_inflight: std::collections::HashMap::new(),
            backfill_cutoff_daa: None,
            backfill_queue: VecDeque::new(),
            backfill_pending: std::collections::HashSet::new(),
            backfill_visited: std::collections::HashSet::new(),
            inference_rx: None,
            pending_traces: restored_traces,
            chain_log: VecDeque::new(),
            trace_store,
            chain_restore_queue: VecDeque::new(),
            chain_restore_pending: std::collections::HashSet::new(),
            chain_restore_starts: VecDeque::new(),
            chain_restore_failures: std::collections::HashMap::new(),
            chain_restore_retry: Vec::new(),
            chain_block_retry: Vec::new(),
            persisting: Vec::new(),
            response_resend: Vec::new(),
            chain_restore_continuations: std::collections::HashSet::new(),
            replay_rx: None,
            opening_inflight: restored_openings,
            publish_rx: None,
            challenge_inference_rx: None,
            opoi_challenge_active: None,
            pending_block_submissions,
            last_known_daa: chain_daa.unwrap_or(0).max(LAST_KNOWN_DAA.load(Ordering::Relaxed)),
            ipfs_url,
            escrow_pubkey,
            escrow_watcher,
            escrow_cert,
            service_identity,
            last_strike_poll: std::time::Instant::now() - std::time::Duration::from_secs(55),
            node_awaited_openings: false,
            awaited_answered: false,
            last_awaited_poll: std::time::Instant::now(),
            awaited_polled: Vec::new(),
            awaited_requests_polled: Vec::new(),
            awaited_poll_pending: false,
            awaited_poll_sent: std::time::Instant::now(),
            force_awaited_poll: false,
            node_checks_requests: false,
            request_checks: std::collections::HashMap::new(),
            request_queued_at: std::collections::HashMap::new(),
            request_retries: std::collections::HashMap::new(),
            sibling_escrows: std::collections::HashMap::new(),
            last_providers_poll: std::time::Instant::now() - std::time::Duration::from_secs(PROVIDERS_POLL_SECS),
            last_message: std::time::Instant::now(),
            last_probe: std::time::Instant::now(),
            strike_status: None,
            stats: None,
            misses_seen: std::collections::HashSet::new(),
            served_requests,
            served_requests_order,
            strike_burns_reported: std::collections::HashSet::new(),
            ai_skipped_unready: std::collections::HashSet::new(),
        });
        handler.restore_parked();
        Ok(handler)
    }

    /// Takes over the request queue and in-flight work of the previous connection.
    fn restore_parked(&mut self) {
        let Some(parked) = PARKED.lock().unwrap_or_else(|p| p.into_inner()).take() else { return };
        // Keep the checks' reference point; restored requests wait for a fresh node check.
        self.node_checks_requests |= parked.node_checks_requests;
        self.force_awaited_poll = true;
        self.ai_answered_by_me.extend(parked.answered);
        self.ai_seen_prefixes.extend(parked.seen);
        self.request_retries.extend(parked.retries);
        for entry in parked.queue {
            self.ai_seen_prefixes.insert(entry.0.clone());
            let queued = parked.queued_at.get(&entry.0).copied().unwrap_or_else(std::time::Instant::now);
            self.request_queued_at.insert(entry.0.clone(), queued);
            self.ai_request_queue.push_back(entry);
        }
        if let Some(job) = parked.inference {
            info!("Inference: resuming request id={} across the reconnect", job.stable_id);
            self.ai_seen_prefixes.insert(job.stable_id.clone());
            self.inference_rx = Some(job);
        }
        if let Some(job) = parked.replay {
            info!("Trace: resuming the replay of response {} across the reconnect", job.response_txid);
            self.replay_rx = Some(job);
        }
        let answered = parked.publish.iter().map(|p| p.request_hash).chain(parked.responses.values().filter_map(|(tx, ..)| {
            hex::decode(&tx.payload).ok().and_then(|b| keryx_inference::AiResponsePayload::deserialize(&b)).map(|r| r.request_hash)
        }));
        for request_hash in answered.collect::<Vec<_>>() {
            let sid = hex::encode(&request_hash[..8]);
            self.ai_seen_prefixes.insert(sid.clone());
            self.ai_answered_by_me.insert(sid);
        }
        self.publish_rx = parked.publish;
        self.ai_response_inflight.extend(parked.responses);
        self.ai_request_txids.extend(parked.request_txids);
    }

    fn create_block_channel(
        send_channel: Sender<KaspadMessage>,
        pending_block_submissions: Arc<Mutex<VecDeque<(String, String)>>>,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        // KaspadMessage::submit_block(block)
        let (send, recv) = mpsc::channel::<BlockSeed>(1);
        (
            send,
            tokio::spawn(async move {
                ReceiverStream::new(recv)
                    .map(move |block_seed| match block_seed {
                        FullBlock { block, device_id } => {
                            let block_hash = block.block_hash().map(|hash| format!("{:x}", hash)).unwrap_or_default();
                            let mut pending = pending_block_submissions.lock().unwrap();
                            pending.push_back((block_hash, device_id));
                            KaspadMessage::submit_block(*block)
                        }
                        PartialBlock { .. } => unreachable!("All blocks sent here should have arrived from here"),
                    })
                    .map(Ok)
                    .forward(PollSender::new(send_channel))
                    .await
            }),
        )
    }

    /// Sends to the node; a channel still full after the silence limit fails like a closed one.
    async fn client_send(&self, msg: impl Into<KaspadMessage>) -> Result<(), SendError<KaspadMessage>> {
        self.send_channel.send_timeout(msg.into(), NODE_SILENCE_LIMIT).await.map_err(|e| SendError(e.into_inner()))
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<KaspadMessage>> {
        let pay_address = self.miner_address.clone();
        // Append a per-request random nonce so that parallel blocks at the same blue_score
        // get distinct coinbase payloads → distinct tx_ids (avoids DAG coinbase collisions).
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // Inference: run the deterministic fixed-point MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Embed escrow pubkey so the node routes 20% to the CSV-locked escrow output.
        let escrow_part = self.escrow_pubkey
            .as_deref()
            .map(|pk| format!("/escrow:{}", pk))
            .unwrap_or_default();
        // Delegation cert binding that escrow key to the payout address. From H6 the node rejects
        // a block whose coinbase carries no valid pair.
        let esig_part = self.escrow_cert
            .as_deref()
            .map(|cert| format!("/esig:{}", cert))
            .unwrap_or_default();
        // Announce loaded model capabilities so the node can enforce model_id matching.
        let cap_part = {
            let ids = keryx_miner::llm::loaded_model_ids();
            if ids.is_empty() {
                String::new()
            } else {
                let hex_ids: Vec<String> = ids.iter().map(hex::encode).collect();
                format!("/ai:cap:{}", hex_ids.join(","))
            }
        };
        let extra_data =
            format!("{}{}{}/{}/ai:v1:{}{}", EXTRA_DATA, escrow_part, esig_part, nonce_hex, opoi_tag, cap_part);
        // Harvest a pending challenge response if the inference task just finished.
        let inference_result = match self.challenge_inference_rx.take() {
            Some((challenge_str, mut rx)) => match rx.try_recv() {
                Ok(Some(text)) => {
                    // challenge_str = "model_id_hex:nonce_hex"
                    let mut parts = challenge_str.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("");
                    let nonce_hex_c  = parts.next().unwrap_or("");
                    info!("Inference: sending challenge response model={:.8}", model_id_hex);
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    // Response format: "model_id_hex:nonce_hex:result_text"
                    format!("{}:{}:{}", model_id_hex, nonce_hex_c, text)
                }
                Ok(None) => {
                    warn!("Inference: challenge inference failed — sending empty result, node will re-challenge");
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    String::new()
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    self.challenge_inference_rx = Some((challenge_str, rx));
                    String::new()
                }
                Err(_) => {
                    warn!("Inference: challenge inference task dropped — sending empty result");
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    String::new()
                }
            },
            None => String::new(),
        };
        self.client_send(GetBlockTemplateRequestMessage { pay_address, extra_data, inference_result }).await
    }

    /// Scans a slice of transactions for AiRequest payloads and pushes new
    /// entries into `ai_request_queue` (deduplication by payload hash prefix).
    ///
    /// Handles two formats:
    ///   - Subnetwork 0x03 + binary `AiRequestPayload` (future on-chain format)
    ///   - Any non-coinbase TX + `KRX:AI:1:` JSON prefix (web wallet format)
    fn scan_txs_for_ai_requests(&mut self, txs: &[crate::proto::RpcTransaction], block_daa: u64) {
        // Identity of a request follows the same gate as the node: transaction id past it, payload
        // digest before. Decided from the daa of the block the request is observed in, so both
        // sides classify a request the same way across the activation.
        let txid_identity = block_daa >= keryx_miner::pom::reward_routing_activation_daa();
        if let Some(mine) = self.escrow_watcher.as_ref().map(|w| w.pubkey_bytes()) {
            for sid in Self::own_answered_request_ids(txs, &mine) {
                if self.ai_answered_by_me.insert(sid.clone()) {
                    log::debug!("Inference: request id={} already answered by this miner on-chain", sid);
                }
                self.ai_request_queue.retain(|(queued, ..)| *queued != sid);
            }
            if self.ai_answered_by_me.len() > MAX_AI_SEEN_IDS {
                self.ai_answered_by_me.clear();
                self.ai_answered_by_me.shrink_to_fit();
            }
        }
        // Hard gate: if no models are ready, refuse to accept any AiRequest.
        // Prevents miners with missing/truncated model files from ever queuing inference work.
        let ready_ids = keryx_miner::llm::loaded_model_ids();
        if ready_ids.is_empty() {
            log::warn!("Inference: no models ready — skipping AiRequest scan (run miner with valid model files)");
            return;
        }
        log::debug!(
            "scan_ai: {} txs, subnetwork_ids: {:?}",
            txs.len(),
            txs.iter().map(|t| t.subnetwork_id.as_str()).collect::<Vec<_>>()
        );
        let escrow_secret = self.escrow_watcher.as_ref().map(|w| w.secret_bytes());
        // Past the private-inference gate every request is sealed; plaintext ones are not served.
        let private_era = block_daa >= keryx_miner::pom::private_inference_activation_daa();
        for tx in txs {
            // (raw, model_id, prompt, plaintext prompt length, max_tokens, inference_reward, private state)
            let extracted: Option<ExtractedRequest> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork).
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        let req = keryx_inference::AiRequestPayload::deserialize(&raw)?;
                        let (prompt, prompt_len, private) = if req.is_private() {
                            Self::open_private_request(&req, escrow_secret.as_ref())?
                        } else if private_era {
                            return None;
                        } else {
                            (String::from_utf8_lossy(&req.prompt).into_owned(), req.prompt.len() as u32, None)
                        };
                        Some((raw, req.model_id, prompt, prompt_len, req.max_tokens as usize, req.inference_reward, private))
                    })
                } else if !tx.inputs.is_empty() && !private_era {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            let prompt_len = prompt.len() as u32;
                            Some((raw, model_id, prompt, prompt_len, max_tokens, 0u64, None))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, prompt_len, max_tokens, inference_reward, private)) = extracted {
                let txid_hex = tx
                    .verbose_data
                    .as_ref()
                    .map(|v| v.transaction_id.clone())
                    .filter(|id| !id.is_empty())
                    .or_else(|| Self::compute_rpc_txid(tx));
                let request_hash: [u8; 32] = if txid_identity {
                    match txid_hex.as_deref().and_then(|h| hex::decode(h).ok()).and_then(|b| <[u8; 32]>::try_from(b).ok()) {
                        Some(id) => id,
                        None => {
                            log::warn!("Inference: cannot resolve the AiRequest transaction id — request skipped");
                            continue;
                        }
                    }
                } else {
                    blake2b_simd::blake2b(&raw).as_bytes()[..32].try_into().unwrap()
                };
                let stable_id = hex::encode(&request_hash[..8]);
                if !ready_ids.contains(&model_id) {
                    log::debug!("Inference: skipping AiRequest id={} — model not supported or files not ready", stable_id);
                    self.ai_skipped_unready.insert(stable_id);
                    if self.ai_skipped_unready.len() > MAX_AI_SEEN_IDS {
                        self.ai_skipped_unready.clear();
                        self.ai_skipped_unready.shrink_to_fit();
                    }
                    continue;
                }
                if self.ai_answered_by_me.contains(&stable_id) {
                    log::debug!("Inference: skipping AiRequest id={} — already answered by this miner", stable_id);
                    continue;
                }
                if !self.ai_seen_prefixes.contains(&stable_id) {
                    info!("Inference: queued AiRequest id={}", stable_id);
                    self.ai_seen_prefixes.insert(stable_id.clone());
                    self.request_queued_at.insert(stable_id.clone(), std::time::Instant::now());
                    self.force_awaited_poll = true;
                    self.ai_request_queue.push_back((stable_id.clone(), request_hash, model_id, prompt, prompt_len, max_tokens, private));
                    if self.ai_request_queue.len() > MAX_AI_QUEUE_SIZE {
                        self.drop_unawaited_requests();
                    }
                    while self.ai_request_queue.len() > MAX_AI_QUEUE_SIZE {
                        if let Some((sid, ..)) = self.ai_request_queue.pop_back() {
                            self.ai_seen_prefixes.remove(&sid);
                            self.request_queued_at.remove(&sid);
                        }
                    }
                    while self.ai_seen_prefixes.len() > MAX_AI_SEEN_IDS {
                        self.ai_seen_prefixes.clear();
                        self.ai_seen_prefixes.shrink_to_fit();
                        for (sid, ..) in &self.ai_request_queue {
                            self.ai_seen_prefixes.insert(sid.clone());
                        }
                    }
                }
                // Track txid for escrow claims. Prefer verbose_data.transaction_id when
                // present, fall back to computing the txid from the transaction fields —
                // verbose_data is not populated for non-coinbase transactions in block
                // template or block notifications, so without this fallback the escrow
                // outpoint is never tracked and the inference_reward is never claimed.
                if inference_reward > 0 {
                    if let Some(txid) = txid_hex {
                        if self.ai_request_txids.len() >= MAX_AI_SEEN_IDS {
                            self.ai_request_txids.clear();
                        }
                        self.ai_request_txids.insert(stable_id, (txid, inference_reward));
                    }
                }
            }
        }
    }

    /// Stable IDs of the requests answered in `txs` by the responder `mine`.
    fn own_answered_request_ids(txs: &[crate::proto::RpcTransaction], mine: &[u8; 32]) -> Vec<String> {
        txs.iter()
            .filter(|tx| tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX)
            .filter_map(|tx| hex::decode(&tx.payload).ok())
            .filter_map(|raw| keryx_inference::AiResponsePayload::deserialize(&raw))
            .filter(|resp| resp.responder.as_ref().is_some_and(|r| r.escrow_pubkey == *mine))
            .map(|resp| hex::encode(&resp.request_hash[..8]))
            .collect()
    }

    /// Opens a private request with the escrow key. `None` when the request is not for this
    /// miner (no escrow key, or not among the recipients); an addressed request that cannot be
    /// opened is still queued, to be answered with a plaintext error body.
    fn open_private_request(
        req: &keryx_inference::AiRequestPayload,
        escrow_secret: Option<&[u8; 32]>,
    ) -> Option<(String, u32, Option<PrivateRequest>)> {
        let Some(secret) = escrow_secret else {
            log::debug!("Inference: skipping private AiRequest — no escrow key configured");
            return None;
        };
        match keryx_inference::open_request(req, secret) {
            Ok(opened) => Some((
                decode_private_prompt(&opened.prompt),
                opened.prompt.len() as u32,
                Some(PrivateRequest::Opened(opened.response_key.clone())),
            )),
            Err(keryx_inference::PrivateError::NotARecipient) => {
                log::debug!("Inference: skipping private AiRequest — this escrow key is not a recipient");
                None
            }
            Err(e) => {
                warn!("Inference: private AiRequest names this miner but cannot be opened ({}) — answering with an error body", e);
                Some((String::new(), 0, Some(PrivateRequest::Unreadable)))
            }
        }
    }

    /// Compute the Kaspa transaction ID for a non-coinbase RpcTransaction.
    ///
    /// Mirrors keryx-node consensus/core/src/hashing/tx.rs `id()` with
    /// EXCLUDE_SIGNATURE_SCRIPT | EXCLUDE_MASS_COMMIT flags (standard for non-coinbase txs).
    ///
    /// Serialization: blake2b-256 keyed "TransactionID" over:
    ///   version(u16 LE) | inputs_count(u64 LE) | inputs... | outputs_count(u64 LE) | outputs...
    ///   | lock_time(u64 LE) | subnetwork_id(20B) | gas(u64 LE) | payload_len(u64 LE) | payload
    ///
    /// For each input (sig script excluded): txid(32B) | index(u32 LE) | 0u64(empty var_bytes) | seq(u64 LE)
    /// For each output: amount(u64 LE) | spk_version(u16 LE) | script_len(u64 LE) | script
    fn compute_rpc_txid(tx: &crate::proto::RpcTransaction) -> Option<String> {
        const KEY: &[u8] = b"TransactionID";
        let mut h = blake2b_simd::Params::new().hash_length(32).key(KEY).to_state();

        h.update(&(tx.version as u16).to_le_bytes());
        h.update(&(tx.inputs.len() as u64).to_le_bytes());
        for input in &tx.inputs {
            let prev = input.previous_outpoint.as_ref()?;
            let txid_bytes = hex::decode(&prev.transaction_id).ok()?;
            if txid_bytes.len() != 32 {
                return None;
            }
            h.update(&txid_bytes);
            h.update(&prev.index.to_le_bytes());
            h.update(&0u64.to_le_bytes()); // write_var_bytes(&[]) — empty sig script
            h.update(&input.sequence.to_le_bytes());
        }

        h.update(&(tx.outputs.len() as u64).to_le_bytes());
        for output in &tx.outputs {
            h.update(&output.amount.to_le_bytes());
            let spk = output.script_public_key.as_ref()?;
            h.update(&(spk.version as u16).to_le_bytes());
            let script = hex::decode(&spk.script_public_key).ok()?;
            h.update(&(script.len() as u64).to_le_bytes());
            h.update(&script);
        }

        h.update(&tx.lock_time.to_le_bytes());
        let subnet = hex::decode(&tx.subnetwork_id).ok()?;
        if subnet.len() != 20 {
            return None;
        }
        h.update(&subnet);
        h.update(&tx.gas.to_le_bytes());
        let payload = hex::decode(&tx.payload).ok()?;
        h.update(&(payload.len() as u64).to_le_bytes());
        h.update(&payload);

        Some(hex::encode(h.finalize().as_bytes()))
    }

    /// Parses a `KRX:AI:1:` JSON payload, returning `(model_name, prompt, max_tokens)`.
    fn parse_krx_ai_payload(raw: &[u8]) -> Option<(String, String, usize)> {
        const PREFIX: &[u8] = b"KRX:AI:1:";
        if raw.len() <= PREFIX.len() || !raw.starts_with(PREFIX) {
            return None;
        }
        let v: serde_json::Value = serde_json::from_slice(&raw[PREFIX.len()..]).ok()?;
        let model = v["m"].as_str().unwrap_or("glm-4-9b-0414").to_string();
        let prompt = v["p"].as_str()?.to_string();
        let max_tokens = v["n"].as_u64().unwrap_or(128).min(MAX_KRX_AI_TOKENS) as usize;
        Some((model, prompt, max_tokens))
    }

    /// Starts LLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    /// Marks whether mining is paused by inference work (inference in flight, or model files not
    /// ready) rather than by the node. Set at every pause decision, so no exit path can leave it
    /// stale. Also suppresses the GPU stall warnings, which a deliberate pause would trip.
    fn set_opoi_pause(&self, paused: bool) {
        keryx_miner::pom_gpu::set_inference_paused(paused);
        if let Some(flag) = &self.opoi_challenge_active {
            flag.store(paused, Ordering::Relaxed);
        }
    }

    /// Whether the GPU is taken by inference work: a request, a challenge or a replay.
    fn inference_busy(&self) -> bool {
        self.inference_rx.is_some() || self.challenge_inference_rx.is_some() || self.replay_rx.is_some()
    }

    /// Estimated duration of the next queued request, in DAA.
    /// Room the node's span test takes for the next request's audit behind the open ones: its
    /// chained work, its attention phases and the network backlog at its cap.
    fn next_request_span_cost(&self) -> u64 {
        let Some((_, _, model_id, _, prompt_len, max_tokens, _)) =
            self.next_request().and_then(|i| self.ai_request_queue.get(i))
        else {
            return 0;
        };
        let tier = keryx_miner::models::pom_tier_index(model_id, self.last_known_daa).unwrap_or(u8::MAX);
        let (n_prompt, max_tokens) = (*prompt_len as u32, *max_tokens as u32);
        let work = keryx_miner::trace::OPEN_AUDIT_WORK_PASSES * keryx_miner::trace::service_work_daa(tier, n_prompt, max_tokens);
        let backlog = keryx_miner::trace::OPENING_BACKLOG_CAP_DAA;
        let phase = keryx_miner::trace::TRACE_SEED_DELAY_DAA
            + keryx_miner::trace::attn_phase_window_daa(tier, max_tokens, n_prompt, n_prompt.saturating_add(max_tokens))
            + backlog;
        work + backlog + AUDIT_PHASES_RESERVED * phase
    }

    fn next_request_work(&self) -> u64 {
        self.next_request().and_then(|i| self.ai_request_queue.get(i)).map_or(0, |q| self.request_work_daa(q))
    }

    /// Estimated DAA to serve `q`: prompt, then generation.
    fn request_work_daa(&self, q: &QueuedRequest) -> u64 {
        let (_, _, model_id, _, prompt_len, max_tokens, _) = q;
        let tier = keryx_miner::models::pom_tier_index(model_id, self.last_known_daa).unwrap_or(u8::MAX);
        match speeds::generation_daa(model_id, *max_tokens as u32) {
            Some(gen) => gen + keryx_miner::trace::work_daa(tier, *prompt_len, 0, 1),
            None => keryx_miner::trace::work_daa(tier, *prompt_len, *max_tokens as u32, 1),
        }
    }

    /// Index of the request the work estimates use: the one this rig is first in line for, or
    /// the first startable one.
    fn next_request(&self) -> Option<usize> {
        self.own_request()
    }

    /// Index of the first queued request that may start now.
    fn startable_request(&self) -> Option<usize> {
        let now = std::time::Instant::now();
        (0..self.ai_request_queue.len()).find(|&i| self.request_startable(i, now))
    }

    /// Whether an audit replay must start before a request of `next_work` DAA: its deadline
    /// leaves no more than both durations.
    fn urgent_replay(&self, next_work: u64) -> bool {
        let now = self.last_known_daa;
        let mut jobs: Vec<(u64, u64, bool)> = self
            .pending_traces
            .values()
            .filter_map(|p| {
                let r = &p.trace.record;
                // Accepted under the audit-all rule: certain to be audited once drawn, due no
                // earlier than its submission plus the seed delay and the window.
                if p.draw.is_none() && p.accepted_hash.is_some() && keryx_miner::trace::audit_all_active(p.submitted_daa) {
                    let n_gen = r.n_pos.saturating_sub(r.n_prompt);
                    let work = speeds::replay_daa(&p.model_id, false, p.gen_secs, n_gen).unwrap_or_else(|| keryx_miner::trace::work_daa(p.tier, r.n_prompt, n_gen, 3));
                    let due = p.submitted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA + keryx_miner::trace::opening_window_daa(p.tier, p.max_tokens);
                    return Some((due, work, false));
                }
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let n_gen = r.n_pos.saturating_sub(r.n_prompt);
                let work = |phase: bool| {
                    speeds::replay_daa(&p.model_id, phase, p.gen_secs, n_gen)
                        .unwrap_or_else(|| keryx_miner::trace::work_daa(p.tier, r.n_prompt, n_gen, if phase { 2 } else { 3 }))
                };
                if d.replay_ready() {
                    return Some((d.deadline_daa, work(false), true));
                }
                p.attn.as_ref().filter(|a| a.ready()).map(|a| (a.deadline_daa, work(true), true))
            })
            .filter(|(deadline, _, _)| *deadline > now)
            .collect();
        jobs.sort_unstable();
        // Earliest deadline first: every replay must still fit after the request and the ones
        // before it. Undrawn audits take their share but cannot start yet, so only a replay that
        // can start now, at or before the first overflow, makes it urgent.
        // The next answer's audit chains behind the open ones of this key; past the node's span
        // it is struck at once, so the open ones go first.
        let inflight: std::collections::HashSet<&str> = self.opening_inflight.values().map(|o| o.response_txid.as_str()).collect();
        let chain_end = self
            .pending_traces
            .iter()
            .filter_map(|(t, p)| {
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let unfinished = !d.opened || inflight.contains(t.as_str()) || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                unfinished.then_some(d.deadline_daa)
            })
            .max();
        if jobs.iter().any(|(_, _, ready)| *ready) && chain_end.is_some_and(|end| end.saturating_add(self.next_request_span_cost()) > now.saturating_add(next_work).saturating_add(AUDIT_SPAN_DAA)) {
            return true;
        }
        let mut at = now.saturating_add(REPLAY_URGENT_BASE_DAA).saturating_add(next_work);
        let mut startable = false;
        jobs.into_iter().any(|(deadline, work, ready)| {
            at = at.saturating_add(work);
            startable |= ready;
            at >= deadline && startable
        })
    }

    /// Keeps, per tier, the escrow keys sharing this rig's identity.
    fn apply_service_providers(&mut self, resp: &crate::proto::GetServiceProvidersResponseMessage) {
        let Some(own) = self.escrow_pubkey.as_deref().map(str::to_ascii_lowercase) else { return };
        let mut siblings: std::collections::HashMap<u32, Vec<[u8; 32]>> = std::collections::HashMap::new();
        for mine in resp.providers.iter().filter(|p| p.escrow_pubkey.eq_ignore_ascii_case(&own)) {
            let keys = siblings.entry(mine.tier).or_default();
            for p in resp.providers.iter().filter(|p| p.tier == mine.tier && p.identity == mine.identity) {
                if let Some(k) = hash32(&p.escrow_pubkey) {
                    keys.push(k);
                }
            }
            keys.sort_unstable();
            keys.dedup();
        }
        self.sibling_escrows = siblings;
    }

    /// Drops the queued requests the node no longer awaits from this rig's identity.
    fn drop_unawaited_requests(&mut self) {
        let dropped: std::collections::HashSet<String> = self
            .ai_request_queue
            .iter()
            .filter(|(sid, ..)| self.request_closed(sid))
            .map(|(sid, ..)| sid.clone())
            .collect();
        if !dropped.is_empty() {
            self.ai_request_queue.retain(|(sid, ..)| !dropped.contains(sid));
        }
        for sid in dropped {
            info!("Inference: request id={} no longer awaited from this address — skipped", sid);
        }
        let queued: std::collections::HashSet<&String> = self.ai_request_queue.iter().map(|(sid, ..)| sid).collect();
        self.request_checks.retain(|sid, _| queued.contains(sid));
        self.request_queued_at.retain(|sid, _| queued.contains(sid));
    }

    /// Whether the node said, since `sid` was queued, that this rig's identity is no longer
    /// awaited for it, or its answer window is already over.
    fn request_closed(&self, sid: &str) -> bool {
        let queued = self.request_queued_at.get(sid);
        self.request_checks.get(sid).is_some_and(|(awaited, window_end, at)| {
            (!awaited && queued.is_none_or(|q| at >= q)) || (*awaited && *window_end > 0 && self.last_known_daa > *window_end)
        })
    }

    /// Rank of this rig among the rigs of its address for the queued request at `index`.
    fn request_rank(&self, index: usize) -> usize {
        let Some((_, request_hash, model_id, ..)) = self.ai_request_queue.get(index) else { return 0 };
        let own = self.escrow_pubkey.as_deref().and_then(hash32);
        keryx_miner::models::pom_tier_index(model_id, self.last_known_daa)
            .and_then(|t| self.sibling_escrows.get(&(t as u32)))
            .zip(own)
            .map_or(0, |(siblings, own)| sibling_rank(request_hash, &own, siblings))
    }

    /// Index of the first queued request this rig is first in line for, or may start now.
    fn own_request(&self) -> Option<usize> {
        let now = std::time::Instant::now();
        (0..self.ai_request_queue.len()).find(|&i| {
            self.ai_request_queue.get(i).is_some_and(|(sid, ..)| !self.request_closed(sid) && !self.retry_waiting(sid, now))
                && (self.request_rank(i) == 0 || self.request_startable(i, now))
        })
    }

    /// Whether the request `sid` waits out its delay after a transient failure.
    fn retry_waiting(&self, sid: &str, now: std::time::Instant) -> bool {
        self.request_retries.get(sid).is_some_and(|(_, at)| *at > now)
    }

    /// Whether the queued request at `index` may start now: at once for the first rig of its
    /// address, after the wait of its rank and a node check for the others.
    fn request_startable(&self, index: usize, now: std::time::Instant) -> bool {
        if self.ai_request_queue.get(index).is_some_and(|(sid, ..)| self.retry_waiting(sid, now)) {
            return false;
        }
        if !self.node_checks_requests {
            return true;
        }
        let Some((sid, _, model_id, _, _, max_tokens, _)) = self.ai_request_queue.get(index) else { return false };
        let queued = self.request_queued_at.get(sid).copied().unwrap_or(now);
        let gen_daa = speeds::generation_daa(model_id, *max_tokens as u32);
        let rank = self.request_rank(index);
        let delay = std::time::Duration::from_secs(rank as u64 * (gen_daa.map_or(60, |d| d / 10) + SIBLING_MARGIN_SECS));
        if self.request_closed(sid) {
            return false;
        }
        if rank == 0 {
            return true;
        }
        let check = self.request_checks.get(sid);
        let window_end = check.map_or(0, |c| c.1);
        let work = self.ai_request_queue.get(index).map_or(0, |q| self.request_work_daa(q));
        let deadline_near = window_end > 0 && self.last_known_daa + work + WINDOW_SAFETY_DAA >= window_end;
        let waited = now.saturating_duration_since(queued);
        if waited < delay && !deadline_near {
            return false;
        }
        deadline_near
            || check.is_some_and(|(awaited, _, at)| *awaited && *at >= queued + delay)
            || waited >= delay + std::time::Duration::from_secs(REQUEST_CHECK_WAIT_SECS)
    }

    /// Puts a request back after a transient failure, to start again after a growing delay, at
    /// most `MAX_REQUEST_RETRIES` times; the node's window still bounds it.
    fn retry_request(&mut self, entry: QueuedRequest) {
        let sid = entry.0.clone();
        let tries = self.request_retries.get(&sid).map_or(0, |(n, _)| *n) + 1;
        if tries > MAX_REQUEST_RETRIES {
            warn!("Inference: request id={} still failing after {} tries — dropped", sid, MAX_REQUEST_RETRIES);
            self.request_retries.remove(&sid);
            return;
        }
        if self.ai_request_queue.len() >= MAX_AI_QUEUE_SIZE {
            warn!("Inference: request id={} not retried — queue full", sid);
            self.request_retries.remove(&sid);
            return;
        }
        let delay = REQUEST_RETRY_BASE.saturating_mul(1 << (tries - 1));
        info!("Inference: request id={} retried in {} s (try {} of {})", sid, delay.as_secs(), tries, MAX_REQUEST_RETRIES);
        self.request_retries.insert(sid.clone(), (tries, std::time::Instant::now() + delay));
        self.request_queued_at.insert(sid, std::time::Instant::now());
        self.ai_request_queue.push_back(entry);
    }

    fn try_start_inference(&mut self) {
        // Retry state lives only as long as its request is queued or running.
        let running = self.inference_rx.as_ref().and_then(|j| j.retry.as_ref()).map(|(sid, ..)| sid.clone());
        let queue = &self.ai_request_queue;
        self.request_retries.retain(|sid, _| running.as_ref() == Some(sid) || queue.iter().any(|(q, ..)| q == sid));
        if self.inference_busy() || keryx_miner::llm::probe_in_flight() || self.urgent_replay(self.next_request_work()) {
            return;
        }
        while self.ai_request_queue.front().is_some_and(|(sid, ..)| self.ai_answered_by_me.contains(sid)) {
            if let Some((sid, ..)) = self.ai_request_queue.pop_front() {
                info!("Inference: request id={} already answered by this miner — skipped", sid);
            }
        }
        self.drop_unawaited_requests();
        let Some(index) = self.startable_request() else { return };
        if let Some(entry) = self.ai_request_queue.remove(index) {
            let retry = Some(entry.clone());
            let (stable_id, request_hash, model_id, prompt, prompt_len, max_tokens, private) = entry;
            self.request_queued_at.remove(&stable_id);
            self.request_checks.remove(&stable_id);
            let (tx_done, rx_done) = oneshot::channel::<(Option<keryx_miner::llm::Generated>, f64, bool)>();
            let traced = self.last_known_daa >= keryx_miner::pom::verifiable_inference_activation_daa();
            let tier = traced.then(|| keryx_miner::models::pom_tier_index(&model_id, self.last_known_daa)).flatten();
            let chain = keryx_miner::trace::chain_active(self.last_known_daa);
            let attention = chain && keryx_miner::trace::attention_active(self.last_known_daa);
            let unreadable = matches!(private, Some(PrivateRequest::Unreadable));
            let job = InferenceJob {
                stable_id: stable_id.clone(),
                request_hash,
                private,
                model_id,
                tier,
                max_tokens: max_tokens as u32,
                rx: rx_done,
                retry,
            };
            if unreadable {
                // No inference to run: the answer is the fixed error text.
                let _ = tx_done.send((Some(keryx_miner::llm::Generated { text: UNREADABLE_PRIVATE_REQUEST_ANSWER.to_string(), trace: None }), 0.0, false));
                self.inference_rx = Some(job);
                return;
            }
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::llm::is_model_ready(&model_id) {
                log::warn!("Inference: model not ready for id={}", stable_id);
                if let Some(entry) = job.retry {
                    self.retry_request(entry);
                }
                return;
            }
            info!("Inference: spawning LLM inference (max_tokens={}{})", max_tokens, if tier.is_some() { ", traced" } else { "" });
            tokio::task::spawn_blocking(move || {
                let started = std::time::Instant::now();
                let result = match tier {
                    Some(tier) => keryx_miner::llm::load_and_run_traced_inference(&model_id, &prompt, max_tokens, tier, prompt_len, &request_hash, chain, attention),
                    None => keryx_miner::llm::load_and_run_inference(&model_id, &prompt, max_tokens)
                        .map(|text| keryx_miner::llm::Generated { text, trace: None }),
                };
                let transient = result.is_none() && keryx_miner::llm::last_failure_transient();
                if result.is_none() {
                    log::warn!("Inference: inference returned no result for id={}{}", stable_id, if transient { " (transient)" } else { " — AiResponse will be skipped" });
                }
                let _ = tx_done.send((result, started.elapsed().as_secs_f64(), transient));
            });
            self.inference_rx = Some(job);
        }
    }

    /// Records the seed block of the block task window of a template, once per window and parent set.
    async fn resolve_task_seed(&mut self, header: &crate::proto::RpcBlockHeader) {
        use keryx_verifiable_inference::block_task::{window, window_start};
        if !keryx_miner::block_task::active(header.daa_score) {
            return;
        }
        let Some(w) = window(header.daa_score) else { return };
        let Some(parents) = header.parents.first().map(|p| p.parent_hashes.clone()) else { return };
        let key = keryx_miner::block_task::parents_key(&parents);
        if keryx_miner::block_task::seed(w, &key).is_some() {
            return;
        }
        if self.task_chain.len() > 4096 {
            self.task_chain.clear();
        }
        let lookup = task_window_seed(&self.client, &parents, window_start(w), &mut self.task_chain);
        match tokio::time::timeout(std::time::Duration::from_secs(5), lookup).await {
            Ok(Some(seed)) => {
                keryx_miner::block_task::set_seed(w, &key, seed);
                keryx_miner::pom_gpu::log_throttled(u32::MAX, "seed", || info!("Block task: seed of window {} resolved", w));
            }
            Ok(None) => warn!("Block task: no seed block found for window {}", w),
            Err(_) => warn!("Block task: seed lookup for window {} timed out", w),
        }
    }

    /// Starts the replay of the audited response closest to its deadline, when the GPU is free.
    /// A response whose opening was already built for its current seed block is submitted
    /// without a replay.
    async fn try_start_replay(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        let ready: Vec<(String, String, Vec<u8>)> = self
            .pending_traces
            .iter_mut()
            .filter_map(|(txid, p)| {
                let d = p.draw.as_mut().filter(|d| d.audited && !d.opened)?;
                let body = p.built.iter().find(|(h, _)| *h == d.seed_hash)?.1.clone();
                d.opened = true;
                Some((txid.clone(), d.seed_hash.clone(), body))
            })
            .collect();
        for (txid, seed_hash, body) in ready {
            info!("Trace: response {} audited at chain block {} — opening already built", txid, seed_hash);
            self.submit_opening(&txid, &seed_hash, body).await;
        }
        if self.inference_busy() || self.publish_rx.is_some() || (self.own_request().is_some() && !self.urgent_replay(self.next_request_work())) || keryx_miner::llm::probe_in_flight() {
            return Ok(());
        }
        let now = self.last_known_daa;
        let released = |p: &PendingTrace, deadline: u64| self.node_awaited_openings && deadline <= now && p.node_awaits != Some(true);
        let first = self.pending_traces.iter().filter_map(|(txid, p)| {
            let d = p.draw.as_ref().filter(|d| d.replay_ready())?;
            (!released(p, d.deadline_daa)).then(|| (d.deadline_daa, txid.clone(), None))
        });
        let phases = self.pending_traces.iter().filter_map(|(txid, p)| {
            p.draw.as_ref().filter(|d| d.audited)?;
            let a = p.attn.as_ref().filter(|a| a.ready())?;
            (!released(p, a.deadline_daa)).then(|| (a.deadline_daa, txid.clone(), Some(a.path.clone())))
        });
        // Deadlines still ahead first, earliest first; past ones only when the node still awaits
        // them (or cannot say) and no request waits.
        let Some((deadline, txid, path)) = first.chain(phases).min_by_key(|(deadline, _, _)| (*deadline <= now, *deadline)) else {
            return Ok(());
        };
        if deadline <= now && self.own_request().is_some() {
            return Ok(());
        }
        if let Some(path) = path {
            let p = self.pending_traces.get_mut(&txid).unwrap();
            let seed_hash = p.draw.as_ref().map(|d| d.seed_hash.clone()).unwrap_or_default();
            if let Some((_, _, body)) = p.built_phase.take_if(|(s, pp, _)| *s == seed_hash && *pp == path) {
                info!("Trace: response {} — attention phase {} already built", txid, path.len());
                if let Some(a) = p.attn.as_mut() {
                    a.opened = true;
                }
                self.on_attention_phase_built(&txid, &seed_hash, path, Some(body)).await;
                return Ok(());
            }
            return self.start_attention_phase(miner, txid, path).await;
        }
        let Some(txid_bytes) = hash32(&txid) else {
            self.pending_traces.remove(&txid);
            self.trace_store.remove(&txid);
            return Ok(());
        };
        let p = self.pending_traces.get_mut(&txid).unwrap();
        let d = p.draw.as_mut().unwrap();
        d.opened = true;
        let (seed, seed_hash, accepted_daa) = (d.seed, d.seed_hash.clone(), d.accepted_daa);
        let (model_id, tier, trace, request_hash) = (p.model_id, p.tier, Arc::clone(&p.trace), p.request_hash);
        info!("Trace: response {} audited at chain block {} — replaying {} points", txid, seed_hash, keryx_miner::trace::POINTS);
        self.set_opoi_pause(true);
        miner.process_block(None).await?;
        let (tx_done, rx_done) = oneshot::channel::<Option<Vec<u8>>>();
        tokio::task::spawn_blocking(move || {
            let body = keryx_miner::llm::open_trace(&model_id, tier, &trace, &seed, txid_bytes, &request_hash, accepted_daa);
            let _ = tx_done.send(body);
        });
        self.replay_rx = Some(ReplayJob { response_txid: txid, seed_hash, path: None, started: std::time::Instant::now(), rx: rx_done });
        Ok(())
    }

    /// Starts the replay that builds the attention phase of `path` of response `txid`.
    async fn start_attention_phase(&mut self, miner: &mut MinerManager, txid: String, path: Vec<u64>) -> Result<(), Error> {
        let Some(txid_bytes) = hash32(&txid) else { return Ok(()) };
        let Some(p) = self.pending_traces.get_mut(&txid) else { return Ok(()) };
        let (Some(d), Some(a)) = (p.draw.as_ref(), p.attn.as_mut()) else { return Ok(()) };
        a.opened = true;
        let (seed, seed_hash, prev) = (d.seed, d.seed_hash.clone(), a.last_partials());
        let (model_id, tier, trace) = (p.model_id, p.tier, Arc::clone(&p.trace));
        info!("Trace: response {} — replaying attention phase {}", txid, path.len());
        self.set_opoi_pause(true);
        miner.process_block(None).await?;
        let (tx_done, rx_done) = oneshot::channel::<Option<Vec<u8>>>();
        let job_path = path.clone();
        tokio::task::spawn_blocking(move || {
            let body = keryx_miner::llm::open_attention_phase(&model_id, tier, &trace, &seed, txid_bytes, &job_path, &prev);
            let _ = tx_done.send(body);
        });
        self.replay_rx = Some(ReplayJob { response_txid: txid, seed_hash, path: Some(path), started: std::time::Instant::now(), rx: rx_done });
        Ok(())
    }

    /// Polls the in-flight replay; a finished body is signed and submitted as the opening.
    /// Returns `true` when the replay just ended.
    async fn poll_replay(&mut self) -> bool {
        let Some(mut job) = self.replay_rx.take() else { return false };
        let body = match job.rx.try_recv() {
            Ok(body) => body,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.replay_rx = Some(job);
                return false;
            }
            Err(_) => None,
        };
        if body.is_some() {
            if let Some(p) = self.pending_traces.get(&job.response_txid) {
                if let Some(base) = p.gen_secs {
                    speeds::note_replay(&p.model_id, job.path.is_some(), job.started.elapsed().as_secs_f64() / base.max(0.001));
                }
            }
        }
        if let Some(path) = job.path {
            self.on_attention_phase_built(&job.response_txid, &job.seed_hash, path, body).await;
            return true;
        }
        let Some(body) = body else {
            let retry = self
                .pending_traces
                .get_mut(&job.response_txid)
                .and_then(|p| p.draw.as_mut())
                .filter(|d| d.seed_hash == job.seed_hash)
                .map(|d| {
                    d.replay_failures += 1;
                    d.opened = false;
                    d.retry_at = Some(replay_backoff(d.replay_failures));
                });
            if retry.is_some() {
                warn!("Trace: no opening built for response {} — retrying later", job.response_txid);
            } else {
                warn!("Trace: no opening built for response {}", job.response_txid);
            }
            return true;
        };
        let Some(p) = self.pending_traces.get_mut(&job.response_txid) else { return true };
        p.built.retain(|(h, _)| *h != job.seed_hash);
        if p.built.len() >= BUILT_OPENINGS_KEPT {
            p.built.remove(0);
        }
        p.built.push((job.seed_hash.clone(), body.clone()));
        match p.draw.as_mut().filter(|d| d.seed_hash == job.seed_hash) {
            Some(d) => d.opened = true,
            None => {
                info!("Trace: seed block of response {} left the chain during the replay — opening kept for its return", job.response_txid);
                return true;
            }
        }
        self.submit_opening(&job.response_txid, &job.seed_hash, body).await;
        true
    }

    /// Signs and submits the opening `body` of response `response_txid`, built for `seed_hash`,
    /// the seed block of its current draw: one transaction per piece it bundles.
    async fn submit_opening(&mut self, response_txid: &str, seed_hash: &str, body: Vec<u8>) {
        let Some(d) = self.pending_traces.get(response_txid).and_then(|p| p.draw.as_ref()) else { return };
        if d.seed_hash != seed_hash {
            info!("Trace: opening of response {} built for a seed the node no longer awaits — dropped", response_txid);
            return;
        }
        let (deadline_daa, seed_daa) = (d.deadline_daa, d.seed_daa);
        let Some(pieces) = keryx_miner::trace_seq::unbundle(&body) else {
            warn!("Trace: malformed opening bundle for response {}", response_txid);
            return;
        };
        let mut txs = Vec::with_capacity(pieces.len());
        let mut phased = None;
        for piece in pieces {
            let Some(tx) = self.opening_tx(response_txid, seed_hash, seed_daa, &piece) else { return };
            if keryx_verifiable_inference::opening::AttnOpening::decode(&piece).is_some_and(|o| !o.partials.is_empty()) {
                phased = Some((piece.clone(), txs.len()));
            }
            txs.push((piece.len(), tx));
        }
        if let Some((body, i)) = phased {
            let tx = &txs[i].1;
            if let (Some(txid), Some(p)) = (Self::compute_rpc_txid(tx), self.pending_traces.get_mut(response_txid)) {
                p.attn = Some(AttnPhases::new(body, txid, tx.payload.clone(), deadline_daa));
            }
        }
        for (len, rpc_tx) in &txs {
            if let Some(txid) = Self::compute_rpc_txid(rpc_tx) {
                info!("Trace: submitting the opening of response {} ({} B)", response_txid, len);
                self.opening_inflight.insert(
                    txid,
                    OpeningInflight {
                        tx: rpc_tx.clone(),
                        response_txid: response_txid.to_string(),
                        deadline_daa,
                        attempts: 1,
                        last: std::time::Instant::now(),
                    },
                );
            }
        }
        if let Some(p) = self.pending_traces.get_mut(response_txid) {
            p.opening = Some(keryx_miner::trace_store::StoredOpening {
                seed_hash: seed_hash.to_string(),
                deadline_daa,
                payload: txs[0].1.payload.clone(),
                extra: txs[1..].iter().map(|(_, t)| t.payload.clone()).collect(),
            });
        }
        self.persist_trace(response_txid);
        for (i, (_, rpc_tx)) in txs.iter().enumerate() {
            if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx.clone())).await {
                warn!("Trace: failed to send the opening tx: {}", e);
                for (_, unsent) in &txs[i..] {
                    if let Some(o) = Self::compute_rpc_txid(unsent).and_then(|t| self.opening_inflight.get_mut(&t)) {
                        o.attempts = 0;
                    }
                }
                break;
            }
        }
    }

    /// The signed opening transaction of `piece`, naming its seed block once that rule is active.
    fn opening_tx(&self, response_txid: &str, seed_hash: &str, seed_daa: u64, piece: &[u8]) -> Option<crate::proto::RpcTransaction> {
        let piece = if self.last_known_daa.max(seed_daa) >= keryx_miner::pom::trace_seed_bind_activation_daa() {
            let Some(b) = hash32(seed_hash).and_then(|h| keryx_verifiable_inference::opening::seeded(piece, &h)) else {
                warn!("Trace: cannot name seed block {} in the opening of response {}", seed_hash, response_txid);
                return None;
            };
            b
        } else {
            piece.to_vec()
        };
        let payload = self.escrow_watcher.as_ref().map(|w| w.sign_opening(&piece))?;
        Some(crate::proto::RpcTransaction {
            version: 0,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
            gas: 0,
            payload: hex::encode(payload),
            mass: 0,
            verbose_data: None,
        })
    }

    /// Submits the attention phase of `path` of response `response_txid` once built, if its path
    /// is still the drawn one.
    async fn on_attention_phase_built(&mut self, response_txid: &str, seed_hash: &str, path: Vec<u64>, body: Option<Vec<u8>>) {
        let Some(p) = self.pending_traces.get_mut(response_txid) else { return };
        let current = p.draw.as_ref().filter(|d| d.seed_hash == seed_hash).map(|d| d.seed_daa).zip(p.attn.as_mut().filter(|a| a.path == path && a.bodies.len() == path.len()));
        let Some((seed_daa, a)) = current else {
            if let Some(body) = body {
                p.built_phase = Some((seed_hash.to_string(), path, body));
            }
            return;
        };
        let Some(body) = body else {
            a.failures += 1;
            a.opened = false;
            a.retry_at = Some(replay_backoff(a.failures));
            warn!("Trace: attention phase {} of response {} not built — retrying later", path.len(), response_txid);
            return;
        };
        let deadline_daa = a.deadline_daa;
        let Some(tx) = self.opening_tx(response_txid, seed_hash, seed_daa, &body) else { return };
        let Some(txid) = Self::compute_rpc_txid(&tx) else { return };
        info!("Trace: submitting attention phase {} of response {} ({} B)", path.len(), response_txid, body.len());
        let a = self.pending_traces.get_mut(response_txid).and_then(|p| p.attn.as_mut()).unwrap();
        a.bodies.push(body);
        a.txids.push(txid.clone());
        a.payloads.push(tx.payload.clone());
        a.accepted.push(a.late.remove(&txid));
        if let Some(Some(block)) = a.accepted.last() {
            info!("Trace: attention phase {} of response {} was already accepted by chain block {}", path.len(), response_txid, block);
        }
        self.opening_inflight.insert(
            txid,
            OpeningInflight { tx: tx.clone(), response_txid: response_txid.to_string(), deadline_daa, attempts: 1, last: std::time::Instant::now() },
        );
        self.persist_trace(response_txid);
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
            warn!("Trace: failed to send the attention phase tx: {}", e);
        }
    }

    /// Resubmits in-flight openings the mempool has not accepted yet.
    async fn retry_pending_openings(&mut self) -> Result<(), Error> {
        if self.opening_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let daa = self.last_known_daa;
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.opening_inflight.retain(|txid, o| {
            if daa > o.deadline_daa + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA {
                warn!("Trace: opening {} of response {} missed its window", txid, o.response_txid);
                return false;
            }
            let wait = if o.attempts < OPENING_FAST_RETRIES { AI_RESPONSE_RETRY_SECS } else { OPENING_SLOW_RETRY_SECS };
            if now.duration_since(o.last).as_secs() < wait {
                return true;
            }
            o.attempts += 1;
            o.last = now;
            resend.push(o.tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// Records a chain block's DAA score once its GetBlock answer arrives.
    /// Writes the on-disk copy of a pending trace.
    fn persist_trace(&self, txid: &str) {
        if let Some(record) = self.trace_record(txid) {
            self.trace_store.save_queued(record);
        }
    }

    /// The on-disk record of a pending trace.
    fn trace_record(&self, txid: &str) -> Option<keryx_miner::trace_store::StoredTraceOwned> {
        let p = self.pending_traces.get(txid)?;
        Some(keryx_miner::trace_store::StoredTraceOwned {
            response_txid: txid.to_string(),
            model_id: p.model_id,
            tier: p.tier,
            request_hash: p.request_hash,
            max_tokens: p.max_tokens,
            submitted_daa: p.submitted_daa,
            accepted_hash: p.accepted_hash.clone(),
            opening: p.opening.clone(),
            trace: Arc::clone(&p.trace),
            attn: p.attn.as_ref().map(AttnPhases::stored),
            anchor_hash: p.anchor_hash.clone(),
            gen_secs: p.gen_secs,
        })
    }

    /// Loads the stored traces of a previous run and the openings they had already submitted.
    fn restore_traces(
        store: &keryx_miner::trace_store::TraceStore,
    ) -> (std::collections::HashMap<String, PendingTrace>, std::collections::HashMap<String, OpeningInflight>) {
        let mut traces = std::collections::HashMap::new();
        let mut openings = std::collections::HashMap::new();
        for t in store.load_all() {
            let mut resend: Vec<(&String, u64)> = t.opening.iter().flat_map(|o| std::iter::once(&o.payload).chain(o.extra.iter()).map(|p| (p, o.deadline_daa))).collect();
            if let Some(a) = t.attn.as_ref() {
                resend.push((&a.payload, a.deadline_daa));
            }
            let mut attn_txid = None;
            for (payload, deadline_daa) in resend {
                let rpc_tx = crate::proto::RpcTransaction {
                    version: 0,
                    inputs: vec![],
                    outputs: vec![],
                    lock_time: 0,
                    subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
                    gas: 0,
                    payload: payload.clone(),
                    mass: 0,
                    verbose_data: None,
                };
                if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
                    if t.attn.as_ref().is_some_and(|a| &a.payload == payload) {
                        attn_txid = Some(txid.clone());
                    }
                    openings.insert(
                        txid,
                        OpeningInflight {
                            tx: rpc_tx,
                            response_txid: t.response_txid.clone(),
                            deadline_daa,
                            attempts: 1,
                            last: std::time::Instant::now() - std::time::Duration::from_secs(60),
                        },
                    );
                }
            }
            let attn = t.attn.zip(attn_txid).and_then(|(a, txid)| AttnPhases::restored(a, txid));
            traces.insert(
                t.response_txid.clone(),
                PendingTrace {
                    model_id: t.model_id,
                    tier: t.tier,
                    request_hash: t.request_hash,
                    max_tokens: t.max_tokens,
                    trace: Arc::new(t.trace),
                    submitted_daa: t.submitted_daa,
                    accepted_hash: t.accepted_hash,
                    anchor_hash: t.anchor_hash,
                    gen_secs: t.gen_secs,
                    draw: None,
                    opening: t.opening,
                    built: Vec::new(),
                    built_phase: None,
                    reorgs: 0,
                    node_awaits: None,
                    attn,
                    response_tx: None,
                },
            );
        }
        if !traces.is_empty() {
            info!("Trace: restored {} pending response(s) from {} ({} opening(s) to resend)", traces.len(), store.dir().display(), openings.len());
        }
        (traces, openings)
    }

    /// After a restart, asks the node for the chain from the accepting block of every restored
    /// trace, or of its last accepted attention phase, so its draws can be resolved again.
    async fn start_chain_restore(&mut self) -> Result<(), Error> {
        let accepted: Vec<String> = self
            .pending_traces
            .values()
            .filter_map(|p| match (&p.draw, &p.attn) {
                (None, _) => p.accepted_hash.clone().or_else(|| p.anchor_hash.clone()),
                (Some(_), Some(a)) => a.accepted.iter().rev().flatten().find(|h| !h.is_empty()).cloned().or_else(|| p.accepted_hash.clone()),
                (Some(_), None) => None,
            })
            .collect();
        let idle = self.chain_restore_starts.is_empty();
        for hash in accepted {
            if self.chain_restore_starts.contains(&hash) {
                continue;
            }
            if !self.chain_log.iter().any(|e| e.hash == hash) {
                self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
                self.chain_restore_queue.push_back(hash.clone());
            }
            self.chain_restore_starts.push_back(hash);
        }
        if idle {
            self.request_next_chain_restore().await?;
        }
        self.chain_restore_pump().await
    }

    /// Asks the node for the chain after the next restored accepting block.
    async fn request_next_chain_restore(&mut self) -> Result<(), Error> {
        let Some(hash) = self.chain_restore_starts.front().cloned() else { return Ok(()) };
        info!("Trace: rebuilding the chain from block {} for a restored response", hash);
        self.client_send(crate::proto::GetVirtualSelectedParentChainFromBlockRequestMessage {
            start_hash: hash,
            include_accepted_transaction_ids: true,
        })
        .await?;
        Ok(())
    }

    /// Places the chain the node returned after restored accepting block `start` right behind it
    /// in the chain log.
    fn place_restored_chain(&mut self, start: &str, removed: &[String], added: Vec<String>) {
        if !removed.is_empty() {
            self.on_chain_blocks_removed(removed);
        }
        let unknown = place_chain_after(&mut self.chain_log, start, added);
        self.chain_restore_queue.extend(unknown);
    }

    /// Schedules a failed chain rebuild from `start` again, later each time; the responses it
    /// serves are kept until their own windows close.
    fn on_chain_restore_failed(&mut self, start: String, error: &str) {
        let failures = self.chain_restore_failures.entry(start.clone()).or_insert(0);
        *failures += 1;
        let delay = CHAIN_RESTORE_RETRY_BASE.saturating_mul(1 << (*failures - 1).min(4)).min(CHAIN_RESTORE_RETRY_MAX);
        warn!("Trace: chain rebuild from block {} failed: {} — retrying in {} s", start, error, delay.as_secs());
        self.chain_restore_retry.push((std::time::Instant::now() + delay, start));
    }

    /// Sends the traced responses whose trace is now on disk.
    async fn release_persisted_responses(&mut self) -> Result<(), Error> {
        let mut ready = Vec::new();
        self.persisting.retain(|(written, txid, tx)| match written.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            _ => {
                ready.push((txid.clone(), tx.clone()));
                false
            }
        });
        for (txid, tx) in ready {
            self.ai_response_inflight.insert(txid, (tx.clone(), 1, std::time::Instant::now()));
            if let Err(e) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
                warn!("Inference: failed to send AiResponse tx: {}", e);
            }
        }
        Ok(())
    }

    /// Sends again the responses a chain block accepted before their request, once the mempool
    /// forgot them: one the ledger did not credit is admitted, one it credited is refused.
    async fn resend_early_responses(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let traces = &self.pending_traces;
        self.response_resend.retain(|(_, txid, sends)| *sends < EARLY_RESPONSE_RESENDS && traces.get(txid).is_some_and(|p| p.response_tx.is_some()));
        let mut due = Vec::new();
        for (at, txid, sends) in self.response_resend.iter_mut().filter(|(at, _, _)| *at <= now) {
            *at = now + std::time::Duration::from_secs(OPENING_SLOW_RETRY_SECS);
            *sends += 1;
            if let Some(tx) = self.pending_traces.get(txid).and_then(|p| p.response_tx.clone()) {
                due.push(tx);
            }
        }
        for tx in due {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// Asks again for the chain-log blocks whose GetBlock failed, once their retry is due.
    async fn retry_chain_blocks(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let log = &self.chain_log;
        self.chain_block_retry.retain(|(_, h, _)| log.iter().any(|c| c.hash == *h && c.daa.is_none()));
        let due: Vec<String> = self.chain_block_retry.iter().filter(|(at, _, _)| *at <= now).map(|(_, h, _)| h.clone()).collect();
        for (at, h, _) in self.chain_block_retry.iter_mut() {
            if due.contains(h) {
                *at = now + CHAIN_RESTORE_RETRY_MAX;
            }
        }
        for hash in due {
            self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
        }
        Ok(())
    }

    /// Requeues the chain rebuilds whose retry is due while a pending response still needs them.
    async fn retry_chain_restores(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.chain_restore_retry).into_iter().partition(|(at, _)| *at <= now);
        self.chain_restore_retry = later;
        let idle = self.chain_restore_starts.is_empty();
        let restoring = self.restore_needed();
        for (_, start) in due {
            let needed = (restoring && self.chain_restore_continuations.contains(&start))
                || self.pending_traces.values().any(|p| {
                    p.accepted_hash.as_ref().or(p.anchor_hash.as_ref()) == Some(&start)
                        || p.attn.as_ref().is_some_and(|a| a.accepted.iter().flatten().any(|h| *h == start))
                });
            if needed {
                self.chain_restore_starts.push_back(start);
            } else {
                self.chain_restore_failures.remove(&start);
                self.chain_restore_continuations.remove(&start);
            }
        }
        if idle && !self.chain_restore_starts.is_empty() {
            self.request_next_chain_restore().await?;
        }
        Ok(())
    }

    /// Notes the chain blocks accepting the last submitted attention phases; returns their
    /// responses.
    fn note_phase_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) -> Vec<String> {
        let mut noted = Vec::new();
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                for (response, p) in self.pending_traces.iter_mut() {
                    let Some(ph) = p.attn.as_mut() else { continue };
                    if ph.txids.last() == Some(txid) && ph.accepted.last().is_some_and(Option::is_none) {
                        *ph.accepted.last_mut().unwrap() = Some(a.accepting_block_hash.clone());
                        info!("Trace: attention phase {} of response {} accepted by chain block {}", ph.bodies.len() - 1, response, a.accepting_block_hash);
                        noted.push(response.clone());
                    } else if ph.dropped.contains(txid) {
                        ph.late.insert(txid.clone(), a.accepting_block_hash.clone());
                    }
                }
            }
        }
        noted
    }

    /// Ends the submissions of the openings chain blocks accepted.
    fn note_opening_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) {
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                if let Some(o) = self.opening_inflight.remove(txid) {
                    info!("Trace: opening {} of response {} accepted by chain block {}", txid, o.response_txid, a.accepting_block_hash);
                }
            }
        }
    }

    /// Notes the acceptances of restored responses that lost their accepting block.
    fn note_restored_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) {
        self.note_opening_acceptances(accepted);
        let mut noted: Vec<String> = self.note_phase_acceptances(accepted);
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                    p.accepted_hash = Some(a.accepting_block_hash.clone());
                    info!("Trace: response {} accepted by chain block {}", txid, a.accepting_block_hash);
                    noted.push(txid.clone());
                }
            }
        }
        for txid in noted {
            self.persist_trace(&txid);
        }
    }

    /// Whether a pending trace still waits for chain blocks to draw it.
    fn restore_needed(&self) -> bool {
        self.pending_traces.values().any(|p| (p.accepted_hash.is_some() && p.draw.is_none()) || p.attn.as_ref().is_some_and(|a| a.awaiting_draw().is_some()))
    }

    /// Fetches the queued chain blocks a few at a time until every restored trace is drawn.
    async fn chain_restore_pump(&mut self) -> Result<(), Error> {
        if !self.restore_needed() {
            self.chain_restore_queue.clear();
            return Ok(());
        }
        while self.chain_restore_pending.len() < CHAIN_RESTORE_INFLIGHT {
            let Some(hash) = self.chain_restore_queue.pop_front() else { break };
            if !self.chain_restore_pending.insert(hash.clone()) {
                continue;
            }
            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
        }
        Ok(())
    }

    fn note_chain_block(&mut self, hash: &str, daa: u64) {
        for e in self.chain_log.iter_mut().filter(|e| e.hash == hash) {
            e.daa = Some(daa);
        }
    }

    /// Forgets removed chain blocks and the acceptance or draw that depended on them.
    fn on_chain_blocks_removed(&mut self, removed: &[String]) {
        if removed.is_empty() {
            return;
        }
        let removed: std::collections::HashSet<&str> = removed.iter().map(String::as_str).collect();
        self.chain_log.retain(|e| !removed.contains(e.hash.as_str()));
        let mut reset: Vec<String> = Vec::new();
        let mut rephased: Vec<String> = Vec::new();
        for (txid, p) in self.pending_traces.iter_mut() {
            if let Some(a) = p.attn.as_mut() {
                a.late.retain(|_, b| !removed.contains(b.as_str()));
            }
            if p.accepted_hash.as_deref().is_some_and(|h| removed.contains(h)) {
                p.accepted_hash = None;
                p.draw = None;
                p.node_awaits = None;
                reset.push(txid.clone());
            } else if p.draw.as_ref().is_some_and(|d| removed.contains(d.seed_hash.as_str())) {
                p.draw = None;
                reset.push(txid.clone());
            } else if let Some(dropped) = p.attn.as_mut().and_then(|a| a.on_removed(&removed)) {
                if let (Some(d), Some(a)) = (p.draw.as_ref(), p.attn.as_mut()) {
                    a.deadline_daa = a.deadline_daa.min(d.deadline_daa);
                }
                log::debug!("Trace: chain reorg — attention phases of response {} back to phase {}", txid, p.attn.as_ref().map_or(0, |a| a.bodies.len() - 1));
                self.opening_inflight.retain(|t, _| !dropped.contains(t));
                rephased.push(txid.clone());
            }
        }
        for txid in rephased {
            self.persist_trace(&txid);
        }
        for txid in reset {
            log::debug!("Trace: chain reorg — the draw of response {} is reset", txid);
            self.opening_inflight.retain(|_, o| o.response_txid != txid);
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.opening = None;
                p.attn = None;
                p.reorgs += 1;
            }
            self.persist_trace(&txid);
        }
    }

    /// Draws every accepted trace whose seed block is known: the first chain block after the
    /// accepting one at `TRACE_SEED_DELAY_DAA` or more.
    /// Whether a chain rebuild that failed, waits for its retry or continues a page starts at or
    /// after `accepted` in the chain log, so the log may still miss blocks past it.
    fn rebuild_pending_after(&self, accepted: &str) -> bool {
        let Some(from) = self.chain_log.iter().position(|e| e.hash == accepted) else { return false };
        let pending = |h: &String| {
            self.chain_restore_failures.contains_key(h)
                || self.chain_restore_continuations.contains(h)
                || self.chain_restore_retry.iter().any(|(_, s)| s == h)
        };
        self.chain_log.iter().skip(from).any(|e| pending(&e.hash))
    }

    fn resolve_traces(&mut self) {
        let mut drawn: Vec<(String, TraceDraw)> = Vec::new();
        for (txid, p) in self.pending_traces.iter() {
            if p.draw.is_some() {
                continue;
            }
            let Some(accepted) = p.accepted_hash.as_deref() else { continue };
            if self.chain_restore_starts.iter().any(|h| h == accepted) || self.rebuild_pending_after(accepted) {
                continue;
            }
            let stored = p.opening.as_ref().and_then(|o| self.chain_log.iter().find(|e| e.hash == o.seed_hash)).and_then(|e| Some((e.hash.as_str(), e.daa?)));
            let accepted_at = self.chain_log.iter().find(|e| e.hash == accepted).and_then(|e| e.daa);
            let Some((seed_hash, daa, accepted_daa)) = (match (stored, accepted_at) {
                (Some((h, d)), Some(a)) => Some((h, d, a)),
                _ => seed_block_after(&self.chain_log, accepted),
            }) else {
                continue;
            };
            let (Some(txid_bytes), Some(seed_block)) = (hash32(txid), hash32(seed_hash)) else { continue };
            let seed = keryx_miner::trace::trace_seed(&seed_block, &txid_bytes);
            let audited = keryx_miner::trace::audit_all_active(accepted_daa) || keryx_miner::trace::audited(&seed);
            let attention = keryx_miner::trace::attention_active(accepted_daa) && keryx_miner::trace::TierManifest::for_attn(p.tier).is_some();
            let prompt = if attention { keryx_miner::trace::attn_first_prompt_daa(p.trace.record.n_prompt) } else { 0 };
            let deadline_daa = daa + keryx_miner::trace::opening_window_daa(p.tier, p.max_tokens) + prompt;
            drawn.push((
                txid.clone(),
                TraceDraw {
                    seed_hash: seed_hash.to_string(),
                    seed_daa: daa,
                    accepted_daa,
                    seed,
                    audited,
                    deadline_daa,
                    opened: p.opening.as_ref().is_some_and(|o| o.seed_hash == seed_hash),
                    replay_failures: 0,
                    retry_at: None,
                },
            ));
        }
        // The node chains the deadline of each new audit of a key behind the open ones of that
        // key, in seed order then txid order; without its network backlog, this stays at or
        // before the node's deadline.
        drawn.sort_by(|(ta, a), (tb, b)| (a.seed_daa, ta).cmp(&(b.seed_daa, tb)));
        let inflight: std::collections::HashSet<String> = self.opening_inflight.values().map(|o| o.response_txid.clone()).collect();
        let mut open: Vec<u64> = self
            .pending_traces
            .iter()
            .filter_map(|(t, p)| {
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let unfinished = !d.opened || inflight.contains(t) || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                unfinished.then(|| d.deadline_daa.max(p.attn.as_ref().map_or(0, |a| a.deadline_daa)))
            })
            .collect();
        for (txid, d) in drawn.iter_mut().filter(|(_, d)| d.audited) {
            let Some(p) = self.pending_traces.get(txid) else { continue };
            let r = &p.trace.record;
            let work = keryx_miner::trace::service_work_daa(p.tier, r.n_prompt, r.n_pos.saturating_sub(r.n_prompt));
            d.deadline_daa = keryx_miner::trace::chained_deadline(d.deadline_daa, &open, d.seed_daa, work);
            open.push(d.deadline_daa);
        }
        let mut children: Vec<String> = Vec::new();
        let held: std::collections::HashSet<String> = self
            .pending_traces
            .values()
            .filter_map(|p| p.attn.as_ref()?.awaiting_draw().map(|h| h.to_string()))
            .filter(|h| self.rebuild_pending_after(h))
            .collect();
        for (txid, p) in self.pending_traces.iter_mut() {
            let (Some(d), Some(a)) = (p.draw.as_ref(), p.attn.as_mut()) else { continue };
            let Some(accepted) = a.awaiting_draw() else { continue };
            if held.contains(accepted) {
                continue;
            }
            let Some((block, _, accepted_daa)) = seed_block_after(&self.chain_log, accepted) else { continue };
            let (Some(txid_bytes), Some(block_bytes)) = (hash32(txid), hash32(block)) else { continue };
            let phase = a.path.len();
            let Some(n) = keryx_miner::trace_seq::attn_children(p.tier, &p.trace.record, &d.seed, &a.path) else { continue };
            let Some(weights) = a.bodies.last().and_then(|b| keryx_miner::trace_seq::attn_child_weights(p.tier, &p.trace.record, &d.seed, b)).filter(|w| w.len() as u64 == n) else {
                warn!("Trace: response {} — no draw weights for attention phase {}", txid, phase + 1);
                continue;
            };
            let child = keryx_verifiable_inference::attn_blocks::draw_weighted_child(&keryx_miner::trace::attn_child_seed(&block_bytes, &txid_bytes, phase as u8), &weights);
            a.path.push(child);
            a.child_blocks.push(Some(block.to_string()));
            a.deadline_daa = d.deadline_daa.max(accepted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA
                    + keryx_miner::trace::attn_phase_window_daa(p.tier, p.max_tokens, p.trace.record.n_prompt, p.trace.record.n_pos));
            a.opened = false;
            a.failures = 0;
            info!("Trace: response {} — attention phase {} drawn child {} of {}, due by DAA {}", txid, phase + 1, child, n, a.deadline_daa);
            children.push(txid.clone());
        }
        for txid in children {
            self.persist_trace(&txid);
        }
        for (txid, d) in drawn {
            if d.audited {
                let reorgs = self.pending_traces.get(&txid).map_or(0, |p| p.reorgs);
                if d.opened {
                    info!("Trace: response {} — audit restored, opening already submitted", txid);
                } else if reorgs == 0 {
                    info!("Trace: response {} drawn for audit, opening due by DAA {}", txid, d.deadline_daa);
                } else {
                    info!("Trace: response {} drawn for audit after {} chain reorgs, opening due by DAA {}", txid, reorgs, d.deadline_daa);
                }
            }
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.draw = Some(d);
            }
        }
    }

    /// Asks the node which of the pending traces it awaits an opening for.
    async fn poll_awaited_openings(&mut self) -> Result<(), Error> {
        let now = self.last_known_daa;
        let mut due: Vec<(u64, &String)> = self
            .pending_traces
            .iter()
            .filter(|(_, p)| {
                p.accepted_hash.is_some()
                    && p.draw.as_ref().is_none_or(|d| {
                        d.audited
                            && (now <= d.deadline_daa + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA
                                || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some()))
                    })
            })
            .map(|(t, p)| (p.draw.as_ref().map_or(0, |d| d.deadline_daa), t))
            .collect();
        due.sort_unstable();
        let response_txids: Vec<String> = due.into_iter().take(AWAITED_POLL_MAX).map(|(_, t)| t.clone()).collect();
        let escrow_pubkey = self.escrow_pubkey.clone().unwrap_or_default();
        let requests: Vec<(String, [u8; 32])> = if escrow_pubkey.is_empty() {
            Vec::new()
        } else {
            self.ai_request_queue.iter().take(AWAITED_POLL_MAX).map(|(sid, rh, ..)| (sid.clone(), *rh)).collect()
        };
        if response_txids.is_empty() && requests.is_empty() {
            return Ok(());
        }
        self.awaited_polled = response_txids.clone();
        let request_hashes = requests.iter().map(|(_, rh)| hex::encode(rh)).collect();
        self.awaited_requests_polled = requests;
        self.awaited_poll_pending = true;
        self.awaited_poll_sent = std::time::Instant::now();
        self.client_send(GetAwaitedOpeningsRequestMessage { response_txids, escrow_pubkey, request_hashes }).await?;
        Ok(())
    }

    /// Takes the node's seed and deadline for each opening it awaits: the miner's own draw is
    /// only an estimate of the node's ledger.
    fn apply_awaited_openings(&mut self, resp: &crate::proto::GetAwaitedOpeningsResponseMessage) {
        self.awaited_poll_pending = false;
        self.awaited_answered = true;
        if resp.requests_checked {
            self.node_checks_requests = true;
            let awaited: std::collections::HashMap<String, u64> =
                resp.awaited_requests.iter().map(|r| (r.request_hash.to_ascii_lowercase(), r.window_end_daa)).collect();
            let closed: std::collections::HashSet<String> = resp.closed_requests.iter().map(|h| h.to_ascii_lowercase()).collect();
            let now = std::time::Instant::now();
            for (sid, rh) in std::mem::take(&mut self.awaited_requests_polled) {
                let key = hex::encode(rh);
                // A request the ledger does not know yet stays queued.
                if let Some(window_end) = awaited.get(&key) {
                    self.request_checks.insert(sid, (true, *window_end, now));
                } else if closed.contains(&key) {
                    self.request_checks.insert(sid, (false, 0, now));
                }
            }
            self.drop_unawaited_requests();
        } else {
            self.node_checks_requests = false;
            self.awaited_requests_polled.clear();
        }
        let mut changed: Vec<String> = Vec::new();
        let mut resend: Vec<(String, keryx_miner::trace_store::StoredOpening, u64)> = Vec::new();
        let listed: std::collections::HashSet<&str> = resp.openings.iter().map(|o| o.response_txid.as_str()).collect();
        for txid in &self.awaited_polled {
            if let Some(p) = self.pending_traces.get_mut(txid) {
                p.node_awaits = Some(listed.contains(txid.as_str()));
            }
        }
        for o in resp.openings.iter() {
            let idle = !self.opening_inflight.values().any(|f| f.response_txid == o.response_txid)
                && self.replay_rx.as_ref().is_none_or(|j| j.response_txid != o.response_txid);
            let (Some(txid_bytes), Some(seed)) = (hash32(&o.response_txid), hash32(&o.seed)) else { continue };
            let Some(p) = self.pending_traces.get(&o.response_txid) else { continue };
            let had_draw = p.draw.is_some();
            let same_seed = p.draw.as_ref().is_some_and(|d| d.seed == seed);
            let seeded = if same_seed {
                None
            } else {
                let from = p.accepted_hash.as_deref().and_then(|h| self.chain_log.iter().position(|e| e.hash == h)).unwrap_or(0);
                let Some(e) = self
                    .chain_log
                    .iter()
                    .skip(from)
                    .find(|e| hash32(&e.hash).is_some_and(|b| keryx_miner::trace::trace_seed(&b, &txid_bytes) == seed))
                else {
                    log::debug!("Trace: response {} — the node's seed block is not in the chain log yet", o.response_txid);
                    continue;
                };
                Some((e.hash.clone(), e.daa.unwrap_or(0)))
            };
            let p = self.pending_traces.get_mut(&o.response_txid).unwrap();
            if let Some((seed_hash, seed_daa)) = seeded {
                if had_draw {
                    info!("Trace: response {} — seed differs from the node, audit seeded by block {}, opening due by DAA {}", o.response_txid, seed_hash, o.deadline_daa);
                } else {
                    info!("Trace: response {} — audit seeded by the node at block {}, opening due by DAA {}", o.response_txid, seed_hash, o.deadline_daa);
                }
                let kept = p.opening.as_ref().is_some_and(|op| op.seed_hash == seed_hash);
                p.draw = Some(TraceDraw {
                    seed_hash,
                    seed_daa,
                    accepted_daa: o.accepted_daa,
                    seed,
                    audited: true,
                    deadline_daa: o.deadline_daa,
                    opened: kept,
                    replay_failures: 0,
                    retry_at: None,
                });
                if kept {
                    if let Some(op) = p.opening.as_mut() {
                        op.deadline_daa = o.deadline_daa;
                    }
                } else {
                    p.attn = None;
                    p.opening = None;
                    self.opening_inflight.retain(|_, f| f.response_txid != o.response_txid);
                }
                changed.push(o.response_txid.clone());
                continue;
            }
            let d = p.draw.as_mut().unwrap();
            let in_phases = p.attn.as_ref().is_some_and(|a| !a.path.is_empty());
            if d.accepted_daa != o.accepted_daa {
                d.accepted_daa = o.accepted_daa;
                changed.push(o.response_txid.clone());
            }
            if !d.audited || (!in_phases && d.deadline_daa != o.deadline_daa) {
                d.audited = true;
                d.deadline_daa = o.deadline_daa;
                let seed_hash = d.seed_hash.clone();
                // The stored opening follows the deadline the node now sets.
                if let Some(op) = p.opening.as_mut().filter(|op| op.seed_hash == seed_hash) {
                    op.deadline_daa = o.deadline_daa;
                }
                changed.push(o.response_txid.clone());
            }
            if d.opened && idle && !in_phases && p.attn.is_none() {
                match p.opening.as_ref().filter(|op| op.seed_hash == d.seed_hash) {
                    Some(op) => resend.push((o.response_txid.clone(), op.clone(), o.deadline_daa)),
                    None => {
                        info!("Trace: the node still awaits the opening of response {} — replaying it", o.response_txid);
                        d.opened = false;
                        changed.push(o.response_txid.clone());
                    }
                }
            }
            if let Some(a) = p.attn.as_mut().filter(|_| in_phases) {
                if !o.attention_path.is_empty() && a.path != o.attention_path {
                    warn!("Trace: response {} — attention path {:?} differs from the node's {:?}", o.response_txid, a.path, o.attention_path);
                }
                if a.deadline_daa != o.deadline_daa {
                    a.deadline_daa = o.deadline_daa;
                    changed.push(o.response_txid.clone());
                }
            }
        }
        for (response_txid, op, deadline_daa) in resend {
            info!("Trace: the node still awaits the opening of response {} — sending it again", response_txid);
            for payload in std::iter::once(&op.payload).chain(op.extra.iter()) {
                let tx = crate::proto::RpcTransaction {
                    version: 0,
                    inputs: vec![],
                    outputs: vec![],
                    lock_time: 0,
                    subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
                    gas: 0,
                    payload: payload.clone(),
                    mass: 0,
                    verbose_data: None,
                };
                if let Some(txid) = Self::compute_rpc_txid(&tx) {
                    let inflight = OpeningInflight {
                        tx,
                        response_txid: response_txid.clone(),
                        deadline_daa,
                        attempts: 1,
                        last: std::time::Instant::now() - std::time::Duration::from_secs(60),
                    };
                    self.opening_inflight.insert(txid, inflight);
                }
            }
        }
        changed.dedup();
        for o in resp.openings.iter() {
            for f in self.opening_inflight.values_mut().filter(|f| f.response_txid == o.response_txid) {
                f.deadline_daa = o.deadline_daa;
            }
        }
        for txid in changed {
            self.persist_trace(&txid);
        }
    }

    /// Drops the traces past their window and the chain log they no longer need.
    fn prune_traces(&mut self) {
        let daa = self.last_known_daa;
        let awaited_answered = self.awaited_answered;
        let before: Vec<String> = self.pending_traces.keys().cloned().collect();
        let inflight: std::collections::HashSet<String> = self.opening_inflight.values().map(|o| o.response_txid.clone()).collect();
        self.pending_traces.retain(|txid, p| match p.draw.as_ref() {
            // Until the node said on this connection whether it awaits an accepted response, it
            // is kept whatever its age.
            None => {
                p.submitted_daa + TRACE_UNDRAWN_LIMIT_DAA > daa
                    || p.node_awaits == Some(true)
                    || (p.accepted_hash.is_some() && p.node_awaits.is_none() && !awaited_answered)
            }
            Some(d) if !d.audited => d.seed_daa + TRACE_UNAUDITED_GRACE_DAA > daa,
            Some(d) => {
                let deadline = d.deadline_daa.max(p.attn.as_ref().map_or(0, |a| a.deadline_daa));
                let released = p.node_awaits == Some(false) && daa > deadline && !inflight.contains(txid) && p.attn.as_ref().is_none_or(|a| a.awaiting_draw().is_none());
                let unfinished = !d.opened || inflight.contains(txid) || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                let keep = !released
                    && (daa <= deadline
                        || (unfinished && daa <= deadline + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA)
                        || p.attn.as_ref().is_some_and(|a| a.awaiting_draw().is_some() && daa <= a.deadline_daa + 2 * keryx_miner::trace::TRACE_SEED_DELAY_DAA));
                if !keep && daa > d.deadline_daa && !d.opened {
                    warn!("Trace: response {} was never opened — window closed", txid);
                }
                keep
            }
        });
        for txid in before.iter().filter(|t| !self.pending_traces.contains_key(*t)) {
            self.trace_store.remove(txid);
        }
        let needed = self
            .pending_traces
            .values()
            .filter_map(|p| p.accepted_hash.as_deref())
            .filter_map(|h| self.chain_log.iter().position(|e| e.hash == h))
            .min()
            .unwrap_or(self.chain_log.len());
        let keep_from = needed.min(self.chain_log.len().saturating_sub(CHAIN_LOG_TAIL));
        self.chain_log.drain(..keep_from);
    }

    /// Seeds the startup backfill from the first block notification.
    fn backfill_seed(&mut self, header: Option<&crate::proto::RpcBlockHeader>) {
        if self.backfill_cutoff_daa.is_some() {
            return;
        }
        let Some(header) = header else { return };
        let cutoff = header.daa_score.saturating_sub(AI_BACKFILL_WINDOW_DAA);
        self.backfill_cutoff_daa = Some(cutoff);
        if let Some(level0) = header.parents.first() {
            for hash in &level0.parent_hashes {
                if self.backfill_visited.insert(hash.clone()) {
                    self.backfill_queue.push_back(hash.clone());
                }
            }
        }
        info!("Inference: backfilling recent blocks for in-window AiRequests (cutoff daa {})", cutoff);
    }

    /// Keeps up to AI_BACKFILL_INFLIGHT backfill block requests in flight.
    async fn backfill_pump(&mut self) -> Result<(), Error> {
        if self.backfill_cutoff_daa.is_none() {
            return Ok(());
        }
        while self.backfill_pending.len() < AI_BACKFILL_INFLIGHT {
            let Some(hash) = self.backfill_queue.pop_front() else { break };
            self.backfill_pending.insert(hash.clone());
            self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
        }
        Ok(())
    }

    /// Notes a fetched backfill block: walks its parents while above the cutoff.
    fn backfill_note_block(&mut self, hash: &str, header: Option<&crate::proto::RpcBlockHeader>) {
        if !self.backfill_pending.remove(hash) {
            return;
        }
        let Some(cutoff) = self.backfill_cutoff_daa else { return };
        if let Some(header) = header {
            if header.daa_score > cutoff {
                if let Some(level0) = header.parents.first() {
                    for parent in &level0.parent_hashes {
                        if self.backfill_visited.insert(parent.clone()) {
                            self.backfill_queue.push_back(parent.clone());
                        }
                    }
                }
            }
        }
        if self.backfill_queue.is_empty() && self.backfill_pending.is_empty() {
            info!("Inference: startup backfill complete ({} blocks scanned)", self.backfill_visited.len());
            self.backfill_visited.clear();
            self.backfill_visited.shrink_to_fit();
        }
    }

    /// Clears a backfill entry whose GetBlock failed (pruned or unknown block).
    fn backfill_note_error(&mut self, message: &str) {
        if let Some(hash) = self.backfill_pending.iter().find(|h| message.contains(h.as_str())).cloned() {
            self.backfill_pending.remove(&hash);
        }
    }

    /// Resubmits in-flight AiResponses the mempool has not accepted yet.
    async fn retry_pending_ai_responses(&mut self) -> Result<(), Error> {
        if self.ai_response_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.ai_response_inflight.retain(|txid, (tx, attempts, last)| {
            if now.duration_since(*last).as_secs() < AI_RESPONSE_RETRY_SECS {
                return true;
            }
            if *attempts >= AI_RESPONSE_MAX_ATTEMPTS {
                warn!("Inference: giving up on AiResponse {} after {} submit attempts", txid, attempts);
                return false;
            }
            *attempts += 1;
            *last = now;
            resend.push(tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// Polls the in-flight inference task, then the publish task. A finished inference hands
    /// its text to a publish task (IPFS upload, then a gateway fetch of the CID); the
    /// AiResponse is submitted only once a public gateway has served the content.
    /// Returns `true` when a request just reached its end (tx submitted or dropped).
    async fn poll_inference(&mut self) -> bool {
        if let Some(mut pending) = self.publish_rx.take() {
            return match pending.rx.try_recv() {
                Ok(Some(cid)) => {
                    self.submit_ai_response(pending.request_hash, &pending.result, cid, None, None).await;
                    true
                }
                Ok(None) => true,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    self.publish_rx = Some(pending);
                    false
                }
                Err(_) => {
                    warn!("Inference: publish task dropped — AiResponse skipped");
                    true
                }
            };
        }
        let Some(mut job) = self.inference_rx.take() else {
            return false;
        };
        let Ok((result_opt, gen_secs, transient)) = job.rx.try_recv() else {
            self.inference_rx = Some(job);
            return false;
        };
        if result_opt.is_none() && transient {
            if let Some(entry) = job.retry.take() {
                self.retry_request(entry);
                return true;
            }
        }
        if let Some((sid, ..)) = job.retry.as_ref() {
            self.request_retries.remove(sid);
        }
        let Some(generated) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("Inference: inference produced no result — AiResponse skipped");
            return true;
        };
        let InferenceJob { request_hash, private, model_id, tier, max_tokens, .. } = job;
        let result = generated.text;
        let trace = match (generated.trace, tier) {
            (Some(trace), Some(tier)) => {
                speeds::note_generation(&model_id, trace.record.n_prompt, trace.record.n_pos.saturating_sub(trace.record.n_prompt), gen_secs);
                Some(TraceContext { model_id, tier, max_tokens, trace, gen_secs })
            }
            _ => None,
        };

        info!("Inference: inference complete, request_hash={}", hex::encode(&request_hash[..8]));

        // A private answer is sealed to the requester; the plaintext is never published. From
        // the gate on it travels inline and the CID field carries its multihash; before the
        // gate the sealed body is pinned on IPFS like a plaintext answer.
        let publish: Vec<u8> = match private {
            None => result.clone().into_bytes(),
            Some(p) => {
                let body = match p {
                    PrivateRequest::Opened(key) => {
                        let plain = match trace.as_ref().filter(|t| t.trace.record.stage == keryx_miner::trace::STAGE_CHAIN) {
                            Some(t) => {
                                let ids: Vec<u32> = t.trace.tokens[t.trace.record.n_prompt as usize..].iter().map(|&v| v as u32).collect();
                                keryx_verifiable_inference::answer::encode(result.as_bytes(), &ids)
                            }
                            None => result.clone().into_bytes(),
                        };
                        keryx_inference::seal_response(&key, &request_hash, &plain)
                    }
                    PrivateRequest::Unreadable => result.clone().into_bytes(),
                };
                let cap = if keryx_miner::trace::chain_active(self.last_known_daa) {
                    keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN
                } else {
                    keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14
                };
                if body.len() > cap {
                    warn!("Inference: sealed answer of {} B exceeds the {} B body limit — AiResponse skipped", body.len(), cap);
                    return true;
                }
                if self.last_known_daa >= keryx_miner::pom::private_inference_activation_daa() {
                    let cid = crate::ipfs::sha256_multihash(&body);
                    self.submit_ai_response(request_hash, &result, cid, Some(body), trace).await;
                    return true;
                }
                body
            }
        };

        let (tx_cid, rx_cid) = oneshot::channel::<Option<[u8; 34]>>();
        let ipfs_url = self.ipfs_url.clone();
        tokio::task::spawn_blocking(move || {
            let cid = match crate::ipfs::upload_bytes_with_recovery(&publish, &ipfs_url) {
                Ok(cid) => cid,
                Err(e) => {
                    warn!("Inference: IPFS upload failed: {} — AiResponse tx skipped", e);
                    let _ = tx_cid.send(None);
                    return;
                }
            };
            info!("Inference: response published as CID {}", crate::ipfs::multihash_to_cid_v0(&cid));
            let _ = tx_cid.send(Some(cid));
        });
        self.publish_rx = Some(PendingPublish { request_hash, result, rx: rx_cid });
        false
    }

    /// Builds and submits the zero-input/zero-output AiResponse for a confirmed CID. A
    /// `private_body` rides inline under the responder signature (V2 only).
    async fn submit_ai_response(
        &mut self,
        request_hash: [u8; 32],
        result: &str,
        cid: [u8; 34],
        private_body: Option<Vec<u8>>,
        trace: Option<TraceContext>,
    ) {
        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
        // The trace commitment rides after the private body only.
        let trace = trace.filter(|_| private_body.is_some());
        let commitment = trace.as_ref().map(|t| t.trace.commitment());
        // H6 service-bond era: sign the response with the escrow key (payload V2) so it counts
        // as served for the tier cohort — an unsigned response no longer cancels a strike. The
        // era rule mirrors the node's: V2 is rejected before the gate, so v1 is kept below it.
        let v2 = self.last_known_daa >= keryx_miner::pom::pom_v3_activation_daa();
        let resp = match (&self.escrow_watcher, v2) {
            (Some(w), true) => {
                let mut unsigned = keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length);
                if let Some(body) = &private_body {
                    unsigned = unsigned.with_private_body(body.clone());
                }
                if let Some(c) = commitment {
                    unsigned = unsigned.with_trace(c);
                }
                let responder = w.sign_responder(&unsigned.signed_bytes());
                let resp = keryx_inference::AiResponsePayload::new_v2(request_hash, challenge_window_end, cid, response_length, responder);
                let resp = match private_body {
                    Some(body) => resp.with_private_body(body),
                    None => resp,
                };
                match commitment {
                    Some(c) => resp.with_trace(c),
                    None => resp,
                }
            }
            (None, true) => {
                warn!("Inference: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length),
        };
        info!("Inference: uploading response CID={}, challenge_window_end={}{}{}{}", resp.cid_v0(), challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" },
            if resp.private_body.is_some() { " (private body inline)" } else { "" },
            if resp.trace.is_some() { " (traced)" } else { "" });

        let rpc_tx = crate::proto::RpcTransaction {
            version: 0,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            gas: 0,
            payload: hex::encode(resp.serialize()),
            mass: 0,
            verbose_data: None,
        };
        if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
            record_served_request(&mut self.served_requests, &mut self.served_requests_order, request_hash, &txid);
            if let Some(t) = trace.filter(|_| resp.trace.is_some()) {
                let pending = PendingTrace {
                    model_id: t.model_id,
                    tier: t.tier,
                    request_hash,
                    max_tokens: t.max_tokens,
                    trace: Arc::new(t.trace),
                    submitted_daa: self.last_known_daa,
                    accepted_hash: None,
                    anchor_hash: self.chain_log.back().map(|e| e.hash.clone()),
                    gen_secs: Some(t.gen_secs),
                    draw: None,
                    opening: None,
                    built: Vec::new(),
                    built_phase: None,
                    reorgs: 0,
                    node_awaits: None,
                    attn: None,
                    response_tx: Some(rpc_tx.clone()),
                };
                self.pending_traces.insert(txid.clone(), pending);
                // The response leaves only once its trace is on disk: a crash in between would
                // leave an audit nothing can open.
                let (done, written) = std::sync::mpsc::channel();
                if let Some(record) = self.trace_record(&txid) {
                    self.trace_store.save_queued_notify(record, done);
                }
                self.persisting.push((written, txid, rpc_tx));
            } else {
                self.ai_response_inflight.insert(txid, (rpc_tx.clone(), 1, std::time::Instant::now()));
                if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
                    warn!("Inference: failed to send AiResponse tx: {}", e);
                }
            }
        }

        // Register inference escrow outpoint for auto-claim after the challenge window.
        let stable_id = hex::encode(&request_hash[..8]);
        if let Some((txid, inference_reward)) = self.ai_request_txids.remove(&stable_id) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }
    }

    /// Logs this miner's service-bond standing when it changes: strike count, burns awaiting
    /// finality and production suspensions, matched by payout-address identity.
    fn report_service_strikes(&mut self, resp: &crate::proto::GetServiceStrikesResponseMessage) {
        let Some(me) = self.service_identity.as_deref() else { return };
        let strike = resp.strikes.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let suspension = resp.suspended.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let burns: Vec<_> = resp.pending_burns.iter().filter(|b| b.miner.eq_ignore_ascii_case(me)).collect();
        let status = if strike.is_none() && suspension.is_none() && burns.is_empty() {
            "clear".to_string()
        } else {
            let mut parts = Vec::new();
            if let Some(s) = strike {
                parts.push(format!("strike {} (last at daa {})", s.consecutive_misses, s.last_strike_daa_score));
            }
            if !burns.is_empty() {
                let claims: u32 = burns.iter().map(|b| b.burned_claims).sum();
                let sompi: u64 = burns.iter().map(|b| b.burned_sompi).sum();
                parts.push(format!("{} escrow claims / {:.2} KRX burning at finality", claims, sompi as f64 / 100_000_000.0));
            }
            if let Some(s) = suspension {
                parts.push(format!("production suspended until daa {}", s.until_daa_score));
            }
            parts.join("; ")
        };
        let active = strike.map(|s| s.consecutive_misses).unwrap_or(0);
        // The node keeps the lifetime tally: it survives restarts and covers the time this miner
        // was down. Fall back to the locally observed misses only when talking to a node that
        // predates the field (it answers with an empty list).
        for burn in &burns {
            self.misses_seen.insert(burn.miss_daa_score);
        }
        // Attribute each pending burn to this rig or another one; each burn is logged once.
        let mut local_cause = false;
        for burn in &burns {
            let Some(cause) = attribute_strike(&self.served_requests, &self.ai_seen_prefixes, &self.ai_skipped_unready, &burn.request_hash) else { continue };
            local_cause |= !matches!(cause, StrikeCause::Elsewhere);
            if self.strike_burns_reported.len() >= STRIKE_BURNS_REPORTED_KEPT {
                self.strike_burns_reported.clear();
            }
            if !self.strike_burns_reported.insert(burn.request_hash.clone()) {
                continue;
            }
            let request = &burn.request_hash[..burn.request_hash.len().min(16)];
            match cause {
                StrikeCause::AnsweredHere(txid) => warn!(
                    "service-bond: miss #{} on request {}… — THIS rig answered it (response {}); check this rig's audit openings for that response",
                    burn.consecutive_misses, request, txid
                ),
                StrikeCause::DroppedHere => warn!(
                    "service-bond: miss #{} on request {}… — THIS rig queued the request and never answered it; this rig caused the strike",
                    burn.consecutive_misses, request
                ),
                StrikeCause::ModelUnreadyHere => warn!(
                    "service-bond: miss #{} on request {}… — THIS rig saw the request but had no model ready to serve it; check the model files — this rig caused the strike",
                    burn.consecutive_misses, request
                ),
                StrikeCause::Elsewhere => warn!(
                    "service-bond: miss #{} on request {}… — this rig never saw the request; another rig sharing the payout address (or this rig while offline) caused the strike",
                    burn.consecutive_misses, request
                ),
            }
        }
        let total = resp
            .lifetime_strikes
            .iter()
            .find(|t| t.miner.eq_ignore_ascii_case(me))
            .map(|t| t.strikes as usize)
            .unwrap_or(self.misses_seen.len());

        if let Some(stats) = &self.stats {
            // `pending` is the only live evidence of sanctions in flight: the third strike resets
            // the active counter (so a later miss re-escalates from one), and the suspension only
            // appears in `suspended` once finality flushes it. Without it the bar reads "0 active"
            // while three penalties are already decided.
            let pending = burns.len();
            let bar = if suspension.is_some() {
                format!("SUSPENDED · {} total", total)
            } else if pending > 0 {
                format!(
                    "{} active · {} pending{} · {} total",
                    active,
                    pending,
                    if local_cause { " (this rig)" } else { "" },
                    total
                )
            } else {
                format!("{} active · {} total", active, total)
            };
            stats.set_service_status(Some(bar));
        }

        if self.strike_status.as_deref() != Some(status.as_str()) {
            match status.as_str() {
                "clear" => info!("service-bond: no strikes against this miner"),
                s => warn!("service-bond: {}", s),
            }
            self.strike_status = Some(status);
        }
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            // BlockAdded: scan confirmed block for AiRequests and escrow UTXOs.
            // Do NOT trigger a new block template here — NewBlockTemplate handles that.
            Payload::BlockAddedNotification(notif) => {
                if let Some(block) = notif.block {
                    self.backfill_seed(block.header.as_ref());
                    self.backfill_pump().await?;
                    if !block.transactions.is_empty() {
                        // Full block — scan directly.
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        // Pause and drain mining BEFORE spawning inference, so no PoM op can
                        // race the model swap: PAUSE -> DRAIN -> SWAP -> GENERATE -> RESUME.
                        if !self.inference_busy() && self.startable_request().is_some()
                        {
                            self.set_opoi_pause(true);
                            keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-a", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                            miner.process_block(None).await?;
                            self.try_start_inference();
                        }
                        // Escrow: check for new escrow UTXOs and mature claims.
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        if let Some(w) = self.escrow_watcher.as_ref() {
                            let (outputs, sompi) = w.pending_escrow();
                            miner.record_escrow_pending(outputs, sompi);
                        }
                        if let Some(tx) = claim_tx {
                            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
                        }
                    } else {
                        // Transactions absent — fetch the full block from the node.
                        let hash = block
                            .verbose_data
                            .as_ref()
                            .map(|v| v.hash.clone())
                            .unwrap_or_default();
                        if !hash.is_empty() {
                            self.client_send(GetBlockRequestMessage {
                                hash,
                                include_transactions: true,
                            })
                            .await?;
                        }
                    }
                }
            }
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetAwaitedOpeningsResponse(resp) => match resp.error.as_ref() {
                Some(e) => {
                    self.awaited_poll_pending = false;
                    log::debug!("Trace: awaited openings unavailable: {}", e.message)
                }
                None => self.apply_awaited_openings(&resp),
            },
            Payload::GetServiceProvidersResponse(resp) => match resp.error.as_ref() {
                Some(e) => log::debug!("Inference: service providers unavailable: {}", e.message),
                None => self.apply_service_providers(&resp),
            },
            Payload::GetServiceStrikesResponse(resp) => match resp.error.as_ref() {
                Some(e) => warn!("service-bond status unavailable: {}", e.message),
                None => self.report_service_strikes(&resp),
            },
            Payload::GetBlockTemplateResponse(template) => {
                static TEMPLATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = TEMPLATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let daa = template.block.as_ref().and_then(|b| b.header.as_ref()).map_or(0, |h| h.daa_score);
                let (synced, err) = (template.is_synced, template.error.is_some());
                keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-count", || {
                    info!("Templates: {} received, last at daa {} (synced {}, error {}, queue {})", n, daa, synced, err, self.ai_request_queue.len())
                });
                // Track DAA score for challenge_window_end computation.
                if let Some(daa) = template.block.as_ref()
                    .and_then(|b| b.header.as_ref())
                    .map(|h| h.daa_score)
                {
                    if daa > self.last_known_daa {
                        self.last_known_daa = daa;
                        keryx_miner::llm::note_chain_daa(daa);
                    }
                }
                // Handle node-issued inference challenge: spawn an inference task if a new
                // challenge arrived and no challenge is already in flight. Ignored under PoM — the
                // per-block possession proof is the capability gate, so no synthetic challenge
                // (defensive: holds even against a node that still issues them post-hardfork).
                if !template.inference_challenge.is_empty()
                    && !self.inference_busy()
                    && self.last_known_daa < keryx_miner::pom::pom_activation_daa()
                {
                    let challenge = template.inference_challenge.clone();
                    let mut parts = challenge.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("").to_string();
                    let nonce_hex = parts.next().unwrap_or("").to_string();
                    if let Ok(model_id_bytes) = hex::decode(&model_id_hex) {
                        if model_id_bytes.len() == 32 {
                            let mut model_id = [0u8; 32];
                            model_id.copy_from_slice(&model_id_bytes);
                            if keryx_miner::llm::is_model_ready(&model_id) {
                                self.set_opoi_pause(true);
                                miner.process_block(None).await?;
                                info!("Inference: challenge received model={:.8} nonce={:.8} — spawning inference", model_id_hex, nonce_hex);
                                let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
                                let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
                                tokio::task::spawn_blocking(move || {
                                    let result = keryx_miner::llm::load_and_run_inference(&model_id, &prompt, 64);
                                    let _ = tx_done.send(result);
                                });
                                self.challenge_inference_rx = Some((challenge, rx_done));
                            } else {
                                warn!("Inference: challenge for unready model={:.8} — cannot respond", model_id_hex);
                            }
                        }
                    }
                }
                // Poll in-flight inference; if done, submit AiResponse tx then get fresh template.
                if self.poll_inference().await {
                    self.client_get_block_template().await?;
                    return Ok(());
                }
                // Inference is mandatory: refuse to mine if no models are ready.
                // Covers miners with missing/truncated model files that somehow passed prefetch.
                if keryx_miner::llm::loaded_model_ids().is_empty() {
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-no-models", || warn!("Template held: no loaded model"));
                    // Throttle to one log per ~200 templates (~every 20s at 10 BPS) to avoid spam.
                    if self.last_known_daa.is_multiple_of(200) {
                        if keryx_miner::llm::publishing_blocked() {
                            log::warn!("Inference: IPFS node unreachable from the public gateways — mining suspended until it is reachable again (kubo port 4001)");
                        } else {
                            log::warn!("Inference: no models ready — mining suspended until model files are available");
                        }
                    }
                    self.set_opoi_pause(true);
                    miner.process_block(None).await?;
                    return Ok(());
                }
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                }
                if !self.inference_busy() && self.startable_request().is_some()
                {
                    self.set_opoi_pause(true);
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-b", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                    miner.process_block(None).await?;
                    self.try_start_inference();
                }
                // Pause GPU mining while any inference is in flight (GPU is occupied by the model).
                // This covers both regular AiRequest inference and node-issued challenge inference.
                if self.inference_busy() {
                    let (inf, chal, rep) = (self.inference_rx.is_some(), self.challenge_inference_rx.is_some(), self.replay_rx.is_some());
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-busy", || {
                        info!("Template held: inference {} challenge {} replay {}", inf, chal, rep)
                    });
                    self.set_opoi_pause(true);
                    miner.process_block(None).await?;
                    return Ok(());
                }
                // Past this point any pause comes from the node, not from us.
                self.set_opoi_pause(false);
                if let Some(header) = template.block.as_ref().and_then(|b| b.header.as_ref()) {
                    self.resolve_task_seed(header).await;
                }
                match (template.block, template.is_synced, template.error) {
                    (Some(b), true, None) => miner.process_block(Some(FullBlock {
                        block: Box::new(b),
                        device_id: "CPU".to_string(),
                    }))
                    .await?,
                    (Some(b), false, None) if self.mine_when_not_synced => {
                        miner.process_block(Some(FullBlock {
                            block: Box::new(b),
                            device_id: "CPU".to_string(),
                        }))
                        .await?
                    }
                    (_, false, None) => {
                        keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-unsynced", || warn!("Template held: node reports not synced"));
                        miner.process_block(None).await?
                    }
                    (_, _, Some(e)) => {
                        return Err(format!("GetTemplate returned with an error: {:?}", e).into());
                    }
                    (None, true, None) => error!("No block and No Error!"),
                }
            }
            // GetBlock response: either a boot-time validation answer, or a full block we
            // requested from BlockAdded (scanned for AiRequests and escrow UTXOs).
            Payload::GetBlockResponse(msg) => {
                let mut was_validation = false;
                if let Some(e) = msg.error {
                    // A chain block missing from the log would shift every seed block after it:
                    // it is asked again until the node answers or a reorg removes it.
                    if let Some(c) = self.chain_log.iter().find(|c| c.daa.is_none() && e.message.contains(c.hash.as_str())) {
                        let failures = self.chain_block_retry.iter().find(|(_, h, _)| *h == c.hash).map_or(0, |(_, _, f)| *f) + 1;
                        let delay = CHAIN_RESTORE_RETRY_BASE.saturating_mul(1 << (failures - 1).min(4)).min(CHAIN_RESTORE_RETRY_MAX);
                        warn!("Trace: chain block {} unavailable — asked again in {} s", c.hash, delay.as_secs());
                        let hash = c.hash.clone();
                        self.chain_block_retry.retain(|(_, h, _)| *h != hash);
                        self.chain_block_retry.push((std::time::Instant::now() + delay, hash, failures));
                    }
                    // Validation answer: "cannot find header <hash>" — unknown to this
                    // node (pruned or not yet synced), the entries are kept.
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .is_some_and(|w| w.on_block_validation_error(&e.message));
                    if let Some(h) = self.chain_restore_pending.iter().find(|h| e.message.contains(h.as_str())).cloned() {
                        self.chain_restore_pending.remove(&h);
                        self.chain_restore_pump().await?;
                    }
                    if !was_validation {
                        if self.backfill_pending.iter().any(|h| e.message.contains(h.as_str())) {
                            self.backfill_note_error(&e.message);
                            self.backfill_pump().await?;
                        } else {
                            warn!("GetBlockResponse error: {}", e.message);
                        }
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                    // Chain membership from the node's live verdict: a stored-but-reorged
                    // block must purge its entries just like a missing one.
                    let is_chain = block.verbose_data.as_ref().is_some_and(|v| v.is_chain_block);
                    if let Some(daa) = block.header.as_ref().map(|h| h.daa_score) {
                        self.note_chain_block(&hash, daa);
                        self.resolve_traces();
                    }
                    if self.chain_restore_pending.remove(&hash) {
                        self.chain_restore_pump().await?;
                    }
                    was_validation = block.transactions.is_empty()
                        && self.escrow_watcher.as_mut().is_some_and(|w| w.consume_validation_ok(&hash, is_chain));
                    if !was_validation {
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        self.backfill_note_block(&hash, block.header.as_ref());
                        self.backfill_pump().await?;
                        if !self.inference_busy() && self.startable_request().is_some()
                        {
                            self.set_opoi_pause(true);
                            keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-a", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                            miner.process_block(None).await?;
                            self.try_start_inference();
                        }
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        if let Some(w) = self.escrow_watcher.as_ref() {
                            let (outputs, sompi) = w.pending_escrow();
                            miner.record_escrow_pending(outputs, sompi);
                        }
                        if let Some(tx) = claim_tx {
                            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
                        }
                    }
                }
                // Self-paced validation flow: every consumed answer pulls the next
                // queued request, keeping at most VALIDATION_WINDOW in flight.
                if was_validation {
                    if let Some(hash) = self.validation_queue.pop_front() {
                        self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                    }
                }
            }
            Payload::SubmitBlockResponse(res) => {
                let attributed_device = self
                    .pending_block_submissions
                    .lock()
                    .unwrap()
                    .pop_front()
                    .map(|(_, device_id)| device_id)
                    .unwrap_or_else(|| "CPU".to_string());
                match res.error {
                    None => {
                        miner.record_block_accepted();
                        miner.record_block_accepted_for_device(&attributed_device);
                        info!("Block submitted successfully!");
                    }
                    Some(e) => {
                        miner.record_block_rejected();
                        miner.record_block_rejected_for_device(&attributed_device);
                        warn!("Failed submitting block: {:?}", e);
                    }
                }
            }
            Payload::SubmitTransactionResponse(res) => {
                // Escrow claims and inference submissions share this stream. Match responses to
                // in-flight claims by identity (txid, or the txid embedded in the rejection
                // text).
                use crate::escrow::SubmitResponseOutcome;
                let err = res.error.as_ref().map(|e| e.message.clone());
                let outcome = self
                    .escrow_watcher
                    .as_mut()
                    .map_or(SubmitResponseOutcome::NotOurs, |w| {
                        w.on_submit_response(&res.transaction_id, err.as_deref())
                    });
                match outcome {
                    SubmitResponseOutcome::Accepted { outputs, amount_sompi } => {
                        miner.record_claim_accepted(outputs, amount_sompi);
                    }
                    SubmitResponseOutcome::Handled => {}
                    SubmitResponseOutcome::NotOurs => {
                        let opening_txid = if self.opening_inflight.contains_key(&res.transaction_id) {
                            Some(res.transaction_id.clone())
                        } else {
                            err.as_ref().and_then(|e| self.opening_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
                        };
                        if let Some(txid) = opening_txid {
                            // Kept until a chain block accepts it: a node restart or an eviction
                            // empties the mempool, and a syncing node awaits nothing yet.
                            match err.as_deref() {
                                None => {
                                    if let Some(o) = self.opening_inflight.get_mut(&txid) {
                                        o.attempts = o.attempts.max(OPENING_FAST_RETRIES);
                                    }
                                    info!("Trace: opening accepted by the mempool");
                                }
                                Some(e) if e.contains("an opening for response") => {
                                    info!("Trace: another opening of this response is in the mempool — retrying");
                                }
                                Some(e) if e.contains("already") => {
                                    if let Some(o) = self.opening_inflight.get_mut(&txid) {
                                        o.attempts = o.attempts.max(OPENING_FAST_RETRIES);
                                    }
                                    log::debug!("Trace: opening already known to the node");
                                }
                                Some(e) if e.contains("not awaited") => {
                                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                                    if attempts <= 1 {
                                        warn!("Trace: opening not awaited by the node yet — retrying until its window ends: {}", e);
                                    } else {
                                        log::debug!("Trace: opening not awaited (attempt {}): {}", attempts, e);
                                    }
                                }
                                Some(e) => {
                                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                                    if attempts <= 1 {
                                        warn!("Trace: opening rejected: {} — retrying until accepted or expired", e);
                                    } else {
                                        log::debug!("Trace: opening rejected (attempt {}): {}", attempts, e);
                                    }
                                }
                            }
                            return Ok(());
                        }
                        let resent = self
                            .response_resend
                            .iter()
                            .position(|(_, t, _)| *t == res.transaction_id || err.as_ref().is_some_and(|e| e.contains(t.as_str())));
                        if let Some(i) = resent {
                            match err.as_deref() {
                                None => {
                                    info!("Inference: response {} sent again and admitted", self.response_resend[i].1);
                                    self.response_resend.remove(i);
                                }
                                Some(e) if e.contains("not awaited") => {
                                    log::debug!("Inference: response {} already credited", self.response_resend[i].1);
                                    self.response_resend.remove(i);
                                }
                                Some(e) => log::debug!("Inference: response sent again refused: {}", e),
                            }
                            return Ok(());
                        }
                        let inflight_txid = if self.ai_response_inflight.contains_key(&res.transaction_id) {
                            Some(res.transaction_id.clone())
                        } else {
                            // Rejections may carry the txid in the message text instead.
                            err.as_ref().and_then(|e| self.ai_response_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
                        };
                        match (inflight_txid, err) {
                            (Some(txid), None) => {
                                self.ai_response_inflight.remove(&txid);
                                info!("Inference: AiResponse accepted by the mempool");
                            }
                            (Some(txid), Some(e)) if (e.contains(&txid) && e.contains("already")) || e.contains("same responder") => {
                                self.ai_response_inflight.remove(&txid);
                                info!("Inference: AiResponse already known to the node — done");
                            }
                            (Some(txid), Some(e)) => {
                                let attempts = self.ai_response_inflight.get(&txid).map(|(_, a, _)| *a).unwrap_or(1);
                                if attempts <= 1 {
                                    warn!("Inference: AiResponse rejected: {} — retrying until accepted or expired", e);
                                } else {
                                    log::debug!("Inference: AiResponse rejected (attempt {}): {}", attempts, e);
                                }
                            }
                            (None, Some(e)) => log::debug!("Inference: submit_transaction error: {}", e),
                            (None, None) => {}
                        }
                    }
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                self.node_awaited_openings = version_at_least(&info.server_version, AWAITED_OPENINGS_SINCE);
                // Register for all notification types:
                // - NewBlockTemplate drives the mining loop
                // - BlockAdded lets us scan confirmed blocks for AiRequests
                //   that were confirmed before the miner saw them in mempool
                // - VirtualChainChanged drives escrow tracking: only chain-block coinbases
                //   materialize UTXOs, so escrow outputs are tracked from chain blocks only
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_send(NotifyBlockAddedRequestMessage {}).await?;
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage { include_accepted_transaction_ids: true }).await?;
                // Boot-time escrow-state validation: check every referenced block against
                // the node so ghost entries (orphaned-chain coinbases) are purged before
                // any claim ships. Send an initial slice; each answer sends the next.
                if let Some(hashes) = self.escrow_watcher.as_mut().map(|w| w.start_state_validation()) {
                    self.validation_queue = hashes.into();
                    for _ in 0..VALIDATION_WINDOW {
                        if let Some(hash) = self.validation_queue.pop_front() {
                            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                        }
                    }
                }
                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block added notifications (AI request scanning)"),
                Some(e) => error!("Failed registering for block added notifications: {:?}", e),
            },
            Payload::NotifyVirtualSelectedParentChainChangedResponse(res) => match res.error {
                None => {
                    info!("Registered for virtual chain notifications (escrow tracking)");
                    self.start_chain_restore().await?;
                }
                Some(e) => error!("Failed registering for virtual chain notifications: {:?}", e),
            },
            // Virtual chain advanced: fetch every added chain block in full. Their coinbases
            // are the only ones that materialize UTXOs, so escrow tracking feeds off this
            // stream (handle_block gates tracking on is_chain_block). Entries of removed
            // chain blocks are purged right away: their coinbase no longer exists.
            Payload::VirtualSelectedParentChainChangedNotification(notif) => {
                if let Some(watcher) = self.escrow_watcher.as_mut() {
                    watcher.on_chain_blocks_removed(&notif.removed_chain_block_hashes);
                }
                self.on_chain_blocks_removed(&notif.removed_chain_block_hashes);
                for hash in &notif.added_chain_block_hashes {
                    if !self.chain_log.iter().any(|e| &e.hash == hash) {
                        self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
                    }
                }
                let mut accepted_now: Vec<String> = Vec::new();
                for accepted in &notif.accepted_transaction_ids {
                    // A request accepted after its response: the ledger parked that response.
                    for txid in &accepted.accepted_transaction_ids {
                        let early: Vec<String> = self
                            .pending_traces
                            .iter()
                            .filter(|(_, p)| p.accepted_hash.as_ref().is_some_and(|h| *h != accepted.accepting_block_hash) && hex::encode(p.request_hash) == *txid)
                            .map(|(t, _)| t.clone())
                            .collect();
                        for t in early {
                            if !self.response_resend.iter().any(|(_, r, _)| *r == t) {
                                info!("Trace: response {} accepted before its request — sent again shortly", t);
                                self.response_resend.push((std::time::Instant::now() + EARLY_RESPONSE_RESEND_DELAY, t, 0));
                            }
                        }
                    }
                    for txid in &accepted.accepted_transaction_ids {
                        if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                            p.accepted_hash = Some(accepted.accepting_block_hash.clone());
                            if p.reorgs == 0 {
                                info!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                            } else {
                                log::debug!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                            }
                            accepted_now.push(txid.clone());
                        }
                    }
                }
                self.note_opening_acceptances(&notif.accepted_transaction_ids);
                accepted_now.extend(self.note_phase_acceptances(&notif.accepted_transaction_ids));
                for txid in accepted_now {
                    self.persist_trace(&txid);
                }
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            Payload::GetVirtualSelectedParentChainFromBlockResponse(msg) => {
                let Some(start) = self.chain_restore_starts.pop_front() else { return Ok(()) };
                match msg.error {
                    Some(e) => self.on_chain_restore_failed(start, &e.message),
                    None => {
                        self.chain_restore_failures.remove(&start);
                        self.chain_restore_continuations.remove(&start);
                        if msg.added_chain_block_hashes.len() >= CHAIN_RESTORE_PAGE_MIN {
                            if let Some(last) = msg.added_chain_block_hashes.last() {
                                self.chain_restore_starts.push_front(last.clone());
                                self.chain_restore_continuations.insert(last.clone());
                            }
                        }
                        self.place_restored_chain(&start, &msg.removed_chain_block_hashes, msg.added_chain_block_hashes);
                        self.note_restored_acceptances(&msg.accepted_transaction_ids);
                    }
                }
                self.resolve_traces();
                self.request_next_chain_restore().await?;
                self.chain_restore_pump().await?;
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
        LAST_KNOWN_DAA.fetch_max(self.last_known_daa, Ordering::Relaxed);
        if !self.persisting.is_empty() {
            self.trace_store.flush();
            for (_, txid, tx) in std::mem::take(&mut self.persisting) {
                self.ai_response_inflight.insert(txid, (tx, 1, std::time::Instant::now()));
            }
        }
        let parked = Parked {
            queue: std::mem::take(&mut self.ai_request_queue),
            inference: self.inference_rx.take(),
            replay: self.replay_rx.take(),
            publish: self.publish_rx.take(),
            responses: std::mem::take(&mut self.ai_response_inflight),
            request_txids: std::mem::take(&mut self.ai_request_txids),
            queued_at: std::mem::take(&mut self.request_queued_at),
            node_checks_requests: self.node_checks_requests,
            answered: std::mem::take(&mut self.ai_answered_by_me),
            seen: std::mem::take(&mut self.ai_seen_prefixes),
            retries: std::mem::take(&mut self.request_retries),
        };
        if !parked.queue.is_empty()
            || parked.inference.is_some()
            || parked.replay.is_some()
            || parked.publish.is_some()
            || !parked.responses.is_empty()
            || !parked.request_txids.is_empty()
            || !parked.answered.is_empty()
            || !parked.seen.is_empty()
        {
            *PARKED.lock().unwrap_or_else(|p| p.into_inner()) = Some(parked);
        }
    }
}

type QueuedRequest = (String, [u8; 32], [u8; 32], String, u32, usize, Option<PrivateRequest>);

/// Request queue and in-flight work left by a closed connection.
struct Parked {
    queue: VecDeque<QueuedRequest>,
    inference: Option<InferenceJob>,
    replay: Option<ReplayJob>,
    publish: Option<PendingPublish>,
    responses: std::collections::HashMap<String, (crate::proto::RpcTransaction, u32, std::time::Instant)>,
    request_txids: std::collections::HashMap<String, (String, u64)>,
    queued_at: std::collections::HashMap<String, std::time::Instant>,
    node_checks_requests: bool,
    /// Requests this worker answered or saw, so a reconnect neither serves nor queues them again.
    answered: std::collections::HashSet<String>,
    seen: std::collections::HashSet<String>,
    retries: std::collections::HashMap<String, (u8, std::time::Instant)>,
}

static PARKED: Mutex<Option<Parked>> = Mutex::new(None);

/// Last DAA score a closed connection knew of.
static LAST_KNOWN_DAA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Leading bytes of a private prompt compressed with raw DEFLATE.
const COMPRESSED_PROMPT_MAGIC: [u8; 4] = [0x00, b'K', b'Z', 0x01];
const MAX_DECOMPRESSED_PROMPT_LEN: u64 = 1 << 20;

/// Plaintext of an opened private prompt: inflated when it carries the compression magic, cut at
/// the cap. A stream that fails to inflate yields its partial output, else the raw bytes.
pub(crate) fn decode_private_prompt(bytes: &[u8]) -> String {
    use std::io::Read;
    let Some(deflated) = bytes.strip_prefix(&COMPRESSED_PROMPT_MAGIC[..]) else {
        return String::from_utf8_lossy(bytes).into_owned();
    };
    let mut inflated = Vec::new();
    let read = flate2::read::DeflateDecoder::new(deflated).take(MAX_DECOMPRESSED_PROMPT_LEN).read_to_end(&mut inflated);
    if read.is_err() && inflated.is_empty() {
        return String::from_utf8_lossy(deflated).into_owned();
    }
    String::from_utf8_lossy(&inflated).into_owned()
}

/// Generation and replay speeds measured on this rig, by model, for the life of the process.
mod speeds {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// DAA scores per second of the network.
    const DAA_PER_SEC: f64 = 10.0;
    /// Weight of a new measurement in the running averages.
    const EMA: f64 = 0.3;
    /// Fewest generated tokens for a generation to update the speed.
    const MIN_TOKENS: u32 = 32;
    /// Prompt positions per generated token above which a generation does not update the speed.
    const PROMPT_PER_TOKEN: u32 = 16;
    /// Largest step of one measurement over the running speed.
    const OUTLIER: f64 = 4.0;
    /// Replay over generation durations assumed before one is measured.
    const FIRST_RATIO: f64 = 3.0;
    const PHASE_RATIO: f64 = 2.0;

    #[derive(Default, Clone, Copy)]
    struct Speed {
        secs_per_token: Option<f64>,
        first_ratio: Option<f64>,
        phase_ratio: Option<f64>,
    }

    static SPEEDS: Mutex<BTreeMap<[u8; 32], Speed>> = Mutex::new(BTreeMap::new());

    fn ema(old: Option<f64>, new: f64) -> Option<f64> {
        if !new.is_finite() || new <= 0.0 {
            return old;
        }
        Some(old.map_or(new, |o| o + EMA * (new - o)))
    }

    fn get(model: &[u8; 32]) -> Speed {
        SPEEDS.lock().unwrap_or_else(|p| p.into_inner()).get(model).copied().unwrap_or_default()
    }

    /// Records a generation of `n_gen` tokens after a prompt of `n_prompt` that took `secs`;
    /// only generations the prompt weighs little in count, each capped at a few times the
    /// running speed.
    pub(super) fn note_generation(model: &[u8; 32], n_prompt: u32, n_gen: u32, secs: f64) {
        if n_gen < MIN_TOKENS.max(n_prompt / PROMPT_PER_TOKEN) {
            return;
        }
        let mut m = SPEEDS.lock().unwrap_or_else(|p| p.into_inner());
        let s = m.entry(*model).or_default();
        let measured = secs / n_gen as f64;
        s.secs_per_token = ema(s.secs_per_token, s.secs_per_token.map_or(measured, |o| measured.min(OUTLIER * o)));
    }

    /// Records a replay that took `ratio` times its generation.
    pub(super) fn note_replay(model: &[u8; 32], phase: bool, ratio: f64) {
        let ratio = ratio.clamp(0.05, 20.0);
        let mut m = SPEEDS.lock().unwrap_or_else(|p| p.into_inner());
        let s = m.entry(*model).or_default();
        if phase {
            s.phase_ratio = ema(s.phase_ratio, ratio);
        } else {
            s.first_ratio = ema(s.first_ratio, ratio);
        }
    }

    /// Estimated duration of a generation of `n_gen` tokens, in DAA; `None` before a measure.
    pub(super) fn generation_daa(model: &[u8; 32], n_gen: u32) -> Option<u64> {
        get(model).secs_per_token.map(|s| (s * n_gen as f64 * DAA_PER_SEC).round() as u64)
    }

    /// Estimated duration of a replay of a response generated in `gen_secs` (or of `n_gen`
    /// tokens when unknown), in DAA; `None` before a measure.
    pub(super) fn replay_daa(model: &[u8; 32], phase: bool, gen_secs: Option<f64>, n_gen: u32) -> Option<u64> {
        let s = get(model);
        let base = gen_secs.or_else(|| s.secs_per_token.map(|t| t * n_gen.max(1) as f64))?;
        let ratio = if phase { s.phase_ratio.unwrap_or(PHASE_RATIO) } else { s.first_ratio.unwrap_or(FIRST_RATIO) };
        Some((base * ratio * DAA_PER_SEC).round() as u64)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn estimates_follow_the_measured_speeds() {
            let model = [0x7Au8; 32];
            assert_eq!(generation_daa(&model, 100), None);
            assert_eq!(replay_daa(&model, false, None, 100), None);
            assert_eq!(replay_daa(&model, false, Some(10.0), 100), Some(300));
            note_generation(&model, 0, 8, 100.0);
            assert_eq!(generation_daa(&model, 100), None, "a short generation is not a speed");
            note_generation(&model, 32_000, 1_000, 200.0);
            assert_eq!(generation_daa(&model, 100), None, "nor one the prompt weighs in");
            note_generation(&model, 100, 1_000, 20.0);
            assert_eq!(generation_daa(&model, 4_096), Some(819));
            note_replay(&model, false, 1.5);
            assert_eq!(replay_daa(&model, false, Some(10.0), 0), Some(150));
            assert_eq!(replay_daa(&model, false, None, 1_000), Some(300));
            assert_eq!(replay_daa(&model, true, Some(10.0), 0), Some(200));
            note_generation(&model, 0, 1_000, 40.0);
            assert_eq!(generation_daa(&model, 1_000), Some(260));
            note_generation(&model, 0, 1_000, 1_000.0);
            assert!(generation_daa(&model, 1_000).unwrap() < 600, "an outlier moves the speed by a bounded step");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn krx_ai_token_budget_is_capped() {
        let raw = br#"KRX:AI:1:{"m":"glm-4-9b-0414","p":"hi","n":999999}"#;
        let (_, prompt, max_tokens) = super::KeryxdHandler::parse_krx_ai_payload(raw).unwrap();
        assert_eq!((prompt.as_str(), max_tokens as u64), ("hi", super::MAX_KRX_AI_TOKENS));
    }

    #[test]
    fn awaited_openings_need_a_recent_node() {
        assert!(super::version_at_least("1.6.6", (1, 6, 6)));
        assert!(super::version_at_least("1.7.0-PoM", (1, 6, 6)));
        assert!(!super::version_at_least("1.6.5", (1, 6, 6)));
        assert!(!super::version_at_least("", (1, 6, 6)));
    }

    /// Every rig of an address ranks the same way, and each rank is held once.
    #[test]
    fn sibling_ranks_are_a_permutation() {
        use super::sibling_rank;
        let keys: Vec<[u8; 32]> = (1u8..=5).map(|i| [i; 32]).collect();
        for r in 0u8..20 {
            let rh = [r; 32];
            let mut ranks: Vec<usize> = keys.iter().map(|k| sibling_rank(&rh, k, &keys)).collect();
            ranks.sort_unstable();
            assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
        }
        assert_eq!(sibling_rank(&[9; 32], &[1; 32], &[[1; 32]]), 0);
        assert_eq!(sibling_rank(&[9; 32], &[1; 32], &[]), 0);
    }

    /// A burn's request hash attributes the strike: answered here, queued but never answered,
    /// skipped for a missing model, or never seen at all. An empty hash (node predates the
    /// field) claims nothing.
    #[test]
    fn strike_attribution_matches_local_evidence() {
        use super::{attribute_strike, StrikeCause};
        let mut served = std::collections::HashMap::new();
        served.insert(hex::encode([1u8; 32]), "txid-answered".to_string());
        served.insert(hex::encode([0xABu8; 32]), "txid-letters".to_string());
        let mut seen = std::collections::HashSet::new();
        seen.insert(hex::encode([2u8; 8]));
        let mut unready = std::collections::HashSet::new();
        unready.insert(hex::encode([4u8; 8]));

        assert_eq!(
            attribute_strike(&served, &seen, &unready, &hex::encode([1u8; 32])),
            Some(StrikeCause::AnsweredHere("txid-answered".to_string()))
        );
        assert_eq!(attribute_strike(&served, &seen, &unready, &hex::encode([2u8; 32])), Some(StrikeCause::DroppedHere));
        assert_eq!(attribute_strike(&served, &seen, &unready, &hex::encode([4u8; 32])), Some(StrikeCause::ModelUnreadyHere));
        assert_eq!(attribute_strike(&served, &seen, &unready, &hex::encode([3u8; 32])), Some(StrikeCause::Elsewhere));
        assert_eq!(attribute_strike(&served, &seen, &unready, ""), None);
        // Node hashes are hex regardless of case; use a hash with letters so the check is real.
        assert_eq!(
            attribute_strike(&served, &seen, &unready, &hex::encode([0xABu8; 32]).to_uppercase()),
            Some(StrikeCause::AnsweredHere("txid-letters".to_string()))
        );
    }

    /// The served set is FIFO-bounded: the oldest answer leaves once the cap is exceeded, and a
    /// duplicate hash refreshes nothing.
    #[test]
    fn served_requests_evict_oldest_beyond_the_cap() {
        use super::{record_served_request, SERVED_REQUESTS_KEPT};
        let mut served = std::collections::HashMap::new();
        let mut order = std::collections::VecDeque::new();
        for i in 0..SERVED_REQUESTS_KEPT + 1 {
            let mut hash = [0u8; 32];
            hash[..8].copy_from_slice(&(i as u64).to_le_bytes());
            record_served_request(&mut served, &mut order, hash, &format!("txid-{i}"));
        }
        assert_eq!(served.len(), SERVED_REQUESTS_KEPT);
        assert!(!served.contains_key(&hex::encode([0u8; 32])));
        let mut last = [0u8; 32];
        last[..8].copy_from_slice(&(SERVED_REQUESTS_KEPT as u64).to_le_bytes());
        let last_key = hex::encode(last);
        assert_eq!(served.get(&last_key).unwrap(), &format!("txid-{SERVED_REQUESTS_KEPT}"));

        let len = order.len();
        record_served_request(&mut served, &mut order, last, "txid-dup");
        assert_eq!(order.len(), len);
        assert_eq!(served.get(&last_key).unwrap(), &format!("txid-{SERVED_REQUESTS_KEPT}"));
    }

    use super::PrivateRequest;
    use keryx_inference::{escrow_pubkey_of, open_response, seal_request, AiRequestPayload};

    /// Only the responses signed by this miner's escrow key mark a request as already answered.
    #[test]
    fn own_answered_request_ids_keeps_only_this_responder() {
        use keryx_inference::{AiResponder, AiResponsePayload};
        let mine = [0xAAu8; 32];
        let other = [0xBBu8; 32];
        let response = |request_hash: [u8; 32], key: [u8; 32]| crate::proto::RpcTransaction {
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            payload: hex::encode(
                AiResponsePayload::new_v2(request_hash, 10, [0x12u8; 34], 3, AiResponder { escrow_pubkey: key, signature: [0u8; 64] })
                    .serialize(),
            ),
            ..Default::default()
        };
        let txs = vec![response([1u8; 32], mine), response([2u8; 32], other), response([3u8; 32], mine)];
        let ids = super::KeryxdHandler::own_answered_request_ids(&txs, &mine);
        assert_eq!(ids, vec![hex::encode([1u8; 8]), hex::encode([3u8; 8])]);
        assert!(super::KeryxdHandler::own_answered_request_ids(&txs, &[0xCCu8; 32]).is_empty());
    }

    /// A private request is opened for a named recipient, skipped for anyone else, and queued as
    /// unreadable when addressed to this key but tampered with.
    #[test]
    fn open_private_request_classifies_recipients() {
        let mine = [0x11u8; 32];
        let other = [0x22u8; 32];
        let recipients = [escrow_pubkey_of(&mine).unwrap()];
        let (req, _secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"sealed prompt", &recipients).unwrap();

        assert!(super::KeryxdHandler::open_private_request(&req, None).is_none());
        assert!(super::KeryxdHandler::open_private_request(&req, Some(&other)).is_none());
        let (prompt, prompt_len, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        assert_eq!(prompt, "sealed prompt");
        assert_eq!(prompt_len, 13);
        assert!(matches!(private, Some(PrivateRequest::Opened(_))));

        // A flipped ciphertext byte still names this key: answered, with the error body.
        let mut tampered = req.serialize();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let tampered = AiRequestPayload::deserialize(&tampered).unwrap();
        let (prompt, _, private) = super::KeryxdHandler::open_private_request(&tampered, Some(&mine)).unwrap();
        assert!(prompt.is_empty());
        assert!(matches!(private, Some(PrivateRequest::Unreadable)));
    }

    /// The sealed answer opens with the requester's root key and nobody else's.
    #[test]
    fn sealed_answer_round_trips_to_the_requester() {
        let mine = [0x11u8; 32];
        let pubkey = escrow_pubkey_of(&mine).unwrap();
        let (req, secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"q", &[pubkey]).unwrap();
        let (_, _, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        let Some(PrivateRequest::Opened(key)) = private else { panic!("not opened") };
        let request_hash = [9u8; 32];
        let body = keryx_inference::seal_response(&key, &request_hash, b"answer");
        assert_eq!(open_response(&secret, &request_hash, &pubkey, &body).unwrap(), b"answer");
        let stranger = keryx_inference::PrivateRequestSecret::from_hex(&"00".repeat(32)).unwrap();
        assert!(open_response(&stranger, &request_hash, &pubkey, &body).is_err());
        let cid = crate::ipfs::sha256_multihash(&body);
        assert_eq!(&cid[..2], &[0x12, 0x20]);
    }

    #[test]
    fn private_prompt_inflates_when_marked() {
        use super::{decode_private_prompt, COMPRESSED_PROMPT_MAGIC, MAX_DECOMPRESSED_PROMPT_LEN};
        use std::io::Write;
        let deflate = |data: &[u8]| {
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(data).unwrap();
            let mut out = COMPRESSED_PROMPT_MAGIC.to_vec();
            out.extend(enc.finish().unwrap());
            out
        };

        assert_eq!(decode_private_prompt("Bonjour, ça va ?".as_bytes()), "Bonjour, ça va ?");
        let text = "Entrée 41 — le code est ORCHIDÉE-7319. ".repeat(500);
        let packed = deflate(text.as_bytes());
        assert!(packed.len() < text.len() / 10);
        assert_eq!(decode_private_prompt(&packed), text);

        let mut corrupt = COMPRESSED_PROMPT_MAGIC.to_vec();
        corrupt.extend([0xFF; 32]);
        assert_eq!(decode_private_prompt(&corrupt), "\u{FFFD}".repeat(32));
        let ascii: String = (0..4000).map(|i| char::from(b'a' + (i * 7 % 26) as u8)).collect();
        let mut truncated = deflate(ascii.as_bytes());
        truncated.truncate(truncated.len() / 2);
        let partial = decode_private_prompt(&truncated);
        assert!(!partial.is_empty() && ascii.starts_with(&partial));
        let bomb = deflate(&vec![b'a'; MAX_DECOMPRESSED_PROMPT_LEN as usize + 1]);
        assert_eq!(decode_private_prompt(&bomb), "a".repeat(MAX_DECOMPRESSED_PROMPT_LEN as usize));
    }

    /// Round-trip against a live node. Ignored by default — run with
    /// `cargo test --bin keryx-miner -- --ignored query_virtual_daa` and a node on 22110.
    #[tokio::test]
    #[ignore]
    async fn query_virtual_daa_reads_the_live_node() {
        let daa = super::query_virtual_daa("grpc://127.0.0.1:22110".to_string()).await;
        assert!(daa.is_some_and(|d| d > 0), "no DAA read from the node: {:?}", daa);
        println!("node virtual daa = {}", daa.unwrap());
    }

    fn phase_body(path: Vec<u64>, partials: usize) -> Vec<u8> {
        keryx_verifiable_inference::opening::AttnOpening { path, partials: vec![7; partials], ..Default::default() }.encode()
    }

    /// Phase 0 accepted by `a0`, child 2 drawn by `b0`, phase 1 accepted by `a1`.
    fn two_phases() -> super::AttnPhases {
        let mut a = super::AttnPhases::new(phase_body(vec![], 8), "t0".into(), "p0".into(), 100);
        assert_eq!(a.awaiting_draw(), None, "a phase awaits its acceptance before its draw");
        a.accepted[0] = Some("a0".into());
        assert_eq!(a.awaiting_draw(), Some("a0"));
        assert!(!a.ready());
        a.path.push(2);
        a.child_blocks.push(Some("b0".into()));
        assert!(a.ready() && a.awaiting_draw().is_none());
        a.bodies.push(phase_body(vec![2], 8));
        a.txids.push("t1".into());
        a.payloads.push("p1".into());
        a.accepted.push(Some("a1".into()));
        a.opened = true;
        a
    }

    #[test]
    fn attention_phases_step_back_to_what_a_reorg_leaves() {
        let removed = |h: &'static str| std::collections::HashSet::from([h]);
        let mut a = two_phases();
        assert_eq!(a.on_removed(&removed("zz")), None);
        assert_eq!(a.on_removed(&removed("a1")), Some(vec![]));
        assert_eq!((a.bodies.len(), a.path.clone(), a.accepted.clone(), a.opened), (2, vec![2], vec![Some("a0".into()), None], false));
        let mut a = two_phases();
        assert_eq!(a.on_removed(&removed("b0")), Some(vec!["t1".to_string()]));
        assert_eq!((a.bodies.len(), a.path.len(), a.child_blocks.len(), a.awaiting_draw()), (1, 0, 0, Some("a0")));
        let mut a = two_phases();
        assert_eq!(a.on_removed(&std::collections::HashSet::from(["a0", "a1"])), Some(vec!["t1".to_string()]));
        assert_eq!((a.bodies.len(), a.accepted.clone(), a.awaiting_draw()), (1, vec![None], None));
    }

    #[test]
    fn the_last_phase_awaits_no_draw() {
        let mut a = two_phases();
        a.bodies[1] = phase_body(vec![2], 0);
        assert_eq!(a.awaiting_draw(), None);
        assert!(!a.ready());
    }

    #[test]
    fn attention_phases_restore_from_the_store() {
        let a = two_phases();
        let back = super::AttnPhases::restored(a.stored(), "t1".into()).unwrap();
        assert_eq!((back.path.clone(), back.txids.last().cloned(), back.accepted.last().cloned()), (vec![2], Some("t1".into()), Some(Some("a1".into()))));
        assert_eq!(back.awaiting_draw(), Some("a1"));
        let mut broken = a.stored();
        broken.bodies[1] = phase_body(vec![2, 1], 8);
        assert!(super::AttnPhases::restored(broken, "t1".into()).is_none());
    }

    #[test]
    fn seed_block_is_the_first_chain_block_past_the_delay() {
        use super::{seed_block_after, ChainEntry};
        let entry = |hash: &str, daa: Option<u64>| ChainEntry { hash: hash.to_string(), daa };
        let delay = keryx_miner::trace::TRACE_SEED_DELAY_DAA;
        let mut log: std::collections::VecDeque<ChainEntry> =
            [entry("a", Some(1_000)), entry("b", Some(1_000 + delay - 1)), entry("c", None), entry("d", Some(1_000 + delay + 5))].into();
        assert_eq!(seed_block_after(&log, "a"), None, "an unknown DAA on the way holds the draw");
        log[2].daa = Some(1_000 + delay);
        assert_eq!(seed_block_after(&log, "a"), Some(("c", 1_000 + delay, 1_000)));
        assert_eq!(seed_block_after(&log, "b"), None, "no block past the delay yet");
        assert_eq!(seed_block_after(&log, "zz"), None);
        log[0].daa = None;
        assert_eq!(seed_block_after(&log, "a"), None, "the accepting block needs its own DAA");
    }

    #[test]
    fn restored_chain_is_placed_behind_its_start() {
        use super::{place_chain_after, seed_block_after, ChainEntry};
        let entry = |hash: &str, daa: Option<u64>| ChainEntry { hash: hash.to_string(), daa };
        let delay = keryx_miner::trace::TRACE_SEED_DELAY_DAA;
        // Accepting blocks of two restored responses, logged in no chain order.
        let mut log: std::collections::VecDeque<ChainEntry> = [entry("b", Some(1_000)), entry("z", Some(1_000 + delay + 30))].into();
        assert_eq!(seed_block_after(&log, "b"), Some(("z", 1_000 + delay + 30, 1_000)), "an unordered log draws the wrong seed block");

        let unknown = place_chain_after(&mut log, "b", vec!["c".into(), "d".into(), "z".into()]);
        assert_eq!(unknown, vec!["c".to_string(), "d".to_string()]);
        let order: Vec<&str> = log.iter().map(|e| e.hash.as_str()).collect();
        assert_eq!(order, ["b", "c", "d", "z"]);
        assert_eq!(log[3].daa, Some(1_000 + delay + 30), "a moved block keeps its DAA");
        log[1].daa = Some(1_000 + delay - 1);
        log[2].daa = Some(1_000 + delay + 1);
        assert_eq!(seed_block_after(&log, "b"), Some(("d", 1_000 + delay + 1, 1_000)));

        assert_eq!(place_chain_after(&mut log, "gone", vec!["d".into(), "x".into()]), vec!["x".to_string()]);
        let order: Vec<&str> = log.iter().map(|e| e.hash.as_str()).collect();
        assert_eq!(order, ["b", "c", "z", "d", "x"], "a chain whose start left the log goes to its end");
    }
}
