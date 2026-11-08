//! Audit replays of traced answers served through a pool.
use super::statum_codec::StratumLine;
use keryx_miner::trace_store::{StoredAttn, StoredOpening, StoredTraceOwned, TraceStore};
use keryx_verifiable_inference::opening::{AttnOpening, VERSION_ATTN, VERSION_RECUR, VERSION_RESID};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc::Sender;
use tokio::sync::{Mutex, Notify};

/// Traces kept at once.
pub(super) const MAX_TRACES: usize = 64;
/// DAA the node keeps a request pending after its acceptance.
pub(super) const LEDGER_HORIZON_DAA: u64 = 72_000;
/// DAA a trace with no draw yet is kept after its answer.
const UNDRAWN_KEEP_DAA: u64 = LEDGER_HORIZON_DAA + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA;
/// DAA a drawn trace is kept past its last deadline, for later attention phases and redraws.
const DRAWN_KEEP_DAA: u64 = 6_000;
/// Latest deadline an audit can reach after its acceptance, and the delay of a phase draw.
pub(super) const AUDIT_SPAN_DAA: u64 = 270_000;
const PHASE_DRAW_DELAY_DAA: u64 = keryx_miner::trace::TRACE_SEED_DELAY_DAA;
/// First delay before a failed replay is retried, doubled at each failure up to the max.
const REPLAY_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(30);
const REPLAY_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(600);
/// Longest attention path a draw may carry.
const MAX_DRAW_PATH: usize = 64;
/// Latest deadline a draw may set past the current DAA.
const MAX_DEADLINE_AHEAD_DAA: u64 = AUDIT_SPAN_DAA + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA;
/// Max DAA rise per second of wall clock, and its slack.
const MAX_DAA_PER_SEC: u64 = 20;
const DAA_SLACK: u64 = 6_000;
/// Bodies kept per trace before those of other seeds are dropped.
const MAX_BUILT: usize = 16;
const MAX_PHASES: usize = 64;
pub(super) const PIECES_ALL: u32 = PIECE_TRACE | PIECE_ATTN | PIECE_RECUR | PIECE_RESID;

const PIECE_TRACE: u32 = 1;
pub(super) const PIECE_ATTN: u32 = 2;
const PIECE_RECUR: u32 = 4;
const PIECE_RESID: u32 = 8;

/// Piece bit of an unseeded opening body, as the node counts the pieces still owed.
fn piece_bit(body: &[u8]) -> u32 {
    match body.first().copied() {
        Some(VERSION_ATTN) => PIECE_ATTN,
        Some(VERSION_RECUR) => PIECE_RECUR,
        Some(VERSION_RESID) => PIECE_RESID,
        _ => PIECE_TRACE,
    }
}

/// A traced answer this worker generated, kept until its audit is over.
pub(super) struct PoolTrace {
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub trace: Arc<keryx_miner::trace::ResponseTrace>,
    created_daa: u64,
    /// Latest deadline of a draw seen for this answer.
    deadline_daa: Option<u64>,
    /// Earliest acceptance DAA a draw gave.
    accepted_daa: Option<u64>,
    /// Bodies built per draw, to answer a repeated draw without replaying.
    built: HashMap<String, Vec<Vec<u8>>>,
    /// Attention phase bodies built, by draw key; each feeds the phase drawn below it.
    pub phases: HashMap<String, Vec<u8>>,
    /// Seed of the latest phase built; only its phases are persisted.
    phase_seed: Option<[u8; 32]>,
    /// Failed replays by draw key, so a draw given up is not replayed on every resend.
    failures: HashMap<String, Failures>,
    /// The answer sent for this trace, resent unchanged when the pool asks again.
    pub response: Option<SentAnswer>,
}

impl PoolTrace {
    pub fn new(
        model_id: [u8; 32],
        tier: u8,
        request_hash: [u8; 32],
        trace: keryx_miner::trace::ResponseTrace,
        daa: u64,
    ) -> Self {
        Self {
            model_id,
            tier,
            request_hash,
            trace: Arc::new(trace),
            created_daa: daa,
            deadline_daa: None,
            accepted_daa: None,
            built: HashMap::new(),
            phases: HashMap::new(),
            phase_seed: None,
            failures: HashMap::new(),
            response: None,
        }
    }

    /// DAA the answer was generated at, close to its acceptance.
    pub fn created_daa(&self) -> u64 {
        self.created_daa
    }

    /// What to do with an opening of the draw `key` that is not cached.
    pub fn opening_action(&mut self, key: &str) -> OpeningAction {
        let now = std::time::Instant::now();
        let action = opening_action(self.failures.get(key), now);
        match action {
            OpeningAction::Replay => {
                if self.failures.get(key).is_some_and(|f| f.count >= MAX_REPLAY_ATTEMPTS) {
                    self.failures.remove(key);
                }
            }
            OpeningAction::Reply => {
                if let Some(f) = self.failures.get_mut(key) {
                    f.last_reply = Some(now);
                }
            }
            OpeningAction::Ignore => {}
        }
        action
    }

    /// Counts one more failed replay of the draw `key`; returns the new count.
    pub fn note_failure(&mut self, key: &str) -> u8 {
        bump_failure(&mut self.failures, key, std::time::Instant::now())
    }

    fn expired(&self, daa: u64) -> bool {
        expired(self.created_daa, self.deadline_daa, self.accepted_daa, daa)
    }

    /// Records the deadline and acceptance of a draw; true when they changed.
    pub fn note_draw(&mut self, deadline_daa: u64, accepted_daa: u64) -> bool {
        let deadline = self.deadline_daa.map_or(deadline_daa, |d| d.max(deadline_daa));
        let accepted = self.accepted_daa.map_or(accepted_daa, |a| a.min(accepted_daa)).min(deadline);
        let changed = self.deadline_daa != Some(deadline) || self.accepted_daa != Some(accepted);
        self.deadline_daa = Some(deadline);
        self.accepted_daa = Some(accepted);
        changed
    }

    /// The trace opening bundle built under `seed`.
    pub fn bundle(&self, seed: &[u8; 32]) -> Option<Vec<Vec<u8>>> {
        self.built.get(&key(seed, &[])).cloned()
    }

    /// Bodies answering `draw` without a replay, or None when a piece it owes is not built yet.
    pub fn cached(&self, draw: &PoolDraw) -> Option<Vec<Vec<u8>>> {
        let k = draw.key();
        if draw.path.is_empty() {
            return self.built.get(&k).cloned().or_else(|| {
                (draw.pieces & !PIECE_ATTN == 0).then(|| self.phases.get(&k).map(|b| vec![b.clone()])).flatten()
            });
        }
        let phase = self
            .built
            .get(&k)
            .and_then(|b| b.iter().find(|x| piece_bit(x) == PIECE_ATTN).cloned())
            .or_else(|| self.phases.get(&k).cloned())?;
        let mut bodies = Vec::new();
        if draw.pieces & !PIECE_ATTN != 0 {
            bodies.extend(self.bundle(&draw.seed)?.into_iter().filter(|b| piece_bit(b) != PIECE_ATTN));
        }
        bodies.push(phase);
        Some(bodies)
    }

    /// Keeps what was built for `draw`; true when phases were added and the record must be
    /// rewritten.
    pub fn store_built(&mut self, draw: &PoolDraw, built: Built) -> bool {
        self.failures.remove(&draw.key());
        let prefix = format!("{}:", hex::encode(draw.seed));
        if self.built.len() >= MAX_BUILT {
            self.built.retain(|k, _| k.starts_with(&prefix));
        }
        if let Some(bundle) = built.bundle {
            self.built.insert(key(&draw.seed, &[]), bundle);
        }
        self.built.insert(draw.key(), built.bodies);
        let new_phases = built.phases;
        if new_phases.is_empty() {
            return false;
        }
        if self.phases.len() + new_phases.len() > MAX_PHASES {
            self.phases.retain(|k, _| k.starts_with(&prefix));
        }
        self.phase_seed = Some(draw.seed);
        self.phases.extend(new_phases);
        true
    }

    /// Phase bodies of `seed`, shallowest first.
    fn phases_of(&self, seed: &[u8; 32]) -> Vec<Vec<u8>> {
        let prefix = format!("{}:", hex::encode(seed));
        let mut bodies: Vec<(usize, Vec<u8>)> = self
            .phases
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .filter_map(|(_, b)| AttnOpening::decode(b).map(|o| (o.path.len(), b.clone())))
            .collect();
        bodies.sort_by_key(|(depth, _)| *depth);
        bodies.into_iter().map(|(_, b)| b).collect()
    }

    fn record(&self, key: &str) -> StoredTraceOwned {
        StoredTraceOwned {
            response_txid: key.to_string(),
            model_id: self.model_id,
            tier: self.tier,
            request_hash: self.request_hash,
            max_tokens: 0,
            submitted_daa: self.created_daa,
            accepted_hash: None,
            opening: self.deadline_daa.map(|d| StoredOpening {
                seed_hash: self.phase_seed.map(hex::encode).unwrap_or_default(),
                deadline_daa: d,
                payload: String::new(),
                extra: Vec::new(),
            }),
            trace: Arc::clone(&self.trace),
            attn: self.phase_seed.map(|s| StoredAttn {
                bodies: self.phases_of(&s),
                payload: String::new(),
                deadline_daa: self.deadline_daa.unwrap_or(0),
                accepted_hash: None,
            }),
            anchor_hash: None,
            gen_secs: None,
        }
    }
}

/// Replays tried for one draw before it is given up for a while.
pub(super) const MAX_REPLAY_ATTEMPTS: u8 = 5;
/// Time after the last failure at which a given-up draw is tried again.
const GIVE_UP_RESET: std::time::Duration = std::time::Duration::from_secs(30 * 60);
/// Least time between two answers to resends of a given-up draw.
const GIVE_UP_REPLY_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// Failed replays of one draw.
#[derive(Clone, Copy, Debug)]
pub(super) struct Failures {
    count: u8,
    last_failure: std::time::Instant,
    last_reply: Option<std::time::Instant>,
}

/// What an opening that is not cached gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpeningAction {
    /// Queue it for a replay.
    Replay,
    /// Answer "replay failed" now.
    Reply,
    /// Drop it: the last answer is recent.
    Ignore,
}

/// The answer an opening gets after the failures `f` of its draw, at `now`.
fn opening_action(f: Option<&Failures>, now: std::time::Instant) -> OpeningAction {
    match f {
        Some(f) if f.count >= MAX_REPLAY_ATTEMPTS && now.duration_since(f.last_failure) < GIVE_UP_RESET => {
            match f.last_reply {
                Some(at) if now.duration_since(at) < GIVE_UP_REPLY_EVERY => OpeningAction::Ignore,
                _ => OpeningAction::Reply,
            }
        }
        _ => OpeningAction::Replay,
    }
}

/// Counts one more failure of `key` in `failures`, bounded to `MAX_BUILT` keys; the new count.
fn bump_failure(failures: &mut HashMap<String, Failures>, key: &str, now: std::time::Instant) -> u8 {
    if failures.len() >= MAX_BUILT && !failures.contains_key(key) {
        if let Some(oldest) = failures.iter().min_by_key(|(_, f)| f.last_failure).map(|(k, _)| k.clone()) {
            failures.remove(&oldest);
        }
    }
    let f = failures.entry(key.to_string()).or_insert(Failures { count: 0, last_failure: now, last_reply: None });
    f.count = f.count.saturating_add(1);
    f.last_failure = now;
    f.count
}

/// An answer sent to the pool: its text and, when traced, the commitment and the sealed body.
#[derive(Clone, Debug)]
pub(super) struct SentAnswer {
    pub text: String,
    pub traced: Option<(String, Vec<u8>)>,
}

/// A trace is past its keep window at `daa`.
fn expired(created_daa: u64, deadline_daa: Option<u64>, accepted_daa: Option<u64>, daa: u64) -> bool {
    match deadline_daa {
        Some(d) => daa > d.saturating_add(DRAWN_KEEP_DAA).max(audit_end(accepted_daa.unwrap_or(d).min(d))),
        None => daa > created_daa.saturating_add(UNDRAWN_KEEP_DAA),
    }
}

/// Last DAA a deadline of an audit accepted at `accepted_daa` can be pushed to, with a phase draw.
fn audit_end(accepted_daa: u64) -> u64 {
    accepted_daa.saturating_add(AUDIT_SPAN_DAA + PHASE_DRAW_DELAY_DAA).saturating_add(DRAWN_KEEP_DAA)
}

/// A draw relayed by the pool for one answer of this worker.
#[derive(Clone, Debug)]
pub(super) struct PoolDraw {
    pub request_hash: String,
    pub response_txid: [u8; 32],
    pub seed: [u8; 32],
    pub deadline_daa: u64,
    pub accepted_daa: u64,
    pub pieces: u32,
    pub path: Vec<u64>,
    pub attempts: u8,
    /// No replay before this instant, after a failed one.
    pub retry_at: Option<std::time::Instant>,
}

impl PoolDraw {
    pub fn new(
        request_hash: String,
        response_txid: [u8; 32],
        seed: [u8; 32],
        deadline_daa: u64,
        accepted_daa: u64,
        pieces: u32,
        path: Vec<u64>,
    ) -> Self {
        Self {
            request_hash,
            response_txid,
            seed,
            deadline_daa,
            accepted_daa,
            pieces,
            path,
            attempts: 0,
            retry_at: None,
        }
    }

    pub fn key(&self) -> String {
        key(&self.seed, &self.path)
    }

    /// The draw to queue again after a failed replay, retried later.
    pub fn retry(self) -> Self {
        let delay = REPLAY_RETRY_BASE.saturating_mul(1u32 << self.attempts.min(5)).min(REPLAY_RETRY_MAX);
        Self { attempts: self.attempts.saturating_add(1), retry_at: Some(std::time::Instant::now() + delay), ..self }
    }

    /// Whether a replay may start now.
    pub fn ready(&self) -> bool {
        self.retry_at.is_none_or(|t| t <= std::time::Instant::now())
    }

    /// Whether the node may still await this draw at `daa`, its deadline pushed or not.
    fn live(&self, daa: u64) -> bool {
        self.deadline_daa > daa || daa <= audit_end(self.accepted_daa.min(self.deadline_daa))
    }
}

/// Takes out of the queue the draws already built, with their owed bodies.
pub(super) async fn take_cached() -> Vec<(PoolDraw, Vec<Vec<u8>>)> {
    let s = shared();
    let mut queue = s.queue.lock().await;
    let traces = s.traces.lock().await;
    let mut ready = Vec::new();
    queue.retain(|d| match traces.get(&d.request_hash).filter(|_| d.ready()).and_then(|t| t.cached(d)) {
        Some(bodies) => {
            ready.push((d.clone(), owed(&bodies, d.pieces)));
            false
        }
        None => true,
    });
    ready
}

/// Why a draw is out of bounds at `daa`, if it is.
pub(super) fn draw_bounds_error(daa: u64, deadline_daa: u64, pieces: u32, path_len: usize) -> Option<&'static str> {
    if daa == 0 {
        return Some("no job yet");
    }
    if path_len > MAX_DRAW_PATH {
        return Some("path too long");
    }
    if pieces == 0 {
        return None;
    }
    if pieces & PIECES_ALL == 0 {
        return Some("no known piece");
    }
    if deadline_daa <= daa || deadline_daa > daa.saturating_add(MAX_DEADLINE_AHEAD_DAA) {
        return Some("deadline out of range");
    }
    None
}

/// `daa` capped by the rise the wall clock allows since `anchor` was seen `elapsed_ms` ago.
fn plausible_daa(anchor: u64, elapsed_ms: u64, daa: u64) -> u64 {
    daa.min(anchor.saturating_add(elapsed_ms.saturating_mul(MAX_DAA_PER_SEC) / 1_000).saturating_add(DAA_SLACK))
}

/// Key of the draw of `path` under `seed`.
pub(super) fn draw_key(seed: &[u8; 32], path: &[u64]) -> String {
    key(seed, path)
}

fn key(seed: &[u8; 32], path: &[u64]) -> String {
    format!("{}:{:?}", hex::encode(seed), path)
}

/// The live pool connection, for work finishing after the connection that started it.
#[derive(Clone)]
pub(super) struct Link {
    pub sender: Sender<StratumLine>,
    pub last_id: Arc<AtomicU32>,
    pub openings_pending: Arc<Mutex<HashSet<u32>>>,
}

/// Audit state shared by every pool connection of the process.
pub(super) struct Shared {
    pub traces: Arc<Mutex<HashMap<String, PoolTrace>>>,
    pub queue: Arc<Mutex<Vec<PoolDraw>>>,
    /// The GPU is taken by a request, a challenge or a replay.
    pub busy: Arc<AtomicBool>,
    pub link: std::sync::Mutex<Option<Link>>,
    /// Signalled when GPU work ends, so queued work starts.
    pub wake: Notify,
    store: OnceLock<TraceStore>,
    /// DAA from the pool, capped by the wall clock.
    clock: std::sync::Mutex<DaaClock>,
}

#[derive(Default)]
struct DaaClock {
    /// First DAA seen and when.
    anchor: Option<(u64, std::time::Instant)>,
    current: u64,
}

/// Takes a job's DAA, capped by the rise the wall clock allows since the last one.
pub(super) fn observe_daa(daa: u64) {
    if daa == 0 {
        return;
    }
    let mut clock = shared().clock.lock().unwrap_or_else(|e| e.into_inner());
    let (accepted, anchor) = match clock.anchor {
        None => (daa, daa),
        Some((a, at)) => {
            let ms = u64::try_from(at.elapsed().as_millis()).unwrap_or(u64::MAX);
            let accepted = plausible_daa(a, ms, daa);
            (accepted, accepted.min(a.saturating_add(ms.saturating_mul(MAX_DAA_PER_SEC) / 1_000)))
        }
    };
    clock.anchor = Some((anchor, std::time::Instant::now()));
    clock.current = clock.current.max(accepted);
}

/// Highest capped DAA seen by any connection of the process; 0 before the first job.
pub(super) fn current_daa() -> u64 {
    shared().clock.lock().unwrap_or_else(|e| e.into_inner()).current
}

pub(super) fn shared() -> &'static Shared {
    static SHARED: OnceLock<Shared> = OnceLock::new();
    SHARED.get_or_init(|| Shared {
        traces: Arc::new(Mutex::new(HashMap::new())),
        queue: Arc::new(Mutex::new(Vec::new())),
        busy: Arc::new(AtomicBool::new(false)),
        link: std::sync::Mutex::new(None),
        wake: Notify::new(),
        store: OnceLock::new(),
        clock: std::sync::Mutex::new(DaaClock::default()),
    })
}

/// Opens the on-disk store next to `escrow_state_file` and restores its traces, once per process.
pub(super) async fn init_store(escrow_state_file: &str) {
    let s = shared();
    if s.store.get().is_some() {
        return;
    }
    let path = std::path::Path::new(escrow_state_file);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("escrow_state");
    let store = TraceStore::new(&path.with_file_name(format!("{}-pool.journal", stem)).to_string_lossy());
    let restored = store.load_all();
    let mut traces = s.traces.lock().await;
    for t in restored {
        let deadline_daa =
            t.opening.as_ref().map(|o| o.deadline_daa).or(t.attn.as_ref().map(|a| a.deadline_daa)).filter(|d| *d > 0);
        let mut seed = [0u8; 32];
        let phase_seed =
            t.opening.as_ref().filter(|o| hex::decode_to_slice(&o.seed_hash, &mut seed).is_ok()).map(|_| seed);
        let phases = match (phase_seed, t.attn) {
            (Some(s), Some(a)) => {
                a.bodies.into_iter().filter_map(|b| AttnOpening::decode(&b).map(|o| (key(&s, &o.path), b))).collect()
            }
            _ => HashMap::new(),
        };
        traces.insert(
            t.response_txid.clone(),
            PoolTrace {
                model_id: t.model_id,
                tier: t.tier,
                request_hash: t.request_hash,
                trace: Arc::new(t.trace),
                created_daa: t.submitted_daa,
                deadline_daa,
                accepted_daa: deadline_daa.map(|d| d.saturating_sub(AUDIT_SPAN_DAA)),
                built: HashMap::new(),
                phases,
                phase_seed,
                failures: HashMap::new(),
                response: None,
            },
        );
    }
    if !traces.is_empty() {
        log::info!("Trace: {} pool answer(s) restored for their audits", traces.len());
    }
    let _ = s.store.set(store);
}

/// Writes (or rewrites the state of) one trace.
pub(super) fn persist(key: &str, t: &PoolTrace) {
    if let Some(store) = shared().store.get() {
        store.save_queued(t.record(key));
    }
}

/// Writes one new trace and waits; blocking. False when its file is not on disk.
pub(super) fn persist_now(key: &str, t: &PoolTrace) -> bool {
    let Some(written) = queue_write(key, t) else { return true };
    let _ = written.recv();
    shared().store.get().is_none_or(|store| store.contains(key))
}

/// Queues the write of one trace, in order with the other writes; the receiver fires once it is
/// applied.
pub(super) fn queue_write(key: &str, t: &PoolTrace) -> Option<std::sync::mpsc::Receiver<()>> {
    let store = shared().store.get()?;
    let (done, written) = std::sync::mpsc::channel();
    store.save_queued_notify(t.record(key), done);
    Some(written)
}

/// Drops a trace and its file.
pub(super) fn remove(traces: &mut HashMap<String, PoolTrace>, key: &str) {
    if traces.remove(key).is_some() {
        if let Some(store) = shared().store.get() {
            store.remove(key);
        }
    }
}

/// Whether queued `d` is the opening (`txid`, `seed`, `path`) again, or an opening of the same
/// request under another response or seed, which this one supersedes.
pub(super) fn same_or_superseded(d: &PoolDraw, request_hash: &str, txid: &[u8; 32], seed: &[u8; 32], path: &[u64]) -> bool {
    (d.response_txid == *txid && d.seed == *seed && d.path == path)
        || (d.request_hash == request_hash && (d.response_txid != *txid || d.seed != *seed))
}

/// Queues `draw` again, ready in a second, unless it is queued already or superseded; blocking.
pub(super) fn requeue(draw: PoolDraw) {
    let mut queue = shared().queue.blocking_lock();
    if !queue.iter().any(|d| same_or_superseded(d, &draw.request_hash, &draw.response_txid, &draw.seed, &draw.path)) {
        let retry_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(1));
        queue.push(PoolDraw { retry_at, ..draw });
    }
}

/// Drops the traces past their keep window with no draw queued, and the draws the node no longer
/// awaits.
pub(super) async fn prune() {
    let daa = current_daa();
    if daa == 0 {
        return;
    }
    let s = shared();
    let mut queue = s.queue.lock().await;
    queue.retain(|d| d.live(daa));
    let mut traces = s.traces.lock().await;
    let gone: Vec<String> = traces
        .iter()
        .filter(|(k, t)| t.expired(daa) && !queue.iter().any(|d| &d.request_hash == *k))
        .map(|(k, _)| k.clone())
        .collect();
    for k in gone {
        remove(&mut traces, &k);
    }
    queue.retain(|d| traces.contains_key(&d.request_hash));
}

/// What a replay built for a draw.
pub(super) struct Built {
    /// Bodies answering the draw.
    pub bodies: Vec<Vec<u8>>,
    /// Attention phase bodies built on the way, by key.
    pub phases: Vec<(String, Vec<u8>)>,
    /// Trace opening bundle built on the way for a phase draw.
    pub bundle: Option<Vec<Vec<u8>>>,
}

/// Builds the bodies answering `draw`, with the attention phases built on the way by key; blocking
/// (GPU replay). A missing parent phase is rebuilt from the trace opening; a phase draw that also
/// owes trace pieces gets them from `bundle` or a fresh trace opening.
pub(super) fn build(
    model_id: &[u8; 32],
    tier: u8,
    request_hash: &[u8; 32],
    trace: &keryx_miner::trace::ResponseTrace,
    phases: &HashMap<String, Vec<u8>>,
    bundle: Option<Vec<Vec<u8>>>,
    draw: &PoolDraw,
) -> Option<Built> {
    let open = || {
        let body = keryx_miner::llm::open_trace(
            model_id,
            tier,
            trace,
            &draw.seed,
            draw.response_txid,
            request_hash,
            draw.accepted_daa,
        )?;
        keryx_miner::trace_seq::unbundle(&body)
    };
    let mut new_phases = Vec::new();
    if draw.path.is_empty() {
        let bodies = open()?;
        if let Some(p) = phase_body(&bodies) {
            new_phases.push((key(&draw.seed, &[]), p));
        }
        return Some(Built { bodies, phases: new_phases, bundle: None });
    }
    let mut bundle = bundle;
    let mut new_bundle = None;
    let n = draw.path.len();
    let known = (0..=n).rev().find_map(|i| phases.get(&key(&draw.seed, &draw.path[..i])).map(|b| (i, b.clone())));
    let (start, mut prev) = match known {
        Some((i, b)) => (i + 1, b),
        None => {
            let b = match bundle.take() {
                Some(b) => b,
                None => {
                    let b = open()?;
                    new_bundle = Some(b.clone());
                    b
                }
            };
            let p = phase_body(&b)?;
            new_phases.push((key(&draw.seed, &[]), p.clone()));
            bundle = Some(b);
            (1, p)
        }
    };
    for i in start..=n {
        let partials = AttnOpening::decode(&prev).map(|o| o.partials).filter(|p| !p.is_empty())?;
        let body = keryx_miner::llm::open_attention_phase(
            model_id,
            tier,
            trace,
            &draw.seed,
            draw.response_txid,
            &draw.path[..i],
            &partials,
        )?;
        new_phases.push((key(&draw.seed, &draw.path[..i]), body.clone()));
        prev = body;
    }
    let mut bodies = Vec::new();
    if draw.pieces & !PIECE_ATTN != 0 {
        let b = match bundle {
            Some(b) => b,
            None => {
                let b = open()?;
                new_bundle = Some(b.clone());
                b
            }
        };
        bodies.extend(b.into_iter().filter(|x| piece_bit(x) != PIECE_ATTN));
    }
    bodies.push(prev);
    Some(Built { bodies, phases: new_phases, bundle: new_bundle })
}

/// The attention body among `bodies` that carries partials for a next phase.
fn phase_body(bodies: &[Vec<u8>]) -> Option<Vec<u8>> {
    bodies.iter().find(|b| AttnOpening::decode(b).is_some_and(|o| !o.partials.is_empty())).cloned()
}

/// Bodies the node still awaits among `bodies`.
pub(super) fn owed(bodies: &[Vec<u8>], pieces: u32) -> Vec<Vec<u8>> {
    bodies.iter().filter(|b| piece_bit(b) & pieces != 0).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_bits_match_the_node() {
        assert_eq!(piece_bit(&[VERSION_ATTN, 0]), PIECE_ATTN);
        assert_eq!(piece_bit(&[VERSION_RECUR]), PIECE_RECUR);
        assert_eq!(piece_bit(&[VERSION_RESID]), PIECE_RESID);
        assert_eq!(piece_bit(&[1, 2, 3]), PIECE_TRACE);
        let bodies = vec![vec![1u8], vec![VERSION_ATTN], vec![VERSION_RESID]];
        assert_eq!(owed(&bodies, PIECE_RESID | PIECE_TRACE).len(), 2);
        assert!(owed(&bodies, 0).is_empty());
    }

    #[test]
    fn replay_failures_count_per_draw_and_stay_bounded() {
        let now = std::time::Instant::now();
        let mut f = HashMap::new();
        let (k1, k2) = (draw_key(&[1; 32], &[]), draw_key(&[1; 32], &[2]));
        assert_eq!(bump_failure(&mut f, &k1, now), 1);
        assert_eq!(bump_failure(&mut f, &k1, now), 2);
        assert_eq!(bump_failure(&mut f, &k2, now), 1);
        let later = now + std::time::Duration::from_secs(1);
        for i in 0..MAX_BUILT as u64 {
            bump_failure(&mut f, &draw_key(&[3; 32], &[i]), later);
        }
        assert_eq!(f.len(), MAX_BUILT);
        assert!(!f.contains_key(&k1) && !f.contains_key(&k2));
        assert_eq!(bump_failure(&mut f, &draw_key(&[3; 32], &[0]), later), 2);
    }

    #[test]
    fn a_given_up_draw_answers_rarely_then_retries() {
        let t0 = std::time::Instant::now();
        assert_eq!(opening_action(None, t0), OpeningAction::Replay);
        let few = Failures { count: MAX_REPLAY_ATTEMPTS - 1, last_failure: t0, last_reply: None };
        assert_eq!(opening_action(Some(&few), t0), OpeningAction::Replay);
        let out = Failures { count: MAX_REPLAY_ATTEMPTS, last_failure: t0, last_reply: None };
        assert_eq!(opening_action(Some(&out), t0), OpeningAction::Reply);
        let replied = Failures { last_reply: Some(t0), ..out };
        assert_eq!(opening_action(Some(&replied), t0 + GIVE_UP_REPLY_EVERY / 2), OpeningAction::Ignore);
        assert_eq!(opening_action(Some(&replied), t0 + GIVE_UP_REPLY_EVERY), OpeningAction::Reply);
        assert_eq!(opening_action(Some(&replied), t0 + GIVE_UP_RESET), OpeningAction::Replay);
    }

    #[test]
    fn requeue_dedup_keeps_other_paths() {
        let queued = PoolDraw::new("r".into(), [1; 32], [1; 32], 10, 5, PIECES_ALL, vec![2]);
        assert!(!same_or_superseded(&queued, "r", &[1; 32], &[1; 32], &[]));
        assert!(same_or_superseded(&queued, "r", &[1; 32], &[1; 32], &[2]));
    }

    #[test]
    fn traces_expire_on_daa() {
        assert!(!expired(1_000, None, None, 1_000 + UNDRAWN_KEEP_DAA));
        assert!(expired(1_000, None, None, 1_001 + UNDRAWN_KEEP_DAA));
        let end = audit_end(4_000);
        assert!(!expired(1_000, Some(5_000), Some(4_000), end));
        assert!(expired(1_000, Some(5_000), Some(4_000), end + 1));
        assert!(!expired(1_000, Some(5_000), None, audit_end(5_000)));
    }

    #[test]
    fn draw_bounds() {
        assert_eq!(draw_bounds_error(1_000, 2_000, PIECE_TRACE, MAX_DRAW_PATH + 1), Some("path too long"));
        assert_eq!(draw_bounds_error(1_000, 0, 0, 0), None);
        assert_eq!(draw_bounds_error(1_000, 2_000, 16, 0), Some("no known piece"));
        assert_eq!(draw_bounds_error(1_000, 1_000, PIECE_TRACE, 0), Some("deadline out of range"));
        assert_eq!(draw_bounds_error(1_000, u64::MAX, PIECE_TRACE, 0), Some("deadline out of range"));
        assert_eq!(draw_bounds_error(1_000, 1_000 + MAX_DEADLINE_AHEAD_DAA, PIECE_ATTN, 3), None);
        assert_eq!(draw_bounds_error(0, u64::MAX, PIECE_TRACE, 0), Some("no job yet"));
    }

    #[test]
    fn failed_replays_back_off() {
        let d = PoolDraw::new(String::new(), [0; 32], [0; 32], 10, 5, 1, vec![]);
        assert!(d.ready());
        let once = d.retry();
        assert!(!once.ready());
        let wait = |d: &PoolDraw| d.retry_at.unwrap().saturating_duration_since(std::time::Instant::now());
        assert!(wait(&once) <= REPLAY_RETRY_BASE && wait(&once) > REPLAY_RETRY_BASE / 2);
        let mut d = once;
        for _ in 0..8 {
            d = d.retry();
        }
        assert!(wait(&d) <= REPLAY_RETRY_MAX && wait(&d) > REPLAY_RETRY_MAX / 2);
    }

    #[test]
    fn prune_daa_follows_the_clock() {
        assert_eq!(plausible_daa(1_000, 0, 1_500), 1_500);
        assert_eq!(plausible_daa(1_000, 0, u64::MAX), 1_000 + DAA_SLACK);
        assert_eq!(plausible_daa(1_000, 100_000, u64::MAX), 1_000 + 100 * MAX_DAA_PER_SEC + DAA_SLACK);
        assert_eq!(plausible_daa(1_000, 500, u64::MAX), 1_000 + MAX_DAA_PER_SEC / 2 + DAA_SLACK);
    }
}
