use futures::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

mod ai;
mod audit;
mod protocol;
mod statum_codec;

use ai::{AiOpening, AiRequest, InferenceGuard};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use protocol::{difficulty_to_target, effective_target, nonce_range, MiningJob};

use crate::client::stratum::statum_codec::{AiRequestParams, ErrorCode, MiningSubmit, NewLineJsonCodecError, StratumLine};
use crate::client::stratum::statum_codec::{
    MiningSubscribe, SetExtranonce, StratumCommand, StratumError, StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::PartialBlock;
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{debug, error, info, warn};
use statum_codec::NewLineJsonCodec;
use std::sync::OnceLock;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;

const KERYX_STRATUM_DAA_CAPABILITY: &str = "keryx-stratum-v3";
const LOG_RATE: Duration = Duration::from_secs(30);
const CHALLENGE_MAX_TOKENS: usize = 128;
/// Reply ids tracked at once for AI responses and openings.
const MAX_PENDING_IDS: usize = 64;
/// Longest a pool may go without sending a job before the connection is renewed.
const JOB_TIMEOUT: Duration = Duration::from_secs(300);
/// Longest interval between two model declarations.
const REDECLARE_EVERY: Duration = Duration::from_secs(90);
/// AI requests a worker holds while its GPU is busy.
const MAX_QUEUED_REQUESTS: usize = 64;
/// DAA after which a queued AI request is dropped.
const REQUEST_QUEUE_TTL_DAA: u64 = audit::LEDGER_HORIZON_DAA;

// ── Phase 2 inference — inference cache & task types ─────────────────────────────

/// AiRequest task dispatched by the bridge in a `mining.notify` 5th parameter (JSON).
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
struct AiTask {
    #[serde(default)]
    stable_id: String,
    model_id_hex: String,
    prompt: String,
    max_tokens: usize,
    #[serde(default)]
    inference_reward: u64,
    #[serde(default)]
    request_hash: String,
}

/// Task attached to the current mining job, cleared on each new `mining.notify`.
struct CurrentTask {
    job_id: String,
    task: AiTask,
}

/// Max cached inference results — evict when full to prevent unbounded growth.
const MAX_INFERENCE_CACHE_SIZE: usize = 1_000;

/// Shared inference result cache — persists across block changes so that if the
/// same AiRequest is included in multiple consecutive job templates the miner can
/// immediately submit with a CID once inference completed for the first occurrence.
struct InferenceCacheInner {
    /// stable_id → base58 CIDv0 string returned by IPFS after upload.
    results: HashMap<String, String>,
    /// stable_ids currently being inferred (guards against duplicate spawn_blocking calls).
    in_progress: HashSet<String>,
}

type InferenceCache = Arc<Mutex<InferenceCacheInner>>;

type BlockHandle = JoinHandle<()>;

/// Capability a pool announces in its subscribe result to take PoM proofs in base64.
const PROOF_B64_CAPABILITY: (&str, &str) = ("keryx.proof_encoding", "b64");

/// Pool addresses that refused a base64 proof; they get hex for the rest of the process.
static HEX_ONLY_POOLS: std::sync::Mutex<Option<HashSet<String>>> = std::sync::Mutex::new(None);

/// How share proofs are encoded on one connection.
#[derive(Default)]
struct ProofEncoding {
    /// The pool takes base64 proofs.
    b64: AtomicBool,
    /// Ids of the shares sent with a base64 proof, awaiting the pool's answer.
    sent_b64: std::sync::Mutex<HashSet<u32>>,
}

impl ProofEncoding {
    /// The proof field of share `id`: `b64:` and base64 when the pool takes it, else hex.
    fn encode(&self, id: u32, proof: &[u8]) -> String {
        if !self.b64.load(Ordering::Acquire) {
            return hex::encode(proof);
        }
        let mut sent = self.sent_b64.lock().unwrap_or_else(|e| e.into_inner());
        if sent.len() >= 1024 {
            sent.clear();
        }
        sent.insert(id);
        format!("b64:{}", BASE64.encode(proof))
    }

    /// Whether share `id` went out in base64; forgets it.
    fn was_b64(&self, id: u32) -> bool {
        self.sent_b64.lock().unwrap_or_else(|e| e.into_inner()).remove(&id)
    }
}

/// Whether `subscriptions` announce base64 proofs.
fn announces_b64(subscriptions: &[(String, String)]) -> bool {
    subscriptions.iter().any(|(k, v)| k == PROOF_B64_CAPABILITY.0 && v == PROOF_B64_CAPABILITY.1)
}
type PendingShares = Arc<Mutex<HashMap<u32, String>>>;

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
}

static SHARE_STATS: OnceLock<Arc<ShareStats>> = OnceLock::new();

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shares: {}{}{}{}",
            match self.accepted.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Accepted: {} ", v),
            },
            match self.stale.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Stale: {} ", v),
            },
            match self.low_diff.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Low difficulty: {} ", v),
            },
            match self.duplicate.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Duplicate: {} ", v),
            },
        )
    }
}

#[allow(dead_code)]
pub struct StratumHandler {
    log_handler: JoinHandle<()>,
    write_handle: JoinHandle<Result<(), NewLineJsonCodecError>>,

    //client: Framed<TcpStream, NewLineJsonCodec>,
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>>,
    miner_address: String,
    mine_when_not_synced: bool,

    target_pool: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,

    shares_stats: Arc<ShareStats>,
    /// AI responses awaiting the pool's ack, by id: when each was sent and its trace key.
    ai_response: Arc<Mutex<HashMap<u32, (Instant, Option<String>)>>>,
    shares_pending: PendingShares,
    keepalive_interval: Duration,
    keepalive_timeout: Duration,
    keepalive_pending: Option<(u32, Instant)>,
    last_activity: Instant,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// IPFS Kubo API URL for uploading inference results (e.g. "http://127.0.0.1:5001").
    ipfs_url: String,
    /// Task dispatched by the bridge for the current mining job (None = no AiRequest in job).
    current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
    /// Completed inferences: stable_id → base58 CIDv0 string (persists across block changes).
    inference_cache: InferenceCache,
    /// True while the GPU runs a request, a challenge or a replay.
    challenge_in_flight: Arc<AtomicBool>,
    /// Traced answers of this worker by request hash, kept for their audit draws.
    pool_traces: Arc<Mutex<HashMap<String, audit::PoolTrace>>>,
    /// Draws relayed by the pool, waiting for the GPU.
    replay_queue: Arc<Mutex<Vec<audit::PoolDraw>>>,
    /// Queued AI requests with the DAA they were queued at.
    request_queue: std::collections::VecDeque<(AiRequest, u64)>,
    /// Capability challenge waiting for the GPU.
    pending_challenge: Option<(String, String)>,
    /// Ids of the `mining.ai_opening` lines awaiting the pool's answer.
    openings_pending: Arc<Mutex<HashSet<u32>>>,
    /// DAA score of the last pool job.
    last_daa: u64,
    /// Models last declared to the pool, and when.
    declared: Option<(Vec<String>, Instant)>,
    /// Ids of the `mining.hello` and `mining.telemetry` lines; their replies are never acted upon.
    info_ids: HashSet<u32>,
    /// When the last job arrived (or the connection opened).
    last_job: Instant,
    /// Password sent with `mining.authorize`.
    pool_password: String,
    /// Address of the pool, to remember a refusal of base64 proofs.
    pool_address: String,
    proof_encoding: Arc<ProofEncoding>,
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    async fn register(&mut self) -> Result<(), Error> {
        let mut id = { Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst)) };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
                    MiningSubscribe::MiningSubscribeOptions((
                        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                        KERYX_STRATUM_DAA_CAPABILITY.into(),
                    )),
                )),
                jsonrpc: None,
                error: None,
            })
            .await?;
        id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));

        let pay_address = self.miner_address.clone();
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    pay_address.clone(),
                    self.pool_password.clone(),
                ))),
                jsonrpc: None,
                error: None,
            })
            .await?;

        self.declared = None;
        self.declare_models().await?;
        let ready = traceable_model();
        let model = match ready {
            Some(id) => serde_json::json!({
                "loaded": true,
                "tier": keryx_miner::models::pom_tier_index(&id, u64::MAX),
                "model_id": hex::encode(id),
            }),
            None => serde_json::json!({ "loaded": false }),
        };
        let hello = serde_json::json!({
            "v": 1,
            "miner": { "name": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION"), "build": "cuda" },
            "trace": u8::from(ready.is_some()),
            "model": model,
        });
        let hello_id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        note_pending(&mut self.info_ids, hello_id);
        self.send_channel
            .send(StratumLine {
                id: Some(hello_id),
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningHello(vec![hello])),
                jsonrpc: None,
                error: None,
            })
            .await?;
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for Stratum traffic");
        let mut maintenance = tokio::time::interval(Duration::from_secs(1));
        maintenance.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                result = &mut self.write_handle => {
                    result??;
                    return Err("Stratum writer closed".into());
                },
                message = self.stream.try_next() => match message? {
                    Some(msg) => {
                        self.last_activity = Instant::now();
                        self.handle_message(msg, miner).await?;
                    }
                    None => return Err("Stratum connection closed".into()),
                },
                _ = maintenance.tick() => {
                    self.maintain_keepalive().await?;
                    self.pump(miner).await;
                }
                _ = audit::shared().wake.notified() => self.pump(miner).await,
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

/// Whether `url` names a stratum pool (`stratum+tcp://`, `stratum+ssl://` or `stratum+tls://`).
pub fn is_stratum_url(url: &str) -> bool {
    ["stratum+tcp://", "stratum+ssl://", "stratum+tls://"].iter().any(|p| url.starts_with(p))
}

/// TLS options of a stratum connection.
pub struct TlsSettings {
    /// Extra PEM certificate authority trusted besides the public roots.
    pub ca_file: Option<String>,
}

trait PoolIo: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> PoolIo for T {}

async fn tls_wrap(socket: TcpStream, address: &str, settings: &TlsSettings) -> Result<Box<dyn PoolIo>, Error> {
    use tokio_rustls::rustls::{self, pki_types::pem::PemObject, pki_types::CertificateDer, pki_types::ServerName, RootCertStore};

    let mut roots = RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    if let Some(path) = &settings.ca_file {
        let mut added = 0;
        for cert in CertificateDer::pem_file_iter(path).map_err(|e| format!("pool CA file {}: {}", path, e))? {
            roots.add(cert.map_err(|e| format!("pool CA file {}: {}", path, e))?)?;
            added += 1;
        }
        if added == 0 {
            return Err(format!("pool CA file {}: no certificate", path).into());
        }
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let host = address.rsplit_once(':').map_or(address, |(h, _)| h).trim_start_matches('[').trim_end_matches(']');
    let name = ServerName::try_from(host.to_string()).map_err(|e| format!("pool host {}: {}", host, e))?;
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let stream = tokio::time::timeout(Duration::from_secs(10), connector.connect(name, socket)).await??;
    Ok(Box::new(stream))
}

impl StratumHandler {
    pub async fn connect(
        address: String,
        tls: Option<TlsSettings>,
        miner_address: String,
        mine_when_not_synced: bool,
        ipfs_url: String,
        keepalive_seconds: u64,
        keepalive_timeout_seconds: u64,
        escrow_state_file: String,
        pool_password: String,
    ) -> Result<Box<Self>, Error> {
        audit::init_store(&escrow_state_file).await;
        info!("Connecting to {}", address);
        let pool_address = address.clone();
        let proof_encoding = Arc::new(ProofEncoding::default());
        let tcp = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(&address)).await??;
        tcp.set_nodelay(true)?;
        let socket: Box<dyn PoolIo> = match &tls {
            Some(settings) => {
                let stream = tls_wrap(tcp, &address, settings).await?;
                info!("TLS session established with {}", address);
                stream
            }
            None => Box::new(tcp),
        };

        let client = Framed::new(socket, NewLineJsonCodec::new());
        let (send_channel, recv) = mpsc::channel::<StratumLine>(3);
        let (sink, stream) = client.split();
        let write_handle = tokio::spawn(async move { ReceiverStream::new(recv).map(Ok).forward(sink).await });

        let share_state = SHARE_STATS.get_or_init(|| Arc::new(ShareStats::default())).clone();
        let shares_pending = Arc::new(Mutex::new(HashMap::new()));
        let last_stratum_id = Arc::new(AtomicU32::new(0));
        let current_task_slot: Arc<Mutex<Option<CurrentTask>>> = Arc::new(Mutex::new(None));
        let inference_cache: InferenceCache =
            Arc::new(Mutex::new(InferenceCacheInner { results: HashMap::new(), in_progress: HashSet::new() }));
        let (block_channel, block_handle) = Self::create_block_channel(
            send_channel.clone(),
            miner_address.clone(),
            last_stratum_id.clone(),
            shares_pending.clone(),
            Arc::clone(&current_task_slot),
            Arc::clone(&inference_cache),
            Arc::clone(&proof_encoding),
        );
        let shared = audit::shared();
        let openings_pending = Arc::new(Mutex::new(HashSet::new()));
        *shared.link.lock().unwrap_or_else(|e| e.into_inner()) = Some(audit::Link {
            sender: send_channel.clone(),
            last_id: last_stratum_id.clone(),
            openings_pending: Arc::clone(&openings_pending),
        });
        Ok(Box::new(Self {
            log_handler: task::spawn(Self::log_shares(share_state.clone(), shares_pending.clone())),
            write_handle,
            stream: Box::pin(stream),
            send_channel,
            miner_address,
            mine_when_not_synced,
            target_pool: Default::default(),
            nonce_mask: u64::MAX, // full nonce space until set_extranonce assigns a sub-range
            nonce_fixed: 0,
            extranonce: None,
            last_stratum_id,
            shares_stats: share_state,
            ai_response: Arc::new(Mutex::new(HashMap::new())),
            shares_pending,
            keepalive_interval: Duration::from_secs(keepalive_seconds.max(1)),
            keepalive_timeout: Duration::from_secs(keepalive_timeout_seconds.max(1)),
            keepalive_pending: None,
            last_activity: Instant::now(),
            block_channel,
            block_handle,
            ipfs_url,
            current_task_slot,
            inference_cache,
            challenge_in_flight: Arc::clone(&shared.busy),
            pool_traces: Arc::clone(&shared.traces),
            replay_queue: Arc::clone(&shared.queue),
            request_queue: std::collections::VecDeque::new(),
            pending_challenge: None,
            openings_pending,
            last_daa: 0,
            declared: None,
            info_ids: HashSet::new(),
            last_job: Instant::now(),
            pool_password,
            pool_address,
            proof_encoding,
        }))
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        miner_address: String,
        last_stratum_id: Arc<AtomicU32>,
        shares_pending: PendingShares,
        current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
        inference_cache: InferenceCache,
        proof_encoding: Arc<ProofEncoding>,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);

        let handle = tokio::spawn(async move {
            let mut recv_stream = ReceiverStream::new(recv);
            while let Some(seed) = recv_stream.next().await {
                let (nonce, job_id, pom_proof, task_root) = match seed {
                    BlockSeed::PartialBlock { nonce, id, pom_proof, task_root, .. } => (nonce, id, pom_proof, task_root),
                    BlockSeed::FullBlock { .. } => unreachable!(),
                };
                let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                shares_pending.lock().await.insert(msg_id, job_id.clone());
                let nonce_hex = format!("{:016x}", nonce);
                let opoi_tag = keryx_inference::tag_fixed(nonce);

                // Phase 2: check inference cache for the current job's task
                let cid_opt = {
                    let task_guard = current_task_slot.lock().await;
                    if let Some(ref ct) = *task_guard {
                        if ct.job_id == job_id && !ct.task.stable_id.is_empty() {
                            let cache_guard = inference_cache.lock().await;
                            cache_guard.results.get(&ct.task.stable_id).cloned()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                let line = if let (false, Some(root)) = (pom_proof.is_empty(), task_root) {
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithPomTask((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid_opt.unwrap_or_default(),
                                proof_encoding.encode(msg_id, &pom_proof),
                                hex::encode(root),
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else if !pom_proof.is_empty() {
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithPom((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid_opt.unwrap_or_default(),
                                proof_encoding.encode(msg_id, &pom_proof),
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else if let Some(cid) = cid_opt {
                    info!("Inference: submitting share with CID for job {}", job_id);
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithCID((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else {
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithTag((miner_address.clone(), job_id, nonce_hex, opoi_tag)),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                };

                if send_channel.send(line).await.is_err() {
                    shares_pending.lock().await.remove(&msg_id);
                    break;
                }
            }
        });
        (send, handle)
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        let result_id = msg.id.filter(|_| matches!(msg.payload, StratumLinePayload::StratumResult { .. }));
        if let Some(id) = result_id {
            if self.openings_pending.lock().await.remove(&id) {
                match &msg.error {
                    None => debug!("Trace: opening accepted by the pool"),
                    Some(e) => warn!("Trace: opening refused by the pool: {:?}", e),
                }
                return Ok(());
            }
        }
        let acked = match result_id {
            Some(id) => self.ai_response.lock().await.remove(&id),
            None => None,
        };
        if let Some((_, traced)) = acked {
            match (&msg.payload, &msg.error) {
                (StratumLinePayload::StratumResult { result: StratumResult::Plain(Some(true)) }, None) => {
                    info!("AI response accepted")
                }
                _ => {
                    warn!("AI response rejected: {:?}", msg.error);
                    if let Some(key) = traced.filter(|_| msg.error.is_some()) {
                        audit::remove(&mut *self.pool_traces.lock().await, &key);
                    }
                }
            }
            self.pump(miner).await;
            return Ok(());
        }
        if result_id.is_some_and(|id| self.info_ids.remove(&id)) {
            debug!("Stratum: pool reply to hello/telemetry ignored: {:?}", msg.error);
            return Ok(());
        }
        if result_id.is_some() && self.keepalive_pending.map(|(id, _)| Some(id)) == Some(result_id) {
            // Any answer proves the connection alive, a refusal of the method included.
            self.keepalive_pending = None;
            return Ok(());
        }
        if msg.id.is_none() && matches!(msg.payload, StratumLinePayload::StratumResult { .. }) {
            match &msg.error {
                None => debug!("Stratum: id-less result ignored"),
                Some(e) => warn!("Stratum: id-less error from the pool ignored: {:?}", e),
            }
            return Ok(());
        }
        match msg.clone() {
            StratumLine { id, payload, error: None, .. } => {
                match payload {
                    StratumLinePayload::StratumResult { result } if id.is_some() => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                self.proof_encoding.was_b64(id.expect("We checked id is not none"));
                                if let Some(_jobid) = self
                                    .shares_pending
                                    .lock()
                                    .await
                                    .remove(&id.expect("We checked id is not none"))
                                {
                                    self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                    info!("Share accepted");
                                } else {
                                    debug!("Stratum result for a non-share request: {:?}", msg);
                                }
                                Ok(())
                            }
                            StratumResult::Subscribe((ref subscriptions, ref extranonce, ref nonce_size)) => {
                                let hex_only = HEX_ONLY_POOLS
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .as_ref()
                                    .is_some_and(|p| p.contains(&self.pool_address));
                                if announces_b64(subscriptions) && !hex_only {
                                    info!("Stratum: the pool takes base64 proofs");
                                    self.proof_encoding.b64.store(true, Ordering::Release);
                                }
                                self.set_extranonce(extranonce.as_str(), nonce_size)
                                /*for (name, value) in _subscriptions {
                                    match name.as_str() {
                                        "mining.set_difficulty" => {self.set_difficulty(&f32::from_str(value.as_str())?)?;},
                                        _ => {warn!("Ignored {} (={})", name, value);}
                                    }
                                }
                                Ok(())*/
                            }
                            _ => {
                                let id = id.expect("We checked id is not none");
                                if let Some(jobid) = self.shares_pending.lock().await.remove(&id) {
                                    self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                                    warn!("Share rejected by pool (Job id: {:?})", jobid);
                                    if self.proof_encoding.was_b64(id) {
                                        self.fall_back_to_hex();
                                    }
                                } else {
                                    debug!("Stratum: negative result for request {} ignored", id);
                                }
                                Ok(())
                            }
                        }
                    }
                    StratumLinePayload::StratumCommand(command) => match command {
                        StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((
                            ref extranonce,
                            ref nonce_size,
                        ))) => self.set_extranonce(extranonce.as_str(), nonce_size),
                        StratumCommand::MiningSetDifficulty((ref difficulty,)) => self.set_difficulty(difficulty),
                        StratumCommand::MiningNotify(notify) => {
                            let job = MiningJob::try_from(notify)?;
                            self.last_job = Instant::now();
                            audit::observe_daa(job.daa_score);
                            self.last_daa = job.daa_score;
                            self.declare_models().await?;
                            self.pump(miner).await;
                            if keryx_miner::llm::loaded_model_ids().is_empty() {
                                return miner.process_block(None).await;
                            }
                            let target = effective_target(self.target_pool, job.block_bits)?;
                            // A whole-rig pause holds the job; a one-device pause lets the others mine.
                            if self.challenge_in_flight.load(Ordering::SeqCst) && keryx_miner::pom_gpu::inference_paused_all() {
                                return miner.process_block(None).await;
                            }
                            if let Some(task_json) = job.task_json {
                                // The task path publishes plaintext answers to IPFS; from the
                                // private-inference gate on only the pool can build an answer.
                                if job.daa_score >= keryx_miner::pom::private_inference_activation_daa() {
                                    warn!("Inference: pool still dispatches inference tasks; private inference needs a v3 pool (mining.ai_request) — task ignored");
                                    *self.current_task_slot.lock().await = None;
                                } else if self.handle_ai_task(job.id.clone(), task_json, miner).await {
                                    return Ok(());
                                }
                            } else {
                                *self.current_task_slot.lock().await = None;
                            }
                            if keryx_miner::block_task::active(job.daa_score) {
                                let window = keryx_verifiable_inference::block_task::window(job.daa_score);
                                let parents = job.header.as_ref().and_then(|h| h.parents.first()).map(|p| keryx_miner::block_task::parents_key(&p.parent_hashes));
                                match (window, parents, job.window_seed) {
                                    (Some(w), Some(p), Some(seed)) => keryx_miner::block_task::set_seed(w, &p, seed),
                                    _ => {
                                        keryx_miner::pom_gpu::log_throttled(u32::MAX, "pool-no-task", || {
                                            warn!("Block task: the pool sends jobs without a block task (header and window seed) — no block can be valid, job skipped")
                                        });
                                        return miner.process_block(None).await;
                                    }
                                }
                            }
                            miner
                                .process_block(Some(PartialBlock {
                                    id: job.id,
                                    header_hash: job.header_hash,
                                    timestamp: job.timestamp,
                                    daa_score: job.daa_score,
                                    nonce: 0,
                                    target,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                    pom_proof: Vec::new(),
                                    header: job.header.map(Box::new),
                                    task_root: None,
                                }))
                                .await
                        }
                        StratumCommand::MiningAiRequest(fields) => {
                            self.handle_ai_request(id, fields).await;
                            self.pump(miner).await;
                            Ok(())
                        }
                        StratumCommand::MiningChallenge((model_id_hex, nonce_hex)) => {
                            if self.pending_challenge.replace((model_id_hex, nonce_hex)).is_some() {
                                warn!("Inference challenge: superseded by a newer challenge");
                            }
                            self.pump(miner).await;
                            Ok(())
                        }
                        StratumCommand::MalformedAi => {
                            if let Some(id) = id {
                                let _ = self.send_channel.send(StratumLine::error_reply(id, ErrorCode::Unknown, "invalid params")).await;
                            }
                            Ok(())
                        }
                        StratumCommand::MiningInferenceRequest(request) => {
                            let req_id = request.get("reqId").cloned().unwrap_or_else(|| serde_json::json!(""));
                            let reply = serde_json::json!({ "reqId": req_id, "ok": false, "error": "busy" });
                            let _ = self
                                .send_channel
                                .send(StratumLine {
                                    id: None,
                                    payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(reply)),
                                    jsonrpc: Some("2.0".into()),
                                    error: None,
                                })
                                .await;
                            Ok(())
                        }
                        StratumCommand::MiningAiOpening(params) => {
                            self.handle_ai_opening(id, params).await;
                            self.pump(miner).await;
                            Ok(())
                        }
                        StratumCommand::Subscribe(_)
                        | StratumCommand::Authorize(_)
                        | StratumCommand::MiningKeepalive(_)
                        | StratumCommand::MiningSubmit(_)
                        | StratumCommand::MiningDeclareCapabilities(_)
                        | StratumCommand::MiningAiResponse(_)
                        | StratumCommand::MiningAiOpeningResult(_)
                        | StratumCommand::MiningHello(_)
                        | StratumCommand::MiningTelemetry(_)
                        | StratumCommand::MiningInferenceResult(_)
                        | StratumCommand::MiningChallengeResponse(_) => {
                            debug!("Stratum: worker-side command echoed by the pool ignored");
                            Ok(())
                        }
                        _ => Err(format!("Unexpected stratum message: {:?}", msg).into()),
                    },
                    _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                }
            }
            StratumLine {
                id: Some(id),
                payload: StratumLinePayload::StratumResult { .. },
                error: Some(StratumError(code, error, _)),
                ..
            } => {
                let Some(jobid) = self.shares_pending.lock().await.remove(&id) else {
                    if matches!(code, ErrorCode::Unauthorized | ErrorCode::NotSubscribed) {
                        return Err(format!("Stratum request {} rejected: {}", id, error).into());
                    }
                    warn!("Stratum: error for request {} ignored: {}", id, error);
                    return Ok(());
                };
                if self.proof_encoding.was_b64(id) {
                    // Any refusal of a base64 proof means hex from now on, on a kept connection.
                    self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                    warn!("Share rejected by pool (Job id: {:?}): {}", jobid, error);
                    self.fall_back_to_hex();
                    return Ok(());
                }
                match code {
                    ErrorCode::Unknown | ErrorCode::Busy => {
                        // A rejected share/block is logged, never fatal.
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Share rejected by pool (Job id: {:?}): {}", jobid, error);
                        Ok(())
                    }
                    ErrorCode::JobNotFound => {
                        self.shares_stats.stale.fetch_add(1, Ordering::SeqCst);
                        warn!("Stale share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::DuplicateShare => {
                        self.shares_stats.duplicate.fetch_add(1, Ordering::SeqCst);
                        warn!("Duplicate share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::LowDifficultyShare => {
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Low difficulty share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::Unauthorized => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                    ErrorCode::NotSubscribed => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                }
            }
            _ => Err(format!("Unhandled stratum response: {:?}", msg).into()),
        }
    }

    /// Declares the loaded models when they changed, when none were declared on this connection,
    /// or every `REDECLARE_EVERY`; an empty list clears what the pool holds for this worker.
    async fn declare_models(&mut self) -> Result<(), Error> {
        let mut ids: Vec<String> = keryx_miner::llm::loaded_model_ids().into_iter().map(hex::encode).collect();
        ids.sort();
        let changed = self.declared.as_ref().is_none_or(|(prev, _)| *prev != ids);
        if !changed && self.declared.as_ref().is_some_and(|(_, at)| at.elapsed() < REDECLARE_EVERY) {
            return Ok(());
        }
        if changed {
            info!("Inference: declaring {} model(s) to the pool", ids.len());
        }
        self.send_channel
            .send(StratumLine {
                id: None,
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(ids.clone())),
                jsonrpc: Some("2.0".into()),
                error: None,
            })
            .await?;
        if self.declared.is_some() {
            let telemetry = serde_json::json!({ "v": 1, "trace": u8::from(traceable_model().is_some()), "models": ids });
            let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
            note_pending(&mut self.info_ids, id);
            self.send_channel
                .send(StratumLine {
                    id: Some(id),
                    payload: StratumLinePayload::StratumCommand(StratumCommand::MiningTelemetry(vec![telemetry])),
                    jsonrpc: None,
                    error: None,
                })
                .await?;
        }
        self.declared = Some((ids, Instant::now()));
        Ok(())
    }

    /// Sends hex proofs from now on, on this connection and the next ones to this pool.
    fn fall_back_to_hex(&self) {
        if self.proof_encoding.b64.swap(false, Ordering::AcqRel) {
            warn!("Stratum: the pool refused a base64 proof — hex proofs from now on");
        }
        HEX_ONLY_POOLS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(self.pool_address.clone());
    }

    async fn maintain_keepalive(&mut self) -> Result<(), Error> {
        let now = Instant::now();
        if now.duration_since(self.last_job) >= JOB_TIMEOUT {
            return Err(format!("No job from the pool for {} s", JOB_TIMEOUT.as_secs()).into());
        }
        self.ai_response.lock().await.retain(|_, (since, _)| now.duration_since(*since) < self.keepalive_timeout);
        if let Some((_, since)) = self.keepalive_pending {
            if now.duration_since(since) >= self.keepalive_timeout {
                return Err("Stratum keepalive timed out".into());
            }
        } else if now.duration_since(self.last_activity) >= self.keepalive_interval {
            let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
            self.send_channel
                .send(StratumLine {
                    id: Some(id),
                    payload: StratumLinePayload::StratumCommand(StratumCommand::MiningKeepalive([])),
                    jsonrpc: Some("2.0".into()),
                    error: None,
                })
                .await?;
            self.keepalive_pending = Some((id, now));
        }
        Ok(())
    }

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        self.target_pool = difficulty_to_target(*difficulty)?;
        info!("Difficulty: {:?}, Target: 0x{}", difficulty, hex::encode(self.target_pool.to_be_bytes()));
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        let (mask, fixed) = nonce_range(extranonce, *nonce_size)?;
        self.extranonce = Some(extranonce.to_string());
        self.nonce_mask = mask;
        self.nonce_fixed = fixed;
        Ok(())
    }

    async fn log_shares(shares_info: Arc<ShareStats>, shares_pending: PendingShares) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut _last_instant = ticker.tick().await;
        loop {
            let _now = ticker.tick().await;
            let pending_count = shares_pending.lock().await.len();
            info!("{}Pending: {}", shares_info, pending_count)
        }
    }

    /// Queues an AI request for the GPU.
    async fn handle_ai_request(&mut self, id: Option<u32>, params: AiRequestParams) {
        let mut request = match AiRequest::parse(params) {
            Ok(request) => request,
            Err(error) => {
                warn!("Invalid AI request: {}", error);
                if let Some(id) = id {
                    let _ = self.send_channel.try_send(StratumLine::error_reply(id, ErrorCode::Unknown, "invalid request"));
                }
                return;
            }
        };
        request.reply_id = id;
        if !keryx_miner::llm::is_model_ready(&request.model_id) {
            self.refuse(&request, "model is not ready");
            return;
        }
        if self.resend_answer(&request).await {
            return;
        }
        if let Some(why) = self.early_refusal(&request).await {
            self.refuse(&request, why);
            return;
        }
        if self.request_queue.len() >= MAX_QUEUED_REQUESTS {
            self.refuse(&request, "worker queue is full");
            return;
        }
        self.request_queue.push_back((request, audit::current_daa()));
    }

    /// Resends the answer kept for `request` when this worker answered it already; true if sent.
    async fn resend_answer(&self, request: &AiRequest) -> bool {
        let key = request.request_hash.to_ascii_lowercase();
        let Some(answer) = self.pool_traces.lock().await.get(&key).and_then(|t| t.response.clone()) else { return false };
        let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        let Ok(line) = request.response(id, self.miner_address.clone(), &answer.text, answer.traced) else { return false };
        {
            let mut pending = self.ai_response.lock().await;
            if pending.len() >= MAX_PENDING_IDS {
                pending.clear();
            }
            pending.insert(id, (Instant::now(), None));
        }
        info!("AI request {}: answered already — same answer resent", request.task_id);
        self.send_channel.send(line).await.is_ok()
    }

    /// Why `request` cannot be served whatever the GPU state, checked before it waits in the queue.
    async fn early_refusal(&self, request: &AiRequest) -> Option<&'static str> {
        let t = request.trace?;
        if !t.chain {
            return Some("untraced-chain answers are not served");
        }
        if keryx_miner::models::pom_tier_index(&request.model_id, u64::MAX) != Some(t.tier) {
            return Some("trace tier does not match the model");
        }
        let traces = self.pool_traces.lock().await;
        if traces.contains_key(&request.request_hash.to_ascii_lowercase()) {
            return Some("already answered by this worker");
        }
        (traces.len() >= audit::MAX_TRACES).then_some("too many answers still under audit")
    }

    /// Tells the pool `request` will not be answered.
    fn refuse(&self, request: &AiRequest, why: &str) {
        warn!("AI request {}: {} — request refused", request.task_id, why);
        if let Some(line) = request.refusal(why) {
            let _ = self.send_channel.try_send(line);
        }
    }

    /// Sends the openings already built, then starts the next GPU work when the GPU is free: a
    /// capability challenge, then an audit replay, then an AI request.
    async fn pump(&mut self, miner: &mut MinerManager) {
        let worker = self.miner_address.clone();
        for (draw, owed) in audit::take_cached().await {
            let worker = worker.clone();
            tokio::task::spawn_blocking(move || {
                if !send_openings(&worker, &draw, &owed) {
                    audit::requeue(draw);
                }
            });
        }
        if self.challenge_in_flight.load(Ordering::SeqCst) {
            return;
        }
        if let Some((model_id_hex, nonce_hex)) = self.pending_challenge.take() {
            let daa = self.last_daa.max(audit::current_daa());
            if daa == 0 {
                self.pending_challenge = Some((model_id_hex, nonce_hex));
                return;
            }
            if daa < keryx_miner::pom::pom_activation_daa() {
                self.start_challenge(model_id_hex, nonce_hex, miner).await;
                return;
            }
            keryx_miner::pom_gpu::log_throttled(u32::MAX, "pool-challenge", || {
                warn!("Inference challenge: ignored under PoM")
            });
        }
        audit::prune().await;
        if !keryx_miner::llm::probe_in_flight() && self.start_replay(miner).await {
            return;
        }
        let now = audit::current_daa();
        for (_, at) in self.request_queue.iter_mut().filter(|(_, at)| *at == 0) {
            *at = now;
        }
        let (fresh, stale): (Vec<_>, Vec<_>) =
            self.request_queue.drain(..).partition(|(_, at)| now <= at.saturating_add(REQUEST_QUEUE_TTL_DAA));
        self.request_queue = fresh.into();
        for (r, _) in stale {
            self.refuse(&r, "waited too long for the GPU");
        }
        if self.request_queue.is_empty() || self.last_daa == 0 || keryx_miner::llm::probe_in_flight() {
            return;
        }
        if let Some((request, at)) = self.request_queue.pop_front() {
            self.start_ai_request(request, at, miner).await;
        }
    }

    async fn start_ai_request(&mut self, request: AiRequest, queued_at: u64, miner: &mut MinerManager) {
        let daa = self.last_daa;
        // The pool decides the tracing: no trace object means an untraced answer.
        let (tier, chain, attention) = match request.trace {
            Some(t) if !t.chain => {
                self.refuse(&request, "untraced-chain answers are not served");
                return;
            }
            Some(t) if keryx_miner::models::pom_tier_index(&request.model_id, u64::MAX) != Some(t.tier) => {
                self.refuse(&request, "trace tier does not match the model");
                return;
            }
            Some(t) => (Some(t.tier), t.chain, t.attention),
            None => (None, false, false),
        };
        if tier.is_some() {
            let traces = self.pool_traces.lock().await;
            let why = if traces.contains_key(&request.request_hash.to_ascii_lowercase()) {
                Some("already answered by this worker")
            } else if traces.len() >= audit::MAX_TRACES {
                Some("too many answers still under audit")
            } else {
                None
            };
            drop(traces);
            if let Some(why) = why {
                self.refuse(&request, why);
                return;
            }
        }
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            self.request_queue.push_front((request, queued_at));
            return;
        }
        let guard = InferenceGuard::new(
            miner.opoi_challenge_flag(),
            self.challenge_in_flight.clone(),
            keryx_miner::pom_gpu::serving_device(&request.model_id),
            request.model_id,
        );
        if guard.whole_rig() {
            if let Err(error) = miner.process_block(None).await {
                warn!("Could not pause mining for AI request: {}", error);
                self.refuse(&request, "mining could not pause");
                return;
            }
        }
        let sender = self.send_channel.clone();
        let worker = self.miner_address.clone();
        let pending = self.ai_response.clone();
        let last_id = self.last_stratum_id.clone();
        let traces = self.pool_traces.clone();
        let created_daa = audit::current_daa().min(daa);
        tokio::task::spawn_blocking(move || {
            let guard = guard;
            let mut request_hash = [0u8; 32];
            let mut traced_key = None;
            let mut record = None;
            let (text, traced) = match tier
                .filter(|_| hex::decode_to_slice(&request.request_hash, &mut request_hash).is_ok())
            {
                Some(tier) => {
                    let prompt_len = request.prompt_len;
                    let Some(generated) = keryx_miner::llm::load_and_run_traced_inference(
                        &request.model_id,
                        &request.prompt,
                        request.max_tokens,
                        tier,
                        prompt_len,
                        &request_hash,
                        chain,
                        attention,
                    ) else {
                        warn!("AI request {}: traced inference failed — no answer", request.task_id);
                        if let Some(line) = request.refusal("inference failed") {
                            let _ = sender.try_send(line);
                        }
                        return;
                    };
                    let (text, device) = (generated.text, generated.device);
                    match generated.trace {
                        Some(trace) => {
                            let answer = if trace.record.stage == keryx_miner::trace::STAGE_CHAIN {
                                let ids: Vec<u32> =
                                    trace.tokens[trace.record.n_prompt as usize..].iter().map(|&v| v as u32).collect();
                                keryx_verifiable_inference::answer::encode(text.as_bytes(), &ids)
                            } else {
                                text.clone().into_bytes()
                            };
                            let commitment = hex::encode(trace.commitment().to_bytes());
                            let key = request.request_hash.to_ascii_lowercase();
                            if traces.blocking_lock().contains_key(&key) {
                                warn!("AI request {}: already answered by this worker — answer dropped", request.task_id);
                                if let Some(line) = request.refusal("already answered by this worker") {
                                    let _ = sender.try_send(line);
                                }
                                return;
                            }
                            let t = audit::PoolTrace::new(request.model_id, tier, request_hash, trace, created_daa);
                            if !audit::persist_now(&key, &t) {
                                warn!("AI request {}: trace not stored — no answer", request.task_id);
                                if let Some(line) = request.refusal("trace not stored") {
                                    let _ = sender.try_send(line);
                                }
                                return;
                            }
                            if let Some(url) = request.upload.clone() {
                                record = Some((url, device, t.trace.clone()));
                            }
                            let mut t = t;
                            t.response = Some(audit::SentAnswer { text: text.clone(), traced: Some((commitment.clone(), answer.clone())) });
                            traces.blocking_lock().insert(key.clone(), t);
                            traced_key = Some(key);
                            (text, Some((commitment, answer)))
                        }
                        None => {
                            warn!("AI request {}: no trace produced — no answer", request.task_id);
                            if let Some(line) = request.refusal("no trace produced") {
                                let _ = sender.try_send(line);
                            }
                            return;
                        }
                    }
                }
                None => (
                    keryx_miner::llm::load_and_run_inference(&request.model_id, &request.prompt, request.max_tokens)
                        .unwrap_or_default(),
                    None,
                ),
            };
            drop(guard);
            // The answer goes out on the live connection, which may have replaced the one the
            // request came on; a traced answer kept with its trace is resent if the pool asks again.
            let link = audit::shared().link.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let (out, ids) = match &link {
                Some(l) => (l.sender.clone(), l.last_id.clone()),
                None => (sender.clone(), last_id.clone()),
            };
            let id = ids.fetch_add(1, Ordering::SeqCst);
            match request.response(id, worker, &text, traced) {
                Ok(line) => {
                    if let Some((url, device, trace)) = record.take() {
                        let (task_id, model_id, max_tokens) = (request.task_id.clone(), request.model_id, request.max_tokens as u32);
                        let tier = tier.unwrap_or_default();
                        let spawned = std::thread::Builder::new().name("trace-upload".into()).spawn(move || {
                            let header = keryx_miner::trace_record::header(model_id, tier, request_hash, max_tokens, device);
                            let blob = keryx_miner::trace_record::encode_record(&header, &trace);
                            match ai::upload_record(&url, &task_id, &blob) {
                                Ok(status) => info!("AI request {}: trace record sent ({} B, HTTP {})", task_id, blob.len(), status),
                                Err(e) => warn!("AI request {}: trace record not sent ({})", task_id, e),
                            }
                        });
                        if let Err(e) = spawned {
                            warn!("AI request {}: trace record not sent ({})", request.task_id, e);
                        }
                    }
                    {
                        let mut pending = pending.blocking_lock();
                        if pending.len() >= MAX_PENDING_IDS {
                            pending.clear();
                        }
                        pending.insert(id, (Instant::now(), traced_key.clone()));
                    }
                    if out.blocking_send(line).is_err() {
                        pending.blocking_lock().remove(&id);
                        warn!("AI response {}: connection closed — kept for a resend", request.task_id);
                    }
                }
                Err(error) => {
                    warn!("AI request {}: {}", request.task_id, error);
                    if let Some(line) = request.refusal("no valid answer") {
                        let _ = sender.try_send(line);
                    }
                    if let Some(key) = traced_key {
                        audit::remove(&mut traces.blocking_lock(), &key);
                    }
                }
            }
        });
    }

    /// Queues the audit opening the pool asks for one of this worker's traced answers; one that
    /// cannot be served is answered at once with the reason.
    async fn handle_ai_opening(&mut self, id: Option<u32>, params: Vec<serde_json::Value>) {
        let opening = match AiOpening::parse(&params) {
            Ok(o) => o,
            Err(e) => {
                warn!("Trace: invalid ai_opening ignored ({}, id {:?})", e, id);
                return;
            }
        };
        let daa = audit::current_daa();
        let pieces = if opening.path.is_empty() { audit::PIECES_ALL } else { audit::PIECE_ATTN };
        let deadline_daa = daa.saturating_add(audit::AUDIT_SPAN_DAA);
        let draw_key = audit::draw_key(&opening.seed, &opening.path);
        let error = {
            let mut queue = self.replay_queue.lock().await;
            let mut traces = self.pool_traces.lock().await;
            match audit::draw_bounds_error(daa, deadline_daa, pieces, opening.path.len()) {
                Some(e) => Some(e),
                None => match traces.get_mut(&opening.request_hash) {
                    None => Some("no kept trace for this request"),
                    Some(t) => {
                        let AiOpening { request_hash, response_txid, seed, path } = opening.clone();
                        let probe = audit::PoolDraw::new(request_hash.clone(), response_txid, seed, deadline_daa, 0, pieces, path.clone());
                        let action = if t.cached(&probe).is_some() { audit::OpeningAction::Replay } else { t.opening_action(&draw_key) };
                        if action == audit::OpeningAction::Ignore {
                            return;
                        }
                        if action == audit::OpeningAction::Reply {
                            drop(traces);
                            drop(queue);
                            warn!("Trace: opening for request {} given up for now", &opening.request_hash[..16]);
                            self.send_opening_error(&opening, "replay failed").await;
                            return;
                        }
                        let backoff = queue
                            .iter()
                            .find(|d| d.response_txid == response_txid && d.seed == seed && d.path == path)
                            .map(|d| (d.attempts, d.retry_at));
                        queue.retain(|d| !audit::same_or_superseded(d, &request_hash, &response_txid, &seed, &path));
                        let accepted_daa = t.created_daa();
                        if backoff.is_none() && t.note_draw(deadline_daa, accepted_daa) {
                            audit::persist(&request_hash, t);
                        }
                        let mut draw =
                            audit::PoolDraw::new(request_hash, response_txid, seed, deadline_daa, accepted_daa, pieces, path);
                        if let Some((attempts, retry_at)) = backoff {
                            draw.attempts = attempts;
                            draw.retry_at = retry_at;
                        }
                        queue.push(draw);
                        None
                    }
                },
            }
        };
        if let Some(e) = error {
            warn!("Trace: opening for request {} not served ({})", &opening.request_hash[..16], e);
            if daa == 0 {
                // Before the first job the pool resends; an answer now would read as final.
                return;
            }
            self.send_opening_error(&opening, e).await;
        }
    }

    /// Answers `opening` with the reason it is not served.
    async fn send_opening_error(&self, opening: &AiOpening, error: &str) {
        let line_id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        note_pending(&mut *self.openings_pending.lock().await, line_id);
        let _ = self.send_channel.send(opening.result(line_id, &self.miner_address, &[], error)).await;
    }

    /// Starts the replay of the first ready draw in `replay_order`; false when none could start.
    async fn start_replay(&mut self, miner: &mut MinerManager) -> bool {
        loop {
            let draw = {
                let mut queue = self.replay_queue.lock().await;
                if queue.is_empty() {
                    return false;
                }
                queue.sort_by_key(replay_order);
                let Some(i) = queue.iter().position(|d| d.ready()) else { return false };
                queue.remove(i)
            };
            let job = {
                let traces = self.pool_traces.lock().await;
                traces.get(&draw.request_hash).map(|t| {
                    (
                        t.model_id,
                        t.tier,
                        t.request_hash,
                        Arc::clone(&t.trace),
                        t.cached(&draw),
                        t.phases.clone(),
                        t.bundle(&draw.seed),
                    )
                })
            };
            let Some((model_id, tier, request_hash, trace, built, phases, bundle)) = job else {
                self.send_opening_error(&opening_of(&draw), "no kept trace for this request").await;
                continue;
            };
            let worker = self.miner_address.clone();
            if let Some(bodies) = built {
                let owed = audit::owed(&bodies, draw.pieces);
                tokio::task::spawn_blocking(move || {
                    if !send_openings(&worker, &draw, &owed) {
                        audit::requeue(draw);
                    }
                });
                continue;
            }
            if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
                self.replay_queue.lock().await.push(draw);
                return true;
            }
            let guard = InferenceGuard::new(
                miner.opoi_challenge_flag(),
                self.challenge_in_flight.clone(),
                keryx_miner::pom_gpu::serving_device(&model_id),
                model_id,
            );
            if guard.whole_rig() {
                if let Err(error) = miner.process_block(None).await {
                    warn!("Could not pause mining for an audit replay: {}", error);
                    self.replay_queue.lock().await.push(draw.retry());
                    return true;
                }
            }
            info!("Trace: replaying the audit of request {} (path {:?})", &draw.request_hash[..16], draw.path);
            let traces = self.pool_traces.clone();
            let queue = self.replay_queue.clone();
            tokio::task::spawn_blocking(move || {
                let built = audit::build(&model_id, tier, &request_hash, &trace, &phases, bundle, &draw);
                let Some(built) = built else {
                    let error = {
                        let mut queue = queue.blocking_lock();
                        let failures = traces.blocking_lock().get_mut(&draw.request_hash).map(|t| t.note_failure(&draw.key()));
                        let queued = queue
                            .iter()
                            .any(|d| audit::same_or_superseded(d, &draw.request_hash, &draw.response_txid, &draw.seed, &draw.path));
                        match failures {
                            None => Some("no kept trace for this request"),
                            Some(n) if n < audit::MAX_REPLAY_ATTEMPTS => {
                                warn!("Trace: no opening built for request {} — retrying later", &draw.request_hash[..16]);
                                if !queued {
                                    queue.push(draw.clone().retry());
                                }
                                None
                            }
                            Some(_) => {
                                warn!("Trace: no opening built for request {} — given up for now", &draw.request_hash[..16]);
                                Some("replay failed")
                            }
                        }
                    };
                    drop(guard);
                    if let Some(e) = error {
                        send_opening_result(&worker, &draw, &[], e);
                    }
                    return;
                };
                let bodies = built.bodies.clone();
                let stored = {
                    let mut queue = queue.blocking_lock();
                    let superseded = queue.iter().any(|d| {
                        d.request_hash == draw.request_hash && (d.seed != draw.seed || d.response_txid != draw.response_txid)
                    });
                    if superseded {
                        info!("Trace: replay of request {} superseded by a newer draw — not sent", &draw.request_hash[..16]);
                        drop(guard);
                        return;
                    }
                    queue.retain(|d| !(d.response_txid == draw.response_txid && d.key() == draw.key() && d.pieces & !draw.pieces == 0));
                    traces
                        .blocking_lock()
                        .get_mut(&draw.request_hash)
                        .map(|t| t.store_built(&draw, built).then(|| audit::queue_write(&draw.request_hash, t)).flatten())
                };
                let Some(written) = stored else {
                    drop(guard);
                    send_opening_result(&worker, &draw, &[], "no kept trace for this request");
                    return;
                };
                if let Some(written) = written {
                    let _ = written.recv();
                }
                drop(guard);
                if !send_openings(&worker, &draw, &audit::owed(&bodies, draw.pieces)) {
                    audit::requeue(draw);
                }
            });
            return true;
        }
    }

    /// Runs a capability challenge and sends `mining.challenge_response`.
    async fn start_challenge(&mut self, model_id_hex: String, nonce_hex: String, miner: &mut MinerManager) {
        let mut model_id = [0u8; 32];
        if hex::decode_to_slice(&model_id_hex, &mut model_id).is_err() {
            warn!("Inference challenge: invalid model_id_hex '{}'", model_id_hex);
            return;
        }
        if !keryx_miner::llm::is_model_ready(&model_id) {
            warn!("Inference challenge: model {:.8} not ready — sending empty response", model_id_hex);
            self.send_channel.send(make_challenge_response_line(&model_id_hex, &nonce_hex, "")).await.ok();
            return;
        }
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            self.pending_challenge = Some((model_id_hex, nonce_hex));
            return;
        }
        let guard = InferenceGuard::new(miner.opoi_challenge_flag(), self.challenge_in_flight.clone(), None, model_id);
        miner.process_block(None).await.ok();
        info!("Inference challenge: PoW suspended — model={:.8} nonce={:.8}", model_id_hex, nonce_hex);

        let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
        let send_channel = self.send_channel.clone();

        tokio::task::spawn_blocking(move || {
            let result = keryx_miner::llm::load_and_run_inference(&model_id, &prompt, CHALLENGE_MAX_TOKENS);
            let text = result.unwrap_or_default();
            if text.is_empty() {
                warn!("Inference challenge: inference returned empty text for model {:.8}", model_id_hex);
            } else {
                info!(
                    "Inference challenge: done for model {:.8} ({} chars) — PoW resumes on next notify",
                    model_id_hex,
                    text.len()
                );
            }
            let line = make_challenge_response_line(&model_id_hex, &nonce_hex, &text);
            if send_channel.blocking_send(line).is_err() {
                warn!("Inference challenge: send_channel closed, could not deliver response");
            }
            drop(guard);
        });
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// Handles an AiTask dispatched by the bridge. Returns `true` if inference was launched
    /// and PoW has been paused — the caller must NOT call `process_block(Some(...))` in that
    /// case; PoW resumes automatically on the next `mining.notify` after inference completes.
    async fn handle_ai_task(&mut self, job_id: String, task_json: String, miner: &mut MinerManager) -> bool {
        let task: AiTask = match serde_json::from_str(&task_json) {
            Ok(t) => t,
            Err(e) => {
                warn!("Inference: failed to parse task JSON from bridge: {}", e);
                *self.current_task_slot.lock().await = None;
                return false;
            }
        };

        // Store task for this job so create_block_channel can look up the CID.
        *self.current_task_slot.lock().await = Some(CurrentTask { job_id, task: task.clone() });

        // Skip inference if stable_id is missing (malformed task) or already done/running.
        if task.stable_id.is_empty() {
            return false;
        }
        let already_handled = {
            let cache = self.inference_cache.lock().await;
            cache.results.contains_key(&task.stable_id) || cache.in_progress.contains(&task.stable_id)
        };
        if already_handled {
            return false;
        }

        // Decode model_id hex and check it is ready on disk.
        let model_id_bytes = match hex::decode(&task.model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("Inference [{}]: invalid model_id_hex '{}'", task.stable_id, task.model_id_hex);
                return false;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::llm::is_model_ready(&model_id) {
            warn!("Inference [{}]: model not ready — inference skipped", task.stable_id);
            return false;
        }

        // Guard against two concurrent inferences (challenge may already hold the GPU).
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("Inference task [{}]: inference already in flight, skipping", task.stable_id);
            return false;
        }

        // Pause PoW — running kHeavyHash and LLM inference simultaneously crashes the GPU.
        let miner_flag = miner.opoi_challenge_flag();
        miner_flag.store(true, Ordering::SeqCst);
        keryx_miner::pom_gpu::set_inference_paused(true);
        miner.process_block(None).await.ok();
        info!("Inference task [{}]: PoW suspended for GPU inference", task.stable_id);

        // Mark in-progress and spawn the blocking inference + IPFS upload.
        {
            let mut cache = self.inference_cache.lock().await;
            cache.in_progress.insert(task.stable_id.clone());
        }
        let stable_id = task.stable_id.clone();
        let prompt = task.prompt.clone();
        let max_tokens = task.max_tokens;
        let ipfs_url = self.ipfs_url.clone();
        let cache_ref = Arc::clone(&self.inference_cache);
        let challenge_flag = Arc::clone(&self.challenge_in_flight);

        tokio::task::spawn_blocking(move || {
            run_inference_and_upload(model_id, prompt, max_tokens, ipfs_url, stable_id, cache_ref);
            // Clear both flags — PoW resumes on the next mining.notify from the bridge.
            keryx_miner::pom_gpu::set_inference_paused(false);
            miner_flag.store(false, Ordering::SeqCst);
            challenge_flag.store(false, Ordering::SeqCst);
        });

        // PoW was paused for GPU inference, so the caller must not feed a new block.
        true
    }
}

impl Drop for StratumHandler {
    fn drop(&mut self) {
        self.log_handler.abort();
        self.write_handle.abort();
        self.block_handle.abort()
    }
}

// ── Phase 2 inference — blocking inference helpers ────────────────────────────────

/// Runs LLM inference, uploads the result to IPFS, then stores the CID in the cache.
/// Called from `spawn_blocking` — must not call async functions.
fn run_inference_and_upload(
    model_id: [u8; 32],
    prompt: String,
    max_tokens: usize,
    ipfs_url: String,
    stable_id: String,
    cache: InferenceCache,
) {
    let cid_opt = do_inference_and_upload(&model_id, &prompt, max_tokens, &ipfs_url, &stable_id);
    let mut guard = cache.blocking_lock();
    guard.in_progress.remove(&stable_id);
    if let Some(cid) = cid_opt {
        if guard.results.len() >= MAX_INFERENCE_CACHE_SIZE {
            guard.results.clear();
            guard.results.shrink_to_fit();
        }
        guard.results.insert(stable_id, cid);
    }
}

fn make_challenge_response_line(model_id_hex: &str, nonce_hex: &str, result: &str) -> StratumLine {
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningChallengeResponse((
            model_id_hex.to_string(),
            nonce_hex.to_string(),
            BASE64.encode(result.as_bytes()),
        ))),
        jsonrpc: Some("2.0".into()),
        error: None,
    }
}

fn do_inference_and_upload(
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    ipfs_url: &str,
    stable_id: &str,
) -> Option<String> {
    info!("Inference [{}]: starting LLM inference (max_tokens={})", stable_id, max_tokens);
    let text = keryx_miner::llm::load_and_run_inference(model_id, prompt, max_tokens)?;
    if text.is_empty() {
        warn!("Inference [{}]: inference returned empty text — skipping IPFS upload", stable_id);
        return None;
    }
    match crate::ipfs::upload_with_recovery(&text, ipfs_url) {
        Ok(cid_bytes) => {
            // Convert raw 34-byte multihash to base58 CIDv0 string via AiResponsePayload helper.
            let cid = keryx_inference::AiResponsePayload::new([0u8; 32], 0, cid_bytes, 0).cid_v0();
            match crate::ipfs::confirm_response_retrievable(&cid) {
                Ok(gateway) => info!("Inference [{}]: inference complete, IPFS CID={} served by {}", stable_id, cid, gateway),
                Err(e) => {
                    warn!("Inference [{}]: response CID {} unreadable from the public gateways ({}) — response skipped; check the pool's kubo port 4001", stable_id, cid, e);
                    return None;
                }
            }
            Some(cid)
        }
        Err(e) => {
            warn!("Inference [{}]: IPFS upload failed: {}", stable_id, e);
            None
        }
    }
}

/// Replay order: attention phases first (their window is short), then by arrival (the
/// deadline of a pool opening is its arrival plus the audit span).
fn replay_order(d: &audit::PoolDraw) -> (bool, u64) {
    (d.path.is_empty(), d.deadline_daa)
}

/// A loaded model that is ready and has a tier, which this worker can trace.
fn traceable_model() -> Option<[u8; 32]> {
    keryx_miner::llm::loaded_model_ids()
        .into_iter()
        .find(|id| keryx_miner::llm::is_model_ready(id) && keryx_miner::models::pom_tier_index(id, u64::MAX).is_some())
}

/// The opening request `draw` answers.
fn opening_of(draw: &audit::PoolDraw) -> AiOpening {
    AiOpening { request_hash: draw.request_hash.clone(), response_txid: draw.response_txid, seed: draw.seed, path: draw.path.clone() }
}

/// Records a reply id awaited from the pool, forgetting all of them past `MAX_PENDING_IDS`.
fn note_pending(pending: &mut HashSet<u32>, id: u32) {
    if pending.len() >= MAX_PENDING_IDS {
        pending.clear();
    }
    pending.insert(id);
}

/// Sends the `mining.ai_opening_result` answering `draw` on the live pool connection, its bodies
/// bundled; blocking. False when the connection is gone.
fn send_openings(worker: &str, draw: &audit::PoolDraw, bodies: &[Vec<u8>]) -> bool {
    if bodies.is_empty() {
        return true;
    }
    send_opening_result(worker, draw, &keryx_miner::trace_seq::bundle(bodies), "")
}

/// Sends one `mining.ai_opening_result` for `draw`: `body`, or `error` when it is empty; blocking.
fn send_opening_result(worker: &str, draw: &audit::PoolDraw, body: &[u8], error: &str) -> bool {
    let Some(link) = audit::shared().link.lock().unwrap_or_else(|e| e.into_inner()).clone() else { return false };
    let opening = opening_of(draw);
    let id = link.last_id.fetch_add(1, Ordering::SeqCst);
    note_pending(&mut link.openings_pending.blocking_lock(), id);
    if link.sender.blocking_send(opening.result(id, worker, body, error)).is_err() {
        warn!("Trace: connection closed before the opening was sent");
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(rh: &str, txid: u8, seed: u8, path: Vec<u64>, deadline: u64) -> audit::PoolDraw {
        audit::PoolDraw::new(rh.into(), [txid; 32], [seed; 32], deadline, 0, audit::PIECES_ALL, path)
    }

    #[test]
    fn phases_replay_before_first_openings() {
        let mut q = vec![draw("a", 1, 1, vec![], 100), draw("b", 2, 2, vec![], 200), draw("c", 3, 3, vec![1], 300)];
        q.sort_by_key(replay_order);
        assert_eq!(q.iter().map(|d| d.request_hash.as_str()).collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    #[test]
    fn a_resend_keeps_other_phases_and_a_new_seed_supersedes() {
        let queued = draw("r", 1, 1, vec![2], 0);
        assert!(!audit::same_or_superseded(&queued, "r", &[1; 32], &[1; 32], &[]));
        assert!(audit::same_or_superseded(&queued, "r", &[1; 32], &[1; 32], &[2]));
        assert!(audit::same_or_superseded(&queued, "r", &[1; 32], &[9; 32], &[]));
        assert!(audit::same_or_superseded(&queued, "r", &[9; 32], &[1; 32], &[]));
        assert!(!audit::same_or_superseded(&queued, "other", &[7; 32], &[7; 32], &[2]));
    }

    #[test]
    fn pending_ids_are_capped() {
        let mut p = HashSet::new();
        for id in 0..MAX_PENDING_IDS as u32 {
            note_pending(&mut p, id);
        }
        assert_eq!(p.len(), MAX_PENDING_IDS);
        note_pending(&mut p, 999);
        assert_eq!(p.len(), 1);
        assert!(p.contains(&999));
    }
}

#[cfg(test)]
mod proof_encoding_tests {
    use super::*;

    #[test]
    fn proofs_go_b64_only_when_announced() {
        assert!(announces_b64(&[("mining.notify".into(), "1".into()), ("keryx.proof_encoding".into(), "b64".into())]));
        assert!(!announces_b64(&[("keryx.proof_encoding".into(), "hex".into())]));
        let e = ProofEncoding::default();
        assert_eq!(e.encode(1, &[0xAB, 0xCD]), "abcd");
        assert!(!e.was_b64(1));
        e.b64.store(true, Ordering::Release);
        assert_eq!(e.encode(2, &[0xAB, 0xCD]), format!("b64:{}", BASE64.encode([0xAB, 0xCD])));
        assert!(e.was_b64(2) && !e.was_b64(2));
    }
}
