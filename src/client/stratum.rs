use futures::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

mod statum_codec;

use crate::client::stratum::statum_codec::{ErrorCode, MiningNotify, MiningSubmit, NewLineJsonCodecError, StratumLine};
use crate::client::stratum::statum_codec::{
    MiningSubscribe, SetExtranonce, StratumCommand, StratumError, StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::PartialBlock;
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{error, info, warn};
use num::Float;
use rand::{thread_rng, RngCore};
use statum_codec::NewLineJsonCodec;
use std::sync::OnceLock;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;

//const DIFFICULTY_1_TARGET: Uint256 = Uint256([0x00000000ffff0000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000]);
const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208); // 0xffff 2^208
const KERYX_STRATUM_DAA_CAPABILITY: &str = "keryx-stratum-v2";
const LOG_RATE: Duration = Duration::from_secs(30);
const CHALLENGE_MAX_TOKENS: usize = 128;

// ── Phase 2 OPoI — inference cache & task types ─────────────────────────────

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

/// Shared inference result cache — persists across block changes so that if the
/// same AiRequest is included in multiple consecutive job templates the miner can
/// immediately submit with a CID once inference completed for the first occurrence.

/// Max cached inference results — evict when full to prevent unbounded growth.
const MAX_INFERENCE_CACHE_SIZE: usize = 1_000;

struct InferenceCacheInner {
    /// stable_id → base58 CIDv0 string returned by IPFS after upload.
    results: HashMap<String, String>,
    /// stable_ids currently being inferred (guards against duplicate spawn_blocking calls).
    in_progress: HashSet<String>,
}

type InferenceCache = Arc<Mutex<InferenceCacheInner>>;

type BlockHandle = JoinHandle<()>;

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
    pub shares_pending: Mutex<HashMap<u32, String>>,
}

static SHARE_STATS: OnceLock<Arc<ShareStats>> = OnceLock::new();

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shares: {}{}{}{}Pending: {}",
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
            self.shares_pending.try_lock().unwrap().len()
        )
    }
}

#[allow(dead_code)]
pub struct StratumHandler {
    log_handler: JoinHandle<()>,

    //client: Framed<TcpStream, NewLineJsonCodec>,
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>>,
    miner_address: String,
    mine_when_not_synced: bool,
    devfund_address: Option<String>,
    devfund_percent: u16,
    mining_dev: Option<bool>,
    block_template_ctr: Arc<AtomicU16>,

    target_pool: Uint256,
    target_real: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,

    shares_stats: Arc<ShareStats>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// IPFS Kubo API URL for uploading inference results (e.g. "http://127.0.0.1:5001").
    ipfs_url: String,
    /// Task dispatched by the bridge for the current mining job (None = no AiRequest in job).
    current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
    /// Completed inferences: stable_id → base58 CIDv0 string (persists across block changes).
    inference_cache: InferenceCache,
    /// True while a capability challenge inference is in flight — prevents duplicate spawns.
    challenge_in_flight: Arc<AtomicBool>,
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    fn add_devfund(&mut self, address: String, percent: u16) {
        self.devfund_address = Some(address);
        self.devfund_percent = percent;
    }

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

        let pay_address = match &self.devfund_address {
            Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
                self.mining_dev = Some(true);
                info!("Mining to devfund");
                devfund_address.clone()
            }
            _ => {
                self.mining_dev = Some(false);
                self.miner_address.clone()
            }
        };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    pay_address.clone(),
                    "x".into(),
                ))),
                jsonrpc: None,
                error: None,
            })
            .await?;

        // Declare loaded SLM models so the bridge can challenge with the right model.
        let model_ids: Vec<String> = keryx_miner::slm::loaded_model_ids()
            .into_iter()
            .map(|id| hex::encode(id))
            .collect();
        if !model_ids.is_empty() {
            info!("OPoI: declaring {} model(s) to pool bridge", model_ids.len());
            self.send_channel
                .send(StratumLine {
                    id: None,
                    payload: StratumLinePayload::StratumCommand(
                        StratumCommand::MiningDeclareCapabilities(model_ids),
                    ),
                    jsonrpc: None,
                    error: None,
                })
                .await?;
        }
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for stuff");
        loop {
            {
                if (!self.mining_dev.unwrap_or(true)
                    && self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent)
                    || (self.mining_dev.unwrap_or(false)
                        && self.block_template_ctr.load(Ordering::SeqCst) > self.devfund_percent)
                {
                    return Ok(());
                }
            }
            match self.stream.try_next().await? {
                Some(msg) => self.handle_message(msg, miner).await?,
                None => return Err("stratum message payload is empty".into()),
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl StratumHandler {
    pub async fn connect(
        address: String,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error> {
        info!("Connecting to {}", address);
        let socket = TcpStream::connect(address).await?;

        let client = Framed::new(socket, NewLineJsonCodec::new());
        let (send_channel, recv) = mpsc::channel::<StratumLine>(3);
        let (sink, stream) = client.split();
        tokio::spawn(async move { ReceiverStream::new(recv).map(Ok).forward(sink).await });

        let share_state = SHARE_STATS.get_or_init(|| Arc::new(ShareStats::default())).clone();
        let last_stratum_id = Arc::new(AtomicU32::new(0));
        let current_task_slot: Arc<Mutex<Option<CurrentTask>>> = Arc::new(Mutex::new(None));
        let inference_cache: InferenceCache = Arc::new(Mutex::new(InferenceCacheInner {
            results: HashMap::new(),
            in_progress: HashSet::new(),
        }));
        let (block_channel, block_handle) = Self::create_block_channel(
            send_channel.clone(),
            miner_address.clone(),
            last_stratum_id.clone(),
            share_state.clone(),
            Arc::clone(&current_task_slot),
            Arc::clone(&inference_cache),
        );
        Ok(Box::new(Self {
            log_handler: task::spawn(Self::log_shares(share_state.clone())),
            stream: Box::pin(stream),
            send_channel,
            miner_address,
            mine_when_not_synced,
            devfund_address: None,
            devfund_percent: 0,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            target_pool: Default::default(),
            target_real: Default::default(),
            nonce_mask: u64::MAX, // full nonce space until set_extranonce assigns a sub-range
            nonce_fixed: 0,
            extranonce: None,
            last_stratum_id,
            shares_stats: share_state,
            mining_dev: None,
            block_channel,
            block_handle,
            ipfs_url,
            current_task_slot,
            inference_cache,
            challenge_in_flight: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        miner_address: String,
        last_stratum_id: Arc<AtomicU32>,
        share_stats: Arc<ShareStats>,
        current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
        inference_cache: InferenceCache,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);

        let handle = tokio::spawn(async move {
            let mut recv_stream = ReceiverStream::new(recv);
            while let Some(seed) = recv_stream.next().await {
                let (nonce, job_id) = match seed {
                    BlockSeed::PartialBlock { nonce, id, .. } => (nonce, id),
                    BlockSeed::FullBlock { .. } => unreachable!(),
                };
                let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                share_stats.shares_pending.try_lock().unwrap().insert(msg_id, job_id.clone());
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

                let line = if let Some(cid) = cid_opt {
                    info!("OPoI Phase 2: submitting share with CID for job {}", job_id);
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
                            MiningSubmit::MiningSubmitWithTag((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                };

                if send_channel.send(line).await.is_err() {
                    break;
                }
            }
        });
        (send, handle)
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        match msg.clone() {
            StratumLine { id, payload, error: None, .. } => {
                match payload {
                    StratumLinePayload::StratumResult { result } if id.is_some() => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                if let Some(_jobid) = self
                                    .shares_stats
                                    .shares_pending
                                    .try_lock()
                                    .unwrap()
                                    .remove(&id.expect("We checked id is not none"))
                                {
                                    self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                    info!("Share accepted");
                                } else {
                                    info!("{:?} (Last: {})", msg.clone(), self.last_stratum_id.load(Ordering::SeqCst));
                                    warn!("Ignoring result for now");
                                }
                                Ok(())
                            }
                            StratumResult::Subscribe((ref _subscriptions, ref extranonce, ref nonce_size)) => {
                                self.set_extranonce(extranonce.as_str(), nonce_size)
                                /*for (name, value) in _subscriptions {
                                    match name.as_str() {
                                        "mining.set_difficulty" => {self.set_difficulty(&f32::from_str(value.as_str())?)?;},
                                        _ => {warn!("Ignored {} (={})", name, value);}
                                    }
                                }
                                Ok(())*/
                            }
                            _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                        }
                    }
                    StratumLinePayload::StratumCommand(command) => match command {
                        StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((
                            ref extranonce,
                            ref nonce_size,
                        ))) => self.set_extranonce(extranonce.as_str(), nonce_size),
                        StratumCommand::MiningSetDifficulty((ref difficulty,)) => self.set_difficulty(difficulty),
                        // Phase 2 OPoI: bridge dispatches an AiRequest task alongside the block.
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyWithTask((
                            id,
                            header_hash,
                            timestamp,
                            daa_score,
                            task_json,
                        ))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            let inference_started =
                                self.handle_ai_task(id.clone(), task_json, miner).await;
                            if inference_started {
                                // PoW already paused inside handle_ai_task — do NOT feed a new
                                // block template or the GPU immediately resumes hashing.
                                Ok(())
                            } else {
                                miner
                                    .process_block(Some(PartialBlock {
                                        id,
                                        header_hash,
                                        timestamp,
                                        daa_score,
                                        nonce: 0,
                                        target: self.target_pool,
                                        nonce_mask: self.nonce_mask,
                                        nonce_fixed: self.nonce_fixed,
                                        hash: None,
                                    }))
                                    .await
                            }
                        }
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShortV2((
                            id,
                            header_hash,
                            timestamp,
                            daa_score,
                        ))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            // No AiRequest in this job — clear the task slot.
                            *self.current_task_slot.lock().await = None;
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    daa_score,
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                }))
                                .await
                        }
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            *self.current_task_slot.lock().await = None;
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    // Short stratum notify carries no daa_score; pin it to the
                                    // current salt era so the host generates the right matrix.
                                    // Post-relaunch the chain is on SALT v4, so force v4.
                                    daa_score: crate::pow::heavy_hash::pow_salt_v4_activation_daa(),
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                }))
                                .await
                        }
                        StratumCommand::MiningChallenge((model_id_hex, nonce_hex)) => {
                            self.handle_challenge(model_id_hex, nonce_hex, miner).await;
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
                let jobid = { self.shares_stats.shares_pending.try_lock().unwrap().remove(&id) }.unwrap();
                match code {
                    ErrorCode::Unknown => {
                        // Match solo-mining behaviour (grpc.rs SubmitBlockResponse): a rejected
                        // share/block is logged but never fatal. Returning Err here tore down the
                        // whole connection and caused an infinite reconnect loop on every share.
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

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        let mut buf = [0u64, 0u64, 0u64, 0u64];
        let (mantissa, exponent, _) = difficulty.recip().integer_decode();
        let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
        let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
        let start = (new_exponent / 64) as usize;
        let remainder = new_exponent % 64;

        buf[start] = new_mantissa << remainder; // bottom
        if start < 3 {
            buf[start + 1] = new_mantissa >> (64 - remainder); // top
        } else if new_mantissa.leading_zeros() < remainder as u32 {
            return Err("Target is too big".into());
        }

        self.target_pool = Uint256::new(buf);
        info!("Difficulty: {:?}, Target: 0x{}", difficulty, hex::encode(self.target_pool.to_be_bytes()));
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        self.extranonce = Some(extranonce.to_string());
        info!("Extra! {:?}", extranonce);
        self.nonce_fixed = u64::from_str_radix(extranonce, 16)? << (nonce_size * 8);
        info!("Extra Done!");
        self.nonce_mask = (1 << (nonce_size * 8)) - 1;
        Ok(())
    }

    async fn log_shares(shares_info: Arc<ShareStats>) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut _last_instant = ticker.tick().await;
        loop {
            let _now = ticker.tick().await;
            info!("{}", shares_info)
        }
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// and spawn a background inference+IPFS upload if the result is not already cached.
    /// Handle a `mining.challenge` from the bridge.
    ///
    /// The bridge relays the node's periodic capability challenge: the miner must prove
    /// it has the requested model loaded and can produce inference output. The result is
    /// sent back as `mining.challenge_response` so the bridge can forward it to the node.
    async fn handle_challenge(&mut self, model_id_hex: String, nonce_hex: String, miner: &mut MinerManager) {
        // Only one challenge in flight at a time — bridge will re-challenge if needed.
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("OPoI challenge: already in flight, dropping new challenge for model {:.8}", model_id_hex);
            return;
        }

        let model_id_bytes = match hex::decode(&model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("OPoI challenge: invalid model_id_hex '{}'", model_id_hex);
                self.challenge_in_flight.store(false, Ordering::SeqCst);
                return;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI challenge: model {:.8} not ready — sending empty response", model_id_hex);
            self.challenge_in_flight.store(false, Ordering::SeqCst);
            self.send_channel.send(make_challenge_response_line(&model_id_hex, &nonce_hex, "")).await.ok();
            return;
        }

        // Pause PoW so the GPU is fully available for the challenge inference.
        let miner_flag = miner.opoi_challenge_flag();
        miner_flag.store(true, Ordering::SeqCst);
        miner.process_block(None).await.ok();
        info!("OPoI challenge: PoW suspended — model={:.8} nonce={:.8}", model_id_hex, nonce_hex);

        let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
        let correlation = format!("challenge:{}", nonce_hex);
        let send_channel = self.send_channel.clone();
        let challenge_flag = Arc::clone(&self.challenge_in_flight);

        tokio::task::spawn_blocking(move || {
            let result = keryx_miner::slm::load_and_run_inference(
                &model_id,
                &prompt,
                CHALLENGE_MAX_TOKENS,
                &correlation,
            );
            let text = result.unwrap_or_default();
            // Clear both flags — PoW resumes on the next mining.notify from the bridge.
            miner_flag.store(false, Ordering::SeqCst);
            challenge_flag.store(false, Ordering::SeqCst);
            if text.is_empty() {
                warn!("OPoI challenge: inference returned empty text for model {:.8}", model_id_hex);
            } else {
                info!("OPoI challenge: done for model {:.8} ({} chars) — PoW resumes on next notify", model_id_hex, text.len());
            }
            let line = make_challenge_response_line(&model_id_hex, &nonce_hex, &text);
            if send_channel.blocking_send(line).is_err() {
                warn!("OPoI challenge: send_channel closed, could not deliver response");
            }
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
                warn!("OPoI: failed to parse task JSON from bridge: {}", e);
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
                warn!("OPoI [{}]: invalid model_id_hex '{}'", task.stable_id, task.model_id_hex);
                return false;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI [{}]: model not ready — inference skipped", task.stable_id);
            return false;
        }

        // Guard against two concurrent inferences (challenge may already hold the GPU).
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("OPoI AiTask [{}]: inference already in flight, skipping", task.stable_id);
            return false;
        }

        // Pause PoW — running kHeavyHash and SLM inference simultaneously crashes the GPU.
        let miner_flag = miner.opoi_challenge_flag();
        miner_flag.store(true, Ordering::SeqCst);
        miner.process_block(None).await.ok();
        info!("OPoI AiTask [{}]: PoW suspended for GPU inference", task.stable_id);

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
        self.block_handle.abort()
    }
}

// ── Phase 2 OPoI — blocking inference helpers ────────────────────────────────

/// Runs SLM inference, uploads the result to IPFS, then stores the CID in the cache.
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
            result.to_string(),
        ))),
        jsonrpc: None,
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
    let correlation = format!("request:{}", stable_id);
    info!(
        "event=ai_inference_queued correlation={} model={} max_tokens={}",
        correlation,
        hex::encode(&model_id[..4]),
        max_tokens
    );
    let text = keryx_miner::slm::load_and_run_inference(model_id, prompt, max_tokens, &correlation)?;
    if text.is_empty() {
        warn!("OPoI [{}]: inference returned empty text — skipping IPFS upload", stable_id);
        return None;
    }
    match crate::ipfs::upload(&text, ipfs_url) {
        Ok(cid_bytes) => {
            // Convert raw 34-byte multihash to base58 CIDv0 string via AiResponsePayload helper.
            let cid = keryx_inference::AiResponsePayload::new([0u8; 32], 0, cid_bytes, 0).cid_v0();
            info!("OPoI [{}]: inference complete, IPFS CID={}", stable_id, cid);
            Some(cid)
        }
        Err(e) => {
            warn!("OPoI [{}]: IPFS upload failed: {}", stable_id, e);
            None
        }
    }
}
