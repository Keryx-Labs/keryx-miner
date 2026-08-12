use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage, GetServiceStrikesRequestMessage,
    KaspadMessage, NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage,
    NotifyVirtualSelectedParentChainChangedRequestMessage,
};
use crate::{miner::MinerManager, Error};

/// Max AiRequest queue size — drop oldest when full to prevent unbounded memory growth.
const MAX_AI_QUEUE_SIZE: usize = 64;
/// Max boot-time escrow-validation GetBlock requests in flight at once — each answer
/// sends the next queued one, so thousands of state entries never overwhelm the
/// HTTP/2 flow-control window or delay the mining stream.
const VALIDATION_WINDOW: usize = 64;
/// Max unique stable-ids tracked for deduplication — evict when full.
const MAX_AI_SEEN_IDS: usize = 10_000;

use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,
    devfund_address: Option<String>,
    devfund_percent: u16,
    block_template_ctr: Arc<AtomicU16>,

    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Queue of AiRequests waiting for inference.
    /// Each entry: (stable_id_hex16, raw_payload_bytes, model_id, prompt, max_tokens).
    /// Fed by both BlockAdded scans and block template scans.
    ai_request_queue: VecDeque<(String, Vec<u8>, [u8; 32], String, usize)>,

    /// Block hashes queued for boot-time escrow-state validation, drained in slices of
    /// VALIDATION_WINDOW so thousands of GetBlock requests never saturate the HTTP/2
    /// flow-control window (each consumed answer sends the next queued request).
    validation_queue: VecDeque<String>,

    /// Stable IDs already queued or in-flight — used for deduplication.
    ai_seen_prefixes: std::collections::HashSet<String>,

    /// Maps stable_id → (txid, inference_reward_sompi) for confirmed AiRequest TXs.
    /// Used by poll_inference to register the escrow outpoint after a successful AiResponse.
    ai_request_txids: std::collections::HashMap<String, (String, u64)>,

    /// In-flight SLM inference task: (request_raw_bytes, result_receiver).
    /// None result means inference failed (model not ready or empty output) — skip IPFS upload.
    inference_rx: Option<(Vec<u8>, oneshot::Receiver<Option<String>>)>,

    /// In-flight inference for a node-issued challenge.
    /// Tuple: (challenge_string, result_receiver) where challenge_string = "model_id_hex:nonce_hex".
    /// When the result arrives, it is sent back via inference_result in the next GetBlockTemplateRequest.
    challenge_inference_rx: Option<(String, oneshot::Receiver<Option<String>>)>,

    /// Shared flag with MinerManager — suppresses GPU stall warnings during OPoI inference.
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

    /// Last service-bond strike poll instant.
    last_strike_poll: std::time::Instant,

    /// Last rendered service-bond status — logged only on change.
    strike_status: Option<String>,
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    fn add_devfund(&mut self, address: String, percent: u16) {
        self.devfund_address = Some(address);
        self.devfund_percent = percent;
    }

    async fn register(&mut self) -> Result<(), Error> {
        // We actually register in connect
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        self.opoi_challenge_active = Some(miner.opoi_challenge_flag());
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
            match maybe_msg {
                Some(Some(m)) => match m.payload {
                    Some(payload) => self.handle_message(payload, miner).await?,
                    None => warn!("keryxd message payload is empty"),
                },
                Some(None) => break, // stream closed by node
                None => {
                    // Timer tick: if a regular inference just finished, get a fresh template.
                    if self.inference_rx.is_some() && self.poll_inference().await {
                        self.client_get_block_template().await?;
                    // If a challenge is in flight, keep pinging the node so the result is
                    // delivered as soon as the inference task completes. This is critical on
                    // sole-producer nodes where mining suspension stops NewBlockTemplate
                    // notifications and the response would otherwise never be sent.
                    } else if self.challenge_inference_rx.is_some() {
                        self.client_get_block_template().await?;
                    }
                    if self.escrow_pubkey.is_some() && self.last_strike_poll.elapsed().as_secs() >= 60 {
                        self.last_strike_poll = std::time::Instant::now();
                        self.client_send(GetServiceStrikesRequestMessage {}).await?;
                    }
                }
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl KeryxdHandler {
    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        escrow_privkey: Option<String>,
        escrow_state_file: String,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        // Build EscrowWatcher from the resolved escrow privkey (derived or loaded from file).
        // The watcher also provides the pubkey to embed in coinbase extra_data.
        let (escrow_pubkey, escrow_watcher) = match escrow_privkey {
            Some(ref privkey) => {
                match crate::escrow::EscrowWatcher::new(privkey, &miner_address, escrow_state_file.into()) {
                    Ok(watcher) => {
                        let pk = watcher.pubkey_hex();
                        info!("OPoI escrow active: pubkey={}", pk);
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

        let mut client = RpcClient::connect(address).await?;
        // Outbound message channel to the node. ALL client->node messages share this:
        // mining (submit_block, GetBlockTemplate) AND OPoI traffic (per-block GetBlock,
        // escrow submit_transaction). With a capacity of 2 the OPoI traffic could fill the
        // buffer and block GetBlockTemplate, stalling template delivery → the GPU sits idle
        // between blocks. A large buffer keeps the mining requests from queuing behind OPoI.
        let (send_channel, recv) = mpsc::channel(1024);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let pending_block_submissions = Arc::new(Mutex::new(VecDeque::new()));
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone(), Arc::clone(&pending_block_submissions));
        Ok(Box::new(Self {
            client,
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            devfund_address: None,
            devfund_percent: 0,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            block_channel,
            block_handle,
            ai_request_queue: VecDeque::new(),
            validation_queue: VecDeque::new(),
            ai_seen_prefixes: std::collections::HashSet::new(),
            ai_request_txids: std::collections::HashMap::new(),
            inference_rx: None,
            challenge_inference_rx: None,
            opoi_challenge_active: None,
            pending_block_submissions,
            last_known_daa: 0,
            ipfs_url,
            escrow_pubkey,
            escrow_watcher,
            last_strike_poll: std::time::Instant::now() - std::time::Duration::from_secs(55),
            strike_status: None,
        }))
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

    async fn client_send(&self, msg: impl Into<KaspadMessage>) -> Result<(), SendError<KaspadMessage>> {
        self.send_channel.send(msg.into()).await
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<KaspadMessage>> {
        let pay_address = match &self.devfund_address {
            Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
                devfund_address.clone()
            }
            _ => self.miner_address.clone(),
        };
        self.block_template_ctr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000)).unwrap();
        // Append a per-request random nonce so that parallel blocks at the same blue_score
        // get distinct coinbase payloads → distinct tx_ids (avoids DAG coinbase collisions).
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // OPoI Phase 2: run the deterministic fixed-point MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Embed escrow pubkey so the node routes 20% to the CSV-locked escrow output.
        let escrow_part = self.escrow_pubkey
            .as_deref()
            .map(|pk| format!("/escrow:{}", pk))
            .unwrap_or_default();
        // Announce loaded model capabilities so the node can enforce model_id matching.
        let cap_part = {
            let ids = keryx_miner::slm::loaded_model_ids();
            if ids.is_empty() {
                String::new()
            } else {
                let hex_ids: Vec<String> = ids.iter().map(|id| hex::encode(id)).collect();
                format!("/ai:cap:{}", hex_ids.join(","))
            }
        };
        let extra_data = format!("{}{}/{}/ai:v1:{}{}", EXTRA_DATA, escrow_part, nonce_hex, opoi_tag, cap_part);
        // Harvest a pending challenge response if the inference task just finished.
        let inference_result = match self.challenge_inference_rx.take() {
            Some((challenge_str, mut rx)) => match rx.try_recv() {
                Ok(Some(text)) => {
                    // challenge_str = "model_id_hex:nonce_hex"
                    let mut parts = challenge_str.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("");
                    let nonce_hex_c  = parts.next().unwrap_or("");
                    info!("OPoI: sending challenge response model={:.8}", model_id_hex);
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    // Response format: "model_id_hex:nonce_hex:result_text"
                    format!("{}:{}:{}", model_id_hex, nonce_hex_c, text)
                }
                Ok(None) => {
                    warn!("OPoI: challenge inference failed — sending empty result, node will re-challenge");
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
                    warn!("OPoI: challenge inference task dropped — sending empty result");
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
    fn scan_txs_for_ai_requests(&mut self, txs: &[crate::proto::RpcTransaction]) {
        // Hard gate: if no models are ready, refuse to accept any AiRequest.
        // Prevents miners with missing/truncated model files from ever queuing inference work.
        let ready_ids = keryx_miner::slm::loaded_model_ids();
        if ready_ids.is_empty() {
            log::warn!("OPoI: no models ready — skipping AiRequest scan (run miner with valid model files)");
            return;
        }
        log::debug!(
            "scan_ai: {} txs, subnetwork_ids: {:?}",
            txs.len(),
            txs.iter().map(|t| t.subnetwork_id.as_str()).collect::<Vec<_>>()
        );
        for tx in txs {
            // (raw, model_id, prompt, max_tokens, inference_reward)
            let extracted: Option<(Vec<u8>, [u8; 32], String, usize, u64)> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork).
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        keryx_inference::AiRequestPayload::deserialize(&raw).map(|req| {
                            let model_id = req.model_id;
                            let prompt = String::from_utf8_lossy(&req.prompt).into_owned();
                            let max_tokens = req.max_tokens as usize;
                            let inference_reward = req.inference_reward;
                            (raw, model_id, prompt, max_tokens, inference_reward)
                        })
                    })
                } else if !tx.inputs.is_empty() {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            Some((raw, model_id, prompt, max_tokens, 0u64))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, max_tokens, inference_reward)) = extracted {
                if !ready_ids.contains(&model_id) {
                    log::debug!("OPoI: skipping AiRequest — model not supported or files not ready");
                    continue;
                }
                let hash = blake2b_simd::blake2b(&raw);
                let stable_id = hex::encode(&hash.as_bytes()[..8]);
                if !self.ai_seen_prefixes.contains(&stable_id) {
                    info!("OPoI: queued AiRequest id={}", stable_id);
                    self.ai_seen_prefixes.insert(stable_id.clone());
                    self.ai_request_queue.push_back((stable_id.clone(), raw, model_id, prompt, max_tokens));
                    while self.ai_request_queue.len() > MAX_AI_QUEUE_SIZE {
                        self.ai_request_queue.pop_front();
                    }
                    while self.ai_seen_prefixes.len() > MAX_AI_SEEN_IDS {
                        self.ai_seen_prefixes.clear();
                        self.ai_seen_prefixes.shrink_to_fit();
                        for (sid, _, _, _, _) in &self.ai_request_queue {
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
                    let txid_opt = tx.verbose_data.as_ref()
                        .map(|v| v.transaction_id.clone())
                        .filter(|id| !id.is_empty())
                        .or_else(|| Self::compute_rpc_txid(tx));
                    if let Some(txid) = txid_opt {
                        self.ai_request_txids.insert(stable_id, (txid, inference_reward));
                    }
                }
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
        let max_tokens = v["n"].as_u64().unwrap_or(128) as usize;
        Some((model, prompt, max_tokens))
    }

    /// Starts SLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    fn try_start_inference(&mut self) {
        if self.inference_rx.is_some() {
            return;
        }
        if let Some((stable_id, raw, model_id, prompt, max_tokens)) = self.ai_request_queue.pop_front() {
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::slm::is_model_ready(&model_id) {
                log::error!("OPoI: model became unavailable after queuing id={} — discarding request", stable_id);
                return;
            }
            info!("OPoI: spawning SLM inference (max_tokens={})", max_tokens);
            let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
            tokio::task::spawn_blocking(move || {
                let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, max_tokens);
                if result.is_none() {
                    log::warn!("OPoI: inference returned no result for id={} — AiResponse will be skipped", stable_id);
                }
                let _ = tx_done.send(result);
            });
            self.inference_rx = Some((raw, rx_done));
        }
    }

    /// Polls the in-flight inference task. When complete, uploads the result to
    /// IPFS and submits a zero-input/zero-output AiResponse transaction.
    /// Returns `true` if inference just finished (regardless of tx success).
    async fn poll_inference(&mut self) -> bool {
        let Some((raw, mut rx)) = self.inference_rx.take() else {
            return false;
        };
        let Ok(result_opt) = rx.try_recv() else {
            self.inference_rx = Some((raw, rx));
            return false;
        };
        let Some(result) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("OPoI: inference produced no result — AiResponse skipped");
            return true;
        };

        let full_hash = blake2b_simd::blake2b(&raw);
        let request_hash: [u8; 32] = full_hash.as_bytes()[..32].try_into().unwrap();
        info!("OPoI: inference complete, request_hash={}", hex::encode(&request_hash[..8]));

        let ipfs_url = self.ipfs_url.clone();
        let result_clone = result.clone();
        let cid = match tokio::task::spawn_blocking(move || crate::ipfs::upload(&result_clone, &ipfs_url)).await {
            Ok(Ok(cid)) => cid,
            Ok(Err(e)) => { warn!("OPoI: IPFS upload failed: {} — AiResponse tx skipped", e); return true; }
            Err(e) => { warn!("OPoI: IPFS spawn_blocking failed: {} — AiResponse tx skipped", e); return true; }
        };

        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
        // H6 service-bond era: sign the response with the escrow key (payload V2) so it counts
        // as served for the tier cohort — an unsigned response no longer cancels a strike. The
        // era rule mirrors the node's: V2 is rejected before the gate, so v1 is kept below it.
        let v2 = self.last_known_daa >= keryx_miner::pom::pom_v3_activation_daa();
        let resp = match (&self.escrow_watcher, v2) {
            (Some(w), true) => {
                let unsigned = keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length);
                let responder = w.sign_responder(&unsigned.signed_bytes());
                keryx_inference::AiResponsePayload::new_v2(request_hash, challenge_window_end, cid, response_length, responder)
            }
            (None, true) => {
                warn!("OPoI: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length),
        };
        info!("OPoI: uploading response CID={}, challenge_window_end={}{}", resp.cid_v0(), challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" });

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
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
            warn!("OPoI: failed to send AiResponse tx: {}", e);
        }

        // Register inference escrow outpoint for auto-claim after the challenge window.
        let stable_id = hex::encode(&full_hash.as_bytes()[..8]);
        if let Some((txid, inference_reward)) = self.ai_request_txids.remove(&stable_id) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }

        true
    }

    /// Logs this miner's service-bond standing when it changes: strike count, burns awaiting
    /// finality and production suspensions, matched by escrow pubkey.
    fn report_service_strikes(&mut self, resp: &crate::proto::GetServiceStrikesResponseMessage) {
        let Some(me) = self.escrow_pubkey.as_deref() else { return };
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
                    if !block.transactions.is_empty() {
                        // Full block — scan directly.
                        self.scan_txs_for_ai_requests(&block.transactions);
                        self.try_start_inference();
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
            Payload::GetServiceStrikesResponse(resp) => match resp.error.as_ref() {
                Some(e) => warn!("service-bond status unavailable: {}", e.message),
                None => self.report_service_strikes(&resp),
            },
            Payload::GetBlockTemplateResponse(template) => {
                // Track DAA score for challenge_window_end computation.
                if let Some(daa) = template.block.as_ref()
                    .and_then(|b| b.header.as_ref())
                    .map(|h| h.daa_score)
                {
                    if daa > self.last_known_daa {
                        self.last_known_daa = daa;
                    }
                }
                // Handle node-issued inference challenge: spawn an inference task if a new
                // challenge arrived and no challenge is already in flight. Ignored under PoM — the
                // per-block possession proof is the capability gate, so no synthetic challenge
                // (defensive: holds even against a node that still issues them post-hardfork).
                if !template.inference_challenge.is_empty()
                    && self.challenge_inference_rx.is_none()
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
                            if keryx_miner::slm::is_model_ready(&model_id) {
                                info!("OPoI: challenge received model={:.8} nonce={:.8} — spawning inference", model_id_hex, nonce_hex);
                                if let Some(flag) = &self.opoi_challenge_active {
                                    flag.store(true, Ordering::Relaxed);
                                }
                                let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
                                let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
                                tokio::task::spawn_blocking(move || {
                                    let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, 64);
                                    let _ = tx_done.send(result);
                                });
                                self.challenge_inference_rx = Some((challenge, rx_done));
                            } else {
                                warn!("OPoI: challenge for unready model={:.8} — cannot respond", model_id_hex);
                            }
                        }
                    }
                }
                // Poll in-flight inference; if done, submit AiResponse tx then get fresh template.
                if self.poll_inference().await {
                    self.client_get_block_template().await?;
                    return Ok(());
                }
                // OPoI is mandatory: refuse to mine if no models are ready.
                // Covers miners with missing/truncated model files that somehow passed prefetch.
                if keryx_miner::slm::loaded_model_ids().is_empty() {
                    // Throttle to one log per ~200 templates (~every 20s at 10 BPS) to avoid spam.
                    if self.last_known_daa % 200 == 0 {
                        log::warn!("OPoI: no models ready — mining suspended until model files are available");
                    }
                    miner.process_block(None).await?;
                    return Ok(());
                }
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(&block.transactions);
                }
                self.try_start_inference();
                // Pause GPU mining while any inference is in flight (GPU is occupied by the model).
                // This covers both regular AiRequest inference and node-issued challenge inference.
                if self.inference_rx.is_some() || self.challenge_inference_rx.is_some() {
                    miner.process_block(None).await?;
                    return Ok(());
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
                    (_, false, None) => miner.process_block(None).await?,
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
                    // Validation answer: "cannot find header <hash>" — the block never
                    // existed on this chain, its escrow entries are ghosts.
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .map_or(false, |w| w.on_block_validation_error(&e.message));
                    if !was_validation {
                        warn!("GetBlockResponse error: {}", e.message);
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                    // Chain membership from the node's live verdict: a stored-but-reorged
                    // block must purge its entries just like a missing one.
                    let is_chain = block.verbose_data.as_ref().map_or(false, |v| v.is_chain_block);
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .map_or(false, |w| w.consume_validation_ok(&hash, is_chain));
                    if !was_validation {
                        self.scan_txs_for_ai_requests(&block.transactions);
                        self.try_start_inference();
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
                // Escrow claims and OPoI submissions share this stream. Match responses to
                // in-flight claims by identity (txid, or the txid embedded in the rejection
                // text) — attributing by position slashed valid escrow entries before.
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
                        if let Some(e) = err {
                            log::debug!("OPoI: submit_transaction error: {}", e);
                        }
                    }
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                // Register for all notification types:
                // - NewBlockTemplate drives the mining loop
                // - BlockAdded lets us scan confirmed blocks for AiRequests
                //   that were confirmed before the miner saw them in mempool
                // - VirtualChainChanged drives escrow tracking: only chain-block coinbases
                //   materialize UTXOs, so escrow outputs are tracked from chain blocks only
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_send(NotifyBlockAddedRequestMessage {}).await?;
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage {}).await?;
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
                None => info!("Registered for virtual chain notifications (escrow tracking)"),
                Some(e) => error!("Failed registering for virtual chain notifications: {:?}", e),
            },
            // Virtual chain advanced: fetch every added chain block in full. Their coinbases
            // are the only ones that materialize UTXOs, so escrow tracking feeds off this
            // stream (handle_block gates tracking on is_chain_block). Removed chain blocks
            // are ignored: entries from reorged-out blocks fail their claims as orphans and
            // are cleaned up by the existing retry/slash machinery.
            Payload::VirtualSelectedParentChainChangedNotification(notif) => {
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
    }
}
