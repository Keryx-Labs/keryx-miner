// Automated OPoI escrow claim module.
//
// After each block, scans for coinbase outputs matching this miner's escrow script.
// When the CSV window (36 000 blocks) expires, builds a Schnorr-signed claim TX and
// broadcasts it via gRPC.  State is persisted to `escrow_state.json` so claims survive
// miner restarts.

use blake2b_simd::Params as Blake2bParams;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{fs, io};

use crate::proto::{
    RpcOutpoint, RpcScriptPublicKey, RpcTransaction, RpcTransactionInput, RpcTransactionOutput,
};

const CHALLENGE_WINDOW_BLOCKS: u64 = 36_000;
const CLAIM_FEE_SOMPI: u64 = 30_000_000;
const NATIVE_SUBNETWORK: &str = "0000000000000000000000000000000000000000";

const OP_CSV: u8 = 0xb1;
const OP_CHECKSIG: u8 = 0xac;
const SIG_HASH_ALL: u8 = 0x01;

/// Drop done (claimed / terminally-slashed) entries every N processed blocks so the
/// in-memory vector and the on-disk state stay bounded under a high block rate.
const COMPACT_EVERY_BLOCKS: u32 = 2_000;
/// Minimum wall-clock interval between state-file writes. Without this debounce the
/// watcher rewrites the entire (multi-thousand-entry) state file on every block, which
/// saturates the async client loop and starves block-template delivery → mining stalls.
const STATE_SAVE_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EscrowEntry {
    /// TXID of the source transaction: coinbase txid for block rewards,
    /// or AiRequest txid for inference escrow entries.
    pub coinbase_txid: String,
    /// Block hash of the coinbase block — used to detect red-set exclusion.
    /// Empty for inference escrow entries (non-coinbase TXs may be re-included).
    #[serde(default)]
    pub block_hash: String,
    pub confirm_daa: u64,
    pub amount_sompi: u64,
    /// Index of the escrow output in the source transaction.
    /// Coinbases: varies by mergeset size. Inference: always 1 (output[1]).
    #[serde(default = "default_output_index")]
    pub output_index: u32,
    pub claimed: bool,
    pub slashed: bool,
    #[serde(default)]
    pub orphan_slashed: bool,
    #[serde(default)]
    pub orphan_retries: u8,
    /// Generic retry cooldown (DAA score): set after orphan, sequence-lock and
    /// unrecognized rejections alike. The entry is skipped until this score passes.
    #[serde(default)]
    pub orphan_retry_after_daa: Option<u64>,
    /// Consecutive unrecognized rejections — drives an exponential retry backoff.
    /// Never leads to a permanent slash: an unspent escrow UTXO stays claimable.
    #[serde(default)]
    pub submit_retries: u8,
    /// Max claim-batch size this entry may join (0 = uncapped). Halved every time a
    /// batch containing this entry is rejected: batches bisect until dead inputs are
    /// isolated into solo claims and valid inputs re-group into accepted batches.
    /// The cap is TRANSIENT: unless re-confirmed by a fresh rejection it expires after
    /// CAP_EXPIRY_DAA (see find_claim) and the entry returns to the full-batch flow.
    #[serde(default)]
    pub batch_cap: u8,
    /// DAA score at which batch_cap was last set/halved — drives cap expiry.
    #[serde(default)]
    pub cap_set_daa: u64,
    /// True for AI inference escrow entries (from AiRequest output[1]).
    /// Red-set slashing is skipped for these since non-coinbase TXs can be re-included.
    #[serde(default)]
    pub is_inference: bool,
}

fn default_output_index() -> u32 {
    1
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct EscrowState {
    pub entries: Vec<EscrowEntry>,
}

/// Max escrow outputs per claim TX. Compute mass is `506 + 1118 * inputs` grams (1,000 per
/// sig-op + 118 serialized bytes per input, plus the single-output overhead) against the
/// node's 100,000-gram standard-TX cap, so 88 inputs is the hard ceiling (89 lands at
/// 100,008 and is rejected). 87 (97,772 grams) keeps one input of margin under the cap.
/// Transient mass (size * 4) and storage mass are never binding here: an N-to-1
/// consolidation has zero storage mass under KIP-9.
/// Batching also amortizes the flat CLAIM_FEE_SOMPI across the whole batch instead of
/// paying it per output — the mass-derived minimum fee (~90k sompi at this size) stays far
/// below the node's flat 0.3 KRX floor, so a bigger batch costs exactly the same.
const MAX_CLAIM_BATCH: usize = 87;
/// Minimum outputs per claim TX: the flat claim fee is amortized across the batch, so
/// nominal claims ship ONLY as full MAX_CLAIM_BATCH batches — partial batches are never
/// submitted, no matter how long the entries have been waiting. Waiting costs nothing
/// beyond deferred liquidity — the outputs are matured UTXOs sitting in the UTXO set,
/// with no expiry and no slash risk. Repair batches (bisection caps, dead-pool grinding)
/// are exempt and keep flowing at any size so dead inputs still get isolated.
const MIN_CLAIM_BATCH: usize = MAX_CLAIM_BATCH;
/// A repair verdict (bisection cap, orphan-death flag) not re-confirmed by a fresh
/// rejection for this many DAA (~1 h at 10 BPS) expires and the entry returns to the
/// nominal full-batch flow. Bisection isolates a dead input within minutes; a verdict
/// that outlives that by an hour is stale state (e.g. inherited from a network
/// incident), and letting it persist would turn the repair path into a permanent
/// small-batch claim regime. The orphan_retries counter survives forgiveness, so real
/// dead entries still converge to the permanent slash across forgiveness cycles.
const CAP_EXPIRY_DAA: u64 = 36_000;
/// Max claim TXs awaiting a SubmitTransactionResponse at once.
const MAX_IN_FLIGHT_CLAIMS: usize = 4;

/// Result of matching a SubmitTransactionResponse against the in-flight claim TXs.
pub enum SubmitResponseOutcome {
    /// The response belongs to other traffic (OPoI submissions) — not a claim of ours.
    NotOurs,
    /// A claim was matched (rejected/retried); no outputs were finalized.
    Handled,
    /// A claim was matched and accepted: its outputs are now claimed.
    Accepted { outputs: u64, amount_sompi: u64 },
}
/// A claim with no response after this many DAA (~1 min at 10 BPS) is released for
/// retry. Claim TXs are deterministic (same inputs → same txid), so a blind retry of a
/// claim that actually went through is rejected as a duplicate and marked claimed —
/// a lost response can no longer wedge the queue for the life of the process.
const CLAIM_RESPONSE_TIMEOUT_DAA: u64 = 600;
/// Emit a status line at most every N processed blocks (~100 s at 10 BPS) while there
/// is claim work outstanding. Debug level, like all claim machinery — at the default
/// INFO level only claim acceptances are logged.
const STATUS_EVERY_BLOCKS: u32 = 1_000;
/// Base DAA cooldown for unrecognized rejections; doubles per consecutive failure.
const UNKNOWN_RETRY_BASE_COOLDOWN_DAA: u64 = 100;
/// Cap for the exponential unknown-rejection backoff (~1.4 h at 10 BPS).
const UNKNOWN_RETRY_MAX_COOLDOWN_DAA: u64 = 50_000;

/// DAA-score cooldown after an orphan rejection before retrying the claim.
const ORPHAN_RETRY_COOLDOWN_BLOCKS: u64 = 100;
/// DAA-score cooldown after a sequence-lock rejection before retrying.
/// The OP_CSV check uses the selected-chain blue score, which can lag the DAA
/// score by a few blocks in the DAG — a short cooldown avoids hammering the
/// node every block for a TX that will be rejected again immediately.
const SEQ_LOCK_RETRY_COOLDOWN_BLOCKS: u64 = 20;
/// After this many consecutive orphan rejections, give up and slash permanently.
const MAX_ORPHAN_RETRIES: u8 = 10;

pub struct EscrowWatcher {
    secp: secp256k1::Secp256k1<secp256k1::All>,
    secret_key: secp256k1::SecretKey,
    pubkey_bytes: [u8; 32],
    escrow_script: Vec<u8>,
    escrow_script_hex: String,
    payout_spk_version: u16,
    payout_spk_script: Vec<u8>,
    payout_spk_script_hex: String,
    pub state: EscrowState,
    state_path: PathBuf,
    /// Claim TXs submitted and awaiting their SubmitTransactionResponse, keyed by the
    /// claim txid (computed at build time). Entries expire after CLAIM_RESPONSE_TIMEOUT_DAA
    /// so a lost response releases its outputs instead of wedging the queue forever.
    /// In-memory only: a reconnect rebuilds the watcher and clears it, which is correct —
    /// responses for the old connection can no longer arrive.
    in_flight: HashMap<String, InFlightClaim>,
    /// "{txid}:{index}" of every outpoint inside an in-flight claim, so eligibility
    /// checks stay O(1).
    in_flight_outpoints: HashSet<String>,
    /// DAA score of the most recent block seen — used to set per-entry orphan cooldowns.
    last_daa_score: u64,
    /// O(1) dedup of tracked outpoints ("{txid}:{output_index}") so tracking a coinbase
    /// output no longer scans all of `state.entries`. Rebuilt from `state` on load/compaction.
    outpoint_set: HashSet<String>,
    /// block_hash -> indices into `state.entries`, for O(reds) red-set slashing instead of
    /// scanning every entry per red hash. Rebuilt on load/compaction; appended on track.
    block_index: HashMap<String, Vec<usize>>,
    /// Debounced persistence: pending unsaved changes + last write time.
    dirty: bool,
    last_save: Instant,
    /// handle_block call counter, used to trigger periodic compaction.
    blocks_since_compact: u32,
    /// handle_block call counter for the periodic INFO status line.
    blocks_since_status: u32,
    /// Block hashes still awaiting boot-time existence validation against the node
    /// (GetBlock round-trip). While non-empty, claim building is disabled so no batch
    /// ships with ghost entries — coinbases of blocks orphaned by a network incident
    /// (e.g. a wedge re-join) whose UTXOs never existed on the surviving chain.
    validation_pending: HashSet<String>,
    /// Entries purged by boot-time validation, for the completion log line.
    validation_purged: u64,
}

/// A claim TX submitted to the node, awaiting its SubmitTransactionResponse.
struct InFlightClaim {
    /// (source txid, output_index) of every escrow output the claim spends.
    outpoints: Vec<(String, u32)>,
    /// DAA score at submission — drives the response timeout.
    submit_daa: u64,
}

impl EscrowWatcher {
    pub fn new(privkey_hex: &str, mining_address: &str, state_path: PathBuf) -> Result<Self, String> {
        let privkey_bytes = hex::decode(privkey_hex)
            .map_err(|e| format!("Invalid --mining-privkey hex: {}", e))?;
        if privkey_bytes.len() != 32 {
            return Err(format!("--mining-privkey must be 32 bytes (64 hex chars), got {}", privkey_bytes.len()));
        }

        let secp = secp256k1::Secp256k1::new();
        let secret_key = secp256k1::SecretKey::from_slice(&privkey_bytes)
            .map_err(|e| format!("Invalid private key: {}", e))?;
        let keypair = secp256k1::Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let pubkey_bytes: [u8; 32] = xonly.serialize();

        let escrow_script = build_escrow_script(&pubkey_bytes);
        let escrow_script_hex = hex::encode(&escrow_script);

        let (payout_spk_version, payout_spk_bytes) = decode_address(mining_address)?;
        let payout_spk_script = build_p2pk_script(&payout_spk_bytes);
        let payout_spk_script_hex = hex::encode(&payout_spk_script);

        let state = load_state(&state_path);

        info!("EscrowWatcher ready: pubkey={}", hex::encode(pubkey_bytes));

        let mut watcher = Self {
            secp,
            secret_key,
            pubkey_bytes,
            escrow_script,
            escrow_script_hex,
            payout_spk_version,
            payout_spk_script,
            payout_spk_script_hex,
            state,
            state_path,
            in_flight: HashMap::new(),
            in_flight_outpoints: HashSet::new(),
            last_daa_score: 0,
            outpoint_set: HashSet::new(),
            block_index: HashMap::new(),
            dirty: false,
            last_save: Instant::now(),
            blocks_since_compact: 0,
            blocks_since_status: 0,
            validation_pending: HashSet::new(),
            validation_purged: 0,
        };
        watcher.rebuild_indexes();
        Ok(watcher)
    }

    /// (Re)build the in-memory lookup indexes from `state.entries`.
    fn rebuild_indexes(&mut self) {
        self.outpoint_set.clear();
        self.block_index.clear();
        for (i, e) in self.state.entries.iter().enumerate() {
            self.outpoint_set.insert(format!("{}:{}", e.coinbase_txid, e.output_index));
            if !e.block_hash.is_empty() {
                self.block_index.entry(e.block_hash.clone()).or_default().push(i);
            }
        }
    }

    /// Drop done (claimed / terminally-slashed) entries and rebuild indexes.
    /// Orphan-retry entries (slashed == false) are kept.
    fn compact(&mut self) {
        let before = self.state.entries.len();
        self.state.entries.retain(|e| !e.claimed && !e.slashed);
        if self.state.entries.len() != before {
            self.rebuild_indexes();
            self.mark_dirty();
        }
    }

    #[inline]
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Persist state at most once per `STATE_SAVE_INTERVAL` to keep disk I/O off the
    /// per-block hot path. Worst case on crash we lose a couple of seconds of tracking,
    /// which is re-derived from the chain on the next blocks.
    fn maybe_flush(&mut self) {
        if self.dirty && self.last_save.elapsed() >= STATE_SAVE_INTERVAL {
            self.save_state();
            self.dirty = false;
            self.last_save = Instant::now();
        }
    }

    /// Return the 64-char hex x-only public key of the mining key.
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.pubkey_bytes)
    }

    /// V2 responder identity for an AiResponse: schnorr signature with the escrow key over the
    /// domain-hashed v1 payload bytes — MUST match the node's `verified_responder`
    /// (blake2b-256("KeryxServiceResponderV1" || signed_bytes)).
    pub fn sign_responder(&self, signed_bytes: &[u8]) -> keryx_inference::AiResponder {
        let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
        hasher.update(b"KeryxServiceResponderV1");
        hasher.update(signed_bytes);
        let msg = secp256k1::Message::from_digest_slice(hasher.finalize().as_bytes()).unwrap();
        let keypair = secp256k1::Keypair::from_secret_key(&self.secp, &self.secret_key);
        let sig = self.secp.sign_schnorr_no_aux_rand(&msg, &keypair);
        keryx_inference::AiResponder { escrow_pubkey: self.pubkey_bytes, signature: *sig.as_ref() }
    }

    /// Scan a confirmed block for the miner's escrow output and check for mature claims.
    /// Returns a claim TX to submit if one is ready; `None` otherwise.
    pub fn handle_block(&mut self, block: &crate::proto::RpcBlock) -> Option<RpcTransaction> {
        let daa_score = block.header.as_ref()?.daa_score;
        self.last_daa_score = daa_score;

        // Release claims whose response never arrived so the queue cannot wedge.
        self.expire_in_flight(daa_score);

        // Periodic compaction so the entry vector / state file stay bounded under a flood.
        self.blocks_since_compact += 1;
        if self.blocks_since_compact >= COMPACT_EVERY_BLOCKS {
            self.blocks_since_compact = 0;
            self.compact();
        }

        // Periodic status line while claim work is outstanding (debug, like the rest of
        // the claim machinery — INFO only reports claim acceptances).
        self.blocks_since_status += 1;
        if self.blocks_since_status >= STATUS_EVERY_BLOCKS {
            self.blocks_since_status = 0;
            let mature = self
                .state
                .entries
                .iter()
                .filter(|e| !e.claimed && !e.slashed && daa_score >= e.confirm_daa + CHALLENGE_WINDOW_BLOCKS + 10)
                .count();
            if mature > 0 || !self.in_flight.is_empty() {
                debug!(
                    "EscrowWatcher: {} tracked, {} mature awaiting claim, {} claim TX(s) in flight",
                    self.state.entries.len(),
                    mature,
                    self.in_flight.len()
                );
            }
        }

        let block_hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();

        // Proactively slash entries whose coinbase block just entered the red set.
        // mergeSetRedsHashes lists blocks newly classified as red by this block's GHOSTDAG —
        // their coinbase UTXOs will never exist in the virtual UTXO set.
        // O(reds) via block_index instead of scanning all entries per red hash.
        if let Some(verbose) = &block.verbose_data {
            for red_hash in &verbose.merge_set_reds_hashes {
                let Some(indices) = self.block_index.get(red_hash).cloned() else { continue };
                for i in indices {
                    if let Some(entry) = self.state.entries.get_mut(i) {
                        if !entry.claimed && !entry.slashed && !entry.is_inference && !entry.block_hash.is_empty() {
                            debug!(
                                "EscrowWatcher: block {} is red — permanently skipping coinbase={}…",
                                &red_hash[..16.min(red_hash.len())],
                                &entry.coinbase_txid[..16.min(entry.coinbase_txid.len())]
                            );
                            entry.slashed = true;
                            self.mark_dirty();
                        }
                    }
                }
            }
        }

        // Track new escrow outputs from CHAIN blocks only. Every block's coinbase pays
        // its mergeset's blues, but only chain-block coinbases enter the UTXO set —
        // an escrow output seen in a non-chain block's coinbase is a phantom outpoint
        // whose claim can only be rejected as an orphan. Chain blocks arrive in full via
        // the VirtualChainChanged → GetBlock path; BlockAdded blocks carry
        // is_chain_block=false at notification time and are skipped here.
        let is_chain_block = block.verbose_data.as_ref().map_or(false, |v| v.is_chain_block);
        if is_chain_block {
            self.track_block_escrows(block, daa_score, &block_hash);
        }

        let claim = self.find_claim(daa_score);
        self.maybe_flush();
        claim
    }

    /// Scan a chain block's coinbase for outputs paying our escrow script and track them.
    fn track_block_escrows(&mut self, block: &crate::proto::RpcBlock, daa_score: u64, block_hash: &str) {
        // Find coinbase TX (no inputs).
        let Some(coinbase) = block.transactions.iter().find(|tx| tx.inputs.is_empty()) else { return };
        let Some(coinbase_txid) = coinbase.verbose_data.as_ref().map(|v| v.transaction_id.clone()) else {
            return;
        };
        if coinbase_txid.is_empty() {
            return;
        }

        // Scan all coinbase outputs — a multi-blue mergeset produces one escrow output
        // per blue block, each at a different index.  Hardcoding index 1 would miss all
        // escrow outputs beyond the first blue's pair. O(1) dedup via outpoint_set.
        for (out_idx, output) in coinbase.outputs.iter().enumerate() {
            if let Some(spk) = &output.script_public_key {
                let key = format!("{}:{}", coinbase_txid, out_idx);
                if spk.script_public_key.to_lowercase() == self.escrow_script_hex
                    && spk.version == 0
                    && !self.outpoint_set.contains(&key)
                {
                    debug!(
                        "EscrowWatcher: tracked escrow coinbase={}…[{}] daa={} amount={}",
                        &coinbase_txid[..16.min(coinbase_txid.len())],
                        out_idx,
                        daa_score,
                        output.amount
                    );
                    let idx = self.state.entries.len();
                    self.state.entries.push(EscrowEntry {
                        coinbase_txid: coinbase_txid.clone(),
                        block_hash: block_hash.to_string(),
                        confirm_daa: daa_score,
                        amount_sompi: output.amount,
                        output_index: out_idx as u32,
                        claimed: false,
                        slashed: false,
                        orphan_slashed: false,
                        orphan_retries: 0,
                        orphan_retry_after_daa: None,
                        submit_retries: 0,
                        batch_cap: 0,
                        cap_set_daa: 0,
                        is_inference: false,
                    });
                    self.outpoint_set.insert(key);
                    if !block_hash.is_empty() {
                        self.block_index.entry(block_hash.to_string()).or_default().push(idx);
                    }
                    self.mark_dirty();
                }
            }
        }
    }

    /// Count and sum the escrow outputs still awaiting claim: tracked, not claimed, and
    /// not proven dead (entries solo-rejected as orphans are excluded — including them
    /// would inflate the figure with outpoints that will never pay).
    pub fn pending_escrow(&self) -> (u64, u64) {
        let mut outputs = 0u64;
        let mut sompi = 0u64;
        for e in &self.state.entries {
            if !e.claimed && !e.slashed && !e.orphan_slashed && e.orphan_retries == 0 {
                outputs += 1;
                sompi += e.amount_sompi;
            }
        }
        (outputs, sompi)
    }

    /// Release in-flight claims whose response never arrived (connection hiccup, node
    /// restart, lost message). Their outputs become eligible again; because claim TXs are
    /// deterministic, retrying one that actually went through is rejected as a duplicate
    /// and resolves to claimed via on_submit_response.
    fn expire_in_flight(&mut self, daa_score: u64) {
        let expired: Vec<String> = self
            .in_flight
            .iter()
            .filter(|(_, c)| daa_score >= c.submit_daa + CLAIM_RESPONSE_TIMEOUT_DAA)
            .map(|(txid, _)| txid.clone())
            .collect();
        for txid in expired {
            let claim = self.in_flight.remove(&txid).unwrap();
            debug!(
                "EscrowWatcher: no response for claim {} after {} DAA — releasing {} output(s) for retry",
                txid,
                CLAIM_RESPONSE_TIMEOUT_DAA,
                claim.outpoints.len()
            );
            for (t, i) in &claim.outpoints {
                self.in_flight_outpoints.remove(&format!("{}:{}", t, i));
            }
        }
    }

    /// Start boot-time state validation: returns every distinct block hash referenced by
    /// a live entry, for the caller to check against the node with GetBlock. Claim
    /// building stays disabled until every hash has been answered, so no batch ships
    /// with ghost entries. Inference entries (empty block_hash) are not checkable this
    /// way and are left alone.
    pub fn start_state_validation(&mut self) -> Vec<String> {
        let mut hashes: HashSet<String> = HashSet::new();
        for e in &self.state.entries {
            if !e.claimed && !e.slashed && !e.block_hash.is_empty() {
                hashes.insert(e.block_hash.clone());
            }
        }
        self.validation_pending = hashes.clone();
        self.validation_purged = 0;
        if !self.validation_pending.is_empty() {
            info!(
                "EscrowWatcher: validating {} block(s) against the node before claiming — ghost entries will be purged",
                self.validation_pending.len()
            );
        }
        hashes.into_iter().collect()
    }

    /// Record a validation answer for one block hash. `exists == false` purges every
    /// live entry of that block: its coinbase never existed on the surviving chain, so
    /// claiming it can only poison a batch.
    pub fn on_block_validated(&mut self, hash: &str, exists: bool) {
        if !self.validation_pending.remove(hash) {
            return;
        }
        if !exists {
            if let Some(indices) = self.block_index.get(hash) {
                for &i in indices {
                    let e = &mut self.state.entries[i];
                    if !e.claimed && !e.slashed {
                        e.slashed = true;
                        self.validation_purged += 1;
                    }
                }
            }
            self.mark_dirty();
        }
        if self.validation_pending.is_empty() {
            info!(
                "EscrowWatcher: state validation complete — {} ghost entr{} purged, claiming enabled",
                self.validation_purged,
                if self.validation_purged == 1 { "y" } else { "ies" }
            );
            self.maybe_flush();
        }
    }

    /// True while boot-time validation is still awaiting node answers.
    pub fn validation_in_progress(&self) -> bool {
        !self.validation_pending.is_empty()
    }

    /// Match a GetBlock error message against the pending validation set (the node's
    /// "cannot find header <hash>" text embeds the hash). Returns true when consumed:
    /// the block does not exist on this chain, its entries are ghosts and get purged.
    pub fn on_block_validation_error(&mut self, message: &str) -> bool {
        let hash = match self.validation_pending.iter().find(|h| message.contains(h.as_str())) {
            Some(h) => h.clone(),
            None => return false,
        };
        self.on_block_validated(&hash, false);
        true
    }

    /// Consume a successful GetBlock answer for a pending validation hash. Returns true
    /// when the response belonged to the validation flow (the caller then skips the
    /// regular block-scan path — validation responses carry no transactions).
    ///
    /// `is_chain_block` is the node's CURRENT verdict: escrow entries come from chain-block
    /// coinbases, so a block that got reorged out of the selected chain (still stored,
    /// no longer chain — invisible to a pure existence check, especially on archival
    /// nodes) never materialized its coinbase and its entries are ghosts too. Entries are
    /// at least a challenge-window old when claimed, so the chain verdict is final here.
    pub fn consume_validation_ok(&mut self, hash: &str, is_chain_block: bool) -> bool {
        if self.validation_pending.contains(hash) {
            self.on_block_validated(hash, is_chain_block);
            true
        } else {
            false
        }
    }

    /// Scan for matured, eligible escrow entries and build a batched claim TX (if any).
    fn find_claim(&mut self, daa_score: u64) -> Option<RpcTransaction> {
        if self.in_flight.len() >= MAX_IN_FLIGHT_CLAIMS {
            return None;
        }
        // No claims while the state is still being validated against the node.
        if self.validation_in_progress() {
            return None;
        }

        // Repair verdicts are transient: a bisection cap or an orphan-death verdict that
        // has not been re-confirmed by a fresh rejection for CAP_EXPIRY_DAA expires and
        // the entry returns to the nominal full-batch flow. A network incident can mark
        // thousands of perfectly live entries dead in one sweep (one orphan rejection
        // each); grinding those through solo claims would take days of 1-output TXs.
        // Forgiveness is safe because orphan_retries is NOT reset: a genuinely dead
        // entry re-fails its full batch, re-bisects to a solo rejection, increments the
        // counter and still converges to the permanent slash at MAX_ORPHAN_RETRIES.
        let mut healed = false;
        for e in self.state.entries.iter_mut() {
            if e.batch_cap != 0 && daa_score >= e.cap_set_daa + CAP_EXPIRY_DAA {
                e.batch_cap = 0;
                healed = true;
            }
            // Forgiveness backs off exponentially with the solo-rejection count: an entry
            // rejected once (network hiccup) is retried in a full batch after ~2 h, but a
            // repeat offender (a ghost outpoint, e.g. inherited from recover-escrow built
            // on a lying indexer) stays quarantined in the dead pool for exponentially
            // longer instead of poisoning a fresh 80-output batch every hour — the solo
            // grinding path slashes it permanently at MAX_ORPHAN_RETRIES meanwhile.
            if e.orphan_slashed {
                let holdoff = CAP_EXPIRY_DAA.saturating_mul(1u64 << e.orphan_retries.min(10) as u32);
                if e.orphan_retry_after_daa.map_or(true, |retry_daa| daa_score >= retry_daa.saturating_add(holdoff)) {
                    e.orphan_slashed = false;
                    healed = true;
                }
            }
        }
        if healed {
            self.mark_dirty();
        }

        // +10 margin: OP_CSV validation uses the selected-chain blue score, which can lag the
        // virtual-state DAA score by several blocks in the BlockDAG.  A single +1 margin is
        // often not enough, causing seq-lock rejections that get retried every block.
        // Per-entry cooldowns are checked individually so they don't block other claims.
        //
        // Selection runs in priority passes; each pass builds its own candidate batch and
        // the first RELEASABLE one wins, so a held-back pass never starves the ones below:
        //   0 — nominal: uncapped live entries, inference first within the batch (they
        //       carry user fees and must not be starved by the coinbase queue). Releasable
        //       ONLY as a full MAX_CLAIM_BATCH batch — the flat claim fee is amortized
        //       across the whole TX and partial batches never ship. Nominal goes FIRST so
        //       full batches are never delayed behind repair grinding;
        //   1 — repair: live entries carrying a bisection cap (survivors of a rejected
        //       batch). They regroup among themselves — never with uncapped entries, which
        //       would bleed fresh outputs into small immediate batches — and flow at any
        //       size (in the gaps between full batches) so dead inputs get isolated;
        //   2 — the known-dead pool (solo-orphaned at least once), ground down whenever
        //       nothing fresher is claimable.
        // A batch never mixes passes: batching live entries with known-dead ones would let
        // one dead input orphan the whole TX and stall the live entries.
        let mut batch: Vec<usize> = Vec::new();
        let mut selected = false;
        for pass in 0..3u8 {
            batch.clear();
            let mut limit = MAX_CLAIM_BATCH;
            // Nominal pass: iterate inference entries first so they get batch priority;
            // the sort is stable, so queue order is preserved within each kind.
            let mut indices: Vec<usize> = (0..self.state.entries.len()).collect();
            if pass == 0 {
                indices.sort_by_key(|&i| !self.state.entries[i].is_inference);
            }
            for &i in &indices {
                let e = &self.state.entries[i];
                // Only the (transient, forgivable) flag decides the dead pool; the
                // orphan_retries counter is the cumulative memory driving the permanent
                // slash and must not keep an entry in the dead pool forever on its own.
                let proven_dead = e.orphan_slashed;
                let in_pass = match pass {
                    0 => e.batch_cap == 0 && !proven_dead,
                    1 => e.batch_cap != 0 && !proven_dead,
                    _ => proven_dead,
                };
                if !in_pass {
                    continue;
                }
                let eligible = !e.claimed
                    && !e.slashed
                    && daa_score >= e.confirm_daa + CHALLENGE_WINDOW_BLOCKS + 10
                    && e.orphan_retry_after_daa.map_or(true, |retry_daa| daa_score >= retry_daa)
                    && !self.in_flight_outpoints.contains(&format!("{}:{}", e.coinbase_txid, e.output_index));
                if !eligible {
                    continue;
                }
                let cap = if e.batch_cap == 0 { MAX_CLAIM_BATCH } else { (e.batch_cap as usize).min(MAX_CLAIM_BATCH) };
                if cap.min(limit) < batch.len() + 1 {
                    continue; // joining would violate this entry's cap (or shrink below current size)
                }
                limit = limit.min(cap);
                batch.push(i);
                if batch.len() >= limit {
                    break; // batch is full for this pass
                }
            }
            let releasable = match pass {
                // Nominal batches ship full or not at all (fee amortization).
                0 => batch.len() >= MIN_CLAIM_BATCH,
                // Repair and dead-pool batches flow at any size.
                _ => !batch.is_empty(),
            };
            if releasable {
                selected = true;
                break;
            }
        }
        if !selected {
            return None;
        }
        let entries: Vec<EscrowEntry> = batch.iter().map(|&i| self.state.entries[i].clone()).collect();

        match self.build_claim_tx(&entries) {
            Ok((claim_txid, tx)) => {
                let total: u64 = entries.iter().map(|e| e.amount_sompi).sum();
                debug!(
                    "EscrowWatcher: claiming {} escrow output(s), {:.8} KRX total (claim txid={})",
                    entries.len(),
                    total as f64 / 1e8,
                    claim_txid
                );
                let outpoints: Vec<(String, u32)> =
                    entries.iter().map(|e| (e.coinbase_txid.clone(), e.output_index)).collect();
                for (t, i) in &outpoints {
                    self.in_flight_outpoints.insert(format!("{}:{}", t, i));
                }
                self.in_flight.insert(claim_txid, InFlightClaim { outpoints, submit_daa: daa_score });
                Some(tx)
            }
            Err(e) => {
                debug!("EscrowWatcher: failed to build claim TX: {}", e);
                None
            }
        }
    }

    /// Called for every SubmitTransactionResponse on the gRPC stream. The response is
    /// matched to one of our in-flight claim TXs — by transaction_id for accepted TXs;
    /// error responses carry an empty transaction_id (the node's error path returns a
    /// default message), so the rejection text, which embeds the offending txid, is
    /// matched against in-flight claim txids instead. Returns `NotOurs` for responses
    /// that belong to other traffic (OPoI submissions), so the caller can log those
    /// itself, and `Accepted` with the claimed totals so the caller can feed stats.
    pub fn on_submit_response(&mut self, response_txid: &str, error: Option<&str>) -> SubmitResponseOutcome {
        let matched_txid = if self.in_flight.contains_key(response_txid) {
            Some(response_txid.to_string())
        } else if let Some(msg) = error {
            self.in_flight.keys().find(|t| msg.contains(t.as_str())).cloned()
        } else {
            None
        };
        let claim_txid = match matched_txid {
            Some(t) => t,
            None => return SubmitResponseOutcome::NotOurs,
        };
        let claim = self.in_flight.remove(&claim_txid).unwrap();
        for (t, i) in &claim.outpoints {
            self.in_flight_outpoints.remove(&format!("{}:{}", t, i));
        }
        let n_outputs = claim.outpoints.len();

        let mut outcome = SubmitResponseOutcome::Handled;
        match error {
            None => {
                info!("EscrowWatcher: claim accepted ({} output(s), txid={})", n_outputs, claim_txid);
                let amount_sompi = self.mark_entries_claimed(&claim.outpoints);
                outcome = SubmitResponseOutcome::Accepted { outputs: n_outputs as u64, amount_sompi };
            }
            // Claim TXs are deterministic (same inputs → same txid), so a retry of one that
            // actually went through is rejected as a duplicate — that IS the lost-response
            // success path.
            Some(msg) if msg.contains("already in the mempool") || msg.contains("already accepted") => {
                info!(
                    "EscrowWatcher: claim accepted ({} output(s), txid={} — confirmed via duplicate rejection)",
                    n_outputs, claim_txid
                );
                let amount_sompi = self.mark_entries_claimed(&claim.outpoints);
                outcome = SubmitResponseOutcome::Accepted { outputs: n_outputs as u64, amount_sompi };
            }
            Some(msg) => {
                // Retriable rejections: sequence-lock timing races and orphan/dag-reorg
                // situations where the source block is off the selected chain.
                let is_orphan = msg.contains("orphan");
                let is_seq_lock = msg.contains("sequence lock");
                let batch_rejected = n_outputs > 1;
                let last_daa = self.last_daa_score;
                // Bisection step: one dead input orphans the whole batch, but most members
                // are usually fine. Halve every member's cap and retry as smaller batches —
                // valid entries re-group into accepted batches within log2(batch) rounds,
                // dead ones converge to solo claims and get slashed there.
                let halved_cap = ((n_outputs / 2).max(1)).min(u8::MAX as usize) as u8;
                for (t, i) in &claim.outpoints {
                    let e = match self
                        .state
                        .entries
                        .iter_mut()
                        .find(|e| e.coinbase_txid == *t && e.output_index == *i)
                    {
                        Some(e) => e,
                        None => continue,
                    };
                    if is_orphan {
                        if batch_rejected {
                            // No retry count and no cooldown here: retries are only counted
                            // on solo claims, so a valid entry repeatedly batched with a
                            // dead one can never be slashed by association, and the halves
                            // retry immediately (bisection strictly shrinks, so it ends).
                            e.batch_cap = halved_cap;
                            e.cap_set_daa = last_daa;
                        } else {
                            e.orphan_retries += 1;
                            if e.orphan_retries >= MAX_ORPHAN_RETRIES {
                                e.orphan_slashed = false;
                                e.slashed = true;
                                debug!(
                                    "EscrowWatcher: coinbase={}[{}] slashed after {} orphan retries",
                                    t, i, e.orphan_retries
                                );
                            } else {
                                e.orphan_slashed = true;
                                // Per-entry cooldown: only this entry waits, other claims proceed.
                                e.orphan_retry_after_daa = Some(last_daa + ORPHAN_RETRY_COOLDOWN_BLOCKS);
                            }
                        }
                    } else if is_seq_lock {
                        // The OP_CSV blue-score check may lag DAA score by a few blocks.
                        e.orphan_retry_after_daa = Some(last_daa + SEQ_LOCK_RETRY_COOLDOWN_BLOCKS);
                    } else {
                        // Unrecognized rejection: bisect too (a size-related rejection heals
                        // that way) with exponential backoff, never a permanent slash — an
                        // unspent escrow UTXO stays claimable, and deriving an irreversible
                        // state from an error string has lost real funds before.
                        if batch_rejected {
                            e.batch_cap = halved_cap;
                            e.cap_set_daa = last_daa;
                        }
                        e.submit_retries = e.submit_retries.saturating_add(1);
                        let cooldown = UNKNOWN_RETRY_BASE_COOLDOWN_DAA
                            .saturating_mul(1u64 << (e.submit_retries.min(9) as u32))
                            .min(UNKNOWN_RETRY_MAX_COOLDOWN_DAA);
                        e.orphan_retry_after_daa = Some(last_daa + cooldown);
                    }
                }
                // Debug level: with boot-time state validation in place, rejections are
                // rare and transient (bisection repair) — not worth operator noise.
                debug!(
                    "EscrowWatcher: claim {} rejected ({} output(s) released{}): {}",
                    claim_txid,
                    n_outputs,
                    if batch_rejected { ", batch cap halved" } else { "" },
                    msg
                );
            }
        }
        self.mark_dirty();
        self.maybe_flush();
        outcome
    }

    /// Mark every entry matching the given outpoints as claimed. Returns the total amount
    /// (sompi) newly marked, for stats reporting.
    fn mark_entries_claimed(&mut self, outpoints: &[(String, u32)]) -> u64 {
        let mut total_sompi = 0u64;
        for (t, i) in outpoints {
            if let Some(e) = self
                .state
                .entries
                .iter_mut()
                .find(|e| e.coinbase_txid == *t && e.output_index == *i)
            {
                if !e.claimed {
                    total_sompi += e.amount_sompi;
                }
                e.claimed = true;
            }
        }
        total_sompi
    }

    /// Build one claim TX spending every given escrow output to the payout address.
    /// All inputs share the miner's escrow script; each carries its own signature and the
    /// CSV sequence. The flat CLAIM_FEE_SOMPI is paid once for the whole batch. Returns
    /// the claim txid (for response matching) together with the transaction.
    fn build_claim_tx(&self, entries: &[EscrowEntry]) -> Result<(String, RpcTransaction), String> {
        if entries.is_empty() {
            return Err("empty claim batch".into());
        }
        let total_in: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let amount_out = total_in
            .checked_sub(CLAIM_FEE_SOMPI)
            .filter(|&a| a > 0)
            .ok_or("escrow amount too small to cover claim fee")?;

        let mut inputs_meta: Vec<([u8; 32], u32, u64)> = Vec::with_capacity(entries.len());
        for entry in entries {
            let txid_bytes: [u8; 32] = hex::decode(&entry.coinbase_txid)
                .map_err(|e| format!("bad source txid: {}", e))?
                .try_into()
                .map_err(|_| "source txid must be 32 bytes")?;
            inputs_meta.push((txid_bytes, entry.output_index, entry.amount_sompi));
        }

        let reused = SighashReused::new(
            &inputs_meta,
            amount_out,
            self.payout_spk_version,
            &self.payout_spk_script,
        );
        let keypair = secp256k1::Keypair::from_secret_key(&self.secp, &self.secret_key);

        let mut inputs: Vec<RpcTransactionInput> = Vec::with_capacity(entries.len());
        for (entry, meta) in entries.iter().zip(&inputs_meta) {
            let sighash = compute_sighash(meta, &self.escrow_script, &reused);
            let msg = secp256k1::Message::from_digest_slice(&sighash)
                .map_err(|e| format!("sighash message error: {}", e))?;
            let sig = self.secp.sign_schnorr_no_aux_rand(&msg, &keypair);

            // signature_script: OpData65 (0x41) | 64-byte sig | SIG_HASH_ALL (0x01)
            let mut sig_script = Vec::with_capacity(66);
            sig_script.push(0x41u8);
            sig_script.extend_from_slice(sig.as_ref());
            sig_script.push(SIG_HASH_ALL);

            inputs.push(RpcTransactionInput {
                previous_outpoint: Some(RpcOutpoint {
                    transaction_id: entry.coinbase_txid.clone(),
                    index: entry.output_index,
                }),
                signature_script: hex::encode(&sig_script),
                sequence: CHALLENGE_WINDOW_BLOCKS,
                sig_op_count: 1,
                verbose_data: None,
            });
        }

        let claim_txid = compute_claim_txid(
            &inputs_meta,
            amount_out,
            self.payout_spk_version,
            &self.payout_spk_script,
        );

        Ok((
            claim_txid,
            RpcTransaction {
                version: 0,
                inputs,
                outputs: vec![RpcTransactionOutput {
                    amount: amount_out,
                    script_public_key: Some(RpcScriptPublicKey {
                        version: self.payout_spk_version as u32,
                        script_public_key: self.payout_spk_script_hex.clone(),
                    }),
                    verbose_data: None,
                }],
                lock_time: 0,
                subnetwork_id: NATIVE_SUBNETWORK.to_string(),
                gas: 0,
                payload: String::new(),
                mass: 0,
                verbose_data: None,
            },
        ))
    }

    /// Record an AI inference escrow outpoint (AiRequest output[1]) to claim after the challenge window.
    /// Called by grpc.rs after a successful AiResponse submission.
    pub fn track_inference_escrow(&mut self, ai_request_txid: String, confirm_daa: u64, escrow_amount: u64) {
        if escrow_amount == 0 {
            return;
        }
        let key = format!("{}:1", ai_request_txid);
        if self.outpoint_set.contains(&key) {
            return;
        }
        debug!(
            "EscrowWatcher: tracking inference escrow txid={}… daa={} amount={}",
            &ai_request_txid[..16.min(ai_request_txid.len())],
            confirm_daa,
            escrow_amount,
        );
        self.state.entries.push(EscrowEntry {
            coinbase_txid: ai_request_txid,
            block_hash: String::new(),
            confirm_daa,
            amount_sompi: escrow_amount,
            output_index: 1,
            claimed: false,
            slashed: false,
            orphan_slashed: false,
            orphan_retries: 0,
            orphan_retry_after_daa: None,
            submit_retries: 0,
            batch_cap: 0,
            cap_set_daa: 0,
            is_inference: true,
        });
        self.outpoint_set.insert(key);
        self.mark_dirty();
        self.maybe_flush();
    }

    fn save_state(&self) {
        match serde_json::to_string_pretty(&self.state) {
            Ok(json) => {
                if let Err(e) = fs::write(&self.state_path, &json) {
                    warn!("EscrowWatcher: failed to save state: {}", e);
                }
            }
            Err(e) => warn!("EscrowWatcher: failed to serialize state: {}", e),
        }
    }
}

/// Load the OPoI escrow private key from `path`. Fails if the file does not exist.
/// Used by --recover-escrow where generating a new key would silently query with the wrong pubkey.
pub fn load_key(path: &str) -> Result<String, String> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Err(format!(
            "Escrow key file '{}' not found. Run the miner once to generate it, or pass --escrow-key-file <path>.",
            path
        ));
    }
    let s = fs::read_to_string(p)
        .map_err(|e| format!("Failed to read escrow key file '{}': {}", path, e))?;
    let privkey = s.trim().to_string();
    if privkey.len() != 64 || !privkey.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "Escrow key file '{}' must contain exactly 64 hex chars",
            path
        ));
    }
    Ok(privkey)
}

/// Load the OPoI escrow private key from `path`, generating a new one if absent.
/// The file contains exactly 64 lowercase hex characters (32-byte Schnorr private key).
pub fn load_or_generate_key(path: &str) -> Result<String, String> {
    use rand::RngCore;
    let p = std::path::Path::new(path);
    if p.exists() {
        let s = fs::read_to_string(p)
            .map_err(|e| format!("Failed to read escrow key file '{}': {}", path, e))?;
        let privkey = s.trim().to_string();
        if privkey.len() != 64 || !privkey.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!(
                "Escrow key file '{}' must contain exactly 64 hex chars — delete it to regenerate",
                path
            ));
        }
        return Ok(privkey);
    }

    let mut privkey_bytes = [0u8; 32];
    loop {
        rand::thread_rng().fill_bytes(&mut privkey_bytes);
        if secp256k1::SecretKey::from_slice(&privkey_bytes).is_ok() {
            break;
        }
    }
    let privkey_hex = hex::encode(privkey_bytes);

    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes).unwrap();
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    let pubkey_hex = hex::encode(xonly.serialize());

    fs::write(p, &privkey_hex)
        .map_err(|e| format!("Failed to write escrow key file '{}': {}", path, e))?;

    info!("OPoI escrow keypair generated — saved to '{}'", path);
    info!("  Escrow pubkey : {}", pubkey_hex);
    info!("  Keep '{}' safe — needed to claim your OPoI escrow rewards.", path);
    Ok(privkey_hex)
}

/// Derive the x-only public key hex (64 hex chars) from a hex-encoded private key.
pub fn pubkey_hex_from_privkey(privkey_hex: &str) -> Result<String, String> {
    let privkey_bytes = hex::decode(privkey_hex)
        .map_err(|e| format!("Invalid privkey hex: {}", e))?;
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes)
        .map_err(|e| format!("Invalid private key: {}", e))?;
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    Ok(hex::encode(xonly.serialize()))
}

fn load_state(path: &PathBuf) -> EscrowState {
    let state = match fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            warn!("EscrowWatcher: could not parse {}: {} — starting fresh", path.display(), e);
            EscrowState::default()
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => EscrowState::default(),
        Err(e) => {
            warn!("EscrowWatcher: could not read {}: {} — starting fresh", path.display(), e);
            EscrowState::default()
        }
    };

    state
}

// ── Script builders ───────────────────────────────────────────────────────────

/// Build the CSV P2PK escrow script identical to the node's `build_escrow_script`.
///
/// `<CHALLENGE_WINDOW_BLOCKS_LE> OP_CSV OP_DATA_32 <pubkey_32> OP_CHECKSIG`
///
/// Keryx's OP_CSV pops its argument, so no OP_DROP is needed after it.
fn build_escrow_script(pubkey: &[u8; 32]) -> Vec<u8> {
    // 36 000 = 0x8CA0 — trim trailing zero bytes for minimal encoding
    let le = CHALLENGE_WINDOW_BLOCKS.to_le_bytes();
    let trimmed_len = 8 - le.iter().rev().position(|&b| b != 0).unwrap_or(8);
    let seq_bytes = &le[..trimmed_len];

    let mut script = Vec::with_capacity(3 + 1 + 33 + 1);
    script.push(seq_bytes.len() as u8); // OpData2 = 0x02
    script.extend_from_slice(seq_bytes);
    script.push(OP_CSV);
    script.push(0x20); // OpData32
    script.extend_from_slice(pubkey);
    script.push(OP_CHECKSIG);
    script
}

/// Standard Schnorr P2PK script: `OP_DATA_32 <pubkey_32> OP_CHECKSIG`
fn build_p2pk_script(pubkey: &[u8; 32]) -> Vec<u8> {
    let mut script = Vec::with_capacity(34);
    script.push(0x20); // OpData32
    script.extend_from_slice(pubkey);
    script.push(OP_CHECKSIG);
    script
}

// ── Sighash ───────────────────────────────────────────────────────────────────

/// Blake2b-256 keyed with the node's transaction-signing domain.
fn sighash_hasher() -> blake2b_simd::State {
    Blake2bParams::new().hash_length(32).key(b"TransactionSigningHash").to_state()
}

fn finalize32(h: blake2b_simd::State) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}

/// Sighash components shared by every input of a claim TX. The node caches these as
/// `SigHashReusedValues` to avoid quadratic hashing; we precompute them once per batch.
struct SighashReused {
    prev_out_hash: [u8; 32],
    seqs_hash: [u8; 32],
    sigops_hash: [u8; 32],
    outs_hash: [u8; 32],
}

impl SighashReused {
    /// `inputs`: (source txid bytes, output index, escrow amount) per claim input.
    fn new(
        inputs: &[([u8; 32], u32, u64)],
        payout_amount: u64,
        payout_spk_version: u16,
        payout_spk_script: &[u8],
    ) -> Self {
        // previous_outputs_hash: Blake2b(txid_32 | index_u32_LE, per input)
        let mut h = sighash_hasher();
        for (txid, index, _) in inputs {
            h.update(txid);
            h.update(&index.to_le_bytes());
        }
        let prev_out_hash = finalize32(h);

        // sequences_hash: Blake2b(sequence_u64_LE, per input)
        let mut h = sighash_hasher();
        for _ in inputs {
            h.update(&CHALLENGE_WINDOW_BLOCKS.to_le_bytes());
        }
        let seqs_hash = finalize32(h);

        // sig_op_counts_hash: Blake2b(sig_op_count_u8, per input)
        let mut h = sighash_hasher();
        for _ in inputs {
            h.update(&[1u8]);
        }
        let sigops_hash = finalize32(h);

        // outputs_hash: Blake2b(value_u64_LE | spk_version_u16_LE | len_u64_LE | spk_script)
        let mut h = sighash_hasher();
        h.update(&payout_amount.to_le_bytes());
        h.update(&payout_spk_version.to_le_bytes());
        h.update(&(payout_spk_script.len() as u64).to_le_bytes()); // write_var_bytes len
        h.update(payout_spk_script);
        let outs_hash = finalize32(h);

        Self { prev_out_hash, seqs_hash, sigops_hash, outs_hash }
    }
}

/// Compute the Keryx Schnorr sighash (SIG_HASH_ALL) for one input of an escrow claim TX.
///
/// Mirrors `calc_schnorr_signature_hash` in consensus/core/src/hashing/sighash.rs.
/// Byte-identical to the historical single-input version when the batch has one entry.
fn compute_sighash(input: &([u8; 32], u32, u64), escrow_script: &[u8], reused: &SighashReused) -> [u8; 32] {
    let (txid, index, amount) = input;
    let mut h = sighash_hasher();
    h.update(&0u16.to_le_bytes()); // tx.version
    h.update(&reused.prev_out_hash);
    h.update(&reused.seqs_hash);
    h.update(&reused.sigops_hash);
    // Input-specific:
    h.update(txid); // prev_outpoint.transaction_id
    h.update(&index.to_le_bytes()); // prev_outpoint.index
    h.update(&0u16.to_le_bytes()); // utxo spk version = 0
    h.update(&(escrow_script.len() as u64).to_le_bytes()); // write_var_bytes len
    h.update(escrow_script); // write_var_bytes data
    h.update(&amount.to_le_bytes()); // utxo.amount
    h.update(&CHALLENGE_WINDOW_BLOCKS.to_le_bytes()); // input.sequence
    h.update(&[1u8]); // input.sig_op_count
    h.update(&reused.outs_hash);
    h.update(&0u64.to_le_bytes()); // tx.lock_time
    h.update(&[0u8; 20]); // tx.subnetwork_id (NATIVE = all zeros 20 bytes)
    h.update(&0u64.to_le_bytes()); // tx.gas
    h.update(&[0u8; 32]); // payload_hash (ZERO_HASH: native subnetwork + empty payload)
    h.update(&[SIG_HASH_ALL]); // hash_type
    finalize32(h)
}

// ── Claim txid ────────────────────────────────────────────────────────────────

/// Compute the claim TX's transaction id, mirroring `tx::id()` in the node's
/// consensus/core/src/hashing/tx.rs: Blake2b-256 keyed `b"TransactionID"` over the TX
/// with signature scripts excluded (empty var-bytes, no sig_op_count byte) and no mass
/// commitment. Needed to match SubmitTransactionResponses to in-flight claims — and
/// because claim TXs are fully deterministic, a resubmission after a lost response
/// carries the same txid and resolves as a duplicate instead of double-spending.
fn compute_claim_txid(
    inputs: &[([u8; 32], u32, u64)],
    payout_amount: u64,
    payout_spk_version: u16,
    payout_spk_script: &[u8],
) -> String {
    let mut h = Blake2bParams::new().hash_length(32).key(b"TransactionID").to_state();
    h.update(&0u16.to_le_bytes()); // tx.version
    h.update(&(inputs.len() as u64).to_le_bytes()); // write_len(inputs)
    for (txid, index, _) in inputs {
        h.update(txid); // outpoint.transaction_id
        h.update(&index.to_le_bytes()); // outpoint.index
        h.update(&0u64.to_le_bytes()); // write_var_bytes(&[]) — sig script excluded
        h.update(&CHALLENGE_WINDOW_BLOCKS.to_le_bytes()); // sequence
    }
    h.update(&1u64.to_le_bytes()); // write_len(outputs)
    h.update(&payout_amount.to_le_bytes()); // output.value
    h.update(&payout_spk_version.to_le_bytes()); // spk version
    h.update(&(payout_spk_script.len() as u64).to_le_bytes()); // write_var_bytes len
    h.update(payout_spk_script);
    h.update(&0u64.to_le_bytes()); // tx.lock_time
    h.update(&[0u8; 20]); // subnetwork_id (NATIVE)
    h.update(&0u64.to_le_bytes()); // tx.gas
    h.update(&0u64.to_le_bytes()); // write_var_bytes(payload = [])
    hex::encode(finalize32(h))
}

// ── Address decoding ──────────────────────────────────────────────────────────

/// Decode a bech32 Keryx address into `(spk_version, spk_32_bytes)`.
fn decode_address(addr: &str) -> Result<(u16, [u8; 32]), String> {
    let colon = addr.find(':').ok_or("Missing ':' in address")?;
    let data_with_checksum = &addr[colon + 1..];
    if data_with_checksum.len() < 9 {
        return Err("Address too short".into());
    }
    let data_str = &data_with_checksum[..data_with_checksum.len() - 8];

    const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    let mut rev = [0xffu8; 128];
    for (i, &c) in CHARSET.iter().enumerate() {
        rev[c as usize] = i as u8;
    }

    let mut data5: Vec<u8> = Vec::with_capacity(data_str.len());
    for c in data_str.chars() {
        let idx = c as usize;
        if idx >= 128 || rev[idx] == 0xff {
            return Err(format!("Invalid bech32 character '{}'", c));
        }
        data5.push(rev[idx]);
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &b in &data5 {
        buf = (buf << 5) | b as u32;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buf >> bits) as u8);
        }
    }

    if bytes.len() < 33 {
        return Err(format!("Expected ≥33 decoded bytes, got {}", bytes.len()));
    }
    let version = bytes[0] as u16;
    let mut spk = [0u8; 32];
    spk.copy_from_slice(&bytes[1..33]);
    Ok((version, spk))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The responder signature must verify exactly the way the node's `verified_responder`
    /// does: schnorr over blake2b-256("KeryxServiceResponderV1" || v1 payload bytes) with the
    /// x-only escrow pubkey.
    #[test]
    fn responder_signature_verifies_like_the_node() {
        let dir = std::env::temp_dir().join(format!("keryx-escrow-test-{}", std::process::id()));
        let privkey = "1111111111111111111111111111111111111111111111111111111111111111";
        let w = EscrowWatcher::new(
            privkey,
            "keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte",
            dir,
        )
        .unwrap();

        let resp = keryx_inference::AiResponsePayload::new([9u8; 32], 123, [7u8; 34], 5);
        let signed_bytes = resp.signed_bytes();
        let r = w.sign_responder(&signed_bytes);
        assert_eq!(r.escrow_pubkey, w.pubkey_bytes);

        // Node-side verification, replicated bit-for-bit.
        let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
        hasher.update(b"KeryxServiceResponderV1");
        hasher.update(&signed_bytes);
        let msg = secp256k1::Message::from_digest_slice(hasher.finalize().as_bytes()).unwrap();
        let pk = secp256k1::XOnlyPublicKey::from_slice(&r.escrow_pubkey).unwrap();
        let sig = secp256k1::schnorr::Signature::from_slice(&r.signature).unwrap();
        assert!(secp256k1::SECP256K1.verify_schnorr(&sig, &msg, &pk).is_ok());

        // A tampered payload byte must fail verification.
        let mut bad = signed_bytes.clone();
        bad[0] ^= 1;
        let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
        hasher.update(b"KeryxServiceResponderV1");
        hasher.update(&bad);
        let bad_msg = secp256k1::Message::from_digest_slice(hasher.finalize().as_bytes()).unwrap();
        assert!(secp256k1::SECP256K1.verify_schnorr(&sig, &bad_msg, &pk).is_err());
    }
}
