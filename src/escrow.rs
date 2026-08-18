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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{fs, io};
use tempfile::NamedTempFile;

use crate::proto::{RpcOutpoint, RpcScriptPublicKey, RpcTransaction, RpcTransactionInput, RpcTransactionOutput};

const CHALLENGE_WINDOW_BLOCKS: u64 = 36_000;
/// CSV lock the node applies to coinbase escrow outputs from the H6 gate on. MUST equal the
/// node's `SERVICE_BOND_CSV_WINDOW_BLOCKS` — the script, and so the output this miner recognizes
/// and can spend, is derived from it.
const SERVICE_BOND_CSV_WINDOW_BLOCKS: u64 = 792_000;
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
    /// CSV lock of this output, in blocks. Coinbase escrows switch to the service-bond
    /// window at the H6 gate; inference escrows keep the legacy one. Entries written by
    /// earlier versions predate the gate and default to legacy.
    #[serde(default = "default_csv_window")]
    pub csv_window: u64,
}

fn default_csv_window() -> u64 {
    CHALLENGE_WINDOW_BLOCKS
}

/// CSV lock the node applies to a coinbase escrow output created at `daa`.
pub fn csv_window_for_daa(daa: u64) -> u64 {
    if daa >= keryx_miner::pom::pom_v3_activation_daa() {
        SERVICE_BOND_CSV_WINDOW_BLOCKS
    } else {
        CHALLENGE_WINDOW_BLOCKS
    }
}

fn default_output_index() -> u32 {
    1
}

/// Pick the next claim batch, returned as indices into `entries` (empty = nothing to ship).
///
/// +10 margin on maturity: OP_CSV validation can lag the virtual-state DAA score by several
/// blocks in the BlockDAG. A single +1 margin is often not enough, causing seq-lock
/// rejections that get retried every block. Per-entry cooldowns are checked individually so
/// they don't block other claims.
///
/// Selection runs in priority passes; each candidate builds its own batch and the first
/// RELEASABLE one wins, so a held-back pass never starves the ones below:
///   0 — nominal: uncapped live entries, inference first within the batch (they carry
///       user fees and must not be starved by the coinbase queue). A batch never mixes
///       CSV windows — one sequence and one escrow script are signed for the whole TX —
///       so each window present gets its own candidate, current-era first. Releasable
///       ONLY as a full MAX_CLAIM_BATCH batch — the flat claim fee is amortized across
///       the whole TX and partial batches never ship. Nominal goes FIRST so full batches
///       are never delayed behind repair grinding;
///   1 — repair: live entries carrying a bisection cap (survivors of a rejected batch).
///       They regroup among themselves — never with uncapped entries, which would bleed
///       fresh outputs into small immediate batches — and flow at any size (in the gaps
///       between full batches) so dead inputs get isolated;
///   2 — drain: live entries whose CSV window no longer matches the one coinbases mint.
///       Their pool can no longer grow to a full nominal batch, so they flow at any size;
///   3 — the known-dead pool (solo-orphaned at least once), ground down whenever nothing
///       fresher is claimable.
/// A batch never mixes passes: batching live entries with known-dead ones would let one
/// dead input orphan the whole TX and stall the live entries.
fn select_claim_batch(entries: &[EscrowEntry], daa_score: u64, in_flight_outpoints: &HashSet<String>) -> Vec<usize> {
    let current_window = csv_window_for_daa(daa_score);
    let mut windows: Vec<u64> = entries.iter().map(|e| e.csv_window).collect();
    windows.sort_unstable_by_key(|&w| (w != current_window, w));
    windows.dedup();

    let mut candidates: Vec<(u8, Option<u64>)> = windows.into_iter().map(|w| (0u8, Some(w))).collect();
    candidates.extend([(1, None), (2, None), (3, None)]);

    let mut batch: Vec<usize> = Vec::new();
    for (pass, forced_window) in candidates {
        batch.clear();
        let mut limit = MAX_CLAIM_BATCH;
        let mut batch_window: Option<u64> = forced_window;
        // Nominal pass: iterate inference entries first so they get batch priority;
        // the sort is stable, so queue order is preserved within each kind.
        let mut indices: Vec<usize> = (0..entries.len()).collect();
        if pass == 0 {
            indices.sort_by_key(|&i| !entries[i].is_inference);
        }
        for &i in &indices {
            let e = &entries[i];
            // Only the (transient, forgivable) flag decides the dead pool; the
            // orphan_retries counter is the cumulative memory driving the permanent
            // slash and must not keep an entry in the dead pool forever on its own.
            let proven_dead = e.orphan_slashed;
            let in_pass = match pass {
                0 => e.batch_cap == 0 && !proven_dead,
                1 => e.batch_cap != 0 && !proven_dead,
                2 => e.batch_cap == 0 && !proven_dead && e.csv_window != current_window,
                _ => proven_dead,
            };
            if !in_pass {
                continue;
            }
            let eligible = !e.claimed
                && !e.slashed
                && daa_score >= e.confirm_daa + e.csv_window + 10
                && e.orphan_retry_after_daa.map_or(true, |retry_daa| daa_score >= retry_daa)
                && !in_flight_outpoints.contains(&format!("{}:{}", e.coinbase_txid, e.output_index));
            if !eligible {
                continue;
            }
            if batch_window.map_or(false, |w| w != e.csv_window) {
                continue;
            }
            let cap = if e.batch_cap == 0 { MAX_CLAIM_BATCH } else { (e.batch_cap as usize).min(MAX_CLAIM_BATCH) };
            if cap.min(limit) < batch.len() + 1 {
                continue; // joining would violate this entry's cap (or shrink below current size)
            }
            limit = limit.min(cap);
            batch_window = Some(e.csv_window);
            batch.push(i);
            if batch.len() >= limit {
                break; // batch is full for this candidate
            }
        }
        let releasable = match pass {
            // Nominal batches ship full or not at all (fee amortization).
            0 => batch.len() >= MIN_CLAIM_BATCH,
            // Repair, drain and dead-pool batches flow at any size.
            _ => !batch.is_empty(),
        };
        if releasable {
            return batch;
        }
    }
    Vec::new()
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
    escrow_script_hex: String,
    /// Service-bond variant of the escrow script, paid by coinbases from the H6 gate on.
    escrow_script_bonded_hex: String,
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
        let privkey_bytes = hex::decode(privkey_hex).map_err(|e| format!("Invalid --mining-privkey hex: {}", e))?;
        if privkey_bytes.len() != 32 {
            return Err(format!("--mining-privkey must be 32 bytes (64 hex chars), got {}", privkey_bytes.len()));
        }

        let secp = secp256k1::Secp256k1::new();
        let secret_key =
            secp256k1::SecretKey::from_slice(&privkey_bytes).map_err(|e| format!("Invalid private key: {}", e))?;
        let keypair = secp256k1::Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let pubkey_bytes: [u8; 32] = xonly.serialize();

        let escrow_script_hex = hex::encode(build_escrow_script(&pubkey_bytes, CHALLENGE_WINDOW_BLOCKS));
        let escrow_script_bonded_hex = hex::encode(build_escrow_script(&pubkey_bytes, SERVICE_BOND_CSV_WINDOW_BLOCKS));

        let (payout_spk_version, payout_spk_bytes) = decode_address(mining_address)?;
        let payout_spk_script = build_p2pk_script(&payout_spk_bytes);
        let payout_spk_script_hex = hex::encode(&payout_spk_script);

        let state = load_state(&state_path)?;

        info!("EscrowWatcher ready: pubkey={}", hex::encode(pubkey_bytes));

        let mut watcher = Self {
            secp,
            secret_key,
            pubkey_bytes,
            escrow_script_hex,
            escrow_script_bonded_hex,
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
            if let Err(e) = self.save_state() {
                warn!("EscrowWatcher: failed to save state: {}", e);
            } else {
                self.dirty = false;
            }
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
                .filter(|e| !e.claimed && !e.slashed && daa_score >= e.confirm_daa + e.csv_window + 10)
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
                let script = spk.script_public_key.to_lowercase();
                let csv_window = if script == self.escrow_script_hex {
                    Some(CHALLENGE_WINDOW_BLOCKS)
                } else if script == self.escrow_script_bonded_hex {
                    Some(SERVICE_BOND_CSV_WINDOW_BLOCKS)
                } else {
                    None
                };
                if let Some(csv_window) = csv_window.filter(|_| spk.version == 0 && !self.outpoint_set.contains(&key))
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
                        csv_window,
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

        let batch = select_claim_batch(&self.state.entries, daa_score, &self.in_flight_outpoints);
        if batch.is_empty() {
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
                // SpendOfBurnedEscrow (node `TxRuleError`): the outpoint is unspendable forever.
                let is_burned = msg.contains("burned escrow outpoint");
                // The node names every burned outpoint in the batch ("...outpoints: txid:idx txid:idx").
                // Parsing it lets us slash exactly those and re-batch the rest — no bisection. An older
                // node sends no list; `burned_set` stays empty and we fall back to bisection below.
                let burned_set: std::collections::HashSet<(String, u32)> = msg
                    .rsplit_once("burned escrow outpoints: ")
                    .map(|(_, list)| {
                        list.split_whitespace()
                            .filter_map(|tok| {
                                let (tx, idx) = tok.split_once(':')?;
                                Some((tx.to_ascii_lowercase(), idx.parse().ok()?))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let batch_rejected = n_outputs > 1;
                let last_daa = self.last_daa_score;
                // Bisection step: one dead input orphans the whole batch, but most members
                // are usually fine. Halve every member's cap and retry as smaller batches —
                // valid entries re-group into accepted batches within log2(batch) rounds,
                // dead ones converge to solo claims and get slashed there.
                let halved_cap = ((n_outputs / 2).max(1)).min(u8::MAX as usize) as u8;
                for (t, i) in &claim.outpoints {
                    let e = match self.state.entries.iter_mut().find(|e| e.coinbase_txid == *t && e.output_index == *i)
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
                    } else if is_burned {
                        // Burn is permanent by consensus. When the node named the burned outpoints,
                        // slash exactly the ones it listed and leave the rest untouched — they
                        // re-batch next round, no bisection. Fall back to bisection only when no list
                        // was parsed (older node): batch → halve to isolate, solo → slash. The exact
                        // node message keeps this off the "irreversible state from an error string"
                        // trap that applies to every OTHER rejection.
                        if !burned_set.is_empty() {
                            if burned_set.contains(&(t.to_ascii_lowercase(), *i)) {
                                e.slashed = true;
                            }
                        } else if batch_rejected {
                            e.batch_cap = halved_cap;
                            e.cap_set_daa = last_daa;
                        } else {
                            e.slashed = true;
                        }
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
                // Burns are terminal and operator-relevant (the miner is being penalised) — surface
                // them once at WARN. Everything else is transient bisection repair, kept at DEBUG.
                if is_burned && !burned_set.is_empty() {
                    warn!(
                        "EscrowWatcher: {} escrow outpoint(s) burned by service-bond — slashed permanently, re-batching the rest.",
                        burned_set.len()
                    );
                } else if is_burned && !batch_rejected {
                    warn!(
                        "EscrowWatcher: escrow outpoint burned by service-bond — abandoning claim {} permanently: {}",
                        claim_txid, msg
                    );
                } else {
                    debug!(
                        "EscrowWatcher: claim {} rejected ({} output(s) released{}): {}",
                        claim_txid,
                        n_outputs,
                        if batch_rejected { ", batch cap halved" } else { "" },
                        msg
                    );
                }
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
            if let Some(e) = self.state.entries.iter_mut().find(|e| e.coinbase_txid == *t && e.output_index == *i) {
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
        // Guaranteed single-valued by the batch selection; the whole TX signs one sequence.
        let csv_window = entries[0].csv_window;
        if entries.iter().any(|e| e.csv_window != csv_window) {
            return Err("claim batch mixes CSV windows".into());
        }
        let escrow_script = build_escrow_script(&self.pubkey_bytes, csv_window);
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
            csv_window,
        );
        let keypair = secp256k1::Keypair::from_secret_key(&self.secp, &self.secret_key);

        let mut inputs: Vec<RpcTransactionInput> = Vec::with_capacity(entries.len());
        for (entry, meta) in entries.iter().zip(&inputs_meta) {
            let sighash = compute_sighash(meta, &escrow_script, &reused, csv_window);
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
                sequence: csv_window,
                sig_op_count: 1,
                verbose_data: None,
            });
        }

        let claim_txid = compute_claim_txid(
            &inputs_meta,
            amount_out,
            self.payout_spk_version,
            &self.payout_spk_script,
            csv_window,
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
            // Built by the requester's wallet, which locks the legacy window on both eras.
            csv_window: CHALLENGE_WINDOW_BLOCKS,
        });
        self.outpoint_set.insert(key);
        self.mark_dirty();
        self.maybe_flush();
    }

    fn save_state(&self) -> Result<(), String> {
        save_state_atomic(&self.state_path, &self.state)
    }

    pub fn flush_state(&mut self) -> Result<(), String> {
        if self.dirty {
            self.save_state()?;
            self.dirty = false;
            self.last_save = Instant::now();
        }
        Ok(())
    }
}

impl Drop for EscrowWatcher {
    fn drop(&mut self) {
        if let Err(e) = self.flush_state() {
            warn!("EscrowWatcher: final state flush failed: {}", e);
        }
    }
}

fn ensure_parent(path: &Path) -> io::Result<&Path> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    create_parent_dirs(parent)?;
    Ok(parent)
}

#[cfg(unix)]
fn create_parent_dirs(parent: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut cursor = parent;
    while !cursor.exists() {
        missing.push(cursor.to_path_buf());
        cursor = cursor.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    }
    if !cursor.is_dir() {
        return Err(io::Error::new(io::ErrorKind::NotADirectory, format!("'{}' is not a directory", cursor.display())));
    }
    for directory in missing.into_iter().rev() {
        match fs::create_dir(&directory) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && directory.is_dir() => {}
            Err(e) => return Err(e),
        }
        sync_parent(directory.parent().unwrap_or_else(|| Path::new(".")))?;
        sync_parent(&directory)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn create_parent_dirs(parent: &Path) -> io::Result<()> {
    fs::create_dir_all(parent)
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> io::Result<()> {
    fs::File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn harden_key_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    if permissions.mode() & 0o077 != 0 {
        permissions.set_mode(0o600);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn harden_key_permissions(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
thread_local! {
    static ATOMIC_REPLACE_FAILURE_STAGE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn inject_atomic_replace_failure(stage: u8) -> io::Result<()> {
    ATOMIC_REPLACE_FAILURE_STAGE.with(|configured| {
        if configured.get() == stage {
            Err(io::Error::new(io::ErrorKind::Other, format!("injected atomic replacement failure at stage {stage}")))
        } else {
            Ok(())
        }
    })
}

#[cfg(windows)]
fn move_file_write_through(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH};

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination.as_os_str().encode_wide().chain(Some(0)).collect();
    let flags = MOVEFILE_WRITE_THROUGH | if replace { MOVEFILE_REPLACE_EXISTING } else { 0 };
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn install_temp_noclobber(temporary: NamedTempFile, path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        move_file_write_through(temporary.path(), path, false)
    }
    #[cfg(not(windows))]
    {
        temporary.persist_noclobber(path).map(|_| ()).map_err(|e| e.error)
    }
}

fn atomic_replace(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = ensure_parent(path)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    #[cfg(test)]
    inject_atomic_replace_failure(1)?;
    temporary.write_all(contents)?;
    #[cfg(test)]
    inject_atomic_replace_failure(2)?;
    temporary.as_file().sync_all()?;
    #[cfg(test)]
    inject_atomic_replace_failure(3)?;
    #[cfg(windows)]
    move_file_write_through(temporary.path(), path, true)?;
    #[cfg(not(windows))]
    temporary.persist(path).map_err(|e| e.error)?;
    #[cfg(test)]
    inject_atomic_replace_failure(4)?;
    sync_parent(parent)
}

/// Persist escrow state without exposing a partially-written JSON file.
pub fn save_state_atomic(path: &Path, state: &EscrowState) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(state).map_err(|e| format!("Failed to serialize escrow state: {}", e))?;
    atomic_replace(path, &json)
        .map_err(|e| format!("Failed to atomically write escrow state '{}': {}", path.display(), e))
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
    // Best effort: a filesystem without Unix permissions (removable media, some network mounts)
    // must not stop a miner whose key is perfectly readable.
    if let Err(e) = harden_key_permissions(p) {
        warn!("Could not restrict permissions on escrow key file '{}': {}", path, e);
    }
    let s = fs::read_to_string(p).map_err(|e| format!("Failed to read escrow key file '{}': {}", path, e))?;
    let privkey = s.trim().to_string();
    if privkey.len() != 64 || !privkey.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("Escrow key file '{}' must contain exactly 64 hex chars", path));
    }
    Ok(privkey)
}

/// Load the OPoI escrow private key from `path`, generating a new one if absent.
/// The file contains exactly 64 lowercase hex characters (32-byte Schnorr private key).
pub fn load_or_generate_key(path: &str) -> Result<String, String> {
    use rand::RngCore;
    let p = std::path::Path::new(path);
    if p.exists() {
        return load_key(path)
            .map_err(|e| format!("{}. Restore the correct key; do not delete a key that may control rewards.", e));
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

    let parent =
        ensure_parent(p).map_err(|e| format!("Failed to create escrow key directory for '{}': {}", path, e))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .map_err(|e| format!("Failed to create temporary escrow key file for '{}': {}", path, e))?;
    temporary
        .write_all(privkey_hex.as_bytes())
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|e| format!("Failed to write escrow key file '{}': {}", path, e))?;

    match install_temp_noclobber(temporary, p) {
        Ok(()) => sync_parent(parent)
            .map_err(|e| format!("Failed to sync escrow key directory '{}': {}", parent.display(), e))?,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return load_key(path),
        Err(e) => return Err(format!("Failed to install escrow key file '{}': {}", path, e)),
    }

    info!("OPoI escrow keypair generated — saved to '{}'", path);
    info!("  Escrow pubkey : {}", pubkey_hex);
    info!("  Keep '{}' safe — needed to claim your OPoI escrow rewards.", path);
    Ok(privkey_hex)
}

/// Derive the x-only public key hex (64 hex chars) from a hex-encoded private key.
pub fn pubkey_hex_from_privkey(privkey_hex: &str) -> Result<String, String> {
    let privkey_bytes = hex::decode(privkey_hex).map_err(|e| format!("Invalid privkey hex: {}", e))?;
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes).map_err(|e| format!("Invalid private key: {}", e))?;
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    Ok(hex::encode(xonly.serialize()))
}

fn load_state(path: &Path) -> Result<EscrowState, String> {
    match fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).map_err(|e| {
            format!("Escrow state '{}' is corrupt: {}. Restore it or run --recover-escrow.", path.display(), e)
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(EscrowState::default()),
        Err(e) => Err(format!("Failed to read escrow state '{}': {}", path.display(), e)),
    }
}

// ── Script builders ───────────────────────────────────────────────────────────

/// Build the CSV P2PK escrow script identical to the node's `build_escrow_script`.
///
/// `<CHALLENGE_WINDOW_BLOCKS_LE> OP_CSV OP_DATA_32 <pubkey_32> OP_CHECKSIG`
///
/// Keryx's OP_CSV pops its argument, so no OP_DROP is needed after it.
fn build_escrow_script(pubkey: &[u8; 32], csv_window: u64) -> Vec<u8> {
    // trim trailing zero bytes for minimal encoding
    let le = csv_window.to_le_bytes();
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
        csv_window: u64,
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
            h.update(&csv_window.to_le_bytes());
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
fn compute_sighash(
    input: &([u8; 32], u32, u64),
    escrow_script: &[u8],
    reused: &SighashReused,
    csv_window: u64,
) -> [u8; 32] {
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
    h.update(&csv_window.to_le_bytes()); // input.sequence
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
    csv_window: u64,
) -> String {
    let mut h = Blake2bParams::new().hash_length(32).key(b"TransactionID").to_state();
    h.update(&0u16.to_le_bytes()); // tx.version
    h.update(&(inputs.len() as u64).to_le_bytes()); // write_len(inputs)
    for (txid, index, _) in inputs {
        h.update(txid); // outpoint.transaction_id
        h.update(&index.to_le_bytes()); // outpoint.index
        h.update(&0u64.to_le_bytes()); // write_var_bytes(&[]) — sig script excluded
        h.update(&csv_window.to_le_bytes()); // sequence
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

// ── Escrow delegation cert ────────────────────────────────────────────────────

/// Mirror of the node's `ESCROW_DELEGATION_DOMAIN`.
const ESCROW_DELEGATION_DOMAIN: &[u8] = b"KeryxEscrowDelegationV1";

fn escrow_delegation_message(escrow_pubkey: &[u8; 32]) -> [u8; 32] {
    let mut h = Blake2bParams::new().hash_length(32).to_state();
    h.update(ESCROW_DELEGATION_DOMAIN);
    h.update(escrow_pubkey);
    finalize32(h)
}

/// Service-ledger identity of a payout address — mirror of the node's `miner_key(spk)`:
/// blake2b-256 keyed "TransactionHash" over `[version_le(2), p2pk_script]`. This is what the
/// node reports strikes, burns and suspensions against; the escrow key is only the hot key.
pub fn service_identity_hex(payout_address: &str) -> Result<String, String> {
    let (version, payout_key) = decode_address(payout_address)?;
    if version != 0 {
        return Err(format!("Payout address version {} has no service identity (schnorr P2PK only)", version));
    }
    let mut h = Blake2bParams::new().hash_length(32).key(b"TransactionHash").to_state();
    h.update(&version.to_le_bytes());
    h.update(&build_p2pk_script(&payout_key));
    Ok(hex::encode(finalize32(h)))
}

/// Verifies a delegation cert exactly the way the node does before accepting a block: schnorr
/// over the domain-hashed escrow key, by the x-only key of the payout address.
pub fn verify_escrow_cert(payout_address: &str, escrow_pubkey_hex: &str, cert_hex: &str) -> Result<(), String> {
    let (version, payout_key) = decode_address(payout_address)?;
    if version != 0 {
        return Err(format!("Payout address version {} cannot carry a delegation (schnorr P2PK only)", version));
    }
    let mut escrow_pubkey = [0u8; 32];
    hex::decode_to_slice(escrow_pubkey_hex, &mut escrow_pubkey)
        .map_err(|e| format!("Invalid escrow pubkey hex: {}", e))?;
    let mut sig_bytes = [0u8; 64];
    hex::decode_to_slice(cert_hex, &mut sig_bytes).map_err(|e| format!("Invalid cert hex: {}", e))?;

    let payout_key =
        secp256k1::XOnlyPublicKey::from_slice(&payout_key).map_err(|e| format!("Invalid payout address key: {}", e))?;
    let sig =
        secp256k1::schnorr::Signature::from_slice(&sig_bytes).map_err(|e| format!("Invalid cert signature: {}", e))?;
    let msg = secp256k1::Message::from_digest_slice(&escrow_delegation_message(&escrow_pubkey)).unwrap();
    secp256k1::Secp256k1::verification_only()
        .verify_schnorr(&sig, &msg, &payout_key)
        .map_err(|_| "Cert does not match this payout address and escrow key".to_string())
}

/// Signs the delegation cert when the payout address IS this escrow key's own address — the only
/// case where the payout key is on this machine. `None` otherwise: a cold payout address must be
/// signed by the wallet that holds it.
pub fn self_sign_cert(privkey_hex: &str, payout_address: &str) -> Option<String> {
    let (version, payout_key) = decode_address(payout_address).ok()?;
    let privkey_bytes = hex::decode(privkey_hex).ok()?;
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes).ok()?;
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    let pubkey: [u8; 32] = xonly.serialize();
    if version != 0 || payout_key != pubkey {
        return None;
    }
    let msg = secp256k1::Message::from_digest_slice(&escrow_delegation_message(&pubkey)).ok()?;
    Some(hex::encode(secp.sign_schnorr_no_aux_rand(&msg, &kp).as_ref()))
}

/// Loads the delegation cert and verifies it against the payout address and escrow key before
/// returning it. Never hands back a cert the node would reject.
pub fn load_cert(path: &str, payout_address: &str, escrow_pubkey_hex: &str) -> Result<String, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("Cannot read escrow delegation cert '{}': {}", path, e))?;
    let cert = raw.trim().to_ascii_lowercase();
    if cert.len() != 128 {
        return Err(format!("Escrow delegation cert '{}' must be 128 hex chars, found {}", path, cert.len()));
    }
    verify_escrow_cert(payout_address, escrow_pubkey_hex, &cert)?;
    Ok(cert)
}

/// Persist a verified delegation cert to `path` so later starts load it without `--escrow-cert`.
/// Idempotent: writes only when the file is missing or holds a different value. Returns whether it
/// wrote. The cert is public (a signature over pubkey↔address), so the file needs no special mode.
pub fn save_cert(path: &str, cert_hex: &str) -> std::io::Result<bool> {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing.trim().eq_ignore_ascii_case(cert_hex) {
            return Ok(false);
        }
    }
    fs::write(path, cert_hex)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Payout address of the private key `0x11..11`, used as a payout key below.
    const TEST_PAYOUT_ADDRESS: &str = "keryx:qp8n2k7uklxq4aegau7vawtptkgxsja4kt99lpv6krctwpq8tpc65uyeddvzr";

    /// The service identity must equal the node's `miner_key(spk)`. The expected value is derived
    /// independently of this code: blake2b-256 keyed "TransactionHash" over
    /// `[version_le(2), 0x20 || key || 0xac]`.
    #[test]
    fn service_identity_matches_the_node_miner_key() {
        assert_eq!(
            service_identity_hex("keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte").unwrap(),
            "cb79bef02d429e0fc8bb2335bf43d9d0df4f5bd6a25a39747d700b173e766e20"
        );
        assert!(service_identity_hex("not-an-address").is_err());
    }

    /// Cross-implementation vector. This cert was produced by the web wallet (noble schnorr over
    /// noble blake2b) for escrow key `0x22..22` and the address of private key `0x11..11`. The
    /// consensus rule must accept it: the two implementations have to agree on the domain string,
    /// the digest and the curve, and nothing else pins that agreement.
    #[test]
    fn web_wallet_cert_verifies_against_the_consensus_rule() {
        let cert = "f3cf9bfc6a29ba5608ebe777ff9e1f87d5bc3f80ab58e3040995631bfc16ab21\
                    8b388c9b2d44140fc9448e639dc81c51ba475a780d1eaecc443c613e6d787c9e";
        assert!(verify_escrow_cert(TEST_PAYOUT_ADDRESS, &"22".repeat(32), cert).is_ok());
        // Same signature against another escrow key must fail, or the test proves nothing.
        assert!(verify_escrow_cert(TEST_PAYOUT_ADDRESS, &"33".repeat(32), cert).is_err());
    }

    /// The miner signs its own delegation only when the payout address is its escrow key's.
    #[test]
    fn self_signing_is_limited_to_the_escrow_key_address() {
        let privkey = "1111111111111111111111111111111111111111111111111111111111111111";
        let own = TEST_PAYOUT_ADDRESS;
        let cert = self_sign_cert(privkey, own).expect("own address must self-sign");
        let escrow_pubkey = pubkey_hex_from_privkey(privkey).unwrap();
        assert!(verify_escrow_cert(own, &escrow_pubkey, &cert).is_ok());

        // Any other payout address: the miner does not hold that key, so it must refuse.
        assert!(self_sign_cert(
            privkey,
            "keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte"
        )
        .is_none());
    }

    /// A cert only verifies against the exact payout address and escrow key it was signed for.
    #[test]
    fn escrow_cert_binds_payout_address_and_escrow_key() {
        let secp = secp256k1::Secp256k1::new();
        let sk = secp256k1::SecretKey::from_slice(&[0x11u8; 32]).unwrap();
        let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
        let escrow_pubkey = [0x22u8; 32];
        let msg = secp256k1::Message::from_digest_slice(&escrow_delegation_message(&escrow_pubkey)).unwrap();
        let cert = hex::encode(secp.sign_schnorr_no_aux_rand(&msg, &kp).as_ref());

        assert!(verify_escrow_cert(TEST_PAYOUT_ADDRESS, &hex::encode(escrow_pubkey), &cert).is_ok());
        // Another escrow key: the delegation message differs.
        assert!(verify_escrow_cert(TEST_PAYOUT_ADDRESS, &hex::encode([0x33u8; 32]), &cert).is_err());
        // Another payout address: not the signer.
        assert!(verify_escrow_cert(
            "keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte",
            &hex::encode(escrow_pubkey),
            &cert
        )
        .is_err());
        assert!(verify_escrow_cert(TEST_PAYOUT_ADDRESS, &hex::encode(escrow_pubkey), "dead").is_err());
    }

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

    /// Both escrow scripts must match what the node's `ScriptBuilder::add_sequence` emits:
    /// the sequence little-endian with trailing zero bytes trimmed, pushed by an OpData
    /// opcode equal to its length. The legacy bytes are pinned so the pre-H6 script — and
    /// every signature over it — stays unchanged.
    #[test]
    fn escrow_scripts_mirror_the_node_for_both_windows() {
        let pk = [0x11u8; 32];
        let legacy = build_escrow_script(&pk, CHALLENGE_WINDOW_BLOCKS);
        let bonded = build_escrow_script(&pk, SERVICE_BOND_CSV_WINDOW_BLOCKS);

        assert_eq!(&legacy[..3], &[0x02, 0xa0, 0x8c]); // 36_000 = 0x8ca0
        assert_eq!(&bonded[..4], &[0x03, 0xc0, 0x15, 0x0c]); // 792_000 = 0x0c15c0

        assert_eq!(legacy[3], OP_CSV);
        assert_eq!(bonded[4], OP_CSV);
        assert_eq!(legacy[4], 0x20);
        assert_eq!(bonded[5], 0x20);
        assert_eq!(&legacy[5..37], &pk);
        assert_eq!(&bonded[6..38], &pk);
        assert_eq!(*legacy.last().unwrap(), OP_CHECKSIG);
        assert_eq!(*bonded.last().unwrap(), OP_CHECKSIG);
        assert_eq!(legacy.len(), 38);
        assert_eq!(bonded.len(), 39);
    }

    /// The window is derived from the creating block's DAA, so entries minted on either
    /// side of the gate keep their own lock.
    #[test]
    fn csv_window_follows_the_gate() {
        let gate = keryx_miner::pom::pom_v3_activation_daa();
        if gate > 0 && gate < u64::MAX {
            assert_eq!(csv_window_for_daa(gate - 1), CHALLENGE_WINDOW_BLOCKS);
        }
        assert_eq!(csv_window_for_daa(gate), SERVICE_BOND_CSV_WINDOW_BLOCKS);
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn state(confirm_daa: u64) -> EscrowState {
        EscrowState {
            entries: vec![EscrowEntry {
                coinbase_txid: "01".repeat(32),
                block_hash: "02".repeat(32),
                confirm_daa,
                amount_sompi: 100,
                output_index: 1,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                submit_retries: 0,
                batch_cap: 0,
                cap_set_daa: 0,
                is_inference: false,
                csv_window: csv_window_for_daa(confirm_daa),
            }],
        }
    }

    fn fail_at(stage: u8) {
        ATOMIC_REPLACE_FAILURE_STAGE.with(|configured| configured.set(stage));
    }

    fn entry(idx: usize, confirm_daa: u64, csv_window: u64) -> EscrowEntry {
        EscrowEntry {
            coinbase_txid: format!("{idx:064x}"),
            block_hash: "02".repeat(32),
            confirm_daa,
            amount_sompi: 100,
            output_index: 1,
            claimed: false,
            slashed: false,
            orphan_slashed: false,
            orphan_retries: 0,
            orphan_retry_after_daa: None,
            submit_retries: 0,
            batch_cap: 0,
            cap_set_daa: 0,
            is_inference: false,
            csv_window,
        }
    }

    /// A sub-full remainder of legacy-window entries at the head of the queue must not
    /// keep a full current-window batch from shipping.
    #[test]
    fn full_current_window_batch_ships_past_legacy_remainder() {
        let gate = keryx_miner::pom::pom_v3_activation_daa();
        let daa = gate + SERVICE_BOND_CSV_WINDOW_BLOCKS + 100_000;
        let mut entries: Vec<EscrowEntry> =
            (0..40).map(|i| entry(i, gate.saturating_sub(50_000), CHALLENGE_WINDOW_BLOCKS)).collect();
        entries.extend((40..40 + MAX_CLAIM_BATCH).map(|i| entry(i, gate + 10, SERVICE_BOND_CSV_WINDOW_BLOCKS)));

        let batch = select_claim_batch(&entries, daa, &HashSet::new());
        assert_eq!(batch.len(), MAX_CLAIM_BATCH);
        assert!(batch.iter().all(|&i| entries[i].csv_window == SERVICE_BOND_CSV_WINDOW_BLOCKS));
    }

    /// Legacy-window entries can no longer grow to a full batch once coinbases mint the
    /// bonded window: they must drain at any size instead of waiting forever.
    #[test]
    fn stranded_legacy_window_drains_at_any_size() {
        let gate = keryx_miner::pom::pom_v3_activation_daa();
        if gate == 0 {
            return; // testnet build: no legacy era exists
        }
        let daa = gate + SERVICE_BOND_CSV_WINDOW_BLOCKS + 100_000;
        let mut entries: Vec<EscrowEntry> =
            (0..40).map(|i| entry(i, gate.saturating_sub(50_000), CHALLENGE_WINDOW_BLOCKS)).collect();
        // Current-window entries below a full batch: nominal must hold them back.
        entries.extend((40..90).map(|i| entry(i, gate + 10, SERVICE_BOND_CSV_WINDOW_BLOCKS)));

        let batch = select_claim_batch(&entries, daa, &HashSet::new());
        assert_eq!(batch.len(), 40);
        assert!(batch.iter().all(|&i| entries[i].csv_window == CHALLENGE_WINDOW_BLOCKS));
    }

    #[test]
    fn state_replacement_is_complete_and_parseable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow_state.json");
        save_state_atomic(&path, &EscrowState::default()).unwrap();
        save_state_atomic(&path, &state(42)).unwrap();

        let loaded: EscrowState = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].confirm_daa, 42);
    }

    #[test]
    fn failures_before_replace_preserve_previous_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow_state.json");
        save_state_atomic(&path, &state(1)).unwrap();
        let original = fs::read(&path).unwrap();

        for stage in 1..=3 {
            fail_at(stage);
            assert!(save_state_atomic(&path, &state(2)).is_err());
            fail_at(0);
            assert_eq!(fs::read(&path).unwrap(), original, "failure stage {stage}");
        }
    }

    #[test]
    fn failure_after_replace_leaves_complete_state_and_allows_retry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow_state.json");
        save_state_atomic(&path, &state(1)).unwrap();

        fail_at(4);
        assert!(save_state_atomic(&path, &state(2)).is_err());
        fail_at(0);
        let loaded: EscrowState = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(loaded.entries[0].confirm_daa, 2);

        save_state_atomic(&path, &state(3)).unwrap();
        let retried: EscrowState = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(retried.entries[0].confirm_daa, 3);
    }

    #[test]
    fn concurrent_key_creation_returns_one_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = Arc::new(dir.path().join("escrow.key"));
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    load_or_generate_key(path.to_str().unwrap()).unwrap()
                })
            })
            .collect();
        let keys: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();

        assert!(keys.iter().all(|key| key == &keys[0]));
        assert_eq!(fs::read_to_string(path.as_ref()).unwrap(), keys[0]);
    }

    #[cfg(unix)]
    #[test]
    fn generated_key_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow.key");
        load_or_generate_key(path.to_str().unwrap()).unwrap();
        assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn existing_key_permissions_are_hardened() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow.key");
        fs::write(&path, "11".repeat(32)).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        load_or_generate_key(path.to_str().unwrap()).unwrap();

        assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn invalid_key_fails_without_changing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow.key");
        let invalid = b"not-a-private-key";
        fs::write(&path, invalid).unwrap();

        assert!(load_or_generate_key(path.to_str().unwrap()).is_err());
        assert_eq!(fs::read(path).unwrap(), invalid);
    }

    #[test]
    fn corrupt_state_fails_closed_without_changing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escrow_state.json");
        let invalid = b"{not-json";
        fs::write(&path, invalid).unwrap();

        let error = load_state(&path).unwrap_err();
        assert!(error.contains("corrupt"));
        assert!(error.contains("--recover-escrow"));
        assert_eq!(fs::read(path).unwrap(), invalid);
    }

    #[test]
    fn explicit_flush_keeps_dirty_state_until_retry_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("state").join("escrow_state.json");
        let address = format!("keryx:{}", "q".repeat(61));
        let mut watcher = EscrowWatcher::new(&"11".repeat(32), &address, path.clone()).unwrap();
        watcher.state = state(77);
        watcher.dirty = true;

        fail_at(1);
        assert!(watcher.flush_state().is_err());
        assert!(watcher.dirty);
        fail_at(0);
        watcher.flush_state().unwrap();

        assert!(!watcher.dirty);
        let loaded: EscrowState = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(loaded.entries[0].confirm_daa, 77);
    }
}
