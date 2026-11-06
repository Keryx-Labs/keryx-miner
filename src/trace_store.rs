//! On-disk copy of each traced response, kept until its opening window closes so a restart or a
//! reconnection does not lose an audit in flight.

use borsh::{BorshDeserialize, BorshSerialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, OnceLock};

const STORE_VERSION: u32 = 5;
/// Records written before the chain anchor: read and kept, without it.
const V4_VERSION: u32 = 4;
/// Records written before attention phases: read and kept, without them.
const V3_VERSION: u32 = 3;
/// Records written before sequence leaves: read and kept, without them.
const LEGACY_VERSION: u32 = 2;
const FILE_EXT: &str = "trace";
/// Mutable part of a record, rewritten on each change while the trace file is written once.
const STATE_EXT: &str = "state";

/// The fields of a record that change after its first write.
#[derive(BorshSerialize, BorshDeserialize)]
struct StoredState {
    accepted_hash: Option<String>,
    opening: Option<StoredOpening>,
    attn: Option<StoredAttn>,
    anchor_hash: Option<String>,
    gen_secs: Option<f64>,
}

/// The opening already built for a response: resent as is after a restart.
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct StoredOpening {
    pub seed_hash: String,
    pub deadline_daa: u64,
    /// Hex payload of the signed opening transaction.
    pub payload: String,
    /// Hex payloads of the other pieces of the opening, one transaction each.
    pub extra: Vec<String>,
}

/// Attention phases after the first, once the first is submitted.
#[derive(Clone, Debug, Default, PartialEq, BorshSerialize, BorshDeserialize)]
pub struct StoredAttn {
    /// Unsigned body of each submitted phase, phase order.
    pub bodies: Vec<Vec<u8>>,
    /// Hex payload of the signed transaction of the last submitted phase.
    pub payload: String,
    pub deadline_daa: u64,
    /// Chain block that accepted the last submitted phase.
    pub accepted_hash: Option<String>,
}

#[derive(BorshDeserialize)]
struct LegacyStoredOpening {
    seed_hash: String,
    deadline_daa: u64,
    payload: String,
}

#[derive(BorshDeserialize)]
pub struct StoredTrace {
    pub version: u32,
    pub response_txid: String,
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub max_tokens: u32,
    pub submitted_daa: u64,
    pub accepted_hash: Option<String>,
    pub opening: Option<StoredOpening>,
    pub trace: crate::trace::ResponseTrace,
    pub attn: Option<StoredAttn>,
    /// Last chain block known when the response was submitted.
    pub anchor_hash: Option<String>,
    /// Measured duration of the generation, in seconds.
    pub gen_secs: Option<f64>,
}

#[derive(BorshDeserialize)]
struct V4StoredTrace {
    version: u32,
    response_txid: String,
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    submitted_daa: u64,
    accepted_hash: Option<String>,
    opening: Option<StoredOpening>,
    trace: crate::trace::ResponseTrace,
    attn: Option<StoredAttn>,
}

impl From<V4StoredTrace> for StoredTrace {
    fn from(t: V4StoredTrace) -> Self {
        Self {
            version: STORE_VERSION,
            response_txid: t.response_txid,
            model_id: t.model_id,
            tier: t.tier,
            request_hash: t.request_hash,
            max_tokens: t.max_tokens,
            submitted_daa: t.submitted_daa,
            accepted_hash: t.accepted_hash,
            opening: t.opening,
            trace: t.trace,
            attn: t.attn,
            anchor_hash: None,
            gen_secs: None,
        }
    }
}

#[derive(BorshDeserialize)]
struct V3StoredTrace {
    version: u32,
    response_txid: String,
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    submitted_daa: u64,
    accepted_hash: Option<String>,
    opening: Option<StoredOpening>,
    trace: crate::trace::ResponseTrace,
}

impl From<V3StoredTrace> for StoredTrace {
    fn from(t: V3StoredTrace) -> Self {
        Self {
            version: STORE_VERSION,
            response_txid: t.response_txid,
            model_id: t.model_id,
            tier: t.tier,
            request_hash: t.request_hash,
            max_tokens: t.max_tokens,
            submitted_daa: t.submitted_daa,
            accepted_hash: t.accepted_hash,
            opening: t.opening,
            trace: t.trace,
            attn: None,
            anchor_hash: None,
            gen_secs: None,
        }
    }
}

#[derive(BorshDeserialize)]
struct LegacyTrace {
    record: crate::trace::TraceRecord,
    tokens: Vec<i32>,
}

#[derive(BorshDeserialize)]
struct LegacyStoredTrace {
    version: u32,
    response_txid: String,
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    submitted_daa: u64,
    accepted_hash: Option<String>,
    opening: Option<LegacyStoredOpening>,
    trace: LegacyTrace,
}

impl From<LegacyStoredTrace> for StoredTrace {
    fn from(t: LegacyStoredTrace) -> Self {
        Self {
            version: STORE_VERSION,
            response_txid: t.response_txid,
            model_id: t.model_id,
            tier: t.tier,
            request_hash: t.request_hash,
            max_tokens: t.max_tokens,
            submitted_daa: t.submitted_daa,
            accepted_hash: t.accepted_hash,
            opening: t.opening.map(|o| StoredOpening { seed_hash: o.seed_hash, deadline_daa: o.deadline_daa, payload: o.payload, extra: Vec::new() }),
            trace: crate::trace::ResponseTrace { record: t.trace.record, tokens: t.trace.tokens, seq: None },
            attn: None,
            anchor_hash: None,
            gen_secs: None,
        }
    }
}

fn decode(b: &[u8]) -> Result<StoredTrace, String> {
    match StoredTrace::try_from_slice(b) {
        Ok(t) => Ok(t),
        Err(e) => match V4StoredTrace::try_from_slice(b) {
            Ok(t) if t.version == V4_VERSION => Ok(t.into()),
            _ => match V3StoredTrace::try_from_slice(b) {
                Ok(t) if t.version == V3_VERSION => Ok(t.into()),
                _ => match LegacyStoredTrace::try_from_slice(b) {
                    Ok(t) if t.version == LEGACY_VERSION => Ok(t.into()),
                    _ => Err(e.to_string()),
                },
            },
        },
    }
}

/// [`StoredTrace`] by reference, for writing.
#[derive(BorshSerialize)]
pub struct StoredTraceRef<'a> {
    pub version: u32,
    pub response_txid: &'a str,
    pub model_id: &'a [u8; 32],
    pub tier: u8,
    pub request_hash: &'a [u8; 32],
    pub max_tokens: u32,
    pub submitted_daa: u64,
    pub accepted_hash: &'a Option<String>,
    pub opening: &'a Option<StoredOpening>,
    pub trace: &'a crate::trace::ResponseTrace,
    pub attn: &'a Option<StoredAttn>,
    pub anchor_hash: &'a Option<String>,
    pub gen_secs: Option<f64>,
}

impl<'a> StoredTraceRef<'a> {
    pub fn version() -> u32 {
        STORE_VERSION
    }
}

/// [`StoredTrace`] owned, for a queued write.
pub struct StoredTraceOwned {
    pub response_txid: String,
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub max_tokens: u32,
    pub submitted_daa: u64,
    pub accepted_hash: Option<String>,
    pub opening: Option<StoredOpening>,
    pub trace: Arc<crate::trace::ResponseTrace>,
    pub attn: Option<StoredAttn>,
    pub anchor_hash: Option<String>,
    pub gen_secs: Option<f64>,
}

impl StoredTraceOwned {
    fn record(&self) -> StoredTraceRef<'_> {
        StoredTraceRef {
            version: STORE_VERSION,
            response_txid: &self.response_txid,
            model_id: &self.model_id,
            tier: self.tier,
            request_hash: &self.request_hash,
            max_tokens: self.max_tokens,
            submitted_daa: self.submitted_daa,
            accepted_hash: &self.accepted_hash,
            opening: &self.opening,
            trace: &self.trace,
            attn: &self.attn,
            anchor_hash: &self.anchor_hash,
            gen_secs: self.gen_secs,
        }
    }
}

enum Job {
    Save(PathBuf, Box<StoredTraceOwned>),
    Remove(PathBuf),
    Flush(mpsc::Sender<()>),
}

/// The process-wide writer: one thread applies every queued write and removal in order.
fn writer() -> &'static mpsc::Sender<Job> {
    static WRITER: OnceLock<mpsc::Sender<Job>> = OnceLock::new();
    WRITER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("trace-store".into())
            .spawn(move || {
                for job in rx {
                    match job {
                        Job::Save(dir, rec) => {
                            let written = dir.join(format!("{}.{}", rec.response_txid, FILE_EXT)).exists();
                            let res = if written { write_state(&dir, &rec) } else { write_record(&dir, &rec.record()) };
                            if let Err(e) = res {
                                log::warn!("Trace: cannot persist response {}: {}", rec.response_txid, e);
                            }
                        }
                        Job::Remove(path) => {
                            let _ = fs::remove_file(path);
                        }
                        Job::Flush(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
            .expect("trace store writer thread");
        tx
    })
}

/// Writes `bytes` to `path` atomically: temporary file, sync, rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        io::Write::write_all(&mut f, bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}

/// Writes the whole record and drops any older state file.
fn write_record(dir: &Path, record: &StoredTraceRef) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let bytes = borsh::to_vec(record).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_atomic(&dir.join(format!("{}.{}", record.response_txid, FILE_EXT)), &bytes)?;
    let _ = fs::remove_file(dir.join(format!("{}.{}", record.response_txid, STATE_EXT)));
    Ok(())
}

/// Writes the mutable fields of a record whose trace file is already written.
fn write_state(dir: &Path, rec: &StoredTraceOwned) -> io::Result<()> {
    let state = StoredState {
        accepted_hash: rec.accepted_hash.clone(),
        opening: rec.opening.clone(),
        attn: rec.attn.clone(),
        anchor_hash: rec.anchor_hash.clone(),
        gen_secs: rec.gen_secs,
    };
    let bytes = borsh::to_vec(&state).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_atomic(&dir.join(format!("{}.{}", rec.response_txid, STATE_EXT)), &bytes)
}

pub struct TraceStore {
    dir: PathBuf,
}

impl TraceStore {
    /// Store next to the escrow state file: `<state>.traces/`.
    pub fn new(escrow_state_file: &str) -> Self {
        Self { dir: Path::new(escrow_state_file).with_extension("traces") }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, txid: &str) -> PathBuf {
        self.dir.join(format!("{}.{}", txid, FILE_EXT))
    }

    /// Writes the record now.
    pub fn save(&self, record: &StoredTraceRef) -> io::Result<()> {
        write_record(&self.dir, record)
    }

    /// Queues the record for the writer thread.
    pub fn save_queued(&self, record: StoredTraceOwned) {
        let _ = writer().send(Job::Save(self.dir.clone(), Box::new(record)));
    }

    /// Queues the record and signals `done` once it is written.
    pub fn save_queued_notify(&self, record: StoredTraceOwned, done: mpsc::Sender<()>) {
        let _ = writer().send(Job::Save(self.dir.clone(), Box::new(record)));
        let _ = writer().send(Job::Flush(done));
    }

    /// Whether the trace file of `txid` is on disk.
    pub fn contains(&self, txid: &str) -> bool {
        self.path(txid).exists()
    }

    /// Queues the removal of a record behind the writes already queued.
    pub fn remove(&self, txid: &str) {
        let _ = writer().send(Job::Remove(self.path(txid)));
        let _ = writer().send(Job::Remove(self.dir.join(format!("{}.{}", txid, STATE_EXT))));
    }

    /// Waits until every queued write and removal is applied.
    pub fn flush(&self) {
        let (done, wait) = mpsc::channel();
        if writer().send(Job::Flush(done)).is_ok() {
            let _ = wait.recv();
        }
    }

    /// Every readable record, once the queued writes are applied; unreadable files are dropped
    /// with a warning.
    pub fn load_all(&self) -> Vec<StoredTrace> {
        self.flush();
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.dir) else { return out };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(FILE_EXT) {
                continue;
            }
            match fs::read(&path).map_err(|e| e.to_string()).and_then(|b| decode(&b)) {
                Ok(mut t) if t.version == STORE_VERSION => {
                    let state = fs::read(path.with_extension(STATE_EXT)).ok().and_then(|b| StoredState::try_from_slice(&b).ok());
                    if let Some(st) = state {
                        t.accepted_hash = st.accepted_hash;
                        t.opening = st.opening;
                        t.attn = st.attn;
                        t.anchor_hash = st.anchor_hash;
                        t.gen_secs = st.gen_secs;
                    }
                    out.push(t)
                }
                Ok(t) => {
                    log::warn!("Trace: stored record {} has version {} — dropped", path.display(), t.version);
                    let _ = fs::remove_file(&path);
                }
                Err(e) => {
                    log::warn!("Trace: stored record {} unreadable ({}) — dropped", path.display(), e);
                    let _ = fs::remove_file(&path);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips_through_the_store() {
        let dir = std::env::temp_dir().join(format!("keryx-trace-store-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = TraceStore { dir: dir.clone() };
        let trace = crate::trace::ResponseTrace::test_sample();
        let opening = Some(StoredOpening { seed_hash: "ab".into(), deadline_daa: 42, payload: "00ff".into(), extra: vec!["11".into()] });
        let attn = Some(StoredAttn { bodies: vec![vec![5, 0, 1], vec![5, 2]], payload: "beef".into(), deadline_daa: 3_400, accepted_hash: Some("f00d".into()) });
        let rec = StoredTraceRef {
            version: StoredTraceRef::version(),
            response_txid: "deadbeef",
            model_id: &[1u8; 32],
            tier: 3,
            request_hash: &[2u8; 32],
            max_tokens: 2048,
            submitted_daa: 7,
            accepted_hash: &Some("cafe".into()),
            opening: &opening,
            trace: &trace,
            attn: &attn,
            anchor_hash: &Some("0a".into()),
            gen_secs: Some(12.5),
        };
        store.save(&rec).unwrap();
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 1);
        let t = &loaded[0];
        assert_eq!(t.response_txid, "deadbeef");
        assert_eq!(t.tier, 3);
        assert_eq!(t.accepted_hash.as_deref(), Some("cafe"));
        assert_eq!(t.opening.as_ref().unwrap().deadline_daa, 42);
        assert_eq!(t.trace.tokens, trace.tokens);
        assert_eq!(t.trace.record.trace_root, trace.record.trace_root);
        assert_eq!(t.attn, attn);
        assert_eq!(t.anchor_hash.as_deref(), Some("0a"));
        assert_eq!(t.gen_secs, Some(12.5));
        store.remove("deadbeef");
        assert!(store.load_all().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn later_writes_keep_the_trace_file_and_update_the_state() {
        let dir = std::env::temp_dir().join(format!("keryx-trace-store-state-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = TraceStore { dir: dir.clone() };
        let trace = Arc::new(crate::trace::ResponseTrace::test_sample());
        let owned = |accepted: Option<&str>| StoredTraceOwned {
            response_txid: "cc".into(),
            model_id: [1u8; 32],
            tier: 1,
            request_hash: [2u8; 32],
            max_tokens: 16,
            submitted_daa: 3,
            accepted_hash: accepted.map(Into::into),
            opening: None,
            trace: Arc::clone(&trace),
            attn: None,
            anchor_hash: None,
            gen_secs: None,
        };
        store.save_queued(owned(None));
        store.flush();
        let trace_file = dir.join("cc.trace");
        let first = fs::metadata(&trace_file).unwrap().modified().unwrap();
        store.save_queued(owned(Some("beef")));
        store.flush();
        assert_eq!(fs::metadata(&trace_file).unwrap().modified().unwrap(), first);
        assert!(dir.join("cc.state").exists());
        let loaded = store.load_all();
        assert_eq!(loaded[0].accepted_hash.as_deref(), Some("beef"));
        store.remove("cc");
        store.flush();
        assert!(!dir.join("cc.state").exists() && !trace_file.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn queued_writes_apply_in_order_and_v4_records_still_load() {
        let dir = std::env::temp_dir().join(format!("keryx-trace-store-queue-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = TraceStore { dir: dir.clone() };
        let trace = Arc::new(crate::trace::ResponseTrace::test_sample());
        let owned = |txid: &str, anchor: Option<&str>| StoredTraceOwned {
            response_txid: txid.into(),
            model_id: [1u8; 32],
            tier: 1,
            request_hash: [2u8; 32],
            max_tokens: 16,
            submitted_daa: 3,
            accepted_hash: None,
            opening: None,
            trace: Arc::clone(&trace),
            attn: None,
            anchor_hash: anchor.map(Into::into),
            gen_secs: None,
        };
        store.save_queued(owned("aa", Some("01")));
        store.remove("aa");
        store.remove("bb");
        store.save_queued(owned("bb", Some("02")));
        store.save_queued(owned("bb", Some("03")));
        #[derive(BorshSerialize)]
        struct V4Ref<'a> {
            version: u32,
            response_txid: &'a str,
            model_id: [u8; 32],
            tier: u8,
            request_hash: [u8; 32],
            max_tokens: u32,
            submitted_daa: u64,
            accepted_hash: Option<String>,
            opening: Option<StoredOpening>,
            trace: &'a crate::trace::ResponseTrace,
            attn: Option<StoredAttn>,
        }
        let attn = Some(StoredAttn { bodies: vec![vec![1]], payload: "ab".into(), deadline_daa: 9, accepted_hash: None });
        let v4 = V4Ref {
            version: V4_VERSION,
            response_txid: "cc",
            model_id: [1u8; 32],
            tier: 2,
            request_hash: [2u8; 32],
            max_tokens: 16,
            submitted_daa: 1,
            accepted_hash: Some("ee".into()),
            opening: None,
            trace: &trace,
            attn: attn.clone(),
        };
        store.flush();
        fs::write(store.path("cc"), borsh::to_vec(&v4).unwrap()).unwrap();
        let mut loaded = store.load_all();
        loaded.sort_by(|a, b| a.response_txid.cmp(&b.response_txid));
        assert_eq!(loaded.iter().map(|t| t.response_txid.as_str()).collect::<Vec<_>>(), vec!["bb", "cc"]);
        assert_eq!(loaded[0].anchor_hash.as_deref(), Some("03"));
        assert_eq!((loaded[1].version, loaded[1].anchor_hash.is_none(), &loaded[1].attn), (STORE_VERSION, true, &attn));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn records_keep_their_sequence_leaves_and_legacy_records_still_load() {
        #[derive(BorshSerialize)]
        struct LegacyRef<'a> {
            version: u32,
            response_txid: &'a str,
            model_id: [u8; 32],
            tier: u8,
            request_hash: [u8; 32],
            max_tokens: u32,
            submitted_daa: u64,
            accepted_hash: Option<String>,
            opening: Option<(String, u64, String)>,
            record: &'a crate::trace::TraceRecord,
            tokens: &'a Vec<i32>,
        }
        let dir = std::env::temp_dir().join(format!("keryx-trace-store-seq-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = TraceStore { dir: dir.clone() };
        let mut trace = crate::trace::ResponseTrace::test_sample();
        trace.seq = crate::trace::SeqTrace::new(vec![[3u8; 32], [4u8; 32], [5u8; 32]]);
        let rec = StoredTraceRef {
            version: StoredTraceRef::version(),
            response_txid: "aa",
            model_id: &[1u8; 32],
            tier: 0,
            request_hash: &[2u8; 32],
            max_tokens: 16,
            submitted_daa: 1,
            accepted_hash: &None,
            opening: &None,
            trace: &trace,
            attn: &None,
            anchor_hash: &None,
            gen_secs: None,
        };
        store.save(&rec).unwrap();
        #[derive(BorshSerialize)]
        struct V3Ref<'a> {
            version: u32,
            response_txid: &'a str,
            model_id: [u8; 32],
            tier: u8,
            request_hash: [u8; 32],
            max_tokens: u32,
            submitted_daa: u64,
            accepted_hash: Option<String>,
            opening: Option<StoredOpening>,
            trace: &'a crate::trace::ResponseTrace,
        }
        let v3 = V3Ref {
            version: V3_VERSION,
            response_txid: "cc",
            model_id: [1u8; 32],
            tier: 2,
            request_hash: [2u8; 32],
            max_tokens: 16,
            submitted_daa: 1,
            accepted_hash: Some("ee".into()),
            opening: None,
            trace: &trace,
        };
        fs::write(store.path("cc"), borsh::to_vec(&v3).unwrap()).unwrap();
        let legacy = LegacyRef {
            version: LEGACY_VERSION,
            response_txid: "bb",
            model_id: [1u8; 32],
            tier: 1,
            request_hash: [2u8; 32],
            max_tokens: 16,
            submitted_daa: 1,
            accepted_hash: None,
            opening: Some(("cd".into(), 9, "beef".into())),
            record: &trace.record,
            tokens: &trace.tokens,
        };
        fs::write(store.path("bb"), borsh::to_vec(&legacy).unwrap()).unwrap();
        let mut loaded = store.load_all();
        loaded.sort_by(|a, b| a.response_txid.cmp(&b.response_txid));
        assert_eq!(loaded.len(), 3);
        assert_eq!((loaded[2].version, loaded[2].tier, loaded[2].attn.is_none()), (STORE_VERSION, 2, true));
        assert_eq!(loaded[2].trace.seq.as_ref().map(|s| s.root), trace.seq.as_ref().map(|s| s.root));
        let seq = loaded[0].trace.seq.as_ref().unwrap();
        assert_eq!((seq.leaves.len(), seq.root), (3, trace.seq.as_ref().unwrap().root));
        assert_ne!(loaded[0].trace.commitment().trace_root, loaded[0].trace.record.trace_root);
        assert!(loaded[1].trace.seq.is_none() && loaded[1].version == STORE_VERSION);
        let o = loaded[1].opening.as_ref().unwrap();
        assert_eq!((o.payload.as_str(), o.deadline_daa, o.extra.len()), ("beef", 9, 0));
        assert_eq!(loaded[1].trace.commitment().trace_root, trace.record.trace_root);
        let _ = fs::remove_dir_all(&dir);
    }
}
