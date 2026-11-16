//! Block task of the mining devices: window seeds, the task trace of each device and its root.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use keryx_verifiable_inference::block_task::window;

use crate::trace::ResponseTrace;

/// The task trace a device commits for one window.
pub struct DeviceTask {
    pub window: u64,
    pub seed: [u8; 32],
    pub model_id: [u8; 32],
    pub tier: u8,
    pub trace: ResponseTrace,
}

impl DeviceTask {
    pub fn root(&self) -> [u8; 32] {
        self.trace.record.trace_root
    }
}

static SEEDS: Mutex<BTreeMap<(u64, String), [u8; 32]>> = Mutex::new(BTreeMap::new());
/// Task of a device for a window seed.
type TaskKey = (u32, [u8; 32]);
static TASKS: Mutex<BTreeMap<TaskKey, Arc<DeviceTask>>> = Mutex::new(BTreeMap::new());
static FAILED: Mutex<BTreeMap<u32, [u8; 32]>> = Mutex::new(BTreeMap::new());

/// Max windows whose seeds are kept.
const SEED_WINDOWS: u64 = 4;

/// Whether a block at `daa` carries the task.
pub fn active(daa: u64) -> bool {
    daa >= crate::pom::block_task_activation_daa()
}

/// Key of a template's direct parents. Two templates of one window may sit on different selected
/// chains, and so on different seeds.
pub fn parents_key(parents: &[String]) -> String {
    let mut p: Vec<String> = parents.iter().map(|h| h.to_ascii_lowercase()).collect();
    p.sort_unstable();
    p.join(",")
}

/// Records the seed block of window `w` for a template over `parents`.
pub fn set_seed(w: u64, parents: &str, seed: [u8; 32]) {
    let mut s = SEEDS.lock().unwrap_or_else(|p| p.into_inner());
    s.insert((w, parents.to_string()), seed);
    s.retain(|&(sw, _), _| sw + SEED_WINDOWS > w);
}

pub fn seed(w: u64, parents: &str) -> Option<[u8; 32]> {
    SEEDS.lock().unwrap_or_else(|p| p.into_inner()).get(&(w, parents.to_string())).copied()
}

/// The task of `device` (mining `model_id` at `tier`) for a block at `daa` over `parents`, traced on
/// first use of its seed. `None` while the seed is unknown or the task cannot be traced.
pub fn task_for(device: u32, model_id: &[u8; 32], tier: u8, daa: u64, parents: &str) -> Option<Arc<DeviceTask>> {
    let w = window(daa)?;
    let seed = seed(w, parents)?;
    if let Some(t) = TASKS.lock().unwrap_or_else(|p| p.into_inner()).get(&(device, seed)) {
        if t.model_id == *model_id && t.tier == tier {
            return Some(Arc::clone(t));
        }
    }
    if FAILED.lock().unwrap_or_else(|p| p.into_inner()).get(&device) == Some(&seed) {
        return None;
    }
    let Some(trace) = crate::llm::run_block_task(model_id, tier, &seed) else {
        log::warn!("Block task: gpu{} cannot trace the task of window {} — not mining on this seed", device, w);
        FAILED.lock().unwrap_or_else(|p| p.into_inner()).insert(device, seed);
        return None;
    };
    let task = Arc::new(DeviceTask { window: w, seed, model_id: *model_id, tier, trace });
    let mut tasks = TASKS.lock().unwrap_or_else(|p| p.into_inner());
    tasks.retain(|&(d, _), t| d != device || t.window + 1 >= w);
    tasks.insert((device, seed), Arc::clone(&task));
    Some(task)
}
