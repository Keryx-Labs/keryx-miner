use cudarc::driver::{result, sys};
use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, structs::device::FieldId, Nvml};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const STATS_READ_TIMEOUT_SECS: u64 = 5;
const STATS_WRITE_TIMEOUT_SECS: u64 = 5;
const MAX_REQUEST_LINE_BYTES: usize = 4096;
const MAX_STATS_CONNECTIONS: usize = 8;

struct StatsConnectionPermit {
    active: Arc<AtomicUsize>,
}

impl StatsConnectionPermit {
    fn acquire(active: &Arc<AtomicUsize>) -> Option<Self> {
        active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_STATS_CONNECTIONS).then_some(count + 1)
            })
            .ok()?;
        Some(Self { active: Arc::clone(active) })
    }
}

impl Drop for StatsConnectionPermit {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Release);
    }
}

static NVML_HANDLE: OnceLock<Option<Nvml>> = OnceLock::new();

#[derive(Default)]
pub struct MinerStats {
    started_at: Mutex<Option<Instant>>,
    started_epoch_s: AtomicU64,
    synced: AtomicBool,
    opoi_challenge_active: AtomicBool,
    total_hashrate_hs: AtomicU64,
    accepted_blocks: AtomicU64,
    rejected_blocks: AtomicU64,
    claimed_outputs: AtomicU64,
    claimed_sompi: AtomicU64,
    escrow_pending_outputs: AtomicU64,
    escrow_pending_sompi: AtomicU64,
    last_update_epoch_s: AtomicU64,
    api_port: AtomicU64,
    mining_address: Mutex<Option<String>>,
    /// Compact service-bond standing for the status bar: "clear", "strike 2", "suspended".
    service_status: Mutex<Option<String>>,
    device_hashrate_hs: Mutex<HashMap<String, u64>>,
    device_blocks_accepted: Mutex<HashMap<String, u64>>,
    device_blocks_rejected: Mutex<HashMap<String, u64>>,
    gpu_telemetry: Mutex<HashMap<u32, GpuTelemetry>>,
    gpu_memory_temp_supported: Mutex<HashMap<u32, bool>>,
    hiveos: AtomicBool,
    // ── IPFS health (issue #19) ───────────────────────────────────────────────
    // Updated at real `/api/v0/version` probes and `/api/v0/add` upload outcomes.
    // Never holds uploaded inference text, prompts, or secrets — only the CIDv0
    // string and short error messages.
    ipfs_api_url: Mutex<Option<String>>,
    ipfs_reachable: AtomicBool,
    kubo_version: Mutex<Option<String>>,
    daemon_managed: AtomicBool,
    last_upload_epoch_s: AtomicU64,
    last_upload_success: AtomicBool,
    last_upload_cid: Mutex<Option<String>>,
    last_error: Mutex<Option<String>>,
    inference_rewards_enabled: AtomicBool,
}

#[derive(Default, Clone, Copy)]
struct GpuTelemetry {
    temp_c: Option<u32>,
    memory_temp_c: Option<u32>,
    fan_percent: Option<u32>,
    power_draw_w: Option<f32>,
}

#[derive(Serialize)]
pub struct DeviceRate {
    pub id: String,
    pub hashrate_hs: u64,
    pub blocks_accepted: u64,
    pub blocks_rejected: u64,
    // Backward-compatible alias for core temp.
    pub temp_c: Option<u32>,
    pub memory_temp_c: Option<u32>,
    pub fan_percent: Option<u32>,
    pub power_draw_w: Option<f32>,
}

#[derive(Serialize)]
pub struct MinerStatsSnapshot {
    pub started_epoch_s: u64,
    pub uptime_s: u64,
    pub synced: bool,
    pub opoi_challenge_active: bool,
    pub mining_address: Option<String>,
    pub service_status: Option<String>,
    pub api_port: Option<u16>,
    pub total_hashrate_hs: u64,
    pub accepted_blocks: u64,
    pub rejected_blocks: u64,
    pub claimed_outputs: u64,
    pub claimed_sompi: u64,
    pub escrow_pending_outputs: u64,
    pub escrow_pending_sompi: u64,
    pub last_update_epoch_s: u64,
    pub devices: Vec<DeviceRate>,
    pub ipfs: IpfsStats,
}

/// Nested IPFS health section of the stats snapshot (issue #19).
/// `last_error` carries a short diagnostic only — never uploaded content or secrets.
#[derive(Serialize, Default)]
pub struct IpfsStats {
    pub api_url: Option<String>,
    pub api_reachable: bool,
    pub kubo_version: Option<String>,
    pub daemon_managed: bool,
    pub last_upload_epoch_s: u64,
    pub last_upload_success: bool,
    pub last_upload_cid: Option<String>,
    pub last_error: Option<String>,
    pub inference_rewards_enabled: bool,
}

impl MinerStats {
    pub fn new(hiveos: bool) -> Self {
        let now = now_epoch_s();
        Self {
            started_at: Mutex::new(Some(Instant::now())),
            started_epoch_s: AtomicU64::new(now),
            synced: AtomicBool::new(true),
            opoi_challenge_active: AtomicBool::new(false),
            total_hashrate_hs: AtomicU64::new(0),
            accepted_blocks: AtomicU64::new(0),
            rejected_blocks: AtomicU64::new(0),
            claimed_outputs: AtomicU64::new(0),
            claimed_sompi: AtomicU64::new(0),
            escrow_pending_outputs: AtomicU64::new(0),
            escrow_pending_sompi: AtomicU64::new(0),
            last_update_epoch_s: AtomicU64::new(now),
            api_port: AtomicU64::new(0),
            mining_address: Mutex::new(None),
            service_status: Mutex::new(None),
            device_hashrate_hs: Mutex::new(HashMap::new()),
            device_blocks_accepted: Mutex::new(HashMap::new()),
            device_blocks_rejected: Mutex::new(HashMap::new()),
            gpu_telemetry: Mutex::new(HashMap::new()),
            gpu_memory_temp_supported: Mutex::new(HashMap::new()),
            hiveos: AtomicBool::new(hiveos),
            ipfs_api_url: Mutex::new(None),
            ipfs_reachable: AtomicBool::new(false),
            kubo_version: Mutex::new(None),
            daemon_managed: AtomicBool::new(false),
            last_upload_epoch_s: AtomicU64::new(0),
            last_upload_success: AtomicBool::new(false),
            last_upload_cid: Mutex::new(None),
            last_error: Mutex::new(None),
            inference_rewards_enabled: AtomicBool::new(false),
        }
    }

    pub fn set_api_port(&self, port: u16) {
        self.api_port.store(port as u64, Ordering::Release);
    }

    pub fn set_service_status(&self, status: Option<String>) {
        if let Ok(mut slot) = self.service_status.lock() {
            *slot = status;
        }
    }

    pub fn set_mining_address(&self, address: Option<String>) {
        if let Ok(mut slot) = self.mining_address.lock() {
            *slot = address;
        }
    }

    pub fn set_synced(&self, synced: bool) {
        self.synced.store(synced, Ordering::Release);
    }

    pub fn set_opoi_challenge_active(&self, active: bool) {
        self.opoi_challenge_active.store(active, Ordering::Release);
    }

    pub fn set_hashrates(&self, total_hs: u64, per_device_hs: &HashMap<String, u64>) {
        self.total_hashrate_hs.store(total_hs, Ordering::Release);
        self.last_update_epoch_s.store(now_epoch_s(), Ordering::Release);
        let mut map = self.device_hashrate_hs.lock().expect("device stats mutex poisoned");
        map.clear();
        map.extend(per_device_hs.iter().map(|(k, v)| (k.clone(), *v)));
    }

    pub fn inc_accepted_blocks(&self) {
        self.accepted_blocks.fetch_add(1, Ordering::AcqRel);
        self.last_update_epoch_s.store(now_epoch_s(), Ordering::Release);
    }

    pub fn inc_device_blocks_accepted(&self, device_id: &str) {
        let mut map = self.device_blocks_accepted.lock().expect("device block count mutex poisoned");
        *map.entry(device_id.to_string()).or_insert(0) += 1;
    }

    pub fn inc_rejected_blocks(&self) {
        self.rejected_blocks.fetch_add(1, Ordering::AcqRel);
        self.last_update_epoch_s.store(now_epoch_s(), Ordering::Release);
    }

    pub fn inc_device_blocks_rejected(&self, device_id: &str) {
        let mut map = self.device_blocks_rejected.lock().expect("device rejected block count mutex poisoned");
        *map.entry(device_id.to_string()).or_insert(0) += 1;
    }

    pub fn add_claimed(&self, outputs: u64, amount_sompi: u64) {
        self.claimed_outputs.fetch_add(outputs, Ordering::AcqRel);
        self.claimed_sompi.fetch_add(amount_sompi, Ordering::AcqRel);
        self.last_update_epoch_s.store(now_epoch_s(), Ordering::Release);
    }

    pub fn set_escrow_pending(&self, outputs: u64, amount_sompi: u64) {
        self.escrow_pending_outputs.store(outputs, Ordering::Release);
        self.escrow_pending_sompi.store(amount_sompi, Ordering::Release);
    }

    // ── IPFS health (issue #19) ────────────────────────────────────────────────

    /// Record the API URL the miner was configured with. Called once at startup.
    pub fn set_ipfs_api_url(&self, url: String) {
        if let Ok(mut slot) = self.ipfs_api_url.lock() {
            *slot = Some(url);
        }
    }

    /// Record the outcome of a `/api/v0/version` probe: reachability, the kubo
    /// version reported by the node, and whether the daemon is auto-managed.
    /// Failures must not panic — an unreachable node only flips the flags.
    pub fn set_ipfs_version_probe(&self, reachable: bool, kubo_version: Option<String>, daemon_managed: bool) {
        self.ipfs_reachable.store(reachable, Ordering::Release);
        self.daemon_managed.store(daemon_managed, Ordering::Release);
        if let Ok(mut slot) = self.kubo_version.lock() {
            *slot = kubo_version;
        }
    }

    /// Record the outcome of a `/api/v0/add` upload: CIDv0 string on success,
    /// short diagnostic on failure. Never stores uploaded text or secrets.
    pub fn set_ipfs_upload_outcome(&self, success: bool, cid: Option<String>, error: Option<String>) {
        self.last_upload_epoch_s.store(now_epoch_s(), Ordering::Release);
        self.last_upload_success.store(success, Ordering::Release);
        if let Ok(mut slot) = self.last_upload_cid.lock() {
            *slot = cid;
        }
        if let Ok(mut slot) = self.last_error.lock() {
            *slot = error;
        }
    }

    /// Whether inference rewards are expected to work: the daemon is reachable
    /// (or locally auto-managed) and the API is configured.
    pub fn set_inference_rewards_enabled(&self, enabled: bool) {
        self.inference_rewards_enabled.store(enabled, Ordering::Release);
    }

    /// Current reachability flag, as last recorded by a `/api/v0/version` probe.
    pub fn is_ipfs_reachable(&self) -> bool {
        self.ipfs_reachable.load(Ordering::Acquire)
    }

    /// Kubo version reported by the last successful `/api/v0/version` probe.
    pub fn kubo_version(&self) -> Option<String> {
        self.kubo_version.lock().ok().and_then(|s| s.clone())
    }

    /// Lock guard for the short last-error diagnostic slot (ipfs.rs uses this
    /// so probe failures stay non-panicking).
    pub fn last_error_lock(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.last_error.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn refresh_gpu_telemetry(&self) {
        let cuda_bus_ids = cuda_device_bus_ids();
        let mut physical_to_logical = HashMap::new();
        let mut fresh = HashMap::new();
        let mut nvml_memory_temps = HashMap::new();
        let mut nvml_fallbacks = HashMap::new();

        let nvml = NVML_HANDLE.get_or_init(|| Nvml::init().ok());
        if let Some(nvml) = nvml.as_ref() {
            if let Ok(device_count) = nvml.device_count() {
                for idx in 0..device_count {
                    let Ok(device) = nvml.device_by_index(idx) else {
                        continue;
                    };
                    let logical_idx = device
                        .pci_info()
                        .ok()
                        .and_then(|pci| logical_device_number(&pci.bus_id, idx, cuda_bus_ids))
                        .or_else(|| cuda_bus_ids.is_empty().then_some(idx));
                    let Some(logical_idx) = logical_idx else {
                        continue;
                    };
                    physical_to_logical.insert(idx, logical_idx);

                    let temp_c = device.temperature(TemperatureSensor::Gpu).ok();
                    let fan_percent = device.fan_speed(0).ok();
                    let power_draw_w = device
                        .power_usage()
                        .ok()
                        .map(|milliwatts| normalize_power_draw_w(Some(milliwatts as f32), None))
                        .flatten();

                    if let Ok(field_values) = device.field_values_for(&[FieldId(82)]) {
                        if let Some(Ok(field_sample)) = field_values.first() {
                            if let Ok(value) = &field_sample.value {
                                let temp = match value {
                                    nvml_wrapper::enums::device::SampleValue::I64(temp) => Some(*temp as i64),
                                    nvml_wrapper::enums::device::SampleValue::U32(temp) => Some(*temp as i64),
                                    nvml_wrapper::enums::device::SampleValue::U64(temp) => Some(*temp as i64),
                                    nvml_wrapper::enums::device::SampleValue::F64(_) => None,
                                };
                                if let Some(temp) = temp.filter(|temp| *temp > 0) {
                                    nvml_memory_temps.insert(logical_idx, temp as u32);
                                }
                            }
                        }
                    }

                    nvml_fallbacks.insert(
                        logical_idx,
                        GpuTelemetry {
                            temp_c: temp_c.map(|temp| temp as u32),
                            memory_temp_c: nvml_memory_temps.get(&logical_idx).copied(),
                            fan_percent: fan_percent.map(|fan| fan as u32),
                            power_draw_w,
                        },
                    );
                }
            }
        }

        if !nvml_fallbacks.is_empty() {
            fresh = nvml_fallbacks.clone();
        }

        let mut memory_temp_supported = self.gpu_memory_temp_supported.lock().expect("gpu telemetry mutex poisoned");
        let should_query_nvidia_smi = should_query_nvidia_smi(
            &fresh,
            &memory_temp_supported,
            self.hiveos.load(Ordering::Acquire),
        );
        let output = if should_query_nvidia_smi {
            Some(
                Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=pci.bus_id,temperature.gpu,temperature.memory,fan.speed,power.draw",
                        "--format=csv,noheader,nounits",
                    ])
                    .output(),
            )
        } else {
            None
        };

        if let Some(Ok(output)) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (fallback_idx, line) in stdout.lines().enumerate() {
                    let mut parts = line.split(',').map(|s| s.trim());
                    let pci_bus_id = parts.next().unwrap_or_default();
                    let Some(gpu_idx) = logical_device_number(pci_bus_id, fallback_idx as u32, cuda_bus_ids) else {
                        continue;
                    };
                    physical_to_logical.insert(fallback_idx as u32, gpu_idx);
                    let temp_c = parts.next().and_then(parse_u32_field);
                    let nvidia_smi_memory_temp_c = parts.next().and_then(parse_u32_field);
                    let fan_percent = parts.next().and_then(parse_u32_field);
                    let power_draw_w = parts.next().and_then(parse_f32_field);

                    if let Some(telemetry) = fresh.get_mut(&gpu_idx) {
                        telemetry.temp_c = prefer_nvml_u32_or_nvidia_smi(telemetry.temp_c, temp_c);
                        telemetry.memory_temp_c = normalize_memory_temp_c(
                            nvidia_smi_memory_temp_c,
                            telemetry.memory_temp_c,
                        );
                        telemetry.fan_percent = prefer_nvml_u32_or_nvidia_smi(telemetry.fan_percent, fan_percent);
                        telemetry.power_draw_w = prefer_nvml_f32_or_nvidia_smi(telemetry.power_draw_w, power_draw_w);
                    } else {
                        fresh.insert(
                            gpu_idx,
                            GpuTelemetry {
                                temp_c: temp_c,
                                memory_temp_c: normalize_memory_temp_c(nvidia_smi_memory_temp_c, None),
                                fan_percent,
                                power_draw_w,
                            },
                        );
                    }

                    if !self.hiveos.load(Ordering::Acquire) {
                        memory_temp_supported.insert(gpu_idx, fresh.get(&gpu_idx).and_then(|entry| entry.memory_temp_c).is_some());
                    }
                }
            }
        }

        let has_missing_memory_temp = fresh.is_empty() || fresh.values().any(|entry| entry.memory_temp_c.is_none());
        if let Ok(mut map) = self.gpu_telemetry.lock() {
            *map = fresh;
        }

        if self.hiveos.load(Ordering::Acquire) && has_missing_memory_temp {
            merge_physical_to_logical(&mut physical_to_logical, nvidia_smi_device_map(cuda_bus_ids));
            if let Some(hiveos_memtemps) = read_hiveos_nvtool_memtemps() {
                if let Ok(mut map) = self.gpu_telemetry.lock() {
                    for (physical_idx, memtemp) in hiveos_memtemps {
                        let Some(logical_idx) =
                            logical_nvtool_device_number(physical_idx, &physical_to_logical, cuda_bus_ids)
                        else {
                            continue;
                        };
                        let entry = map.entry(logical_idx).or_default();
                        if entry.memory_temp_c.is_none() {
                            entry.memory_temp_c = Some(memtemp);
                        }
                    }
                }
            }
        }
    }

    pub fn snapshot(&self) -> MinerStatsSnapshot {
        let started_epoch_s = self.started_epoch_s.load(Ordering::Acquire);
        let uptime_s = self
            .started_at
            .lock()
            .expect("start time mutex poisoned")
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);

        let telemetry = self
            .gpu_telemetry
            .lock()
            .expect("gpu telemetry mutex poisoned")
            .clone();
        let service_status = self.service_status.lock().ok().and_then(|s| s.clone());
        let mining_address = self
            .mining_address
            .lock()
            .expect("mining address mutex poisoned")
            .clone();

        let device_blocks_accepted = self.device_blocks_accepted.lock().expect("device block count mutex poisoned").clone();
        let device_blocks_rejected = self.device_blocks_rejected.lock().expect("device rejected block count mutex poisoned").clone();
        let mut devices = self
            .device_hashrate_hs
            .lock()
            .expect("device stats mutex poisoned")
            .iter()
            .map(|(id, rate)| {
                let gpu_idx = parse_device_number(id);
                let telem = gpu_idx.and_then(|idx| telemetry.get(&idx).copied());
                DeviceRate {
                    id: id.clone(),
                    hashrate_hs: *rate,
                    blocks_accepted: device_blocks_accepted.get(id).copied().unwrap_or(0),
                    blocks_rejected: device_blocks_rejected.get(id).copied().unwrap_or(0),
                    temp_c: telem.and_then(|t| t.temp_c),
                    memory_temp_c: telem.and_then(|t| t.memory_temp_c),
                    fan_percent: telem.and_then(|t| t.fan_percent),
                    power_draw_w: telem.and_then(|t| t.power_draw_w),
                }
            })
            .collect::<Vec<_>>();
        devices.sort_by(|a, b| {
            let a_num = parse_device_number(&a.id);
            let b_num = parse_device_number(&b.id);
            match (a_num, b_num) {
                (Some(x), Some(y)) => x.cmp(&y),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.id.cmp(&b.id),
            }
        });

        MinerStatsSnapshot {
            started_epoch_s,
            uptime_s,
            synced: self.synced.load(Ordering::Acquire),
            opoi_challenge_active: self.opoi_challenge_active.load(Ordering::Acquire),
            mining_address,
            service_status,
            api_port: match self.api_port.load(Ordering::Acquire) {
                0 => None,
                p => Some(p as u16),
            },
            total_hashrate_hs: self.total_hashrate_hs.load(Ordering::Acquire),
            accepted_blocks: self.accepted_blocks.load(Ordering::Acquire),
            rejected_blocks: self.rejected_blocks.load(Ordering::Acquire),
            claimed_outputs: self.claimed_outputs.load(Ordering::Acquire),
            claimed_sompi: self.claimed_sompi.load(Ordering::Acquire),
            escrow_pending_outputs: self.escrow_pending_outputs.load(Ordering::Acquire),
            escrow_pending_sompi: self.escrow_pending_sompi.load(Ordering::Acquire),
            last_update_epoch_s: self.last_update_epoch_s.load(Ordering::Acquire),
            devices,
            ipfs: IpfsStats {
                api_url: self.ipfs_api_url.lock().ok().and_then(|s| s.clone()),
                api_reachable: self.ipfs_reachable.load(Ordering::Acquire),
                kubo_version: self.kubo_version.lock().ok().and_then(|s| s.clone()),
                daemon_managed: self.daemon_managed.load(Ordering::Acquire),
                last_upload_epoch_s: self.last_upload_epoch_s.load(Ordering::Acquire),
                last_upload_success: self.last_upload_success.load(Ordering::Acquire),
                last_upload_cid: self.last_upload_cid.lock().ok().and_then(|s| s.clone()),
                last_error: self.last_error.lock().ok().and_then(|s| s.clone()),
                inference_rewards_enabled: self.inference_rewards_enabled.load(Ordering::Acquire),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_accepted_block_counts_are_reported_separately_from_rejections() {
        let stats = MinerStats::new(false);
        let mut per_device_hashrates = HashMap::new();
        per_device_hashrates.insert("GPU0".to_string(), 100);
        stats.set_hashrates(100, &per_device_hashrates);

        stats.inc_device_blocks_accepted("GPU0");
        stats.inc_device_blocks_rejected("GPU0");

        let snapshot = stats.snapshot();
        let device = snapshot.devices.iter().find(|device| device.id == "GPU0").unwrap();
        assert_eq!(device.blocks_accepted, 1);
        assert_eq!(device.blocks_rejected, 1);
    }

    #[test]
    fn ipfs_section_defaults_are_observable_without_a_daemon() {
        let stats = MinerStats::new(false);
        let snapshot = stats.snapshot();

        // Pre-existing fields stay intact and the nested ipfs section is present.
        assert_eq!(snapshot.synced, true);
        assert_eq!(snapshot.api_port, None);
        assert_eq!(snapshot.ipfs.api_url, None);
        assert!(!snapshot.ipfs.api_reachable);
        assert_eq!(snapshot.ipfs.kubo_version, None);
        assert!(!snapshot.ipfs.daemon_managed);
        assert_eq!(snapshot.ipfs.last_upload_epoch_s, 0);
        assert!(!snapshot.ipfs.last_upload_success);
        assert_eq!(snapshot.ipfs.last_upload_cid, None);
        assert_eq!(snapshot.ipfs.last_error, None);
        assert!(!snapshot.ipfs.inference_rewards_enabled);
    }

    #[test]
    fn daemon_managed_flag_distinguishes_auto_started_daemons() {
        let stats = MinerStats::new(false);
        // External daemon: reachable, not auto-managed.
        stats.set_ipfs_version_probe(true, Some("0.41.0".to_string()), false);
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert!(!snapshot.ipfs.daemon_managed);

        // Daemon started by the miner itself: same reachability, managed flag set.
        stats.set_ipfs_version_probe(true, Some("0.41.0".to_string()), true);
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.daemon_managed);

        // Auto-managed daemon that failed to come up stays observable.
        stats.set_ipfs_version_probe(false, None, true);
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.daemon_managed);
    }

    #[test]
    fn ipfs_probe_and_upload_outcomes_are_reflected_in_the_snapshot() {
        let stats = MinerStats::new(false);
        stats.set_ipfs_api_url("http://127.0.0.1:5001".to_string());
        stats.set_ipfs_version_probe(true, Some("0.41.0".to_string()), false);
        stats.set_ipfs_upload_outcome(true, Some("QmXp1ExampleCidv0String".to_string()), None);
        stats.set_inference_rewards_enabled(true);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.ipfs.api_url.as_deref(), Some("http://127.0.0.1:5001"));
        assert!(snapshot.ipfs.api_reachable);
        assert_eq!(snapshot.ipfs.kubo_version.as_deref(), Some("0.41.0"));
        assert!(!snapshot.ipfs.daemon_managed);
        assert!(snapshot.ipfs.last_upload_success);
        assert_eq!(snapshot.ipfs.last_upload_cid.as_deref(), Some("QmXp1ExampleCidv0String"));
        assert_eq!(snapshot.ipfs.last_error, None);
        assert!(snapshot.ipfs.inference_rewards_enabled);
        assert!(snapshot.ipfs.last_upload_epoch_s > 0);

        // Failure transition: reachability loss and a failed upload stay observable.
        stats.set_ipfs_version_probe(false, None, false);
        stats.set_ipfs_upload_outcome(false, None, Some("connection refused".to_string()));
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.api_reachable);
        assert!(!snapshot.ipfs.last_upload_success);
        assert_eq!(snapshot.ipfs.last_upload_cid, None);
        assert_eq!(snapshot.ipfs.last_error.as_deref(), Some("connection refused"));
        // The previous successful CID is gone once a later attempt fails.
        assert!(snapshot.ipfs.last_upload_epoch_s > 0);
    }

    #[test]
    fn serialized_stats_never_contain_uploaded_text() {
        let stats = MinerStats::new(false);
        stats.set_ipfs_api_url("http://127.0.0.1:5001".to_string());
        // Simulate what an upload records — never the text itself.
        stats.set_ipfs_upload_outcome(true, Some("QmSecretCid".to_string()), None);

        let json = serde_json::to_string(&stats.snapshot()).expect("snapshot serializes");
        assert!(json.contains("QmSecretCid"));
        assert!(!json.contains("super secret inference output"));
        assert!(!json.contains("prompt"));
    }
}

fn parse_f32_field(value: &str) -> Option<f32> {
    value
        .split_whitespace()
        .next()
        .and_then(|x| x.parse::<f32>().ok())
}

fn prefer_nvml_u32_or_nvidia_smi(nvml_value: Option<u32>, nvidia_smi_value: Option<u32>) -> Option<u32> {
    nvml_value.filter(|temp| *temp > 0).or_else(|| nvidia_smi_value.filter(|temp| *temp > 0))
}

fn prefer_nvml_f32_or_nvidia_smi(nvml_value: Option<f32>, nvidia_smi_value: Option<f32>) -> Option<f32> {
    nvml_value.filter(|value| *value > 0.0).or_else(|| nvidia_smi_value.filter(|value| *value > 0.0))
}

fn normalize_power_draw_w(nvml_power_mw: Option<f32>, nvidia_smi_power_w: Option<f32>) -> Option<f32> {
    let nvml_power_w = nvml_power_mw.map(|mw| mw / 1000.0);
    let nvidia_smi_power_w = nvidia_smi_power_w.filter(|value| *value > 0.0);
    nvml_power_w.filter(|value| *value > 0.0).or(nvidia_smi_power_w)
}

fn should_query_nvidia_smi(
    telemetry: &HashMap<u32, GpuTelemetry>,
    memory_temp_supported: &HashMap<u32, bool>,
    hiveos: bool,
) -> bool {
    telemetry.is_empty()
        || telemetry.iter().any(|(gpu_idx, entry)| {
            entry.temp_c.is_none()
                || entry.fan_percent.is_none()
                || entry.power_draw_w.is_none()
                || (!hiveos
                    && entry.memory_temp_c.is_none()
                    // Re-query only while support is unknown or expected: a recorded `false`
                    // (probed, no memory-temp sensor) must stop the per-tick nvidia-smi spawns.
                    && memory_temp_supported.get(gpu_idx).copied().unwrap_or(true))
        })
}

fn normalize_memory_temp_c(nvidia_smi_temp_c: Option<u32>, nvml_temp_c: Option<u32>) -> Option<u32> {
    prefer_nvml_u32_or_nvidia_smi(nvml_temp_c, nvidia_smi_temp_c)
}

fn read_hiveos_nvtool_memtemps() -> Option<HashMap<u32, u32>> {
    let output = Command::new("nvtool").arg("--memtemp").output().ok()?;
    if !output.status.success() {
        return None;
    }

    Some(parse_nvtool_memtemp_output(&String::from_utf8_lossy(&output.stdout)))
}

fn parse_nvtool_memtemp_output(output: &str) -> HashMap<u32, u32> {
    let mut memtemps = HashMap::new();
    let mut current_device: Option<u32> = None;

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(device_id) = trimmed.strip_prefix("DEVICE #") {
            if let Some(idx_str) = device_id.split(':').next() {
                if let Ok(idx) = idx_str.parse::<u32>() {
                    current_device = Some(idx);
                }
            }
            continue;
        }

        if let Some(idx) = current_device {
            if let Some(temp_text) = trimmed.split("MEM TEMPERATURE:").nth(1) {
                let temp_text = temp_text.split('C').next().unwrap_or_default().trim();
                if let Ok(temp) = temp_text.parse::<u32>() {
                    if temp > 0 {
                        memtemps.insert(idx, temp);
                    }
                }
                current_device = None;
            }
        }
    }

    memtemps
}

#[cfg(test)]
mod telemetry_tests {
    use super::{
        logical_device_number, logical_nvtool_device_number, merge_physical_to_logical, normalize_memory_temp_c,
        normalize_pci_bus_id, parse_nvidia_smi_device_map, parse_nvtool_memtemp_output,
        prefer_nvml_f32_or_nvidia_smi, prefer_nvml_u32_or_nvidia_smi, should_query_nvidia_smi,
    };

    #[test]
    fn logical_device_number_follows_cuda_pci_mapping() {
        let cuda_bus_ids =
            HashMap::from([(normalize_pci_bus_id("0000:02:00.0"), 0), (normalize_pci_bus_id("0000:01:00.0"), 1)]);

        assert_eq!(logical_device_number("00000000:01:00.0", 0, &cuda_bus_ids), Some(1));
        assert_eq!(logical_device_number("00000000:02:00.0", 1, &cuda_bus_ids), Some(0));
        assert_eq!(logical_device_number("00000000:03:00.0", 2, &cuda_bus_ids), None);
    }

    #[test]
    fn logical_device_number_falls_back_when_cuda_is_unavailable() {
        assert_eq!(logical_device_number("00000000:01:00.0", 2, &HashMap::new()), Some(2));
    }

    #[test]
    fn maps_nvidia_smi_physical_ordinals_by_pci_identity() {
        let cuda_bus_ids = HashMap::from([("0:02:00.0".to_string(), 0), ("0:01:00.0".to_string(), 1)]);
        let output = b"0, 00000000:01:00.0\n1, 00000000:02:00.0\n";

        assert_eq!(parse_nvidia_smi_device_map(output, &cuda_bus_ids), HashMap::from([(0, 1), (1, 0)]));
    }

    #[test]
    fn maps_hiveos_nvtool_ordinals_to_cuda_devices() {
        let physical_to_logical = HashMap::from([(0, 2), (1, 0)]);
        let cuda_bus_ids = HashMap::from([("0:01:00.0".to_string(), 2)]);

        assert_eq!(logical_nvtool_device_number(0, &physical_to_logical, &cuda_bus_ids), Some(2));
        assert_eq!(logical_nvtool_device_number(1, &physical_to_logical, &cuda_bus_ids), Some(0));
        assert_eq!(logical_nvtool_device_number(3, &physical_to_logical, &cuda_bus_ids), None);
        assert_eq!(logical_nvtool_device_number(3, &HashMap::new(), &HashMap::new()), Some(3));
    }

    #[test]
    fn completes_partial_physical_device_mapping_without_overwriting_nvml() {
        let mut mapping = HashMap::from([(0, 2)]);
        merge_physical_to_logical(&mut mapping, HashMap::from([(0, 7), (1, 0)]));

        assert_eq!(mapping, HashMap::from([(0, 2), (1, 0)]));
    }
    use std::collections::HashMap;

    #[test]
    fn prefers_nvml_memory_temp_when_available() {
        assert_eq!(normalize_memory_temp_c(Some(70), Some(55)), Some(55));
    }

    #[test]
    fn prefers_nvml_u32_values_over_nvidia_smi_when_available() {
        assert_eq!(prefer_nvml_u32_or_nvidia_smi(Some(55), Some(70)), Some(55));
    }

    #[test]
    fn falls_back_to_nvidia_smi_when_nvml_is_missing() {
        assert_eq!(prefer_nvml_u32_or_nvidia_smi(None, Some(70)), Some(70));
    }

    #[test]
    fn treats_zero_as_missing_for_nvml_u32_values() {
        assert_eq!(prefer_nvml_u32_or_nvidia_smi(Some(0), Some(70)), Some(70));
    }

    #[test]
    fn prefers_nvml_f32_values_over_nvidia_smi_when_available() {
        assert_eq!(prefer_nvml_f32_or_nvidia_smi(Some(320.0), Some(350.0)), Some(320.0));
    }

    #[test]
    fn falls_back_to_nvidia_smi_memory_temp_when_nvml_is_missing() {
        assert_eq!(normalize_memory_temp_c(Some(70), None), Some(70));
    }

    #[test]
    fn ignores_zero_values() {
        assert_eq!(normalize_memory_temp_c(Some(0), Some(0)), None);
    }

    #[test]
    fn parses_hiveos_nvtool_memtemps() {
        let output = r#"HiveOS Nvtool 1.8.6
DEVICE #0:
  MEM TEMPERATURE: 72 C
DEVICE #1:
  MEM TEMPERATURE: 0 C [Not Supported]"#;

        let memtemps = parse_nvtool_memtemp_output(output);
        assert_eq!(memtemps.get(&0), Some(&72));
        assert!(memtemps.get(&1).is_none());
    }

    #[test]
    fn skips_nvidia_smi_when_nvml_already_has_complete_telemetry() {
        let mut telemetry = HashMap::new();
        telemetry.insert(
            0,
            super::GpuTelemetry {
                temp_c: Some(70),
                memory_temp_c: Some(72),
                fan_percent: Some(80),
                power_draw_w: Some(250.0),
            },
        );

        assert!(!should_query_nvidia_smi(&telemetry, &HashMap::new(), false));
    }

    #[test]
    fn queries_nvidia_smi_when_any_nvml_value_is_missing() {
        let mut telemetry = HashMap::new();
        telemetry.insert(
            0,
            super::GpuTelemetry {
                temp_c: Some(70),
                memory_temp_c: Some(72),
                fan_percent: Some(80),
                power_draw_w: None,
            },
        );

        assert!(should_query_nvidia_smi(&telemetry, &HashMap::new(), false));
    }
    #[test]
    fn skips_nvidia_smi_when_memory_temp_is_known_unsupported() {
        let mut telemetry = HashMap::new();
        telemetry.insert(
            0,
            super::GpuTelemetry {
                temp_c: Some(70),
                memory_temp_c: None,
                fan_percent: Some(80),
                power_draw_w: Some(250.0),
            },
        );
        let mut supported = HashMap::new();
        supported.insert(0, false);

        assert!(!should_query_nvidia_smi(&telemetry, &supported, false));
    }
}

pub fn spawn_stats_server(stats: Arc<MinerStats>, bind_addr: String, port: u16) -> std::io::Result<thread::JoinHandle<()>> {
    let listener = TcpListener::bind((bind_addr.as_str(), port))?;
    Ok(thread::spawn(move || {
        let active_connections = Arc::new(AtomicUsize::new(0));
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let Some(permit) = StatsConnectionPermit::acquire(&active_connections) else {
                        continue;
                    };
                    let stats = Arc::clone(&stats);
                    let _ = thread::Builder::new().name("stats-handler".into()).spawn(move || {
                        let _permit = permit;
                        let _ = handle_connection(stream, &stats);
                    });
                }
                Err(_) => continue,
            }
        }
    }))
}

fn handle_connection(mut stream: TcpStream, stats: &MinerStats) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(STATS_READ_TIMEOUT_SECS)))?;
    stream.set_write_timeout(Some(Duration::from_secs(STATS_WRITE_TIMEOUT_SECS)))?;

    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = Vec::with_capacity(256);
    let read_res = reader
        .by_ref()
        .take((MAX_REQUEST_LINE_BYTES + 1) as u64)
        .read_until(b'\n', &mut request_line);
    let bytes_read = match read_res {
        Ok(n) => n,
        Err(err)
            if err.kind() == std::io::ErrorKind::WouldBlock || err.kind() == std::io::ErrorKind::TimedOut =>
        {
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    if bytes_read == 0 {
        return Ok(());
    }
    if request_line.len() > MAX_REQUEST_LINE_BYTES {
        return write_json_response(
            &mut stream,
            "414 URI Too Long",
            b"{\"error\":\"request line too long\"}".to_vec(),
        );
    }

    let request_line = String::from_utf8_lossy(&request_line);
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");

    let (status, body) = if path == "/stats" || path == "/v1/miner/stats" {
        (
            "200 OK",
            serde_json::to_vec(&stats.snapshot()).unwrap_or_else(|_| b"{\"error\":\"failed to serialize stats\"}".to_vec()),
        )
    } else {
        ("404 Not Found", b"{\"error\":\"not found\"}".to_vec())
    };

    write_json_response(&mut stream, status, body)
}

fn write_json_response(stream: &mut TcpStream, status: &str, body: Vec<u8>) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn now_epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn parse_device_number(id: &str) -> Option<u32> {
    id.strip_prefix('#')
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u32>().ok())
}

/// CUDA logical ordinal per PCI bus id. NVML, nvidia-smi and nvtool all number GPUs by bus
/// order, while CUDA's default order is FASTEST_FIRST — on a mixed rig the two disagree and
/// telemetry lands on the wrong card.
fn cuda_device_bus_ids() -> &'static HashMap<String, u32> {
    static BUS_IDS: OnceLock<HashMap<String, u32>> = OnceLock::new();
    BUS_IDS.get_or_init(|| {
        let mut bus_ids = HashMap::new();
        if result::init().is_err() {
            return bus_ids;
        }

        let count = result::device::get_count().unwrap_or(0);
        for ordinal in 0..count {
            let Ok(device) = result::device::get(ordinal) else {
                continue;
            };
            let mut buffer = [0i8; 32];
            if unsafe { sys::cuDeviceGetPCIBusId(buffer.as_mut_ptr(), buffer.len() as i32, device).result() }.is_err() {
                continue;
            }
            let Ok(bus_id) = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str() else {
                continue;
            };
            bus_ids.insert(normalize_pci_bus_id(bus_id), ordinal as u32);
        }
        bus_ids
    })
}

fn logical_device_number(pci_bus_id: &str, fallback_idx: u32, cuda_bus_ids: &HashMap<String, u32>) -> Option<u32> {
    if cuda_bus_ids.is_empty() {
        Some(fallback_idx)
    } else {
        cuda_bus_ids.get(&normalize_pci_bus_id(pci_bus_id)).copied()
    }
}

fn nvidia_smi_device_map(cuda_bus_ids: &HashMap<String, u32>) -> HashMap<u32, u32> {
    let output =
        Command::new("nvidia-smi").args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"]).output();
    let Ok(output) = output else {
        return HashMap::new();
    };
    if !output.status.success() {
        return HashMap::new();
    }

    parse_nvidia_smi_device_map(&output.stdout, cuda_bus_ids)
}

fn parse_nvidia_smi_device_map(output: &[u8], cuda_bus_ids: &HashMap<String, u32>) -> HashMap<u32, u32> {
    output
        .split(|byte| *byte == b'\n')
        .filter_map(|line| {
            let line = String::from_utf8_lossy(line);
            let mut fields = line.split(',').map(str::trim);
            let physical_idx = fields.next()?.parse::<u32>().ok()?;
            let bus_id = normalize_pci_bus_id(fields.next()?);
            cuda_bus_ids.get(&bus_id).copied().map(|logical_idx| (physical_idx, logical_idx))
        })
        .collect()
}

fn merge_physical_to_logical(existing: &mut HashMap<u32, u32>, supplemental: HashMap<u32, u32>) {
    for (physical_idx, logical_idx) in supplemental {
        existing.entry(physical_idx).or_insert(logical_idx);
    }
}

fn logical_nvtool_device_number(
    physical_idx: u32,
    physical_to_logical: &HashMap<u32, u32>,
    cuda_bus_ids: &HashMap<String, u32>,
) -> Option<u32> {
    physical_to_logical.get(&physical_idx).copied().or_else(|| cuda_bus_ids.is_empty().then_some(physical_idx))
}

fn normalize_pci_bus_id(pci_bus_id: &str) -> String {
    let pci_bus_id = pci_bus_id.trim().to_ascii_lowercase();
    let Some((domain, device)) = pci_bus_id.split_once(':') else {
        return pci_bus_id;
    };
    let domain = domain.trim_start_matches('0');
    format!("{}:{device}", if domain.is_empty() { "0" } else { domain })
}

fn parse_u32_field(value: &str) -> Option<u32> {
    let filtered = value
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if filtered.is_empty() {
        None
    } else {
        filtered.parse::<u32>().ok()
    }
}
#[cfg(test)]
mod connection_tests {
    use super::{StatsConnectionPermit, MAX_STATS_CONNECTIONS};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn stats_connection_limit_is_bounded_and_released() {
        let active = Arc::new(AtomicUsize::new(0));
        let permits = (0..MAX_STATS_CONNECTIONS)
            .map(|_| StatsConnectionPermit::acquire(&active).expect("connection slot"))
            .collect::<Vec<_>>();

        assert!(StatsConnectionPermit::acquire(&active).is_none());
        drop(permits);
        assert!(StatsConnectionPermit::acquire(&active).is_some());
    }
}
