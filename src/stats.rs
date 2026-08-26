use cudarc::driver::{result, sys};
use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, structs::device::FieldId, Nvml};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
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

/// Schema version of the stats snapshot. v2 adds `phase`, `fatal_error`, `stats_schema_version`,
/// nested `keryxd`/`escrow` state and per-device readiness fields while keeping every v1 key.
pub const STATS_SCHEMA_VERSION: u32 = 2;

/// Process-wide registration of the miner's stats handle, installed once by main at startup.
/// The stats server and any readiness reporters read through this.
static READINESS_STATS: OnceLock<Arc<MinerStats>> = OnceLock::new();

pub fn install_readiness_stats(stats: Arc<MinerStats>) {
    let _ = READINESS_STATS.set(stats);
}

pub fn readiness_stats() -> Option<Arc<MinerStats>> {
    READINESS_STATS.get().cloned()
}

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
    gpu_identity: Mutex<HashMap<u32, GpuIdentity>>,
    hiveos: AtomicBool,
    // Readiness state (schema v2). All optional/mutable — old clients see the same v1 fields.
    devices_configured: Mutex<HashMap<u32, ConfiguredDevice>>,
    fatal_error: Mutex<Option<String>>,
    keryxd_connected: AtomicBool,
    keryxd_version: Mutex<Option<String>>,
    keryxd_synced: AtomicBool,
    template_notifications: AtomicBool,
    escrow_key_loaded: AtomicBool,
    escrow_certificate_valid: AtomicBool,
    escrow_public_key: Mutex<Option<String>>,
}

#[derive(Default, Clone, Copy)]
struct GpuTelemetry {
    temp_c: Option<u32>,
    memory_temp_c: Option<u32>,
    fan_percent: Option<u32>,
    power_draw_w: Option<f32>,
}

/// Physical GPU identity for readiness rows, keyed by logical CUDA device id.
#[derive(Default, Clone)]
struct GpuIdentity {
    uuid: Option<String>,
    name: Option<String>,
}

/// Per-CUDA-device readiness state tracked by the miner. `tier`/`model_id`/`model` are set by
/// `configure_device`; the sticky flags are latched by their setters and never reset by a `false`
/// (a `false` only means "no new evidence this cycle", not "no longer ready"). Reconfiguring a
/// device to a **different** model resets the whole record — the old model's progress and any
/// latched error must not leak into the new assignment.
#[derive(Default, Clone)]
struct ConfiguredDevice {
    tier: Option<String>,
    model_id: Option<String>,
    model: Option<String>,
    /// Cumulative bytes of the model file downloaded so far.
    downloaded: u64,
    /// Total bytes when known; `None` while unknown (server did not send a size). Never
    /// downgraded by a later `None` after a known total was recorded.
    download_total: Option<u64>,
    download_complete: bool,
    integrity_verified: bool,
    model_loaded: bool,
    inference_verified: bool,
    pom_ready: bool,
    error: Option<String>,
}

/// Overall startup/mine readiness, derived from per-device and node state. Serialized snake_case.
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessPhase {
    #[default]
    Starting,
    Downloading,
    Verifying,
    Loading,
    Preparing,
    Partial,
    Degraded,
    Mining,
    Fatal,
}

#[derive(Default, Clone, Serialize)]
pub struct KeryxdReadiness {
    pub connected: bool,
    pub version: Option<String>,
    pub synced: bool,
}

#[derive(Default, Clone, Serialize)]
pub struct EscrowReadiness {
    pub key_loaded: bool,
    pub certificate_valid: bool,
    pub public_key: Option<String>,
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
    // Readiness enrichment (schema v2). `uuid`/`name` are the physical GPU identity from
    // NVML/nvidia-smi telemetry when available; `tier`/`model` come from configure_device.
    pub uuid: Option<String>,
    pub name: Option<String>,
    pub tier: Option<String>,
    pub model_id: Option<String>,
    pub model: Option<String>,
    pub phase: ReadinessPhase,
    pub download_percent: Option<u32>,
    pub download_complete: bool,
    pub integrity_verified: bool,
    pub model_loaded: bool,
    pub inference_verified: bool,
    pub pom_ready: bool,
    pub mining: bool,
    pub error: Option<String>,
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
    // Readiness (schema v2).
    pub stats_schema_version: u32,
    pub phase: ReadinessPhase,
    pub fatal_error: Option<String>,
    pub keryxd: KeryxdReadiness,
    pub escrow: EscrowReadiness,
    pub template_notifications: bool,
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
            gpu_identity: Mutex::new(HashMap::new()),
            hiveos: AtomicBool::new(hiveos),
            devices_configured: Mutex::new(HashMap::new()),
            fatal_error: Mutex::new(None),
            keryxd_connected: AtomicBool::new(false),
            keryxd_version: Mutex::new(None),
            keryxd_synced: AtomicBool::new(false),
            template_notifications: AtomicBool::new(false),
            escrow_key_loaded: AtomicBool::new(false),
            escrow_certificate_valid: AtomicBool::new(false),
            escrow_public_key: Mutex::new(None),
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

    /// Record that CUDA device `id` is configured to mine tier `tier` with model `model_id`
    /// (human label `model`). Reconfiguration to the **same** model preserves the progress
    /// already latched for it (download, verification flags, pom readiness); reconfiguration
    /// to a different model restarts the device's pipeline — the old model's download/flags
    /// and any latched error must not leak into the new assignment.
    pub fn configure_device(&self, id: u32, tier: String, model_id: String, model: String) {
        if let Ok(mut devices) = self.devices_configured.lock() {
            let device = devices.entry(id).or_default();
            if device.model_id.as_deref() != Some(model_id.as_str()) {
                *device = ConfiguredDevice {
                    tier: Some(tier),
                    model_id: Some(model_id),
                    model: Some(model),
                    ..Default::default()
                };
                return;
            }
            device.tier = Some(tier);
            device.model_id = Some(model_id);
            device.model = Some(model);
        }
    }

    /// Record cumulative download progress for a model. `downloaded` is cumulative bytes;
    /// `total` is the file size when known. A known total is never lost: a later `None` (or a
    /// degenerate zero size) keeps the previously recorded size, and the download is latched
    /// complete when the cumulative bytes reach the total.
    pub fn set_device_download(&self, model_id: &str, downloaded: u64, total: Option<u64>) {
        if let Ok(mut devices) = self.devices_configured.lock() {
            // Every device sharing this model gets the same progress — a mixed rig can serve one
            // model on several GPUs and each row must advance in lockstep, not just the first.
            for device in devices.values_mut().filter(|d| d.model_id.as_deref() == Some(model_id)) {
                if let Some(total) = total.filter(|total| *total > 0) {
                    device.download_total = Some(total);
                }
                device.downloaded = downloaded;
                if let Some(total) = device.download_total {
                    if downloaded >= total {
                        device.download_complete = true;
                    }
                }
            }
        }
    }

    /// Latch model integrity verification on every device sharing `model_id` (true-only; a
    /// `false` is a no-op).
    pub fn set_model_integrity(&self, model_id: &str, verified: bool) {
        if !verified {
            return;
        }
        if let Ok(mut devices) = self.devices_configured.lock() {
            for device in devices.values_mut().filter(|d| d.model_id.as_deref() == Some(model_id)) {
                device.integrity_verified = true;
            }
        }
    }

    /// Latch model load state on every device sharing `model_id` (true-only; a `false` is a no-op).
    pub fn set_model_loaded(&self, model_id: &str, loaded: bool) {
        if !loaded {
            return;
        }
        if let Ok(mut devices) = self.devices_configured.lock() {
            for device in devices.values_mut().filter(|d| d.model_id.as_deref() == Some(model_id)) {
                device.model_loaded = true;
            }
        }
    }

    /// Latch inference verification on every device sharing `model_id` (true-only; a `false` is
    /// a no-op).
    pub fn set_model_inference_verified(&self, model_id: &str, verified: bool) {
        if !verified {
            return;
        }
        if let Ok(mut devices) = self.devices_configured.lock() {
            for device in devices.values_mut().filter(|d| d.model_id.as_deref() == Some(model_id)) {
                device.inference_verified = true;
            }
        }
    }

    /// Latch that the PoM kernel is installed and ready for a device (true-only).
    pub fn set_device_pom_ready(&self, id: u32, ready: bool) {
        if !ready {
            return;
        }
        if let Ok(mut devices) = self.devices_configured.lock() {
            devices.entry(id).or_default().pom_ready = true;
        }
    }

    /// Set/clear a device-level error. `None` clears the failure and lets the device resume
    /// its truthful phase.
    pub fn set_device_error(&self, id: u32, error: Option<String>) {
        if let Ok(mut devices) = self.devices_configured.lock() {
            devices.entry(id).or_default().error = error;
        }
    }

    /// Set a fatal, miner-wide error. `None` clears it.
    pub fn set_fatal_error(&self, error: Option<String>) {
        if let Ok(mut slot) = self.fatal_error.lock() {
            *slot = error;
        }
    }

    pub fn set_escrow_readiness(&self, key_loaded: bool, certificate_valid: bool, public_key: Option<String>) {
        self.escrow_key_loaded.store(key_loaded, Ordering::Release);
        self.escrow_certificate_valid.store(certificate_valid, Ordering::Release);
        if let Ok(mut slot) = self.escrow_public_key.lock() {
            *slot = public_key;
        }
    }

    pub fn set_keryxd_connected(&self, connected: bool) {
        self.keryxd_connected.store(connected, Ordering::Release);
    }

    pub fn set_keryxd_version(&self, version: Option<String>) {
        if let Ok(mut slot) = self.keryxd_version.lock() {
            *slot = version;
        }
    }

    pub fn set_keryxd_synced(&self, synced: bool) {
        self.keryxd_synced.store(synced, Ordering::Release);
    }

    pub fn set_template_notifications(&self, enabled: bool) {
        self.template_notifications.store(enabled, Ordering::Release);
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
                    // Physical GPU identity for the readiness rows; keyed by logical device id.
                    let uuid = device.uuid().ok();
                    let name = device.name().ok();
                    if uuid.is_some() || name.is_some() {
                        if let Ok(mut identity) = self.gpu_identity.lock() {
                            identity.insert(
                                logical_idx,
                                GpuIdentity {
                                    uuid: uuid.filter(|u| !u.is_empty()),
                                    name: name.filter(|n| !n.is_empty()),
                                },
                            );
                        }
                    }

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
        let should_query_nvidia_smi =
            should_query_nvidia_smi(&fresh, &memory_temp_supported, self.hiveos.load(Ordering::Acquire));
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
                        telemetry.memory_temp_c =
                            normalize_memory_temp_c(nvidia_smi_memory_temp_c, telemetry.memory_temp_c);
                        telemetry.fan_percent = prefer_nvml_u32_or_nvidia_smi(telemetry.fan_percent, fan_percent);
                        telemetry.power_draw_w = prefer_nvml_f32_or_nvidia_smi(telemetry.power_draw_w, power_draw_w);
                    } else {
                        fresh.insert(
                            gpu_idx,
                            GpuTelemetry {
                                temp_c,
                                memory_temp_c: normalize_memory_temp_c(nvidia_smi_memory_temp_c, None),
                                fan_percent,
                                power_draw_w,
                            },
                        );
                    }

                    if !self.hiveos.load(Ordering::Acquire) {
                        memory_temp_supported
                            .insert(gpu_idx, fresh.get(&gpu_idx).and_then(|entry| entry.memory_temp_c).is_some());
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
        let uptime_s =
            self.started_at.lock().expect("start time mutex poisoned").map(|t| t.elapsed().as_secs()).unwrap_or(0);

        let telemetry = self.gpu_telemetry.lock().expect("gpu telemetry mutex poisoned").clone();
        let service_status = self.service_status.lock().ok().and_then(|s| s.clone());
        let mining_address = self.mining_address.lock().expect("mining address mutex poisoned").clone();

        let device_blocks_accepted =
            self.device_blocks_accepted.lock().expect("device block count mutex poisoned").clone();
        let device_blocks_rejected =
            self.device_blocks_rejected.lock().expect("device rejected block count mutex poisoned").clone();
        let configured = self.devices_configured.lock().expect("configured devices mutex poisoned").clone();
        let identities = self.gpu_identity.lock().expect("gpu identity mutex poisoned").clone();
        let hashrates = self.device_hashrate_hs.lock().expect("device stats mutex poisoned");
        let mut devices = hashrates
            .iter()
            .map(|(id, rate)| {
                let gpu_idx = parse_device_number(id);
                let telem = gpu_idx.and_then(|idx| telemetry.get(&idx).copied());
                let identity = gpu_idx.and_then(|idx| identities.get(&idx).cloned());
                let device = gpu_idx.and_then(|idx| configured.get(&idx));
                let mining = *rate > 0 && device.map(|d| d.error.is_none()).unwrap_or(true);
                DeviceRate {
                    id: id.clone(),
                    hashrate_hs: *rate,
                    blocks_accepted: device_blocks_accepted.get(id).copied().unwrap_or(0),
                    blocks_rejected: device_blocks_rejected.get(id).copied().unwrap_or(0),
                    temp_c: telem.and_then(|t| t.temp_c),
                    memory_temp_c: telem.and_then(|t| t.memory_temp_c),
                    fan_percent: telem.and_then(|t| t.fan_percent),
                    power_draw_w: telem.and_then(|t| t.power_draw_w),
                    uuid: identity.as_ref().and_then(|i| i.uuid.clone()),
                    name: identity.as_ref().and_then(|i| i.name.clone()),
                    tier: device.and_then(|d| d.tier.clone()),
                    model_id: device.and_then(|d| d.model_id.clone()),
                    model: device.and_then(|d| d.model.clone()),
                    phase: device_phase(*rate, device),
                    download_percent: device.and_then(|d| download_percent(d)),
                    download_complete: device.map(|d| d.download_complete).unwrap_or(false),
                    integrity_verified: device.map(|d| d.integrity_verified).unwrap_or(false),
                    model_loaded: device.map(|d| d.model_loaded).unwrap_or(false),
                    inference_verified: device.map(|d| d.inference_verified).unwrap_or(false),
                    pom_ready: device.map(|d| d.pom_ready).unwrap_or(false),
                    mining,
                    error: device.and_then(|d| d.error.clone()),
                }
            })
            .collect::<Vec<_>>();
        // Every configured CUDA device gets a row even before its first hashrate tick (it may
        // still be downloading/preparing); the numeric id is the canonical `GPU #N` label.
        let mut seen = hashrates.keys().filter_map(|id| parse_device_number(id)).collect::<HashSet<_>>();
        for (gpu_idx, device) in configured.iter() {
            if !seen.insert(*gpu_idx) {
                continue;
            }
            let telem = telemetry.get(gpu_idx).copied();
            let identity = identities.get(gpu_idx).cloned();
            devices.push(DeviceRate {
                id: format!("#{}", gpu_idx),
                hashrate_hs: 0,
                blocks_accepted: 0,
                blocks_rejected: 0,
                temp_c: telem.and_then(|t| t.temp_c),
                memory_temp_c: telem.and_then(|t| t.memory_temp_c),
                fan_percent: telem.and_then(|t| t.fan_percent),
                power_draw_w: telem.and_then(|t| t.power_draw_w),
                uuid: identity.as_ref().and_then(|i| i.uuid.clone()),
                name: identity.as_ref().and_then(|i| i.name.clone()),
                tier: device.tier.clone(),
                model_id: device.model_id.clone(),
                model: device.model.clone(),
                phase: device_phase(0, Some(device)),
                download_percent: download_percent(device),
                download_complete: device.download_complete,
                integrity_verified: device.integrity_verified,
                model_loaded: device.model_loaded,
                inference_verified: device.inference_verified,
                pom_ready: device.pom_ready,
                mining: false,
                error: device.error.clone(),
            });
        }
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

        let fatal_error = self.fatal_error.lock().ok().and_then(|f| f.clone());
        let phase = overall_phase(&devices, &configured, fatal_error.as_deref());
        let escrow_public_key = self.escrow_public_key.lock().ok().and_then(|k| k.clone());
        let keryxd_version = self.keryxd_version.lock().ok().and_then(|v| v.clone());

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
            stats_schema_version: STATS_SCHEMA_VERSION,
            phase,
            fatal_error,
            keryxd: KeryxdReadiness {
                connected: self.keryxd_connected.load(Ordering::Acquire),
                version: keryxd_version,
                synced: self.keryxd_synced.load(Ordering::Acquire),
            },
            escrow: EscrowReadiness {
                key_loaded: self.escrow_key_loaded.load(Ordering::Acquire),
                certificate_valid: self.escrow_certificate_valid.load(Ordering::Acquire),
                public_key: escrow_public_key,
            },
            template_notifications: self.template_notifications.load(Ordering::Acquire),
        }
    }
}

/// Download progress as a percentage, bounded to 100. `None` when the total size is unknown
/// (or zero), since a percentage would be meaningless.
fn download_percent(device: &ConfiguredDevice) -> Option<u32> {
    let total = device.download_total?;
    if total == 0 {
        return None;
    }
    let percent = (u128::from(device.downloaded.min(total)) * 100) / u128::from(total);
    Some(percent.min(100) as u32)
}

/// Per-device readiness phase: failed when a device error is set (overrides a stale hashrate),
/// mining when the hashrate is non-zero (ground truth — applies even to v1 rows that were never
/// configured), otherwise the furthest truthful startup step (downloading → verifying → loading →
/// preparing). Rows without any readiness configuration report `Starting`.
fn device_phase(hashrate_hs: u64, device: Option<&ConfiguredDevice>) -> ReadinessPhase {
    match device {
        Some(device) if device.error.is_some() => ReadinessPhase::Fatal,
        _ => {
            if hashrate_hs > 0 {
                return ReadinessPhase::Mining;
            }
            match device {
                None => ReadinessPhase::Starting,
                Some(device) => {
                    if !device.download_complete {
                        return ReadinessPhase::Downloading;
                    }
                    if !device.integrity_verified {
                        return ReadinessPhase::Verifying;
                    }
                    if !device.model_loaded {
                        return ReadinessPhase::Loading;
                    }
                    ReadinessPhase::Preparing
                }
            }
        }
    }
}

/// Derive the overall readiness phase from per-device state.
///
/// Order of precedence (checked before the truthful startup progression so a stuck device can
/// never masquerade as healthy):
/// - `Fatal` when a miner-wide fatal error is set;
/// - `Degraded` when at least one tracked device failed and at least one other still mines.
///   "Tracked" is any device the stats hold a `ConfiguredDevice` entry for — even one whose
///   `configure_device` never ran (a device that failed before a tier was assigned must still
///   degrade the aggregate, not silently vanish from it);
/// - `Partial` when at least one device mines while another is still preparing;
/// - `Mining` when every configured CUDA device mines (each row non-fatal with a positive rate);
/// - otherwise the furthest truthful startup phase across devices (download → verify → load →
///   prepare), which is also `Mining` for the empty set (no GPUs configured).
fn overall_phase(
    devices: &[DeviceRate],
    configured: &HashMap<u32, ConfiguredDevice>,
    fatal_error: Option<&str>,
) -> ReadinessPhase {
    if fatal_error.is_some() {
        return ReadinessPhase::Fatal;
    }
    if devices.is_empty() {
        return ReadinessPhase::Mining;
    }

    let mut failed = 0usize;
    let mut mining = 0usize;
    let mut preparing = 0usize;
    let mut furthest = ReadinessPhase::Starting;
    for device in devices {
        let gpu_idx = parse_device_number(&device.id);
        let device_entry = gpu_idx.and_then(|idx| configured.get(&idx));
        // A device must hold a tier to drive the startup progression; any tracked entry at all
        // still counts as a failed device when it has an error.
        let is_configured = device_entry.map(|d| d.tier.is_some()).unwrap_or(false);
        if device.phase == ReadinessPhase::Fatal || device.error.is_some() {
            if device_entry.is_some() {
                failed += 1;
        }
            continue;
        }
        if device.mining {
            mining += 1;
        } else if is_configured {
            preparing += 1;
        }
        if is_configured {
            furthest = furthest.max(device.phase);
        }
    }

    if failed > 0 && mining > 0 {
        return ReadinessPhase::Degraded;
    }
    if mining > 0 && preparing > 0 {
        return ReadinessPhase::Partial;
    }
    if mining > 0 && failed == 0 && preparing == 0 {
        return ReadinessPhase::Mining;
    }
    if furthest != ReadinessPhase::Starting {
        return furthest;
    }
    // No configured device is on a truthful startup path (all idle or failed) and none mines:
    // stay neutral. Only the vacuous case — no GPUs at all — reports Mining.
    if configured.is_empty() {
        ReadinessPhase::Mining
    } else {
        ReadinessPhase::Starting
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
}

#[cfg(test)]
mod readiness_tests {
    use super::*;

    fn hashrates(per_device: &[(&str, u64)]) -> HashMap<String, u64> {
        per_device.iter().map(|(id, rate)| (id.to_string(), *rate)).collect()
    }

    /// Ready device row: configured, model downloaded/verified/loaded, inference verified, PoM ready.
    fn fully_ready(stats: &MinerStats, id: u32) {
        let model_id = format!("model-a{id}");
        let model = format!("Keryx 7B-{id}");
        stats.configure_device(id, "tier-1".into(), model_id.clone(), model);
        stats.set_device_download(&model_id, 100, Some(100));
        stats.set_model_integrity(&model_id, true);
        stats.set_model_loaded(&model_id, true);
        stats.set_model_inference_verified(&model_id, true);
        stats.set_device_pom_ready(id, true);
    }

    #[test]
    fn mixed_tier_phase_serializes_enriched_rows_with_old_fields() {
        let stats = MinerStats::new(false);
        fully_ready(&stats, 0);
        stats.configure_device(1, "tier-2".into(), "model-2".into(), "Keryx 14B".into());
        stats.set_device_download("model-2", 25, Some(100));

        let per_device_hashrates = hashrates(&[("#0", 4_000_000_000u64)]);
        stats.set_hashrates(4_000_000_000, &per_device_hashrates);

        let snapshot = stats.snapshot();

        // v2 envelope: schema version and overall mixed phase.
        assert_eq!(snapshot.stats_schema_version, 2);
        assert_eq!(snapshot.phase, ReadinessPhase::Partial);

        assert_eq!(snapshot.devices.len(), 2);
        let gpu0 = snapshot.devices.iter().find(|d| d.id == "#0").unwrap();
        let gpu1 = snapshot.devices.iter().find(|d| d.id == "#1").unwrap();

        // GPU0: ready + actively mining.
        assert_eq!(gpu0.tier.as_deref(), Some("tier-1"));
        assert_eq!(gpu0.model_id.as_deref(), Some("model-a0"));
        assert_eq!(gpu0.model.as_deref(), Some("Keryx 7B-0"));
        assert_eq!(gpu0.phase, ReadinessPhase::Mining);
        assert_eq!(gpu0.hashrate_hs, 4_000_000_000);
        assert!(gpu0.mining);
        assert_eq!(gpu0.download_percent, Some(100));
        assert!(gpu0.download_complete);
        assert!(gpu0.integrity_verified);
        assert!(gpu0.model_loaded);
        assert!(gpu0.inference_verified);
        assert!(gpu0.pom_ready);
        assert!(gpu0.error.is_none());

        // GPU1: still downloading, not mining.
        assert_eq!(gpu1.tier.as_deref(), Some("tier-2"));
        assert_eq!(gpu1.model_id.as_deref(), Some("model-2"));
        assert_eq!(gpu1.phase, ReadinessPhase::Downloading);
        assert_eq!(gpu1.hashrate_hs, 0);
        assert!(!gpu1.mining);
        assert_eq!(gpu1.download_percent, Some(25));
        assert!(!gpu1.download_complete);
        assert!(!gpu1.integrity_verified);
        assert!(!gpu1.model_loaded);
        assert!(!gpu1.inference_verified);
        assert!(!gpu1.pom_ready);

        // Old (v1) fields retained and serialized.
        let json = serde_json::to_value(&snapshot).unwrap();
        let device_json = &json["devices"][0];
        assert_eq!(device_json["hashrate_hs"], 4_000_000_000u64);
        assert!(device_json.get("blocks_accepted").is_some());
        assert!(device_json.get("temp_c").is_some());
        assert!(device_json.get("memory_temp_c").is_some());
        assert!(device_json.get("fan_percent").is_some());
        assert!(device_json.get("power_draw_w").is_some());
        assert!(json.get("started_epoch_s").is_some());
        assert!(json.get("total_hashrate_hs").is_some());
        assert_eq!(json["phase"], "partial");
    }

    #[test]
    fn fully_ready_devices_mine() {
        let stats = MinerStats::new(false);
        fully_ready(&stats, 0);
        fully_ready(&stats, 1);
        let rates = hashrates(&[("#0", 2_000_000_000u64), ("#1", 3_000_000_000u64)]);
        stats.set_hashrates(5_000_000_000, &rates);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.phase, ReadinessPhase::Mining);
        assert!(snapshot.devices.iter().all(|d| d.mining && d.phase == ReadinessPhase::Mining));
    }

    #[test]
    fn degraded_when_one_device_fails_and_another_mines() {
        let stats = MinerStats::new(false);
        fully_ready(&stats, 0);
        fully_ready(&stats, 1);
        let rates = hashrates(&[("#0", 2_000_000_000u64), ("#1", 3_000_000_000u64)]);
        stats.set_hashrates(5_000_000_000, &rates);

        stats.set_device_error(1, Some("cuda error 999: device lost".into()));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.phase, ReadinessPhase::Degraded);
        let failed = snapshot.devices.iter().find(|d| d.id == "#1").unwrap();
        assert_eq!(failed.phase, ReadinessPhase::Fatal);
        assert_eq!(failed.error.as_deref(), Some("cuda error 999: device lost"));
        assert!(!failed.mining);
        assert!(snapshot.devices.iter().find(|d| d.id == "#0").unwrap().mining);

        // Clearing the error restores the healthy phase.
        stats.set_device_error(1, None);
        assert_eq!(stats.snapshot().phase, ReadinessPhase::Mining);
    }

    #[test]
    fn tier_less_failed_device_still_degrades_a_mining_rig() {
        // A device that failed before configure_device ever assigned a tier must still count in
        // the aggregate: a failed card plus a mining card reads Degraded, not healthy Mining.
        let stats = MinerStats::new(false);
        fully_ready(&stats, 0);
        let rates = hashrates(&[("#0", 2_000_000_000u64)]);
        stats.set_hashrates(2_000_000_000, &rates);

        // Device 1 errors without ever being configured (no tier/model rows in readiness).
        stats.set_device_error(1, Some("cuda error 999: device lost".into()));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.phase, ReadinessPhase::Degraded);
        let failed = snapshot.devices.iter().find(|d| d.id == "#1").unwrap();
        assert_eq!(failed.phase, ReadinessPhase::Fatal);
        assert_eq!(failed.error.as_deref(), Some("cuda error 999: device lost"));
        assert!(failed.tier.is_none());
        assert!(!failed.mining);
        assert!(snapshot.devices.iter().find(|d| d.id == "#0").unwrap().mining);

        // Clearing the error restores the healthy phase.
        stats.set_device_error(1, None);
        assert_eq!(stats.snapshot().phase, ReadinessPhase::Mining);
    }

    #[test]
    fn all_tier_less_devices_failed_without_mining_stays_neutral() {
        // Deliberate all-failed/no-mining behavior is preserved: with no device mining, the
        // aggregate must not invent Degraded (or Mining) — it stays on the neutral Starting path.
        let stats = MinerStats::new(false);
        stats.set_device_error(0, Some("no mining tier assigned".into()));
        stats.set_device_error(1, Some("cuda error 999: device lost".into()));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.phase, ReadinessPhase::Starting);
        assert_ne!(snapshot.phase, ReadinessPhase::Degraded);
        assert_ne!(snapshot.phase, ReadinessPhase::Mining);
        assert_ne!(snapshot.phase, ReadinessPhase::Fatal);
        assert!(snapshot.devices.iter().all(|d| d.error.is_some() && !d.mining));
    }

    #[test]
    fn fatal_error_overrides_every_phase() {
        let stats = MinerStats::new(false);
        fully_ready(&stats, 0);
        let rates = hashrates(&[("#0", 2_000_000_000u64)]);
        stats.set_hashrates(2_000_000_000, &rates);

        stats.set_fatal_error(Some("keryxd node unreachable".into()));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.phase, ReadinessPhase::Fatal);
        assert_eq!(snapshot.fatal_error.as_deref(), Some("keryxd node unreachable"));
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["fatal_error"], "keryxd node unreachable");
        assert_eq!(json["phase"], "fatal");

        stats.set_fatal_error(None);
        assert_eq!(stats.snapshot().phase, ReadinessPhase::Mining);
    }

    #[test]
    fn keryxd_and_escrow_nested_state_serialize() {
        let stats = MinerStats::new(false);
        stats.set_keryxd_connected(true);
        stats.set_keryxd_version(Some("0.9.1".into()));
        stats.set_keryxd_synced(true);
        stats.set_template_notifications(true);
        stats.set_escrow_readiness(true, true, Some("0x1a2b3c".into()));

        let snapshot = stats.snapshot();
        assert!(snapshot.keryxd.connected);
        assert_eq!(snapshot.keryxd.version.as_deref(), Some("0.9.1"));
        assert!(snapshot.keryxd.synced);
        assert!(snapshot.escrow.key_loaded);
        assert!(snapshot.escrow.certificate_valid);
        assert_eq!(snapshot.escrow.public_key.as_deref(), Some("0x1a2b3c"));

        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["keryxd"]["connected"], true);
        assert_eq!(json["keryxd"]["version"], "0.9.1");
        assert_eq!(json["keryxd"]["synced"], true);
        assert_eq!(json["escrow"]["key_loaded"], true);
        assert_eq!(json["escrow"]["certificate_valid"], true);
        assert_eq!(json["escrow"]["public_key"], "0x1a2b3c");
        assert_eq!(json["stats_schema_version"], 2);
        assert_eq!(json["template_notifications"], true);

        // Older clients still see their keys (v1 keys untouched, phase snake_case).
        assert_eq!(json["phase"], "mining");
        assert!(json.get("started_epoch_s").is_some());
        assert!(json.get("service_status").is_some());
        assert!(json.get("opoi_challenge_active").is_some());
    }

    #[test]
    fn configure_device_to_a_new_model_resets_download_flags_and_error() {
        let stats = MinerStats::new(false);
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "Keryx 7B".into());
        stats.set_device_download("model-1", 100, Some(100));
        stats.set_model_integrity("model-1", true);
        stats.set_model_loaded("model-1", true);
        stats.set_model_inference_verified("model-1", true);
        stats.set_device_pom_ready(0, true);
        stats.set_device_error(0, Some("cuda error 999: device lost".into()));
        let before = stats.snapshot();
        assert_eq!(before.devices[0].phase, ReadinessPhase::Fatal);
        assert_eq!(before.devices[0].error.as_deref(), Some("cuda error 999: device lost"));

        // Reconfigure to a different model: the old model's progress and the latched error must
        // not leak into the new assignment.
        stats.configure_device(0, "tier-2".into(), "model-2".into(), "Keryx 14B".into());
        let snapshot = stats.snapshot();
        let device = snapshot.devices.iter().find(|d| d.id == "#0").unwrap();
        assert_eq!(device.tier.as_deref(), Some("tier-2"));
        assert_eq!(device.model_id.as_deref(), Some("model-2"));
        assert_eq!(device.model.as_deref(), Some("Keryx 14B"));
        assert_eq!(device.download_percent, None);
        assert!(!device.download_complete);
        assert!(!device.integrity_verified);
        assert!(!device.model_loaded);
        assert!(!device.inference_verified);
        assert!(!device.pom_ready);
        assert!(device.error.is_none());
        // With the error cleared and no hashrate for the fresh assignment, the device reports
        // its truthful startup step instead of the old model's failure.
        assert_eq!(device.phase, ReadinessPhase::Downloading);
        assert_eq!(snapshot.phase, ReadinessPhase::Downloading);
    }

    #[test]
    fn configure_device_with_same_model_preserves_latched_progress() {
        let stats = MinerStats::new(false);
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "Keryx 7B".into());
        stats.set_device_download("model-1", 100, Some(100));
        stats.set_model_integrity("model-1", true);
        stats.set_model_loaded("model-1", true);
        stats.set_model_inference_verified("model-1", true);
        stats.set_device_pom_ready(0, true);

        // Reconfiguration to the same model (e.g. a tier-label refresh) keeps the pipeline.
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "Keryx 7B".into());
        let device = &stats.snapshot().devices[0];
        assert!(device.download_complete);
        assert!(device.integrity_verified);
        assert!(device.model_loaded);
        assert!(device.inference_verified);
        assert!(device.pom_ready);
        assert_eq!(device.phase, ReadinessPhase::Preparing);
    }

    #[test]
    fn model_setters_update_every_device_sharing_the_model() {
        // A mixed rig serves one model on several GPUs; progress latched for the model must
        // reach every configured device, not just the first row that matches.
        let stats = MinerStats::new(false);
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "Keryx 7B".into());
        stats.configure_device(1, "tier-1".into(), "model-1".into(), "Keryx 7B".into());

        stats.set_device_download("model-1", 100, Some(100));
        stats.set_model_integrity("model-1", true);
        stats.set_model_loaded("model-1", true);
        stats.set_model_inference_verified("model-1", true);
        stats.set_device_pom_ready(0, true);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.devices.len(), 2);
        for device in &snapshot.devices {
            assert_eq!(device.model_id.as_deref(), Some("model-1"));
            assert_eq!(device.download_percent, Some(100), "device {} missed download", device.id);
            assert!(device.download_complete, "device {} missed download_complete", device.id);
            assert!(device.integrity_verified, "device {} missed integrity", device.id);
            assert!(device.model_loaded, "device {} missed model_loaded", device.id);
            assert!(device.inference_verified, "device {} missed inference_verified", device.id);
        }
    }

    #[test]
    fn download_progress_updates_and_bounds() {
        let stats = MinerStats::new(false);
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "Keryx 7B".into());
        stats.set_device_download("model-1", 50, Some(100));
        assert_eq!(stats.snapshot().devices[0].download_percent, Some(50));

        // Cumulative progress advances.
        stats.set_device_download("model-1", 100, Some(100));
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.devices[0].download_percent, Some(100));
        assert!(snapshot.devices[0].download_complete);

        // A later unknown total keeps the known one; progress beyond the total stays bounded.
        stats.set_device_download("model-1", 4096, None);
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.devices[0].download_percent, Some(100));
        assert!(snapshot.devices[0].download_complete);

        // Zero/unknown totals stay unknown (None, not 0%) and never flag completion on a
        // device that has not finished yet.
        stats.configure_device(1, "tier-1".into(), "model-zero".into(), "Keryx 7B".into());
        stats.set_device_download("model-zero", 0, Some(0));
        stats.set_device_download("model-zero", 10, None);
        let snapshot = stats.snapshot();
        let zero = snapshot.devices.iter().find(|d| d.id == "#1").unwrap();
        assert_eq!(zero.download_percent, None);
        assert!(!zero.download_complete);

        // Downloads for unknown models are ignored; the configured row keeps its identity.
        stats.set_device_download("model-ghost", 10, Some(50));
        assert_eq!(stats.snapshot().devices[0].id, "#0");
    }

    #[test]
    fn ready_device_advances_phase_through_startup_steps() {
        let stats = MinerStats::new(false);
        stats.configure_device(0, "tier-1".into(), "model-1".into(), "K-1".into());

        // Downloading → verifying once the file is complete.
        stats.set_device_download("model-1", 42, Some(100));
        assert_eq!(stats.snapshot().devices[0].phase, ReadinessPhase::Downloading);
        stats.set_device_download("model-1", 100, Some(100));
        assert_eq!(stats.snapshot().devices[0].phase, ReadinessPhase::Verifying);

        // Verifying → loading → preparing as flags latch.
        stats.set_model_integrity("model-1", true);
        assert_eq!(stats.snapshot().devices[0].phase, ReadinessPhase::Loading);
        stats.set_model_loaded("model-1", true);
        assert_eq!(stats.snapshot().devices[0].phase, ReadinessPhase::Preparing);
        stats.set_device_pom_ready(0, true);

        // A false after a true never downgrades (sticky latch).
        stats.set_model_integrity("model-1", false);
        stats.set_model_loaded("model-1", false);
        let snapshot = stats.snapshot();
        assert!(snapshot.devices[0].integrity_verified);
        assert!(snapshot.devices[0].model_loaded);
        assert_eq!(snapshot.devices[0].phase, ReadinessPhase::Preparing);

        // Mining once hashrate flows.
        let rates = hashrates(&[("#0", 1_000_000_000u64)]);
        stats.set_hashrates(1_000_000_000, &rates);
        assert_eq!(stats.snapshot().phase, ReadinessPhase::Mining);
    }

    #[test]
    fn hashrate_without_configuration_reports_rows_without_regressing() {
        // v1 behavior: devices exist purely from hashrate reports; no readiness config set.
        let stats = MinerStats::new(false);
        let rates = hashrates(&[("#0", 500u64)]);
        stats.set_hashrates(500, &rates);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.devices.len(), 1);
        let device = &snapshot.devices[0];
        assert_eq!(device.id, "#0");
        assert_eq!(device.hashrate_hs, 500);
        assert_eq!(device.phase, ReadinessPhase::Mining);
        assert!(device.mining);
        assert!(device.tier.is_none());
        assert!(device.uuid.is_none());
        assert_eq!(device.download_percent, None);
        assert_eq!(snapshot.phase, ReadinessPhase::Mining);
        assert_eq!(snapshot.stats_schema_version, 2);
    }
}

fn parse_f32_field(value: &str) -> Option<f32> {
    value.split_whitespace().next().and_then(|x| x.parse::<f32>().ok())
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
        normalize_pci_bus_id, parse_nvidia_smi_device_map, parse_nvtool_memtemp_output, prefer_nvml_f32_or_nvidia_smi,
        prefer_nvml_u32_or_nvidia_smi, should_query_nvidia_smi,
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

pub fn spawn_stats_server(
    stats: Arc<MinerStats>,
    bind_addr: String,
    port: u16,
) -> std::io::Result<thread::JoinHandle<()>> {
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
    let read_res = reader.by_ref().take((MAX_REQUEST_LINE_BYTES + 1) as u64).read_until(b'\n', &mut request_line);
    let bytes_read = match read_res {
        Ok(n) => n,
        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock || err.kind() == std::io::ErrorKind::TimedOut => {
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    if bytes_read == 0 {
        return Ok(());
    }
    if request_line.len() > MAX_REQUEST_LINE_BYTES {
        return write_json_response(&mut stream, "414 URI Too Long", b"{\"error\":\"request line too long\"}".to_vec());
    }

    let request_line = String::from_utf8_lossy(&request_line);
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");

    let (status, body) = if path == "/stats" || path == "/v1/miner/stats" {
        (
            "200 OK",
            serde_json::to_vec(&stats.snapshot())
                .unwrap_or_else(|_| b"{\"error\":\"failed to serialize stats\"}".to_vec()),
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
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn parse_device_number(id: &str) -> Option<u32> {
    id.strip_prefix('#').and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<u32>().ok())
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
    let filtered = value.chars().take_while(|c| c.is_ascii_digit()).collect::<String>();
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
