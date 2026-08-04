use clap::{ArgMatches, FromArgMatches};
use keryx_miner::declare_plugin;
use keryx_miner::{Plugin, Worker, WorkerSpec};
use log::{info, warn, LevelFilter};
use opencl3::device::{Device, CL_DEVICE_TYPE_ALL};
use opencl3::platform::{get_platforms, Platform};
use opencl3::types::cl_device_id;
use std::error::Error as StdError;

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

mod cli;
mod worker;

use crate::cli::{NonceGenEnum, OpenCLOpt};
use crate::worker::OpenCLGPUWorker;

fn cuda_ordinal_for_opencl_device(device_id: cl_device_id) -> Option<u32> {
    use cl3::ffi::cl_ext::CL_DEVICE_PCI_BUS_INFO_KHR;

    let pci = cl3::device::get_device_info(device_id, CL_DEVICE_PCI_BUS_INFO_KHR)
        .ok()
        .map(Vec::<u8>::from)
        .filter(|bytes| {
            bytes.len() == std::mem::size_of::<cl3::ffi::cl_ext::cl_device_pci_bus_info_khr>()
        })
        .map(|bytes| cl3::device::get_device_pci_bus_info_khr(&bytes))
        .and_then(|pci| {
            keryx_miner::cuda_ordinal_for_pci(
                pci.pci_domain,
                pci.pci_bus,
                pci.pci_device,
                pci.pci_function,
            )
        });

    pci.or_else(|| {
        Device::new(device_id)
            .pci_bus_id_nv()
            .ok()
            .and_then(keryx_miner::cuda_ordinal_for_pci_bus)
    })
}

// Sentinel: user did not pass --opencl-workload, so the worker resolves a
// capability-driven default ratio from the GPU arch (see worker::default_workload_scale).
const AUTO_WORKLOAD: f32 = 0.;

pub struct OpenCLPlugin {
    specs: Vec<OpenCLWorkerSpec>,
    _enabled: bool,
}

impl OpenCLPlugin {
    fn new() -> Result<Self, Error> {
        // try_init (not init): when this plugin and libkeryxcuda.so are both dlopen'd
        // into one binary (a mixed AMD+NVIDIA rig, or any NVIDIA box where both .so
        // resolve), whichever plugin's `_plugin_create` runs second would panic on a
        // second `init()`. The CUDA plugin already uses try_init; match it so the two
        // can coexist in a single "both worlds" binary regardless of load order.
        let _ = env_logger::builder().filter_level(LevelFilter::Info).parse_default_env().try_init();
        Ok(Self { specs: Vec::new(), _enabled: false })
    }
}

impl Plugin for OpenCLPlugin {
    fn name(&self) -> &'static str {
        "OpenCL Worker"
    }

    fn enabled(&self) -> bool {
        self._enabled
    }

    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        self.specs.iter().map(|spec| Box::new(*spec) as Box<dyn WorkerSpec>).collect::<Vec<Box<dyn WorkerSpec>>>()
    }

    //noinspection RsTypeCheck
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, keryx_miner::Error> {
        let opts: OpenCLOpt = OpenCLOpt::from_arg_matches(matches)?;

        self._enabled = opts.opencl_enable;
        let platforms = match get_platforms() {
            Ok(p) => p,
            Err(e) => {
                return Err(e.to_string().into());
            }
        };
        info!("OpenCL Found Platforms:");
        info!("=======================");
        for platform in &platforms {
            let vendor = &platform.vendor().unwrap_or_else(|_| "Unk".into());
            let name = &platform.name().unwrap_or_else(|_| "Unk".into());
            let num_devices = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default().len();
            info!("{}: {} ({} devices available)", vendor, name, num_devices);
        }
        let amd_platforms = (&platforms)
            .iter()
            .filter(|p| {
                p.vendor().unwrap_or_else(|_| "Unk".into()) == "Advanced Micro Devices, Inc."
                    && !p.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default().is_empty()
            })
            .collect::<Vec<&Platform>>();
        let _platform: &Platform = match opts.opencl_platform {
            Some(idx) => {
                self._enabled = true;
                &platforms[idx as usize]
            }
            None if !opts.opencl_amd_disable && !amd_platforms.is_empty() => {
                self._enabled = true;
                let amd = amd_platforms[0];
                let plat_name = amd.name().unwrap_or_else(|_| "Unk".into());
                if !plat_name.contains("ROCm") && !plat_name.contains("AMD Accelerated") {
                    warn!(
                        "AMD OpenCL platform detected but does not appear to be the ROCm runtime. \
                         RDNA 3+ GPUs (RX 7000/9000) may have issues. \
                         Install rocm-opencl-runtime for best support."
                    );
                }
                amd
            }
            None => &platforms[0],
        };
        if self._enabled {
            info!(
                "Chose to mine on {}: {}.",
                &_platform.vendor().unwrap_or_else(|_| "Unk".into()),
                &_platform.name().unwrap_or_else(|_| "Unk".into())
            );

            let device_ids = _platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap();
            let gpus: Vec<(usize, cl_device_id)> = match opts.opencl_device {
                Some(dev) => {
                    self._enabled = true;
                    dev.iter()
                        .map(|d| (*d as usize, device_ids[*d as usize]))
                        .collect()
                }
                None => device_ids.into_iter().enumerate().collect(),
            };

            let is_nvidia = _platform
                .vendor()
                .map(|vendor| vendor.contains("NVIDIA"))
                .unwrap_or(false);
            self.specs = (0..gpus.len())
                .filter_map(|i| {
                    let physical_index = gpus[i].0;
                    let device_id = Device::new(gpus[i].1);
                    let index = match cuda_ordinal_for_opencl_device(device_id.id()) {
                        Some(ordinal) => ordinal as usize,
                        None if is_nvidia => {
                            warn!(
                                "Skipping NVIDIA OpenCL device {} because its PCI identity cannot be mapped safely to a visible CUDA device",
                                device_id.name().unwrap_or_else(|_| "unknown".into())
                            );
                            return None;
                        }
                        None => physical_index,
                    };
                    Some(OpenCLWorkerSpec {
                        _platform: *_platform,
                        index,
                        device_id,
                        workload: match &opts.opencl_workload {
                            Some(workload) if i < workload.len() => workload[i],
                            Some(workload) if !workload.is_empty() => *workload.last().unwrap(),
                            // AUTO: no --opencl-workload given. 0.0 is a sentinel that
                            // tells the worker to pick a capability-driven default ratio
                            // per GPU arch (the old flat 512 under-saturated big cards).
                            _ => AUTO_WORKLOAD,
                        },
                        is_absolute: opts.opencl_workload_absolute,
                        experimental_amd: opts.experimental_amd,
                        use_amd_binary: !opts.opencl_no_amd_binary,
                        random: opts.opencl_nonce_gen,
                    })
                })
                .collect();
        }
        Ok(self.specs.len())
    }
}

#[derive(Copy, Clone)]
struct OpenCLWorkerSpec {
    _platform: Platform,
    index: usize,
    device_id: Device,
    workload: f32,
    is_absolute: bool,
    experimental_amd: bool,
    use_amd_binary: bool,
    random: NonceGenEnum,
}

impl WorkerSpec for OpenCLWorkerSpec {
    fn id(&self) -> String {
        format!(
            "#{} {}",
            self.index,
            self.device_id
                .board_name_amd()
                .unwrap_or_else(|_| self.device_id.name().unwrap_or_else(|_| "Unknown Device".into()))
        )
    }

    fn build(&self) -> Box<dyn Worker> {
        Box::new(
            OpenCLGPUWorker::new(
                self.index as u32,
                self.device_id,
                self.workload,
                self.is_absolute,
                self.experimental_amd,
                self.use_amd_binary,
                &self.random,
            )
            .unwrap(),
        )
    }
}

declare_plugin!(OpenCLPlugin, OpenCLPlugin::new, OpenCLOpt);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_nvidia_opencl_devices_map_to_cuda_ordinals() {
        let Ok(platforms) = get_platforms() else {
            return;
        };
        for platform in platforms {
            let Ok(devices) = platform.get_devices(CL_DEVICE_TYPE_ALL) else {
                continue;
            };
            for device_id in devices {
                let device = Device::new(device_id);
                if !device
                    .vendor()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .contains("nvidia")
                {
                    continue;
                }
                assert!(
                    cuda_ordinal_for_opencl_device(device_id).is_some(),
                    "NVIDIA OpenCL device {} was not mapped to a visible CUDA ordinal",
                    device.name().unwrap_or_else(|_| "unknown".into())
                );
            }
        }
    }
}
