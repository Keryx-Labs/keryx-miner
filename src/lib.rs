use clap::ArgMatches;
use std::any::Any;
use std::error::Error as StdError;
use std::ffi::CStr;
use std::io::IsTerminal;

pub mod gguf;
pub mod inference;
pub mod integrity;
pub mod llama_engine;
pub mod models;
pub mod pom;
pub mod pom_gpu;
pub mod slm;
pub mod xoshiro256starstar;
use libloading::{Library, Symbol};

pub type Error = Box<dyn StdError + Send + Sync + 'static>;
pub type PluginLogSink = extern "C" fn(level: u8, msg_ptr: *const u8, msg_len: usize);
pub const PLUGIN_ABI_VERSION: u32 = 2;

pub const PLUGIN_LOG_ERROR: u8 = 1;
pub const PLUGIN_LOG_WARN: u8 = 2;
pub const PLUGIN_LOG_INFO: u8 = 3;
pub const PLUGIN_LOG_DEBUG: u8 = 4;
pub const PLUGIN_LOG_TRACE: u8 = 5;

/// Resolve an unambiguous physical PCI bus number to the CUDA logical ordinal used by the miner.
/// Returns `None` when multiple PCI domains contain the same bus number.
pub fn cuda_ordinal_for_pci_bus(bus_id: u32) -> Option<u32> {
    use cudarc::driver::{result, sys};

    result::init().ok()?;
    let count = result::device::get_count().ok()?;
    let mut matched = None;
    for ordinal in 0..count {
        let Ok(device) = result::device::get(ordinal) else {
            continue;
        };
        let Ok(device_bus) = (unsafe {
            result::device::get_attribute(device, sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
        }) else {
            continue;
        };
        if device_bus as u32 == bus_id {
            if matched.is_some() {
                return None;
            }
            matched = Some(ordinal as u32);
        }
    }
    matched
}

/// Resolve a full PCI domain/bus/device/function identity to the CUDA logical ordinal.
pub fn cuda_ordinal_for_pci(domain: u32, bus: u32, device_id: u32, function: u32) -> Option<u32> {
    use cudarc::driver::{result, sys};

    result::init().ok()?;
    let count = result::device::get_count().ok()?;
    for ordinal in 0..count {
        let Ok(device) = result::device::get(ordinal) else {
            continue;
        };
        let mut buffer = [0i8; 32];
        let status = unsafe { sys::cuDeviceGetPCIBusId(buffer.as_mut_ptr(), buffer.len() as i32, device) };
        if status.result().is_err() {
            continue;
        }
        let Ok(value) = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str() else {
            continue;
        };
        if parse_pci_address(value) == Some((domain, bus, device_id, function)) {
            return Some(ordinal as u32);
        }
    }
    None
}

fn parse_pci_address(value: &str) -> Option<(u32, u32, u32, u32)> {
    let (domain, rest) = value.trim().split_once(':')?;
    let (bus, rest) = rest.split_once(':')?;
    let (device, function) = rest.split_once('.')?;
    Some((
        u32::from_str_radix(domain, 16).ok()?,
        u32::from_str_radix(bus, 16).ok()?,
        u32::from_str_radix(device, 16).ok()?,
        u32::from_str_radix(function, 16).ok()?,
    ))
}

#[derive(Default)]
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
    loaded_libraries: Vec<Library>,
    startup_warnings: Vec<String>,
}

/**
 Plugin Manager class - allows inserting your own hashers
 Inspired by https://michael-f-bryan.github.io/rust-ffi-guide/dynamic_loading.html
*/
impl PluginManager {
    pub fn new() -> Self {
        Self { plugins: Vec::new(), loaded_libraries: Vec::new(), startup_warnings: Vec::new() }
    }

    fn record_startup_warning(&mut self, message: String) {
        self.startup_warnings.push(message.clone());
        if should_emit_startup_stderr() {
            eprintln!("{}", message);
        }
    }

    pub fn drain_startup_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.startup_warnings)
    }

    pub(crate) unsafe fn load_single_plugin<'help>(
        &mut self,
        app: clap::App<'help>,
        path: &str,
    ) -> Result<clap::App<'help>, (clap::App<'help>, Error)> {
        type PluginAbiVersion = unsafe extern "C" fn() -> u32;
        type PluginCreate<'help> =
            unsafe extern "C" fn(*mut clap::App<'help>) -> (*mut clap::App<'help>, *mut dyn Plugin, *mut Error);

        let lib = match Library::new(path) {
            Ok(l) => l,
            Err(e) => return Err((app, e.to_string().into())),
        };

        let abi_version: Symbol<PluginAbiVersion> = match lib.get(b"_plugin_abi_version") {
            Ok(version) => version,
            Err(_) => {
                return Err((
                    app,
                    format!("Plugin {} has no ABI version; install plugins packaged with this miner", path).into(),
                ))
            }
        };
        let found_abi = abi_version();
        if found_abi != PLUGIN_ABI_VERSION {
            return Err((
                app,
                format!(
                    "Plugin {} uses ABI {}, but this miner requires ABI {}; install the matching plugin package",
                    path, found_abi, PLUGIN_ABI_VERSION
                )
                .into(),
            ));
        }

        self.loaded_libraries.push(lib); // Save library so it persists in memory
        let lib = self.loaded_libraries.last().unwrap();

        let constructor: Symbol<PluginCreate> = match lib.get(b"_plugin_create") {
            Ok(cons) => cons,
            Err(e) => return Err((app, e.to_string().into())),
        };

        let (app, boxed_raw, error) = constructor(Box::into_raw(Box::new(app)));
        let app = *Box::from_raw(app);

        if boxed_raw.is_null() {
            return Err((app, *Box::from_raw(error)));
        }
        let plugin = Box::from_raw(boxed_raw);
        self.plugins.push(plugin);

        Ok(app)
    }

    pub fn build(&self) -> Result<Vec<Box<dyn WorkerSpec + 'static>>, Error> {
        let mut specs = Vec::<Box<dyn WorkerSpec + 'static>>::new();
        for plugin in &self.plugins {
            if plugin.enabled() {
                specs.extend(plugin.get_worker_specs());
            }
        }
        Ok(specs)
    }

    /**
    Process the options for a plugin, and reports how many workers are available
    */
    pub fn process_options(&mut self, matchs: &ArgMatches) -> Result<usize, Error> {
        let mut count = 0usize;
        let mut warnings = Vec::new();
        for plugin in self.plugins.iter_mut() {
            count += match plugin.process_option(matchs) {
                Ok(n) => n,
                Err(e) => {
                    warnings.push(format!(
                        "WARNING: Failed processing options for {} (ignore if you do not intend to use): {}",
                        plugin.name(),
                        e
                    ));
                    0
                }
            };
        }
        for warning in warnings {
            self.record_startup_warning(warning);
        }
        Ok(count)
    }

    pub fn has_specs(&self) -> bool {
        !self.plugins.is_empty()
    }

    pub fn set_log_sink(&mut self, sink: Option<PluginLogSink>) {
        for plugin in self.plugins.iter_mut() {
            plugin.set_log_sink(sink);
        }
    }
}

#[inline]
fn should_emit_startup_stderr() -> bool {
    !std::io::stdout().is_terminal()
}

pub trait Plugin: Any + Send + Sync {
    fn name(&self) -> &'static str;
    fn enabled(&self) -> bool;
    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>>;
    fn process_option(&mut self, matchs: &ArgMatches) -> Result<usize, Error>;
    fn set_log_sink(&mut self, _sink: Option<PluginLogSink>) {}
}

pub trait WorkerSpec: Any + Send + Sync {
    /*type_: GPUWorkType,
    opencl_platform: u16,
    device_id: u32,
    workload: f32,
    is_absolute: bool*/
    fn id(&self) -> String;
    fn build(&self) -> Box<dyn Worker>;
}

pub trait Worker {
    //fn new(device_id: u32, workload: f32, is_absolute: bool) -> Result<Self, Error>;
    fn id(&self) -> String;
    fn load_block_constants(
        &mut self,
        hash_header: &[u8; 72],
        matrix: &[[u16; 64]; 64],
        target: &[u64; 4],
    ) -> Result<(), Error>;

    fn calculate_hash(&mut self, nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) -> Result<(), Error>;
    fn sync(&self) -> Result<(), Error>;

    fn get_workload(&self) -> usize;
    fn read_winner(&mut self) -> Result<Option<u64>, Error>;
}

pub fn load_plugins<'help>(
    app: clap::App<'help>,
    paths: &[String],
) -> Result<(clap::App<'help>, PluginManager), Error> {
    let mut factory = PluginManager::new();
    let mut app = app;
    for path in paths {
        app = unsafe {
            factory.load_single_plugin(app, path.as_str()).unwrap_or_else(|(app, e)| {
                factory.record_startup_warning(format!(
                    "WARNING: Failed loading plugin {} (ignore if you do not intend to use): {}",
                    path, e
                ));
                app
            })
        };
    }
    Ok((app, factory))
}

#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty, $constructor:path, $args:ty) => {
        use clap::Args;
        #[no_mangle]
        pub extern "C" fn _plugin_abi_version() -> u32 {
            $crate::PLUGIN_ABI_VERSION
        }

        #[no_mangle]
        pub unsafe extern "C" fn _plugin_create(
            app: *mut clap::App,
        ) -> (*mut clap::App, *mut dyn $crate::Plugin, *mut $crate::Error) {
            // make sure the constructor is the correct type.
            let constructor: fn() -> Result<$plugin_type, $crate::Error> = $constructor;

            let object = match constructor() {
                Ok(obj) => obj,
                Err(e) => {
                    return (
                        app,
                        std::ptr::null_mut::<$plugin_type>() as *mut dyn $crate::Plugin,
                        Box::into_raw(Box::new(e)),
                    );
                }
            };

            let boxed: Box<dyn $crate::Plugin> = Box::new(object);

            let boxed_app = Box::new(<$args>::augment_args(unsafe { *Box::from_raw(app) }));
            (Box::into_raw(boxed_app), Box::into_raw(boxed), std::ptr::null_mut::<Error>())
        }
    };
}

#[cfg(test)]
mod pci_tests {
    use super::parse_pci_address;

    #[test]
    fn parses_full_cuda_pci_identity() {
        assert_eq!(parse_pci_address("00000000:01:02.3"), Some((0, 1, 2, 3)));
        assert_eq!(parse_pci_address("0000:af:00.0"), Some((0, 0xaf, 0, 0)));
        assert_eq!(parse_pci_address("invalid"), None);
    }
}
