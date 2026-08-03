use clap::ArgMatches;
use std::any::Any;
use std::error::Error as StdError;
use std::io::IsTerminal;

pub mod gguf;
pub mod integrity;
pub mod inference;
pub mod llama_engine;
pub mod models;
pub mod pom;
pub mod pom_gpu;
pub mod slm;
pub mod xoshiro256starstar;
use libloading::{Library, Symbol};

pub type Error = Box<dyn StdError + Send + Sync + 'static>;
pub type PluginLogSink = extern "C" fn(level: u8, msg_ptr: *const u8, msg_len: usize);

pub const PLUGIN_LOG_ERROR: u8 = 1;
pub const PLUGIN_LOG_WARN: u8 = 2;
pub const PLUGIN_LOG_INFO: u8 = 3;
pub const PLUGIN_LOG_DEBUG: u8 = 4;
pub const PLUGIN_LOG_TRACE: u8 = 5;

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
        Self {
            plugins: Vec::new(),
            loaded_libraries: Vec::new(),
            startup_warnings: Vec::new(),
        }
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
        type PluginCreate<'help> =
            unsafe fn(*const clap::App<'help>) -> (*mut clap::App<'help>, *mut dyn Plugin, *mut Error);

        let lib = match Library::new(path) {
            Ok(l) => l,
            Err(e) => return Err((app, e.to_string().into())),
        };

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
    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]);

    fn calculate_hash(&mut self, nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64);
    fn sync(&self) -> Result<(), Error>;

    fn get_workload(&self) -> usize;
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error>;
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
        pub unsafe extern "C" fn _plugin_create(
            app: *mut clap::App,
        ) -> (*mut clap::App, *mut dyn $crate::Plugin, *const $crate::Error) {
            // make sure the constructor is the correct type.
            let constructor: fn() -> Result<$plugin_type, $crate::Error> = $constructor;

            let object = match constructor() {
                Ok(obj) => obj,
                Err(e) => {
                    return (
                        app,
                        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }, // Translates to null pointer
                        Box::into_raw(Box::new(e)),
                    );
                }
            };

            let boxed: Box<dyn $crate::Plugin> = Box::new(object);

            let boxed_app = Box::new(<$args>::augment_args(unsafe { *Box::from_raw(app) }));
            (Box::into_raw(boxed_app), Box::into_raw(boxed), std::ptr::null::<Error>())
        }
    };
}
