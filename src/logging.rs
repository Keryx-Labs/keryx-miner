use env_logger::Logger;
use log::{LevelFilter, Metadata, Record, SetLoggerError};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::Arc;

use crate::ui::UiState;

/// Plain log file used when none is given on the command line or in `KERYX_PLAIN_LOG_FILE`.
pub const DEFAULT_PLAIN_LOG_FILE: &str = "keryx-miner.log";
/// Size at which the plain log file is moved to `<path>.1` and started again.
const PLAIN_LOG_ROTATE_BYTES: u64 = 64 * 1024 * 1024;

/// Append-only plain log file, rotated once it reaches `PLAIN_LOG_ROTATE_BYTES`.
pub struct PlainLogFile {
    path: PathBuf,
    file: File,
    written: u64,
}

impl PlainLogFile {
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut log = Self { path, file, written };
        if log.written >= PLAIN_LOG_ROTATE_BYTES {
            log.rotate();
        }
        Ok(log)
    }

    fn rotate(&mut self) {
        let mut old = self.path.clone().into_os_string();
        old.push(".1");
        let _ = std::fs::remove_file(&old);
        let _ = std::fs::rename(&self.path, &old);
        if let Ok(file) = OpenOptions::new().create(true).write(true).truncate(true).open(&self.path) {
            self.file = file;
        }
        self.written = 0;
    }

    fn write_line(&mut self, line: &str) {
        if self.written >= PLAIN_LOG_ROTATE_BYTES {
            self.rotate();
        }
        if writeln!(self.file, "{}", line).is_ok() {
            self.written += line.len() as u64 + 1;
        }
    }
}

struct MinerLogger {
    inner: Logger,
    emit_inner: bool,
    ui: Option<Arc<UiState>>,
    plain_log_file: Option<Mutex<PlainLogFile>>,
}

impl log::Log for MinerLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        if self.emit_inner {
            self.inner.log(record);
        }

        if let Some(ui) = &self.ui {
            ui.push_log(record.level(), &format!("{}", record.args()));
        }

        if let Some(file) = &self.plain_log_file {
            if let Ok(mut f) = file.lock() {
                f.write_line(&format!("[{}] {}", record.level(), record.args()));
            }
        }
    }

    fn flush(&self) {
        self.inner.flush();
        if let Some(file) = &self.plain_log_file {
            if let Ok(mut f) = file.lock() {
                let _ = f.file.flush();
            }
        }
    }
}

pub fn init_logging(
    level: LevelFilter,
    ui: Option<Arc<UiState>>,
    emit_inner: bool,
    plain_log_file: Option<PlainLogFile>,
) -> Result<(), SetLoggerError> {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(level).parse_default_env();
    let inner = builder.build();
    let max_level = inner.filter();

    let logger = MinerLogger {
        inner,
        emit_inner,
        ui,
        plain_log_file: plain_log_file.map(Mutex::new),
    };

    log::set_boxed_logger(Box::new(logger))?;
    log::set_max_level(max_level);
    Ok(())
}
