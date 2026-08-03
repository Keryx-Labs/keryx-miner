use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use crossterm::{execute, queue};
use log::Level;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use time::{macros::format_description, OffsetDateTime};

use crate::stats::MinerStats;

const MAX_LOG_LINES: usize = 2000;
const UI_TICK_RATE: Duration = Duration::from_millis(100);
const MIN_LOG_ROWS: u16 = 5;
const BLOCK_CELEBRATION_DURATION: Duration = Duration::from_millis(2500);
const BLOCK_COIN_FRAME_RATE: Duration = Duration::from_millis(100);
const BLOCK_COIN_FRAME_COUNT: usize = 25;
const BLOCK_COIN_PIXEL_WIDTH: usize = 2;
// 18x18 indexed-color raster generated from the approved pixel coin concept.
const BLOCK_COIN_ART: [&str; 18] = [
    "...BBGGGHHGGGBB...",
    "..BGHHHHHHHHHHGB..",
    "..GHBBGGGGGGBBHG..",
    ".GHBGGHHHHHHGGBHG.",
    "BGBGHHHKHHHHHHGBGB",
    "BGBGHHKKHHKKHHGBGB",
    "BHBGHHKKHKKKHBGBDB",
    "BGBGHHKKKKDHHBGBDB",
    "BHBGHHKKKHHHHBGBDB",
    "BHBGHHKKKKHHHBGBDB",
    "BGBGHHKKHKDHHBGBDB",
    "BGBGHHKKHHKKHBGBDB",
    "BGBGHHKKHHHHHBGBDB",
    "BGBGHHHKHHHHBHGBDB",
    ".GGBGGHHHHHHGGBDG.",
    "..GGBBGGGGGGBBDG..",
    "..BGGGGGGGGGGDGB..",
    "...BBGGGDDGGGBB...",
];

#[derive(Copy, Clone)]
struct Palette {
    bg: Color,
    panel: Color,
    accent: Color,
    text: Color,
    muted: Color,
    ok: Color,
    dim: Color,
    bright: Color,
    warn: Color,
    mid: Color,
    err: Color,
}

const PALETTE_TRUECOLOR: Palette = Palette {
    bg: Color::Rgb {
        r: 0x07,
        g: 0x0a,
        b: 0x08,
    },
    panel: Color::Rgb {
        r: 0x0d,
        g: 0x11,
        b: 0x0d,
    },
    accent: Color::Rgb {
        r: 0x23,
        g: 0xd6,
        b: 0x58,
    },
    text: Color::Rgb {
        r: 0x6f,
        g: 0xd5,
        b: 0x83,
    },
    muted: Color::Rgb {
        r: 0x35,
        g: 0x44,
        b: 0x3a,
    },
    ok: Color::Rgb {
        r: 0x2e,
        g: 0xe3,
        b: 0x58,
    },
    dim: Color::Rgb {
        r: 0x4f,
        g: 0x9a,
        b: 0x60,
    },
    bright: Color::Rgb {
        r: 0x30,
        g: 0xff,
        b: 0x67,
    },
    warn: Color::Rgb {
        r: 0xff,
        g: 0xbf,
        b: 0x3f,
    },
    mid: Color::Rgb {
        r: 0x4e,
        g: 0xcb,
        b: 0x6a,
    },
    err: Color::Rgb {
        r: 0xe0,
        g: 0x52,
        b: 0x52,
    },
};

const PALETTE_ANSI: Palette = Palette {
    bg: Color::AnsiValue(232),
    panel: Color::AnsiValue(233),
    accent: Color::AnsiValue(41),
    text: Color::AnsiValue(114),
    muted: Color::AnsiValue(240),
    ok: Color::AnsiValue(46),
    dim: Color::AnsiValue(71),
    bright: Color::AnsiValue(82),
    warn: Color::AnsiValue(220),
    mid: Color::AnsiValue(78),
    err: Color::AnsiValue(203),
};

fn palette() -> Palette {
    static CHOSEN: OnceLock<Palette> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        if supports_truecolor() {
            PALETTE_TRUECOLOR
        } else {
            PALETTE_ANSI
        }
    })
}

fn supports_truecolor() -> bool {
    if let Ok(force) = env::var("KERYX_TRUECOLOR") {
        let force = force.trim().to_ascii_lowercase();
        if matches!(force.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
        if matches!(force.as_str(), "0" | "false" | "no" | "off") {
            return false;
        }
    }

    let colorterm = env::var("COLORTERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return true;
    }

    let term = env::var("TERM").unwrap_or_default().to_ascii_lowercase();
    if term.starts_with("screen") || term == "linux" {
        return false;
    }

    term.contains("direct")
        || term.contains("kitty")
        || term.contains("wezterm")
        || term.contains("alacritty")
        || term.contains("foot")
}

pub struct UiState {
    lines: Mutex<VecDeque<String>>,
    scrollback_lines: AtomicU64,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            lines: Mutex::new(VecDeque::with_capacity(MAX_LOG_LINES)),
            scrollback_lines: AtomicU64::new(0),
        }
    }

    pub fn push_log(&self, level: Level, message: &str) {
        if matches!(level, Level::Info | Level::Warn | Level::Error) {
            let mut lines = self.lines.lock().expect("ui log mutex poisoned");
            for entry in sanitize_log_message(message) {
                let line = format!("{} [{}] {}", tty_timestamp(), level, entry);
                lines.push_back(line);
            }
            while lines.len() > MAX_LOG_LINES {
                lines.pop_front();
            }
            if self.scrollback_lines.load(Ordering::Acquire) > 0 {
                let max_offset = lines.len().saturating_sub(1) as u64;
                let current = self.scrollback_lines.load(Ordering::Acquire);
                self.scrollback_lines
                    .store(current.saturating_add(1).min(max_offset), Ordering::Release);
            }
        }

    }

    fn visible_lines(&self, n: usize) -> Vec<String> {
        let lines = self.lines.lock().expect("ui log mutex poisoned");
        let len = lines.len();
        let offset = self.scrollback_lines.load(Ordering::Acquire) as usize;
        let end = len.saturating_sub(offset);
        let start = end.saturating_sub(n);
        lines.iter().skip(start).cloned().collect()
    }

    fn scroll_up(&self, amount: usize) {
        let len = self.lines.lock().expect("ui log mutex poisoned").len();
        let max_offset = len.saturating_sub(1) as u64;
        let current = self.scrollback_lines.load(Ordering::Acquire);
        self.scrollback_lines
            .store(current.saturating_add(amount as u64).min(max_offset), Ordering::Release);
    }

    fn scroll_down(&self, amount: usize) {
        let current = self.scrollback_lines.load(Ordering::Acquire);
        self.scrollback_lines
            .store(current.saturating_sub(amount as u64), Ordering::Release);
    }

    fn scroll_to_top(&self) {
        let len = self.lines.lock().expect("ui log mutex poisoned").len();
        self.scrollback_lines
            .store(len.saturating_sub(1) as u64, Ordering::Release);
    }

    fn scroll_to_live(&self) {
        self.scrollback_lines.store(0, Ordering::Release);
    }

    fn is_scrolled(&self) -> bool {
        self.scrollback_lines.load(Ordering::Acquire) > 0
    }

    fn scrollback_lines(&self) -> u64 {
        self.scrollback_lines.load(Ordering::Acquire)
    }
}

fn sanitize_log_message(message: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in message.lines() {
        // Drop escape/control characters so dependency output cannot execute
        // terminal control sequences inside the alternate-screen UI.
        let mut cleaned = String::new();
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                // Skip basic ANSI CSI/OSC/SS3 escape payloads.
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&next) || next == '\u{7}' {
                        break;
                    }
                }
                continue;
            }
            if c.is_control() {
                continue;
            }
            cleaned.push(c);
        }

        if cleaned.trim().is_empty() {
            continue;
        }
        out.push(cleaned);
    }

    if out.is_empty() {
        vec!["(log)".to_string()]
    } else {
        out
    }
}

fn tty_timestamp() -> String {
    let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
    OffsetDateTime::now_utc()
        .format(&format)
    .map(|stamp| format!("{} UTC", stamp))
    .unwrap_or_else(|_| "0000-00-00 00:00:00 UTC".to_string())
}

pub struct UiGuard {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for UiGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn_ui(stats: Arc<MinerStats>, ui_state: Arc<UiState>, shutdown_requested: Arc<AtomicBool>) -> UiGuard {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);

    let handle = thread::spawn(move || {
        let mut out = stdout();
        let _ = enable_raw_mode();
        let _ = wait_for_stable_terminal_size();
        let _ = execute!(out, EnterAlternateScreen, Hide);
        let mut last_size: Option<(u16, u16)> = None;
        let mut pending_resize: Option<(u16, u16)> = None;
        let mut first_frame = true;
        let mut last_draw_key: Option<(
            u16,
            u16,
            bool,
            bool,
            u64,
            usize,
            u64,
            u64,
            u64,
            bool,
        )> = None;
        let mut last_drawn_at = Instant::now();
        let mut last_block_coin_frame = None;
        let mut last_accepted_blocks = stats.snapshot().accepted_blocks;
        let mut block_celebration_started: Option<Instant> = None;
        let mut block_celebrations_enabled = true;

        while !stop_clone.load(Ordering::Acquire) {
            if handle_input(&ui_state, &shutdown_requested, &mut block_celebrations_enabled) {
                break;
            }
            if !block_celebrations_enabled {
                block_celebration_started = None;
            }
            let current_size = terminal::size()
                .ok()
                .filter(|(w, h)| *w > 0 && *h > 0);

            let Some(current_size) = current_size else {
                thread::sleep(UI_TICK_RATE);
                continue;
            };

            let should_clear = if first_frame {
                first_frame = false;
                last_size = Some(current_size);
                pending_resize = None;
                true
            } else if last_size == Some(current_size) {
                pending_resize = None;
                false
            } else if pending_resize == Some(current_size) {
                last_size = Some(current_size);
                pending_resize = None;
                true
            } else {
                // Ignore a single-frame size wobble; clear only after stable resize confirmation.
                pending_resize = Some(current_size);
                thread::sleep(UI_TICK_RATE);
                continue;
            };

            let snapshot = stats.snapshot();
            if block_celebrations_enabled && snapshot.accepted_blocks > last_accepted_blocks {
                block_celebration_started = Some(Instant::now());
            }
            last_accepted_blocks = snapshot.accepted_blocks;
            let block_coin_frame = block_celebration_started
                .and_then(|started| block_animation_frame(started.elapsed()));
            if block_celebration_started.is_some() && block_coin_frame.is_none() {
                block_celebration_started = None;
            }
            let draw_key = (
                current_size.0,
                current_size.1,
                snapshot.synced,
                snapshot.opoi_challenge_active,
                snapshot.total_hashrate_hs,
                snapshot.devices.len(),
                snapshot.last_update_epoch_s,
                snapshot.accepted_blocks,
                snapshot.rejected_blocks,
                block_celebrations_enabled,
            );
            let animation_ended = last_block_coin_frame.is_some() && block_coin_frame.is_none();
            let periodic_refresh_due =
                block_coin_frame.is_none() && last_drawn_at.elapsed() >= Duration::from_secs(1);
            if should_clear || animation_ended || periodic_refresh_due || last_draw_key != Some(draw_key) {
                draw_frame(
                    &mut out,
                    current_size,
                    &snapshot,
                    &ui_state,
                    should_clear,
                    block_coin_frame,
                );
                last_draw_key = Some(draw_key);
                last_drawn_at = Instant::now();
                last_block_coin_frame = block_coin_frame;
            } else if block_coin_frame != last_block_coin_frame {
                if let Some(frame) = block_coin_frame {
                    draw_block_celebration(
                        &mut out,
                        current_size.0,
                        current_size.1,
                        frame,
                        snapshot.accepted_blocks,
                    );
                    let _ = out.flush();
                }
                last_block_coin_frame = block_coin_frame;
            }
            thread::sleep(UI_TICK_RATE);
        }

        let _ = execute!(out, Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    });

    UiGuard {
        stop,
        handle: Some(handle),
    }
}

fn wait_for_stable_terminal_size() -> Option<(u16, u16)> {
    let mut last = None;
    for _ in 0..10 {
        let current = terminal::size().ok();
        if let Some((w, h)) = current {
            if w > 0 && h > 0 && last == current {
                return current;
            }
            last = current;
        }
        thread::sleep(Duration::from_millis(50));
    }
    last
}

enum PanelRow {
    Plain { text: String, fg: Color, bold: bool },
    Segments(Vec<(String, Color)>),
}

fn metric_row(label: &str, value: String, value_color: Color) -> PanelRow {
    PanelRow::Segments(vec![
        (format!(" {:<16}", label), palette().dim),
        (value, value_color),
    ])
}

fn draw_frame(
    out: &mut std::io::Stdout,
    size: (u16, u16),
    snapshot: &crate::stats::MinerStatsSnapshot,
    ui_state: &UiState,
    _clear_screen: bool,
    block_coin_frame: Option<usize>,
) {
    let (w, h) = size;
    let total_width = w as usize;

    // Avoid terminal clear escapes (ED/EL): on some terminals they erase using
    // the shell default background, causing brief color flashes and smear.
    // We repaint the full visible frame explicitly instead.
    let _ = queue!(out, MoveTo(0, 0), SetBackgroundColor(palette().bg));

    let kernel_by_device: HashMap<u32, keryx_miner::pom_gpu::GpuKernelInfo> =
        keryx_miner::pom_gpu::list_gpu_kernel_info()
            .into_iter()
            .map(|k| (k.device_id, k))
            .collect();
    let loaded_model_by_device: HashMap<u32, String> =
        keryx_miner::pom_gpu::list_mining_model_labels()
            .into_iter()
            .collect();

    // Bias layout toward the device pane, but keep left metrics moderately large: ~43% / ~57%.
    let left_w = total_width.saturating_sub(1) * 43 / 100;
    let right_w = total_width.saturating_sub(left_w + 1);
    let divider_x = left_w as u16;
    let right_x = left_w as u16 + 1;
    let compact = total_width < 132;

    let hashrate_value = format_hashrate(snapshot.total_hashrate_hs);
    let opoi_pause_value = if snapshot.opoi_challenge_active { "Active" } else { "Idle" };
    let accepted_blocks_value = snapshot.accepted_blocks;
    let rejected_value = snapshot.rejected_blocks;
    let claims_value = format!(
        "{} ({:.2} KRX)",
        snapshot.claimed_outputs,
        snapshot.claimed_sompi as f64 / 1e8
    );
    let escrow_pending_value = format!(
        "{} ({:.2} KRX)",
        snapshot.escrow_pending_outputs,
        snapshot.escrow_pending_sompi as f64 / 1e8
    );
    let uptime_value = format_duration(snapshot.uptime_s);
    let last_update_age = seconds_since(snapshot.last_update_epoch_s);
    let last_update_value = format!("{}s ago", last_update_age);
    let (memory_value, memory_color) = match system_memory_summary() {
        Some((used_mb, total_mb)) if total_mb > 0 => {
            let used_pct = ((used_mb as f64 / total_mb as f64) * 100.0).round() as u32;
            let color = if used_pct >= 97 {
                palette().err
            } else if used_pct >= 90 {
                palette().warn
            } else if used_pct >= 80 {
                palette().mid
            } else {
                palette().ok
            };
            (
                format!(
                    "{:.1}/{:.1} GB ({}%)",
                    used_mb as f64 / 1024.0,
                    total_mb as f64 / 1024.0,
                    used_pct
                ),
                color,
            )
        }
        _ => ("--".to_string(), palette().muted),
    };
    let (load_value, load_color) = match load_average_summary() {
        Some((load_1m, load_5m, load_15m)) => {
            let cores = std::thread::available_parallelism().ok().map(|n| n.get()).unwrap_or(1) as f64;
            let pct = if cores > 0.0 {
                ((load_1m / cores) * 100.0).round().clamp(0.0, 999.0) as u32
            } else {
                0
            };
            let color = if pct >= 95 {
                palette().err
            } else if pct >= 80 {
                palette().warn
            } else if pct >= 60 {
                palette().mid
            } else {
                palette().ok
            };
            (
                format!("{:.2}/{:.2}/{:.2}", load_1m, load_5m, load_15m),
                color,
            )
        }
        None => ("--".to_string(), palette().muted),
    };
    let address_value = snapshot
        .mining_address
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("(recovery mode)");
    let api_port_value = snapshot
        .api_port
        .map(|p| p.to_string())
        .unwrap_or_else(|| "--".to_string());
    let address_metric = truncate_middle(address_value, left_w.saturating_sub(12));
    let log_mode_value = if ui_state.is_scrolled() {
        format!("{} +{}", uppercase_first_char("scrolled"), ui_state.scrollback_lines())
    } else {
        uppercase_first_char("live")
    };

    let top_left = vec![
        (" Keryx Miner ".to_string(), palette().accent),
        ("| ".to_string(), palette().muted),
        (format!("{}  ", hashrate_value), palette().ok),
        ("| Node ".to_string(), palette().muted),
        (
            format!("{}  ", if snapshot.synced { "Synced" } else { "Not Synced" }),
            if snapshot.synced { palette().ok } else { palette().warn },
        ),
        ("| OPoI ".to_string(), palette().muted),
        (
            format!("{}", opoi_pause_value),
            if snapshot.opoi_challenge_active {
                palette().bright
            } else {
                palette().dim
            },
        ),
    ];

    let top_right = vec![
        (" API ".to_string(), palette().muted),
        (":".to_string(), palette().ok),
        (api_port_value, palette().ok),
        ("/v1/miner/stats".to_string(), palette().ok),
        (" | RAM ".to_string(), palette().muted),
        (memory_value, memory_color),
        (" | Load Avg [1m/5m/15m] ".to_string(), palette().muted),
        (load_value, load_color),
    ];

    draw_colored_segments_cell(
        out,
        0,
        0,
        left_w,
        &top_left,
        palette().panel,
    );
    draw_colored_cell(out, divider_x, 0, 1, "|", palette().muted, palette().panel, false);
    draw_colored_segments_cell(out, right_x, 0, right_w, &top_right, palette().panel);

    let available_content_rows = h.saturating_sub(MIN_LOG_ROWS + 3) as usize;
    let content_budget = available_content_rows.max(4);

    let mut left_rows: Vec<PanelRow> = vec![
        PanelRow::Plain {
            text: " Metrics".to_string(),
            fg: palette().accent,
            bold: true,
        },
        metric_row("Hashrate", hashrate_value.clone(), palette().ok),
        metric_row("Uptime", uptime_value.clone(), palette().text),
        PanelRow::Segments(vec![
            (format!(" {:<16}", "Address"), palette().dim),
            (address_metric, palette().ok),
        ]),
        metric_row("Blocks Accepted", accepted_blocks_value.to_string(), palette().bright),
        metric_row(
            "Blocks Rejected",
            rejected_value.to_string(),
            if rejected_value > 0 { palette().err } else { palette().ok },
        ),
        metric_row(
            "Escrow Claimed",
            claims_value.clone(),
            if snapshot.claimed_outputs > 0 { palette().bright } else { palette().dim },
        ),
        metric_row(
            "Escrow Pending",
            escrow_pending_value.clone(),
            if snapshot.escrow_pending_outputs > 0 { palette().text } else { palette().dim },
        ),
        metric_row(
            "Stats Updated",
            last_update_value,
            if last_update_age <= 3 { palette().ok } else { palette().warn },
        ),
        metric_row(
            "Log Mode",
            log_mode_value.clone(),
            if ui_state.is_scrolled() { palette().bright } else { palette().dim },
        ),
        PanelRow::Plain {
            text: "".to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: " Controls".to_string(),
            fg: palette().accent,
            bold: true,
        },
        PanelRow::Plain {
            text: if compact { " U/D Scroll" } else { " Up/Down      Scroll" }.to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: if compact {
                " Pg Fast"
            } else {
                " PgUp/PgDn    Scroll faster"
            }
            .to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: if compact {
                " Home/End L/O"
            } else {
                " Home/End     Oldest/Live"
            }
            .to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: if compact { " B Block FX" } else { " B            Block animation" }.to_string(),
            fg: palette().text,
            bold: false,
        },
    ];

    if left_rows.len() > content_budget {
        left_rows.truncate(content_budget);
    }

    let mut right_rows: Vec<PanelRow> = Vec::new();
    right_rows.push(PanelRow::Plain {
        text: " Devices".to_string(),
        fg: palette().accent,
        bold: true,
    });

    let id_w = if compact { 3usize } else { 4usize };
    let model_w = if compact { 10usize } else { 14usize };
    let rate_w = if compact { 9usize } else { 11usize };
    let cc_w = if compact { 3usize } else { 5usize };
    let cm_w = if compact { 7usize } else { 9usize };
    let fan_w = 4usize;
    let bar_w = if compact {
        (right_w / 10).clamp(3, 7)
    } else {
        (right_w / 8).clamp(5, 11)
    };
    let free_w = right_w.saturating_sub(id_w + model_w + rate_w + cc_w + cm_w + fan_w + bar_w + 12);
    // Keep Loaded Model closer to Kernel: trim Kernel by ~5 chars when possible.
    let kernel_min = if compact { 7 } else { 10 };
    let kernel_w = ((free_w * 2 / 5).max(kernel_min)).saturating_sub(3).max(kernel_min);
    let loaded_w = free_w.saturating_sub(kernel_w).max(if compact { 9 } else { 12 });
    right_rows.push(PanelRow::Plain {
        text: format!(
            " {:<id_w$} {:<model_w$} {:<rate_w$} {:<cc_w$} {:<cm_w$} {:<fan_w$} {:<bar_w$} {:<kernel_w$} {:<loaded_w$}",
            "ID",
            "Model",
            "Hashrate",
            "CC",
            "Core/Mem",
            "Fan",
            "Load",
            "Kernel",
            "Loaded Model",
            id_w = id_w,
            model_w = model_w,
            rate_w = rate_w,
            cc_w = cc_w,
            cm_w = cm_w,
            fan_w = fan_w,
            bar_w = bar_w,
            kernel_w = kernel_w,
            loaded_w = loaded_w,
        ),
        fg: palette().muted,
        bold: true,
    });

    let max_rate = snapshot
        .devices
        .iter()
        .map(|d| d.hashrate_hs)
        .max()
        .unwrap_or(1)
        .max(1);

    let max_device_rows = ((content_budget.saturating_sub(2)) / 2).max(1);
    let shown = snapshot.devices.len().min(max_device_rows);
    for d in snapshot.devices.iter().take(shown) {
        let dev_id = parse_device_id(&d.id);
        let (compute, kernel) = dev_id
            .and_then(|id| kernel_by_device.get(&id))
            .map(|k| {
                let compute = match (k.cc_major, k.cc_minor) {
                    (Some(maj), Some(min)) => format!("{}.{}", maj, min),
                    _ => "n/a".to_string(),
                };
                (compute, k.image.clone())
            })
            .unwrap_or_else(|| ("n/a".to_string(), "n/a".to_string()));
        let loaded_model = dev_id
            .and_then(|id| loaded_model_by_device.get(&id).cloned())
            .unwrap_or_else(|| "n/a".to_string());

        let load_blocks = (((d.hashrate_hs as f64 / max_rate as f64) * bar_w as f64).round() as usize)
            .clamp(0, bar_w);
        let load_bar = format!("{}{}", "=".repeat(load_blocks), ".".repeat(bar_w.saturating_sub(load_blocks)));
        let id_short = parse_device_id(&d.id)
            .map(|id| format!("#{}", id))
            .unwrap_or_else(|| trim_to_width(&d.id, id_w));
        let model_short = trim_to_width(&short_device_model(&d.id), model_w);
        let rate_short = trim_to_width(&format_hashrate(d.hashrate_hs), rate_w);
        let temp = d.temp_c.map(|v| format!("{}C", v)).unwrap_or_else(|| "--".to_string());
        let mem = d
            .memory_temp_c
            .map(|v| format!("{}C", v))
            .unwrap_or_else(|| "--".to_string());
        let core_mem = trim_to_width(&format!("{}/{}", temp, mem), cm_w);
        let fan = d
            .fan_percent
            .map(|v| format!("{}%", v))
            .unwrap_or_else(|| "--".to_string());
        let kernel_short = trim_to_width(&kernel, kernel_w);
        let loaded_short = trim_to_width(&loaded_model, loaded_w);
        let power_short = d
            .power_draw_w
            .map(format_power_draw)
            .unwrap_or_else(|| "--".to_string());
        let efficiency_short = d
            .power_draw_w
            .map(|w| format_efficiency(d.hashrate_hs, w))
            .unwrap_or_else(|| "--".to_string());
        let detail_color = if power_short == "--" || efficiency_short == "--" {
            palette().muted
        } else {
            palette().ok
        };
        let brand_color = device_brand_color(&d.id);
        right_rows.push(PanelRow::Segments(vec![
            (
                format!(
                    " {:<id_w$} {:<model_w$} {:<rate_w$} {:<cc_w$} {:<cm_w$} {:<fan_w$} {:<bar_w$} {:<kernel_w$} {:<loaded_w$}",
                    id_short,
                    model_short,
                    rate_short,
                    compute,
                    core_mem,
                    fan,
                    load_bar,
                    kernel_short,
                    loaded_short,
                    id_w = id_w,
                    model_w = model_w,
                    rate_w = rate_w,
                    cc_w = cc_w,
                    cm_w = cm_w,
                    fan_w = fan_w,
                    bar_w = bar_w,
                    kernel_w = kernel_w,
                    loaded_w = loaded_w,
                ),
                brand_color,
            ),
        ]));
        let blocks_accepted_color = if d.blocks_accepted > 0 { palette().bright } else { detail_color };
        let blocks_rejected_color = if d.blocks_rejected > 0 { palette().err } else { detail_color };
        right_rows.push(PanelRow::Segments(vec![
            ("   ".to_string(), palette().muted),
            ("P: ".to_string(), palette().muted),
            (format!("{:<8}", power_short), detail_color),
            ("  ".to_string(), palette().muted),
            ("Eff: ".to_string(), palette().muted),
            (efficiency_short, detail_color),
            ("  ".to_string(), palette().muted),
            ("Blocks Accepted: ".to_string(), palette().muted),
            (d.blocks_accepted.to_string(), blocks_accepted_color),
            ("  ".to_string(), palette().muted),
            ("Blocks Rejected: ".to_string(), palette().muted),
            (d.blocks_rejected.to_string(), blocks_rejected_color),
        ]));
    }

    if snapshot.devices.len() > shown {
        right_rows.push(PanelRow::Plain {
            text: format!(" +{} more devices", snapshot.devices.len().saturating_sub(shown)),
            fg: palette().muted,
            bold: false,
        });
    }

    let content_rows = left_rows.len().max(right_rows.len());

    for row in 0..content_rows {
        let y = row as u16 + 1;
        let left_bg = if row == 0 {
            palette().panel
        } else if row % 2 == 0 {
            palette().bg
        } else {
            palette().panel
        };
        let right_bg = if row == 0 {
            palette().panel
        } else if row % 2 == 0 {
            palette().bg
        } else {
            palette().panel
        };

        match left_rows.get(row) {
            Some(PanelRow::Plain { text, fg, bold }) => {
                draw_colored_cell(out, 0, y, left_w, text, *fg, left_bg, *bold);
            }
            Some(PanelRow::Segments(segments)) => {
                draw_colored_segments_cell(out, 0, y, left_w, segments, left_bg);
            }
            None => draw_colored_cell(out, 0, y, left_w, "", palette().text, left_bg, false),
        }

        draw_colored_cell(out, divider_x, y, 1, "|", palette().muted, palette().panel, false);

        match right_rows.get(row) {
            Some(PanelRow::Plain { text, fg, bold }) => {
                draw_colored_cell(out, right_x, y, right_w, text, *fg, right_bg, *bold);
            }
            Some(PanelRow::Segments(segments)) => {
                draw_colored_segments_cell(out, right_x, y, right_w, segments, right_bg);
            }
            None => draw_colored_cell(out, right_x, y, right_w, "", palette().text, right_bg, false),
        }
    }

    let separator_y = content_rows as u16 + 1;
    draw_colored_line(
        out,
        separator_y,
        &"-".repeat(total_width),
        palette().muted,
        Some(palette().panel),
        false,
    );

    let log_header_y = separator_y + 1;
    draw_colored_segments_cell(
        out,
        0,
        log_header_y,
        total_width,
        &[
            (" Logs ".to_string(), palette().accent),
            ("| ".to_string(), palette().muted),
            (
                format!("{}  ", log_mode_value),
                if ui_state.is_scrolled() { palette().bright } else { palette().dim },
            ),
            (
                if compact {
                    "| U/D Scroll  Pg Fast  Home/End  B FX"
                } else {
                    "| Up/Down Scroll  PgUp/PgDn Fast  Home Oldest  End Live  B Block FX"
                }
                .to_string(),
                palette().text,
            ),
        ],
        palette().panel,
    );

    let log_start_y = log_header_y + 1;
    let log_rows = h.saturating_sub(log_start_y);
    let log_lines = ui_state.visible_lines(log_rows as usize);

    for (i, line) in log_lines.iter().enumerate() {
        let y = log_start_y + i as u16;
        if y >= h {
            break;
        }
        let clipped = trim_to_width(line, total_width);
        draw_colored_segments_cell(
            out,
            0,
            y,
            total_width,
            &[(clipped, color_for_log_line(line))],
            palette().bg,
        );
    }

    // Clear any remaining rows in the log panel (mostly relevant during startup / resize).
    for y in (log_start_y + log_lines.len() as u16)..h {
        draw_colored_cell(out, 0, y, total_width, "", palette().text, palette().bg, false);
    }

    if let Some(frame) = block_coin_frame {
        draw_block_celebration(out, w, h, frame, snapshot.accepted_blocks);
    }

    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
    let _ = out.flush();
}

fn block_animation_frame(elapsed: Duration) -> Option<usize> {
    if elapsed >= BLOCK_CELEBRATION_DURATION {
        return None;
    }
    Some(
        ((elapsed.as_millis() / BLOCK_COIN_FRAME_RATE.as_millis()) as usize)
            .min(BLOCK_COIN_FRAME_COUNT - 1),
    )
}

fn draw_block_celebration(out: &mut std::io::Stdout, w: u16, h: u16, frame: usize, accepted_blocks: u64) {
    let title = format!("KRX BLOCK ACCEPTED  #{}", accepted_blocks);
    let art_width = BLOCK_COIN_ART
        .iter()
        .map(|line| line.chars().count() * BLOCK_COIN_PIXEL_WIDTH)
        .max()
        .unwrap_or(0);
    let overlay_width = art_width.max(title.len() + 2);
    let overlay_height = BLOCK_COIN_ART.len() + 2;

    if (w as usize) < overlay_width || (h as usize) < overlay_height {
        if h > 0 {
            let clipped_title: String = title.chars().take(w as usize).collect();
            draw_colored_cell(out, 0, 0, w as usize, &clipped_title, palette().bright, palette().panel, true);
        }
        return;
    }

    let x = (w as usize - overlay_width) as u16 / 2;
    let y = (h as usize - overlay_height) as u16 / 2;
    let title_x = x + ((overlay_width - title.len() - 2) / 2) as u16;
    draw_colored_cell(
        out,
        title_x,
        y,
        title.len() + 2,
        &format!(" {} ", title),
        block_coin_color('H', false),
        palette().panel,
        true,
    );

    let logical_width = art_width / BLOCK_COIN_PIXEL_WIDTH;
    let sweep = frame * (logical_width + 4) / (BLOCK_COIN_FRAME_COUNT - 1);
    let art_x = x + ((overlay_width - art_width) / 2) as u16;
    for (row, line) in BLOCK_COIN_ART.iter().enumerate() {
        for (column, glyph) in line.chars().enumerate() {
            if glyph == '.' {
                if block_coin_halo(row, column) {
                    let _ = queue!(
                        out,
                        MoveTo(art_x + (column * BLOCK_COIN_PIXEL_WIDTH) as u16, y + row as u16 + 2),
                        SetBackgroundColor(palette().panel),
                        Print("  "),
                        ResetColor
                    );
                }
                continue;
            }
            let color = block_coin_color(glyph, glyph != 'K' && column.abs_diff(sweep) <= 1);
            let _ = queue!(
                out,
                MoveTo(art_x + (column * BLOCK_COIN_PIXEL_WIDTH) as u16, y + row as u16 + 2),
                SetBackgroundColor(color),
                Print("  "),
                ResetColor
            );
        }
    }
}

fn block_coin_halo(row: usize, column: usize) -> bool {
    let row_start = row.saturating_sub(1);
    let row_end = (row + 1).min(BLOCK_COIN_ART.len() - 1);
    let column_start = column.saturating_sub(1);
    let column_end = (column + 1).min(BLOCK_COIN_ART[0].len() - 1);

    BLOCK_COIN_ART[row_start..=row_end].iter().any(|line| {
        line.as_bytes()[column_start..=column_end]
            .iter()
            .any(|pixel| *pixel != b'.')
    })
}

fn block_coin_color(glyph: char, highlighted: bool) -> Color {
    let truecolor = matches!(palette().warn, Color::Rgb { .. });
    if highlighted {
        return if truecolor {
            Color::Rgb { r: 0xff, g: 0xf4, b: 0xb0 }
        } else {
            Color::AnsiValue(230)
        };
    }

    if truecolor {
        match glyph {
            'K' => Color::Rgb { r: 0x17, g: 0x1a, b: 0x20 },
            'D' => Color::Rgb { r: 0x6f, g: 0x35, b: 0x0c },
            'B' => Color::Rgb { r: 0xa9, g: 0x57, b: 0x0d },
            'G' => Color::Rgb { r: 0xe8, g: 0xa5, b: 0x1f },
            'H' => Color::Rgb { r: 0xff, g: 0xdd, b: 0x67 },
            _ => palette().panel,
        }
    } else {
        match glyph {
            'K' => Color::AnsiValue(235),
            'D' => Color::AnsiValue(94),
            'B' => Color::AnsiValue(136),
            'G' => Color::AnsiValue(178),
            'H' => Color::AnsiValue(221),
            _ => palette().panel,
        }
    }
}

fn draw_colored_line(
    out: &mut std::io::Stdout,
    y: u16,
    text: &str,
    fg: Color,
    bg: Option<Color>,
    bold: bool,
) {
    let _ = queue!(out, MoveTo(0, y));
    if let Some(bg) = bg {
        let _ = queue!(out, SetBackgroundColor(bg));
    }
    let _ = queue!(out, SetForegroundColor(fg));
    if bold {
        let _ = queue!(out, SetAttribute(Attribute::Bold));
    }
    let _ = queue!(out, Print(text));
    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn draw_colored_cell(
    out: &mut std::io::Stdout,
    x: u16,
    y: u16,
    width: usize,
    text: &str,
    fg: Color,
    bg: Color,
    bold: bool,
) {
    let clipped = trim_to_width(text, width);
    let pad = width.saturating_sub(clipped.len());
    let _ = queue!(out, MoveTo(x, y), SetBackgroundColor(bg), SetForegroundColor(fg));
    if bold {
        let _ = queue!(out, SetAttribute(Attribute::Bold));
    }
    let _ = queue!(out, Print(clipped));
    if pad > 0 {
        let _ = queue!(out, Print(" ".repeat(pad)));
    }
    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn draw_colored_segments_cell(
    out: &mut std::io::Stdout,
    x: u16,
    y: u16,
    width: usize,
    segments: &[(String, Color)],
    bg: Color,
) {
    let _ = queue!(out, MoveTo(x, y), SetBackgroundColor(bg));
    let mut remaining = width;

    for (text, fg) in segments {
        if remaining == 0 {
            break;
        }
        let clipped = trim_to_width(text, remaining);
        let taken = clipped.len();
        if taken == 0 {
            continue;
        }
        let _ = queue!(out, SetForegroundColor(*fg), Print(clipped));
        remaining = remaining.saturating_sub(taken);
    }

    if remaining > 0 {
        let _ = queue!(out, SetForegroundColor(palette().text), Print(" ".repeat(remaining)));
    }

    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn color_for_log_line(line: &str) -> Color {
    let lower = line.to_ascii_lowercase();
    if lower.contains("[error") || lower.contains(" error ") {
        palette().err
    } else if lower.contains("[warn") || lower.contains(" warning ") {
        palette().warn
    } else if lower.contains("opoi inference in progress") {
        palette().bright
    } else if lower.contains("inference") {
        palette().mid
    } else if lower.contains("escrowwatcher") && lower.contains("claim accepted") {
        palette().bright
    } else if lower.contains("found a block") || lower.contains("found a share") || lower.contains("block submitted successfully") {
        palette().bright
    } else {
        palette().text
    }
}

fn seconds_since(epoch_s: u64) -> u64 {
    let now = OffsetDateTime::now_utc().unix_timestamp();
    if now <= 0 {
        0
    } else {
        (now as u64).saturating_sub(epoch_s)
    }
}

fn format_duration(seconds: u64) -> String {
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;

    if h > 0 {
        format!("{}h {}m", h, m)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return "...".chars().take(max_chars).collect();
    }

    let keep = max_chars - 3;
    let left = keep / 2;
    let right = keep - left;
    let mut out = String::new();
    out.extend(chars.iter().take(left));
    out.push_str("...");
    out.extend(chars.iter().skip(chars.len().saturating_sub(right)));
    out
}

fn system_memory_summary() -> Option<(u64, u64)> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = None;
    let mut avail_kb = None;

    for line in meminfo.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total_kb = parse_meminfo_kb(v);
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_meminfo_kb(v);
        }
        if total_kb.is_some() && avail_kb.is_some() {
            break;
        }
    }

    let total_kb = total_kb?;
    let avail_kb = avail_kb?;
    let used_kb = total_kb.saturating_sub(avail_kb);
    Some((used_kb / 1024, total_kb / 1024))
}

fn parse_meminfo_kb(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()
        .and_then(|x| x.parse::<u64>().ok())
}

fn load_average_summary() -> Option<(f64, f64, f64)> {
    let loadavg = std::fs::read_to_string("/proc/loadavg").ok()?;
    let mut values = loadavg
        .split_whitespace()
        .take(3)
        .map(|v| v.parse::<f64>().ok());
    let load_1m = values.next().flatten()?;
    let load_5m = values.next().flatten()?;
    let load_15m = values.next().flatten()?;
    Some((load_1m, load_5m, load_15m))
}

fn handle_input(
    ui_state: &UiState,
    shutdown_requested: &AtomicBool,
    block_celebrations_enabled: &mut bool,
) -> bool {
    while event::poll(Duration::from_millis(0)).unwrap_or(false) {
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            let mut out = stdout();
            let _ = execute!(out, Show, LeaveAlternateScreen);
            let _ = disable_raw_mode();
            shutdown_requested.store(true, Ordering::Release);
            return true;
        }
        match key.code {
            KeyCode::Up => ui_state.scroll_up(1),
            KeyCode::Down => ui_state.scroll_down(1),
            KeyCode::PageUp => ui_state.scroll_up(10),
            KeyCode::PageDown => ui_state.scroll_down(10),
            KeyCode::Home => ui_state.scroll_to_top(),
            KeyCode::End => ui_state.scroll_to_live(),
            KeyCode::Char('b') | KeyCode::Char('B') => {
                *block_celebrations_enabled = !*block_celebrations_enabled;
                ui_state.push_log(
                    Level::Info,
                    if *block_celebrations_enabled {
                        "Block celebration animation enabled"
                    } else {
                        "Block celebration animation disabled"
                    },
                );
            }
            _ => {}
        }
    }
    false
}

fn trim_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    for c in s.chars() {
        if out.len() + c.len_utf8() > width {
            break;
        }
        out.push(c);
    }
    out
}

fn format_hashrate(hs: u64) -> String {
    let v = hs as f64;
    if v < 1_000.0 {
        format!("{} hs", hs)
    } else if v < 1_000_000.0 {
        format!("{:.2} Khs", v / 1_000.0)
    } else if v < 1_000_000_000.0 {
        format!("{:.2} Mhs", v / 1_000_000.0)
    } else if v < 1_000_000_000_000.0 {
        format!("{:.2} Ghs", v / 1_000_000_000.0)
    } else {
        format!("{:.2} Ths", v / 1_000_000_000_000.0)
    }
}

fn format_power_draw(power_w: f32) -> String {
    if power_w.is_finite() && power_w >= 0.0 {
        format!("{:.0}W", power_w.round())
    } else {
        "--".to_string()
    }
}

fn format_efficiency(hashrate_hs: u64, power_w: f32) -> String {
    if hashrate_hs == 0 || !power_w.is_finite() || power_w <= 0.0 {
        return "--".to_string();
    }

    let hashrate = hashrate_hs as f64;
    if hashrate <= 0.0 {
        return "--".to_string();
    }

    let efficiency = hashrate / power_w as f64;
    if efficiency <= 0.0 {
        return "--".to_string();
    }

    let (scaled_value, unit) = if efficiency < 1_000.0 {
        (efficiency, "H/W")
    } else if efficiency < 1_000_000.0 {
        (efficiency / 1_000.0, "kH/W")
    } else if efficiency < 1_000_000_000.0 {
        (efficiency / 1_000_000.0, "MH/W")
    } else if efficiency < 1_000_000_000_000.0 {
        (efficiency / 1_000_000_000.0, "GH/W")
    } else {
        (efficiency / 1_000_000_000_000.0, "TH/W")
    };

    format!("{:.0} {}", scaled_value, unit)
}

fn parse_device_id(worker_id: &str) -> Option<u32> {
    worker_id
        .strip_prefix('#')
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u32>().ok())
}

fn short_device_model(worker_id: &str) -> String {
    let raw = worker_id
        .split_once('(')
        .and_then(|(_, rhs)| rhs.strip_suffix(')'))
        .unwrap_or(worker_id)
        .trim();

    let lower = raw.to_ascii_lowercase();
    let cleaned = if lower.contains("nvidia") {
        raw.replace("NVIDIA", "")
            .replace("GeForce", "")
            .replace("Corporation", "")
            .replace("Graphics Device", "")
    } else if lower.contains("amd") || lower.contains("radeon") {
        raw.replace("Advanced Micro Devices, Inc.", "")
            .replace("AMD", "")
            .replace("Radeon", "")
            .replace("Graphics", "")
    } else {
        raw.to_string()
    };

    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn device_brand_color(worker_id: &str) -> Color {
    let lower = worker_id.to_ascii_lowercase();
    if lower.contains("nvidia") {
        palette().ok
    } else if lower.contains("amd") || lower.contains("radeon") {
        palette().err
    } else {
        palette().text
    }
}

fn uppercase_first_char(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_animation_advances_and_expires() {
        assert_eq!(block_animation_frame(Duration::ZERO), Some(0));
        assert_eq!(block_animation_frame(BLOCK_COIN_FRAME_RATE), Some(1));
        assert_eq!(
            block_animation_frame(BLOCK_CELEBRATION_DURATION - Duration::from_millis(1)),
            Some(BLOCK_COIN_FRAME_COUNT - 1),
        );
        assert_eq!(block_animation_frame(BLOCK_CELEBRATION_DURATION), None);
    }

    #[test]
    fn block_coin_art_is_a_complete_indexed_grid() {
        let art = BLOCK_COIN_ART.join("\n");
        assert!(BLOCK_COIN_ART.iter().all(|row| row.chars().count() == 18));
        assert!(['D', 'B', 'G', 'H', 'K'].into_iter().all(|pixel| art.contains(pixel)));
        assert!(!block_coin_halo(0, 0));
        assert!(block_coin_halo(0, 2));
    }
}

