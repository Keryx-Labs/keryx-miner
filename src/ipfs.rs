/// IPFS integration — upload inference results and auto-manage kubo daemon.
use crate::stats::MinerStats;
use std::sync::Arc;
use std::time::Duration;

const KUBO_VERSION_FALLBACK: &str = "0.41.0";

/// Prefix written by `short_error("version probe", ..)` into `last_error`.
/// The successful-probe clear matches on this so it can remove its own stale
/// diagnostic without erasing a newer upload failure.
const PROBE_ERROR_MARKER: &str = "version probe: ";

/// Probe the Kubo API at `/api/v0/version` and record the outcome on `stats`.
/// Never panics: any failure only flips the reachability flag and records a short
/// diagnostic in `last_error`. `daemon_managed` is the flag recorded with the
/// probe outcome: `false` for an externally running daemon, `true` once this
/// miner has decided to auto-manage the local daemon.
pub fn probe_version(stats: &MinerStats, api_url: &str, daemon_managed: bool) {
    let url = format!("{}/api/v0/version", api_url.trim_end_matches('/'));
    let outcome = ureq::post(&url).timeout(Duration::from_secs(2)).call();
    match outcome {
        Ok(response) => {
            let version = response.into_string().ok().and_then(|body| parse_version_response(&body));
            stats.set_ipfs_version_probe(true, version, daemon_managed);
            // A successful probe no longer reports the previous probe failure.
            // Only the probe's own diagnostic is cleared (matched by prefix) so
            // a concurrent, newer upload failure in `last_error` is preserved.
            // The check-and-clear happens under the same mutex `upload_with_stats`
            // uses, so no real event is lost to a stale clear.
            let mut slot = stats.last_error_lock();
            if slot.as_deref().is_some_and(|msg| msg.starts_with(PROBE_ERROR_MARKER)) {
                *slot = None;
            }
        }
        Err(e) => {
            stats.set_ipfs_version_probe(false, None, daemon_managed);
            *stats.last_error_lock() = Some(short_error("version probe", &e.to_string()));
        }
    }
}

/// Parse the kubo version from a protocol-faithful `/api/v0/version` JSON body.
fn parse_version_response(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|json| json["Version"].as_str().map(|s| s.to_string()))
}

/// Truncate `error` to at most 160 bytes without splitting a UTF-8 character.
/// `String::truncate` panics when the byte index is not a char boundary, and
/// arbitrary I/O error text (hostnames, paths, headers) can carry multi-byte
/// characters — so walk back to the previous boundary before truncating.
fn short_error(prefix: &str, error: &str) -> String {
    let mut message = format!("{}: {}", prefix, error);
    if message.len() > 160 {
        let mut end = 160;
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        message.truncate(end);
    }
    message
}

/// Record an upload failure that was not observed by `upload_with_stats` itself
/// (e.g. the blocking task failed before the upload ran): short diagnostic,
/// rewards disabled. Mirrors the failure side of `upload_with_stats`.
pub fn record_upload_failure(stats: &MinerStats, error: &str) {
    stats.set_ipfs_upload_outcome(false, None, Some(short_error("upload", error)));
    stats.set_inference_rewards_enabled(false);
}

/// Upload `text` to the IPFS node at `api_url` and return the raw 34-byte multihash.
/// The multihash format is: [0x12, 0x20, <32-byte sha2-256 digest>].
pub fn upload(text: &str, api_url: &str) -> anyhow::Result<[u8; 34]> {
    let url = format!("{}/api/v0/add?pin=true&quieter=true", api_url.trim_end_matches('/'));
    let boundary = "keryxboundary1234567890";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"result.txt\"\r\nContent-Type: text/plain\r\n\r\n{text}\r\n--{boundary}--\r\n",
        boundary = boundary,
        text = text,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let response = ureq::post(&url)
        .set("Content-Type", &content_type)
        .timeout(Duration::from_secs(30))
        .send_bytes(body.as_bytes())
        .map_err(|e| anyhow::anyhow!("IPFS upload failed: {}", e))?;
    let body = response.into_string()
        .map_err(|e| anyhow::anyhow!("IPFS response read error: {}", e))?;
    let cid_str = parse_add_response(&body)?;
    cid_v0_to_multihash(&cid_str)
}

/// Parse the CIDv0 string from a protocol-faithful `/api/v0/add` JSON body
/// (`{"Name": "...", "Hash": "Qm...", "Size": "..."}`).
/// Errors never embed the response body: a broken gateway or proxy may echo the
/// uploaded inference text (or other secrets) back inside it.
fn parse_add_response(body: &str) -> anyhow::Result<String> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| anyhow::anyhow!("IPFS response parse error: {}", e))?;
    json["Hash"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("IPFS response missing Hash field"))
}

/// Upload wrapper that records the outcome on the shared stats: CIDv0 string on
/// success, short diagnostic on failure. The uploaded text never reaches stats.
/// The inference-rewards flag follows the actual upload outcome: a successful
/// upload (re)enables rewards, a failed upload disables them, so a later upload
/// loss turns rewards off even while the daemon stays reachable, and a
/// successful recovery turns them back on.
pub fn upload_with_stats(text: &str, api_url: &str, stats: &MinerStats) -> anyhow::Result<[u8; 34]> {
    match upload(text, api_url) {
        Ok(cid) => {
            let cid_str = cid_v0_string(&cid);
            stats.set_ipfs_upload_outcome(true, Some(cid_str), None);
            stats.set_inference_rewards_enabled(true);
            Ok(cid)
        }
        Err(e) => {
            stats.set_ipfs_upload_outcome(false, None, Some(short_error("upload", &e.to_string())));
            stats.set_inference_rewards_enabled(false);
            Err(e)
        }
    }
}

/// Encode a 34-byte raw multihash as a base58btc CIDv0 string (e.g. "Qm...").
/// Prefer the same encoding the inference crate uses for wire-format CIDs.
fn cid_v0_string(cid: &[u8; 34]) -> String {
    keryx_inference::AiResponsePayload::new([0u8; 32], 0, *cid, 0).cid_v0()
}

/// Decode a base58btc CIDv0 string (e.g. "Qm...") into a 34-byte raw multihash.
fn cid_v0_to_multihash(cid: &str) -> anyhow::Result<[u8; 34]> {
    let bytes = base58btc_decode(cid)
        .ok_or_else(|| anyhow::anyhow!("Invalid base58 CID: {}", cid))?;
    if bytes.len() != 34 || bytes[0] != 0x12 || bytes[1] != 0x20 {
        return Err(anyhow::anyhow!("CID is not a CIDv0 sha2-256 multihash: {}", cid));
    }
    let mut out = [0u8; 34];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn base58btc_decode(input: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let mut table = [0xFF_u8; 128];
    for (i, &c) in ALPHABET.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    let mut result: Vec<u8> = vec![0];
    for &c in input.as_bytes() {
        if c >= 128 || table[c as usize] == 0xFF {
            return None;
        }
        let mut carry = table[c as usize] as u32;
        for byte in result.iter_mut() {
            carry += (*byte as u32) * 58;
            *byte = (carry & 0xFF) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            result.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }
    let leading_zeros = input.bytes().take_while(|&b| b == b'1').count();
    let mut out = vec![0u8; leading_zeros];
    out.extend(result.iter().rev());
    Some(out)
}

/// Check whether the IPFS API at `api_url` is reachable, recording the probe
/// outcome on `stats` as an external daemon (not auto-managed by this miner).
pub fn probe(stats: &MinerStats, api_url: &str) -> bool {
    probe_version(stats, api_url, false);
    stats.is_ipfs_reachable()
}

/// Check reachability of a daemon this miner auto-manages. Identical to `probe`
/// except every probe records the `daemon_managed` flag as `true`, so a snapshot
/// taken while the daemon is still starting up never flickers it back to `false`.
pub fn probe_managed(stats: &MinerStats, api_url: &str) -> bool {
    probe_version(stats, api_url, true);
    stats.is_ipfs_reachable()
}

/// Ensure the IPFS daemon is running. If not, download kubo and start it.
/// Non-fatal: logs warnings on failure so the miner can still work (without inference rewards).
/// Records reachability/version/daemon-managed state on `stats` at every real
/// `/api/v0/version` probe.
pub fn ensure_daemon(api_url: String, stats: Arc<MinerStats>) {
    if probe(&stats, &api_url) {
        log::info!("IPFS daemon reachable at {}", api_url);
        stats.set_inference_rewards_enabled(true);
        return;
    }

    // Only auto-manage local daemon.
    if !api_url.contains("127.0.0.1") && !api_url.contains("localhost") {
        log::warn!("IPFS daemon not reachable at {} — inference rewards disabled", api_url);
        stats.set_inference_rewards_enabled(false);
        return;
    }

    log::info!("IPFS daemon not running — attempting to start kubo...");

    let ipfs_bin = match find_or_download_kubo() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("Could not obtain kubo binary: {} — inference rewards disabled", e);
            stats.set_ipfs_version_probe(false, None, true);
            stats.set_inference_rewards_enabled(false);
            return;
        }
    };

    // Init repo if first run.
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let ipfs_repo = std::path::PathBuf::from(&home).join(".ipfs");
    if !ipfs_repo.exists() {
        log::info!("Initialising IPFS repo...");
        let _ = std::process::Command::new(&ipfs_bin)
            .arg("init")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    // Start daemon in background, redirecting output to a log file so
    // mDNS/discovery noise does not pollute the miner terminal while
    // keeping Kubo logs accessible for inference debugging.
    log::info!("Starting IPFS daemon...");
    let log_dir = std::path::PathBuf::from(&home).join(".keryx");
    let _ = std::fs::create_dir_all(&log_dir);
    let kubo_log = log_dir.join("kubo.log");
    let (stdout, stderr) = match std::fs::OpenOptions::new().create(true).append(true).open(&kubo_log) {
        Ok(f) => match f.try_clone() {
            Ok(f2) => {
                log::info!("Kubo output redirected to {}", kubo_log.display());
                (std::process::Stdio::from(f), std::process::Stdio::from(f2))
            }
            Err(_) => (std::process::Stdio::null(), std::process::Stdio::null()),
        },
        Err(_) => (std::process::Stdio::null(), std::process::Stdio::null()),
    };
    match std::process::Command::new(&ipfs_bin)
        .args(["daemon", "--routing=dhtclient"])
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
    {
        Ok(_) => {
            // Wait up to 15 seconds for the API to be ready. Every probe in this
            // loop records the managed flag, so the startup window never reports
            // `daemon_managed=false` for a daemon this miner is bringing up.
            for _ in 0..15 {
                std::thread::sleep(Duration::from_secs(1));
                if probe_managed(&stats, &api_url) {
                    log::info!("IPFS daemon ready");
                    break;
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to start IPFS daemon: {} — inference rewards disabled", e);
            stats.set_ipfs_version_probe(false, None, true);
            stats.set_inference_rewards_enabled(false);
            return;
        }
    }

    // Auto-managed local daemon: preserve ownership while reflecting the final probe state.
    let reachable = stats.is_ipfs_reachable();
    stats.set_ipfs_version_probe(reachable, stats.kubo_version(), true);
    stats.set_inference_rewards_enabled(reachable);
    if !reachable {
        log::warn!("IPFS daemon started but API not ready — inference rewards may be delayed");
    }
}

fn find_or_download_kubo() -> anyhow::Result<std::path::PathBuf> {
    // 1. Check PATH.
    if let Ok(out) = std::process::Command::new("ipfs").arg("version").output() {
        if out.status.success() {
            return Ok(std::path::PathBuf::from("ipfs"));
        }
    }

    // 2. Check next to the miner executable.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let local_bin = exe_dir.join("ipfs");
    if local_bin.exists() {
        return Ok(local_bin);
    }

    // 3. Download kubo for the current platform.
    let version = fetch_latest_kubo_version();
    let (os, arch) = detect_platform()?;
    let archive_ext = if cfg!(target_os = "windows") { "zip" } else { "tar.gz" };
    let archive_name = format!("kubo_v{}_{}-{}.{}", version, os, arch, archive_ext);
    let url = format!("https://dist.ipfs.tech/kubo/v{}/{}", version, archive_name);
    let archive_path = exe_dir.join(&archive_name);

    log::info!("Downloading kubo {}...", version);
    download_file(&url, &archive_path)?;

    extract_ipfs_binary(&archive_path, &exe_dir)?;
    std::fs::remove_file(&archive_path).ok();

    let bin = exe_dir.join(if cfg!(target_os = "windows") { "ipfs.exe" } else { "ipfs" });
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bin)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bin, perms)?;
    }

    log::info!("kubo installed at {}", bin.display());
    Ok(bin)
}

fn fetch_latest_kubo_version() -> String {
    let result = ureq::get("https://api.github.com/repos/ipfs/kubo/releases/latest")
        .set("User-Agent", "keryx-miner")
        .timeout(Duration::from_secs(10))
        .call();
    match result {
        Ok(resp) => {
            if let Ok(body) = resp.into_string() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(tag) = json["tag_name"].as_str() {
                        let version = tag.trim_start_matches('v').to_string();
                        log::info!("Latest kubo version: {}", version);
                        return version;
                    }
                }
            }
        }
        Err(e) => log::warn!("Could not fetch latest kubo version: {} — using fallback {}", e, KUBO_VERSION_FALLBACK),
    }
    KUBO_VERSION_FALLBACK.to_string()
}

fn detect_platform() -> anyhow::Result<(&'static str, &'static str)> {
    let os = match std::env::consts::OS {
        "linux" => "linux",
        "macos" => "darwin",
        "windows" => "windows",
        other => return Err(anyhow::anyhow!("Unsupported OS: {}", other)),
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        other => return Err(anyhow::anyhow!("Unsupported arch: {}", other)),
    };
    Ok((os, arch))
}

fn download_file(url: &str, dest: &std::path::Path) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    let response = ureq::get(url)
        .timeout(Duration::from_secs(300))
        .call()
        .map_err(|e| anyhow::anyhow!("Download {}: {}", url, e))?;
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)?;
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
    }
    Ok(())
}

fn extract_ipfs_binary(archive: &std::path::Path, dest_dir: &std::path::Path) -> anyhow::Result<()> {
    if archive.extension().and_then(|e| e.to_str()) == Some("zip") {
        let file = std::fs::File::open(archive)?;
        let mut zip = zip::ZipArchive::new(file)?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            let name = entry.name().to_string();
            let file_name = std::path::Path::new(&name)
                .file_name()
                .unwrap_or_default()
                .to_os_string();
            if file_name == "ipfs.exe" {
                let mut out = std::fs::File::create(dest_dir.join(&file_name))?;
                std::io::copy(&mut entry, &mut out)?;
                return Ok(());
            }
        }
        return Err(anyhow::anyhow!("ipfs.exe not found in kubo zip archive"));
    }

    let file = std::fs::File::open(archive)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let file_name = path.file_name().unwrap_or_default().to_os_string();
        if file_name == "ipfs" {
            entry.unpack(dest_dir.join(file_name))?;
            return Ok(());
        }
    }
    Err(anyhow::anyhow!("ipfs binary not found in kubo archive"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Protocol-faithful `/api/v0/add`/`/api/v0/version` bodies used below.
    const VERSION_BODY: &str =
        r#"{"Version":"0.41.0","Commit":"f3bbdf958","Repo":"17","System":"amd64/linux","Golang":"go1.24.3"}"#;
    const ADD_BODY: &str =
        r#"{"Name":"result.txt","Hash":"QmVC672p5SEjPGkMb9ztkVjxpYYmfG8e3oV39wAjzKtgU8","Size":"11"}"#;
    const ADD_CID: &str = "QmVC672p5SEjPGkMb9ztkVjxpYYmfG8e3oV39wAjzKtgU8";

    /// Bind an ephemeral port and serve one HTTP response with the given JSON
    /// body for whichever `/api/v0/*` path the client probes. Fully offline:
    /// exercises the real ureq client against protocol-faithful Kubo responses.
    fn serve_kubo(json_body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept test client");
            let mut buf = [0u8; 8192];
            let mut total = 0;
            // Read until end of headers so ureq's request (and multipart body
            // for /add) is fully consumed before we answer.
            let header_end = loop {
                let n = stream.read(&mut buf[total..]).expect("read request");
                if n == 0 {
                    break total;
                }
                total += n;
                if let Some(off) = buf[..total].windows(4).position(|w| w == b"\r\n\r\n") {
                    break off;
                }
            };
            // Parse Content-Length from the header section and own it as a
            // plain usize so no borrow of `buf` outlives the read below.
            let content_len = {
                let head = String::from_utf8_lossy(&buf[..header_end]);
                head.lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().to_string()))
                    .and_then(|len_str| len_str.parse::<usize>().ok())
            };
            // Consume exactly header + delimiter + body: body bytes that
            // arrived with the headers are already counted in `total`, so
            // only the remainder is read here.
            if let Some(len) = content_len {
                let needed = header_end + 4 + len;
                while total < needed {
                    let n = stream.read(&mut buf[total..]).expect("read body");
                    if n == 0 {
                        break;
                    }
                    total += n;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                json_body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(json_body.as_bytes()).unwrap();
        });
        drop(handle); // detached: the response is written before the test proceeds
        format!("http://{}", addr)
    }

    /// Real `/api/v0/version` bodies carry "Version", "Commit", "Repo", "System"...
    #[test]
    fn parses_kubo_version_response() {
        assert_eq!(parse_version_response(VERSION_BODY).as_deref(), Some("0.41.0"));
    }

    #[test]
    fn rejects_malformed_version_response() {
        assert_eq!(parse_version_response("not json"), None);
        assert_eq!(parse_version_response(r#"{"Commit":"x"}"#), None);
        assert_eq!(parse_version_response(""), None);
    }

    #[test]
    fn probe_records_version_from_a_real_version_response() {
        let url = serve_kubo(VERSION_BODY);
        let stats = crate::stats::MinerStats::new(false);
        assert!(probe(&stats, &url));
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert_eq!(snapshot.ipfs.kubo_version.as_deref(), Some("0.41.0"));
        assert_eq!(snapshot.ipfs.last_error, None);
    }

    #[test]
    fn probe_failure_stays_observable_and_non_panicking() {
        // Nothing listens on this port (TCP_NODELAY loopback, no service):
        // the probe must fail without panicking and record the diagnostic.
        let url = "http://127.0.0.1:1";
        let stats = crate::stats::MinerStats::new(false);
        assert!(!probe(&stats, url));
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.last_error.is_some());
        assert_eq!(snapshot.ipfs.kubo_version, None);
    }

    /// Managed startup probes must never flicker `daemon_managed` back to false:
    /// from the moment the miner decides to auto-manage, every probe records the
    /// managed flag, so a snapshot taken mid-startup still reports `true`.
    #[test]
    fn managed_startup_probes_never_flicker_daemon_managed_false() {
        let stats = crate::stats::MinerStats::new(false);

        // Daemon still starting: the managed probe fails but keeps the flag true.
        assert!(!probe_managed(&stats, "http://127.0.0.1:1"));
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.daemon_managed);

        // Next tick the API answers: still managed, version recorded.
        let url = serve_kubo(VERSION_BODY);
        assert!(probe_managed(&stats, &url));
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.daemon_managed);
        assert_eq!(snapshot.ipfs.kubo_version.as_deref(), Some("0.41.0"));

        // An external daemon probe still records the non-managed flag.
        assert!(!probe(&stats, "http://127.0.0.1:1"));
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.daemon_managed);
    }

    /// A protocol-faithful `/api/v0/add` response with a CIDv0 sha2-256 hash
    /// (QmVC... = sha2-256 of a fixed test payload, so it passes CID validation).
    #[test]
    fn parses_add_response_cidv0() {
        let cid = parse_add_response(ADD_BODY).expect("valid add response parses");
        assert_eq!(cid, ADD_CID);
    }

    #[test]
    fn rejects_add_response_without_hash() {
        assert!(parse_add_response(r#"{"Name":"result.txt","Size":"11"}"#).is_err());
        assert!(parse_add_response("not json").is_err());
        assert!(parse_add_response("").is_err());
    }

    /// Parse errors must never embed the response body: a broken gateway or
    /// proxy may echo the uploaded inference text (or other secrets) inside it.
    #[test]
    fn add_parse_errors_never_echo_the_response_body() {
        let secret = "super secret inference text echoed by a broken gateway";
        let err = parse_add_response(secret).expect_err("non-JSON body is rejected");
        assert!(err.to_string().contains("parse error"));
        assert!(!err.to_string().contains(secret));

        let err = parse_add_response(r#"{"Name":"result.txt","Size":"11","Prompt":"super secret prompt"}"#)
            .expect_err("missing Hash is rejected");
        assert!(err.to_string().contains("missing Hash"));
        assert!(!err.to_string().contains("super secret prompt"));
    }

    #[test]
    fn upload_records_cidv0_and_serialized_stats_omit_text() {
        let url = serve_kubo(ADD_BODY);
        let stats = crate::stats::MinerStats::new(false);
        let cid = upload_with_stats("super secret inference text", &url, &stats).expect("upload succeeds");
        assert_eq!(cid.len(), 34);
        assert_eq!(cid[0], 0x12);
        assert_eq!(cid[1], 0x20);
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.last_upload_success);
        assert_eq!(snapshot.ipfs.last_upload_cid.as_deref(), Some(ADD_CID));
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(serialized.contains(ADD_CID));
        assert!(!serialized.contains("super secret inference text"));
        assert!(!serialized.contains("inference text"));
    }

    /// Inference rewards follow actual upload outcomes: a failed upload turns
    /// them off, a later success turns them back on, and a fresh upload loss
    /// disables them again.
    #[test]
    fn upload_outcome_drives_inference_rewards() {
        let stats = crate::stats::MinerStats::new(false);
        assert!(!stats.snapshot().ipfs.inference_rewards_enabled);

        // Upload loss: nothing listens on this loopback port, so the upload fails
        // and rewards must go off (even though no probe ran here).
        let dead_url = "http://127.0.0.1:1";
        assert!(upload_with_stats("super secret", dead_url, &stats).is_err());
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.inference_rewards_enabled);
        assert!(!snapshot.ipfs.last_upload_success);
        assert!(snapshot.ipfs.last_error.as_deref().is_some_and(|e| e.starts_with("upload:")));

        // Recovery: a real /api/v0/add success re-enables rewards.
        let url = serve_kubo(ADD_BODY);
        upload_with_stats("super secret", &url, &stats).expect("upload succeeds");
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.inference_rewards_enabled);
        assert!(snapshot.ipfs.last_upload_success);
        assert_eq!(snapshot.ipfs.last_error, None);

        // Later upload loss disables rewards again...
        assert!(upload_with_stats("super secret", dead_url, &stats).is_err());
        assert!(!stats.snapshot().ipfs.inference_rewards_enabled);

        // ...and one more recovery success re-enables them.
        let url = serve_kubo(ADD_BODY);
        upload_with_stats("super secret", &url, &stats).expect("upload succeeds");
        assert!(stats.snapshot().ipfs.inference_rewards_enabled);
    }

    /// The grpc JoinError path (spawn_blocking task never ran) records the
    /// failure via `record_upload_failure`: short diagnostic, rewards disabled,
    /// no uploaded text ever stored.
    #[test]
    fn record_upload_failure_disables_rewards_with_short_diagnostic() {
        let stats = crate::stats::MinerStats::new(false);

        // Pre-condition: rewards were enabled by an earlier successful upload.
        let url = serve_kubo(ADD_BODY);
        upload_with_stats("super secret", &url, &stats).expect("upload succeeds");
        assert!(stats.snapshot().ipfs.inference_rewards_enabled);

        // The blocking task failed before the upload ran — record it exactly the
        // way the grpc JoinError arm does.
        let join_error = "task panicked while running";
        crate::ipfs::record_upload_failure(&stats, join_error);
        let snapshot = stats.snapshot();
        assert!(!snapshot.ipfs.inference_rewards_enabled);
        assert!(!snapshot.ipfs.last_upload_success);
        assert!(snapshot.ipfs.last_error.as_deref().is_some_and(|e| e.starts_with("upload:")));
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(!serialized.contains("super secret"));
    }

    /// Arbitrary UTF-8 in error text must never panic the 160-byte truncation:
    /// the result stays on a char boundary and within the size budget.
    #[test]
    fn short_error_is_utf8_safe_and_bounded() {
        // Pure ASCII: truncated to exactly 160 bytes.
        let long_ascii = "a".repeat(500);
        let s = short_error("upload", &long_ascii);
        assert_eq!(s.len(), 160);
        assert!(s.starts_with("upload: "));
        assert!(s.is_char_boundary(s.len()));

        // Multi-byte text where byte 160 lands inside a character: must not
        // panic, must stay <= 160 bytes, and must remain valid UTF-8.
        let long_utf8 = "雪".repeat(200); // 3 bytes per char, 600 bytes total
        let s = short_error("version probe", &long_utf8);
        assert!(s.len() <= 160);
        assert!(s.is_char_boundary(s.len()));
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
        assert!(s.starts_with("version probe: "));

        // A boundary-exact 160-byte message with multi-byte characters.
        let mixed = format!("{}", "é".repeat(200)); // 2 bytes per char, 400 bytes
        let s = short_error("upload", &mixed);
        assert!(s.len() <= 160);
        assert!(s.is_char_boundary(s.len()));

        // Short messages are passed through untouched.
        assert_eq!(short_error("upload", "boom"), "upload: boom");
    }

    /// A successful version probe clears the stale probe diagnostic but must
    /// not erase a newer, unrelated upload failure recorded concurrently.
    #[test]
    fn successful_probe_clears_stale_probe_error_only() {
        let stats = crate::stats::MinerStats::new(false);

        // Failed probe: records its own diagnostic.
        assert!(!probe(&stats, "http://127.0.0.1:1"));
        assert!(stats.snapshot().ipfs.last_error.as_deref().unwrap().starts_with("version probe:"));

        // Recovery: the next probe succeeds and the stale probe error is gone.
        let url = serve_kubo(VERSION_BODY);
        assert!(probe(&stats, &url));
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert_eq!(snapshot.ipfs.last_error, None);

        // A probe failure followed by an upload failure, then a successful
        // probe: the upload failure (newest real event) must survive the clear.
        assert!(!probe(&stats, "http://127.0.0.1:1"));
        assert!(upload_with_stats("super secret", "http://127.0.0.1:1", &stats).is_err());
        let url = serve_kubo(VERSION_BODY);
        assert!(probe(&stats, &url));
        let snapshot = stats.snapshot();
        assert!(snapshot.ipfs.api_reachable);
        assert!(snapshot.ipfs.last_error.as_deref().is_some_and(|e| e.starts_with("upload:")));
    }
}
