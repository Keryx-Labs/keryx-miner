/// IPFS integration — upload inference results and auto-manage kubo daemon.
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

const KUBO_VERSION_FALLBACK: &str = "0.41.0";

/// How many seconds `ensure_daemon` waits for the Kubo API to become reachable after
/// spawning the daemon. The wait still fails immediately if the child process exits.
const READINESS_POLL_SECONDS: u32 = 60;

/// Fixed timeout for a single `is_running` probe.
const PROBE_TIMEOUT_SECS: u64 = 2;

/// Sleep interval between readiness probes. Each sleep — and each probe — is capped by
/// the time remaining before the deadline so sleep plus probe time stays within budget.
const READINESS_POLL_INTERVAL_SECS: u64 = 1;

/// Normalize an IPFS API value into a canonical URL form. Scheme-less values such as
/// `127.0.0.1:5001` become `http://127.0.0.1:5001`; explicit `http://` / `https://`
/// schemes (and any other scheme) are preserved as-is. Applied consistently before
/// probing, uploading, and local/remote classification.
fn normalize_api_url(api_url: &str) -> String {
    let trimmed = api_url.trim();
    if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{}", trimmed)
    }
}

/// Upload `text` to the IPFS node at `api_url` and return the raw 34-byte multihash.
/// The multihash format is: [0x12, 0x20, <32-byte sha2-256 digest>].
pub fn upload(text: &str, api_url: &str) -> anyhow::Result<[u8; 34]> {
    let api_url = normalize_api_url(api_url);
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
    let body = response.into_string().map_err(|e| anyhow::anyhow!("IPFS response read error: {}", e))?;
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| anyhow::anyhow!("IPFS response parse error: {}", e))?;
    let cid_str =
        json["Hash"].as_str().ok_or_else(|| anyhow::anyhow!("IPFS response missing Hash field: {:?}", json))?;
    cid_v0_to_multihash(cid_str)
}

/// Decode a base58btc CIDv0 string (e.g. "Qm...") into a 34-byte raw multihash.
fn cid_v0_to_multihash(cid: &str) -> anyhow::Result<[u8; 34]> {
    let bytes = base58btc_decode(cid).ok_or_else(|| anyhow::anyhow!("Invalid base58 CID: {}", cid))?;
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

/// Check that the IPFS API at `api_url` is reachable.
pub fn is_running(api_url: &str) -> bool {
    let api_url = normalize_api_url(api_url);
    probe_running(&api_url, Duration::from_secs(PROBE_TIMEOUT_SECS))
}

/// Probe the IPFS API with an explicit per-request timeout. Callers bound the timeout by
/// the time remaining before their deadline so a slow probe cannot overshoot it.
fn probe_running(api_url: &str, timeout: Duration) -> bool {
    let url = format!("{}/api/v0/version", api_url.trim_end_matches('/'));
    ureq::post(&url).timeout(timeout).call().is_ok()
}

/// Upload `text` to the IPFS node at `api_url`, and on failure recover a local daemon and
/// retry exactly once. Remote endpoints are never auto-managed: their failures propagate
/// unchanged.
pub fn upload_with_recovery(data: &str, api_url: &str) -> anyhow::Result<[u8; 34]> {
    match upload(data, api_url) {
        Ok(cid) => Ok(cid),
        Err(first_err) => match recovery_action(api_url) {
            RecoveryAction::FailImmediately => Err(first_err),
            RecoveryAction::RestartDaemonAndRetry => {
                log::warn!("IPFS upload failed ({}); restarting local daemon and retrying once", first_err);
                if let Err(restore_err) = ensure_daemon(api_url) {
                    return Err(recovery_failed_error(first_err, restore_err));
                }
                upload(data, api_url).map_err(|e| retry_failed_error(first_err, e))
            }
        },
    }
}

fn recovery_failed_error(first_err: anyhow::Error, restore_err: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("IPFS daemon recovery failed (original error: {}): {}", first_err, restore_err)
}

fn retry_failed_error(first_err: anyhow::Error, retry_err: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("IPFS upload failed after daemon recovery (original error: {}): {}", first_err, retry_err)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAction {
    FailImmediately,
    RestartDaemonAndRetry,
}

fn recovery_action(api_url: &str) -> RecoveryAction {
    if is_local_endpoint(api_url) {
        RecoveryAction::RestartDaemonAndRetry
    } else {
        RecoveryAction::FailImmediately
    }
}

/// A URL is local only when its host is exactly a loopback address or `localhost`.
fn is_local_endpoint(api_url: &str) -> bool {
    match url_host(api_url) {
        Some("127.0.0.1") | Some("::1") => true,
        Some(host) => host.eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

fn url_host(url: &str) -> Option<&str> {
    let rest = url.trim();
    let authority = match rest.split_once("://") {
        Some((_, authority)) => authority,
        None => rest,
    };
    let authority = authority.split(|c: char| c == '/' || c == '?' || c == '#').next()?;
    if authority.is_empty() {
        return None;
    }
    if authority.starts_with('[') {
        let mut parts = authority[1..].split(']');
        let host = parts.next()?;
        let suffix = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        return match suffix {
            "" => Some(host),
            rest => {
                let port = rest.strip_prefix(':').unwrap_or(rest);
                if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                Some(host)
            }
        };
    }
    if authority == "::1" {
        return Some("::1");
    }
    let (host, port) = match authority.split_once(':') {
        Some((host, port)) => (host, port),
        None => (authority, ""),
    };
    if !port.is_empty() && !port.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(host)
}

fn resolve_home(candidate_home: Option<&str>, fallback_cwd: &std::path::Path) -> std::path::PathBuf {
    match candidate_home {
        Some(home) if !home.trim().is_empty() => std::path::PathBuf::from(home),
        _ => fallback_cwd.to_path_buf(),
    }
}

fn ipfs_home() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    resolve_home(std::env::var("HOME").ok().as_deref(), &cwd)
}

fn kubo_child_env(home: &std::path::Path) -> Vec<(String, String)> {
    vec![
        ("HOME".to_string(), home.display().to_string()),
        ("IPFS_PATH".to_string(), home.join(".ipfs").display().to_string()),
    ]
}

/// Time remaining before `deadline`, or zero once the deadline has passed.
fn remaining_budget(deadline: std::time::Instant, now: std::time::Instant) -> Duration {
    deadline.saturating_duration_since(now)
}

/// Duration of the next inter-probe sleep: the full interval while budget remains,
/// truncated to whatever budget is left so the sleep alone never overshoots.
fn sleep_duration(deadline: std::time::Instant, now: std::time::Instant) -> Duration {
    remaining_budget(deadline, now).min(Duration::from_secs(READINESS_POLL_INTERVAL_SECS))
}

/// Timeout for the next API probe: bounded by the fixed probe timeout and by the budget
/// remaining *after* the preceding sleep, so sleep plus probe stays within the deadline.
fn probe_timeout(deadline: std::time::Instant, now: std::time::Instant) -> Duration {
    remaining_budget(deadline, now).min(Duration::from_secs(PROBE_TIMEOUT_SECS))
}

/// Serializes local daemon spawn/recovery process-wide so simultaneous failed uploads can
/// never double-spawn Kubo. Acquired only for local (auto-managed) endpoints; remote
/// endpoints never touch it.
fn local_daemon_lock() -> &'static Mutex<()> {
    static LOCAL_DAEMON_RECOVERY_LOCK: Mutex<()> = Mutex::new(());
    &LOCAL_DAEMON_RECOVERY_LOCK
}

/// Lock the process-wide daemon recovery mutex, tolerating a poisoned lock left behind by a
/// previous recovery that panicked while holding it. Poisoning is never re-raised: it
/// becomes a plain recovery error the caller surfaces alongside the original upload error
/// instead of panicking again. Kubo startup is idempotent and the readiness wait fails fast
/// if the API does not come up, so retrying a recovery after a poison is safe.
fn acquire_recovery_lock() -> Result<MutexGuard<'static, ()>, anyhow::Error> {
    lock_recovery(local_daemon_lock())
}

fn lock_recovery(lock: &'static Mutex<()>) -> Result<MutexGuard<'static, ()>, anyhow::Error> {
    lock.lock().map_err(|_| {
        anyhow::anyhow!("IPFS daemon recovery lock is poisoned (a previous recovery panicked while holding it)")
    })
}

/// Under the recovery lock, decide whether a daemon spawn is still needed. The API was
/// unreachable before the lock was acquired; the lock winner may have restored the daemon
/// while this caller waited, so the decision is made on a fresh probe — the waiter reuses
/// the winner's daemon instead of spawning a second one.
fn spawn_still_needed(api_url: &str) -> bool {
    !is_running(api_url)
}

/// Ensure the IPFS daemon at `api_url` is running. Returns only once its API is reachable,
/// waiting up to 60 seconds after spawning the daemon (failing immediately if the child
/// exits). Local daemons are auto-managed; remote endpoints are never auto-managed.
///
/// Local recovery is serialized process-wide: once a caller discovers the API is down it
/// acquires the recovery lock, and under the lock re-probes before spawning. If the daemon
/// was restored by another caller (the lock winner) in the meantime, the waiter reuses it
/// instead of spawning a second Kubo. The lock is held for the entire spawn + readiness
/// wait, so at most one daemon spawn happens at a time.
pub fn ensure_daemon(api_url: &str) -> anyhow::Result<()> {
    let api_url = normalize_api_url(api_url);
    if is_running(&api_url) {
        log::info!("IPFS daemon reachable at {}", api_url);
        return Ok(());
    }

    if !is_local_endpoint(&api_url) {
        return Err(anyhow::anyhow!(
            "IPFS daemon not reachable at {} — remote endpoints are not auto-managed",
            api_url
        ));
    }

    // Serialize spawn/recovery. The lock guard lives for the whole function body below.
    let _recovery_guard = acquire_recovery_lock()?;

    // The API was unreachable before we took the lock; another concurrent caller may have
    // restored the daemon while we waited. Re-probe before spawning so a waiter reuses the
    // daemon the winner started instead of double-spawning.
    if !spawn_still_needed(&api_url) {
        log::info!("IPFS daemon reachable at {}", api_url);
        return Ok(());
    }

    log::info!("IPFS daemon not running — attempting to start kubo...");
    let ipfs_bin = find_or_download_kubo()?;

    // Every Kubo child gets explicit HOME and IPFS_PATH derived from the same home path.
    let home = ipfs_home();
    let ipfs_repo = home.join(".ipfs");
    let child_env = kubo_child_env(&home);
    if !ipfs_repo.exists() {
        log::info!("Initialising IPFS repo at {}", ipfs_repo.display());
        let status = std::process::Command::new(&ipfs_bin)
            .arg("init")
            .envs(child_env.iter().cloned())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to run kubo init: {}", e))?;
        if !status.success() {
            return Err(anyhow::anyhow!("kubo init failed with status {}", status));
        }
    }

    // Keep Kubo output out of the miner terminal but available for diagnostics.
    log::info!("Starting IPFS daemon...");
    let log_dir = home.join(".keryx");
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
    let mut child = std::process::Command::new(&ipfs_bin)
        .args(["daemon", "--routing=dhtclient"])
        .envs(child_env)
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start IPFS daemon: {}", e))?;

    let deadline = std::time::Instant::now() + Duration::from_secs(READINESS_POLL_SECONDS as u64);
    loop {
        let now = std::time::Instant::now();
        if remaining_budget(deadline, now).is_zero() {
            break;
        }
        // Cap the sleep by the time remaining so sleep alone can never overshoot.
        std::thread::sleep(sleep_duration(deadline, now));

        // Probe with a timeout capped by the budget remaining after the sleep, so sleep
        // plus probe time stays within the deadline. If the sleep consumed the budget,
        // the deadline is reached and there is nothing left to probe.
        let now = std::time::Instant::now();
        let budget = remaining_budget(deadline, now);
        if budget.is_zero() {
            break;
        }
        if probe_running(&api_url, probe_timeout(deadline, now)) {
            log::info!("IPFS daemon ready");
            return Ok(());
        }
        if let Some(status) = child.try_wait().map_err(|e| anyhow::anyhow!("Failed to poll IPFS daemon: {}", e))? {
            return Err(anyhow::anyhow!("IPFS daemon exited before the API was ready (status {})", status));
        }
    }

    // Deadline reached. The child may still be running — kill and reap exactly the child
    // this call spawned before reporting the timeout.
    let _ = child.kill();
    let _ = child.wait();
    Err(anyhow::anyhow!("IPFS daemon started but API not ready after {} seconds", READINESS_POLL_SECONDS))
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
        if n == 0 {
            break;
        }
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
            let file_name = std::path::Path::new(&name).file_name().unwrap_or_default().to_os_string();
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

    #[test]
    fn normalize_api_url_adds_http_only_to_scheme_less_values() {
        assert_eq!(normalize_api_url("127.0.0.1:5001"), "http://127.0.0.1:5001");
        assert_eq!(normalize_api_url("localhost:5001"), "http://localhost:5001");
        assert_eq!(normalize_api_url("[::1]:5001"), "http://[::1]:5001");
        assert_eq!(normalize_api_url("  127.0.0.1:5001  "), "http://127.0.0.1:5001");
        // Explicit schemes are preserved as-is.
        assert_eq!(normalize_api_url("http://127.0.0.1:5001"), "http://127.0.0.1:5001");
        assert_eq!(normalize_api_url("https://gateway.keryx.network"), "https://gateway.keryx.network");
        assert_eq!(normalize_api_url("http://localhost"), "http://localhost");
    }

    #[test]
    fn is_local_endpoint_classifies_exact_loopback_hosts() {
        assert!(is_local_endpoint("http://127.0.0.1:5001"));
        assert!(is_local_endpoint("http://localhost:5001"));
        assert!(is_local_endpoint("http://[::1]:5001"));
        assert!(is_local_endpoint("http://::1"));
        assert!(is_local_endpoint("localhost:5001"));
        assert!(!is_local_endpoint("http://::1:5001"));
        assert!(!is_local_endpoint("http://192.168.1.10:5001"));
        assert!(!is_local_endpoint("http://localhost.evil.example:5001"));
        assert!(!is_local_endpoint("http://127.0.0.1.5.nip.io:5001"));
        assert!(!is_local_endpoint("http://myhost/path/localhost"));
        assert!(!is_local_endpoint("http://myhost?redirect=localhost"));
        assert!(!is_local_endpoint("http://localhost:notaport"));
        assert!(!is_local_endpoint("http://[::1]:5001]"));
    }

    #[test]
    fn kubo_child_env_sets_explicit_home_and_derived_ipfs_path() {
        let env = kubo_child_env(std::path::Path::new("/custom/home"));
        let home = env.iter().find(|(k, _)| k == "HOME").expect("HOME env set");
        let ipfs_path = env.iter().find(|(k, _)| k == "IPFS_PATH").expect("IPFS_PATH env set");
        assert_eq!(home.1, "/custom/home");
        assert_eq!(ipfs_path.1, "/custom/home/.ipfs");
    }

    #[test]
    fn resolve_home_preserves_explicit_and_falls_back_for_missing_or_empty() {
        let cwd = std::path::Path::new("/fallback/cwd");
        assert_eq!(resolve_home(Some("/custom/home"), cwd), std::path::PathBuf::from("/custom/home"));
        assert_eq!(resolve_home(None, cwd), cwd.to_path_buf());
        assert_eq!(resolve_home(Some(""), cwd), cwd.to_path_buf());
        assert_eq!(resolve_home(Some("   "), cwd), cwd.to_path_buf());
    }

    #[test]
    fn recovery_action_restarts_local_but_never_remote() {
        assert_eq!(recovery_action("http://127.0.0.1:5001"), RecoveryAction::RestartDaemonAndRetry);
        assert_eq!(recovery_action("http://localhost:5001"), RecoveryAction::RestartDaemonAndRetry);
        assert_eq!(recovery_action("http://10.0.0.5:5001"), RecoveryAction::FailImmediately);
        assert_eq!(recovery_action("https://gateway.keryx.network"), RecoveryAction::FailImmediately);
    }

    #[test]
    fn recovery_errors_retain_original_upload_error_in_plain_display() {
        let err = recovery_failed_error(
            anyhow::anyhow!("IPFS upload failed: boom"),
            anyhow::anyhow!("kubo init failed with status 1"),
        );
        let text = format!("{err}");
        assert!(text.contains("IPFS upload failed: boom"), "{text}");
        assert!(text.contains("IPFS daemon recovery failed"), "{text}");
        assert!(text.contains("kubo init failed with status 1"), "{text}");

        let err =
            retry_failed_error(anyhow::anyhow!("IPFS upload failed: boom"), anyhow::anyhow!("connection refused"));
        let text = format!("{err}");
        assert!(text.contains("IPFS upload failed after daemon recovery"), "{text}");
        assert!(text.contains("original error: IPFS upload failed: boom"), "{text}");
        assert!(text.contains("connection refused"), "{text}");
    }

    #[test]
    fn lock_recovery_acquires_when_uncontended_and_survives_poisoning() {
        static TEST_RECOVERY_LOCK: Mutex<()> = Mutex::new(());

        // Uncontended acquisition succeeds and the guard releases the lock when dropped.
        {
            let guard = lock_recovery(&TEST_RECOVERY_LOCK).expect("uncontended lock acquired");
            drop(guard);
        }

        // Simulate a recovery that panicked while holding the lock, exactly as a panic
        // inside the serialized spawn/readiness section would poison it. The panic is
        // caught here so it does not fail the test; the default panic hook only writes
        // to stderr, which the test harness captures per-test and surfaces only when
        // this test fails. No process-global panic hook swap is needed.
        let outcome = std::panic::catch_unwind(|| {
            let _guard = TEST_RECOVERY_LOCK.lock().expect("test lock acquired");
            panic!("simulated recovery panic");
        });
        assert!(outcome.is_err(), "the simulated panic must poison the lock");

        // Poisoning is surfaced as a plain recovery error — never a panic — and names the
        // poison so `recovery_failed_error` can carry it alongside the original upload error.
        let err = lock_recovery(&TEST_RECOVERY_LOCK).expect_err("poisoned lock must not acquire");
        let text = format!("{err:#}");
        assert!(text.contains("poisoned"), "{text}");
        assert!(text.contains("IPFS daemon recovery lock"), "{text}");
    }

    #[test]
    fn spawn_still_needed_re_probes_so_a_waiter_reuses_the_restored_daemon() {
        // No daemon behind the endpoint: the probe must fail fast and deterministically,
        // so a spawn is still needed. The listener stays bound for the whole probe — it is
        // never dropped into the ephemeral port range, where another test or process could
        // grab the port and make the endpoint appear reachable. Any connection the probe
        // opens is accepted and closed immediately, so the request fails with a closed
        // connection instead of waiting out the probe timeout.
        let dead_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind dead listener");
        let dead_url = format!("http://{}", dead_listener.local_addr().expect("dead local addr"));
        let dead_reaper = std::thread::spawn(move || {
            use std::io::Read as _;
            // Accept and close the probe's connection so it sees a clean EOF instead of
            // waiting out the probe timeout.
            for conn in dead_listener.incoming().take(1) {
                let Ok(mut stream) = conn else { break };
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
            }
        });
        assert!(spawn_still_needed(&dead_url), "dead API must still need a spawn");
        // The probe above already contacted the endpoint, but guarantee the accept is
        // satisfied so the reaper thread cannot outlive the request if it ever did not
        // connect (e.g. the probe failed before the TCP handshake).
        let _ = std::net::TcpStream::connect(dead_url.trim_start_matches("http://"));
        dead_reaper.join().expect("dead reaper thread joined");

        // A restored daemon answers the probe: the waiter must NOT spawn again — it reuses
        // the daemon the lock winner started.
        let live = std::net::TcpListener::bind("127.0.0.1:0").expect("bind live listener");
        let live_url = format!("http://{}", live.local_addr().expect("live local addr"));
        let responder = std::thread::spawn(move || {
            use std::io::{Read as _, Write as _};
            let (mut stream, _) = live.accept().expect("accept probe connection");
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let response = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(response);
        });
        assert!(!spawn_still_needed(&live_url), "restored API must not trigger a spawn");
        responder.join().expect("responder thread joined");
    }

    #[test]
    fn deadline_helpers_cap_sleep_and_probe_by_remaining_budget() {
        let now = std::time::Instant::now();
        let deadline = now + Duration::from_secs(READINESS_POLL_SECONDS as u64);

        // Full budget: sleep uses the full interval, probe uses the fixed probe timeout.
        assert_eq!(remaining_budget(deadline, now), Duration::from_secs(READINESS_POLL_SECONDS as u64));
        assert_eq!(sleep_duration(deadline, now), Duration::from_secs(READINESS_POLL_INTERVAL_SECS));
        assert_eq!(probe_timeout(deadline, now), Duration::from_secs(PROBE_TIMEOUT_SECS));

        // One second left: sleep and probe are capped by the remaining budget so sleep
        // plus probe time cannot overshoot the deadline.
        let near_deadline = now + Duration::from_secs(READINESS_POLL_SECONDS as u64 - 1);
        assert_eq!(sleep_duration(deadline, near_deadline), Duration::from_secs(1));
        assert_eq!(probe_timeout(deadline, near_deadline), Duration::from_secs(1));

        // Half a second left: both truncate to the remaining budget.
        let half_second_left = now + Duration::from_millis(READINESS_POLL_SECONDS as u64 * 1000 - 500);
        assert_eq!(sleep_duration(deadline, half_second_left), Duration::from_millis(500));
        assert_eq!(probe_timeout(deadline, half_second_left), Duration::from_millis(500));

        // Deadline reached or passed: no budget remains for sleep or probe.
        let at_deadline = now + Duration::from_secs(READINESS_POLL_SECONDS as u64);
        let past_deadline = now + Duration::from_secs(READINESS_POLL_SECONDS as u64 + 1);
        assert!(remaining_budget(deadline, at_deadline).is_zero());
        assert!(remaining_budget(deadline, past_deadline).is_zero());
        assert!(sleep_duration(deadline, past_deadline).is_zero());
        assert!(probe_timeout(deadline, past_deadline).is_zero());
    }

    #[test]
    fn readiness_bound_names_the_60_second_limit() {
        assert_eq!(READINESS_POLL_SECONDS, 60);
        let err = anyhow::anyhow!("IPFS daemon started but API not ready after {} seconds", READINESS_POLL_SECONDS);
        let text = format!("{err:#}");
        assert!(text.contains("60 seconds"), "{text}");
    }
}
