/// IPFS integration — upload inference results and auto-manage kubo daemon.
use std::path::{Path, PathBuf};
use std::time::Duration;

const KUBO_VERSION_FALLBACK: &str = "0.41.0";

fn home_dir() -> PathBuf {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    if let Ok(h) = std::env::var("USERPROFILE") {
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    PathBuf::from(".")
}

fn miner_exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn bundled_ipfs_bin(exe_dir: &Path) -> Option<PathBuf> {
    #[cfg(windows)]
    let names: [&str; 2] = ["ipfs.exe", "ipfs"];
    #[cfg(not(windows))]
    let names: [&str; 1] = ["ipfs"];
    for name in names {
        let p = exe_dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
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
    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("IPFS response parse error: {}", e))?;
    let cid_str = json["Hash"].as_str()
        .ok_or_else(|| anyhow::anyhow!("IPFS response missing Hash field: {:?}", json))?;
    cid_v0_to_multihash(cid_str)
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

/// Check that the IPFS API at `api_url` is reachable.
pub fn is_running(api_url: &str) -> bool {
    let url = format!("{}/api/v0/version", api_url.trim_end_matches('/'));
    ureq::post(&url)
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok()
}

/// Ensure the IPFS daemon is running. If not, download kubo and start it.
/// Non-fatal: logs warnings on failure so the miner can still work (without inference rewards).
pub fn ensure_daemon(api_url: &str) {
    if is_running(api_url) {
        log::info!("IPFS daemon reachable at {}", api_url);
        return;
    }

    // Only auto-manage local daemon.
    if !api_url.contains("127.0.0.1") && !api_url.contains("localhost") {
        log::warn!("IPFS daemon not reachable at {} — inference rewards disabled", api_url);
        return;
    }

    log::info!("IPFS daemon not running — attempting to start kubo...");

    let ipfs_bin = match find_or_download_kubo() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("Could not obtain kubo binary: {} — inference rewards disabled", e);
            return;
        }
    };

    // Init repo if first run.
    let home = home_dir();
    let ipfs_repo = home.join(".ipfs");
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
    match std::process::Command::new(&ipfs_bin)
        .args(["daemon", "--routing=dhtclient"])
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
    {
        Ok(_) => {
            // Wait up to 15 seconds for the API to be ready.
            for _ in 0..15 {
                std::thread::sleep(Duration::from_secs(1));
                if is_running(api_url) {
                    log::info!("IPFS daemon ready");
                    return;
                }
            }
            log::warn!("IPFS daemon started but API not ready — inference rewards may be delayed");
        }
        Err(e) => log::warn!("Failed to start IPFS daemon: {} — inference rewards disabled", e),
    }
}

fn find_or_download_kubo() -> anyhow::Result<PathBuf> {
    let exe_dir = miner_exe_dir();

    // 1. Prefer the Kubo binary shipped next to the miner (release packages).
    if let Some(local) = bundled_ipfs_bin(&exe_dir) {
        log::info!("Using bundled IPFS at {}", local.display());
        return Ok(local);
    }

    // 2. Check PATH.
    if let Ok(out) = std::process::Command::new("ipfs").arg("version").output() {
        if out.status.success() {
            return Ok(PathBuf::from("ipfs"));
        }
    }

    // 3. Download kubo for the current platform into the miner directory.
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

    let bin = bundled_ipfs_bin(&exe_dir).ok_or_else(|| {
        anyhow::anyhow!("ipfs binary missing after extracting {}", archive_name)
    })?;
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
