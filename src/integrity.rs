use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Result};
use std::path::Path;

/// SHA-256 of a file's bytes, reporting (done, total) as it streams.
pub fn sha256_file(path: &Path, mut progress: impl FnMut(u64, u64)) -> Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    let mut done = 0u64;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        done += read as u64;
        progress(done, total);
    }

    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn hashes_file_bytes_and_reports_progress() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"abc").unwrap();
        file.flush().unwrap();
        let mut progress = Vec::new();

        let digest = sha256_file(file.path(), |done, total| progress.push((done, total))).unwrap();

        assert_eq!(hex::encode(digest), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(progress.last(), Some(&(3, 3)));
    }
}
