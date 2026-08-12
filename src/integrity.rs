use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Result};
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

/// Rebuild the legacy Kubo UnixFS CIDv0 root and return its SHA-256 digest.
///
/// Model ids are the digest inside the pinned CIDv0, not a SHA-256 of the raw file bytes. The
/// legacy Kubo importer uses 256 KiB DAG-PB leaves and a balanced layout with 174 links per node.
pub fn unixfs_v0_digest_file(path: &Path, mut progress: impl FnMut(u64, u64)) -> anyhow::Result<[u8; 32]> {
    use ipfs_unixfs::file::adder::{BalancedCollector, Chunker, FileAdderBuilder};

    let file = File::open(path)?;
    let total = file.metadata()?.len();
    let mut adder = FileAdderBuilder::default()
        .with_chunker(Chunker::Size(256 * 1024))
        .with_collector(BalancedCollector::with_branching_factor(174))
        .build();
    let mut reader = BufReader::with_capacity(adder.size_hint(), file);
    let mut last_cid = None;
    let mut done = 0u64;

    loop {
        let input = reader.fill_buf()?;
        if input.is_empty() {
            for (cid, _) in adder.finish() {
                last_cid = Some(cid);
            }
            break;
        }

        let mut consumed_total = 0usize;
        while consumed_total < input.len() {
            let (blocks, consumed) = adder.push(&input[consumed_total..]);
            anyhow::ensure!(consumed > 0, "UnixFS importer made no progress");
            for (cid, _) in blocks {
                last_cid = Some(cid);
            }
            consumed_total += consumed;
        }
        reader.consume(consumed_total);
        done += consumed_total as u64;
        progress(done, total);
    }

    let cid = last_cid.ok_or_else(|| anyhow::anyhow!("UnixFS importer produced no root"))?;
    let digest = cid.hash().digest();
    anyhow::ensure!(digest.len() == 32, "UnixFS root is not SHA-256");
    let mut result = [0u8; 32];
    result.copy_from_slice(digest);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn computes_legacy_unixfs_cid_digest() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"hello world").unwrap();
        file.flush().unwrap();

        let digest = unixfs_v0_digest_file(file.path(), |_, _| {}).unwrap();

        // CIDv0 Qmf412jQZiuVUtdgnB36FXFX7xg5V6KEbSJ4dpQuhkLyfD.
        assert_eq!(hex::encode(digest), "f852c7fa62f971817f54d8a80dcd63fcf7098b3cbde9ae8ec1ee449013ec5db0");
    }

    #[test]
    #[ignore = "requires KERYX_TEST_MODEL and KERYX_TEST_MODEL_ID"]
    fn unixfs_digest_matches_the_pinned_model_id() {
        let path = std::env::var("KERYX_TEST_MODEL").expect("KERYX_TEST_MODEL");
        let expected = std::env::var("KERYX_TEST_MODEL_ID").expect("KERYX_TEST_MODEL_ID");

        let digest = unixfs_v0_digest_file(Path::new(&path), |_, _| {}).unwrap();

        assert_eq!(hex::encode(digest), expected.to_ascii_lowercase());
    }

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
