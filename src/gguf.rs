//! Minimal GGUF header parser — the only thing PoM needs from a GGUF: each tensor's name,
//! absolute file offset and exact on-disk byte length, in the file's canonical layout.
//!
//! The chunk enumeration derived from this (name-sorted tensors, `floor(nbytes/32)` 32 B
//! chunks) is CONSENSUS-CRITICAL: it must reproduce byte-for-byte the layout `pom-rt-builder`
//! pinned in `R_T`. Two nets guard any divergence: the recomputed root is checked against the
//! consensus-pinned `R_T`, and the total chunk count against the pinned `n_chunks` — a mismatch
//! refuses to mine instead of producing invalid blocks. Unknown GGML dtypes are a hard error
//! for the same reason (a guessed size would silently shift every later chunk).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{anyhow, bail, Context, Result};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// Exact on-disk size of one tensor: `nelements / block_size * type_size`, the same arithmetic
/// ggml/llama.cpp use. `(block_size, type_size)` per GGML dtype id — only well-known, stable
/// entries; anything else errors out (see module doc).
fn ggml_type_layout(dtype: u32) -> Result<(u64, u64)> {
    Ok(match dtype {
        0 => (1, 4),    // F32
        1 => (1, 2),    // F16
        2 => (32, 18),  // Q4_0
        3 => (32, 20),  // Q4_1
        6 => (32, 22),  // Q5_0
        7 => (32, 24),  // Q5_1
        8 => (32, 34),  // Q8_0
        9 => (32, 36),  // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        20 => (32, 18),   // IQ4_NL
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),   // I8
        25 => (1, 2),   // I16
        26 => (1, 4),   // I32
        27 => (1, 8),   // I64
        28 => (1, 8),   // F64
        30 => (1, 2),   // BF16
        other => bail!(
            "GGUF: unsupported GGML dtype {} — add its (block_size, type_size) to ggml_type_layout",
            other
        ),
    })
}

/// One tensor as laid out in the GGUF data section.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    /// Byte offset of the tensor's data *within the data section* (already alignment-padded
    /// by the writer; absolute file offset = `tensor_data_offset + offset`).
    pub offset: u64,
    /// Exact on-disk byte length of the tensor's (quantized) data.
    pub nbytes: u64,
}

/// Parsed GGUF header: tensor directory + where the data section starts.
pub struct GgufMeta {
    pub tensors: HashMap<String, TensorMeta>,
    pub tensor_data_offset: u64,
}

fn read_u32(f: &mut File) -> Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(f: &mut File) -> Result<String> {
    let len = read_u64(f)?;
    if len > 64 * 1024 * 1024 {
        bail!("GGUF: unreasonable string length {}", len);
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("GGUF: non-UTF-8 string: {}", e))
}

/// Skip (or, for `general.alignment`, capture) one metadata value of GGUF value type `vt`.
/// Returns the value as u64 for the scalar integer types so the caller can read alignment.
fn skip_value(f: &mut File, vt: u32) -> Result<Option<u64>> {
    match vt {
        0 | 1 => { f.seek(SeekFrom::Current(1))?; Ok(None) }         // u8 / i8
        2 | 3 => { f.seek(SeekFrom::Current(2))?; Ok(None) }         // u16 / i16
        4 => Ok(Some(read_u32(f)? as u64)),                          // u32 (alignment candidate)
        5 | 6 => { f.seek(SeekFrom::Current(4))?; Ok(None) }         // i32 / f32
        7 => { f.seek(SeekFrom::Current(1))?; Ok(None) }             // bool
        8 => { read_string(f)?; Ok(None) }                           // string
        9 => {                                                       // array
            let elem_vt = read_u32(f)?;
            let n = read_u64(f)?;
            // Fixed-size element types can be skipped in one seek; strings/nested arrays walk.
            let fixed: Option<i64> = match elem_vt {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4 | 5 | 6 => Some(4),
                10 | 11 | 12 => Some(8),
                _ => None,
            };
            if let Some(sz) = fixed {
                f.seek(SeekFrom::Current(sz * n as i64))?;
            } else {
                for _ in 0..n {
                    skip_value(f, elem_vt)?;
                }
            }
            Ok(None)
        }
        10 => Ok(Some(read_u64(f)?)),                                // u64 (alignment candidate)
        11 | 12 => { f.seek(SeekFrom::Current(8))?; Ok(None) }       // i64 / f64
        other => bail!("GGUF: unknown metadata value type {}", other),
    }
}

impl GgufMeta {
    /// Parse a GGUF v2/v3 header from the start of `f` (seeks to 0 first).
    pub fn read(f: &mut File) -> Result<Self> {
        f.seek(SeekFrom::Start(0))?;
        let magic = read_u32(f).context("GGUF: read magic")?;
        if magic != GGUF_MAGIC {
            bail!("GGUF: bad magic {:#x}", magic);
        }
        let version = read_u32(f)?;
        if !(2..=3).contains(&version) {
            bail!("GGUF: unsupported version {}", version);
        }
        let tensor_count = read_u64(f)?;
        let kv_count = read_u64(f)?;
        if tensor_count > 1_000_000 || kv_count > 1_000_000 {
            bail!("GGUF: unreasonable header counts ({} tensors, {} kvs)", tensor_count, kv_count);
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..kv_count {
            let key = read_string(f)?;
            let vt = read_u32(f)?;
            if let Some(v) = skip_value(f, vt)? {
                if key == "general.alignment" && v > 0 {
                    alignment = v;
                }
            }
        }

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(f)?;
            let n_dims = read_u32(f)?;
            if n_dims > 8 {
                bail!("GGUF: tensor '{}' has {} dims", name, n_dims);
            }
            let mut nelements: u64 = 1;
            let mut ne0: u64 = 1;
            for d in 0..n_dims {
                let ne = read_u64(f)?;
                if d == 0 {
                    ne0 = ne;
                }
                nelements = nelements
                    .checked_mul(ne)
                    .ok_or_else(|| anyhow!("GGUF: tensor '{}' element-count overflow", name))?;
            }
            let dtype = read_u32(f)?;
            let offset = read_u64(f)?;
            let (block, ts) = ggml_type_layout(dtype)
                .with_context(|| format!("tensor '{}'", name))?;
            // ggml lays blocks out per row; sizes divide cleanly only when ne0 is a whole
            // number of blocks (true for every model llama.cpp can write).
            if ne0 % block != 0 {
                bail!("GGUF: tensor '{}' row size {} not divisible by block {}", name, ne0, block);
            }
            let nbytes = nelements / block * ts;
            tensors.insert(name, TensorMeta { offset, nbytes });
        }

        // Data section starts at the first `alignment`-multiple at/after the header end.
        let pos = f.stream_position()?;
        let tensor_data_offset = pos.div_ceil(alignment) * alignment;

        Ok(Self { tensors, tensor_data_offset })
    }

    /// Tensor names in canonical (sorted) order — the layout `R_T` pins.
    pub fn sorted_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Minimum on-disk byte length required for every tensor's data to be present.
    pub fn required_file_len(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| self.tensor_data_offset.saturating_add(t.offset).saturating_add(t.nbytes))
            .max()
            .unwrap_or(self.tensor_data_offset)
    }
}

/// True when `path` is a parseable GGUF whose on-disk size covers every tensor.
/// Used to reuse already-downloaded models without hitting the network again.
pub fn is_complete_file(path: &std::path::Path) -> bool {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let meta = match GgufMeta::read(&mut f) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let file_len = match f.metadata() {
        Ok(m) => m.len(),
        Err(_) => return false,
    };
    file_len >= meta.required_file_len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny in-memory GGUF v3 (1 kv + 2 tensors) and check the directory arithmetic.
    #[test]
    fn parses_minimal_gguf() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        // kv: general.alignment = u32 32
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // value type u32
        buf.extend_from_slice(&32u32.to_le_bytes());
        // tensor 0: "a" F32 [8] @0
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(b'a');
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes());
        // tensor 1: "b" Q4_K [256, 2] @64 (32 B of F32 data padded to alignment 64? — offset is
        // whatever the writer says; the parser must just echo it)
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(b'b');
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
        buf.extend_from_slice(&64u64.to_le_bytes());

        let dir = std::env::temp_dir().join(format!("keryx-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.gguf");
        std::fs::File::create(&path).unwrap().write_all(&buf).unwrap();

        let mut f = File::open(&path).unwrap();
        let meta = GgufMeta::read(&mut f).unwrap();
        assert_eq!(meta.tensors.len(), 2);
        assert_eq!(meta.tensors["a"].nbytes, 32); // 8 × f32
        assert_eq!(meta.tensors["a"].offset, 0);
        assert_eq!(meta.tensors["b"].nbytes, 2 * 144); // 512 elems / 256 × 144
        assert_eq!(meta.tensors["b"].offset, 64);
        // Header length rounded up to the 32 B alignment.
        assert_eq!(meta.tensor_data_offset % 32, 0);
        assert!(meta.tensor_data_offset >= buf.len() as u64);
        assert_eq!(meta.sorted_names(), vec!["a".to_string(), "b".to_string()]);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
