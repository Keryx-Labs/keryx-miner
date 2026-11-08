//! `KRXREC01` compact trace records, kept by a pool for its audits:
//! `"KRXREC01" | u32 LE header_len | borsh(RecordHeader) | borsh(ResponseTrace) | blake3(header || trace)`.

use crate::trace::ResponseTrace;
use borsh::BorshDeserialize;

pub const RECORD_MAGIC: &[u8; 8] = b"KRXREC01";

/// ABI of the trace engine this build loads.
pub const ENGINE_ABI: u32 = crate::llama_engine::ABI as u32;

#[derive(Debug, Clone, PartialEq, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct RecordHeader {
    pub engine_abi: u32,
    pub patch_set: [u8; 32],
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub max_tokens: u32,
    pub gpu_model: String,
    pub sm_count: u32,
    pub driver_version: String,
}

/// BLAKE3 over the llama.cpp patch series the engine is built with, in order.
pub fn patch_set_hash() -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for p in [
        &include_bytes!("../tools/keryx-llama/patches/0001-canonical-trace.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0002-sequence-leaves.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0003-fattn-tile-f32.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0004-rope-f64-angle.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0005-attn-lse-capture.patch")[..],
    ] {
        h.update(&(p.len() as u64).to_le_bytes());
        h.update(p);
    }
    *h.finalize().as_bytes()
}

/// GPU model name, SM count and driver version of CUDA device `gpu` (empty / 0 when unknown).
pub fn gpu_identity(gpu: u32) -> (String, u32, String) {
    use cudarc::driver::{result, sys};
    let dev = || {
        result::init().ok()?;
        result::device::get(gpu as i32).ok()
    };
    let name = dev().and_then(|d| result::device::get_name(d).ok()).unwrap_or_default();
    let sm = dev()
        .and_then(|d| {
            unsafe { result::device::get_attribute(d, sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) }.ok()
        })
        .map_or(0, |n| n.max(0) as u32);
    let driver = nvml_wrapper::Nvml::init().ok().and_then(|n| n.sys_driver_version().ok()).unwrap_or_default();
    (name, sm, driver)
}

/// Header of a record made on CUDA device `gpu`.
pub fn header(model_id: [u8; 32], tier: u8, request_hash: [u8; 32], max_tokens: u32, gpu: u32) -> RecordHeader {
    let (gpu_model, sm_count, driver_version) = gpu_identity(gpu);
    RecordHeader {
        engine_abi: ENGINE_ABI,
        patch_set: patch_set_hash(),
        model_id,
        tier,
        request_hash,
        max_tokens,
        gpu_model,
        sm_count,
        driver_version,
    }
}

/// Serializes a record.
pub fn encode_record(header: &RecordHeader, trace: &ResponseTrace) -> Vec<u8> {
    let h = borsh::to_vec(header).expect("borsh header");
    let b = borsh::to_vec(trace).expect("borsh trace");
    let mut hasher = blake3::Hasher::new();
    hasher.update(&h);
    hasher.update(&b);
    let mut out = Vec::with_capacity(8 + 4 + h.len() + b.len() + 32);
    out.extend_from_slice(RECORD_MAGIC);
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&b);
    out.extend_from_slice(hasher.finalize().as_bytes());
    out
}

/// Parses and checks a record.
pub fn decode_record(blob: &[u8]) -> Option<(RecordHeader, ResponseTrace)> {
    if blob.len() < 8 + 4 + 32 || &blob[..8] != RECORD_MAGIC {
        return None;
    }
    let hl = u32::from_le_bytes(blob[8..12].try_into().ok()?) as usize;
    let rest = blob.get(12..)?;
    if hl > rest.len() {
        return None;
    }
    let (hb, rest) = rest.split_at(hl);
    let (bb, digest) = rest.split_at(rest.len().checked_sub(32)?);
    let mut hasher = blake3::Hasher::new();
    hasher.update(hb);
    hasher.update(bb);
    if hasher.finalize().as_bytes() != digest {
        return None;
    }
    Some((RecordHeader::try_from_slice(hb).ok()?, ResponseTrace::try_from_slice(bb).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> RecordHeader {
        RecordHeader {
            engine_abi: ENGINE_ABI,
            patch_set: patch_set_hash(),
            model_id: [1; 32],
            tier: 2,
            request_hash: [3; 32],
            max_tokens: 512,
            gpu_model: "NVIDIA GeForce RTX 3090".into(),
            sm_count: 82,
            driver_version: "595.45.04".into(),
        }
    }

    #[test]
    fn a_record_round_trips() {
        let trace = ResponseTrace::test_sample();
        let blob = encode_record(&sample_header(), &trace);
        assert_eq!(&blob[..8], RECORD_MAGIC);
        let (h, t) = decode_record(&blob).expect("decodes");
        assert_eq!(h, sample_header());
        assert_eq!(borsh::to_vec(&t).unwrap(), borsh::to_vec(&trace).unwrap());
    }

    #[test]
    fn a_damaged_record_is_refused() {
        let blob = encode_record(&sample_header(), &ResponseTrace::test_sample());
        for i in [0, 9, 20, blob.len() - 40, blob.len() - 1] {
            let mut bad = blob.clone();
            bad[i] ^= 1;
            assert!(decode_record(&bad).is_none(), "byte {i}");
        }
        assert!(decode_record(&blob[..blob.len() - 1]).is_none());
        let mut huge = blob.clone();
        huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_record(&huge).is_none());
    }
}
