use std::collections::HashMap;
use anyhow::{anyhow, Result};
use crate::gguf::{GgufMeta, TensorMeta};

/// One fixed-size, layer-aligned slice of a model's canonical GGUF. Fully determined by the
/// GGUF's own tensor directory plus `target_bytes` — no operator choice enters the boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardManifest {
    /// Position of this shard among its siblings, 0-based, in ascending layer order.
    pub index: u16,
    /// Transformer layer range this shard owns, `[layer_lo, layer_hi)`. Non-layer tensors are
    /// attached to shard 0 (input side) or the last shard (output side) — see `pack_shards`.
    pub layer_lo: u32,
    pub layer_hi: u32,
    /// Tensor names in this shard, already in canonical (sorted) order.
    pub tensor_names: Vec<String>,
}

/// Split a GGUF's tensors into fixed-byte-target, layer-aligned shards.
///
/// Grouping rule: tensors named `blk.<N>.*` belong to layer `N`; everything else is
/// "non-layer" (`token_embd.*` → shard 0, `output*` → last shard). Layers are walked in
/// numeric order and packed greedily: a shard closes when adding the next layer would push it
/// over `target_bytes`, unless the shard has no layers yet (a single layer bigger than
/// `target_bytes` is a hard error — the caller must raise the target). Layer ranges are
/// contiguous and cover every layer in the GGUF exactly once.
pub fn pack_shards(meta: &GgufMeta, target_bytes: u64) -> Result<Vec<ShardManifest>> {
    if target_bytes == 0 {
        return Err(anyhow!("shard: target_bytes must be > 0"));
    }

    // Bucket every tensor by layer index; collect non-layer names separately.
    let mut by_layer: HashMap<u32, Vec<String>> = HashMap::new();
    let mut input_side: Vec<String> = Vec::new(); // token_embd.*
    let mut output_side: Vec<String> = Vec::new(); // output*, output_norm.*
    let mut max_layer: Option<u32> = None;

    for name in meta.sorted_names() {
        if let Some(rest) = name.strip_prefix("blk.") {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && rest.as_bytes().get(digits.len()) == Some(&b'.') {
                let layer: u32 = digits.parse().map_err(|e| anyhow!("shard: bad layer index in '{name}': {e}"))?;
                max_layer = Some(max_layer.map_or(layer, |m| m.max(layer)));
                by_layer.entry(layer).or_default().push(name);
                continue;
            }
        }
        if name.starts_with("output") {
            output_side.push(name);
        } else {
            input_side.push(name);
        }
    }
    let Some(max_layer) = max_layer else {
        return Err(anyhow!("shard: GGUF has no 'blk.<N>.*' tensors to shard"));
    };
    let n_layers = max_layer + 1;

    let tensor_bytes = |name: &str| -> Result<u64> {
        meta.tensors.get(name).map(|t: &TensorMeta| t.nbytes).ok_or_else(|| anyhow!("shard: tensor '{name}' missing from GgufMeta"))
    };
    let layer_bytes = |l: u32| -> Result<u64> {
        by_layer.get(&l).ok_or_else(|| anyhow!("shard: layer {l} has no tensors (gap in blk.N numbering)"))?
            .iter().map(|n| tensor_bytes(n)).sum::<Result<u64>>()
    };

    let mut shards: Vec<ShardManifest> = Vec::new();
    let mut cur_names: Vec<String> = input_side.clone();
    let mut cur_bytes: u64 = input_side.iter().map(|n| tensor_bytes(n)).sum::<Result<u64>>()?;
    let mut cur_lo: u32 = 0;

    for l in 0..n_layers {
        let lb = layer_bytes(l)?;
        if lb > target_bytes {
            return Err(anyhow!("shard: layer {l} alone is {lb} bytes, over target_bytes={target_bytes} — raise the target"));
        }
        let would_be = cur_bytes + lb;
        let shard_has_layers = cur_lo < l; // at least one layer already accepted into this shard
        if shard_has_layers && would_be > target_bytes {
            shards.push(ShardManifest { index: shards.len() as u16, layer_lo: cur_lo, layer_hi: l, tensor_names: { let mut v = cur_names.clone(); v.sort(); v } });
            cur_names = Vec::new();
            cur_bytes = 0;
            cur_lo = l;
        }
        cur_names.extend(by_layer[&l].iter().cloned());
        cur_bytes += lb;
    }
    cur_names.extend(output_side.iter().cloned());
    shards.push(ShardManifest { index: shards.len() as u16, layer_lo: cur_lo, layer_hi: n_layers, tensor_names: { let mut v = cur_names; v.sort(); v } });

    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(layers: u32, bytes_per_layer: u64, extra: &[(&str, u64)]) -> GgufMeta {
        let mut tensors = HashMap::new();
        for l in 0..layers {
            tensors.insert(format!("blk.{l}.attn.weight"), TensorMeta { offset: 0, nbytes: bytes_per_layer });
        }
        for (name, nbytes) in extra {
            tensors.insert((*name).to_string(), TensorMeta { offset: 0, nbytes: *nbytes });
        }
        GgufMeta { tensors, tensor_data_offset: 0 }
    }

    #[test]
    fn packs_by_fixed_byte_target() {
        // 10 layers x 100 bytes, target 250 -> 2 layers per shard except a possible remainder.
        let m = meta(10, 100, &[("token_embd.weight", 10), ("output.weight", 10)]);
        let shards = pack_shards(&m, 250).unwrap();
        // Every layer covered exactly once, ranges contiguous, ranges cover 0..10.
        let mut lo = 0u32;
        for s in &shards {
            assert_eq!(s.layer_lo, lo);
            assert!(s.layer_hi > s.layer_lo);
            lo = s.layer_hi;
        }
        assert_eq!(lo, 10);
        // input tensor is in shard 0, output tensor is in the last shard.
        assert!(shards[0].tensor_names.contains(&"token_embd.weight".to_string()));
        assert!(shards.last().unwrap().tensor_names.contains(&"output.weight".to_string()));
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let m = meta(7, 50, &[]);
        let a = pack_shards(&m, 130).unwrap();
        let b = pack_shards(&m, 130).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_target_gives_different_boundaries() {
        let m = meta(7, 50, &[]);
        let small = pack_shards(&m, 60).unwrap();
        let big = pack_shards(&m, 400).unwrap();
        assert!(small.len() > big.len());
    }

    #[test]
    fn single_layer_too_big_is_an_error() {
        let m = meta(3, 500, &[]);
        let err = pack_shards(&m, 100).unwrap_err();
        assert!(err.to_string().contains("over target_bytes"));
    }

    #[test]
    fn tensor_names_within_a_shard_are_sorted() {
        let m = meta(3, 10, &[]);
        let shards = pack_shards(&m, 1000).unwrap();
        for s in &shards {
            let mut sorted = s.tensor_names.clone();
            sorted.sort();
            assert_eq!(s.tensor_names, sorted);
        }
    }
}
