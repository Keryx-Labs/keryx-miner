# Shard Possession PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fixed-size, fingerprinted slice ("shard") of a model's GGUF independently mineable via PoM on one GPU, and accepted by a testnet node — without changing `PomProof`/consensus wire format.

**Architecture:** A shard is a deterministic, layer-aligned byte-range of the canonical GGUF with its own Merkle root (built with the same chunk/tree machinery the miner already uses for whole models, `WeightIndex`, just over a name subset). A shard device skips the miner's existing whole-model/llama-engine machinery entirely and takes a new, simpler path: raw upload of only its own tensors (`load_raw_subset`), its own possession index, its own tier index (5, 6, …, additive to the existing 5-tier table). The node accepts shard blocks by adding rows to its tier table behind a fork gate that is `never()` outside a private testnet — zero new verification code.

**Tech Stack:** Rust (miner: CUDA via `cudarc`; node: existing consensus pipeline). No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-network-fatia-shard-poc-design.md` (this plan implements sections 1–4 and Testing-plan Phase A only; §3.5 ggml-rpc pipeline and Phase B are a separate follow-up plan).

## Global Constraints

- No change to `PomProof`, block header format (beyond using an already-`u8` `pom_tier`/`proof.tier` field with a higher value), difficulty, cohorts, strikes, or escrow. (Spec §"Explicitly not doing".)
- The node change is gated behind a new `ForkActivation` that is `never()` on `MAINNET_PARAMS`/`DEVNET`/`SIMNET`; only the user's private testnet params activate it. (Spec §4.)
- Miner repo root for this plan: `C:\Testnet\keryx-miner\.worktrees\shard-poc` (branch `shard-poc`, off `main`).
- Node repo root for this plan: `C:\Testnet\keryx-node\.claude\worktrees\shard-poc` (branch `shard-poc`, off `feature/watchdog-relay-v160` — NOT `master`, which is stale at v1.4.5).
- Every new/modified function below was checked against the actual code in these two worktrees on 2026-09-09 (not the stale `tensor-split` worktree, not `master`). Two corrections vs. the spec doc, found during this planning pass — trust this plan over the spec doc where they differ:
  1. **`POM_INDICES` (miner `src/pom.rs:1602`) is keyed by `model_id: [u8; 32]`, not by tier.** A shard shares its parent model's `model_id`, so a shard's index cannot live in that map (it would either collide with, or masquerade as, the whole-model index). This plan adds a **separate, device-keyed** shard index store instead (Task 4).
  2. Line numbers in the spec doc (e.g. `pom_gpu.rs:1533`, `body_validation_in_isolation.rs:222-330`) were read on a slightly different point in history; this plan re-cites the exact lines as they stand in the two `shard-poc` worktrees today. Re-check with `grep -n` before editing if any drift further during execution — other work may land on these branches concurrently.

---

## File Structure

**Miner (`C:\Testnet\keryx-miner\.worktrees\shard-poc`):**
- Create `src/shard.rs` — shard packing (pure function, no I/O in the packing logic itself beyond the caller-supplied `GgufMeta`).
- Modify `src/pom.rs` — add `WeightIndex::build_from_gguf_subset`, a device-keyed shard index store.
- Modify `src/pom_gpu.rs` — add `load_raw_subset`, shard device registry, shard-aware `ensure_installed_inner` branch, shard-aware `current_tier`/index lookup.
- Modify `src/models.rs` — add `pom_shard_tier_index`.
- Modify `src/cli.rs` — add `--shard`, `--print-shards` flags.
- Modify `src/main.rs` — parse `--shard`, register shard devices, handle `--print-shards`.
- Modify `src/miner.rs` — swap one call site to the new shard-aware index lookup.

**Node (`C:\Testnet\keryx-node\.claude\worktrees\shard-poc`):**
- Modify `consensus/core/src/config/params.rs` — add `POM_SHARDS_POC`, `shard_poc_activation` field + wiring, extend `pom_tiers()`.
- Modify `consensus/src/pipeline/body_processor/body_validation_in_isolation.rs` — pass the new activation flag into `pom_tiers(...)`.

---

### Task 1: Shard packing (`src/shard.rs`)

**Files:**
- Create: `src/shard.rs`
- Modify: `src/lib.rs` (add `pub mod shard;` — check the existing `pub mod pom;`-style list near the top and insert alongside it, alphabetically or matching the existing convention)
- Test: inline `#[cfg(test)] mod tests` in `src/shard.rs`

**Interfaces:**
- Consumes: `crate::gguf::GgufMeta { tensors: HashMap<String, TensorMeta>, tensor_data_offset: u64 }`, `crate::gguf::TensorMeta { offset: u64, nbytes: u64 }`, `GgufMeta::sorted_names(&self) -> Vec<String>` (all already exist, unchanged).
- Produces: `pub struct ShardManifest { pub index: u16, pub layer_lo: u32, pub layer_hi: u32, pub tensor_names: Vec<String> }`, `pub fn pack_shards(meta: &crate::gguf::GgufMeta, target_bytes: u64) -> anyhow::Result<Vec<ShardManifest>>` — consumed by Task 2 (fingerprint) and Task 6 (CLI/`--print-shards`).

- [ ] **Step 1: Write the failing tests**

```rust
// src/shard.rs
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
```

- [ ] **Step 2: Add the module and run the tests to verify they fail to compile (module doesn't exist yet), then pass**

Find the module list in `src/lib.rs` (`grep -n "^pub mod " src/lib.rs`) and add `pub mod shard;` alongside the existing entries (e.g. next to `pub mod pom;`).

Run: `cargo test --lib shard::tests -- --nocapture`
Expected: 5 tests pass (`packs_by_fixed_byte_target`, `deterministic_same_input_same_output`, `different_target_gives_different_boundaries`, `single_layer_too_big_is_an_error`, `tensor_names_within_a_shard_are_sorted`).

- [ ] **Step 3: Commit**

```bash
git add src/shard.rs src/lib.rs
git commit -m "feat(shard): add fixed-size, layer-aligned shard packing"
```

---

### Task 2: Shard-scoped `WeightIndex` (`src/pom.rs`)

**Files:**
- Modify: `src/pom.rs` (functions `open_existing_tree` at line 926, `WeightIndex::build_from_gguf` at line 1021 — verify with `grep -n "fn open_existing_tree\|pub fn build_from_gguf\b" src/pom.rs` before editing, in case line numbers drifted)
- Test: extend the existing `#[cfg(test)] mod tests` in `src/pom.rs` (already has a `synth_index`-style helper per the design doc — search for it with `grep -n "fn synth_index\|ChunkSource::Ram" src/pom.rs` and reuse its pattern)

**Interfaces:**
- Consumes: `crate::gguf::GgufMeta`, existing `WeightIndex` fields/methods (all unchanged).
- Produces: `pub fn WeightIndex::build_from_gguf_subset(path: &str, model_id: [u8; 32], tensor_names: &[String], tree_filename: &str) -> Result<Self>` — consumed by Task 4's shard install path.

This task **refactors** `open_existing_tree` and `build_from_gguf` to accept an optional tensor-name filter and an explicit tree filename, without changing behavior for existing callers (they keep calling the unchanged `build_from_gguf(path, model_id)`, which becomes a thin wrapper).

- [ ] **Step 1: Write the failing test** (subset root matches a manually-built dense tree over the same subset)

Add to `src/pom.rs`'s existing test module:

```rust
    #[test]
    fn build_from_gguf_subset_root_matches_full_dense_tree_over_same_leaves() {
        // Build a tiny synthetic GGUF-like fixture on disk (reuse whatever helper the existing
        // `build_from_gguf`-based tests in this file already use to write a real temp GGUF —
        // search this file for an existing test that calls `WeightIndex::build_from_gguf` with a
        // real path, e.g. near line 1751/1851, and copy its fixture-writing helper verbatim).
        // With that helper (call it `write_synthetic_gguf(path, tensor_sizes: &[(&str, u64)])`):
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        write_synthetic_gguf(&gguf_path, &[
            ("blk.0.a", 320), ("blk.0.b", 320), ("blk.1.a", 320), ("output.weight", 320),
        ]);
        let model_id = [7u8; 32];

        // Full index over everything.
        let full = WeightIndex::build_from_gguf(gguf_path.to_str().unwrap(), model_id).unwrap();

        // Subset index over just layer 0's tensors + output (skip blk.1.a).
        let subset_names = vec!["blk.0.a".to_string(), "blk.0.b".to_string(), "output.weight".to_string()];
        let subset = WeightIndex::build_from_gguf_subset(
            gguf_path.to_str().unwrap(), model_id, &subset_names, "pom-tree.shard0-test.bin",
        ).unwrap();

        // Subset root must differ from the full root (fewer leaves) but subset n_chunks must
        // equal exactly the chunk count of the 3 included tensors (320*3/32 = 30).
        assert_ne!(subset.r_t, full.r_t);
        assert_eq!(subset.n_chunks, 30);
    }
```

If no existing `write_synthetic_gguf`-style helper exists in this file, write one now (a minimal GGUF writer matching whatever `GgufMeta::read` expects — check `src/gguf.rs`'s reader for the exact header format before writing it, since this must round-trip through the real parser, not a mock).

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib pom::tests::build_from_gguf_subset_root_matches_full_dense_tree_over_same_leaves`
Expected: FAIL — `build_from_gguf_subset` not found.

- [ ] **Step 3: Implement — thread an optional subset through `open_existing_tree` and `build_from_gguf`**

In `src/pom.rs`, change the signature of `open_existing_tree` (currently `fn open_existing_tree(tree_path: &Path, gguf_path: &str, expected_model_id: [u8; 32]) -> Result<WeightIndex>`, line 926) to add a subset parameter, and filter `names` right after it's computed (line 929: `let names = meta.sorted_names();`):

```rust
fn open_existing_tree(
    tree_path: &Path,
    gguf_path: &str,
    expected_model_id: [u8; 32],
    subset: Option<&[String]>,
) -> Result<WeightIndex> {
    let mut file = File::open(gguf_path)?;
    let meta = crate::gguf::GgufMeta::read(&mut file)?;
    let names: Vec<String> = match subset {
        Some(s) => {
            let want: std::collections::HashSet<&str> = s.iter().map(|x| x.as_str()).collect();
            meta.sorted_names().into_iter().filter(|n| want.contains(n.as_str())).collect()
        }
        None => meta.sorted_names(),
    };
    // ...rest of the function body is UNCHANGED from here on (n_chunks/table loop, checkpoint
    // math, mmap, WeightIndex construction) — it already only depends on `names`, not on
    // "all tensors in the file".
```

The one call site of `open_existing_tree` (inside `build_from_gguf`, line 1039) becomes `open_existing_tree(&tree_path, path, model_id, subset)` once `build_from_gguf` itself gains the parameter (next).

Rename the existing `pub fn build_from_gguf(path: &str, model_id: [u8; 32]) -> Result<Self>` body to a private `fn build_from_gguf_impl(path: &str, model_id: [u8; 32], subset: Option<&[String]>, tree_filename: &str) -> Result<Self>`, with exactly two changes inside the existing body:

1. Line ~1023, `let tree_path = dir.join("pom-tree.bin");` → `let tree_path = dir.join(tree_filename);`
2. Line ~1054, `let names = meta.sorted_names(); // canonical order` → the same subset-filtering snippet as above (`match subset { Some(s) => ..., None => meta.sorted_names() }`).
3. The `open_existing_tree(&tree_path, path, model_id)` call (in the "reuse existing checkpoint tree" branch, ~line 1039) becomes `open_existing_tree(&tree_path, path, model_id, subset)`.

Then add two thin public wrappers where the old `pub fn build_from_gguf` used to be:

```rust
    pub fn build_from_gguf(path: &str, model_id: [u8; 32]) -> Result<Self> {
        Self::build_from_gguf_impl(path, model_id, None, "pom-tree.bin")
    }

    /// Same as `build_from_gguf`, but over only `tensor_names` (already sorted or not — this
    /// re-sorts). `tree_filename` must be unique per (model, shard) pair so different shards'
    /// checkpoint trees never collide on disk next to the same GGUF.
    pub fn build_from_gguf_subset(
        path: &str,
        model_id: [u8; 32],
        tensor_names: &[String],
        tree_filename: &str,
    ) -> Result<Self> {
        Self::build_from_gguf_impl(path, model_id, Some(tensor_names), tree_filename)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib pom::tests -- --nocapture`
Expected: all existing `pom::tests` still pass (unchanged behavior for `build_from_gguf`/whole-model callers), plus the new subset test passes.

- [ ] **Step 5: Add a device-keyed shard index store** (needed by Task 4; belongs in `pom.rs` next to `POM_INDICES` at line 1602)

```rust
/// Possession index for a SHARD, keyed by the CUDA device mining it (not by model_id — a shard
/// shares its parent model's model_id, so keying by model_id would collide with or shadow the
/// whole-model entry in `POM_INDICES`). This phase is one shard per device per process, so
/// device-keying is sufficient; a future multi-shard-per-device design would need a composite key.
static POM_SHARD_INDICES: OnceLock<Mutex<HashMap<u32, Arc<WeightIndex>>>> = OnceLock::new();

fn pom_shard_indices() -> &'static Mutex<HashMap<u32, Arc<WeightIndex>>> {
    POM_SHARD_INDICES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn set_shard_index(device_id: u32, index: WeightIndex) {
    if let Ok(mut g) = pom_shard_indices().lock() {
        g.insert(device_id, Arc::new(index));
    }
}

pub fn active_shard_index(device_id: u32) -> Option<Arc<WeightIndex>> {
    pom_shard_indices().lock().ok()?.get(&device_id).cloned()
}
```

(Mirror the exact style of `set_index`/`active_index_for_model`/`clear_index` at lines 1609/1616/1623 — read them first with `grep -n -A6 "pub fn set_index" src/pom.rs` so the new functions match the codebase's existing locking idiom exactly.)

- [ ] **Step 6: Run full pom.rs test suite, then commit**

Run: `cargo test --lib pom::`
Expected: PASS, no regressions.

```bash
git add src/pom.rs
git commit -m "feat(pom): add WeightIndex::build_from_gguf_subset + device-keyed shard index store"
```

---

### Task 3: Scoped raw upload (`src/pom_gpu.rs`)

**Files:**
- Modify: `src/pom_gpu.rs` (new method next to `pub fn load_raw` at line 635 — re-verify line with `grep -n "pub fn load_raw\b" src/pom_gpu.rs`)

**Interfaces:**
- Consumes: `crate::gguf::GgufMeta`, existing `PomGpuMiner` struct/fields (unchanged), `select_pom_kernel` (unchanged).
- Produces: `pub fn PomGpuMiner::load_raw_subset(gguf_path: &str, device_id: usize, tensor_names: &[String]) -> Result<Self>` — consumed by Task 4.

This is not independently unit-testable (it opens a real CUDA device) — same category as the existing `load_raw`/`load_llama`, which have no unit tests either (search confirms: `grep -n "fn load_raw\|#\[test\]" src/pom_gpu.rs` shows no test directly exercising `load_raw`). Verification for this task is the Task 4 guard-order test (compile-time / mock-level) plus the Task 8 hardware integration pass.

- [ ] **Step 1: Implement, mirroring `load_raw` exactly except for a name filter**

Read the existing `load_raw` body first (`src/pom_gpu.rs:635-681`) to copy it precisely, then add, directly below it:

```rust
    /// Same as `load_raw`, but uploads only `tensor_names` instead of every tensor in the GGUF.
    /// Used for a shard: the walk gathers over exactly this device's own slice of the canonical
    /// bytes, so no cross-GPU pointer dependency exists and no other GPU needs to be involved.
    pub fn load_raw_subset(gguf_path: &str, device_id: usize, tensor_names: &[String]) -> Result<Self> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();

        let mut file = std::fs::File::open(gguf_path)?;
        let meta = crate::gguf::GgufMeta::read(&mut file)?;
        let want: std::collections::HashSet<&str> = tensor_names.iter().map(|s| s.as_str()).collect();
        let names: Vec<String> = meta.sorted_names().into_iter().filter(|n| want.contains(n.as_str())).collect();
        if names.len() != tensor_names.len() {
            return Err(anyhow!(
                "PoM GPU: shard subset requested {} tensors, but only {} were found in the GGUF — manifest/GGUF mismatch",
                tensor_names.len(), names.len()
            ));
        }

        let mut uploads: Vec<CudaSlice<u8>> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut host_buf: Vec<u8> = Vec::new();
        for name in &names {
            let t = &meta.tensors[name];
            let chunks = t.nbytes / CHUNK_BYTES as u64;
            if chunks == 0 {
                continue;
            }
            host_buf.resize(t.nbytes as usize, 0);
            crate::pom::read_exact_at(&file, &mut host_buf, meta.tensor_data_offset + t.offset)?;
            let dev = stream.clone_htod(host_buf.as_slice())?;
            bases.push(dev.device_ptr(&stream).0 as u64);
            uploads.push(dev);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(anyhow!("PoM GPU: shard produced 0 chunks"));
        }

        let bases_dev = stream.clone_htod(bases.as_slice())?;
        let prefix_dev = stream.clone_htod(prefix.as_slice())?;
        let kernel = select_pom_kernel(device_id)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _uploads: uploads,
        })
    }
```

- [ ] **Step 2: Build**

Run: `cargo build --features cuda 2>&1 | tail -50`
Expected: clean build (this function has no callers yet, so it must compile standalone — check for unused-function warnings, which are expected and fine until Task 4 wires a caller).

- [ ] **Step 3: Commit**

```bash
git add src/pom_gpu.rs
git commit -m "feat(pom_gpu): add load_raw_subset for shard-scoped raw upload"
```

---

### Task 4: Shard device registry + install path (`src/pom_gpu.rs`)

**Files:**
- Modify: `src/pom_gpu.rs` — add shard registry next to `MINING_TIERS` (line 1045), branch `ensure_installed_inner` (line 1434), branch `current_tier` (line 1301), add `active_index_for_device` helper.

**Interfaces:**
- Consumes: `Task 1`'s `crate::shard::ShardManifest`, `Task 2`'s `crate::pom::{WeightIndex::build_from_gguf_subset, set_shard_index, active_shard_index}`, `Task 3`'s `PomGpuMiner::load_raw_subset`, `Task 5`'s `crate::models::pom_shard_tier_index`.
- Produces:
  - `pub fn set_shard_for_device(device_id: u32, model_id: [u8; 32], gguf_path: String, target_bytes: u64, shard: crate::shard::ShardManifest)`
  - `pub fn shard_for_device(device_id: u32) -> Option<ShardAssignment>` (new `pub struct ShardAssignment { model_id: [u8;32], gguf_path: String, target_bytes: u64, manifest: crate::shard::ShardManifest }`)
  - `pub fn active_index_for_device(device_id: u32, model_id: [u8; 32]) -> Option<Arc<crate::pom::WeightIndex>>` — consumed by Task 7 (miner.rs call site).

- [ ] **Step 1: Add the shard registry, next to `MINING_TIERS`/`mining_tiers()` (line 1045)**

```rust
/// A device's shard assignment (from `--shard`). Disjoint from `MINING_TIERS`: a device is
/// either a whole-tier miner (existing path) or a shard miner (this path), never both.
#[derive(Clone)]
pub struct ShardAssignment {
    pub model_id: [u8; 32],
    pub gguf_path: String,
    pub target_bytes: u64,
    pub manifest: crate::shard::ShardManifest,
}

static SHARD_ASSIGNMENTS: OnceLock<Mutex<HashMap<u32, ShardAssignment>>> = OnceLock::new();

fn shard_assignments() -> &'static Mutex<HashMap<u32, ShardAssignment>> {
    SHARD_ASSIGNMENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn set_shard_for_device(device_id: u32, model_id: [u8; 32], gguf_path: String, target_bytes: u64, manifest: crate::shard::ShardManifest) {
    if let Ok(mut g) = shard_assignments().lock() {
        g.insert(device_id, ShardAssignment { model_id, gguf_path, target_bytes, manifest });
    }
}

pub fn shard_for_device(device_id: u32) -> Option<ShardAssignment> {
    shard_assignments().lock().ok()?.get(&device_id).cloned()
}

/// The possession index for whatever this device is mining — its shard's index if it's a shard
/// device, else the whole-model index (existing `active_index_for_model` path). Single lookup
/// point so callers (the mining loop) don't need to know which kind of device they're on.
pub fn active_index_for_device(device_id: u32, model_id: [u8; 32]) -> Option<Arc<crate::pom::WeightIndex>> {
    if shard_for_device(device_id).is_some() {
        crate::pom::active_shard_index(device_id)
    } else {
        crate::pom::active_index_for_model(&model_id)
    }
}
```

`Arc` must be in scope in `pom_gpu.rs` — check the existing `use std::sync::{..., Arc, ...};` at the top of the file (it's almost certainly already imported, since `Arc<CudaContext>` is used at line 619); if not, add it.

- [ ] **Step 2: Branch `ensure_installed_inner` (line 1434) — shard devices take a separate, simpler path**

At the very top of `fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {`, right after the existing `if inference_paused() { return false; }` (line 1437-1439), insert:

```rust
    if let Some(shard) = shard_for_device(device_id) {
        return ensure_shard_installed_inner(device_id, &shard);
    }
```

Then add a new sibling function, right after `ensure_installed_inner` ends (after line 1621's closing `}`):

```rust
/// Install path for a shard device: raw scoped upload only, no llama engine, no era-crossing,
/// own possession index (keyed by device, not model_id — see `pom.rs::POM_SHARD_INDICES`).
fn ensure_shard_installed_inner(device_id: u32, shard: &ShardAssignment) -> bool {
    if is_oom_banlisted(device_id, &shard.model_id) {
        return false;
    }
    if crate::pom::active_shard_index(device_id).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_shard_index(device_id).is_none() {
            let tree_filename = format!("pom-tree.shard{}-{}g.bin", shard.manifest.index, shard.target_bytes / 1_000_000_000);
            info!(
                "PoM[gpu{}]: building host index for shard {} (layers {}..{}) — this can take a while...",
                device_id, shard.manifest.index, shard.manifest.layer_lo, shard.manifest.layer_hi
            );
            match crate::pom::WeightIndex::build_from_gguf_subset(&shard.gguf_path, shard.model_id, &shard.manifest.tensor_names, &tree_filename) {
                Ok(idx) => {
                    info!("PoM[gpu{}]: shard {} host index ready — N={} chunks", device_id, shard.manifest.index, idx.n_chunks);
                    crate::pom::set_shard_index(device_id, idx);
                }
                Err(e) => {
                    log::error!("PoM[gpu{}]: shard {} host index build failed: {}", device_id, shard.manifest.index, e);
                    return false;
                }
            }
        }
    }

    // Shard devices never run the llama engine and never zero-dup: raw scoped upload only.
    info!(
        "PoM[gpu{}]: shard {}/{} — walk uses a raw scoped upload (no zero-dup, no llama engine)",
        device_id, shard.manifest.index, /* total shard count isn't known to this device in
        this phase — each process is handed exactly one shard via --shard; log index only */ shard.manifest.index
    );
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        PomGpuMiner::load_raw_subset(&shard.gguf_path, device_id as usize, &shard.manifest.tensor_names)
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            log::error!("PoM[gpu{}]: shard {} miner load failed: {}", device_id, shard.manifest.index, e);
            oom_banlist_add(device_id, shard.model_id);
            return false;
        }
        Err(_) => {
            log::error!("PoM[gpu{}]: shard {} miner load panicked (likely OOM)", device_id, shard.manifest.index);
            oom_banlist_add(device_id, shard.model_id);
            return false;
        }
    };
    let n = gm.n_chunks();
    if let Some(idx) = crate::pom::active_shard_index(device_id) {
        if n != idx.n_chunks {
            log::error!(
                "PoM[gpu{}]: shard {} gather N={} != shard index N={} — refusing to mine",
                device_id, shard.manifest.index, n, idx.n_chunks
            );
            return false;
        }
    }
    install(device_id, gm);
    info!("PoM[gpu{}]: shard {} miner ready — N={} chunks resident", device_id, shard.manifest.index, n);
    true
}
```

Before writing this, run `grep -n "fn is_oom_banlisted\|fn oom_banlist_add\|fn index_build_lock\|fn install\b\|n_chunks(&self)" src/pom_gpu.rs` to confirm these helper names/signatures exactly (they are used unchanged from the existing `ensure_installed_inner`/its neighborhood — copy the exact call shape, e.g. whether `oom_banlist_add` takes `model_id` by value or reference, from its existing call site at line 1590).

- [ ] **Step 3: Branch `current_tier` (line 1301) for shard devices**

Current body:
```rust
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}
```

Change to:
```rust
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    if let Some(shard) = shard_for_device(device_id) {
        return crate::models::pom_shard_tier_index(&shard.model_id, shard.target_bytes, shard.manifest.index, daa);
    }
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}
```

- [ ] **Step 4: Build**

Run: `cargo build --features cuda 2>&1 | tail -80`
Expected: clean build (still no CLI/main.rs wiring yet — Task 5's `pom_shard_tier_index` must exist first for this to compile; if doing tasks in order, do Task 5 before this build step, or stub `pom_shard_tier_index` temporarily and finish it properly in Task 5).

- [ ] **Step 5: Commit**

```bash
git add src/pom_gpu.rs
git commit -m "feat(pom_gpu): shard device registry + shard-aware install/current_tier paths"
```

---

### Task 5: Shard tier index (`src/models.rs`)

**Files:**
- Modify: `src/models.rs` (next to `pom_tier_index` at line 158)

**Interfaces:**
- Consumes: nothing new.
- Produces: `pub fn pom_shard_tier_index(model_id: &[u8; 32], target_bytes: u64, shard_index: u16, daa: u64) -> Option<u8>` — consumed by Task 4.

This function's output **must match, index-for-index, the order the node pins in `POM_SHARDS_POC`** (Task 9) — that table's row order is what `proof.tier` numbers actually mean on-chain. For this POC there is exactly one sharded model/target combination, so the mapping is trivial; document that constraint so nobody adds a second sharded model without updating both sides together.

- [ ] **Step 1: Write the failing test**

Add near the existing `pom_tier_index` tests (search `grep -n "mod tests" src/models.rs`):

```rust
    #[test]
    fn pom_shard_tier_index_offsets_from_h6_table_size() {
        let daa = crate::pom::pom_v3_activation_daa(); // any DAA at/after H6
        // Shard 0 and shard 1 of the POC model land at 5 and 6 (H6 table has 5 whole-tier rows).
        assert_eq!(pom_shard_tier_index(&KIMI_LINEAR_48B.model_id, POC_SHARD_TARGET_BYTES, 0, daa), Some(5));
        assert_eq!(pom_shard_tier_index(&KIMI_LINEAR_48B.model_id, POC_SHARD_TARGET_BYTES, 1, daa), Some(6));
        // Wrong model_id -> None.
        assert_eq!(pom_shard_tier_index(&QWEN3_6_27B.model_id, POC_SHARD_TARGET_BYTES, 0, daa), None);
        // Wrong target_bytes -> None (not the pinned POC target).
        assert_eq!(pom_shard_tier_index(&KIMI_LINEAR_48B.model_id, POC_SHARD_TARGET_BYTES + 1, 0, daa), None);
        // Before H6 activation -> None.
        assert_eq!(pom_shard_tier_index(&KIMI_LINEAR_48B.model_id, POC_SHARD_TARGET_BYTES, 0, 0), None);
    }
```

(`KIMI_LINEAR_48B` and `QWEN3_6_27B` are the existing `ModelSpec` statics already used elsewhere in this file's tests, e.g. line 341/347 — reuse them, don't redefine.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib models::tests::pom_shard_tier_index_offsets_from_h6_table_size`
Expected: FAIL — `pom_shard_tier_index`/`POC_SHARD_TARGET_BYTES` not found.

- [ ] **Step 3: Implement**

```rust
/// The one (model, target_bytes) combination this POC shards. 8 GiB, matching the spec's
/// suggested target for Kimi-Linear-48B on a 24 GB card (raw-scoped upload leaves headroom for
/// the rest of the process). A second sharded model/target needs a second constant here AND a
/// matching second block of rows in the node's `POM_SHARDS_POC` (keryx-node
/// consensus/core/src/config/params.rs) — the two must stay in lockstep by construction; there
/// is no dynamic negotiation in this phase.
pub const POC_SHARD_TARGET_BYTES: u64 = 8 * 1_000_000_000;

/// Shard tier index for the POC sharded model. Mirrors the node's `POM_SHARDS_POC` row order
/// (index 0 -> tier 5, index 1 -> tier 6, ... immediately after the 5-row H6 table). `None` for
/// any model/target this POC doesn't shard, or before the H6 gate.
pub fn pom_shard_tier_index(model_id: &[u8; 32], target_bytes: u64, shard_index: u16, daa: u64) -> Option<u8> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    if *model_id != KIMI_LINEAR_48B.model_id || target_bytes != POC_SHARD_TARGET_BYTES {
        return None;
    }
    5u8.checked_add(shard_index as u8)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib models::tests::pom_shard_tier_index_offsets_from_h6_table_size`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models.rs
git commit -m "feat(models): add pom_shard_tier_index for the POC sharded model"
```

---

### Task 6: CLI flags (`src/cli.rs`)

**Files:**
- Modify: `src/cli.rs` (add flags near `force_model`, line 43-51)

**Interfaces:**
- Consumes: nothing new.
- Produces: `opt.shard: Option<String>`, `opt.print_shards: Option<String>` — consumed by Task 7.

- [ ] **Step 1: Add the two flags**, matching the existing `force_model` style exactly (same struct, same `help_heading`):

```rust
    #[clap(
        long = "shard",
        value_name = "TIER:TARGET_GIB:IDX[,TIER:TARGET_GIB:IDX...]",
        help = "Mine one fixed-size shard of a model instead of the whole tier (CUDA-driver order, CSV): \
                e.g. --shard very-high:8:0 -> GPU0 mines shard 0 of an 8 GiB split of the very-high model. \
                A GPU listed here mines ONLY its shard, never the whole tier. Requires the model's GGUF to \
                be present (combine with --force-model naming the same tier so it gets prefetched).",
        help_heading = "OPoI / Inference"
    )]
    pub shard: Option<String>,

    #[clap(
        long = "print-shards",
        value_name = "TIER:TARGET_GIB",
        help = "Compute and print the shard manifest (layer ranges, chunk counts, Merkle roots) for a \
                tier at a given byte target, then exit without mining. Used to produce the rows pinned \
                into a private testnet node's shard tier table.",
        help_heading = "OPoI / Inference"
    )]
    pub print_shards: Option<String>,
```

- [ ] **Step 2: Build**

Run: `cargo build --features cuda 2>&1 | tail -30`
Expected: clean build (flags parsed but unused yet).

- [ ] **Step 3: Commit**

```bash
git add src/cli.rs
git commit -m "feat(cli): add --shard and --print-shards flags"
```

---

### Task 7: Wire `--shard` and `--print-shards` into `main.rs` and the mining loop

**Files:**
- Modify: `src/main.rs` (two insertion points, both inside the `#[cfg(any(feature = "cuda", ...))]` block — the CPU-only and Vulkan cfg blocks are NOT touched, this POC targets CUDA only)
- Modify: `src/miner.rs` (one call-site swap, line 523)

**Interfaces:**
- Consumes: `crate::shard::pack_shards`, `crate::pom_gpu::{set_shard_for_device, active_index_for_device}`, `opt.shard`, `opt.print_shards`.
- Produces: nothing new (this is the integration task).

- [ ] **Step 1: `--print-shards` early exit**

Find where `Opt::parse()` runs in `main.rs` (`grep -n "Opt::parse()" src/main.rs`) — right after it, before any network/GPU setup, insert:

```rust
    if let Some(spec) = opt.print_shards.as_deref() {
        let parts: Vec<&str> = spec.split(':').collect();
        let (tier_name, target_gib) = match parts.as_slice() {
            [t, g] => (*t, g.parse::<u64>().map_err(|e| format!("--print-shards: bad TARGET_GIB: {e}"))?),
            _ => return Err("--print-shards expects TIER:TARGET_GIB, e.g. very-high:8".into()),
        };
        let tier = parse_tier_name(tier_name).ok_or_else(|| format!("--print-shards: unrecognised tier '{tier_name}'"))?;
        let spec = keryx_miner::models::spec_for_tier(tier);
        let gguf_path = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        if !std::path::Path::new(&gguf_path).exists() {
            return Err(format!("--print-shards: {gguf_path} not found — run once without --print-shards first so the model is fetched, then retry").into());
        }
        let mut file = std::fs::File::open(&gguf_path)?;
        let meta = keryx_miner::gguf::GgufMeta::read(&mut file)?;
        let target_bytes = target_gib * 1_000_000_000;
        let shards = keryx_miner::shard::pack_shards(&meta, target_bytes)
            .map_err(|e| format!("--print-shards: packing failed: {e}"))?;
        println!("model_id = {}", hex::encode(spec.model_id));
        println!("target_bytes = {target_bytes}");
        for s in &shards {
            let tree_filename = format!("pom-tree.shard{}-{}g.bin", s.index, target_gib);
            let idx = keryx_miner::pom::WeightIndex::build_from_gguf_subset(&gguf_path, spec.model_id, &s.tensor_names, &tree_filename)?;
            println!(
                "shard {} layers=[{},{}) n_chunks={} root={}",
                s.index, s.layer_lo, s.layer_hi, idx.n_chunks, hex::encode(idx.r_t)
            );
        }
        return Ok(());
    }
```

Check the exact function signature `main.rs` uses for its top-level error type (grep for `fn main(` and the surrounding `-> Result<..., ...>` or similar) — `.into()`/`?` above must match whatever error type that function returns; adjust if it's not a plain `String`-convertible boxed error (the existing code around line 1337-1341 already does `return Err(e.into())` in the same function, so mirror that exactly).

- [ ] **Step 2: Register shard devices after the existing tier-assignment loop** (inside the `#[cfg(any(feature = "cuda", all(feature = "cuda-bridge-client", not(feature = "vulkan"))))]` block, right after the loop ending at line 1466 that calls `set_mining_tier`/`set_device_tier`)

```rust
    // --shard: additive per-device override. A device named here mines ONLY its shard (never
    // the whole tier) — the tier-assignment loop above already ensured its model's GGUF is on
    // disk (via --force-model naming the same tier), which load_raw_subset needs.
    if let Some(raw) = opt.shard.as_deref() {
        for entry in raw.split(',') {
            let parts: Vec<&str> = entry.trim().split(':').collect();
            let (tier_name, target_gib_s, idx_s) = match parts.as_slice() {
                [t, g, i] => (*t, *g, *i),
                _ => {
                    warn!("--shard: malformed entry '{entry}' (want TIER:TARGET_GIB:IDX) — ignoring.");
                    continue;
                }
            };
            let (Some(tier), Ok(target_gib), Ok(idx)) =
                (parse_tier_name(tier_name), target_gib_s.parse::<u64>(), idx_s.parse::<u16>())
            else {
                warn!("--shard: malformed entry '{entry}' — ignoring.");
                continue;
            };
            let spec = keryx_miner::models::spec_for_tier(tier);
            let gguf_path = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            let target_bytes = target_gib * 1_000_000_000;
            let mut file = match std::fs::File::open(&gguf_path) {
                Ok(f) => f,
                Err(e) => { warn!("--shard: cannot open {gguf_path}: {e} — ignoring entry '{entry}'."); continue; }
            };
            let meta = match keryx_miner::gguf::GgufMeta::read(&mut file) {
                Ok(m) => m,
                Err(e) => { warn!("--shard: cannot parse {gguf_path}: {e} — ignoring entry '{entry}'."); continue; }
            };
            let shards = match keryx_miner::shard::pack_shards(&meta, target_bytes) {
                Ok(s) => s,
                Err(e) => { warn!("--shard: packing failed for '{entry}': {e} — ignoring."); continue; }
            };
            let Some(manifest) = shards.into_iter().find(|s| s.index == idx) else {
                warn!("--shard: index {idx} out of range for '{entry}' — ignoring.");
                continue;
            };
            // Device id: --shard is positional-CSV like --force-model, so its Nth entry (in
            // parse order) targets CUDA device N. This mirrors forced_tiers' indexing exactly.
            let device_id = raw.split(',').position(|e| e.trim() == entry.trim()).unwrap_or(0) as u32;
            info!("--shard: GPU {device_id} -> shard {idx} of {} ({target_gib} GiB target, layers [{},{}))", spec.dir_name, manifest.layer_lo, manifest.layer_hi);
            keryx_miner::pom_gpu::set_shard_for_device(device_id, spec.model_id, gguf_path, target_bytes, manifest);
        }
    }
```

Note on `device_id` derivation above: this positional scheme is simple but fragile if two entries are textually identical; for this POC (one shard per process, one `--shard` entry per invocation per the design's "two processes" model) this is acceptable — flag it as a known simplification, not a bug, in the commit message.

- [ ] **Step 3: Swap the miner.rs call site to the shard-aware lookup** (line 523)

Current:
```rust
                                let idx = keryx_miner::pom::active_index_for_model(&model_id)?;
```
Change to:
```rust
                                let idx = keryx_miner::pom_gpu::active_index_for_device(worker_device_id, model_id)?;
```

- [ ] **Step 4: Build**

Run: `cargo build --features cuda 2>&1 | tail -80`
Expected: clean build.

- [ ] **Step 5: Unit-test `--shard`/`--print-shards` CLI parsing behavior in isolation**

Since the parsing logic above lives inline in `main.rs`'s `main()` (not a free function), and `main.rs` functions are typically not directly unit-tested here (confirm: `grep -n "#\[cfg(test)\]" src/main.rs` — if there is no existing test module in `main.rs`, this codebase's convention is to keep `main.rs` parsing logic thin and tested via the CLI/integration path, not unit tests). If that's confirmed, skip a dedicated unit test for this step and rely on Task 8's hardware integration pass to exercise it end-to-end; note this explicitly rather than inventing a test harness the codebase doesn't otherwise use.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs src/miner.rs
git commit -m "feat(main): wire --shard/--print-shards into tier assignment and the mining loop"
```

---

### Task 8: Node — testnet-gated shard tier table

**Files:**
- Modify: `C:\Testnet\keryx-node\.claude\worktrees\shard-poc\consensus\core\src\config\params.rs`
- Modify: `C:\Testnet\keryx-node\.claude\worktrees\shard-poc\consensus\src\pipeline\body_processor\body_validation_in_isolation.rs`
- Modify: wherever the struct that owns `check_pom_proof` (the `self` in that method) is constructed from `Params` — find it before editing.

**Interfaces:**
- Consumes: `crate::pom::PomTier { model_id, root, chunks }` (unchanged), `ForkActivation::{new, never}` (unchanged), the exact shard roots printed by Task 6's `--print-shards` (must be pasted in verbatim, not invented).
- Produces: `pom_tiers(..., shard_poc_active: bool)` (signature change — the one existing call site is updated in the same task).

**This task cannot be completed until Task 6/7 produce real `--print-shards` output on the actual Kimi-Linear-48B GGUF** (the roots are pinned data, not computable from this plan alone). Steps 1-3 below are the code scaffold; Step 4 is where the real roots get pasted in, gated on having run `--print-shards` for real.

- [ ] **Step 1: Add `POM_SHARDS_POC` with placeholder-but-structurally-real rows**

In `params.rs`, right after `POM_TIERS_H6` (ends at line 579), add:

```rust
/// Shard possession anchors for the network-shard PoC (see keryx-miner's
/// `docs/superpowers/specs/2026-09-09-network-fatia-shard-poc-design.md`). Each row is one
/// fixed-size, layer-aligned slice of Kimi-Linear-48B at the POC's 8 GiB target
/// (`models::POC_SHARD_TARGET_BYTES` on the miner side — the two must be kept in lockstep by
/// hand; there is no dynamic negotiation in this phase). Tier indices are 5, 6, ... immediately
/// after POM_TIERS_H6's 5 rows (`pom_tiers()` below concatenates them). Roots/chunks are pasted
/// verbatim from `keryx-miner --print-shards very-high:8` output — DO NOT hand-derive them.
pub const POM_SHARDS_POC: &[crate::pom::PomTier] = &[
    // TODO(shard-poc): paste real output of `keryx-miner --print-shards very-high:8` here, one
    // crate::pom::PomTier { model_id: KIMI_LINEAR_48B_MODEL_ID, root: [...], chunks: ... } per
    // shard line printed. This is the ONLY acceptable source for these values — do not compute
    // them by hand or guess. Placeholder below MUST be replaced before this branch is usable;
    // it is intentionally a compile error (deliberately wrong array length vs a real N) to make
    // "forgot to fill this in" impossible to miss:
    crate::pom::PomTier { model_id: KIMI_LINEAR_48B_MODEL_ID, root: [0u8; 32], chunks: 0 },
];
```

(Use whatever the existing constant for Kimi-Linear-48B's model_id is actually named in this file — `grep -n "KIMI_LINEAR_48B" consensus/core/src/config/params.rs` to get the exact identifier; the miner side's `KIMI_LINEAR_48B.model_id` and the node side's constant must be byte-identical, they're two independent copies of the same 32 bytes pinned in both repos today for the existing tiers too, same pattern.)

- [ ] **Step 2: Add the fork gate**

Find the `ForkActivation` fields already on whatever struct holds `pom_v3_activation` (search `grep -rn "pom_v3_activation:" consensus/core/src/config/params.rs` — likely a `Params` struct field plus one entry per network in `MAINNET_PARAMS`/`TESTNET_PARAMS`/`DEVNET_PARAMS`/`SIMNET_PARAMS`-style consts). Add a new field `shard_poc_activation: ForkActivation` in the struct definition, and in every existing network const's initializer, set it to `ForkActivation::never()` — **except** the user's own private testnet params, which the user will set up separately (this task does not create a new network; it just makes sure the field exists and defaults safe everywhere that already exists). If the user has not yet created private-testnet params in this repo, add a comment marking exactly where they'll add `ForkActivation::new(0)` later, rather than inventing a new network config file.

- [ ] **Step 3: Extend `pom_tiers()` (line 586) and its one call site**

```rust
pub fn pom_tiers(
    pom_v3_active: bool,
    h5_active: bool,
    coin_age_active: bool,
    very_light_active: bool,
    shard_poc_active: bool,
) -> &'static [crate::pom::PomTier] {
    if shard_poc_active && pom_v3_active {
        return pom_tiers_with_shards();
    }
    if pom_v3_active {
        POM_TIERS_H6
    } else if h5_active {
        POM_TIERS_H5
    } else if coin_age_active {
        POM_TIERS_H4
    } else if very_light_active {
        POM_TIERS_H2
    } else {
        POM_TIERS
    }
}

/// H6 tiers plus the shard-PoC rows, concatenated once. `shard_poc_active` is `never()`
/// everywhere except a private testnet, so this path is dead code on every real network.
fn pom_tiers_with_shards() -> &'static [crate::pom::PomTier] {
    static COMBINED: std::sync::OnceLock<Vec<crate::pom::PomTier>> = std::sync::OnceLock::new();
    COMBINED.get_or_init(|| POM_TIERS_H6.iter().copied().chain(POM_SHARDS_POC.iter().copied()).collect())
}
```

In `body_validation_in_isolation.rs`, the call site at line 252-257 gains a 5th argument:

```rust
        let tiers = pom_tiers(
            pom_v3,
            self.h5_activation.is_active(header.daa_score),
            self.coin_age_verification_activation.is_active(header.daa_score),
            self.very_light_activation.is_active(header.daa_score),
            self.shard_poc_activation.is_active(header.daa_score),
        );
```

This requires `self` (whatever struct `check_pom_proof` is a method on) to have a `shard_poc_activation: ForkActivation` field, populated the same way `h5_activation` etc. already are — find that struct's constructor (search `grep -rn "h5_activation:" consensus/src/` outside `params.rs` to find where the processor struct is built from `Params`) and add the mirror line.

- [ ] **Step 4: Build**

Run: `cargo build 2>&1 | tail -80`
Expected: clean build, with the placeholder `POM_SHARDS_POC` row present (it compiles fine — it's structurally valid Rust, just semantically wrong data — that's fine until Task 9 replaces it with real numbers).

- [ ] **Step 5: Commit**

```bash
git add consensus/core/src/config/params.rs consensus/src/pipeline/body_processor/body_validation_in_isolation.rs
git commit -m "feat(pom): add testnet-gated shard tier table (POM_SHARDS_POC), rows pending real manifest"
```

---

### Task 9: Pin real shard roots and run Phase A integration (manual, hardware-only)

This task is **not a code-loop task** — same category as the old `tensor-split` design's Task 6 (see its `progress.md`: "manual/hardware-only, not a code-loop task"). It requires the real rig, real models, and a running private testnet node. Do not attempt to simulate or mock this.

- [ ] **Step 1:** Build the miner (`cargo build --release --features cuda`) and run `keryx-miner --print-shards very-high:8`. Record the printed `model_id`, and each shard's `layer_range`, `n_chunks`, `root`.

- [ ] **Step 2:** Paste those exact values into `POM_SHARDS_POC` in the node worktree (replacing Task 8 Step 1's placeholder row, one `PomTier` entry per shard printed). Set `shard_poc_activation` in the private testnet's params to `ForkActivation::new(0)` (or whatever DAA the user's testnet uses for its own PoM-from-genesis convention — match the existing `pom_v3_activation` value already used for that testnet, per this repo's existing "PoM from genesis" pattern noted in the design doc's Testing plan).

- [ ] **Step 3:** Build and run the node (`cargo build --release`, start `keryxd` on the private testnet).

- [ ] **Step 4:** Run two miner processes on the rig, one per GPU (check CUDA ordinal vs. `nvidia-smi` index first — this rig has a known mismatch, see the design spec's citation of `feedback_cuda_ordinal_vs_nvidia_smi_index`):
   - Process A: `keryx-miner --pom-cuda-device 0 --force-model very-high --shard very-high:8:0 ...`
   - Process B: `keryx-miner --pom-cuda-device 1 --force-model very-high --shard very-high:8:1 ...`
   (Exact device ordinals depend on which physical card is 0 vs 1 after the `CUDA_DEVICE_ORDER=PCI_BUS_ID` fix — verify per the existing project precedent before assuming.)

- [ ] **Step 5:** Confirm in each process's logs: `shard N — walk uses a raw scoped upload`, `shard N miner ready — N=<n_chunks> chunks resident` matching the manifest from Step 1, and no OOM.

- [ ] **Step 6:** Let both mine. Gate: blocks with `proof.tier ∈ {5, 6}` are accepted by the node (no `PomUnknownTier`, no `BadPomProofV3` in the node's logs). This is the possession half, end to end, against the real verifier — the exact thing the old `tensor-split` PoC's Task 6 could not achieve.

- [ ] **Step 7 (guard regression check):** Temporarily edit one shard's `tensor_names` in the running miner's manifest computation (e.g. drop one tensor) and confirm the N-guard in `ensure_shard_installed_inner` refuses to mine (logs the "gather N != shard index N" error) rather than silently producing a proof. Revert the edit afterward.

- [ ] **Step 8:** Report results back — this closes out the plan's goal. If OPoI/pipeline serving across the two shards is wanted next, that's the separate follow-up plan for spec §3.3/§3.5 (ggml-rpc), not part of this one.

---

## Self-Review Notes (for whoever picks this plan up)

- **Spec coverage:** This plan implements spec §1 (shard definition) and §4 (node registration) in full, §2 (`--resident-tree` reuse — via the shared `WeightIndex` machinery, not the dense in-RAM tree itself, which stays opt-in and untouched) and §3.1-3.2 (CLI, scoped raw upload, guard) in full. It does **not** implement §3.3-3.5 (ggml-rpc pipeline) — that's the deferred follow-up plan the brainstorming session already flagged.
- **Corrections vs. the spec doc:** `POM_INDICES` keying (see Global Constraints) and several line-number citations were corrected after reading the actual `shard-poc` worktrees. If further drift is found while executing (concurrent work on `main`/`feature/watchdog-relay-v160`), re-run the relevant `grep -n` before trusting any line number in this plan.
- **Type consistency check:** `ShardManifest` (Task 1) is used identically in Tasks 2, 4, 6, 7 (`index: u16`, `layer_lo/hi: u32`, `tensor_names: Vec<String>`) — no renamed fields across tasks. `ShardAssignment` (Task 4) is used identically in Task 7's `set_shard_for_device` call and Task 4's `current_tier`/`ensure_shard_installed_inner`.
- **Placeholder scan:** The only intentional placeholder is Task 8 Step 1's `POM_SHARDS_POC` row, which is explicitly a deliberate compile-fine/semantically-wrong stand-in that Task 9 Step 2 replaces with real, measured data — this is not a "TBD" left for later without a plan to close it; Task 9 is that plan.
