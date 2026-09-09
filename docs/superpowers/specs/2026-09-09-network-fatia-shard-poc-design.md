# Fixed-size model shards: PoM possession per shard + ggml-rpc pipeline pinned to shard boundaries (design)

Date: 2026-09-09
Status: proposed, not yet implemented
Supersedes (for the "tier bigger than one GPU" goal): `2026-08-08-local-tensor-split-design.md`

## Problem

The local tensor-split POC (`tensor-split` branch, commits `dc24b41..3552ce3`) wires
`--tensor-split` cleanly, but Task 6 on the real rig proved it cannot *mine* a tier bigger than
the main GPU. Ledger: `.superpowers/sdd/2026-08-08-local-tensor-split/progress.md`, last entry.
The exact mechanism:

1. `src/pom_gpu.rs::ensure_installed_inner` (worktree, lines 960-973) forces `load_raw` whenever
   split is active. That guard is correct: the walk kernel dereferences raw device pointers on ONE
   device, and llama.cpp may have placed a tensor on the helper card. (`main` now has an even more
   explicit version of the same guard: `llama_engine::foreign_device_tensor`, `src/pom_gpu.rs:1533`.)
2. `PomGpuMiner::load_raw` (worktree lines 369-415, `main` line 635) uploads **every** tensor of the
   canonical GGUF (`meta.sorted_names()`) to one GPU. For Kimi-Linear-48B that is 29.7 GB into a
   24 GB card → `CUDA_ERROR_OUT_OF_MEMORY`, confirmed in the rig log.
3. The split FFI (`keryx_llama_load_split`) is never reached by any production caller, so the split
   engine does not serve inference either.

Root cause is not a bug: possession (walk over the whole tier) and hosting (weights spread across
GPUs) are coupled by the definition "1 tier = 1 model = 1 root `R_T`". A tier that does not fit one
GPU cannot be walked from one GPU. The way out is to make the **unit of possession smaller than the
model**: a shard with its own root, small enough for one GPU, walked entirely from that GPU. That is
also exactly the unit the network vision needs (one miner = one shard, pipelines assembled from
complementary shards), so this is the necessary first step before any of the pipeline economics.

Two facts already established, reused as-is:

- Local layer-split inference works and is fast: `llama-bench`, Kimi-Linear-48B Q4_K_M across
  4090 + 3090, `split_mode=LAYER`: `pp512 = 3081 tok/s`, `tg128 = 103.5 tok/s`.
- `--resident-tree` (commit `0d64adc`, `src/pom.rs::WeightIndex::build_dense`) and the sparse
  checkpoint tree (`build_from_gguf`, `merkle_path`) already produce canonical roots and paths over
  an arbitrary ordered tensor list — the `ChunkSource::Gguf { table }` reader is a list of
  `(first_chunk_index, file_offset)` per tensor, not a whole-file assumption.

## Branch reality (read before planning tasks)

The `tensor-split` worktree is **153 commits behind `main`** (`git rev-list --count
tensor-split..main`). `main` has, and the worktree does NOT have:

- `cuda-bridge/` crate + `src/pom_gpu_cuda_bridge.rs` + `--serve-cuda-bridge` (`src/main.rs:805`).
- The H6/v3 matrix walk (`src/pom_v3.rs`), the v4 re-walk (`src/pom_v4.rs`), H10 one-way seed
  (`63b7619`), the per-tensor llama reconcile (`19d19c1`), `ensure_loaded(gguf, gpu)` returning
  `Result` with `is_busy()/is_oom()` (`src/pom_gpu.rs:1507`).
- The H6 tier table (`models.rs::pom_tier_index`, 5 tiers gated on `pom_v3_activation_daa`).

The node's `master` branch is stuck at v1.4.5 (2026-08-14) and is stale. The correct current base
is `feature/watchdog-relay-v160` (v1.6.0, 2026-09-04) — strictly ahead of `master`, carries the
IBD body-swarm sync fix and the watchdog-relay work already deployed for live testing. It is already
at H6: `check_pom_proof` (`consensus/src/pipeline/body_processor/body_validation_in_isolation.rs:227`)
dispatches to `verify_pom_proof_v3_container` when `pom_v3_activation` is active (testnet gate = DAA
1000); `pom_tiers()` is at `consensus/core/src/config/params.rs:586` (line numbers shift slightly
from `master`, re-check against the actual worktree before editing).

**Decision: this design is implemented on new worktrees off the correct current base on each repo**,
not on `tensor-split` (miner) or `master` (node):

- Miner: `C:\Testnet\keryx-miner\.worktrees\shard-poc`, branch `shard-poc`, off `main`.
- Node: `C:\Testnet\keryx-node\.claude\worktrees\shard-poc`, branch `shard-poc`, off
  `feature/watchdog-relay-v160`.

Only two things are cherry-picked from `tensor-split`: `keryx_llama_load_split`'s pattern for
building `llama_model_params` (§3.3) and the "one-time `info!` at the guard branch point" testing
idea. The `--tensor-split` flag itself is not carried forward (it is the ratio-based split this
design replaces). The `cuda-bridge` crate (§5) stays untouched and out of scope for this phase per
the user's own steer — it solves a different problem (no-GPU clients), not the shard/pipeline
question this design is about.

## Scope

**In scope**

1. Shard definition: a deterministic, layer-aligned, fixed-byte-target slice of a canonical GGUF,
   with its own `(n_chunks, root)` fingerprint, computed with the existing `WeightIndex` machinery.
2. `load_raw` with scope: upload only the shard's tensors, build the gather over them. Same kernel,
   same `bases/prefix` tables, same N-guard.
3. `ensure_installed_inner` guard adjusted so a shard is always walked from a raw scoped upload on
   the device that holds it (never zero-dup in this phase).
4. OPoI pipeline over `ggml-rpc`, with llama.cpp's layer→device assignment **pinned by
   `tensor_buft_overrides`** to the same layer ranges the shards were cut on (not ratios).
5. Minimal node change: a testnet-only tier-table extension so the existing verifier can look up a
   shard's `(root, chunks)` by `proof.tier`. Zero new verification code.
6. Test plan: local 2-GPU rig as two independent miners, then a private testnet with two machines.

**Out of scope** (phase 2, per project leadership's "no consensus change until the physics is
measured"): any change to `PomProof`/header wire format, difficulty, cohorts, strikes, escrow;
dynamic pipeline marketplace; payment split per link; link failure detection/penalty/replacement;
mainnet tier tables. Listed again in "Explicitly not doing".

## Design

### 1. Shard definition

A shard is identified by `ShardId { model_id: [u8;32], target_bytes: u64, index: u16 }` and fully
determined by the canonical GGUF plus `target_bytes`. No operator choice enters the boundary.

**Packing rule** (new `src/shard.rs`, pure function over `gguf::GgufMeta`):

1. Group tensors by transformer layer: name matches `^blk\.(\d+)\.` → layer `L`. Everything else
   (`token_embd.weight`, `output_norm.weight`, `output.weight`, rope/attn misc without `blk.`) is
   "non-layer".
2. Walk layers in numeric order `0..n_layer`. Open shard 0 with all non-layer tensors that llama.cpp
   places on the *input* side (`token_embd.*`; see `llama-model.cpp:1309`, `dev_input` is always
   CPU — it still belongs to shard 0 for possession). Append layer bytes greedily: close the current
   shard when adding the next layer would exceed `target_bytes`, unless the shard is still empty of
   layers (a single layer larger than the target is a hard error — pick a bigger target). The last
   shard additionally takes the output-side non-layer tensors (`output_norm.*`, `output.*`).
3. Result: `Vec<ShardManifest { index, layer_range: [a, b), tensor_names: Vec<String> }>`. Layer
   ranges are contiguous and cover `0..n_layer` exactly once.

Why layer-aligned rather than a raw byte range of the canonical file: the canonical chunk order is
**name-sorted** (`gguf.rs::sorted_names`, `blk.1.` < `blk.10.` < `blk.2.`), so a contiguous byte
range of the canonical layout is not a contiguous group of layers, and the pipeline (§3.5) can only
cut on whole layers. Layer alignment gives a boundary that both PoM and ggml-rpc can agree on. The
byte size is "fixed target, quantised to whole layers" — the deviation is at most one layer.

**Fingerprint**: within a shard the tensors are ordered by the same canonical rule applied to the
subset (`tensor_names.sort()`), chunked `floor(nbytes/32)` 32 B chunks, blake3 leaves,
duplicate-last Merkle — i.e. `WeightIndex::build_from_gguf` restricted to `tensor_names`. This
yields `(n_chunks_shard, root_shard)`. A shard is therefore *structurally a `PomTier`*
(`keryx-node/consensus/core/src/pom.rs:61`: `{ model_id, root, chunks }`), which is what lets the
node stay unchanged in §4.

Tile constraint (v3/v4 walks address 64 KB / 1 KB tiles, `n_tiles = n_chunks / TILE_CHUNKS`,
`src/pom_v4.rs:87`): a shard needs `n_chunks ≥ 2048`; any real `target_bytes` (GiB-scale) satisfies
this trivially. Tile alignment is relative to the shard's own chunk 0, so no padding is needed.

**Manifest file**: `<models>/<dir_name>/shards-<target_gib>g.json` — `{ model_id, target_bytes,
shards: [{ index, layer_range, n_chunks, root_hex, tensor_names }] }`. Written by the new
`--print-shards` CLI (§3.1) and by the node-side pin step (§4). It is derived data (recomputable from
the GGUF), never trusted over recomputation.

### 2. Reusing `--resident-tree` / `WeightIndex`

Add `WeightIndex::build_from_gguf_subset(path, tensor_names: &[String], tree_path)` in `src/pom.rs`:
identical to `build_from_gguf` (lines 769-879) except (a) `names` is the sorted subset instead of
`meta.sorted_names()`, and (b) the tree file is `pom-tree.shard<idx>-<target_gib>g.bin` next to the
GGUF. `open_existing_tree` (line 686) recomputes expected size from the tensor list, so it gets the
same subset parameter. `build_dense` (`--resident-tree`) works unchanged over the subset —
`read_all_chunks_hashed` iterates the `table`. `read_chunk_bytes`, `merkle_path`, the v3/v4 tile
paths (`pom_v4.rs:101`) are untouched: they only see `n_chunks` and the table.

`POM_INDICES` (`pom.rs:1267`) stays keyed by `tier: u8`; the shard's tier index (§4) is the key.

### 3. `keryx-miner` changes

#### 3.1 CLI (`src/cli.rs`)

```
--shard <TIER>:<TARGET_GIB>:<IDX>[,<TIER>:<TARGET_GIB>:<IDX>...]
```
Per-GPU CSV in CUDA-driver order, same convention as `--force-model` (`main.rs::parse_tier_name`
call site). Example on the 2-GPU rig, two *processes* (see §5): process A `--shard very-high:8:0`
binds GPU 0, process B `--shard very-high:8:1` binds GPU 1. Malformed entries: warn + ignore
(mirrors `--force-model`). A GPU listed here mines that shard instead of a whole tier.

```
--print-shards <TIER>:<TARGET_GIB>      # compute + print the manifest (roots), exit
--rpc-serve <BIND_ADDR>                 # run llama.cpp rpc-server for this GPU's shard (§3.5)
--pipeline <ENDPOINT>[,<ENDPOINT>...]   # head: build the pipeline over these rpc-servers (§3.5)
```

#### 3.2 `src/pom_gpu.rs` — scoped raw upload

`PomGpuMiner::load_raw_subset(gguf_path, device_id, tensor_names: &[String])`: the body of
`load_raw` (`main` line 635) with `names.retain(|n| subset.contains(n))` after
`meta.sorted_names()`. Nothing else changes: `bases/prefix` are built over the retained tensors,
`n_total_chunks` becomes the shard's N, `select_pom_kernel` is the same. Also `n_tiles` for the
v3/v4 launches derives from that N exactly as today.

Guard in `ensure_installed_inner` (`main` lines 1503-1560): before the `use_llama` block,

```rust
if let Some(shard) = shard_for_device(device_id) {   // registry populated from --shard
    info!("PoM[gpu{}]: shard {}/{} of {} — walk uses a raw scoped upload (no zero-dup)", ...);
    // use_llama stays false; load path is load_raw_subset(&gguf, dev, &shard.tensor_names)
}
```

The existing N-guard (`gm.n_chunks() != idx.n_chunks → refuse`) now compares against the *shard*
index. This is the one guard that must not regress: a wrong subset silently produces proofs the node
rejects (no crash), so the one-time `info!` above names the shard and N for any log reviewer.

`set_mining_tier(device_id, model_id, gguf)` (line 819 area) gains the shard: the `MINING_TIERS`
value becomes `(model_id, gguf, Option<ShardManifest>)`. `current_tier` returns the shard's tier
index (§4) via a new `models::pom_shard_tier_index(model_id, target_bytes, idx, daa)` that mirrors
the node table. `advance_mining_tier_if_due` is a no-op for shard devices (no era swap in the POC).

`device_for_model` / OPoI routing (`slm::load_and_run_inference`, `main` around line 454): a shard
device never hosts a whole-model engine. Inference for a sharded model goes through the pipeline
head (§3.5); in this phase the head is selected by `--pipeline` on exactly one process.

#### 3.3 FFI (`tools/keryx-llama/keryx_llama.cpp`)

New export, alongside the existing `keryx_llama_load` (single GPU, untouched):

```c
KeryxLlama* keryx_llama_load_pipeline(const char* gguf_path, int n_ctx,
                                      const char** endpoints, int n_endpoints,
                                      const int* layer_lo, const int* layer_hi,   // per endpoint, [lo,hi)
                                      int local_gpu, int local_lo, int local_hi); // this process's own shard, or -1
```

Body, in order:

1. `ggml_backend_load_all()` once (the RPC backend is a loadable module; also needs the build change
   in §3.5).
2. For each endpoint: `ggml_backend_rpc_add_server(endpoint)` (resolved via
   `ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server")` exactly like
   `common/arg.cpp:959-965`). Devices come back named `RPC0`, `RPC1`, … in registration order
   (`ggml-rpc.cpp:1946`); their buffer type is `ggml_backend_dev_buffer_type(dev)`.
3. Build `llama_model_params.devices` = `[RPC devs..., local CUDA dev]` (NULL-terminated), and
   `tensor_buft_overrides` = one entry per layer range: pattern
   `^blk\.(a|a+1|…|b-1)\.` → that device's buft, plus `^output(_norm)?\.` → the last shard's buft
   (`llama.h:290`, `llama_model_tensor_buft_override { pattern, buft }`; applied per tensor in
   `llama-model-loader.cpp:1165-1167` before the ratio-based `get_layer_buft_list`). Ratios are left
   at zero: with overrides covering every `blk.` tensor the ratio path only decides KV/compute
   placement, and llama.cpp allocates a layer's KV on the device that owns the layer.
4. `split_mode = LLAMA_SPLIT_MODE_LAYER`, `main_gpu = local_gpu` (or the first RPC device when the
   head has no local shard), `n_gpu_layers = 999`, `use_mmap = true`.

`keryx_llama_load_split` from `tensor-split` is **not** kept: the ratio it forwards is the thing
this design removes.

`src/llama_engine.rs`: `LoadPipelineFn` type + `ensure_loaded_pipeline(gguf, cfg)`; `Engine` gains
`pipeline: Option<PipelineConfig>`; `active_pipeline()` accessor. `tensors()` is unchanged but
unused in shard mode (§3.2 never zero-dups).

#### 3.4 `src/miner.rs` / `src/main.rs`

`launch_gpu_miner`'s PoM branch (`miner.rs:353-425`) is unchanged in shape: a shard device calls
`ensure_installed` → `mine` → `generate_block_if_pom(nonce, idx, tier)` where `idx` is the shard's
`WeightIndex` and `tier` the shard's tier index. `pow.rs::generate_block_if_pom` builds the era's
proof (`build_proof_v2`/v3/v4) over `index.n_chunks`/`index.r_t` — already shard-agnostic.

`main.rs`: parse `--shard` into a `ShardAssignment` registry next to `pom_assignments`; a shard GPU
is *not* given a whole-tier `set_mining_tier`; `lineup_from_assignments` announces the parent
model in `ai:cap` only on the pipeline head (a shard alone cannot serve). Startup computes the
manifest for each requested `(tier, target)` once (cheap: header arithmetic only, no tensor reads —
`open_existing_tree`'s N computation shows the pattern) and builds the shard index lazily in
`ensure_installed_inner` like today.

#### 3.5 `ggml-rpc` bridge — forcing the cut

Mechanism (vendored source, `target/llama-src-b10015`):

- **Holder side** (`--rpc-serve`): run `ggml_backend_rpc_start_server(endpoint, cache_dir,
  n_threads, 1, &local_dev)` in-process (`ggml-rpc.h:27`; `tools/rpc/rpc-server.cpp:340` is the
  reference) exposing exactly the one CUDA device that holds the shard. The **same process** also
  runs the PoM walk over its raw scoped upload — two VRAM copies of the shard on that GPU (raw walk
  copy + llama's RPC buffer). For an 8 GiB target that is 16 GiB on a 24 GB card; accepted cost this
  phase (same trade the local design made in its §4). Zero-dup over the RPC buffer is a follow-up.
- **Weights never cross the wire** — but only if the holder's rpc cache is pre-seeded. The RPC
  client sends `RPC_CMD_SET_TENSOR_HASH` for tensors > 10 MB (`ggml-rpc.cpp:79,468-479`) carrying
  an FNV-1a-64 of the tensor bytes (`fnv_hash`, line 233); the server answers from
  `<cache_dir>/<hash-hex>` if present (`set_tensor_hash`, line 1114; `get_cached_file`). New
  `--rpc-serve` startup step: for every tensor in the shard, `pread` the raw bytes from the local
  GGUF (the holder already has the file — it built the fingerprint from it), compute `fnv_hash`,
  write `<cache_dir>/<hash-hex>` (hard-link or copy). Then the head's load sends only hashes for
  large tensors. Tensors ≤ 10 MB (norms, biases) still travel; that is KBs-to-MBs once at load, not
  per token. Measured in §5.
- **Head side** (`--pipeline`): `keryx_llama_load_pipeline` (§3.3). The head needs the full GGUF on
  **disk** (llama.cpp mmaps it to compute hashes / send small tensors) but not in VRAM. This is a
  real asymmetry vs the "every miner is equal" vision — recorded in §6.
- **Build**: `build.rs` (`main` line 274 area) adds `-DGGML_RPC=ON` next to `-DGGML_CUDA=ON`, and
  `hiveos/build-keryx-llama.sh:50` likewise. The RPC backend is small and has no extra deps.
- **Security posture**: `rpc-server` is unauthenticated and its own banner says never expose it
  (`rpc-server.cpp:301-309`). For this phase it binds to LAN/VPN addresses on the private testnet
  only. Framing it in the cuda-bridge's authenticated transport is phase 2.

What is deliberately NOT used: `tensor_split[]` ratios (`llama-model.cpp:1261-1305`, the
`upper_bound(splits, il/act_gpu_layers)` rule) — a float boundary that can shift by one layer under
rounding and is not something a remote party can verify. `tensor_buft_overrides` is exact.

### 4. `keryx-node` — minimal registration, not new validation

What the node needs, and nothing more: when a block arrives with `proof.tier = t`, `check_pom_proof`
does `tiers.get(t)` (`body_validation_in_isolation.rs:252`) and feeds `tier.chunks` / `tier.root`
to the unchanged verifier (lines 291-301 for v3). For a shard to be accepted, `pom_tiers()` must
return a table that has the shard's `(model_id, root, chunks)` at index `t`.

Change (testnet-only, `consensus/core/src/config/params.rs`):

- New `pub const POM_SHARDS_POC: &[PomTier]` — one row per shard of the POC model (Kimi-Linear-48B
  at the chosen target), `model_id` = the parent model's id, `root`/`chunks` from the manifest.
  Roots are pinned the same way `POM_TIERS_H6` roots were (line 516: "from pom-rt-builder over the
  pinned GGUFs") — here computed by `WeightIndex::build_from_gguf_subset` and cross-checked by the
  existing `#[ignore]` test pattern `weight_index_matches_pinned_root` (`pom.rs:1405`) extended with
  a subset variant.
- New `ForkActivation` field `shard_poc_activation`, `never()` in `MAINNET_PARAMS`, `DEVNET`,
  `SIMNET`; set (e.g. `new(0)`) only in the user's private testnet params. `pom_tiers()` (line 543)
  gets one more branch: when active, return `POM_TIERS_H6 ++ POM_SHARDS_POC` (a `&'static` concat
  built once via `OnceLock`, or a second const slice with the H6 rows repeated). Shard tier indices
  are therefore `5, 6, …` — `proof.tier` is `u8` and `header.pom_tier` (H6) is `u8`, so up to 251
  shard rows fit without touching the wire.

That is the whole node diff. Explicitly **not** consensus logic: `check_pom_proof`, the v3
verifier, `PomProof`, the header, IBD, pruning, `pom_proof_store`/`pom_tier_store` are untouched.
It is a params-table row behind a gate that is `never()` everywhere except the private testnet.

Two side effects to know about (not fix) in this phase:

- Tier reward: `tier_reward_bps(..).get(tier).unwrap_or(TIER_REWARD_BPS_DIVISOR)`
  (`utxo_validation.rs:670`) — a shard tier index beyond the 5-entry schedule silently earns the
  100 % rate. Fine for a private testnet; the real bareme for shards ("mine proportionally to what
  you carry") is phase-2 economics.
- Difficulty: shards mine against the same target as tiers. A shard walk over N_shard chunks has
  the same per-nonce cost as a tier walk (K steps), so a small shard is a cheap way to produce
  blocks on the testnet. Irrelevant for the physics test; must be part of the phase-2 design.

Optional, purely informational (no consensus, no gate): a `/ai:shard:<model_hex>:<idx>/<n>` tag
in the coinbase extra data next to `/ai:cap:` (`src/client/grpc.rs:387-391`, parsed like
`keryx_inference::parse_ai_caps`, `inference/src/lib.rs:95`), so a block explorer / the node's
`GetBlock` verbose data can show which shard a block was mined on. Payload cap is 2048 bytes on
testnet (`max_coinbase_payload_len`), so the tag must stay short. Nice-to-have for the test plan's
log review, not required.

### 5. Relationship with the existing `cuda-bridge`

`cuda-bridge` (`main`, `cuda-bridge/src/{client,server,session,registry,framing,protocol}.rs`)
moves the **PoM walk** of a whole model to a remote GPU and publishes it by name
(`PublishModel`/`BindModel`, `registry.rs::ResidentModel { bases_ptr, prefix_ptr, t_count,
n_total_chunks, tensor_ptrs }`). It does not move inference activations between GPUs; `Generate`
runs the whole model on the server's local engine.

`ggml-rpc` moves **activations** between GPUs for inference; it has nothing to do with the walk.

They are complementary and stay in parallel:

- Miner **with** a GPU holding a shard: local `load_raw_subset` walk (§3.2) + `--rpc-serve` (§3.5).
  No cuda-bridge involved.
- Miner **without** any GPU (the `.21` case): keeps using `cuda-bridge-client` against a
  `--serve-cuda-bridge` host. Later that host can publish shards instead of whole models — the
  registry is keyed by name, so `bridge_model_name(spec) + "#shard<i>"` and `load_raw_subset` on the
  server side is a small follow-up (`pom_gpu_cuda_bridge::publish_zero_dup` would become
  `publish_shard_raw`). Not in this phase.
- Transport reuse: the bridge's framing/proto is a candidate wrapper to give `ggml-rpc` an
  authenticated, signed-per-link channel in phase 2 (each hop signs its activation). Not now.

## Testing plan

Precedent: the cuda-bridge was validated cross-machine on this rig (`.14` = GPU host,
`.21` = client). Same pattern here, after the local phase.

### Phase A — local, two GPUs as two miners (no network hop for PoM)

0. Build off `main` with the changes above; unit tests: shard packing determinism (same GGUF +
   target → same manifest, twice; different target → different boundaries; single-layer-too-big →
   error), `build_from_gguf_subset` root == dense `merkle_root` over the subset leaves (extend the
   `synth_index`-based tests in `pom.rs:1372-1397` with a subset), `--shard` parsing
   (malformed → warn + ignore; same GPU twice → reject).
1. `keryx-miner --print-shards very-high:8` → manifest for Kimi-Linear-48B at 8 GiB. Record each
   shard's `layer_range`, `n_chunks`, `root`. Pin those rows into the node's `POM_SHARDS_POC`,
   build `keryxd`, run the private testnet node (testnet params already have PoM from genesis and
   `pom_v3_activation = 1000`).
2. Two miner processes on the rig, each bound to one GPU (verify CUDA ordinal vs nvidia-smi index
   before starting — this rig has bitten us on that): A `--shard very-high:8:0`, B
   `--shard very-high:8:1` (or `:2`, whichever fits the 3090). Confirm in logs, per process:
   `shard i/n … raw scoped upload`, `GPU miner ready — N=<n_chunks_shard>` equal to the manifest,
   no OOM. Let them mine. Gate: blocks with `proof.tier ∈ {5,6,…}` accepted by the node (no
   `PomUnknownTier`, no `BadPomProofV3`). This is the possession half, end to end, with the real
   verifier.
3. Pipeline, same rig: process B adds `--rpc-serve 127.0.0.1:50052` (cache pre-seeded from its
   GGUF; confirm on the head side that `SET_TENSOR_HASH` hits for every tensor > 10 MB — count
   `SET_TENSOR` bytes sent, must be ≈ only the small tensors). Process A adds
   `--pipeline 127.0.0.1:50052` with layer ranges from the manifest. Run the OPoI generate path
   (`keryx_llama_generate`) on a fixed prompt and measure: `tg` tok/s, per-token wall time, bytes on
   the socket per token. Reference: 103.5 tok/s for the same model native-local split; the delta is
   the loopback RPC cost.
4. Guard check: a deliberately wrong manifest (edit one tensor name) must be refused by the N-guard
   before any block is produced — proves the "silent wrong proof" failure mode is closed.

### Phase B — private testnet, two machines

5. Move process B to the second machine (real NIC, real latency). Same `--shard … --rpc-serve
   <lan-ip>:50052` there; head on `.14` with `--pipeline <lan-ip>:50052`. Each machine runs its
   own `keryxd` on the private testnet (or both point at one). Repeat step 2's gate (blocks from
   both shards accepted by both nodes) and step 3's measurements; now the per-hop latency is the
   number leadership asked for (their reference estimate ~30 ms/hop). Record tok/s vs the Phase A
   loopback number and vs the 103.5 native figure.
6. Failure physics (measure, do not fix): kill the rpc-server mid-generation → observe how the head
   fails (`RPC_STATUS_ASSERT` aborts the process today — document it); restart holder → does the
   head recover without reload? These observations feed the phase-2 "link fails → penalise and
   replace" design.
7. Accepted-block corroboration as in the local design's §6.4: presence of accepted shard blocks
   over a long window is strong evidence; absence in a short window is inconclusive.

## Open questions / risks

- **Execution correctness (Texas's point, unresolved, deliberately deferred).** PoM proves the
  holder *has* the shard bytes; a per-link signature proves *who* produced an activation; nothing
  proves the activation was computed *correctly*. A malicious holder can emit garbage with a valid
  signature and a valid PoM. Leadership's position: real, separate, handled after the pipeline
  exists via deterministic execution + random redundant stages / challenge, not expensive
  cryptographic proofs up front. This design does not narrow or widen that gap; it makes the
  boundary explicit (a shard is a fixed layer range, so a redundant re-execution of one stage has
  a well-defined input/output) — a prerequisite for whatever audit mechanism phase 2 picks. Logged
  here so it is not lost.
- **Head asymmetry.** `ggml-rpc` is one driver + N workers: the head owns the graph, the KV
  scheduling, and needs the full GGUF on disk. In the network vision every link is a peer. Either
  the head role becomes a paid pipeline role in phase 2, or the transport is replaced by a
  peer-to-peer activation relay (each holder runs its own llama instance for its layers — llama.cpp
  does not support that today). Physics from Phase B decides which is worth building.
- **Small tensors still travel** (≤ 10 MB, `HASH_THRESHOLD`). Once at load, not per token; measure
  the total in step 3. If it matters, lowering the threshold is a one-line patch to the vendored
  `ggml-rpc.cpp`.
- **`rpc-server` is unauthenticated** and single-purpose-trusting; private testnet only. Wrapping it
  in an authenticated channel (cuda-bridge framing or otherwise) is phase 2.
- **Double VRAM on the holder** (raw walk copy + RPC buffer). Fine for 8 GiB shards on 24 GB cards;
  zero-dup over the RPC buffer needs the `foreign_device_tensor`-style ownership check applied to
  the RPC buffer's device pointer — follow-up.
- **Economics not addressed on purpose**: shard difficulty/reward bareme (a shard currently gets
  the 100 % fallback rate and the same target as a tier), shard replication incentives, which
  target size(s) the network standardises on. All phase 2, all consensus.
- **Manifest determinism across GGUF revisions**: the manifest is a function of the pinned CID's
  bytes; a re-quantised model is a different `model_id` with its own manifest. No versioning field
  needed in this phase.
- **Branch drift**: implementing on `tensor-split` instead of `main` would miss the v3/v4 walk and
  the cuda-bridge entirely (see "Branch reality"). The 8 `tensor-split` commits are reference
  material, not a base.

## Explicitly not doing

- Not changing `PomProof`, the block header, `verify_pom_proof*`, difficulty, cohorts, strikes,
  escrow, or any mainnet params. The only node diff is a testnet-gated tier-table row set.
- Not building the dynamic pipeline marketplace, per-link payment split, or link
  failure/penalty/replacement — phase 2, with the hard-fork rigour leadership asked for (enumerate
  non-derivable state, IBD/pruning consequences, adversarial testnet).
- Not solving execution correctness of intermediate activations (recorded above as a known,
  deferred risk).
- Not using `tensor_split[]` ratios anywhere; not carrying `--tensor-split` /
  `keryx_llama_load_split` forward.
- Not zero-dupping the walk over llama's RPC buffer on the holder; raw scoped upload only.
- Not making `models.rs` tier assignment "shard-aware" automatically — the operator states the
  shard explicitly via `--shard`, exactly as `--force-model` states a tier.
- Not replacing the cuda-bridge; it stays for the no-GPU client and is not part of this phase's
  test plan.
- Not touching the CUDA walk kernels (`cuda/pom_mine*.cu`): they receive a smaller `bases/prefix`
  table and a smaller N, nothing else.
