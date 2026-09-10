# Shard-Possession PoM PoC ("network-fatia")

Status: **PROOF OF CONCEPT — incomplete, exploratory.** Not activated on Mainnet or Testnet.
Only exercised on an isolated, private Simnet build created for this work. Nothing in this
branch changes consensus on any shared network; `shard_poc_activation` is `never()` everywhere
except the private `SIMNET_PARAMS` added here.
Branch: `shard-poc` in both `keryx-miner` and `keryx-node`.
Companion repo: this document is duplicated verbatim in both `keryx-node` and `keryx-miner`
(the two halves of the PoC live in separate repos and have to be read together).

## 1. Motivation

PoM tiers currently require one GPU to hold and walk an entire model's weights. As served
models grow (e.g. a 48B-parameter tier), that stops fitting on a single consumer GPU. This PoC
explores an alternative: split a model into fixed-byte-size **shards** along layer boundaries,
and let a GPU prove possession of only its own shard instead of the whole model — so a model
too large for any one card can still be mined, cooperatively, across several cards.

This is a possession-proof question, not a serving question: it does **not** attempt to make a
shard device capable of running real OPoI inference over the full model (see §6, "Not done").

## 2. What changed — miner side (`keryx-miner`, branch `shard-poc`)

- `src/shard.rs`: `ShardManifest { index: u16, layer_lo: u32, layer_hi: u32, tensor_names:
  Vec<String> }` and `pack_shards(meta, target_bytes) -> Vec<ShardManifest>` — splits a GGUF's
  tensors into layer-range shards, each close to `target_bytes` on disk.
- `src/pom.rs`: `WeightIndex::build_from_gguf_subset(...)` — builds a host chunk index (the same
  Merkle-style structure used for a whole-tier index) over only a shard's tensor subset.
- `src/pom_gpu.rs`:
  - `load_raw_subset(...)` — raw GPU upload restricted to a shard's tensor subset only. Shard
    devices never run the llama engine and never zero-dup against a resident whole-model tree —
    always a raw scoped upload.
  - Per-device shard registry (`set_shard_for_device` / `shard_for_device`) and
    `ensure_shard_installed_inner`, which contains the **N-guard**: after installing, it compares
    the GPU gather's chunk count against the host index's chunk count and refuses to install a
    miner (`"gather N != shard index N — refusing to mine"`) on any mismatch, rather than
    silently producing a proof over the wrong data. Verified live on real hardware — see §5.
- `src/models.rs`: `pom_shard_tier_index(model_id, target_bytes, shard_index, daa)` — maps a
  shard assignment to a PoM tier index. In this PoC: `5 + shard_index` for a 4-shard, 8 GB split
  of the `very-high` tier (tiers 5, 6, 7, 8). This offset is a **hand-kept convention**, not
  negotiated by any protocol — see §6.
- `src/cli.rs` / `src/main.rs`: `--shard TIER:TARGET_GIB:IDX[,...]` (mine one fixed-size shard
  instead of the whole tier) and `--print-shards TIER:TARGET_GIB` (compute and print a shard
  manifest, then exit — used to produce the exact values pasted into the node's shard tier
  table; see §3).
- Two real defects found and fixed while running this live (`src/client/grpc.rs`):
  1. The existing "OPoI is mandatory: refuse to mine with no model loaded" gate had no exemption
     for shard-only devices, which never load a full model by design. Fixed with
     `pom_gpu::any_shard_devices_active()` and a gate exemption for processes that only run
     shard devices — no change for ordinary whole-tier mining.
  2. A sole-producer / idle chain (this PoC's Simnet, with zero peers) never re-requested a
     fresh block template outside of an in-flight inference cycle, so mining stalled at DAA 0
     indefinitely. Fixed with an unconditional ~500ms fallback re-poll.
- `examples/shard_walk_poc.rs` and `examples/shard_guard_poc.rs` — see §5.

## 3. What changed — node side (`keryx-node`, branch `shard-poc`)

- `consensus/core/src/config/params.rs`:
  - `POM_SHARDS_POC`: a shard tier table (`crate::pom::PomTier { model_id, root, chunks }` per
    shard) — real, measured values pasted verbatim from `keryx-miner --print-shards
    very-high:8` run against the real model file (see §4). Gated behind `shard_poc_activation`.
  - `shard_poc_activation`: a new fork-activation gate. `never()` on `MAINNET_PARAMS` and the
    shared public `TESTNET_PARAMS` — this PoC never touches either. Set to `new(1)` only on
    `SIMNET_PARAMS`, alongside a full, internally-consistent PoM/OPoI/H-series activation
    profile built specifically for a from-genesis private Simnet (every field's reasoning is
    documented inline in the diff, including one corrected discrepancy against the public
    `TESTNET_PARAMS`'s `pom_v4_activation`/`h10_activation` values — the real miner binary gates
    those at DAA 500 under `--testnet`, not DAA 1 as `TESTNET_PARAMS` currently says; that
    pre-existing drift is dormant on the real Testnet only because its tip is already past DAA
    500, and is not copied into `SIMNET_PARAMS`).

## 4. Environment used for validation

- Hardware: 2x NVIDIA GPU on one rig — CUDA ordinal 0 = RTX 3090 (24 GB), CUDA ordinal 1 = RTX
  4090 (24 GB). This rig's CUDA driver ordinal does not match the `nvidia-smi` device index by
  default; validation used `CUDA_DEVICE_ORDER=PCI_BUS_ID` to pin them.
- Model: Kimi-Linear-48B, Q4_K_M quantization, ~29.7 GB GGUF on disk — too large to fit whole on
  either single card in this rig.
- Split: 8 GB target per shard → 4 shards. Real manifest (captured via `--print-shards
  very-high:8`):

  | Shard | Layers | Chunks | Merkle root |
  |---|---|---|---|
  | 0 | [0, 7) | 219,852,216 | `a73a38fe1ef0cddc6e108eca25eb1de1adcdc7a65c01f87058530ac06df669b4` |
  | 1 | [7, 14) | 238,829,076 | `db22407efc04f5d261498afc7acb397c87e692699484a9bcf2a747636adda9e5` |
  | 2 | [14, 21) | 243,638,100 | `9c3aed4cc240baea0928f8282ea91ce6b0cd275f8286c45c440017c457f8cb8d` |
  | 3 | [21, 27) | 225,674,672 | `233fdb1f9b551bb7895a0fca931ce5eddd16d87a12980f7e4361e020cfe29ff2` |

- Network: a private, isolated Simnet instance (no DNS seeders, own genesis) — never the shared
  public Testnet or Mainnet.
- Build: Windows, MSVC toolchain via the Visual Studio Developer Shell;
  `KERYX_LLAMA_SKIP=1` (skips only the llama.cpp CMake library build used for OPoI inference
  serving — irrelevant to the PoM CUDA kernel path this PoC exercises).

## 5. Results — proven on real hardware

1. **In-process concept proof** (`keryx-miner/examples/shard_walk_poc.rs`): both GPUs
   independently packed their shard's manifest, installed it (real raw scoped upload), ran a
   real GPU PoM walk against a synthetic (trivially easy) target, built a real `PomProofV3`
   witness from the walk, and self-verified it with the same `verify_proof_v3` function a node
   uses — no live node needed for this step. Both self-verified `true`.
2. **Live network acceptance** (real gRPC mining loop, against the private Simnet node running
   the `SIMNET_PARAMS` profile from §3): the node genuinely accepted shard-tier PoM blocks over
   the real block-submission path, sustained, with zero rejections — 817 blocks on shard 0
   (GPU0/3090, ~1.05M h/s, `v4 tensor-core` kernel) and 413 blocks on shard 1 (GPU1/4090, ~1.68M
   h/s), 1,230 combined, 0 rejected.
3. **Guard regression check** (`keryx-miner/examples/shard_guard_poc.rs`): installs a shard with
   its correct manifest (succeeds), then reinstalls after dropping one tensor name from the
   manifest — the cached host index still reflects the correct (larger) chunk count, so the
   second gather's chunk count diverges and the N-guard correctly refuses to install a miner,
   confirming a corrupted/wrong manifest cannot silently produce an acceptable proof.

## 6. Explicitly NOT done — open, incomplete by design

This is a concept proof, not a finished feature. In particular:

- **No distributed OPoI inference serving.** A shard device still cannot itself serve real
  inference over the full model — the OPoI-mandatory-mining gate exemption in §2 only lets a
  shard-only process skip that requirement for *mining*; it does not make inference work. A
  real ggml-rpc-style pipeline across shards (already scoped separately as spec §3.3–3.5) is
  future work, not part of this PoC.
- **The pre-PoM bootstrap block still needs a whole model.** A fresh chain's very first block
  (below `pom_activation`) must be mined under the legacy kHeavyHash+OPoI path, which needs one
  device holding the *entire* model — a pure shard-only rig has no path through that block on
  its own.
- **The shard→tier offset (`5 + shard_index`) is a hand-kept convention**, not negotiated by any
  protocol. The node's `POM_SHARDS_POC` row order must match the miner's shard numbering by
  agreement, not by any on-chain mechanism. A mismatch here is silently wrong, not rejected —
  this is a real weakness a production design would need to close.
- **Not tested beyond this exact configuration**: 2 GPUs, 4 shards, one model
  (Kimi-Linear-48B), one rig. Behavior with more shards, more devices, a different model, or
  cross-machine (rather than cross-GPU-on-one-machine) sharding is unexplored.
- **No reward/economics design.** A shard-only miner's partial possession currently just reuses
  the ordinary tier-based reward table positionally. Whether that's fair or game-theoretically
  sound versus a whole-tier miner's possession was not evaluated.
- **`SIMNET_PARAMS` activation is deliberately Simnet-only.** Extending any of this to a shared
  network (Testnet or Mainnet) is a separate, much larger decision — governance, rollout timing,
  and backward compatibility are all out of scope here.

## 7. Where to look / how to reproduce

- `keryx-node`, branch `shard-poc`: commit `76aa3c6f` (shard tier table + gating scaffold, then
  a placeholder row), commit `ecb0f09e` (real manifest data + the `SIMNET_PARAMS` activation
  profile from §3).
- `keryx-miner`, branch `shard-poc`: commits `3cf1bd4` (shard registry + install path), `15649ec`
  (`--shard`/`--print-shards` CLI), `ba2aa2f` (wiring into tier assignment and the mining loop),
  `b9c0990` (OPoI-gate exemption for shard devices), `183d3e5` (idle-chain template re-poll
  fix), `645fd5d` (the two diagnostic examples in §5).
- To rerun the in-process proof or the guard check: `cargo run --example shard_walk_poc
  --features cuda` / `cargo run --example shard_guard_poc --features cuda` from the miner repo
  (both accept `--gguf-path`, `--target-bytes`, and `--mainnet` flags; default to the
  Kimi-Linear-48B path and testnet-style DAA gates used above).
