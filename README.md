# Keryx Miner

A high-performance GPU miner for **Keryx**.

Proof of work is **PoM — Proof of Model**: every nonce walks the weights of the AI model your GPU holds, so mining requires genuine possession of that model, and the tier you prove scales your share of the block reward. The same resident model answers on-chain inference requests (**OPoI** — Optimistic Proof of Inference).

---

## Precompiled Binaries

Download the latest release from the [Releases page](https://github.com/Keryx-Labs/keryx-miner/releases). Each release ships three packages, built by CI from the tagged source:

| Package | For |
|---------|-----|
| `keryx-miner-<version>-linux-amd64.zip` | Linux desktop and servers (glibc 2.34+) |
| `keryx-miner-<version>_hiveos.tar.gz` | HiveOS rigs (install as a custom miner from the archive URL) |
| `keryx-miner-<version>-win64-amd64.zip` | Windows 10/11 |

Every package carries native CUDA kernels for **every GPU from Volta to Blackwell** (sm_70 to sm_120: GTX 16xx, RTX 20/30/40/50, V100, A100, H100, B200) and the CUDA runtime libraries it needs, next to the binary. The only host requirement is an NVIDIA driver from the CUDA 12 era (**525 or newer**; HiveOS 535 included). Pascal (GTX 10xx) cannot hold the smallest model and is not supported.

---

## Build from Source

### Requirements

- Rust + Cargo ([rustup.rs](https://rustup.rs/))
- `protoc` (`protobuf-compiler`), `cmake`, `git` (the inference engine builds llama.cpp from source)
- **CUDA toolkit 12.8 or newer** (releases use 12.9) with a host compiler it supports (Ubuntu 22.04 / GCC 11 works out of the box). `nvcc` is mandatory: `build.rs` compiles the PoM mining kernel into a PTX ladder and builds the inference engine. A 12.2 toolkit builds too with `KERYX_LLAMA_ARCHS="70;75;80;86;89;90"`, but the engine then has no Blackwell kernels and an RTX 50 is refused at startup by the engine check.

### Build

```bash
git clone https://github.com/Keryx-Labs/keryx-miner.git
cd keryx-miner
cargo build --release
```

`nvcc` must be on your `PATH` (or set `NVCC=/path/to/nvcc`); the toolkit root is derived from it. Produces three files in `target/release/`, which must stay together: `keryx-miner`, `libkeryxcuda.so` (the mining plugin) and `libkeryx-llama.so` (the inference engine, loaded at runtime).

The first build clones and compiles llama.cpp for every supported GPU generation (long; cached under `target/`, near no-op afterwards). For a faster development build on one card, restrict the list, e.g. `KERYX_LLAMA_ARCHS="86" cargo build --release` for an RTX 30xx. `KERYX_LLAMA_SKIP=1` skips the engine build entirely; a prebuilt `libkeryx-llama.so` must then sit next to the binary.

Windows builds the same way with the CUDA toolkit and Visual Studio 2022 installed; see `.github/workflows/release.yml` for the exact steps the releases use.

### Runtime libraries

The engine links `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12` from the toolkit you built with. The binary and the engine look for them **next to themselves first**, so copy the three files from `$CUDA_PATH/targets/x86_64-linux/lib/` into `target/release/` (this is what the release packages do), or keep the toolkit's library directory on the system library path. Mining itself needs only `libcuda.so.1`, which comes with the driver.

The two prebuilt PoM fatbins are committed to the repo and embedded as they are; regenerate them only if you change `cuda/pom_mine.cu` (see [cuda/README.md](cuda/README.md)).

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

Inference is not optional. A miner that holds no model cannot prove possession and cannot mine — there is no PoW-only mode.

### Startup checks

Before mining, the miner verifies that the CUDA runtime and the inference engine load, then runs a tiny kernel on **each mining GPU** through the engine. A library without kernels for one of your cards makes the miner stop with a message naming the GPU, instead of aborting on the first inference request hours later (which costs strikes). `--skip-engine-probe` bypasses that last check if you are sure it is wrong.

`--exit-on-disconnect` makes the miner exit instead of reconnecting when the node or pool connection drops while the GPUs are active, for supervised setups (HiveOS, pm2) that restart it.

### Model tiers

One tier, one model. The flag you pick decides which model your GPU must hold, and the tier you prove through PoM (Proof of Model) scales your share of the block reward: the higher the tier, the larger the miner cut.

| Flag | Model | Quant | Min VRAM |
|------|-------|-------|----------|
| `--very-light` | Qwen3.5-9B-abliterated | Q5_K_M | 8 GB+ |
| `--light` | GLM-4-9B-0414 | Q6_K | 12 GB+ |
| *(none, default)* | Gemma-4-12B-abliterated | Q6_K | 16 GB+ |
| `--high` | Qwen3.6-27B | Q4_K_M | 24 GB+ |
| `--very-high` | Kimi-Linear-48B | Q4_K_M | 32 GB+ |

Tiers are **not cumulative**: each one serves exactly one model, and a card that cannot hold the model you asked for falls back to a tier it can actually serve.

On a multi-GPU rig the tier is assigned per card from its VRAM, so a mixed rig runs several tiers side by side. `--force-model` overrides that per GPU, in CUDA driver order:

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS --force-model light,very-high
```

The model is loaded **on demand** when a request arrives and cached between requests. Mining pauses on that GPU during inference, then resumes automatically.

### Faster proof build (optional)

`--resident-tree` holds the full Merkle tree in RAM, so building a block's proof is a lookup instead of an on-the-fly recompute. It needs roughly **2× the model size in system RAM** (not VRAM) — about 13 GB for the smallest tier, ~40 GB for the largest. If the machine doesn't have the RAM, the miner logs a warning and keeps the default disk-backed path, so enabling it is always safe.

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS --resident-tree
```

### Getting the models

Nothing to download by hand: on first run the miner fetches the model for your tier over IPFS and caches it. It looks for the weights at:

```
<directory of the keryx-miner executable>/models/<Model-Name>/model.gguf
```

Point it somewhere else with `--models-dir /path/to/models` (or the `KERYX_MODELS_DIR` environment variable). The path you give is the **root** — the miner still appends `<Model-Name>/model.gguf` under it.

If IPFS is slow or blocked on your network, download the archive and unzip it into that models folder. Keep the folder name exactly as listed below, and use `--ipfs-url` if you would rather point at a different gateway.

| Model | Hugging Face | Direct | Torrent |
|-------|--------------|--------|---------|
| Qwen3.5-9B-abliterated | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Qwen3.5-9B-abliterated.zip) | [zip](https://keryx-labs.com/Qwen3.5-9B-abliterated.zip) | [torrent](https://keryx-labs.com/Qwen3.5-9B-abliterated.zip.torrent) |
| GLM-4-9B-0414 | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/GLM-4-9B-0414.zip) | [zip](https://keryx-labs.com/GLM-4-9B-0414.zip) | [torrent](https://keryx-labs.com/GLM-4-9B-0414.zip.torrent) |
| Gemma-4-12B-abliterated | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Gemma-4-12B-abliterated.zip) | [zip](https://keryx-labs.com/Gemma-4-12B-abliterated.zip) | [torrent](https://keryx-labs.com/Gemma-4-12B-abliterated.zip.torrent) |
| Qwen3.6-27B | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Qwen3.6-27B.zip) | [zip](https://keryx-labs.com/Qwen3.6-27B.zip) | [torrent](https://keryx-labs.com/Qwen3.6-27B.zip.torrent) |
| Kimi-Linear-48B | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Kimi-Linear-48B.zip) | [zip](https://keryx-labs.com/Kimi-Linear-48B.zip) | [torrent](https://keryx-labs.com/Kimi-Linear-48B.zip.torrent) |

A correct manual install looks like this — the miner writes the `.ok` marker itself once it has validated the file, so there is no need to create it:

```
keryx-miner
models/
└── Qwen3.6-27B/
    └── model.gguf
```

If the miner still downloads a model although the folder is there, check your tier flag before anything else: the flag decides **which** model is requested, `--models-dir` only says **where** to look.

### Escrow state durability

Escrow claim state lives in two files next to the miner: `escrow_state.json`, a snapshot rewritten every 10 minutes (and on shutdown), and `escrow_state.journal`, an append-only log of every change since that snapshot, synced within 2 seconds. On start the snapshot is loaded and the journal replayed, so a crash loses at most the last 2 seconds of claims. Both files are written through a same-directory temporary file and atomically installed; an unreadable key or a malformed snapshot stops escrow initialization without replacing the existing files.

On HiveOS, the durable files live outside the replaceable miner package, under `/hive/miners/custom/keryx-miner-state/` (`escrow.key`, `escrow_state.json`, `escrow_state.journal`), with directory mode `0700`.

### All options

```bash
./keryx-miner --help
```

### Block celebration

Block celebration is excluded from default builds so its native audio dependency does not affect release portability. To include it, build with `--features block-celebration`, then pass `--block-celebration` when starting the miner. Both opt-ins are required; animation and sound start disabled otherwise.

When enabled in the terminal UI, it displays a short coin animation and plays a sound on the miner host after a locally mined block is accepted. Press `B` to toggle the animation and `M` to mute or enable the sound independently during the session. If the host has no audio device, mining continues without sound.

The embedded sound is a trimmed adaptation of "Coin Drop" by Universfield under the Pixabay Content License. See `assets/coin-sound-LICENSE.txt` for details.


---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

> "Intelligence is the message. Keryx is the messenger."
