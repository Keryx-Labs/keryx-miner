# Keryx Miner

A high-performance GPU miner for **Keryx**.

Proof of work is **PoM — Proof of Model**: every nonce walks the weights of the AI model your GPU holds, so mining requires genuine possession of that model, and the tier you prove scales your share of the block reward. The same resident model answers on-chain inference requests (**OPoI** — Optimistic Proof of Inference).

---

## Precompiled Binaries

Download the latest release from the [Releases page](https://github.com/Keryx-Labs/keryx-miner/releases).

---

## Build from Source

### Requirements

- Rust + Cargo ([rustup.rs](https://rustup.rs/))
- `protoc` (`protobuf-compiler`)
- `cmake` and `git` (the inference engine builds llama.cpp from source)
- **CUDA 12.2 toolkit** — `nvcc` is mandatory, there is no CUDA-less build: `build.rs` compiles the PoM mining kernel into a PTX ladder on every build, and the miner cannot mine without it
- **GCC ≤ 12** (Ubuntu 22.04 / GCC 11 works out of the box); on newer hosts use Option B

CUDA **12.2** specifically: nvcc 12.2 emits code that runs on **NVIDIA driver ≥ 535**, whereas newer toolkits raise that floor (HiveOS commonly ships 535.x). The two prebuilt PoM fatbins are committed to the repo and are simply embedded by the build — you never need to regenerate them unless you change `cuda/pom_mine.cu`, in which case see [cuda/README.md](cuda/README.md).

### Option A — CUDA 12.2 toolkit installed on host (recommended)

Install the toolkit side-by-side (runfile, toolkit-only, no driver), then point the build at it:

```bash
# one-time: install the CUDA 12.2 toolkit to ~/cuda-12.2 (no driver, no root needed)
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
bash cuda_12.2.2_535.104.05_linux.run --silent --toolkit --toolkitpath="$HOME/cuda-12.2" --override

git clone https://github.com/Keryx-Labs/keryx-miner.git
cd keryx-miner
CUDA_ROOT="$HOME/cuda-12.2" CUDA_PATH="$HOME/cuda-12.2" \
  PATH="$HOME/cuda-12.2/bin:$PATH" \
  cargo build --release
```

Produces `target/release/keryx-miner` plus `libkeryx-llama.so` next to it. **Both are needed** — the inference engine is loaded from that shared object at runtime, so keep them together when you move the binary.

The first build clones and compiles llama.cpp (cached under `target/`, near no-op on rebuilds). Its GPU architectures come from `KERYX_LLAMA_ARCHS`, default `75-real;80-real;86-real;89-real;89-virtual`. Official releases use a wider set:

```bash
KERYX_LLAMA_ARCHS="70;75;80;86;89;90" cargo build --release
```

`KERYX_LLAMA_SKIP=1` skips that step entirely, but a prebuilt `libkeryx-llama.so` must then sit next to the binary or no tier can be mined.

### Option B — CUDA 13.x or incompatible gcc on host (build via container)

If your system has CUDA 13.x or gcc 13+ (e.g. Fedora 40+, Ubuntu 25+), build inside a CUDA 12.2 container. The binary runs on the host via driver forward-compatibility.

Requires: [Podman](https://podman.io/) (rootless) or Docker, NVIDIA driver ≥ 535.

```bash
cd keryx-miner
podman run --rm --security-opt label=disable \
  -v "$PWD":/src -w /src \
  -e CARGO_TARGET_DIR=/src/target-cuda \
  docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq \
      curl build-essential cmake git pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null 2>&1
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    . "$HOME/.cargo/env"
    export CUDA_PATH=/usr/local/cuda PROTOC=/usr/bin/protoc
    export KERYX_LLAMA_ARCHS="70;75;80;86;89;90"
    cargo build --release'
```

Binary and `libkeryx-llama.so`: `target-cuda/release/`.

> This path has **not been re-tested since the llama.cpp migration**. `cmake`, `git` and an explicit `KERYX_LLAMA_ARCHS` were added because the container has no GPU to auto-detect; if it fails, Option A on a distro with GCC ≤ 12 is the supported route.

> **Runtime dependencies.** Mining needs only `libcuda.so.1` (the driver). Inference additionally `dlopen`s `libcublas.so.12` and `libcurand.so.10`, so the host must have the matching CUDA 12.2 runtime libs (`libcublas-12-2`, `libcurand-12-2`). On HiveOS the miner installs and registers them automatically on first run; on other hosts install them via your package manager or the CUDA 12.2 toolkit.

> **Plugin compatibility.** Always deploy the miner and its CUDA/OpenCL plugins from the same build or release archive. The loader rejects missing or mismatched plugin ABI versions instead of calling an incompatible Rust trait interface.

> **Blackwell (RTX 50xx).** Nothing to configure: the committed nextgen fatbin carries native `sm_89`/`sm_90`/`sm_100`/`sm_120` SASS, so a 50-series card runs native code with no JIT. This matters — a Blackwell card falling back to JIT from PTX emitted by CUDA 12.2 loses roughly half its hashrate (measured on RTX 5090 and 5080). If you ever rebuild the fatbins yourself, the nextgen one requires CUDA ≥ 12.8; see [cuda/README.md](cuda/README.md).

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

Inference is not optional. A miner that holds no model cannot prove possession and cannot mine — there is no PoW-only mode.

### Model tiers

One tier, one model. The flag you pick decides which model your GPU must hold, and the tier you prove through PoM (Proof of Model) scales your share of the block reward: the higher the tier, the larger the miner cut.

| Flag | Model | Quant | Min VRAM |
|------|-------|-------|----------|
| `--very-light` | Qwen3-8B-abliterated | Q4_K_S | 6 GB+ |
| `--light` | Mistral-7B-v0.3 | Q6_K | 8 GB+ |
| *(none, default)* | GLM-4-9B-0414 | Q6_K | 12 GB+ |
| `--high` | Qwen3.6-27B | Q4_K_M | 24 GB+ |
| `--very-high` | Kimi-Linear-48B | Q4_K_M | 32 GB+ |

Tiers are **not cumulative**: each one serves exactly one model, and a card that cannot hold the model you asked for falls back to a tier it can actually serve.

On a multi-GPU rig the tier is assigned per card from its VRAM, so a mixed rig runs several tiers side by side. `--force-model` overrides that per GPU, in CUDA driver order:

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS --force-model light,very-high
```

The model is loaded **on demand** when a request arrives and cached between requests. Mining pauses on that GPU during inference, then resumes automatically.

### Getting the models

Nothing to download by hand: on first run the miner fetches the model for your tier over IPFS and caches it. It looks for the weights at:

```
<directory of the keryx-miner executable>/models/<Model-Name>/model.gguf
```

Point it somewhere else with `--models-dir /path/to/models` (or the `KERYX_MODELS_DIR` environment variable). The path you give is the **root** — the miner still appends `<Model-Name>/model.gguf` under it.

If IPFS is slow or blocked on your network, download the archive and unzip it into that models folder. Keep the folder name exactly as listed below, and use `--ipfs-url` if you would rather point at a different gateway.

| Model | Hugging Face | Direct | Torrent |
|-------|--------------|--------|---------|
| Qwen3-8B-abliterated | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Qwen3-8B-abliterated.zip) | [zip](https://keryx-labs.com/Qwen3-8B-abliterated.zip) | [torrent](https://keryx-labs.com/Qwen3-8B-abliterated.zip.torrent) |
| Mistral-7B-v0.3 | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Mistral-7B-v0.3.zip) | [zip](https://keryx-labs.com/Mistral-7B-v0.3.zip) | [torrent](https://keryx-labs.com/Mistral-7B-v0.3.zip.torrent) |
| GLM-4-9B-0414 | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/GLM-4-9B-0414.zip) | [zip](https://keryx-labs.com/GLM-4-9B-0414.zip) | [torrent](https://keryx-labs.com/GLM-4-9B-0414.zip.torrent) |
| Qwen3.6-27B | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Qwen3.6-27B.zip) | [zip](https://keryx-labs.com/Qwen3.6-27B.zip) | [torrent](https://keryx-labs.com/Qwen3.6-27B.zip.torrent) |
| Kimi-Linear-48B | [zip](https://huggingface.co/datasets/Keryx-Labs/models/resolve/main/Kimi-Linear-48B.zip) | [zip](https://keryx-labs.com/Kimi-Linear-48B.zip) | [torrent](https://keryx-labs.com/Kimi-Linear-48B.zip.torrent) |

A correct manual install looks like this — the miner writes the `.ok` marker itself once it has validated the file, so there is no need to create it:

Before advertising a model or mining with it, the miner reconstructs the complete GGUF's UnixFS CIDv0 identity and compares it with the pinned model ID. The sparse `pom-tree.bin` cache is also authenticated and bound to that model and cache format; stale or corrupted caches are rebuilt automatically.

```
keryx-miner
models/
└── Qwen3.6-27B/
    └── model.gguf
```

If the miner still downloads a model although the folder is there, check your tier flag before anything else: the flag decides **which** model is requested, `--models-dir` only says **where** to look.

### All options

```bash
./keryx-miner --help
```

### Block celebration

The terminal UI displays a short coin animation and plays a sound through the miner host's default audio output when a block is accepted. Press `B` to toggle the visual and `M` to mute or enable sound independently. No terminal configuration or external media player is required.

On a remote or headless rig, audio plays on the mining host rather than the viewing terminal. If no audio device is available, mining continues normally without sound.

The embedded sound is a trimmed adaptation of "Coin Drop" by Universfield under the Pixabay Content License. See `coin-sound-LICENSE.txt` in the release archive.


---

### HiveOS escrow state

HiveOS stores `escrow.key` and `escrow_state.json` in `/hive/miners/custom/keryx-miner-state`, outside versioned miner packages. Upgrades migrate legacy package-local files without overwriting an existing durable copy. Back up `escrow.key`: it is required to claim accrued OPoI rewards.

Custom `--escrow-key-file` and `--escrow-state-file` values remain supported and take precedence over the HiveOS defaults.

## Dev Fund

2% of mining rewards support development by default.

```bash
--devfund-percent XX.YY
```

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

> "Intelligence is the message. Keryx is the messenger."
