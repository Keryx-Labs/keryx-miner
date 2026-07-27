#!/usr/bin/env bash
# Build Linux/HiveOS release artifacts inside CUDA 12.8 + Ubuntu 22.04 (glibc/gcc compatible).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${KERYX_BUILD_IMAGE:-docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04}"
TARGET_DIR="${CARGO_TARGET_DIR:-target-linux}"

docker pull "${IMAGE}"

docker run --rm --network host \
  -v "${REPO}:/src" -w /src \
  -e CARGO_TARGET_DIR="/src/${TARGET_DIR}" \
  -e POM_SM_LIST=120,90,89,86,80,75,70,61 \
  -e CUDA_COMPUTE_CAP=89 \
  -e KERYX_LLAMA_ARCHS='75-real;80-real;86-real;89-real;120-real;89-virtual' \
  -e NUM_JOBS="${NUM_JOBS:-8}" \
  -e NVCC=/usr/local/cuda/bin/nvcc \
  -e CUDA_PATH=/usr/local/cuda \
  -e CUDA_HOME=/usr/local/cuda \
  "${IMAGE}" \
  bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq curl build-essential pkg-config libssl-dev ca-certificates \
      protobuf-compiler cmake git >/dev/null
    if ! command -v cargo >/dev/null; then
      curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    fi
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    export PATH="/usr/local/cuda/bin:$PATH"
    echo "nvcc=$(nvcc --version | head -4 | tr "\n" " ")"
    echo "gcc=$(gcc --version | head -1)"
    cargo build --release --bin keryx-miner
    cargo build --release -p keryxcuda
    ls -lh "/src/'"${TARGET_DIR}"'/release/keryx-miner" \
           "/src/'"${TARGET_DIR}"'/release/libkeryx-llama.so" \
           "/src/'"${TARGET_DIR}"'/release/libkeryxcuda.so"
  '

echo "Linux build complete → ${REPO}/${TARGET_DIR}/release"
