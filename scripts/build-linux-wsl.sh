#!/usr/bin/env bash
# Full Linux/HiveOS release build inside WSL.
set -euo pipefail
export PATH="/usr/local/cuda/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export PROTOC="${PROTOC:-${HOME}/.local/bin/protoc}"
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export NVCC=/usr/local/cuda/bin/nvcc
export POM_SM_LIST="${POM_SM_LIST:-120,90,89,86,80,75,70,61}"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-89}"
# Include Blackwell real arch when toolkit supports it; virtual 89 remains a JIT fallback.
export KERYX_LLAMA_ARCHS="${KERYX_LLAMA_ARCHS:-75-real;80-real;86-real;89-real;89-virtual;120-real}"
export NUM_JOBS="${NUM_JOBS:-8}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target-linux}"
# CUDA 12.8 rejects host gcc > 14 (Ubuntu 25+/WSL often ships 15).
export NVCC_FLAGS="${NVCC_FLAGS:--allow-unsupported-compiler}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

echo "protoc=$(${PROTOC} --version)"
echo "nvcc=$(${NVCC} --version | head -1)"
echo "gcc=$(gcc --version | head -1)"
echo "Starting Linux build at $(date) → ${CARGO_TARGET_DIR}"

# llama.cpp/nvcc also needs the unsupported-compiler escape on gcc 15+.
export CMAKE_CUDA_FLAGS="-allow-unsupported-compiler ${CMAKE_CUDA_FLAGS:-}"
export CUDAFLAGS="-allow-unsupported-compiler ${CUDAFLAGS:-}"

cargo build --release --bin keryx-miner
echo "BIN_OK $(date)"
cargo build --release -p keryxcuda
echo "DONE $(date)"
ls -lh "${CARGO_TARGET_DIR}/release/keryx-miner" \
       "${CARGO_TARGET_DIR}/release/libkeryx-llama.so" \
       "${CARGO_TARGET_DIR}/release/libkeryxcuda.so"
