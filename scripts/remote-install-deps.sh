#!/usr/bin/env bash
# Permanent compile deps for keryx-miner on Ubuntu 22.04 / HiveOS.
set -euo pipefail

echo "==> apt packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades -qq \
  build-essential pkg-config libssl-dev ca-certificates curl git \
  cmake ninja-build protobuf-compiler ocl-icd-opencl-dev \
  wget xz-utils unzip zip tar gcc g++ make python3 || \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential pkg-config libssl-dev ca-certificates curl git \
  cmake ninja-build protobuf-compiler \
  wget xz-utils unzip zip tar gcc g++ make python3

echo "==> rustup (stable)"
if ! command -v rustc >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
  fi
fi
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
rustup default stable
grep -q 'cargo/env' "$HOME/.bashrc" 2>/dev/null || echo '. "$HOME/.cargo/env"' >> "$HOME/.bashrc"

echo "==> CUDA toolkit 12.8 (nvcc) if missing"
if [ ! -x /usr/local/cuda/bin/nvcc ] && [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  cd /tmp
  # apt toolkit from NVIDIA repo — keeps driver untouched (HiveOS already has 595)
  if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ] && [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
    sudo dpkg -i cuda-keyring.deb
    rm -f cuda-keyring.deb
  fi
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-8
  sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
fi

# Permanent env for all shells
sudo tee /etc/profile.d/keryx-build.sh >/dev/null <<'EOF'
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export POM_SM_LIST=120,90,89,86,80,75,70,61
export CUDA_COMPUTE_CAP=89
export KERYX_LLAMA_ARCHS='75-real;80-real;86-real;89-real;120-real;89-virtual'
EOF
sudo chmod 644 /etc/profile.d/keryx-build.sh

# shellcheck disable=SC1091
. /etc/profile.d/keryx-build.sh
. "$HOME/.cargo/env"

echo "==> versions"
rustc --version
cargo --version
gcc --version | head -1
cmake --version | head -1
protoc --version
nvcc --version | sed -n '4p'
nvidia-smi -L | head -3
echo "DEPS_OK"
