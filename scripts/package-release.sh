#!/usr/bin/env bash
# Build release archives with runtime deps bundled (CUDA libs + Kubo IPFS).
#
# Usage:
#   ./scripts/package-release.sh windows   # from MSYS/Git Bash on Windows (binaries already built)
#   ./scripts/package-release.sh hiveos    # from Linux/WSL (binaries already built)
#   ./scripts/package-release.sh all       # package whatever local artifacts exist
#
# Env:
#   VERSION          default: Cargo.toml version + -OPoI
#   DIST_DIR         default: dist
#   WIN_TARGET_DIR   default: target/release
#   LINUX_TARGET_DIR default: target/release  (or target-cuda/release)
#   CUDA_WIN_BIN     Windows CUDA bin dir (auto-detect 12.x)
#   KUBO_VERSION     default: 0.41.0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VERSION="${VERSION:-$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)-OPoI}"
DIST_DIR="${DIST_DIR:-$REPO/dist}"
WIN_TARGET_DIR="${WIN_TARGET_DIR:-$REPO/target/release}"
LINUX_TARGET_DIR="${LINUX_TARGET_DIR:-$REPO/target/release}"
if [[ ! -x "$LINUX_TARGET_DIR/keryx-miner" && -x "$REPO/target-cuda/release/keryx-miner" ]]; then
  LINUX_TARGET_DIR="$REPO/target-cuda/release"
fi
KUBO_VERSION="${KUBO_VERSION:-0.41.0}"
DEPS_CACHE="${DEPS_CACHE:-$REPO/dist/deps-cache}"
mkdir -p "$DIST_DIR" "$DEPS_CACHE"

log() { echo "==> $*"; }

fetch() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    return 0
  fi
  log "Downloading $(basename "$out")"
  curl -fsSL --retry 5 --retry-delay 2 -o "$out.partial" "$url"
  mv "$out.partial" "$out"
}

ensure_kubo() {
  local os="$1" # windows|linux
  local arch="amd64"
  local ext archive bin
  if [[ "$os" == "windows" ]]; then
    ext="zip"
    archive="$DEPS_CACHE/kubo_v${KUBO_VERSION}_windows-${arch}.zip"
    bin="ipfs.exe"
  else
    ext="tar.gz"
    archive="$DEPS_CACHE/kubo_v${KUBO_VERSION}_linux-${arch}.tar.gz"
    bin="ipfs"
  fi
  fetch "https://dist.ipfs.tech/kubo/v${KUBO_VERSION}/kubo_v${KUBO_VERSION}_${os}-${arch}.${ext}" "$archive"
  local extract_dir="$DEPS_CACHE/kubo-${os}"
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  if [[ "$ext" == "zip" ]]; then
    unzip -qo "$archive" -d "$extract_dir"
  else
    tar -xzf "$archive" -C "$extract_dir"
  fi
  local found
  found="$(find "$extract_dir" -type f -name "$bin" | head -1)"
  [[ -n "$found" ]] || { echo "ERROR: $bin not found in kubo archive"; exit 1; }
  chmod a+rx "$found" 2>/dev/null || true
  echo "$found"
}

copy_win_cuda() {
  local dest="$1"
  local cuda_bin="${CUDA_WIN_BIN:-}"
  if [[ -z "$cuda_bin" ]]; then
    for v in v12.8 v12.6 v12.5 v12.4 v12.2 v12.1 v12.0; do
      local candidate="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/${v}/bin"
      if [[ -f "$candidate/cublas64_12.dll" ]]; then
        cuda_bin="$candidate"
        break
      fi
    done
  fi
  [[ -n "$cuda_bin" && -f "$cuda_bin/cublas64_12.dll" ]] || {
    echo "ERROR: Windows CUDA 12 bin dir with cublas64_12.dll not found (set CUDA_WIN_BIN)"
    exit 1
  }
  log "Bundling Windows CUDA libs from $cuda_bin"
  cp -f "$cuda_bin/cublas64_12.dll" "$dest/"
  cp -f "$cuda_bin/cublasLt64_12.dll" "$dest/"
  # Optional helpers some hosts still dlopen
  [[ -f "$cuda_bin/cudart64_12.dll" ]] && cp -f "$cuda_bin/cudart64_12.dll" "$dest/" || true
  [[ -f "$cuda_bin/curand64_10.dll" ]] && cp -f "$cuda_bin/curand64_10.dll" "$dest/" || true
}

ensure_linux_cuda_libs() {
  local dest="$1"
  local stamp="$DEPS_CACHE/cuda12.2-libs"
  mkdir -p "$stamp"
  if [[ ! -f "$stamp/libcublas.so.12" ]]; then
    log "Fetching CUDA 12.2 runtime debs (Ubuntu 22.04)"
    local base="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
    # Pin known 12.2 package filenames; refresh via Packages index if missing.
    local pkgs=(
      "libcublas-12-2_12.2.5.6-1_amd64.deb"
      "cuda-cudart-12-2_12.2.140-1_amd64.deb"
      "libcurand-12-2_10.3.3.141-1_amd64.deb"
    )
    local tmp="$DEPS_CACHE/cuda-debs"
    mkdir -p "$tmp"
    for pkg in "${pkgs[@]}"; do
      fetch "$base/$pkg" "$tmp/$pkg"
      rm -rf "$tmp/extract"
      mkdir -p "$tmp/extract"
      if command -v dpkg-deb >/dev/null 2>&1; then
        dpkg-deb -x "$tmp/$pkg" "$tmp/extract"
      else
        (
          cd "$tmp/extract"
          ar x "$tmp/$pkg"
          tar -xf data.tar.* 2>/dev/null || tar -xf data.tar
        )
      fi
      # Collect versioned shared objects.
      find "$tmp/extract" -type f \( -name 'libcublas.so.12*' -o -name 'libcublasLt.so.12*' \
        -o -name 'libcudart.so.12*' -o -name 'libcurand.so.10*' \) -exec cp -a {} "$stamp/" \;
    done
    # Normalize unversioned soname symlinks expected by dlopen.
    (
      cd "$stamp"
      [[ -e libcublas.so.12 ]] || ln -sf "$(ls libcublas.so.12.* 2>/dev/null | head -1)" libcublas.so.12
      [[ -e libcublasLt.so.12 ]] || ln -sf "$(ls libcublasLt.so.12.* 2>/dev/null | head -1)" libcublasLt.so.12
      [[ -e libcudart.so.12 ]] || ln -sf "$(ls libcudart.so.12.* 2>/dev/null | head -1)" libcudart.so.12
      [[ -e libcurand.so.10 ]] || ln -sf "$(ls libcurand.so.10.* 2>/dev/null | head -1)" libcurand.so.10
    )
  fi
  log "Bundling Linux CUDA 12.2 runtime libs"
  cp -a "$stamp"/. "$dest/"
}

package_windows() {
  local out_name="keryx-miner-v${VERSION}-win64-amd64"
  local stage="$DIST_DIR/$out_name"
  rm -rf "$stage"
  mkdir -p "$stage"

  [[ -f "$WIN_TARGET_DIR/keryx-miner.exe" ]] || { echo "ERROR: missing $WIN_TARGET_DIR/keryx-miner.exe"; exit 1; }
  [[ -f "$WIN_TARGET_DIR/keryx-llama.dll" ]] || { echo "ERROR: missing $WIN_TARGET_DIR/keryx-llama.dll"; exit 1; }

  cp -f "$WIN_TARGET_DIR/keryx-miner.exe" "$stage/"
  cp -f "$WIN_TARGET_DIR/keryx-llama.dll" "$stage/"
  if [[ -f "$WIN_TARGET_DIR/keryxcuda.dll" ]]; then
    cp -f "$WIN_TARGET_DIR/keryxcuda.dll" "$stage/"
  elif [[ -f "$WIN_TARGET_DIR/keryxcuda.dll" ]]; then
    cp -f "$WIN_TARGET_DIR/keryxcuda.dll" "$stage/"
  else
    # plugin may live under plugins target
    local plug
    plug="$(find "$REPO/target" -name 'keryxcuda.dll' 2>/dev/null | head -1 || true)"
    [[ -n "$plug" ]] && cp -f "$plug" "$stage/" || echo "WARN: keryxcuda.dll not found"
  fi

  copy_win_cuda "$stage"
  local ipfs
  ipfs="$(ensure_kubo windows)"
  cp -f "$ipfs" "$stage/ipfs.exe"

  # Convenience launcher
  cat > "$stage/mine.bat" <<'EOF'
@echo off
echo ============================================================
echo = Edit this file and set your keryx: address / pool host  =
echo ============================================================
:start
keryx-miner.exe --mining-address keryx:YOUR_ADDRESS
goto start
EOF

  local zip_path="$DIST_DIR/${out_name}.zip"
  rm -f "$zip_path"
  (
    cd "$DIST_DIR"
    if command -v 7z >/dev/null 2>&1; then
      7z a -tzip -mx=9 "${out_name}.zip" "$out_name" >/dev/null
    else
      powershell.exe -NoProfile -Command "Compress-Archive -Path '$out_name' -DestinationPath '${out_name}.zip' -Force"
    fi
  )
  log "Windows package: $zip_path"
  ls -lh "$zip_path"
}

package_hiveos() {
  local out_name="keryx-miner"
  local stage="$DIST_DIR/$out_name"
  rm -rf "$stage"
  mkdir -p "$stage"

  [[ -x "$LINUX_TARGET_DIR/keryx-miner" || -f "$LINUX_TARGET_DIR/keryx-miner" ]] \
    || { echo "ERROR: missing $LINUX_TARGET_DIR/keryx-miner"; exit 1; }

  cp -f "$LINUX_TARGET_DIR/keryx-miner" "$stage/keryx-miner"
  chmod a+rx "$stage/keryx-miner"
  if [[ -f "$LINUX_TARGET_DIR/libkeryx-llama.so" ]]; then
    cp -f "$LINUX_TARGET_DIR/libkeryx-llama.so" "$stage/"
  else
    echo "ERROR: missing libkeryx-llama.so next to linux binary"; exit 1
  fi
  local cuda_plugin
  cuda_plugin="$(find "$LINUX_TARGET_DIR" "$REPO/target" -name 'libkeryxcuda.so' 2>/dev/null | head -1 || true)"
  [[ -n "$cuda_plugin" ]] && cp -f "$cuda_plugin" "$stage/" || echo "WARN: libkeryxcuda.so not found"

  ensure_linux_cuda_libs "$stage"
  local ipfs
  ipfs="$(ensure_kubo linux)"
  cp -f "$ipfs" "$stage/ipfs"
  chmod a+rx "$stage/ipfs"

  # HiveOS scripts + stable shared-models manifest (keeps CUSTOM_MINER_DIR resolution).
  local ver_tag="$VERSION"
  ver_tag="${ver_tag%-OPoI}"
  sed "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=${ver_tag}-OPoI/" \
    "$REPO/integrations/hiveos/h-manifest.conf" > "$stage/h-manifest.conf"
  cp -f "$REPO/integrations/hiveos/"h-*.sh "$stage/"
  chmod a+rx "$stage/"h-*.sh

  # HIVEOS_README: keryx-miner-v<version>_OPoI_hiveos.tar.gz
  local tarball="$DIST_DIR/keryx-miner-v${ver_tag}_OPoI_hiveos.tar.gz"
  rm -f "$tarball"
  tar -czf "$tarball" -C "$DIST_DIR" "$out_name"

  # Also ship a plain linux zip with the same payload (non-HiveOS)
  local linux_name="keryx-miner-v${VERSION}-linux-gnu-amd64"
  rm -rf "$DIST_DIR/$linux_name"
  mkdir -p "$DIST_DIR/$linux_name"
  cp -a "$stage"/. "$DIST_DIR/$linux_name/"
  # Hive helper scripts are harmless for bare linux; keep binary deps identical.
  local linux_zip="$DIST_DIR/${linux_name}.zip"
  rm -f "$linux_zip" "$DIST_DIR/${linux_name}.tar.gz"
  tar -czf "$DIST_DIR/${linux_name}.tar.gz" -C "$DIST_DIR" "$linux_name"
  (
    cd "$DIST_DIR"
    if command -v zip >/dev/null 2>&1; then
      zip -qr "${linux_name}.zip" "$linux_name"
    elif command -v 7z >/dev/null 2>&1; then
      7z a -tzip -mx=9 "${linux_name}.zip" "$linux_name" >/dev/null
    else
      echo "WARN: zip/7z unavailable — skipped ${linux_name}.zip"
    fi
  )

  log "HiveOS package: $tarball"
  ls -lh "$tarball" "$DIST_DIR/${linux_name}.tar.gz" 2>/dev/null || true
}

cmd="${1:-all}"
case "$cmd" in
  windows) package_windows ;;
  hiveos|linux) package_hiveos ;;
  all)
    if [[ -f "$WIN_TARGET_DIR/keryx-miner.exe" ]]; then package_windows; fi
    if [[ -f "$LINUX_TARGET_DIR/keryx-miner" ]]; then package_hiveos; fi
    if [[ ! -f "$WIN_TARGET_DIR/keryx-miner.exe" && ! -f "$LINUX_TARGET_DIR/keryx-miner" ]]; then
      echo "ERROR: no built binaries found under $WIN_TARGET_DIR or $LINUX_TARGET_DIR"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {windows|hiveos|all}"
    exit 1
    ;;
esac

log "Done. Artifacts in $DIST_DIR"
ls -lh "$DIST_DIR"/*.{zip,tar.gz} 2>/dev/null || ls -lh "$DIST_DIR"
