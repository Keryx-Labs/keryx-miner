#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO}/dist/ptx"
mkdir -p "${OUT}"
cd "${REPO}"
for sm in 120 90 89 86 80 75 70 61; do
  echo "=== sm_${sm} ==="
  if "${NVCC}" -ptx -O3 -allow-unsupported-compiler -arch="sm_${sm}" \
      cuda/pom_mine.cu -o "${OUT}/pom_mine_sm${sm}.ptx"; then
    ls -lh "${OUT}/pom_mine_sm${sm}.ptx"
  else
    echo "FAILED sm_${sm}"
    if [ "${sm}" = "120" ]; then
      echo "// pom_mine sm_120 unavailable" > "${OUT}/pom_mine_sm${sm}.ptx"
    else
      exit 1
    fi
  fi
done
ls -lh "${OUT}"
