#!/usr/bin/env bash
set -euo pipefail
DEBDIR="${HOME}/opt/gcc13-debs"
ROOT="${HOME}/opt/gcc13"
mkdir -p "${DEBDIR}" "${ROOT}"
cd "${DEBDIR}"
apt-get download \
  gcc-13 g++-13 cpp-13 gcc-13-base \
  gcc-13-x86-64-linux-gnu g++-13-x86-64-linux-gnu cpp-13-x86-64-linux-gnu \
  libgcc-13-dev libstdc++-13-dev \
  libasan8 libtsan2 liblsan0 libubsan1 libitm1 libatomic1 libcc1-0 libgomp1

rm -rf "${ROOT}"
mkdir -p "${ROOT}"
shopt -s nullglob
for deb in "${DEBDIR}"/*.deb; do
  echo "extract ${deb}"
  dpkg-deb -x "${deb}" "${ROOT}"
done

GXX="${ROOT}/usr/bin/x86_64-linux-gnu-g++-13"
GCC="${ROOT}/usr/bin/x86_64-linux-gnu-gcc-13"
"${GXX}" --version | head -1

export PATH="/usr/local/cuda/bin:${PATH}"
REPO="/mnt/h/Dev/Mining/KERYX-LABS/keryx-miner"
mkdir -p "${REPO}/dist/ptx"
cd "${REPO}"
for sm in 120 90 89 86 80 75 70 61; do
  echo "=== sm_${sm} ==="
  if nvcc -ptx -O3 -allow-unsupported-compiler -ccbin "${GXX}" \
      -arch="sm_${sm}" cuda/pom_mine.cu -o "dist/ptx/pom_mine_sm${sm}.ptx"; then
    ls -lh "dist/ptx/pom_mine_sm${sm}.ptx"
  else
    if [ "${sm}" = "120" ]; then
      echo "// unavailable" > "dist/ptx/pom_mine_sm${sm}.ptx"
    else
      exit 1
    fi
  fi
done
ls -lh "${REPO}/dist/ptx"
echo "GCC13=${GCC}"
echo "GXX13=${GXX}"
