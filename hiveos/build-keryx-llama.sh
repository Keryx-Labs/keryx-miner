#!/usr/bin/env bash
# Shippable `libkeryx-llama.so` — the miner's in-process llama.cpp engine (Phase 2
# in-process engine: hosts the model for the PoM walk zero-dup + serves OPoI inference).
# Built inside the glibc-2.31 container; links the SAME CUDA line as the package it ships in
# (cublas/cudart resolve from the package's lib/ at runtime).
#
# Usage: build-keryx-llama.sh <modern|legacy|pascal> [JOBS]
#   modern: container CUDA 12.9, archs 75;80;86;89;90;120
#   legacy: /tmp/cuda124 (12.4), archs 70;75;80;86;89;90
#   pascal: /tmp/cuda124 (12.4), archs 60;61
# Output: hiveos/dist-<line>/libkeryx-llama.so  (package-line.sh bundles it when present)
#
# llama.cpp PINNED to b10032 — the SAME pin as build-llama-server.sh and the byte-identity
# proof in tools/llama_zerodup_spike. Bump all together, then re-verify the spike.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINE="$1"; JOBS="${2:-16}"
TAG=b10032
case "$LINE" in
  modern) ARCHS="75;80;86;89;90;120"; CUDAMOUNT=(); KCUDA=/usr/local/cuda ;;
  legacy) ARCHS="70;75;80;86;89;90";  CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  pascal) ARCHS="60;61";              CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  no-avx) ARCHS="70;75;80;86;89;90";  CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda
          # BMI2 (shlx/mulx/pdep) traps on pre-Haswell CPUs — the exact audience of this line;
          # -march=nehalem pins a no-AVX/no-BMI ISA baseline for everything the guards miss.
          CPUFLAGS="-DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF"
          CPUFLAGS="$CPUFLAGS -DCMAKE_C_FLAGS=-march=nehalem -DCMAKE_CXX_FLAGS=-march=nehalem"
          WRAPFLAGS="-march=nehalem" ;;
  *) echo "usage: $0 <modern|legacy|pascal|no-avx> [JOBS]"; exit 1 ;;
esac
: "${CPUFLAGS:=}"
: "${WRAPFLAGS:=}"
OUT="$REPO/hiveos/dist-$LINE"
mkdir -p "$OUT"
SRC=/tmp/llama-src-$TAG
if [ ! -d "$SRC" ]; then
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" checkout -- .
for P in "$REPO"/tools/keryx-llama/patches/*.patch; do
  git -C "$SRC" apply "$P"
done

docker run --rm --network host \
  -v "$SRC":/llama -v "$REPO":/repo:ro -v "$OUT":/out "${CUDAMOUNT[@]}" \
  -e KCUDA="$KCUDA" -e ARCHS="$ARCHS" -e JOBS="$JOBS" -e CPUFLAGS="$CPUFLAGS" -e WRAPFLAGS="$WRAPFLAGS" -e KERYX_SPIKE="${KERYX_SPIKE:-0}" \
  keryx-build:offline bash -euo pipefail -c '
    if [ ! -x /tmp/cmk/bin/cmake ]; then
      curl -sL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz \
        | tar xz -C /tmp && mv /tmp/cmake-3.28.6-linux-x86_64 /tmp/cmk
    fi
    export PATH=/tmp/cmk/bin:$KCUDA/bin:$PATH CUDA_HOME=$KCUDA
    B=/tmp/llama-pic-build
    /tmp/cmk/bin/cmake -S /llama -B $B -DGGML_CUDA=ON -DGGML_RPC=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF $CPUFLAGS -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$KCUDA/bin/nvcc
    /tmp/cmk/bin/cmake --build $B --target llama -j "$JOBS"
    g++ -O2 -std=c++17 $WRAPFLAGS -shared -fPIC -fopenmp /repo/tools/keryx-llama/keryx_llama.cpp \
      -I /llama/include -I /llama/ggml/include -I /llama/src -I /llama/common \
      -I $KCUDA/include \
      -Wl,--start-group $B/src/libllama.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
        $B/ggml/src/ggml-rpc/libggml-rpc.a \
        $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a \
      -Wl,--end-group \
      -L$KCUDA/lib64 -L$KCUDA/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt \
      -L$KCUDA/lib64/stubs -L$KCUDA/targets/x86_64-linux/lib/stubs -lcuda \
      -lpthread -ldl -o /out/libkeryx-llama.so
    chmod a+rx /out/libkeryx-llama.so

    # Byte-identity spike (KERYX_SPIKE=1): proves the llama-resident tensor bytes are byte-identical
    # to the on-disk GGUF (canonical name-sorted order — what the PoM walk gathers and R_T pins).
    # Reuses the static libs already built in $B. Run: ./spike <model.gguf> <gpu_ordinal>; exit 0 = OK.
    if [ "${KERYX_SPIKE:-0}" = "1" ]; then
      g++ -O2 -std=c++17 /repo/tools/llama_zerodup_spike/spike.cpp \
        -I /llama/include -I /llama/ggml/include -I /llama/src \
        -I $KCUDA/include \
        -Wl,--start-group $B/src/libllama.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
          $B/ggml/src/ggml-rpc/libggml-rpc.a \
          $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a \
        -Wl,--end-group \
        -L$KCUDA/lib64 -L$KCUDA/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt \
        -L$KCUDA/lib64/stubs -L$KCUDA/targets/x86_64-linux/lib/stubs -lcuda \
        -lpthread -ldl -o /out/spike
      chmod a+rx /out/spike
    fi
  '
echo ">> $LINE libkeryx-llama.so: $(ls -la "$OUT/libkeryx-llama.so" | awk '{print $5}') bytes, glibc=$(objdump -T "$OUT/libkeryx-llama.so" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1), syms=$(nm -D "$OUT/libkeryx-llama.so" | grep -c keryx_llama)"
