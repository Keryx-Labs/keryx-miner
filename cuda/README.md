# PoM Fatbin Build Instructions

This directory contains the CUDA source for PoM mining kernel builds.

The miner expects two prebuilt PoM fatbins:

- `cuda/pom_mine_legacy.fatbin` — cc < 8.9 (GTX 10xx, Volta, Turing, Ampere)
- `cuda/pom_mine_nextgen.fatbin` — cc >= 8.9 (Ada, Hopper, Blackwell)

They are embedded at build time (`build.rs` stages them into `OUT_DIR`) and loaded at runtime
with a per-GPU fatbin-first ladder: nextgen-first on cc >= 8.9, legacy-first otherwise, with a
cross fallback. Together they cover SASS from sm_61 to sm_120, so no supported card ever needs
to JIT from PTX.

**Both fatbins are committed to the repository.** Do not regenerate them as part of a release
build: shipping fatbins that differ from the ones in the repo makes a kernel regression
invisible between a source build and a release archive. Regenerate them only when
`cuda/pom_mine.cu` changes, then commit the result.

## Prerequisites

- CUDA 12.9 `nvcc` at `/home/slash/cuda-129/bin/nvcc` (nextgen fatbin build)
- CUDA 12.2 `nvcc` at `/home/slash/cuda-12.2/bin/nvcc` (legacy fatbin build, also the pinned
  `nvcc` used by `build.rs` for the runtime PTX ladder)
- Run commands from repository root

Toolkit choice is not interchangeable — see the warnings on each build step below.

## Manual Build

### 1) Build nextgen PoM fatbin (CUDA 12.9)

```sh
/home/slash/cuda-129/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_103,code=sm_103 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_89,code=compute_89 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_nextgen.fatbin
```

CUDA 12.8 or newer is required: earlier toolkits cannot emit `sm_100`/`sm_120` SASS, so
Blackwell cards would fall back to driver-JIT from the embedded PTX. Accepted-block throughput
is the same either way (verified on RTX 5090 by counting accepted blocks) — native SASS is
about not depending on the installed driver's JIT quality, not about speed. CUDA 13.x should
also work for nextgen but has not been tested here.

When comparing fatbin builds, measure accepted blocks over time, never the displayed MH/s:
the hashrate display currently over-counts (~2×) on the JIT-from-PTX path, which makes
identical kernels look wildly different between module paths.

The nextgen fatbin deliberately carries no SASS below `sm_89` — those cards are served by the
legacy fatbin, and duplicating the architectures only inflates the binary.

### 2) Build legacy PoM fatbin (CUDA 12.2)

```sh
/home/slash/cuda-12.2/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_89,code=compute_89 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_legacy.fatbin
```

Keep this one on CUDA 12.2. The legacy fatbin is the compatibility floor (driver r535,
glibc 2.34): a newer toolkit would raise its fallback PTX above ISA 8.2, which an r535 driver
cannot JIT, and `sm_61` is deprecated in 12.9 and gone in CUDA 13.

## Verify Outputs

`cuobjdump` from CUDA 12.2 aborts on the `sm_100`/`sm_120` entries it does not know, and the
pinned 12.9 install ships `nvcc` only. Parse the fatbin container directly instead:

```sh
python3 - cuda/pom_mine_nextgen.fatbin cuda/pom_mine_legacy.fatbin <<'PYEOF'
import struct,sys
KIND={1:'PTX',2:'SASS'}
for path in sys.argv[1:]:
    d=open(path,'rb').read()
    magic,ver,hsz,fatsz=struct.unpack_from('<IHHQ',d,0)
    assert magic==0xBA55ED50, f"{path}: not a fatbin"
    print(f"\n{path}  ({len(d)} bytes)")
    off,end=hsz,hsz+fatsz
    while off<end:
        kind,_u,ehsz,size=struct.unpack_from('<HHIQ',d,off)
        minor,major=struct.unpack_from('<HH',d,off+24)
        arch,=struct.unpack_from('<I',d,off+28)
        extra=f"ISA {major}.{minor}" if kind==1 else ""
        print(f"  {KIND.get(kind,kind):5s} sm_{arch:<4d} {extra}")
        off+=ehsz+size
PYEOF
```

Expected:

- nextgen — `PTX sm_89 ISA 8.8` plus SASS `sm_89`, `sm_90`, `sm_100`, `sm_103`, `sm_120`
- legacy — SASS `sm_61`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `PTX sm_89 ISA 8.2`, SASS `sm_89`, `sm_90`

If nextgen reports ISA 8.2 or has no `sm_120`, it was built with the wrong toolkit: redo step 1.

## Rebuild Miner

```sh
export PATH="/home/slash/cuda-12.2/bin:$PATH"
export CUDA_HOME="/home/slash/cuda-12.2"
export CUDA_ROOT="/home/slash/cuda-12.2"
export CUDA_PATH="/home/slash/cuda-12.2"

KERYX_LLAMA_ARCHS="70;75;80;86;89;90" cargo build --release
```

The 12.2 toolkit here is intentional: it drives the runtime PTX ladder and the multi-arch
`libkeryx-llama.so`, both of which sit on the same r535/glibc 2.34 floor. The nextgen fatbin is
consumed as a prebuilt artifact and is unaffected by these variables.

## Runtime Check

At startup, look for PoM logs that show per-GPU selection, for example:

- `startup loaded nextgen fatbin`
- `startup loaded legacy fatbin`

If fatbins are not available/compatible on a card, runtime falls back to the PoM PTX ladder.
