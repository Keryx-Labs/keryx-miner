#!/usr/bin/env python3
"""Pre-fill a ggml-rpc-server tensor cache from a GGUF file.

For every tensor larger than the RPC HASH_THRESHOLD (10 MiB), writes its raw bytes
to <cache_dir>/<fnv1a64 hex16>, exactly what rpc-server -c would have saved after
receiving it over the wire. A server started with -c <cache_dir> then answers
SET_TENSOR_HASH with "present" and the client never sends the bytes.

usage: gguf2cache.py MODEL.gguf CACHE_DIR [--filter blk.0.] [--manifest OUT.tsv]
"""
import argparse, os, sys, time
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
from gguf import GGUFReader  # noqa: E402

HASH_THRESHOLD = 10 * 1024 * 1024
FNV_PRIME = 0x100000001b3
FNV_OFFSET = 0xcbf29ce484222325
MASK = (1 << 64) - 1

def fnv1a64(buf: bytes) -> int:
    h = FNV_OFFSET
    for b in buf:
        h ^= b
        h = (h * FNV_PRIME) & MASK
    return h

import ctypes, subprocess
_here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_here, "libfnv.so")):
    subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", "-o", os.path.join(_here, "libfnv.so"), os.path.join(_here, "libfnv.c")])
_lib = ctypes.CDLL(os.path.join(_here, "libfnv.so"))
_lib.fnv1a64.restype = ctypes.c_uint64
_lib.fnv1a64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

def fnv1a64_fast(mv: memoryview) -> int:
    buf = (ctypes.c_uint8 * len(mv)).from_buffer(mv) if not mv.readonly else None
    if buf is None:
        raw = mv.tobytes()
        return _lib.fnv1a64(raw, len(raw))
    return _lib.fnv1a64(ctypes.addressof(buf), len(mv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("cache_dir")
    ap.add_argument("--filter", default=None, help="only tensors whose name starts with this")
    ap.add_argument("--manifest", default=None, help="write name\\tbytes\\thash per cached tensor")
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    r = GGUFReader(a.model)
    man = open(a.manifest, "w") if a.manifest else None
    n = 0; total = 0; t0 = time.time()
    for t in r.tensors:
        if a.filter and not t.name.startswith(a.filter):
            continue
        nb = int(t.n_bytes)
        if nb <= HASH_THRESHOLD:
            continue
        mv = memoryview(t.data).cast("B")
        assert len(mv) == nb, (t.name, len(mv), nb)
        h = fnv1a64_fast(mv)
        path = os.path.join(a.cache_dir, f"{h:016x}")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(mv)
        if man:
            man.write(f"{t.name}\t{nb}\t{h:016x}\n")
        n += 1; total += nb
        print(f"{t.name:40s} {nb:>12d} {h:016x}", file=sys.stderr, flush=True)
    print(f"cached {n} tensors, {total/2**20:.0f} MiB, {time.time()-t0:.0f} s", file=sys.stderr)

if __name__ == "__main__":
    main()
