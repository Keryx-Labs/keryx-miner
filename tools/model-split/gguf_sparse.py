#!/usr/bin/env python3
"""Write a sparse copy of a GGUF: same header and tensor table, but the data of the
tensors matching --hole (and larger than the RPC hash threshold) is left as file holes.
The head of a pipeline runs on this file; the holed tensors are served by hash from a
manifest, so their bytes never need to exist locally.

usage: gguf_sparse.py IN.gguf OUT.gguf --hole blk.0. --hole blk.1.
"""
import argparse, os, sys
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
from gguf import GGUFReader  # noqa: E402

HASH_THRESHOLD = 10 * 1024 * 1024

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--hole", action="append", default=[], help="tensor name prefix to leave as a hole")
    a = ap.parse_args()
    r = GGUFReader(a.src)
    data_offset = int(r.data_offset)
    src_size = os.path.getsize(a.src)
    kept = holed = 0; kept_bytes = holed_bytes = 0
    with open(a.src, "rb") as fi, open(a.dst, "wb") as fo:
        fo.write(fi.read(data_offset))          # header + kv + tensor table, verbatim
        for t in r.tensors:
            off = int(t.data_offset); nb = int(t.n_bytes)   # absolute file offset
            if nb > HASH_THRESHOLD and any(t.name.startswith(h) for h in a.hole):
                holed += 1; holed_bytes += nb
                continue
            fi.seek(off); fo.seek(off); fo.write(fi.read(nb))
            kept += 1; kept_bytes += nb
        fo.truncate(src_size)                   # same logical size as the source
    print(f"kept {kept} tensors ({kept_bytes/2**20:.0f} MiB), holed {holed} ({holed_bytes/2**20:.0f} MiB)", file=sys.stderr)

if __name__ == "__main__":
    main()
