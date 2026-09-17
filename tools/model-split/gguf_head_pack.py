#!/usr/bin/env python3
"""Pack a pipeline head into a sparse container the miner rebuilds into a sparse GGUF.

Keeps the whole GGUF header (metadata + the complete tensor table at its original offsets) and
the data of every tensor outside the layers (token_embd, output, output_norm, ...). A head
bound to resident shards never reads a layer tensor from its file, so everything else becomes a
hole when the miner unpacks it: same byte offsets as the source, a fraction of the disk.

Container "KRXSPRS1": magic(8) | total_len u64 | n u64 | n x (offset u64, len u64) | segment bytes.

usage: gguf_head_pack.py SOURCE.gguf OUT.krxh
"""
import os, struct, sys
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
from gguf import GGUFReader  # noqa: E402

MAGIC = b"KRXSPRS1"

def main():
    src, out = sys.argv[1], sys.argv[2]
    r = GGUFReader(src)
    total = os.path.getsize(src)
    segments = [(0, r.data_offset)]
    for t in r.tensors:
        if t.name.startswith("blk."):
            continue
        segments.append((int(t.data_offset), int(t.n_bytes)))
    segments.sort()
    with open(src, "rb") as f, open(out, "wb") as o:
        o.write(MAGIC)
        o.write(struct.pack("<QQ", total, len(segments)))
        for off, ln in segments:
            o.write(struct.pack("<QQ", off, ln))
        for off, ln in segments:
            f.seek(off)
            left = ln
            while left:
                chunk = f.read(min(left, 64 << 20))
                if not chunk:
                    raise SystemExit(f"short read at {off}")
                o.write(chunk)
                left -= len(chunk)
    kept = sum(ln for _, ln in segments)
    print(f"{out}: {len(segments)} segments, {kept/2**30:.3f} GiB kept of {total/2**30:.1f} GiB", file=sys.stderr)

if __name__ == "__main__":
    main()
