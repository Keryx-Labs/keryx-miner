#!/usr/bin/env python3
"""Cut a GGUF into layer-aligned shard files plus a sparse head file and a hash manifest.

usage: gguf_split_layers.py MODEL.gguf OUT_DIR --shard 0-5 --shard 6-11 ...

Each --shard A-B produces OUT_DIR/shard-<k>.gguf holding every tensor of layers A..B (blk.A.* to
blk.B.*), with the model's metadata copied verbatim. OUT_DIR/head.gguf is a sparse copy of the
source: embeddings, output head, the small tensors of every layer, and the tensors of the layers
no --shard claims; the large tensors of shard layers are file holes. OUT_DIR/manifest.tsv lists
name, bytes and FNV-1a 64 of every tensor above the rpc hash threshold.
"""
import argparse, os, re, sys
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
import gguf  # noqa: E402
from gguf2cache import HASH_THRESHOLD, fnv1a64_fast  # noqa: E402

BLK = re.compile(r"^blk\.(\d+)\.")

def layer_of(name):
    m = BLK.match(name)
    return int(m.group(1)) if m else None

def write_shard(reader, tensors, path):
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(path, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    w.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out_dir")
    ap.add_argument("--shard", action="append", default=[], help="layer range A-B, one per shard, in pipeline order")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ranges = []
    for s in a.shard:
        lo, hi = s.split("-"); ranges.append((int(lo), int(hi)))
    r = gguf.GGUFReader(a.model)
    owner = {}
    for k, (lo, hi) in enumerate(ranges):
        for l in range(lo, hi + 1):
            assert l not in owner, f"layer {l} in two shards"
            owner[l] = k

    # manifest over the source bytes (identical bytes in the shard files)
    with open(os.path.join(a.out_dir, "manifest.tsv"), "w") as man:
        for t in r.tensors:
            nb = int(t.n_bytes)
            if nb > HASH_THRESHOLD:
                h = fnv1a64_fast(memoryview(t.data).cast("B"))
                man.write(f"{t.name}\t{nb}\t{h:016x}\n")

    for k, (lo, hi) in enumerate(ranges):
        ts = [t for t in r.tensors if owner.get(layer_of(t.name)) == k]
        path = os.path.join(a.out_dir, f"shard-{k}.gguf")
        write_shard(r, ts, path)
        print(f"shard-{k}: layers {lo}-{hi}, {len(ts)} tensors, {sum(int(t.n_bytes) for t in ts)/2**30:.2f} GiB", file=sys.stderr)

    # sparse head: verbatim copy with holes for the shard layers' large tensors
    data_offset = int(r.data_offset); src_size = os.path.getsize(a.model)
    kept = holed = 0
    with open(a.model, "rb") as fi, open(os.path.join(a.out_dir, "head.gguf"), "wb") as fo:
        fo.write(fi.read(data_offset))
        for t in r.tensors:
            off = int(t.data_offset); nb = int(t.n_bytes)
            if nb > HASH_THRESHOLD and layer_of(t.name) in owner:
                holed += nb; continue
            fi.seek(off); fo.seek(off); fo.write(fi.read(nb)); kept += nb
        fo.truncate(src_size)
    print(f"head.gguf: {kept/2**30:.2f} GiB on disk, {holed/2**30:.2f} GiB of holes", file=sys.stderr)

if __name__ == "__main__":
    main()
