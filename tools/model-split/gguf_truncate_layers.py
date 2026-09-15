#!/usr/bin/env python3
"""Write a GGUF keeping only the first N layers (block_count = N, per-layer arrays cut to N).

usage: gguf_truncate_layers.py IN.gguf OUT.gguf N
Header-only parse of the source (no GGUFReader), tensor bytes copied by offset. The result is a
loadable model whose output is meaningless; it exists to measure the per-token RPC protocol of an
architecture too large for one machine.
"""
import os, re, struct, sys
import numpy as np
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType

T = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
NUMPY_DTYPES = {GGMLQuantizationType.F32: np.float32, GGMLQuantizationType.F16: np.float16, GGMLQuantizationType.I8: np.int8, GGMLQuantizationType.I16: np.int16, GGMLQuantizationType.I32: np.int32, GGMLQuantizationType.I64: np.int64, GGMLQuantizationType.F64: np.float64}
PER_LAYER_ARRAYS = ("swiglu_clamp_exp", "swiglu_clamp_shexp", "attention.compress_ratios")

def parse(path):
    f = open(path, 'rb')
    def rd(fmt):
        fmt = '<' + fmt
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    def rstr():
        (n,) = rd('Q'); return f.read(n).decode('utf-8')
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            (et,) = rd('I'); (n,) = rd('Q'); return (et, [rval(et) for _ in range(n)])
        return rd(T[t])[0]
    assert f.read(4) == b'GGUF'
    (ver, nt, nkv) = rd('IQQ')
    kvs = []
    for _ in range(nkv):
        k = rstr(); (t,) = rd('I'); kvs.append((k, t, rval(t)))
    tensors = []
    for _ in range(nt):
        name = rstr(); (nd,) = rd('I'); dims = rd('Q' * nd); (typ,) = rd('I'); (off,) = rd('Q')
        bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(typ)]
        ne = 1
        for d in dims: ne *= d
        tensors.append((name, dims, typ, off, ne * ts // bs))
    alignment = next((v for k, t, v in kvs if k == 'general.alignment'), 32)
    end = f.tell(); data_offset = (end + alignment - 1) // alignment * alignment
    return kvs, tensors, data_offset

def main():
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kvs, tensors, data_offset = parse(src)
    arch = next(v for k, t, v in kvs if k == 'general.architecture')
    w = gguf.GGUFWriter(dst, arch=arch)
    for k, t, v in kvs:
        if k == 'general.architecture' or k.startswith('GGUF.'):
            continue
        if k == f'{arch}.block_count':
            w.add_key_value(k, n, GGUFValueType(t)); continue
        if t == 9:
            et, items = v
            if k in (f'{arch}.{s}' for s in PER_LAYER_ARRAYS):
                assert len(items) >= n, k
                items = items[:n]
            w.add_key_value(k, items, GGUFValueType.ARRAY, sub_type=GGUFValueType(et)); continue
        w.add_key_value(k, v, GGUFValueType(t))
    keep = []
    for name, dims, typ, off, nb in tensors:
        m = re.match(r'blk\.(\d+)\.', name)
        if m and int(m.group(1)) >= n:
            continue
        keep.append((name, dims, typ, off, nb))
        qt = GGMLQuantizationType(typ); bs, ts = GGML_QUANT_SIZES[qt]
        shape = list(reversed(dims))
        if qt in NUMPY_DTYPES:
            w.add_tensor_info(name, shape, NUMPY_DTYPES[qt], nb, raw_dtype=qt)
        else:
            shape[-1] = shape[-1] * ts // bs
            w.add_tensor_info(name, shape, np.uint8, nb, raw_dtype=qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    mm = np.memmap(src, dtype=np.uint8, mode='r')
    total = 0
    for name, dims, typ, off, nb in keep:
        w.write_tensor_data(mm[data_offset + off: data_offset + off + nb]); total += nb
    w.close()
    print(f'{dst}: {len(keep)} tensors, {total/2**30:.2f} GiB, block_count={n}', file=sys.stderr)

if __name__ == '__main__':
    main()
