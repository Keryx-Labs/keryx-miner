#!/usr/bin/env python3
"""Per-layer byte sizes from a GGUF header only (works on a partially downloaded file)."""
import struct, re, sys, os, collections
sys.path.insert(0, os.environ.get("KERYX_GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "llama-src-b10015", "gguf-py")))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

LARGE = 10 * 2**20  # ggml-rpc HASH_THRESHOLD

def main(path):
    f = open(path, 'rb')
    def rd(fmt):
        fmt = '<' + fmt
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    def rstr():
        (n,) = rd('Q'); return f.read(n).decode('utf-8', 'replace')
    T = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            (et,) = rd('I'); (n,) = rd('Q'); return [rval(et) for _ in range(n)]
        return rd(T[t])[0]
    magic = f.read(4); (ver, nt, nkv) = rd('IQQ')
    print(magic, 'v', ver, 'tensors', nt, 'kv', nkv)
    kvs = {}
    for _ in range(nkv):
        k = rstr(); (t,) = rd('I'); kvs[k] = rval(t)
    arch = kvs.get('general.architecture')
    for k, v in kvs.items():
        if (k.startswith('general.') and k not in ('general.license',)) or (k.startswith(arch + '.') and 'token' not in k):
            s = str(v); print(' ', k, s if len(s) < 100 else s[:100] + '…')
    per = collections.defaultdict(int); big = collections.defaultdict(int); other = {}
    types = collections.Counter(); per_kind = collections.defaultdict(int)
    for _ in range(nt):
        name = rstr(); (nd,) = rd('I'); dims = rd('Q' * nd); (typ,) = rd('I'); (off,) = rd('Q')
        bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(typ)]
        ne = 1
        for d in dims: ne *= d
        nb = ne * ts // bs
        types[GGMLQuantizationType(typ).name] += nb
        m = re.match(r'blk\.(\d+)\.(.*)', name)
        if m:
            i = int(m.group(1)); per[i] += nb
            if nb > LARGE: big[i] += nb
            per_kind[m.group(2)] += nb
        else:
            other[name] = nb
    tot = sum(per.values()) + sum(other.values())
    print(f'total {tot} bytes = {tot/2**30:.2f} GiB ; header ends at {f.tell()}')
    print('by type GiB:', {k: round(v/2**30, 2) for k, v in types.items()})
    print('non-layer:', {k: f'{v/2**20:.0f} MiB' for k, v in other.items()})
    for i in sorted(per):
        print(f'layer {i:2d}: {per[i]/2**30:6.2f} GiB  large {big[i]/2**30:6.2f}  small {(per[i]-big[i])/2**20:7.1f} MiB')
    print('per tensor kind (summed over layers), GiB:')
    for k, v in sorted(per_kind.items(), key=lambda x: -x[1])[:16]:
        print(f'  {k:40s} {v/2**30:8.2f}')

if __name__ == '__main__':
    main(sys.argv[1])
