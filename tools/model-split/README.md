# model-split tools

Host-side tooling for serving one model across several `ggml-rpc-server` shards.

- `gguf2cache.py MODEL.gguf CACHE_DIR [--filter blk.0.] [--manifest OUT.tsv]` — pre-fills a
  shard's rpc tensor cache from its own GGUF (tensors above the rpc hash threshold, keyed by
  FNV-1a 64) and writes the manifest (`name\tbytes\thash`). Start the shard with
  `LLAMA_CACHE=<parent of CACHE_DIR named rpc> ggml-rpc-server -c`.
- `gguf_sparse.py IN.gguf OUT.gguf --hole blk.0. ...` — sparse copy for the head: same
  header and tensor table, the listed layers' large tensors left as file holes.
- The head runs with `KERYX_RPC_MANIFEST=manifest.tsv` (see
  `tools/keryx-llama/patches/0001-rpc-manifest.patch`); `KERYX_RPC_TRACE=1` logs the path
  taken per tensor.

`gguf-py` is taken from `target/llama-src-b10015` or `KERYX_GGUF_PY`. `libfnv.so` is built
on first use with gcc.
