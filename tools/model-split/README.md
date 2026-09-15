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
- `patches/0002-rpc-memset-tensor.patch`: `memset_tensor` on rpc buffers (needed by the
  DeepSeek-V4 KV cache).
- `patches/0003-sched-coalesce-view-inputs.patch`: split inputs that are views of one graph tensor
  are transferred once, as their source, and re-viewed on the split backend (one round trip per
  shard boundary instead of one per view).
- `patches/0004-rpc-cache-alloc-size.patch`: `GET_ALLOC_SIZE` answers cached per endpoint and
  tensor shape (one round trip per distinct shape instead of one per graph node).
- `patches/0005-dsv4-pin-hc-norm-to-layer.patch`: the DeepSeek-V4 hyper-connection input norm is
  named `norm` so llama.cpp pins it to its layer's device (otherwise it runs on the previous
  shard and crosses the link as a second tensor per token).

`gguf-py` is taken from `target/llama-src-b10015` or `KERYX_GGUF_PY`. `libfnv.so` is built
on first use with gcc.
- `gguf_split_layers.py MODEL.gguf OUT_DIR --shard 0-5 --shard 6-11 ...` — one valid GGUF per
  shard (its layers' tensors, metadata copied), `head.gguf` (sparse: everything but the shard
  layers' large tensors) and `manifest.tsv`. A shard file is what `pom-rt-builder` roots and what
  `gguf2cache.py` pre-fills the shard's cache from.
- `gguf_layer_sizes.py MODEL.gguf` — per-layer byte sizes from the header alone (works on a partial
  download); the input of the per-VRAM-class partition.
- `gguf_truncate_layers.py MODEL.gguf OUT.gguf N` — the first N layers as a loadable model
  (per-layer metadata arrays cut to N). Output is meaningless; it is for measuring the per-token
  rpc protocol of a model too large for one machine.
