// libkeryx-llama.{so,dylib} — the miner's in-process llama.cpp engine (Phase 2
// on CUDA, Phase 3b on Apple Silicon Metal).
//
// One llama.cpp instance per loaded model: it OWNS the resident GGUF copy on the inference GPU
// and exposes (a) per-tensor device pointers so the PoM walk gathers straight over the SAME VRAM
// (zero-dup — proven byte-identical to the on-disk GGUF by tools/llama_zerodup_spike on CUDA), and
// (b) text generation for OPoI. On Apple Silicon (Metal) the walk uses its own packed buffer
// (`pom_gpu_metal` Phase 3a) so the tensor-pointer contract there only feeds the future zero-dup
// Metal walk; today it just satisfies the loader-side count/name enumeration.
//
// The miner dlopens this next to its own binary; absent = inference is unavailable.
// Built by hiveos/build-keryx-llama.sh (CUDA) or hiveos/build-keryx-llama-macos.sh (Metal).
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#ifdef __APPLE__
// Metal: llama.cpp's ggml-metal backend stores quantized tensors in unified-memory MTLBuffers.
// `t->data` is a CPU-readable pointer into that unified memory (also GPU-visible on Apple Silicon
// via the shared address space), so we don't need cudaPointerGetAttributes — `is_device` is
// always 1 for tensors llama.cpp reports.
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

// Windows DLLs export nothing by default — mark the ABI surface explicitly so the miner's
// GetProcAddress finds it. No-op on ELF/Mach-O (default visibility already exports).
#if defined(_WIN32)
#define KERYX_EXPORT __declspec(dllexport)
#else
#define KERYX_EXPORT
#endif

// Load failures are reported by a null return, which tells the miner nothing. Keep the reason
// (and the VRAM figures behind an OOM) for `keryx_llama_last_error`.
static thread_local std::string keryx_last_error;

static std::string keryx_cuda_diagnostics() {
#ifdef __APPLE__
    return std::string();
#else
    std::string out;
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess) {
        out += std::string(" [cuda: ") + cudaGetErrorString(pending) + "]";
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        out += " [vram: " + std::to_string(free_bytes / (1024 * 1024)) + " MiB free / "
             + std::to_string(total_bytes / (1024 * 1024)) + " MiB total]";
    }
    return out;
#endif
}

static void keryx_set_error(const char* stage, const std::string& detail) {
    keryx_last_error = std::string(stage) + ": " + detail + keryx_cuda_diagnostics();
}

// Opt-in debug trace (KERYX_TRACE_DIR).
struct KeryxTrace {
    std::vector<std::string> match;
    std::vector<std::string> nodes;
    FILE* f = nullptr;
    uint64_t max_bytes = 0, written = 0, seq = 0, skipped = 0, decode = 0;
    int64_t pos0 = 0;
    std::vector<uint8_t> buf;
    const ggml_tensor* q8_dst = nullptr;
    int q8_layout = -1;
    std::vector<uint8_t> q8;
    std::vector<uint8_t> q8_ids;
};

static void keryx_trace_put(KeryxTrace* tr, const void* p, size_t n) {
    fwrite(p, 1, n, tr->f);
    tr->written += n;
}

static std::vector<std::string> keryx_split(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s; *p; ++p) {
        if (*p == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += *p;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static bool keryx_trace_wants(const KeryxTrace* tr, const ggml_tensor* t) {
    if ((t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32
        || t->src[1]->type != GGML_TYPE_F32) return false;
    for (const auto& m : tr->match) if (std::strstr(t->src[0]->name, m.c_str())) return true;
    return false;
}

static bool keryx_trace_wants_node(const KeryxTrace* tr, const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32 || t->op == GGML_OP_MUL_MAT) return false;
    for (const auto& n : tr->nodes) if (n == t->name) return true;
    return false;
}

static void keryx_trace_node(KeryxTrace* tr, const ggml_tensor* t) {
    const size_t nb = ggml_nbytes(t);
    const size_t name_len = std::strlen(t->name);
    const char* op = ggml_op_name(t->op);
    const size_t op_len = std::strlen(op);
    if (!ggml_is_contiguous(t) || tr->written + nb + 96 + name_len + op_len > tr->max_bytes) {
        tr->skipped++;
        return;
    }
    const uint8_t kind = 7;
    const uint32_t nl = (uint32_t)name_len;
    const uint32_t ol = (uint32_t)op_len;
    keryx_trace_put(tr, &kind, 1);
    keryx_trace_put(tr, &tr->decode, 8);
    keryx_trace_put(tr, &tr->pos0, 8);
    keryx_trace_put(tr, &nl, 4);
    keryx_trace_put(tr, t->name, name_len);
    keryx_trace_put(tr, &ol, 4);
    keryx_trace_put(tr, op, op_len);
    keryx_trace_put(tr, t->ne, 4 * sizeof(int64_t));
    tr->buf.resize(nb);
    ggml_backend_tensor_get(t, tr->buf.data(), 0, nb);
    keryx_trace_put(tr, tr->buf.data(), nb);
}

static void keryx_trace_q8_flush(KeryxTrace* tr, const ggml_tensor* t) {
    if (tr->q8_dst == t && !tr->q8.empty()) {
        const uint8_t k3 = 3;
        const int32_t layout = tr->q8_layout;
        const uint64_t nb = tr->q8.size();
        keryx_trace_put(tr, &k3, 1);
        keryx_trace_put(tr, &tr->seq, 8);
        keryx_trace_put(tr, &layout, 4);
        keryx_trace_put(tr, &nb, 8);
        keryx_trace_put(tr, tr->q8.data(), tr->q8.size());
        if (!tr->q8_ids.empty()) {
            const uint8_t k5 = 5;
            const uint64_t ni = tr->q8_ids.size();
            keryx_trace_put(tr, &k5, 1);
            keryx_trace_put(tr, &tr->seq, 8);
            keryx_trace_put(tr, &ni, 8);
            keryx_trace_put(tr, tr->q8_ids.data(), tr->q8_ids.size());
        }
    }
    tr->q8_dst = nullptr;
    tr->q8.clear();
    tr->q8_ids.clear();
}

static bool keryx_trace_mmid(KeryxTrace* tr, const ggml_tensor* t) {
    const ggml_tensor* w = t->src[0];
    const ggml_tensor* x = t->src[1];
    const ggml_tensor* ids = t->src[2];
    const int64_t K = w->ne[0], M = w->ne[1], n_exp = w->ne[2];
    const int64_t ne11 = x->ne[1], ne12 = x->ne[2], n_used = ids ? ids->ne[0] : 0;
    const size_t xb = ggml_nbytes(x), yb = ggml_nbytes(t);
    const size_t name_len = std::strlen(w->name);
    const size_t ib = sizeof(int32_t) * (size_t)(n_used * ne12);
    const size_t need = 1 + 8 * 3 + 4 + name_len + 4 + 8 * 6 + xb + ib + yb;
    if (!ids || ids->type != GGML_TYPE_I32 || x->ne[0] != K || x->ne[3] != 1 || t->ne[0] != M
        || t->ne[1] != n_used || t->ne[2] != ne12 || !ggml_is_contiguous(x) || !ggml_is_contiguous(t)
        || tr->written + need > tr->max_bytes) {
        tr->skipped++;
        tr->q8_dst = nullptr;
        tr->q8.clear();
        tr->q8_ids.clear();
        return true;
    }
    const uint8_t kind = 6;
    const uint32_t nl = (uint32_t)name_len;
    const int32_t wtype = (int32_t)w->type;
    keryx_trace_put(tr, &kind, 1);
    keryx_trace_put(tr, &tr->seq, 8);
    keryx_trace_put(tr, &tr->decode, 8);
    keryx_trace_put(tr, &tr->pos0, 8);
    keryx_trace_put(tr, &nl, 4);
    keryx_trace_put(tr, w->name, name_len);
    keryx_trace_put(tr, &wtype, 4);
    const int64_t dims[6] = { K, M, n_exp, ne11, ne12, n_used };
    keryx_trace_put(tr, dims, sizeof dims);
    tr->buf.resize(std::max(std::max(xb, yb), ggml_nbytes(ids)));
    ggml_backend_tensor_get(x, tr->buf.data(), 0, xb);
    keryx_trace_put(tr, tr->buf.data(), xb);
    ggml_backend_tensor_get(ids, tr->buf.data(), 0, ggml_nbytes(ids));
    std::vector<int32_t> flat((size_t)(n_used * ne12));
    for (int64_t i1 = 0; i1 < ne12; ++i1) {
        for (int64_t i0 = 0; i0 < n_used; ++i0) {
            std::memcpy(&flat[(size_t)(i1 * n_used + i0)], tr->buf.data() + i1 * ids->nb[1] + i0 * ids->nb[0], sizeof(int32_t));
        }
    }
    keryx_trace_put(tr, flat.data(), ib);
    ggml_backend_tensor_get(t, tr->buf.data(), 0, yb);
    keryx_trace_put(tr, tr->buf.data(), yb);
    keryx_trace_q8_flush(tr, t);
    tr->seq++;
    return true;
}

static bool keryx_trace_cb(ggml_tensor* t, bool ask, void* ud) {
    auto* tr = static_cast<KeryxTrace*>(ud);
    if (ask) return keryx_trace_wants(tr, t) || keryx_trace_wants_node(tr, t);
    if (keryx_trace_wants_node(tr, t)) {
        keryx_trace_node(tr, t);
        return true;
    }
    if (!keryx_trace_wants(tr, t)) return true;
    if (t->op == GGML_OP_MUL_MAT_ID) return keryx_trace_mmid(tr, t);
    const ggml_tensor* w = t->src[0];
    const ggml_tensor* x = t->src[1];
    const int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1];
    const size_t xb = ggml_nbytes(x), yb = ggml_nbytes(t);
    const size_t name_len = std::strlen(w->name);
    const size_t need = 1 + 8 * 3 + 4 + name_len + 4 + 8 * 3 + xb + yb;
    if (x->ne[0] != K || x->ne[2] != 1 || x->ne[3] != 1 || t->ne[0] != M || t->ne[1] != N
        || !ggml_is_contiguous(x) || !ggml_is_contiguous(t) || tr->written + need > tr->max_bytes) {
        tr->skipped++;
        return true;
    }
    const uint8_t kind = 1;
    const uint32_t nl = (uint32_t)name_len;
    const int32_t wtype = (int32_t)w->type;
    keryx_trace_put(tr, &kind, 1);
    keryx_trace_put(tr, &tr->seq, 8);
    keryx_trace_put(tr, &tr->decode, 8);
    keryx_trace_put(tr, &tr->pos0, 8);
    keryx_trace_put(tr, &nl, 4);
    keryx_trace_put(tr, w->name, name_len);
    keryx_trace_put(tr, &wtype, 4);
    keryx_trace_put(tr, &K, 8);
    keryx_trace_put(tr, &M, 8);
    keryx_trace_put(tr, &N, 8);
    tr->buf.resize(std::max(xb, yb));
    ggml_backend_tensor_get(x, tr->buf.data(), 0, xb);
    keryx_trace_put(tr, tr->buf.data(), xb);
    ggml_backend_tensor_get(t, tr->buf.data(), 0, yb);
    keryx_trace_put(tr, tr->buf.data(), yb);
    keryx_trace_q8_flush(tr, t);
    tr->seq++;
    return true;
}

#ifndef __APPLE__
// Canonical trace leaves (ggml-cuda/q8canon.cu, applied by tools/keryx-llama/patches).
extern "C" {
int  ggml_cuda_q8c_configure(int device, const void* manifest, size_t manifest_len, uint32_t max_pos);
void ggml_cuda_q8c_release(int device);
int  ggml_cuda_q8c_leaf_count(int device);
int  ggml_cuda_q8c_begin_response(int device);
int  ggml_cuda_q8c_arm_capture(int device, uint32_t pos, uint32_t leaf_x, uint32_t leaf_y);
int  ggml_cuda_q8c_read(int device, uint8_t* leaves, size_t leaves_cap, uint32_t* ubatch, size_t ubatch_cap, uint32_t* n_pos);
int  ggml_cuda_q8c_read_capture(int device, uint8_t* x, size_t x_cap, uint8_t* y, size_t y_cap, uint32_t* x_len, uint32_t* y_len);
int  ggml_cuda_q8c_arm_captures(int device, const uint32_t* pos, const uint32_t* leaf, uint32_t n, uint32_t slot_bytes);
int  ggml_cuda_q8c_read_captures(int device, uint8_t* out, size_t out_cap, uint32_t* lens, uint32_t n);
// Sequence leaves (ggml-cuda/q8seq.cu, applied by tools/keryx-llama/patches).
int  ggml_cuda_q8s_configure(int device, const uint32_t* layers, uint32_t n, uint32_t n_attn, uint32_t n_recur,
                             uint32_t seq_heads, uint32_t interval, uint32_t max_tokens);
void ggml_cuda_q8s_release(int device);
int  ggml_cuda_q8s_begin_response(int device);
int  ggml_cuda_q8s_roots(int device, uint8_t* out, size_t out_cap, uint32_t* n_leaves);
int  ggml_cuda_q8s_arm(int device, const uint32_t* stream, const uint64_t* lo, const uint64_t* hi, uint32_t n,
                       uint32_t max_chunks);
int  ggml_cuda_q8s_read_captures(int device, uint8_t* bytes, size_t bytes_cap, uint64_t* lens, uint8_t* cvs,
                                 size_t cvs_cap, uint32_t* n_cvs, uint32_t n);
}
#endif

static void keryx_trace_batch(KeryxTrace* tr, const llama_token* toks, int n, int64_t pos0) {
    if (!tr) return;
    tr->decode++;
    tr->pos0 = pos0;
    const uint8_t kind = 2;
    const int64_t nn = n;
    keryx_trace_put(tr, &kind, 1);
    keryx_trace_put(tr, &tr->decode, 8);
    keryx_trace_put(tr, &pos0, 8);
    keryx_trace_put(tr, &nn, 8);
    keryx_trace_put(tr, toks, sizeof(llama_token) * (size_t)n);
    fflush(tr->f);
}

static KeryxTrace* keryx_trace_open(int gpu) {
    const char* dir = getenv("KERYX_TRACE_DIR");
    if (!dir || !*dir) return nullptr;
    const std::string base = std::string(dir) + "/trace-" + std::to_string((long)getpid()) + "-gpu" + std::to_string(gpu);
    FILE* f = fopen((base + ".bin").c_str(), "wb");
    if (!f) return nullptr;
    auto* tr = new KeryxTrace();
    tr->f = f;
    const char* m = getenv("KERYX_TRACE_MATCH");
    tr->match = keryx_split((m && *m) ? m : ".ffn_down.");
    const char* nd = getenv("KERYX_TRACE_NODES");
    if (nd && *nd) tr->nodes = keryx_split(nd);
    const char* mb = getenv("KERYX_TRACE_MAX_MB");
    tr->max_bytes = (uint64_t)((mb && *mb) ? std::strtoull(mb, nullptr, 10) : 1024) << 20;
    keryx_trace_put(tr, "KXTRACE1", 8);
    return tr;
}

static void keryx_trace_meta(int gpu, const char* gguf_path, const std::string& ctx_info, const KeryxTrace* tr) {
    const std::string base = std::string(getenv("KERYX_TRACE_DIR")) + "/trace-" + std::to_string((long)getpid()) + "-gpu" + std::to_string(gpu);
    FILE* f = fopen((base + ".meta").c_str(), "w");
    if (!f) return;
    std::string match, nodes;
    for (const auto& s : tr->match) match += (match.empty() ? "" : ",") + s;
    for (const auto& s : tr->nodes) nodes += (nodes.empty() ? "" : ",") + s;
    fprintf(f, "gguf=%s\nmatch=%s\nnodes=%s\n%s\n", gguf_path, match.c_str(), nodes.c_str(), ctx_info.c_str());
#ifndef __APPLE__
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) == cudaSuccess) {
        fprintf(f, "device=%s\nsm=%d%d\n", prop.name, prop.major, prop.minor);
    }
#endif
    fclose(f);
}

static void keryx_trace_close(KeryxTrace* tr) {
    if (!tr) return;
#ifdef KERYX_BENCH_Q8_TAP
    if (keryx_q8_trace == tr) {
        ggml_cuda_q8_tap = nullptr;
        keryx_q8_trace = nullptr;
    }
#endif
    if (tr->f) fclose(tr->f);
    if (tr->skipped) fprintf(stderr, "keryx trace: %llu matmul(s) skipped\n", (unsigned long long)tr->skipped);
    delete tr;
}

// Patch 0007: canonical weight layout switch and the canonical buffer the last load produced.
#ifdef __APPLE__
static thread_local bool keryx_llama_canonical_weights = false;
static thread_local void * keryx_llama_canonical_base = nullptr;
static thread_local size_t keryx_llama_canonical_bytes = 0;
#else
extern "C" { extern thread_local bool keryx_llama_canonical_weights; extern thread_local void * keryx_llama_canonical_base; extern thread_local size_t keryx_llama_canonical_bytes; }
#endif

static thread_local bool keryx_want_canonical = true;

// Whether the next load on this thread may use the canonical layout (on by default).
extern "C" KERYX_EXPORT void keryx_llama_want_canonical(int on) {
    keryx_want_canonical = on != 0;
}

struct KeryxLlama {
    void*          canon_base  = nullptr;
    size_t         canon_bytes = 0;
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    KeryxTrace*    trace = nullptr;
    int            gpu   = 0;
    bool           traced = false;
    bool           sequenced = false;
    std::vector<std::string> names; // canonical (byte-lexicographic) order — matches pom.rs
    std::mutex gen_lock;
    std::string ctx_info;           // "n_ctx=… kv=… flash_attn=…" for the miner log
    llama_context_params cparams{}; // parameters the context was created with at load
    std::string kv;
};

static std::string keryx_ctx_info(KeryxLlama* h) {
    if (!h->ctx) return "no context";
    return "n_ctx=" + std::to_string(llama_n_ctx(h->ctx)) + " kv=" + h->kv +
           " flash_attn=" + llama_flash_attn_type_name(h->cparams.flash_attn_type) +
           " n_batch=" + std::to_string(llama_n_batch(h->ctx)) + " n_ubatch=" + std::to_string(llama_n_ubatch(h->ctx));
}

static bool keryx_has_ctx(KeryxLlama* h) {
    if (h->ctx) return true;
    keryx_set_error("context", "the engine has no context");
    return false;
}

// llama.cpp/ggml emit a large INFO-level dump on every model load (full tensor list, per-layer
// device assignment, kv-cache map, sched-reserve…). The miner's own Rust logging already covers
// what matters, so by default we install a log callback that forwards only WARN/ERROR. Set
// KERYX_LLAMA_VERBOSE=1 to restore llama.cpp's full default stderr logging (debugging).
static void keryx_llama_log_cb(enum ggml_log_level level, const char* text, void* /*ud*/) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}
static void keryx_install_log_filter() {
    static bool done = false;
    if (done) return;
    done = true;
    if (getenv("KERYX_LLAMA_VERBOSE")) return; // leave llama's default stderr logging in place
    llama_log_set(keryx_llama_log_cb, nullptr);
    ggml_log_set(keryx_llama_log_cb, nullptr);
}

// GLM-4-0414 rotates 64 of its 128 head dimensions; a GGUF missing the key rotates all of them.
static bool keryx_needs_glm4_rope_fix(const char* gguf_path) {
    gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(gguf_path, params);
    if (!g) return false;
    bool fix = false;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING
        && std::strcmp(gguf_get_val_str(g, arch), "glm4") == 0) {
        fix = gguf_find_key(g, "glm4.rope.dimension_count") < 0;
    }
    gguf_free(g);
    return fix;
}

// Feeds the prompt in n_batch chunks: the replay must split it the same way.
static int keryx_feed_prompt(KeryxLlama* h, const llama_token* toks, int n) {
    const int n_batch = std::max(1, (int)llama_n_batch(h->ctx));
    for (int i = 0; i < n; i += n_batch) {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(toks) + i, std::min(n_batch, n - i));
        keryx_trace_batch(h->trace, toks + i, std::min(n_batch, n - i), i);
        if (llama_decode(h->ctx, batch) != 0) {
            keryx_set_error("decode", "llama_decode failed while reading the prompt");
            return -3;
        }
    }
    return 0;
}

// Token fed as padding: a newline, else the end-of-sequence token.
static llama_token keryx_pad_token(const llama_vocab* vocab) {
    llama_token t[4];
    const int k = llama_tokenize(vocab, "\n", 1, t, 4, false, false);
    return k >= 1 ? t[0] : llama_vocab_eos(vocab);
}

// Starts a fresh trace for the next response when tracing is configured. Returns 0 on success.
static int keryx_trace_reset(KeryxLlama* h) {
#ifdef __APPLE__
    (void)h;
    return 0;
#else
    if (h->sequenced && ggml_cuda_q8s_begin_response(h->gpu) != 0) return -1;
    return h->traced ? ggml_cuda_q8c_begin_response(h->gpu) : 0;
#endif
}

extern "C" {

// ABI version — the miner refuses to use a mismatched .so.
KERYX_EXPORT int keryx_llama_abi() { return 9; }

// Reason for the last failed call on this thread; empty when none. Valid until the next call.
KERYX_EXPORT const char* keryx_llama_last_error() { return keryx_last_error.c_str(); }

KERYX_EXPORT KeryxLlama* keryx_llama_load(const char* gguf_path, int gpu, int n_ctx) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE; // ONE GPU — never layer-split across mining cards
    mp.main_gpu     = gpu;
    mp.use_mmap     = true;
    llama_model_kv_override overrides[2] = {};
    if (keryx_needs_glm4_rope_fix(gguf_path)) {
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strncpy(overrides[0].key, "glm4.rope.dimension_count", sizeof(overrides[0].key) - 1);
        overrides[0].val_i64 = 64;
        mp.kv_overrides = overrides;
    }
    {
        // On by default; KERYX_CANONICAL_WEIGHTS=0 keeps llama's own layout.
        const char* cw = std::getenv("KERYX_CANONICAL_WEIGHTS");
        keryx_llama_canonical_weights = keryx_want_canonical && !(cw && cw[0] == '0');
        keryx_llama_canonical_base = nullptr;
        keryx_llama_canonical_bytes = 0;
    }
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    void* const canon_base = keryx_llama_canonical_base;
    const size_t canon_bytes = keryx_llama_canonical_bytes;
    keryx_llama_canonical_weights = false;
    if (!model) {
        keryx_set_error("model", std::string("llama_model_load_from_file failed for ") + gguf_path);
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    // The logical batch bounds one decode call; the physical one bounds the compute buffers,
    // which grow with it and not with the context.
    cp.n_batch = std::min(2048u, cp.n_ctx);
    cp.n_ubatch = std::min(512u, cp.n_batch);
    // The cache is cleared before every request, so sliding-window layers only need their window.
    cp.swa_full = false;
    // 8-bit KV cache with flash attention: half the per-token VRAM of f16. Falls back to the
    // default cache for an architecture the fast path cannot serve.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    const char* kv = "q8_0";
    KeryxTrace* trace = keryx_trace_open(gpu);
    if (trace) {
        cp.cb_eval = keryx_trace_cb;
        cp.cb_eval_user_data = trace;
#ifdef KERYX_BENCH_Q8_TAP
        keryx_q8_trace = trace;
        ggml_cuda_q8_tap = keryx_q8_tap;
#endif
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        // An MLA cache holds no V rows, but its value length still has to divide a quantized V type.
        cp.type_v = GGML_TYPE_F16;
        kv = "q8_0/f16";
        ctx = llama_init_from_model(model, cp);
    }
    if (!ctx) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        kv = "f16";
        ctx = llama_init_from_model(model, cp);
    }
    if (!ctx) {
        keryx_set_error("context", "llama_init_from_model failed");
        keryx_trace_close(trace);
        llama_model_free(model);
        return nullptr;
    }

    // User-facing sampling (DRY -> repeat penalty -> temperature 0.7 / top_p 0.9) — the OPoI
    // text is not consensus-relevant, but keep the flavor consistent.
    //
    // The flat repeat penalty alone does not stop these models: it charges a token once for
    // having appeared, no matter how many times, so a repeated *sentence* pays almost nothing
    // per token and the loop survives (observed on-chain with GLM-4-9B at 1.10 / 256). DRY
    // penalises the continuation of an already-seen sequence, growing with its length, which is
    // the failure mode we actually have. Standard settings: multiplier 0.8, base 1.75, loops
    // allowed up to 2 tokens, scanning the whole context (-1). The break tokens keep normal
    // structure (newlines, list markers, quotes) from counting as repetition.
    static const char* dry_breakers[] = { "\n", ":", "\"", "*" };
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_dry(
        llama_model_get_vocab(model), llama_model_n_ctx_train(model),
        0.8f, 1.75f, 2, -1, dry_breakers, sizeof(dry_breakers) / sizeof(dry_breakers[0])));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto* h = new KeryxLlama();
    h->model = model; h->ctx = ctx; h->smpl = smpl; h->trace = trace; h->gpu = gpu;
    h->canon_base = canon_base; h->canon_bytes = canon_bytes;
    h->cparams = cp; h->kv = kv;
    h->ctx_info = keryx_ctx_info(h);
    if (trace) keryx_trace_meta(gpu, gguf_path, h->ctx_info, trace);
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

KERYX_EXPORT size_t keryx_llama_tensor_count(KeryxLlama* h) { return h ? h->names.size() : 0; }
// Device address and byte length of the model's canonical weight stream, 0 when the weights were
// allocated the ordinary way. Holes (tensors kept elsewhere) are zero until filled.
KERYX_EXPORT size_t keryx_llama_canonical_blob(KeryxLlama* h, void** base) {
    if (!h || !base) return 0;
    *base = h->canon_base;
    return h->canon_base ? h->canon_bytes : 0;
}

// Replaces the context with one of `n_ctx` tokens (the load size when <= 0); the model and its
// weights stay. Returns the context size, or -1 with no context left.
KERYX_EXPORT int keryx_llama_context_acquire(KeryxLlama* h, int n_ctx) {
    if (!h) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    llama_context_params cp = h->cparams;
    if (n_ctx > 0) cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = std::min(2048u, cp.n_ctx);
    cp.n_ubatch = std::min(512u, cp.n_batch);
    if (h->ctx && llama_n_ctx(h->ctx) == cp.n_ctx) return (int)cp.n_ctx;
    if (h->ctx) {
        llama_free(h->ctx);
        h->ctx = nullptr;
    }
    h->ctx = llama_init_from_model(h->model, cp);
    h->ctx_info = keryx_ctx_info(h);
    if (!h->ctx) {
        keryx_set_error("context", "llama_init_from_model failed for a " + std::to_string(cp.n_ctx) + "-token context");
        return -1;
    }
    return (int)llama_n_ctx(h->ctx);
}

// Frees the context (KV cache and compute buffers); the model and its weights stay.
KERYX_EXPORT void keryx_llama_context_release(KeryxLlama* h) {
    if (!h) return;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (h->ctx) llama_free(h->ctx);
    h->ctx = nullptr;
    h->ctx_info = keryx_ctx_info(h);
}

// Context parameters the engine ended up with, for the miner log.
KERYX_EXPORT const char* keryx_llama_context_info(KeryxLlama* h) { return h ? h->ctx_info.c_str() : ""; }

// Tensor i in CANONICAL order. *is_device = the data pointer is CUDA device memory (walkable
// in-place); 0 = host memory (the caller uploads its own device copy for the walk).
KERYX_EXPORT bool keryx_llama_tensor_info(KeryxLlama* h, size_t i, const char** name, void** data,
                                          size_t* nbytes, int* is_device) {
    if (!h || i >= h->names.size()) return false;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return false;
    *name = h->names[i].c_str();
    *data = t->data;
    *nbytes = ggml_nbytes(t);
#ifdef __APPLE__
    // Metal / Apple Silicon unified memory: tensor bytes are in an MTLBuffer that's both CPU- and
    // GPU-visible via the same address. The Metal PoM walk (Phase 3a) doesn't consume `data` for
    // its own gather (it pre-packs from GGUF), so the semantic here is "there's a live pointer
    // to the tensor bytes for anyone who wants to check byte-exactness against GGUF".
    *is_device = 1;
#else
    cudaPointerAttributes attr{};
    cudaPointerGetAttributes(&attr, t->data);
    *is_device = attr.type == cudaMemoryTypeDevice ? 1 : 0;
#endif
    return true;
}

// Generate up to max_tokens; writes UTF-8 into out (cap bytes, NUL-terminated). Returns written
// length, or -1 on error. Serialized — one generation at a time (OPoI challenges are rare).
// CUDA ordinal owning tensor i's bytes, or -1 (host memory, unified memory, unknown, or a
// context in error). The walk gathers over these pointers, so it must launch on this device.
KERYX_EXPORT int keryx_llama_tensor_device(KeryxLlama* h, size_t i) {
#ifdef __APPLE__
    (void)h; (void)i;
    return -1;
#else
    if (!h || i >= h->names.size()) return -1;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return -1;
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, t->data) != cudaSuccess) return -1;
    return attr.type == cudaMemoryTypeDevice ? attr.device : -1;
#endif
}

// Returns the bytes written, or -1 (arguments / tokenizer), -2 (prompt longer than the context,
// detail in keryx_llama_last_error), -3 (prompt decode failed). The prompt is fed in batches of
// at most n_batch tokens; generation stops when the context is full.
KERYX_EXPORT int keryx_llama_generate(KeryxLlama* h, const char* prompt, int max_tokens, char* out, int cap) {
    if (!h || !prompt || !out || cap < 2) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (!keryx_has_ctx(h)) return -2;
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    std::vector<llama_token> toks(strlen(prompt) + 16);
    int n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks.data(), (int32_t)toks.size(), true, true);
    if (n <= 0) return -1;
    toks.resize(n);

    const int n_ctx = (int)llama_n_ctx(h->ctx);
    if (n > n_ctx - 16) {
        keryx_set_error("prompt", std::to_string(n) + " prompt tokens do not fit the " + std::to_string(n_ctx) + "-token context");
        return -2;
    }

    llama_memory_clear(llama_get_memory(h->ctx), true);
    // Penalty history must not carry over from earlier requests.
    llama_sampler_reset(h->smpl);
    keryx_trace_reset(h);
    if (keryx_feed_prompt(h, toks.data(), n) != 0) return -3;
    int written = 0;
    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (pn < 0) break;
        if (written + pn >= cap - 1) break;
        memcpy(out + written, piece, pn);
        written += pn;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        keryx_trace_batch(h->trace, &tok, 1, n + i);
        if (llama_decode(h->ctx, batch) != 0) break;
    }
    out[written] = 0;
    return written;
}

// Like keryx_llama_generate, and records every token fed to the model (prompt, then each decoded
// token) in `tokens`: that sequence replays the response. Generation stops when `tokens` is full.
// The prompt is padded to `min_prompt` tokens, and padding positions are decoded after the answer
// until `answer bytes / answer_bytes_per_position` (at most `max_tokens`) have been generated.
// With `n_prefix > 0`, the fed prompt is `prompt` tokenized alone (it must give `head_tokens`
// tokens, else -8), then the `prefix` token ids, then `rest` tokenized without special tokens.
KERYX_EXPORT int keryx_llama_generate_traced(KeryxLlama* h, const char* prompt, const int32_t* prefix, int n_prefix,
                                             int head_tokens, const char* rest, int max_tokens, char* out, int cap,
                                             int32_t* tokens, int tokens_cap, int* n_prompt_out, int* n_fed_out,
                                             int min_prompt, int answer_bytes_per_position) {
    if (!h || !prompt || !out || cap < 2 || !tokens || tokens_cap < 1 || !n_prompt_out || !n_fed_out) return -1;
    if (n_prefix > 0 && (!prefix || !rest)) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (!keryx_has_ctx(h)) return -2;
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    std::vector<llama_token> toks(strlen(prompt) + 16);
    int n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks.data(), (int32_t)toks.size(), true, true);
    if (n <= 0) return -1;
    toks.resize(n);
    if (n_prefix > 0) {
        if (n != head_tokens) {
            keryx_set_error("prefix", "the prompt head tokenizes to " + std::to_string(n) + " tokens, " +
                                      std::to_string(head_tokens) + " expected");
            return -8;
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_prefix; i++) {
            if (prefix[i] < 0 || prefix[i] >= n_vocab) {
                keryx_set_error("prefix", "prefix token out of the vocabulary");
                return -1;
            }
            toks.push_back((llama_token)prefix[i]);
        }
        std::vector<llama_token> rt(strlen(rest) + 16);
        const int nr = llama_tokenize(vocab, rest, (int32_t)strlen(rest), rt.data(), (int32_t)rt.size(), false, true);
        if (nr < 0) return -1;
        toks.insert(toks.end(), rt.begin(), rt.begin() + nr);
        n = (int)toks.size();
    }
    const llama_token pad = keryx_pad_token(vocab);
    if (min_prompt > n) {
        toks.resize((size_t)min_prompt, pad);
        n = min_prompt;
    }

    const int n_ctx = (int)llama_n_ctx(h->ctx);
    if (n > n_ctx - 16) {
        keryx_set_error("prompt", std::to_string(n) + " prompt tokens do not fit the " + std::to_string(n_ctx) + "-token context");
        return -2;
    }
    if (n > tokens_cap) {
        keryx_set_error("prompt", "token buffer too small for the prompt");
        return -1;
    }

    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_sampler_reset(h->smpl);
    if (keryx_trace_reset(h) != 0) {
        keryx_set_error("trace", "the trace could not be reset");
        return -5;
    }
    memcpy(tokens, toks.data(), (size_t)n * sizeof(int32_t));
    int fed = n;
    if (keryx_feed_prompt(h, toks.data(), n) != 0) return -3;
    int written = 0;
    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (pn < 0) break;
        if (written + pn >= cap - 1) break;
        memcpy(out + written, piece, pn);
        written += pn;
        if (fed >= tokens_cap) break;
        tokens[fed++] = tok;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        keryx_trace_batch(h->trace, &tok, 1, fed - 1);
        if (llama_decode(h->ctx, batch) != 0) break;
    }
    if (answer_bytes_per_position > 0) {
        const int need = std::min(written / answer_bytes_per_position, max_tokens);
        while (fed - n < need && fed < tokens_cap && fed < n_ctx - 1) {
            llama_token tok = pad;
            tokens[fed++] = tok;
            llama_batch batch = llama_batch_get_one(&tok, 1);
            keryx_trace_batch(h->trace, &tok, 1, fed - 1);
            if (llama_decode(h->ctx, batch) != 0) break;
        }
    }
    out[written] = 0;
    *n_prompt_out = n;
    *n_fed_out = fed;
    return written;
}

// Traced evaluation of `tokens` as a prompt with nothing generated; the trace is read with
// keryx_llama_trace_read. Returns 0 on success, -1 (arguments), -2 (context), -3 (decode),
// -5 (trace).
KERYX_EXPORT int keryx_llama_trace_tokens(KeryxLlama* h, const int32_t* tokens, int n_tokens) {
#ifdef __APPLE__
    (void)h; (void)tokens; (void)n_tokens;
    return -1;
#else
    if (!h || !h->traced || !tokens || n_tokens <= 0) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (!keryx_has_ctx(h)) return -2;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(h->model));
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) return -1;
    }
    if (n_tokens > (int)llama_n_ctx(h->ctx)) return -2;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_sampler_reset(h->smpl);
    if (keryx_trace_reset(h) != 0) return -5;
    return keryx_feed_prompt(h, tokens, n_tokens);
#endif
}

// Enables the canonical trace on the engine's GPU for the model described by `manifest`
// (`max_pos` positions per response; the context size when 0). Returns 0 on success.
KERYX_EXPORT int keryx_llama_trace_configure(KeryxLlama* h, const void* manifest, size_t manifest_len, int max_pos) {
#ifdef __APPLE__
    (void)h; (void)manifest; (void)manifest_len; (void)max_pos;
    return -1;
#else
    if (!h || !manifest) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (max_pos <= 0 && !keryx_has_ctx(h)) return -2;
    const uint32_t mp = max_pos > 0 ? (uint32_t)max_pos : (uint32_t)llama_n_ctx(h->ctx);
    const int rc = ggml_cuda_q8c_configure(h->gpu, manifest, manifest_len, mp);
    h->traced = rc == 0;
    if (rc != 0) keryx_set_error("trace", "trace configuration failed (" + std::to_string(rc) + ")");
    return rc;
#endif
}

// Leaves per traced position, or -1 when the trace is not configured.
KERYX_EXPORT int keryx_llama_trace_leaf_count(KeryxLlama* h) {
#ifdef __APPLE__
    (void)h;
    return -1;
#else
    return h && h->traced ? ggml_cuda_q8c_leaf_count(h->gpu) : -1;
#endif
}

// Leaves (n_pos × leaf_count × 32 bytes) and ubatch sizes of the last traced run. Returns 0 on
// success; `n_pos` is set even when the buffers are null, to size them.
KERYX_EXPORT int keryx_llama_trace_read(KeryxLlama* h, uint8_t* leaves, size_t leaves_cap, uint32_t* ubatch,
                                        size_t ubatch_cap, uint32_t* n_pos) {
#ifdef __APPLE__
    (void)h; (void)leaves; (void)leaves_cap; (void)ubatch; (void)ubatch_cap; (void)n_pos;
    return -1;
#else
    if (!h || !h->traced) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    return ggml_cuda_q8c_read(h->gpu, leaves, leaves_cap, ubatch, ubatch_cap, n_pos);
#endif
}

#ifndef __APPLE__
// Feeds `tokens` up to position `pos` with the batching of the original run. The caller holds
// `gen_lock` and has armed the captures.
static int keryx_replay_to(KeryxLlama* h, const int32_t* tokens, int n_prompt, uint32_t pos) {
    const int n_batch = std::max(1, (int)llama_n_batch(h->ctx));
    llama_token* toks = const_cast<llama_token*>(tokens);
    // The prompt is fed in the same n_batch chunks as the original run, up to the chunk holding `pos`.
    const int prompt_end = pos < (uint32_t)n_prompt ? std::min(n_prompt, ((int)pos / n_batch + 1) * n_batch) : n_prompt;
    for (int i = 0; i < prompt_end; i += n_batch) {
        llama_batch batch = llama_batch_get_one(toks + i, std::min(n_batch, prompt_end - i));
        if (llama_decode(h->ctx, batch) != 0) return -3;
    }
    for (int i = n_prompt; pos >= (uint32_t)n_prompt && i <= (int)pos; i++) {
        llama_batch batch = llama_batch_get_one(toks + i, 1);
        if (llama_decode(h->ctx, batch) != 0) return -3;
    }
    return 0;
}

static int keryx_replay_begin(KeryxLlama* h, int n_tokens) {
    if (!keryx_has_ctx(h)) return -2;
    if (n_tokens > (int)llama_n_ctx(h->ctx)) return -2;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_sampler_reset(h->smpl);
    if (h->sequenced && ggml_cuda_q8s_begin_response(h->gpu) != 0) return -5;
    return ggml_cuda_q8c_begin_response(h->gpu) != 0 ? -5 : 0;
}
#endif

// Replays the fed token sequence of a traced response up to position `pos` with the same batching
// and copies the bytes of leaves `leaf_x` and `leaf_y` at that position. Returns 0 on success,
// -1 (arguments), -2 (context), -3 (decode), -5 (trace), -6 (buffers), -7 (nothing captured).
KERYX_EXPORT int keryx_llama_replay(KeryxLlama* h, const int32_t* tokens, int n_tokens, int n_prompt, uint32_t pos,
                                    uint32_t leaf_x, uint32_t leaf_y, uint8_t* x, size_t x_cap, uint8_t* y, size_t y_cap,
                                    uint32_t* x_len, uint32_t* y_len) {
#ifdef __APPLE__
    (void)h; (void)tokens; (void)n_tokens; (void)n_prompt; (void)pos; (void)leaf_x; (void)leaf_y;
    (void)x; (void)x_cap; (void)y; (void)y_cap; (void)x_len; (void)y_len;
    return -1;
#else
    if (!h || !h->traced || !tokens || n_prompt <= 0 || n_tokens < n_prompt || pos >= (uint32_t)n_tokens ||
        !x || !y || !x_len || !y_len) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    int rc = keryx_replay_begin(h, n_tokens);
    if (rc != 0) return rc;
    if (ggml_cuda_q8c_arm_capture(h->gpu, pos, leaf_x, leaf_y) != 0) return -5;
    rc = keryx_replay_to(h, tokens, n_prompt, pos);
    if (rc != 0) return rc;
    return ggml_cuda_q8c_read_capture(h->gpu, x, x_cap, y, y_cap, x_len, y_len);
#endif
}

// Replays a traced response up to the last of `n` positions and copies leaf `leaf[k]` at
// position `pos[k]` into `out + k * slot_bytes`, its length into `lens[k]`. Same return codes as
// keryx_llama_replay.
KERYX_EXPORT int keryx_llama_replay_multi(KeryxLlama* h, const int32_t* tokens, int n_tokens, int n_prompt,
                                          const uint32_t* pos, const uint32_t* leaf, uint32_t n, uint32_t slot_bytes,
                                          uint8_t* out, size_t out_cap, uint32_t* lens) {
#ifdef __APPLE__
    (void)h; (void)tokens; (void)n_tokens; (void)n_prompt; (void)pos; (void)leaf; (void)n; (void)slot_bytes;
    (void)out; (void)out_cap; (void)lens;
    return -1;
#else
    if (!h || !h->traced || !tokens || n_prompt <= 0 || n_tokens < n_prompt || !pos || !leaf || n == 0 || !out || !lens) return -1;
    uint32_t last = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (pos[k] >= (uint32_t)n_tokens) return -1;
        last = std::max(last, pos[k]);
    }
    std::lock_guard<std::mutex> g(h->gen_lock);
    int rc = keryx_replay_begin(h, n_tokens);
    if (rc != 0) return rc;
    if (ggml_cuda_q8c_arm_captures(h->gpu, pos, leaf, n, slot_bytes) != 0) return -5;
    rc = keryx_replay_to(h, tokens, n_prompt, last);
    if (rc != 0) return rc;
    return ggml_cuda_q8c_read_captures(h->gpu, out, out_cap, lens, n);
#endif
}

// Enables the sequence leaves on the engine's GPU: `layers` holds `n` rows of (layer, kind, slot,
// heads, head_dim), `interval` positions between recurrent state checkpoints; `n` = 0 disables
// them. Returns 0 on success.
KERYX_EXPORT int keryx_llama_seq_configure(KeryxLlama* h, const uint32_t* layers, uint32_t n, uint32_t n_attn,
                                           uint32_t n_recur, uint32_t seq_heads, uint32_t interval) {
#ifdef __APPLE__
    (void)h; (void)layers; (void)n; (void)n_attn; (void)n_recur; (void)seq_heads; (void)interval;
    return -1;
#else
    if (!h || (n > 0 && !layers)) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (n == 0) {
        if (h->sequenced) ggml_cuda_q8s_release(h->gpu);
        h->sequenced = false;
        return 0;
    }
    if (!keryx_has_ctx(h)) return -2;
    const int rc = ggml_cuda_q8s_configure(h->gpu, layers, n, n_attn, n_recur, seq_heads, interval, llama_n_ubatch(h->ctx));
    h->sequenced = rc == 0;
    if (rc != 0) keryx_set_error("trace", "sequence leaf configuration failed (" + std::to_string(rc) + ")");
    return rc;
#endif
}

// Sequence leaf hashes of the last traced run (`n_leaves` × 32 bytes). Returns 0 on success.
KERYX_EXPORT int keryx_llama_seq_roots(KeryxLlama* h, uint8_t* out, size_t out_cap, uint32_t* n_leaves) {
#ifdef __APPLE__
    (void)h; (void)out; (void)out_cap; (void)n_leaves;
    return -1;
#else
    if (!h || !h->sequenced || !out || !n_leaves) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    return ggml_cuda_q8s_roots(h->gpu, out, out_cap, n_leaves);
#endif
}

// Replays a traced response to its end and copies bytes `[lo[k], hi[k])` of sequence leaf
// `stream[k]` end to end into `bytes` (`lens[k]` each), with the chunk chaining values of each leaf
// (`n_cvs[k]` × 32 bytes) into `cvs`. Same return codes as keryx_llama_replay.
KERYX_EXPORT int keryx_llama_seq_replay(KeryxLlama* h, const int32_t* tokens, int n_tokens, int n_prompt,
                                        const uint32_t* stream, const uint64_t* lo, const uint64_t* hi, uint32_t n,
                                        uint32_t max_chunks, uint8_t* bytes, size_t bytes_cap, uint64_t* lens,
                                        uint8_t* cvs, size_t cvs_cap, uint32_t* n_cvs) {
#ifdef __APPLE__
    (void)h; (void)tokens; (void)n_tokens; (void)n_prompt; (void)stream; (void)lo; (void)hi; (void)n; (void)max_chunks;
    (void)bytes; (void)bytes_cap; (void)lens; (void)cvs; (void)cvs_cap; (void)n_cvs;
    return -1;
#else
    if (!h || !h->traced || !h->sequenced || !tokens || n_prompt <= 0 || n_tokens < n_prompt || !stream || !lo || !hi
        || n == 0 || !bytes || !lens || !cvs || !n_cvs) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (ggml_cuda_q8s_arm(h->gpu, stream, lo, hi, n, max_chunks) != 0) return -5;
    int rc = keryx_replay_begin(h, n_tokens);
    if (rc != 0) return rc;
    rc = keryx_replay_to(h, tokens, n_prompt, (uint32_t)n_tokens - 1);
    if (rc != 0) return rc;
    uint32_t n_leaves = 0;
    std::vector<uint8_t> roots((size_t)2 * 1024 * 1024);
    if (ggml_cuda_q8s_roots(h->gpu, roots.data(), roots.size(), &n_leaves) != 0) return -5;
    return ggml_cuda_q8s_read_captures(h->gpu, bytes, bytes_cap, lens, cvs, cvs_cap, n_cvs, n) != 0 ? -6 : 0;
#endif
}

// Runs one tiny graph on `gpu` so a library without kernels for that device fails here rather
// than on the first request. ggml aborts the process on a missing kernel image: call from a
// child process. Returns 0 on success.
KERYX_EXPORT int keryx_llama_probe_device(int gpu) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    if (gpu < 0 || gpu >= ggml_backend_cuda_get_device_count()) {
        keryx_set_error("probe", "no such CUDA device");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        keryx_set_error("probe", "ggml_backend_cuda_init failed");
        return 2;
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    int rc = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        keryx_set_error("probe", "ggml_backend_alloc_ctx_tensors failed");
        rc = 3;
    } else {
        const float v[4] = { 1.f, 2.f, 3.f, 4.f };
        ggml_backend_tensor_set(a, v, 0, sizeof v);
        ggml_backend_tensor_set(b, v, 0, sizeof v);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            keryx_set_error("probe", "graph compute failed");
            rc = 4;
        } else {
            float out[4] = { 0.f, 0.f, 0.f, 0.f };
            ggml_backend_tensor_get(c, out, 0, sizeof out);
            if (out[0] != 2.f || out[3] != 8.f) {
                keryx_set_error("probe", "graph compute returned wrong values");
                rc = 5;
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return rc;
}

KERYX_EXPORT void keryx_llama_free(KeryxLlama* h) {
#ifndef __APPLE__
    if (h && h->traced) ggml_cuda_q8c_release(h->gpu);
    if (h && h->sequenced) ggml_cuda_q8s_release(h->gpu);
#endif
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    keryx_trace_close(h->trace);
    delete h;
}

} // extern "C"
