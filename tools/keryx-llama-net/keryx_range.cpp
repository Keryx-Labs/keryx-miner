// libkeryx-llama-net.so — layer ranges of the network model, on its own llama.cpp build.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define KERYX_EXPORT __attribute__((visibility("default")))

#ifdef KERYX_RANGE_TRACE
extern "C" {
int ggml_cuda_q8c_configure(int device, const void * manifest, size_t manifest_len, uint32_t max_pos);
int ggml_cuda_q8c_leaf_count(int device);
int ggml_cuda_q8c_set_pos(int device, uint32_t pos0, uint32_t n);
int ggml_cuda_q8c_read_range(int device, uint32_t pos0, uint32_t n, uint8_t * leaves, size_t leaves_cap,
                             uint32_t * ubatch, size_t ubatch_cap);
int ggml_cuda_q8c_arm_captures(int device, const uint32_t * pos, const uint32_t * leaf, uint32_t n, uint32_t slot_bytes);
int ggml_cuda_q8c_read_captures(int device, uint8_t * out, size_t out_cap, uint32_t * lens, uint32_t n);
}
#endif

extern "C" {
extern thread_local int32_t keryx_llama_range_start;
extern thread_local int32_t keryx_llama_range_end;
extern thread_local ggml_backend_buffer_t keryx_llama_local_resident_buf;
extern thread_local size_t                keryx_llama_local_resident_n;
extern thread_local const char * const *  keryx_llama_local_resident_names;
extern thread_local void * const *        keryx_llama_local_resident_datas;
extern thread_local const size_t *        keryx_llama_local_resident_nbytes;
}

static thread_local std::string keryx_last_error;

static void keryx_set_error(const char* stage, const std::string& detail) {
    std::string diag;
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess) diag += std::string(" [cuda: ") + cudaGetErrorString(pending) + "]";
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        diag += " [vram: " + std::to_string(free_bytes / (1024 * 1024)) + " MiB free / "
              + std::to_string(total_bytes / (1024 * 1024)) + " MiB total]";
    }
    keryx_last_error = std::string(stage) + ": " + detail + diag;
}

static void keryx_log_cb(enum ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static void keryx_install_log_filter() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (getenv("KERYX_LLAMA_VERBOSE")) return;
        llama_log_set(keryx_log_cb, nullptr);
        ggml_log_set(keryx_log_cb, nullptr);
    });
}

// Same sampling chain as the lineup engine.
static llama_sampler* keryx_make_sampler(const llama_model* model) {
    static const char* dry_breakers[] = { "\n", ":", "\"", "*" };
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_dry(
        llama_model_get_vocab(model),
        0.8f, 1.75f, 2, -1, dry_breakers, sizeof(dry_breakers) / sizeof(dry_breakers[0])));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(llama_model_get_vocab(model)), 256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
    return smpl;
}

#ifdef KERYX_RANGE_TRACE
static std::mutex keryx_range_cfg_mu;
static std::string keryx_range_cfg[16];
#endif

extern "C" {

KERYX_EXPORT int keryx_range_abi() { return 1; }

KERYX_EXPORT const char* keryx_range_last_error() { return keryx_last_error.c_str(); }

struct KeryxRange {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    ggml_backend_buffer_t resident = nullptr;
    int  first = 0, last = 0;
    bool head  = false;
    int  n_hc  = 0;
    int  n_vocab = 0;
    int  n_ctx = 0;
    int  n_seq = 0;
    int  n_ubatch = 0;
    std::vector<int32_t> len;
    llama_sampler* smpl = nullptr;
    int  sample_seq = -1;
    int  gpu    = 0;
    bool traced = false;
    std::mutex mu;
};

static const int KERYX_RANGE_MAX_UBATCH = 512;

// Loads layers [first, last] of `gguf_path` on `gpu`. The `n_resident` tensors already held in
// device memory [resident_base, resident_base + resident_size) are bound by name, not loaded.
KERYX_EXPORT KeryxRange* keryx_range_load(const char* gguf_path, int gpu, int first, int last,
                                          void* resident_base, size_t resident_size, size_t n_resident,
                                          const char* const* names, void* const* datas, const size_t* nbytes,
                                          int n_ctx, int n_seq) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    if (!gguf_path || first < 0 || last < first || n_ctx <= 0 || n_seq <= 0 || n_seq > 16
        || (n_resident > 0 && (!resident_base || !names || !datas || !nbytes))) {
        keryx_set_error("range", "bad range parameters");
        return nullptr;
    }
    for (size_t i = 0; i < n_resident; i++) {
        const char* d = (const char*) datas[i];
        const char* b = (const char*) resident_base;
        if (!names[i] || d < b || nbytes[i] > resident_size || (size_t) (d - b) > resident_size - nbytes[i]) {
            keryx_set_error("range", "resident tensor outside the resident buffer");
            return nullptr;
        }
    }
    llama_backend_init();
    ggml_backend_buffer_t resident = nullptr;
    if (n_resident > 0) {
        resident = ggml_backend_cuda_buffer_from_ptr(gpu, resident_base, resident_size);
        if (!resident) {
            keryx_set_error("range", "cannot wrap the resident buffer");
            return nullptr;
        }
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.main_gpu = gpu;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    keryx_llama_range_start = first;
    keryx_llama_range_end = last;
    keryx_llama_local_resident_buf = resident;
    keryx_llama_local_resident_n = n_resident;
    keryx_llama_local_resident_names = names;
    keryx_llama_local_resident_datas = datas;
    keryx_llama_local_resident_nbytes = nbytes;
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    keryx_llama_range_start = -1;
    keryx_llama_range_end = -1;
    keryx_llama_local_resident_buf = nullptr;
    keryx_llama_local_resident_n = 0;
    keryx_llama_local_resident_names = nullptr;
    keryx_llama_local_resident_datas = nullptr;
    keryx_llama_local_resident_nbytes = nullptr;
    if (!model) {
        if (resident) ggml_backend_buffer_free(resident);
        keryx_set_error("range", std::string("llama_model_load_from_file failed for ") + gguf_path);
        return nullptr;
    }
    const int n_layer = llama_model_n_layer(model);
    if (last >= n_layer) {
        llama_model_free(model);
        if (resident) ggml_backend_buffer_free(resident);
        keryx_set_error("range", "range beyond the model's layers");
        return nullptr;
    }
    const bool head = last == n_layer - 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_seq_max = (uint32_t) n_seq;
    cp.n_ctx = (uint32_t) n_ctx * (uint32_t) n_seq;
    cp.n_batch = KERYX_RANGE_MAX_UBATCH;
    cp.n_ubatch = KERYX_RANGE_MAX_UBATCH;
    cp.swa_full = false;
    cp.kv_unified = false;
    cp.embeddings = !head;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        ctx = llama_init_from_model(model, cp);
    }
    if (!ctx) {
        llama_model_free(model);
        if (resident) ggml_backend_buffer_free(resident);
        keryx_set_error("range", "llama_init_from_model failed");
        return nullptr;
    }
    auto* r = new KeryxRange();
    r->model = model;
    r->ctx = ctx;
    r->resident = resident;
    r->first = first;
    r->last = last;
    r->head = head;
    r->n_hc = head ? (int) llama_model_n_embd_inp(model) : (int) llama_model_n_embd_out(model);
    r->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    r->n_ctx = n_ctx;
    r->n_seq = n_seq;
    r->n_ubatch = KERYX_RANGE_MAX_UBATCH;
    r->len.assign((size_t) n_seq, 0);
    r->gpu = gpu;
    if (head) r->smpl = keryx_make_sampler(model);
    return r;
}

// Residual floats per token and vocabulary size of a loaded range.
KERYX_EXPORT void keryx_range_dims(KeryxRange* r, int* n_hc, int* n_vocab, int* head) {
    if (!r) return;
    if (n_hc) *n_hc = r->n_hc;
    if (n_vocab) *n_vocab = r->n_vocab;
    if (head) *head = r->head ? 1 : 0;
}

// Forgets sequence `seq`.
KERYX_EXPORT void keryx_range_clear(KeryxRange* r, int seq) {
    if (!r || seq < 0 || seq >= r->n_seq) return;
    std::lock_guard<std::mutex> lock(r->mu);
    llama_memory_seq_rm(llama_get_memory(r->ctx), seq, -1, -1);
    r->len[(size_t) seq] = 0;
}

static int keryx_range_step_locked(KeryxRange* r, int seq, int pos0, int n, const int32_t* tokens, const float* in,
                                   float* out, size_t out_floats, uint8_t* leaves, size_t leaves_cap, uint32_t* ubatch,
                                   size_t ubatch_cap) {
    r->sample_seq = -1;
    if (seq < 0 || seq >= r->n_seq || n <= 0 || n > r->n_ubatch) return -2;
    if (pos0 != r->len[(size_t) seq] || pos0 + n > r->n_ctx) return -3;
    const bool from_tokens = r->first == 0;
    if (from_tokens ? (tokens == nullptr || in != nullptr) : (in == nullptr || tokens != nullptr)) return -4;
    if (from_tokens) {
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= r->n_vocab) return -5;
        }
    } else {
        const size_t m = (size_t) n * (size_t) r->n_hc;
        for (size_t i = 0; i < m; i++) {
            if (!std::isfinite(in[i])) return -6;
        }
    }
    const size_t want = r->head ? (size_t) r->n_vocab : (size_t) n * (size_t) r->n_hc;
    if (!out || out_floats != want) return -7;

    llama_batch batch = llama_batch_init(n, from_tokens ? 0 : r->n_hc, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        if (from_tokens) batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = r->head ? (i == n - 1) : 1;
    }
    if (!from_tokens) {
        std::memcpy(batch.embd, in, (size_t) n * (size_t) r->n_hc * sizeof(float));
    }
#ifdef KERYX_RANGE_TRACE
    if (r->traced && ggml_cuda_q8c_set_pos(r->gpu, (uint32_t) pos0, (uint32_t) n) != 0) {
        llama_batch_free(batch);
        return -10;
    }
#else
    (void) leaves; (void) leaves_cap; (void) ubatch; (void) ubatch_cap;
#endif
    const int rc = llama_decode(r->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        llama_memory_seq_rm(llama_get_memory(r->ctx), seq, -1, -1);
        r->len[(size_t) seq] = 0;
        return -8;
    }
    const float* res = r->head ? llama_get_logits_ith(r->ctx, n - 1) : llama_get_embeddings(r->ctx);
    if (!res) {
        llama_memory_seq_rm(llama_get_memory(r->ctx), seq, -1, -1);
        r->len[(size_t) seq] = 0;
        return -9;
    }
    std::memcpy(out, res, want * sizeof(float));
#ifdef KERYX_RANGE_TRACE
    if (leaves && ggml_cuda_q8c_read_range(r->gpu, (uint32_t) pos0, (uint32_t) n, leaves, leaves_cap, ubatch, ubatch_cap) != 0) {
        llama_memory_seq_rm(llama_get_memory(r->ctx), seq, -1, -1);
        r->len[(size_t) seq] = 0;
        return -10;
    }
#endif
    r->len[(size_t) seq] += n;
    if (r->head) r->sample_seq = seq;
    return 0;
}

// Runs `n` tokens at positions [pos0, pos0+n) of sequence `seq` through the range. The first
// range takes `tokens`, the others the residual `in` (n × n_hc floats). Writes the residual
// (n × n_hc floats) or, for the head range, the last token's logits (n_vocab floats) to `out`.
// 0 on success; a negative code leaves the sequence cleared.
KERYX_EXPORT int keryx_range_step(KeryxRange* r, int seq, int pos0, int n, const int32_t* tokens, const float* in,
                                  float* out, size_t out_floats) {
    if (!r) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    return keryx_range_step_locked(r, seq, pos0, n, tokens, in, out, out_floats, nullptr, 0, nullptr, 0);
}

// keryx_range_step on a traced range, also copying the trace leaves (n × leaf count × 32 bytes) and
// ubatch sizes of the step's positions. -10 when the trace fails.
KERYX_EXPORT int keryx_range_step_traced(KeryxRange* r, int seq, int pos0, int n, const int32_t* tokens, const float* in,
                                         float* out, size_t out_floats, uint8_t* leaves, size_t leaves_cap,
                                         uint32_t* ubatch, size_t ubatch_cap) {
    if (!r || !leaves || !ubatch) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    if (!r->traced) return -10;
    return keryx_range_step_locked(r, seq, pos0, n, tokens, in, out, out_floats, leaves, leaves_cap, ubatch, ubatch_cap);
}

// Traces the range on its GPU for `manifest` (`max_pos` positions per sequence). 0 on success.
KERYX_EXPORT int keryx_range_trace_configure(KeryxRange* r, const void* manifest, size_t manifest_len, int max_pos) {
#ifdef KERYX_RANGE_TRACE
    if (!r || !manifest || max_pos <= 0 || r->gpu < 0 || r->gpu >= 16) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    std::lock_guard<std::mutex> cl(keryx_range_cfg_mu);
    std::string* cfg = keryx_range_cfg;
    const std::string want = std::string((const char*) manifest, manifest_len) + std::to_string(max_pos);
    int rc = 0;
    if (cfg[r->gpu] != want) {
        rc = ggml_cuda_q8c_configure(r->gpu, manifest, manifest_len, (uint32_t) max_pos);
        cfg[r->gpu] = rc == 0 ? want : std::string();
    }
    r->traced = rc == 0;
    if (rc != 0) keryx_set_error("range", "trace configuration failed (" + std::to_string(rc) + ")");
    return rc;
#else
    (void) r; (void) manifest; (void) manifest_len; (void) max_pos;
    keryx_set_error("range", "built without range traces");
    return -1;
#endif
}

// Leaves per traced position of the range, -1 when not traced.
KERYX_EXPORT int keryx_range_trace_leaf_count(KeryxRange* r) {
#ifdef KERYX_RANGE_TRACE
    return r && r->traced ? ggml_cuda_q8c_leaf_count(r->gpu) : -1;
#else
    (void) r;
    return -1;
#endif
}

// Arms the capture of leaf `leaf[k]` at position `pos[k]` for the next traced steps.
KERYX_EXPORT int keryx_range_arm_captures(KeryxRange* r, const uint32_t* pos, const uint32_t* leaf, uint32_t n,
                                          uint32_t slot_bytes) {
#ifdef KERYX_RANGE_TRACE
    if (!r || !r->traced) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    return ggml_cuda_q8c_arm_captures(r->gpu, pos, leaf, n, slot_bytes);
#else
    (void) r; (void) pos; (void) leaf; (void) n; (void) slot_bytes;
    return -1;
#endif
}

// Copies the armed captures into `out + k * slot_bytes`, their lengths into `lens`.
KERYX_EXPORT int keryx_range_read_captures(KeryxRange* r, uint8_t* out, size_t out_cap, uint32_t* lens, uint32_t n) {
#ifdef KERYX_RANGE_TRACE
    if (!r || !r->traced) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    return ggml_cuda_q8c_read_captures(r->gpu, out, out_cap, lens, n);
#else
    (void) r; (void) out; (void) out_cap; (void) lens; (void) n;
    return -1;
#endif
}

// Tokenizes `text` (`len` bytes) with the model's vocabulary, special tokens parsed. Returns the
// token count, or minus the count needed when `cap` is too small.
KERYX_EXPORT int keryx_range_tokenize(KeryxRange* r, const char* text, int len, int32_t* out, int cap) {
    if (!r || !text || len < 0 || cap < 0 || (!out && cap > 0)) return INT32_MIN;
    const llama_vocab* vocab = llama_model_get_vocab(r->model);
    return llama_tokenize(vocab, text, len, out, cap, true, true);
}

// Clears the head range's sampling history (start of a new generation).
KERYX_EXPORT void keryx_range_sampler_reset(KeryxRange* r) {
    if (!r || !r->smpl) return;
    std::lock_guard<std::mutex> lock(r->mu);
    llama_sampler_reset(r->smpl);
}

// Samples the next token of `seq` from the logits its last head step left; -1 when there are none.
KERYX_EXPORT int32_t keryx_range_sample(KeryxRange* r, int seq) {
    if (!r || !r->smpl) return -1;
    std::lock_guard<std::mutex> lock(r->mu);
    if (r->sample_seq != seq || seq < 0) return -1;
    r->sample_seq = -1;
    return llama_sampler_sample(r->smpl, r->ctx, -1);
}

// Text of token `tok` into `buf`: its byte length, -2 for an end-of-generation token, -1 on error.
KERYX_EXPORT int keryx_range_piece(KeryxRange* r, int32_t tok, char* buf, int cap) {
    if (!r || !buf || cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(r->model);
    if (tok < 0 || tok >= llama_vocab_n_tokens(vocab)) return -1;
    if (llama_vocab_is_eog(vocab, tok)) return -2;
    const int n = llama_token_to_piece(vocab, tok, buf, cap, 0, true);
    return n < 0 ? -1 : n;
}

KERYX_EXPORT void keryx_range_free(KeryxRange* r) {
    if (!r) return;
    if (r->smpl) llama_sampler_free(r->smpl);
    if (r->ctx) llama_free(r->ctx);
    if (r->model) llama_model_free(r->model);
    if (r->resident) ggml_backend_buffer_free(r->resident);
    delete r;
}

} // extern "C"
