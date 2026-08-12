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

struct KeryxLlama {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    std::vector<std::string> names; // canonical (byte-lexicographic) order — matches pom.rs
    std::mutex gen_lock;
};

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

extern "C" {

// ABI version — the miner refuses to use a mismatched .so.
KERYX_EXPORT int keryx_llama_abi() { return 3; }

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
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        keryx_set_error("model", std::string("llama_model_load_from_file failed for ") + gguf_path);
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        keryx_set_error("context", "llama_init_from_model failed");
        llama_model_free(model);
        return nullptr;
    }

    // User-facing sampling (repeat penalty -> temperature 0.7 / top_p 0.9) — the OPoI
    // text is not consensus-relevant, but keep the flavor consistent.
    // The repetition penalty is essential: without it the small quantized models
    // (9B Q4 especially) degenerate into verbatim sentence loops. 256-token window
    // because the observed loops are whole sentences (~25 tokens each), far beyond
    // the classic 64-token window; 1.10 is the battle-tested llama.cpp default.
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto* h = new KeryxLlama();
    h->model = model; h->ctx = ctx; h->smpl = smpl;
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

KERYX_EXPORT size_t keryx_llama_tensor_count(KeryxLlama* h) { return h ? h->names.size() : 0; }

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
KERYX_EXPORT int keryx_llama_generate(KeryxLlama* h, const char* prompt, int max_tokens, char* out, int cap) {
    if (!h || !prompt || !out || cap < 2) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    std::vector<llama_token> toks(strlen(prompt) + 16);
    int n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) return -1;
    toks.resize(n);

    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t)toks.size());
    int written = 0;
    for (int i = 0; i < max_tokens; i++) {
        if (llama_decode(h->ctx, batch) != 0) break;
        llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (pn < 0) break;
        if (written + pn >= cap - 1) break;
        memcpy(out + written, piece, pn);
        written += pn;
        batch = llama_batch_get_one(&tok, 1);
    }
    out[written] = 0;
    return written;
}

KERYX_EXPORT void keryx_llama_free(KeryxLlama* h) {
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

} // extern "C"
