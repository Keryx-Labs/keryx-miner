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
#include "ggml-rpc.h"
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
#include <sstream>
#include <thread>
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
KERYX_EXPORT int keryx_llama_abi() { return 5; }

// Reason for the last failed call on this thread; empty when none. Valid until the next call.
KERYX_EXPORT const char* keryx_llama_last_error() { return keryx_last_error.c_str(); }

static KeryxLlama* keryx_load_with(const char* gguf_path, int n_ctx, const llama_model_params& mp) {
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        keryx_set_error("model", std::string("llama_model_load_from_file failed for ") + gguf_path);
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    cp.n_batch = std::min(cp.n_batch, cp.n_ctx);
    cp.n_ubatch = std::min(cp.n_ubatch, std::max(1u, cp.n_ctx / 4));
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        keryx_set_error("context", "llama_init_from_model failed");
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
    h->model = model; h->ctx = ctx; h->smpl = smpl;
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

KERYX_EXPORT KeryxLlama* keryx_llama_load(const char* gguf_path, int gpu, int n_ctx) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE; // ONE GPU — never layer-split across mining cards
    mp.main_gpu     = gpu;
    mp.use_mmap     = true;
    return keryx_load_with(gguf_path, n_ctx, mp);
}

// Pipeline head: layers split across the listed rpc-server shards (in order) and local GPU `gpu`
// (last). `tensor_split` = one proportion per device, same order. `manifest` (optional) lets the
// head load from a GGUF that holds no data for the remote layers. The handle is inference-only:
// remote tensors have no local device pointer, never walk it.
KERYX_EXPORT KeryxLlama* keryx_llama_load_split(const char* gguf_path, int gpu, int n_ctx,
                                                const char* rpc_endpoints, const char* tensor_split,
                                                const char* manifest) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    if (manifest && *manifest) {
#if defined(_WIN32)
        _putenv_s("KERYX_RPC_MANIFEST", manifest);
#else
        setenv("KERYX_RPC_MANIFEST", manifest, 1);
#endif
    }
    llama_backend_init();
    static std::vector<ggml_backend_dev_t> devices; // llama keeps the pointer for the model's life
    static std::vector<float> split;
    devices.clear(); split.clear();
    std::stringstream eps(rpc_endpoints ? rpc_endpoints : "");
    std::string ep;
    while (std::getline(eps, ep, ',')) {
        if (ep.empty()) continue;
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(ep.c_str());
        if (!reg) {
            keryx_set_error("rpc", "cannot register shard " + ep);
            return nullptr;
        }
        ggml_backend_register(reg);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            devices.push_back(ggml_backend_reg_dev_get(reg, i));
        }
    }
#ifdef __APPLE__
    const std::string local = "MTL" + std::to_string(gpu);
#else
    const std::string local = "CUDA" + std::to_string(gpu);
#endif
    ggml_backend_dev_t local_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (local == ggml_backend_dev_name(d)) { local_dev = d; break; }
    }
    if (!local_dev) {
        keryx_set_error("device", "local device " + local + " not found");
        return nullptr;
    }
    devices.push_back(local_dev);
    std::stringstream ts(tensor_split ? tensor_split : "");
    std::string f;
    while (std::getline(ts, f, ',')) {
        if (!f.empty()) split.push_back(std::stof(f));
    }
    if (split.size() != devices.size()) {
        keryx_set_error("split", "tensor_split has " + std::to_string(split.size()) + " entries for "
                        + std::to_string(devices.size()) + " devices");
        return nullptr;
    }
    split.resize(llama_max_devices(), 0.0f);
    devices.push_back(nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers  = 999;
    mp.split_mode    = LLAMA_SPLIT_MODE_LAYER;
    mp.devices       = devices.data();
    mp.tensor_split  = split.data();
    mp.use_mmap      = true;
    return keryx_load_with(gguf_path, n_ctx, mp);
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
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

// ── Network-model shard (H14) ─────────────────────────────────────────────────────────────────
// A shard GGUF holds a layer range of the network model and is not a model llama can run. The
// miner loads it ONCE on its mining GPU: the PoM walk gathers over these tensors exactly like
// over llama's (same info/device contract), and the in-process rpc server hands the same buffer
// to a pipeline head that binds by name — one VRAM copy for walk and service.
struct KeryxShard {
    gguf_context*          gguf = nullptr;
    ggml_context*          ctx  = nullptr;
    ggml_backend_buffer_t  buf  = nullptr;
    int                    gpu  = -1;
    std::vector<std::string> names;              // canonical (byte-lexicographic) order
    std::vector<ggml_tensor*> tensors;           // parallel to names
    std::string            endpoint;             // set once serving
};

KERYX_EXPORT KeryxShard* keryx_shard_load(const char* gguf_path, int gpu) {
    keryx_last_error.clear();
    keryx_install_log_filter();
#ifdef __APPLE__
    (void)gguf_path; (void)gpu;
    keryx_set_error("shard", "shard serving is CUDA-only");
    return nullptr;
#else
    llama_backend_init();
    ggml_context* meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ &meta };
    gguf_context* g = gguf_init_from_file(gguf_path, gp);
    if (!g || !meta) {
        keryx_set_error("shard", std::string("gguf_init_from_file failed for ") + gguf_path);
        if (g) gguf_free(g);
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(gpu);
    if (!buft) {
        keryx_set_error("shard", "no CUDA buffer type for gpu " + std::to_string(gpu));
        ggml_free(meta); gguf_free(g);
        return nullptr;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(meta, buft);
    if (!buf) {
        keryx_set_error("shard", "ggml_backend_alloc_ctx_tensors_from_buft failed (out of memory?)");
        ggml_free(meta); gguf_free(g);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE* f = fopen(gguf_path, "rb");
    if (!f) {
        keryx_set_error("shard", std::string("cannot open ") + gguf_path);
        ggml_backend_buffer_free(buf); ggml_free(meta); gguf_free(g);
        return nullptr;
    }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> chunk(64u << 20);
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; i++) {
        const char* name = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(meta, name);
        if (!t || !t->data) {
            keryx_set_error("shard", std::string("tensor without device storage: ") + name);
            fclose(f); ggml_backend_buffer_free(buf); ggml_free(meta); gguf_free(g);
            return nullptr;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (nbytes != gguf_get_tensor_size(g, i)) {
            keryx_set_error("shard", std::string("tensor size mismatch: ") + name);
            fclose(f); ggml_backend_buffer_free(buf); ggml_free(meta); gguf_free(g);
            return nullptr;
        }
#if defined(_WIN32)
        _fseeki64(f, (long long)(data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
#else
        fseeko(f, (off_t)(data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
#endif
        size_t done = 0;
        while (done < nbytes) {
            const size_t want = std::min(chunk.size(), nbytes - done);
            if (fread(chunk.data(), 1, want, f) != want) {
                keryx_set_error("shard", std::string("short read in ") + name);
                fclose(f); ggml_backend_buffer_free(buf); ggml_free(meta); gguf_free(g);
                return nullptr;
            }
            ggml_backend_tensor_set(t, chunk.data(), done, want);
            done += want;
        }
    }
    fclose(f);
    auto* h = new KeryxShard();
    h->gguf = g; h->ctx = meta; h->buf = buf; h->gpu = gpu;
    for (int64_t i = 0; i < n; i++) h->names.push_back(gguf_get_tensor_name(g, i));
    std::sort(h->names.begin(), h->names.end());
    for (auto& nm : h->names) h->tensors.push_back(ggml_get_tensor(meta, nm.c_str()));
    return h;
#endif
}

KERYX_EXPORT size_t keryx_shard_tensor_count(KeryxShard* h) { return h ? h->names.size() : 0; }

KERYX_EXPORT bool keryx_shard_tensor_info(KeryxShard* h, size_t i, const char** name, void** data,
                                          size_t* nbytes, int* is_device) {
    if (!h || i >= h->names.size()) return false;
    const ggml_tensor* t = h->tensors[i];
    if (!t || !t->data) return false;
    *name = h->names[i].c_str();
    *data = t->data;
    *nbytes = ggml_nbytes(t);
    *is_device = 1;
    return true;
}

KERYX_EXPORT int keryx_shard_tensor_device(KeryxShard* h, size_t i) {
    if (!h || i >= h->names.size()) return -1;
    return h->gpu;
}

// Serve the resident shard on `endpoint` (host:port, loopback — the miner's gateway fronts it).
// Registers the buffer as the rpc device 0 resident set and runs the server for the life of
// the process. 0 ok, -1 error, 1 already serving.
KERYX_EXPORT int keryx_shard_serve(KeryxShard* h, const char* endpoint, int n_threads) {
#ifdef __APPLE__
    (void)h; (void)endpoint; (void)n_threads;
    return -1;
#else
    if (!h || !endpoint) return -1;
    if (!h->endpoint.empty()) return 1;
    std::string dev_name = "CUDA" + std::to_string(h->gpu);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        keryx_set_error("shard", "no backend device " + dev_name);
        return -1;
    }
    std::vector<const char*> names; std::vector<const void*> datas; std::vector<size_t> sizes;
    for (size_t i = 0; i < h->names.size(); i++) {
        names.push_back(h->names[i].c_str());
        datas.push_back(h->tensors[i]->data);
        sizes.push_back(ggml_nbytes(h->tensors[i]));
    }
    ggml_backend_rpc_set_resident(0, h->buf, names.size(), names.data(), datas.data(), sizes.data());
    h->endpoint = endpoint;
    std::string ep = h->endpoint;
    int threads = n_threads > 0 ? n_threads : 4;
    std::thread([ep, threads, dev]() {
        ggml_backend_dev_t devs[1] = { dev };
        ggml_backend_rpc_start_server(ep.c_str(), nullptr, (size_t)threads, 1, devs);
    }).detach();
    return 0;
#endif
}

KERYX_EXPORT void keryx_shard_free(KeryxShard* h) {
    if (!h) return;
    if (!h->endpoint.empty()) {
        // the server thread keeps running: drop the resident registration so no client can bind
        ggml_backend_rpc_set_resident(0, nullptr, 0, nullptr, nullptr, nullptr);
    }
    if (h->buf) ggml_backend_buffer_free(h->buf);
    if (h->ctx) ggml_free(h->ctx);
    if (h->gguf) gguf_free(h->gguf);
    delete h;
}

} // extern "C"
