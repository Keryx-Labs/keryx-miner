// Keryx Proof-of-Model mining kernel (design A), loaded into candle's CUDA context.
// Per nonce: seed-fold + data-dependent gather walk over the resident weight blob +
// pow-fold + target check. Only mix64 + memory reads (light → high hashrate). Mirrors
// pom-q4-probe::pom_mine. The seed/pow folds MUST match the host (src/pom.rs build_proof).

#include <cstdint>

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ unsigned long long pom_seed_fold(
    unsigned long long nonce, unsigned long long time_,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3) {
    unsigned long long s = mix64(nonce ^ 0x4B65727978531ULL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

__device__ __forceinline__ void pom_pow_fold(
    unsigned long long fin, unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long out[4]) {
    out[0] = mix64(fin ^ p0 ^ 0x9E3779B97F4A7C15ULL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FULL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9ULL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93ULL);
}

__device__ __forceinline__ bool pom_le_leq(const unsigned long long a[4],
                                           unsigned long long b0, unsigned long long b1,
                                           unsigned long long b2, unsigned long long b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// H5.1: the seed fold reads its own word set (s0..s3, host-salted per era) while the pow fold
// keeps p0..p3 — pre-H5.1 the host passes identical words, keeping behavior bit-identical.
//
// ILP x2: each thread grinds TWO nonces with their walk steps interleaved. The walk is
// memory-latency bound (a 3090 sits at ~24% of its bandwidth at 1 nonce/thread), so the
// second walk's search/mix work overlaps the first walk's DRAM fetch and vice versa.
// Byte-exact per nonce: each walk's math is untouched, only the instruction scheduling
// interleaves. The host launches ceil(batch/2) threads (grid calc in src/pom_gpu.rs).
extern "C" __global__ void pom_mine(const unsigned long long* bases, const unsigned long long* prefix,
                                    unsigned int T, unsigned long long n_total_chunks, unsigned int K,
                                    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
                                    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
                                    unsigned long long time_,
                                    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
                                    unsigned long long nonce_base, unsigned long long n_nonces,
                                    unsigned long long* winner, unsigned int walk_v2) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i0 = tid * 2ULL;
    if (i0 >= n_nonces) return;
    unsigned long long i1 = i0 + 1ULL;
    bool has1 = i1 < n_nonces;
    unsigned long long nonce0 = nonce_base + i0;
    // Boundary thread of an odd batch: walk nonce0 twice (result discarded below) rather than
    // branching the whole thread body on has1.
    unsigned long long nonce1 = nonce_base + (has1 ? i1 : i0);

    unsigned long long state0 = pom_seed_fold(nonce0, time_, s0, s1, s2, s3);
    unsigned long long state1 = pom_seed_fold(nonce1, time_, s0, s1, s2, s3);
    unsigned long long off0 = state0 % n_total_chunks;
    unsigned long long off1 = state1 % n_total_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned int lo0 = 0, hi0 = T;
        while (lo0 + 1 < hi0) {
            unsigned int mid = (lo0 + hi0) >> 1;
            if (prefix[mid] <= off0) lo0 = mid; else hi0 = mid;
        }
        unsigned int lo1 = 0, hi1 = T;
        while (lo1 + 1 < hi1) {
            unsigned int mid = (lo1 + hi1) >> 1;
            if (prefix[mid] <= off1) lo1 = mid; else hi1 = mid;
        }
        // 128-bit vector loads: read each 32-byte chunk as 2x ulonglong2 (a.x,a.y,c.x,c.y =
        // w0..w3), BYTE-EXACT vs the 4x u64 form. Both chunks' fetches issue back-to-back so
        // their DRAM latencies overlap (the whole point of the x2 interleave).
        const ulonglong2* q0 = (const ulonglong2*)bases[lo0];
        const ulonglong2* q1 = (const ulonglong2*)bases[lo1];
        unsigned long long b0 = (off0 - prefix[lo0]) * 2ULL;
        unsigned long long b1 = (off1 - prefix[lo1]) * 2ULL;
        ulonglong2 a0 = q0[b0], c0 = q0[b0 + 1];
        ulonglong2 a1 = q1[b1], c1 = q1[b1 + 1];
        unsigned long long h0 = state0, h1 = state1;
        if (walk_v2) {
            // H5 non-foldable walk: chain mix64 through each chunk word so all 32 bytes are
            // load-bearing (closes the pre-H5 XOR-fold that let a miner hold a 4x-smaller table).
            // walk_v2 is uniform across all threads -> no warp divergence. The two chains are
            // independent and interleave in the pipeline.
            h0 = mix64(h0 ^ a0.x); h1 = mix64(h1 ^ a1.x);
            h0 = mix64(h0 ^ a0.y); h1 = mix64(h1 ^ a1.y);
            h0 = mix64(h0 ^ c0.x); h1 = mix64(h1 ^ c1.x);
            h0 = mix64(h0 ^ c0.y); h1 = mix64(h1 ^ c1.y);
            state0 = h0; state1 = h1;
        } else {
            // Pre-H5 fold (frozen — validates all blocks below H5_ACTIVATION_DAA).
            h0 ^= a0.x; h0 ^= a0.y; h0 ^= c0.x; h0 ^= c0.y;
            h1 ^= a1.x; h1 ^= a1.y; h1 ^= c1.x; h1 ^= c1.y;
            state0 = mix64(h0); state1 = mix64(h1);
        }
        off0 = state0 % n_total_chunks;
        off1 = state1 % n_total_chunks;
    }
    unsigned long long pv[4];
    pom_pow_fold(state0, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) {
        atomicMin(winner, nonce0);
    }
    if (has1) {
        pom_pow_fold(state1, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) {
            atomicMin(winner, nonce1);
        }
    }
}

// ============================ PoM v3 (H6 matrix-state walk) ============================
// Byte-exact mirror of the node's consensus/core/src/pom_v3.rs / POM_V3_SPEC.md.
// One CUDA block = one nonce; blockDim.x == 256 == D; thread x owns state row x as 64
// packed uint32 registers. The 64 KB tile lives in dynamic shared memory (requires the
// opt-in shared attribute; cc >= 7.0).

#define V3_D 256
#define V3_D4 (V3_D / 4)
#define V3_K_MAX 256
#define V3_TILE_BYTES 65536
#define V3_TILE_CHUNKS 2048

#define V3_S0_ROW_SALT 0x6B61F28F3CC48744ULL
#define V3_OFFSET_FIRST_SALT 0x3F1F886D659E316AULL
#define V3_OFFSET_STEP_SALT 0xD4C194F3ADB3B1C7ULL

// --- blake3 (single-chunk path: inputs <= 1024 B, counter 0) ---

__device__ static const unsigned int B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

__device__ static const unsigned char B3_SCHED[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

#define B3_CHUNK_START 1u
#define B3_CHUNK_END 2u
#define B3_ROOT 8u

__device__ __forceinline__ unsigned int b3_rotr(unsigned int x, int n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ void b3_g(unsigned int* v, int a, int b, int c, int d,
                                     unsigned int mx, unsigned int my) {
    v[a] += v[b] + mx; v[d] = b3_rotr(v[d] ^ v[a], 16);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 12);
    v[a] += v[b] + my; v[d] = b3_rotr(v[d] ^ v[a], 8);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 7);
}

__device__ __forceinline__ void b3_compress(unsigned int cv[8], const unsigned int m[16],
                                            unsigned int block_len, unsigned int flags) {
    unsigned int v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    v[8] = B3_IV[0]; v[9] = B3_IV[1]; v[10] = B3_IV[2]; v[11] = B3_IV[3];
    v[12] = 0; v[13] = 0; v[14] = block_len; v[15] = flags;
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        const unsigned char* s = B3_SCHED[r];
        b3_g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        b3_g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        b3_g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        b3_g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        b3_g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        b3_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        b3_g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        b3_g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[i + 8];
}

// blake3 of one 256 B state row (64 packed words): 4 blocks of one chunk.
__device__ __forceinline__ void b3_hash_row(const unsigned int row[64], unsigned int out[8]) {
    unsigned int cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        unsigned int flags = (b == 0 ? B3_CHUNK_START : 0u) | (b == 3 ? (B3_CHUNK_END | B3_ROOT) : 0u);
        b3_compress(cv, row + b * 16, 64, flags);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// blake3 of a 64 B concat (two child hashes): single block.
__device__ __forceinline__ void b3_hash_pair(const unsigned int m[16], unsigned int out[8]) {
    unsigned int cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    b3_compress(cv, m, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// --- v3 walk primitives ---

__device__ __forceinline__ unsigned int v3_rho8(int acc, unsigned int tweak) {
    unsigned int z = (unsigned int)acc ^ tweak;
    z *= 0x9E3779B9u; z ^= z >> 16;
    z *= 0x85EBCA6Bu; z ^= z >> 13;
    return z & 0xffu;
}

// Gather one 64 KB tile (2048 canonical chunks) from the segmented blob into shared.
__device__ __forceinline__ void v3_load_tile(const unsigned long long* bases,
                                             const unsigned long long* prefix, unsigned int T,
                                             unsigned long long tile_index, unsigned int* s_tile32) {
    const unsigned long long chunk0 = tile_index * (unsigned long long)V3_TILE_CHUNKS;
    for (unsigned int c = threadIdx.x; c < V3_TILE_CHUNKS; c += blockDim.x) {
        unsigned long long idx = chunk0 + c;
        unsigned int lo = 0, hi = T;
        while (lo + 1 < hi) {
            unsigned int mid = (lo + hi) >> 1;
            if (prefix[mid] <= idx) lo = mid; else hi = mid;
        }
        const ulonglong2* q = (const ulonglong2*)bases[lo];
        unsigned long long b = (idx - prefix[lo]) * 2ULL;
        ulonglong2* dst = (ulonglong2*)(s_tile32 + c * 8);
        dst[0] = q[b];
        dst[1] = q[b + 1];
    }
}

// The walk body shared by grind and dump. Fits in EXACTLY 64 KB of dynamic shared (the
// tile; reused as the hash-tree scratch after the last step) with zero static shared — the
// Turing per-block ceiling. Each thread redundantly derives the offset chain in registers
// (8 mix64 per step) instead of a shared broadcast. When states_out != nullptr every state
// S_0..S_K is written there (row x of S_t at states_out + t*65536 + x*256) and each step's
// snippet to snippets_out (32 B per step); proof-build consumes the dump. The returned
// fold64(root_K) is valid on thread 0 ONLY.
__device__ __forceinline__ unsigned long long v3_walk_block(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K, unsigned long long seed,
    unsigned int* s_tile32, unsigned char* states_out, unsigned char* snippets_out) {
    const unsigned int x = threadIdx.x;

    // S_0: mix64 keystream per row (spec §4.2).
    unsigned int row4[V3_D4];
    {
        unsigned long long h = mix64(seed ^ (V3_S0_ROW_SALT + (unsigned long long)x));
        #pragma unroll
        for (int k4 = 0; k4 < V3_D4; k4++) { h = mix64(h); row4[k4] = (unsigned int)h; }
    }
    if (states_out) {
        unsigned int* dst = (unsigned int*)(states_out + (unsigned long long)x * V3_D);
        #pragma unroll
        for (int k4 = 0; k4 < V3_D4; k4++) dst[k4] = row4[k4];
    }

    unsigned long long off = mix64(seed ^ V3_OFFSET_FIRST_SALT) % n_tiles;

    for (unsigned int step = 1; step <= K; step++) {
        v3_load_tile(bases, prefix, T, off, s_tile32);
        __syncthreads();

        if (x == 0 && snippets_out) {
            unsigned int* sn = (unsigned int*)(snippets_out + (unsigned long long)(step - 1) * 32);
            #pragma unroll
            for (int w = 0; w < 8; w++) sn[w] = s_tile32[w];
        }
        // Next offset from the CURRENT tile's snippet (spec §4.3), derived by every thread.
        {
            unsigned long long sf = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) sf = mix64(sf ^ (unsigned long long)s_tile32[w]);
            off = mix64(seed ^ (unsigned long long)(step + 1) * V3_OFFSET_STEP_SALT ^ sf) % n_tiles;
        }

        // S_t[x][j] = rho8(dot_i8(row, col_j), tweak(step, x, j)) — fully unrolled dp4a.
        const unsigned int step_tweak = step * 0x9E3779B9u + x * 0xC2B2AE35u;
        unsigned int new4[V3_D4];
        #pragma unroll
        for (int j4 = 0; j4 < V3_D4; j4++) {
            unsigned int packed = 0;
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                const int j = j4 * 4 + jj;
                const uint4* col = (const uint4*)&s_tile32[j * V3_D4];
                int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                #pragma unroll
                for (int k16 = 0; k16 < V3_D4 / 4; k16++) {
                    const uint4 tv = col[k16];
                    a0 = __dp4a((int)row4[k16 * 4 + 0], (int)tv.x, a0);
                    a1 = __dp4a((int)row4[k16 * 4 + 1], (int)tv.y, a1);
                    a2 = __dp4a((int)row4[k16 * 4 + 2], (int)tv.z, a2);
                    a3 = __dp4a((int)row4[k16 * 4 + 3], (int)tv.w, a3);
                }
                packed |= v3_rho8((a0 + a1) + (a2 + a3),
                                  step_tweak + (unsigned int)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        #pragma unroll
        for (int k4 = 0; k4 < V3_D4; k4++) row4[k4] = new4[k4];
        if (states_out) {
            unsigned int* dst =
                (unsigned int*)(states_out + (unsigned long long)step * V3_TILE_BYTES + (unsigned long long)x * V3_D);
            #pragma unroll
            for (int k4 = 0; k4 < V3_D4; k4++) dst[k4] = new4[k4];
        }
        __syncthreads();
    }

    // root_K: blake3 row leaves + complete depth-8 tree (spec §4.5-4.6), scratch = the tile
    // region (the last step's trailing barrier fenced all reads of it).
    b3_hash_row(row4, s_tile32 + x * 8);
    __syncthreads();
    unsigned int* src = s_tile32;
    unsigned int* dst = s_tile32 + V3_D * 8;
    for (unsigned int n = V3_D; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        __syncthreads();
        unsigned int* tmp = src; src = dst; dst = tmp;
    }
    return (unsigned long long)src[0] | ((unsigned long long)src[1] << 32);
}

// v3 grind: one block per nonce. Dynamic shared = 64 KB (the tile).
extern "C" __global__ void pom_mine_v3(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long time_,
    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
    unsigned long long nonce_base, unsigned long long n_nonces,
    unsigned long long* winner) {
    extern __shared__ unsigned int s_tile32[];
    if ((unsigned long long)blockIdx.x >= n_nonces) return;
    const unsigned long long nonce = nonce_base + blockIdx.x;
    const unsigned long long seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);

    const unsigned long long fin = v3_walk_block(bases, prefix, T, n_tiles, K, seed, s_tile32, nullptr, nullptr);

    if (threadIdx.x == 0) {
        unsigned long long pv[4];
        pom_pow_fold(fin, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) {
            atomicMin(winner, nonce);
        }
    }
}

// v3 dump: re-walk ONE (winning) nonce, writing all K+1 states and K snippets for the host
// proof-build. Launch with a single block.
extern "C" __global__ void pom_mine_v3_dump(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long time_, unsigned long long nonce,
    unsigned char* states_out, unsigned char* snippets_out, unsigned long long* final_state_out) {
    extern __shared__ unsigned int s_tile32[];
    const unsigned long long seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);

    const unsigned long long fin = v3_walk_block(bases, prefix, T, n_tiles, K, seed, s_tile32, states_out, snippets_out);

    if (threadIdx.x == 0) *final_state_out = fin;
}

// ============================================================================
// PoM v4 (D=32 re-walk). Byte-exact mirror of consensus/core/src/pom_v4.rs.
// One block of 32 threads per nonce; the moat is the K dependent 1 KB tile reads.
// ============================================================================
#define V4_D 32
#define V4_D4 (V4_D / 4)          // 8 uints per row/column
#define V4_TILE_BYTES 1024
#define V4_TILE_CHUNKS 32
static_assert(V4_D == 32, "PoM v4 requires D=32");

#define V4_S0_ROW_SALT       0x03421325594C3C51ULL
#define V4_OFFSET_FIRST_SALT 0x6D1CCF96AC4D76F9ULL
#define V4_OFFSET_STEP_SALT  0x89050E78D34609EFULL

// One 1 KB tile (32 canonical chunks) into shared, one chunk per lane.
__device__ __forceinline__ void v4_load_tile(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long tile_index, unsigned int* s_tile, unsigned int lane) {
    const unsigned long long idx = tile_index * (unsigned long long)V4_TILE_CHUNKS + lane;
    unsigned int lo = 0, hi = T;
    while (lo + 1 < hi) { unsigned int mid = (lo + hi) >> 1; if (prefix[mid] <= idx) lo = mid; else hi = mid; }
    const ulonglong2* q = (const ulonglong2*)bases[lo];
    const unsigned long long b = (idx - prefix[lo]) * 2ULL;
    ulonglong2* dst = (ulonglong2*)(s_tile + lane * 8);
    dst[0] = q[b];
    dst[1] = q[b + 1];
}

// blake3 of a 32-byte state row (single partial block).
__device__ __forceinline__ void b3_hash_row32(const unsigned int row[8], unsigned int out[8]) {
    unsigned int cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    unsigned int m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = row[i];
    #pragma unroll
    for (int i = 8; i < 16; i++) m[i] = 0u;
    b3_compress(cv, m, 32, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// fold64(v4_state_root(S_K)); valid on thread 0 only.
__device__ __forceinline__ unsigned long long v4_walk_block(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K, unsigned long long seed, unsigned int* s_tile) {
    const unsigned int x = threadIdx.x;   // state row, 0..31

    // S_0: mix64 keystream per row.
    unsigned int row4[V4_D4];
    {
        unsigned long long h = mix64(seed ^ (V4_S0_ROW_SALT + (unsigned long long)x));
        #pragma unroll
        for (int k4 = 0; k4 < V4_D4; k4++) { h = mix64(h); row4[k4] = (unsigned int)h; }
    }

    unsigned long long off = mix64(seed ^ V4_OFFSET_FIRST_SALT) % n_tiles;

    for (unsigned int step = 1; step <= K; step++) {
        v4_load_tile(bases, prefix, T, off, s_tile, x);
        __syncthreads();

        // Next offset from the CURRENT tile's snippet (first 32 bytes = 8 words).
        {
            unsigned long long sf = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) sf = mix64(sf ^ (unsigned long long)s_tile[w]);
            off = mix64(seed ^ (unsigned long long)(step + 1) * V4_OFFSET_STEP_SALT ^ sf) % n_tiles;
        }

        // Transition: this thread's new row = rho(row . tile_col_j) for j in 0..32.
        const unsigned int step_tweak = step * 0x9E3779B9u + x * 0xC2B2AE35u;
        unsigned int new4[V4_D4];
        #pragma unroll
        for (int j4 = 0; j4 < V4_D4; j4++) {
            unsigned int packed = 0;
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                const int j = j4 * 4 + jj;
                const unsigned int* col = &s_tile[j * V4_D4];   // column j = 32 bytes
                int acc = 0;
                #pragma unroll
                for (int k = 0; k < V4_D4; k++) acc = __dp4a((int)row4[k], (int)col[k], acc);
                packed |= (unsigned int)v3_rho8(acc, step_tweak + (unsigned int)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        #pragma unroll
        for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = new4[k4];
        __syncthreads();
    }

    // root_K: 32 blake3 row leaves + complete depth-5 tree (scratch reuses the tile region).
    b3_hash_row32(row4, s_tile + x * 8);
    __syncthreads();
    unsigned int* src = s_tile;
    unsigned int* dst = s_tile + V4_D * 8;
    for (unsigned int n = V4_D; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        __syncthreads();
        unsigned int* tmp = src; src = dst; dst = tmp;
    }
    return (unsigned long long)src[0] | ((unsigned long long)src[1] << 32);
}

// v4 grind: one block of 32 threads per nonce. Dynamic shared = 2 KB (tile + fold scratch).
extern "C" __global__ void pom_mine_v4(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long time_,
    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
    unsigned long long nonce_base, unsigned long long n_nonces,
    unsigned long long* winner) {
    extern __shared__ unsigned int s_shared[];
    if ((unsigned long long)blockIdx.x >= n_nonces) return;
    const unsigned long long nonce = nonce_base + blockIdx.x;
    const unsigned long long seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    const unsigned long long fin = v4_walk_block(bases, prefix, T, n_tiles, K, seed, s_shared);
    if (threadIdx.x == 0) {
        unsigned long long pv[4];
        pom_pow_fold(fin, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) atomicMin(winner, nonce);
    }
}

// Tensor-core v4 solver, two phases. pom_mine_v4_chase resolves the whole tile-offset
// chain (it depends only on tile snippets, never on the walk state); pom_mine_v4_tc then
// walks with a cp.async tile pipeline and 8x mma.sync.m16n8k32.s8 per step.
// Byte-exact vs pom_mine_v4 — any divergence produces blocks the node rejects.
// mma.sync.m16n8k32.s8 needs sm_80+; below that pom_mine_v4_tc is a stub and the host
// must dispatch pom_mine_v4. V4_TC_WARPS must match the host launch config.

__device__ __forceinline__ const ulonglong2* v4_chunk_addr(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long tile_index, unsigned int lane) {
    const unsigned long long idx = tile_index * (unsigned long long)V4_TILE_CHUNKS + lane;
    unsigned int lo = 0, hi = T;
    while (lo + 1 < hi) { unsigned int mid = (lo + hi) >> 1; if (prefix[mid] <= idx) lo = mid; else hi = mid; }
    const ulonglong2* q = (const ulonglong2*)bases[lo];
    return q + (idx - prefix[lo]) * 2ULL;
}

extern "C" __global__ void pom_mine_v4_chase(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long n_tiles, unsigned int K,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long time_, unsigned long long nonce_base, unsigned long long n_nonces,
    unsigned int* offsets /* [n_nonces][K] */) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_nonces) return;
    const unsigned long long nonce = nonce_base + i;
    const unsigned long long seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    unsigned long long off = mix64(seed ^ V4_OFFSET_FIRST_SALT) % n_tiles;
    unsigned int* my = offsets + i * (unsigned long long)K;
    for (unsigned int step = 1; step <= K; step++) {
        my[step - 1] = (unsigned int)off;
        const ulonglong2* q = v4_chunk_addr(bases, prefix, T, off, 0);
        const ulonglong2 c0 = q[0], c1 = q[1];
        unsigned long long sf = 0;
        sf = mix64(sf ^ (unsigned int)(c0.x)); sf = mix64(sf ^ (unsigned int)(c0.x >> 32));
        sf = mix64(sf ^ (unsigned int)(c0.y)); sf = mix64(sf ^ (unsigned int)(c0.y >> 32));
        sf = mix64(sf ^ (unsigned int)(c1.x)); sf = mix64(sf ^ (unsigned int)(c1.x >> 32));
        sf = mix64(sf ^ (unsigned int)(c1.y)); sf = mix64(sf ^ (unsigned int)(c1.y >> 32));
        off = mix64(seed ^ (unsigned long long)(step + 1) * V4_OFFSET_STEP_SALT ^ sf) % n_tiles;
    }
}

#if __CUDA_ARCH__ >= 800

#define V4_TC_WARPS 4    // nonces (warps) per block
#define V4_TC_PIPE  3    // cp.async tile buffers per warp

__device__ __forceinline__ void v4_cp_async16(void* smem_dst, const void* gmem_src) {
    unsigned long long sdst = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "l"(sdst), "l"(gmem_src));
}

__device__ __forceinline__ void v4_tile_cp_async(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned long long tile_index, unsigned int* s_tile, unsigned int lane) {
    const ulonglong2* q = v4_chunk_addr(bases, prefix, T, tile_index, lane);
    ulonglong2* dst = (ulonglong2*)(s_tile + lane * 8);
    v4_cp_async16(dst, q);
    v4_cp_async16(dst + 1, q + 1);
}

// One v4 step on tensor cores. s_state = 256-word shared state (row r = words r*8..r*8+7,
// same packed layout as the host); s_tile = the 1 KB column-major tile.
__device__ __forceinline__ void v4_imma_step(
    unsigned int* s_state, const unsigned int* s_tile, unsigned int step, unsigned int x) {
    const unsigned int gid = x >> 2, tig = x & 3u;
    unsigned int a[2][4];
    #pragma unroll
    for (unsigned int g = 0; g < 2; g++) {
        const unsigned int r0 = g * 16u + gid, r1 = r0 + 8u;
        a[g][0] = s_state[r0 * 8u + tig];
        a[g][1] = s_state[r1 * 8u + tig];
        a[g][2] = s_state[r0 * 8u + tig + 4u];
        a[g][3] = s_state[r1 * 8u + tig + 4u];
    }
    __syncwarp();   // every lane read the old state before anyone overwrites it
    unsigned short* s_state16 = (unsigned short*)s_state;
    const unsigned int step_base = step * 0x9E3779B9u;
    #pragma unroll
    for (unsigned int cg = 0; cg < 4; cg++) {
        const unsigned int cb = (cg * 8u + gid) * 8u;
        const unsigned int b0 = s_tile[cb + tig];
        const unsigned int b1 = s_tile[cb + tig + 4u];
        #pragma unroll
        for (unsigned int g = 0; g < 2; g++) {
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a[g][0]), "r"(a[g][1]), "r"(a[g][2]), "r"(a[g][3]), "r"(b0), "r"(b1));
            const unsigned int r0 = g * 16u + gid, r1 = r0 + 8u;
            const unsigned int j0 = cg * 8u + tig * 2u;
            const unsigned int tw0 = step_base + r0 * 0xC2B2AE35u + j0 * 0x85EBCA6Bu;
            const unsigned int tw1 = step_base + r1 * 0xC2B2AE35u + j0 * 0x85EBCA6Bu;
            s_state16[r0 * 16u + (j0 >> 1)] =
                (unsigned short)(v3_rho8(c0, tw0) | (v3_rho8(c1, tw0 + 0x85EBCA6Bu) << 8));
            s_state16[r1 * 16u + (j0 >> 1)] =
                (unsigned short)(v3_rho8(c2, tw1) | (v3_rho8(c3, tw1 + 0x85EBCA6Bu) << 8));
        }
    }
    __syncwarp();
}

extern "C" __global__ void pom_mine_v4_tc(
    const unsigned long long* bases, const unsigned long long* prefix, unsigned int T,
    unsigned int K,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long time_,
    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
    unsigned long long nonce_base, unsigned long long n_nonces,
    const unsigned int* offsets, unsigned long long* winner) {
    extern __shared__ unsigned int s_shared[];
    const unsigned int w = threadIdx.x >> 5;
    const unsigned long long i = (unsigned long long)blockIdx.x * V4_TC_WARPS + w;
    if (i >= n_nonces) return;
    const unsigned int x = threadIdx.x & 31u;
    unsigned int* s_buf = s_shared + w * (256u * (V4_TC_PIPE + 1));
    unsigned int* s_state = s_buf + 256u * V4_TC_PIPE;
    const unsigned long long nonce = nonce_base + i;
    const unsigned long long seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    const unsigned int* my = offsets + i * (unsigned long long)K;

    // S_0 straight into the shared state (spec keystream, same packing as the host).
    { unsigned long long h = mix64(seed ^ (V4_S0_ROW_SALT + (unsigned long long)x));
      #pragma unroll
      for (int k4 = 0; k4 < V4_D4; k4++) { h = mix64(h); s_state[x * 8u + k4] = (unsigned int)h; } }
    __syncwarp();

    #pragma unroll
    for (unsigned int p = 0; p < V4_TC_PIPE - 1; p++) {
        if (p < K) { v4_tile_cp_async(bases, prefix, T, my[p], s_buf + p * 256u, x); }
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned int step = 1; step <= K; step++) {
        unsigned int* cur = s_buf + ((step - 1u) % V4_TC_PIPE) * 256u;
        asm volatile("cp.async.wait_group %0;" :: "n"(V4_TC_PIPE - 2));
        __syncwarp();
        if (step + V4_TC_PIPE - 2 < K) {
            v4_tile_cp_async(bases, prefix, T, my[step + V4_TC_PIPE - 2],
                             s_buf + ((step + V4_TC_PIPE - 2u) % V4_TC_PIPE) * 256u, x);
        }
        asm volatile("cp.async.commit_group;");
        v4_imma_step(s_state, cur, step, x);
    }
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    // root_K + fold, verbatim reference tail (row regs restored from the shared state).
    unsigned int row4[V4_D4];
    #pragma unroll
    for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = s_state[x * 8u + k4];
    unsigned int* s_tile = s_buf;
    b3_hash_row32(row4, s_tile + x * 8);
    __syncwarp();
    unsigned int* src = s_tile; unsigned int* dst = s_tile + V4_D * 8;
    for (unsigned int n = V4_D; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        __syncwarp();
        unsigned int* tmp = src; src = dst; dst = tmp;
    }
    if (x == 0) {
        const unsigned long long fin = (unsigned long long)src[0] | ((unsigned long long)src[1] << 32);
        unsigned long long pv[4];
        pom_pow_fold(fin, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) atomicMin(winner, nonce);
    }
}

#else   // __CUDA_ARCH__ < 800: no int8 mma — stub so the module still loads; the host
        // checks compute capability and dispatches pom_mine_v4 instead.
extern "C" __global__ void pom_mine_v4_tc(
    const unsigned long long*, const unsigned long long*, unsigned int, unsigned int,
    unsigned long long, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long,
    unsigned long long, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, const unsigned int*, unsigned long long*) {}
#endif
