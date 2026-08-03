#include<stdint.h>
#include <assert.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"



typedef uint8_t Hash[32];

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X,Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])
#define LE_U256(X,Y) (!LT_U256(Y,X))

__constant__ uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;
__constant__ static const uint8_t powP[Plen] = { 0x3d, 0xd8, 0xf6, 0xa1, 0x0d, 0xff, 0x3c, 0x11, 0x3c, 0x7e, 0x02, 0xb7, 0x55, 0x88, 0xbf, 0x29, 0xd2, 0x44, 0xfb, 0x0e, 0x72, 0x2e, 0x5f, 0x1e, 0xa0, 0x69, 0x98, 0xf5, 0xa3, 0xa4, 0xa5, 0x1b, 0x65, 0x2d, 0x5e, 0x87, 0xca, 0xaf, 0x2f, 0x7b, 0x46, 0xe2, 0xdc, 0x29, 0xd6, 0x61, 0xef, 0x4a, 0x10, 0x5b, 0x41, 0xad, 0x1e, 0x98, 0x3a, 0x18, 0x9c, 0xc2, 0x9b, 0x78, 0x0c, 0xf6, 0x6b, 0x77, 0x40, 0x31, 0x66, 0x88, 0x33, 0xf1, 0xeb, 0xf8, 0xf0, 0x5f, 0x28, 0x43, 0x3c, 0x1c, 0x65, 0x2e, 0x0a, 0x4a, 0xf1, 0x40, 0x05, 0x07, 0x96, 0x0f, 0x52, 0x91, 0x29, 0x5b, 0x87, 0x67, 0xe3, 0x44, 0x15, 0x37, 0xb1, 0x25, 0xa4, 0xf1, 0x70, 0xec, 0x89, 0xda, 0xe9, 0x82, 0x8f, 0x5d, 0xc8, 0xe6, 0x23, 0xb2, 0xb4, 0x85, 0x1f, 0x60, 0x1a, 0xb2, 0x46, 0x6a, 0xa3, 0x64, 0x90, 0x54, 0x85, 0x34, 0x1a, 0x85, 0x2f, 0x7a, 0x1c, 0xdd, 0x06, 0x0f, 0x42, 0xb1, 0x3b, 0x56, 0x1d, 0x02, 0xa2, 0xc1, 0xe4, 0x68, 0x16, 0x45, 0xe4, 0xe5, 0x1d, 0xba, 0x8d, 0x5f, 0x09, 0x05, 0x41, 0x57, 0x02, 0xd1, 0x4a, 0xcf, 0xce, 0x9b, 0x84, 0x4e, 0xca, 0x89, 0xdb, 0x2e, 0x74, 0xa8, 0x27, 0x94, 0xb0, 0x48, 0x72, 0x52, 0x8b, 0xe7, 0x9c, 0xce, 0xfc, 0xb1, 0xbc, 0xa5, 0xaf, 0x82, 0xcf, 0x29, 0x11, 0x5d, 0x83, 0x43, 0x82, 0x6f, 0x78, 0x7c, 0xb9, 0x02 };
__constant__ static const uint8_t heavyP[Plen] = { 0x09, 0x85, 0x24, 0xb2, 0x52, 0x4c, 0xd7, 0x3a, 0x16, 0x42, 0x9f, 0x2f, 0x0e, 0x9b, 0x62, 0x79, 0xee, 0xf8, 0xc7, 0x16, 0x48, 0xff, 0x14, 0x7a, 0x98, 0x64, 0x05, 0x80, 0x4c, 0x5f, 0xa7, 0x11, 0xda, 0xce, 0xee, 0x44, 0xdf, 0xe0, 0x20, 0xe7, 0x69, 0x40, 0xf3, 0x14, 0x2e, 0xd8, 0xc7, 0x72, 0xba, 0x35, 0x89, 0x93, 0x2a, 0xff, 0x00, 0xc1, 0x62, 0xc4, 0x0f, 0x25, 0x40, 0x90, 0x21, 0x5e, 0x48, 0x6a, 0xcf, 0x0d, 0xa6, 0xf9, 0x39, 0x80, 0x0c, 0x3d, 0x2a, 0x79, 0x9f, 0xaa, 0xbc, 0xa0, 0x26, 0xa2, 0xa9, 0xd0, 0x5d, 0xc0, 0x31, 0xf4, 0x3f, 0x8c, 0xc1, 0x54, 0xc3, 0x4c, 0x1f, 0xd3, 0x3d, 0xcc, 0x69, 0xa7, 0x01, 0x7d, 0x6b, 0x6c, 0xe4, 0x93, 0x24, 0x56, 0xd3, 0x5b, 0xc6, 0x2e, 0x44, 0xb0, 0xcd, 0x99, 0x3a, 0x4b, 0xf7, 0x4e, 0xb0, 0xf2, 0x34, 0x54, 0x83, 0x86, 0x4c, 0x77, 0x16, 0x94, 0xbc, 0x36, 0xb0, 0x61, 0xe9, 0x07, 0x07, 0xcc, 0x65, 0x77, 0xb1, 0x1d, 0x8f, 0x7e, 0x39, 0x6d, 0xc4, 0xba, 0x80, 0xdb, 0x8f, 0xea, 0x58, 0xca, 0x34, 0x7b, 0xd3, 0xf2, 0x92, 0xb9, 0x57, 0xb9, 0x81, 0x84, 0x04, 0xc5, 0x76, 0xc7, 0x2e, 0xc2, 0x12, 0x51, 0x67, 0x9f, 0xc3, 0x47, 0x0a, 0x0c, 0x29, 0xb5, 0x9d, 0x39, 0xbb, 0x92, 0x15, 0xc6, 0x9f, 0x2f, 0x31, 0xe0, 0x9a, 0x54, 0x35, 0xda, 0xb9, 0x10, 0x7d, 0x32, 0x19, 0x16 };

/* ======================================================================
 * KERYX WAVE-MIX — ARX post-processing after matrix multiply
 *
 * Implements `fn wave_mix()` from consensus/pow/src/matrix.rs exactly.
 * Must be called on the 32-byte matrix-product BEFORE the final
 * KeryxHash (heavyP) Keccak absorption.
 *
 * Round constants (must NOT be changed — changing them = hard fork):
 *   [0] = 0x9e3779b97f4a7c15  (frac. bits of φ)
 *   [1] = 0x6c62272e07bb0142  (Keryx network discriminator)
 *   [2] = 0xb5ad4eceda1ce2a9  (frac. bits of √3)
 *   [3] = 0x243f6a8885a308d3  (frac. bits of π)
 * Rotation schedule: [17, 47, 31, 13]
 * ====================================================================== */
__constant__ static const uint64_t WAVE_MIX_KEYS[4] = {
    0x9e3779b97f4a7c15ULL,
    0x6c62272e07bb0142ULL,
    0xb5ad4eceda1ce2a9ULL,
    0x243f6a8885a308d3ULL,
};

/* Apply 4 rounds of ARX to the 32-byte hash stored in h->number[0..3] (LE).
 * The uint256_t union gives us the 4 × uint64_t words directly. */
__device__ __inline__ void wave_mix(uint256_t *h) {
    uint64_t w0 = h->number[0];
    uint64_t w1 = h->number[1];
    uint64_t w2 = h->number[2];
    uint64_t w3 = h->number[3];

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        /* Step A — vertical pairs: independent, compiler can schedule in parallel */
        w0 += w1; w0 = (w0 << 17) | (w0 >> 47); w0 ^= WAVE_MIX_KEYS[r & 3];
        w2 += w3; w2 = (w2 << 47) | (w2 >> 17); w2 ^= WAVE_MIX_KEYS[(r + 2) & 3];
        /* Step B — diagonal pairs: cross-pollinate all 256 bits */
        w1 += w2; w1 = (w1 << 31) | (w1 >> 33); w1 ^= WAVE_MIX_KEYS[(r + 1) & 3];
        w3 += w0; w3 = (w3 << 13) | (w3 >> 51); w3 ^= WAVE_MIX_KEYS[(r + 3) & 3];
    }

    h->number[0] = w0;
    h->number[1] = w1;
    h->number[2] = w2;
    h->number[3] = w3;
}

__device__ __inline__ void amul4bit(uint32_t packed_vec1[32], uint32_t packed_vec2[32], uint32_t *ret) {
    // We assume each 32 bits have four values: A0 B0 C0 D0
    unsigned int res = 0;
    #if __CUDA_ARCH__ < 610
    char4 *a4 = (char4*)packed_vec1;
    char4 *b4 = (char4*)packed_vec2;
    #endif
    #pragma unroll
    for (int i=0; i<QUARTER_MATRIX_SIZE; i++) {
        #if __CUDA_ARCH__ >= 610
        res = __dp4a(packed_vec1[i], packed_vec2[i], res);
        #else
        res += a4[i].x*b4[i].x;
        res += a4[i].y*b4[i].y;
        res += a4[i].z*b4[i].z;
        res += a4[i].w*b4[i].w;
        #endif
    }

    *ret = res;
}


/*
 * Arch-specific launch bounds. On sm_80 (GA100 — A100 / CMP 170HX) the
 * unrolled kernel naturally lands at 72 registers, which is just over
 * the 65536/(2*512)=64 threshold, so it runs at only 1 block/SM (~25%
 * occupancy) — the 170HX then idles at ~99W/250W, latency-bound. Forcing
 * `__launch_bounds__(512, 2)` caps it at 64 regs (a trivial 12-byte
 * spill) and unlocks 2 blocks/SM = 50% occupancy.
 *
 * Gated on __CUDA_ARCH__ so sm_120 (RTX 5090) is untouched — it already
 * sits at 64 regs / 2 blocks/SM and is power-bound at 3.28 GH/s; pinning
 * a block-size hint there could only constrain the driver's launch
 * config, so we leave it alone.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
#define HEAVY_HASH_BOUNDS __launch_bounds__(512, 2)
#else
#define HEAVY_HASH_BOUNDS
#endif

extern "C" {


    /*
     * *** BIG WIN (2026-06): unroll the Keccak round loop ***
     * `keccakf()` in keccak-tiny.c had a ROLLED `for (i=0;i<24;i++)`
     * round loop. Rolled, the 25-lane state stayed an addressable
     * local array (the rho/pi step indexes a[pi[x]]), which pinned the
     * kernel at 229 regs offline / 126 regs after driver JIT → only
     * 1 block/SM, 33% occupancy, and no cross-round ILP (fatal on a
     * latency-bound kernel at low occupancy). Adding `#pragma unroll`
     * to that loop lets the π-permutation become pure register
     * renaming: 229 → 64 regs, 0 spill, 2 blocks/SM, and the runtime
     * workload 4×'d (44M → 178M nonces). Measured 2.57 → 3.28 GH/s
     * (+28%) on the 5090 at stock 575W. Shares accepted, 0 rejects.
     * The card is now power-bound at the TDP cap rather than
     * compute-bound; further kernel wins must cut energy-per-hash.
     *
     * --- earlier (now superseded) launch-config sweep ---
     * Tried `__launch_bounds__(512, 2)` (force 2 blocks/SM by capping
     * regs/thread at ~64). Measured 2.78 → 2.01 GH/s — the compiler
     * spilled the Keccak f-1600 state to local memory and the latency
     * blew up. (This was BEFORE the unroll: the rolled loop made 64
     * regs impossible without spilling. Post-unroll, 64 regs fits
     * naturally with 0 spill — the lever was the loop, not the bound.)
     *
     * Profiling baseline (nsight 2026.1.1, ncu --section SpeedOfLight,
     * sm_120 RTX 5090, --light tier, 89 M-nonce launch):
     *   Compute SM throughput    91.50 %        ← compute-bound
     *   Memory throughput         1.87 %        ← entirely cached
     *   Mem pipes busy            4.90 %
     *   Registers per thread      126           ← hard limiter
     *   Block size                512
     *   Block limit (registers)   1
     *   Theoretical occupancy     33.33 %
     *   Achieved occupancy        32.44 %       ← ≈ theoretical
     *   L1 hit rate              100 %          ← matrix lives in L1
     *   L2 hit rate               91 %
     *
     * `__launch_bounds__(1024, 1)` (committed but later measured at
     * 2.55 GH/s — also a regression). The cust runtime's
     * cuOccupancyMaxPotentialBlockSize picks block_size=512 to
     * maximize occupancy at the 126-reg compute. Forcing 1024 either
     * blew up regs (compiler had to budget across 1024 threads) or
     * inflated the in-flight warp count past the SM scheduler's
     * sweet spot — either way, slower.
     *
     * Net of the launch-config sweep: upstream defaults win. The lever
     * for speed is on the kernel body, not the launch dispatcher.
     */
    __global__ void HEAVY_HASH_BOUNDS heavy_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t *final_nonce, uint32_t *winner_found) {
        // assuming header_len is 72
        int nonceId = threadIdx.x + blockIdx.x*blockDim.x;
        if (nonceId < nonces_len) {
            uint64_t nonce;
            switch (random_type) {
                case RANDOM_LEAN:
                    nonce = ((uint64_t *)states)[0] ^ nonceId;
                    break;
                case RANDOM_XOSHIRO:
                default:
                    nonce = xoshiro256_next(((ulonglong4 *)states) + nonceId);
                    break;
            }
            nonce = (nonce & nonce_mask) | nonce_fixed;
            uint256_t hash_;
            /*
             * First Keccak call (powP / pre-pow hash).
             *
             * Upstream materialised an 80-byte `uint8_t input[80]` in
             * local memory and `memcpy`'d 72 bytes of header + 8 bytes
             * of nonce into it before handing the pointer to `hash()`.
             * That 80-byte stack buffer is a non-trivial register / spill
             * cost (we're already at 126 regs/thread) and the memcpy
             * adds 9 loads + 9 stores that the compiler can't always
             * elide.
             *
             * The Keccak state itself only ever reads:
             *   • bytes 0..71  from `hash_header`  (constant memory)
             *   • bytes 72..79 from the 64-bit nonce
             *   • bytes 80..   are zero (sponge padding)
             * which lets us absorb the rate block directly with 9 XOR-
             * with-constant + 1 XOR-with-nonce ops, plus a straight copy
             * of the 15-u64 capacity tail from `powP`. No local buffer,
             * no memcpy.
             */
            {
                uint64_t a[25];
                const uint64_t *hp = (const uint64_t *) hash_header;
                const uint64_t *p  = (const uint64_t *) powP;
                #pragma unroll
                for (int i = 0; i < 9; i++) a[i] = p[i] ^ hp[i];
                a[9] = p[9] ^ nonce;
                #pragma unroll
                for (int i = 10; i < 25; i++) a[i] = p[i];
                P(a);
                #pragma unroll
                for (int i = 0; i < 4; i++) ((uint64_t *) hash_.hash)[i] = a[i];
            }

            //assert((rowId != 0) || (hashId != 0) );
            uchar4 packed_hash[QUARTER_MATRIX_SIZE] = {0};
            #pragma unroll
            for (int i=0; i<QUARTER_MATRIX_SIZE; i++) {
                packed_hash[i] = make_uchar4(
                    (hash_.hash[2*i] & 0xF0) >> 4 ,
                    (hash_.hash[2*i] & 0x0F),
                    (hash_.hash[2*i+1] & 0xF0) >> 4,
                    (hash_.hash[2*i+1] & 0x0F)
                );
            }
            uint32_t product1, product2;
            #pragma unroll
            for (int rowId=0; rowId<HALF_MATRIX_SIZE; rowId++){

                amul4bit((uint32_t *)(matrix[(2*rowId)]), (uint32_t *)(packed_hash), &product1);
                amul4bit((uint32_t *)(matrix[(2*rowId+1)]), (uint32_t *)(packed_hash), &product2);
                product1 >>= 6;
                product1 &= 0xF0;
                product2 >>= 10;
                /*
                 * On sm_120 the compiler-driven C path actually beats
                 * an explicit lop3.b32 asm here — tried widening the
                 * `__CUDA_ARCH__ < 500 || > 700` upstream gate to
                 * sm_50+ and the rebuild measured 2.78 → 2.55 GH/s.
                 * Hypothesis: the C variant works on u8 hash_.hash[rowId]
                 * and nvcc lowers it to a byte-permute fused with the
                 * `xor` step; lifting to u32 + lop3 + truncate forced an
                 * extra promotion path. Kept upstream behaviour for now.
                 */
                #if __CUDA_ARCH__ < 500 || __CUDA_ARCH__ > 700
                hash_.hash[rowId] = hash_.hash[rowId] ^ ((uint8_t)(product1) | (uint8_t)(product2));
                #else
                uint32_t lop_temp = hash_.hash[rowId];
                asm("lop3.b32" " %0, %1, %2, %3, 0x56;": "=r" (lop_temp): "r" (product1), "r" (product2), "r" (lop_temp));
                hash_.hash[rowId] = lop_temp;
                #endif
            }
            /* Keryx wave-mix: ARX post-processing step.
             * Nodes that skip this compute a different hash and can never
             * satisfy the target — protocol compliance is enforced implicitly. */
            wave_mix(&hash_);

            /*
             * Second Keccak call (heavyP / final KeryxHash).
             *
             * Upstream zeroed the 80-byte `input[]` and memcpy'd 32 bytes
             * of `hash_` over the front, then called `hash(heavyP, ...)`.
             * Inside `hash()` the first 80 input bytes get XOR'd with
             * the first 80 of `heavyP`. Bytes 32..79 are guaranteed
             * zero, so 6 of the 10 input XORs are `x ^ 0 = x` — pure
             * waste. Inline + skip:
             *   • XOR 4 u64 of the matrix-product hash with heavyP[0..3]
             *   • COPY 6 u64 of heavyP[4..9] (no XOR — input is zero)
             *   • COPY 15 u64 of heavyP[10..24] (capacity tail)
             * Saves: 80-byte memset, 32-byte memcpy, 6 XORs, and the
             * call frame for `hash()`.
             */
            {
                uint64_t a[25];
                const uint64_t *hh = (const uint64_t *) hash_.hash;
                const uint64_t *p  = (const uint64_t *) heavyP;
                #pragma unroll
                for (int i = 0; i < 4; i++) a[i] = hh[i] ^ p[i];
                #pragma unroll
                for (int i = 4; i < 25; i++) a[i] = p[i];
                P(a);
                #pragma unroll
                for (int i = 0; i < 4; i++) ((uint64_t *) hash_.hash)[i] = a[i];
            }
            if (LE_U256(hash_, target) && atomicCAS(winner_found, 0, 1) == 0){
                *final_nonce = nonce;
            }
        }
    }

}
