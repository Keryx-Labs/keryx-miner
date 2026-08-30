// Keccak-f[1600] permutation (device). MUST match the host `keccak::f1600`.
#pragma once

__device__ __forceinline__ unsigned long long keccak_rotl64(unsigned long long x, unsigned int n) {
    return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ void keccak_f1600(unsigned long long st[25]) {
    const unsigned long long RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
    const unsigned int RHO[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const unsigned int PI[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    unsigned long long bc[5];
    #pragma unroll 1
    for (int round = 0; round < 24; round++) {
        #pragma unroll
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            const unsigned long long t = bc[(i + 4) % 5] ^ keccak_rotl64(bc[(i + 1) % 5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        unsigned long long t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            const unsigned int j = PI[i];
            const unsigned long long tmp = st[j];
            st[j] = keccak_rotl64(t, RHO[i]);
            t = tmp;
        }
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            #pragma unroll
            for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= RC[round];
    }
}

// H10 seed: lane 0 of the sponge after absorbing the nonce into lane 9 of `state25`
// (pre_pow_hash and timestamp already absorbed host-side).
__device__ __forceinline__ unsigned long long pom_seed_h10(unsigned long long nonce, const unsigned long long* state25) {
    unsigned long long st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = state25[i];
    st[9] ^= nonce;
    keccak_f1600(st);
    return st[0];
}
