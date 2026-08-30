// Prints the H10 seed for fixed (pre_pow_hash, timestamp, nonce) vectors, computed on the GPU
// with the kernel's keccak. Compare against the host `pom::seed_h10_tests`.
//   nvcc -O2 -arch=sm_86 cuda/tests/seed_h10_check.cu -o /tmp/seed_h10_check && /tmp/seed_h10_check
#include <cstdio>
#include <cstring>
#include "../keccak_f1600.cuh"

static const unsigned long long INITIAL[25] = {
    1242148031264380989ULL, 3008272977830772284ULL, 2188519011337848018ULL, 1992179434288343456ULL, 8876506674959887717ULL,
    5399642050693751366ULL, 1745875063082670864ULL, 8605242046444978844ULL, 17936695144567157056ULL, 3343109343542796272ULL,
    1123092876221303306ULL, 4963925045340115282ULL, 17037383077651887893ULL, 16629644495023626889ULL, 12833675776649114147ULL,
    3784524041015224902ULL, 1082795874807940378ULL, 13952716920571277634ULL, 13411128033953605860ULL, 15060696040649351053ULL,
    9928834659948351306ULL, 5237849264682708699ULL, 12825353012139217522ULL, 6706187291358897596ULL, 196324915476054915ULL};

__global__ void seeds(const unsigned long long* states, const unsigned long long* nonces, unsigned long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = pom_seed_h10(nonces[i], states + 25 * i);
}

int main() {
    const int N = 3;
    unsigned char pph[N][32]; unsigned long long ts[N] = {0, 1788000000000ULL, 1788000000000ULL};
    unsigned long long nonce[N] = {0, 0x0123456789abcdefULL, 0xffffffffffffffffULL};
    memset(pph[0], 0x00, 32); memset(pph[1], 0x5a, 32); memset(pph[2], 0xa5, 32);
    unsigned long long states[N * 25];
    for (int i = 0; i < N; i++) {
        memcpy(states + 25 * i, INITIAL, sizeof(INITIAL));
        for (int w = 0; w < 4; w++) {
            unsigned long long v = 0;
            for (int b = 0; b < 8; b++) v |= (unsigned long long)pph[i][w * 8 + b] << (8 * b);
            states[25 * i + w] ^= v;
        }
        states[25 * i + 4] ^= ts[i];
    }
    unsigned long long *d_states, *d_nonces, *d_out, out[N];
    cudaMalloc(&d_states, sizeof(states)); cudaMalloc(&d_nonces, sizeof(nonce)); cudaMalloc(&d_out, sizeof(out));
    cudaMemcpy(d_states, states, sizeof(states), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonces, nonce, sizeof(nonce), cudaMemcpyHostToDevice);
    seeds<<<1, 32>>>(d_states, d_nonces, d_out, N);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("gpu seed_h10[%d] = 0x%016llx\n", i, out[i]);
    return 0;
}
