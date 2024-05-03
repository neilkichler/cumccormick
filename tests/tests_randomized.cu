#include "tests_common.h"
#include "tests_randomized.h"

#include <cumccormick/arithmetic/basic.cuh>

#include <cstdio>
#include <curand_kernel.h>

#define CURAND_CHECK(x)                               \
    do {                                              \
        if ((x) != CURAND_STATUS_SUCCESS) {           \
            printf("CuRand error in %s at %s:%d\n",   \
                   __FUNCTION__, __FILE__, __LINE__); \
            abort();                                  \
        }                                             \
    } while (0)

using u64       = unsigned long long;
using rng_state = curandStateScrambledSobol64_t;

constexpr int BLOCK_COUNT       = 64;
constexpr int THREADS_PER_BLOCK = 64;
constexpr int TOTAL_THREADS     = BLOCK_COUNT * THREADS_PER_BLOCK;
constexpr int VECTOR_SIZE       = 64;

constexpr int n_dims = 4;

__global__ void setup_randomized_kernel(u64 *sobol_directions,
                                        u64 *sobol_scrambled_constants,
                                        rng_state *state)
{
    int idx = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int dim = n_dims * idx;

    for (int i = 0; i < n_dims; i++) {
        curand_init(sobol_directions + VECTOR_SIZE * (dim + 0),
                    sobol_scrambled_constants[dim + i],
                    0, // offset
                    &state[dim + i]);
    }
}

__global__ void generate_kernel(rng_state *state, int n)
{
    int idx = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int dim = n_dims * idx;

    mc<double> x;

    for (int i = 0; i < n; i++) {
        x.cv     = curand_uniform_double(&state[dim + 0]);
        x.cc     = curand_uniform_double(&state[dim + 1]);
        x.box.lb = curand_uniform_double(&state[dim + 2]);
        x.box.ub = curand_uniform_double(&state[dim + 3]);

        if (x.cv <= x.cc && x.box.lb <= x.box.ub) {
            printf("[%d][%d] [%g, (%g, %g), %g]\n", dim, i, x.box.lb, x.cv, x.cc, x.box.ub);
        }
    }
}

void tests_randomized(cudaStream_t stream, cudaEvent_t event)
{

    curandStateScrambledSobol64_t *states;
    curandDirectionVectors64_t *h_directions;

    u64 *h_scrambled_constants;
    u64 *d_directions;
    u64 *d_scrambled_constants;

    CUDA_CHECK(cudaMalloc(&states, TOTAL_THREADS * n_dims * sizeof(*states)));

    CURAND_CHECK(curandGetDirectionVectors64(&h_directions, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
    CURAND_CHECK(curandGetScrambleConstants64(&h_scrambled_constants));

    // we need 4 values x.lb x.cv x.cc x.ub with restriction x.lb <=  x.ub, x.cv <= x.cc
    CUDA_CHECK(cudaMalloc(&d_directions, n_dims * TOTAL_THREADS * VECTOR_SIZE * sizeof(*d_directions)));
    CUDA_CHECK(cudaMemcpy(d_directions, h_directions, n_dims * TOTAL_THREADS * VECTOR_SIZE * sizeof(*d_directions), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_scrambled_constants, n_dims * TOTAL_THREADS * sizeof(*d_scrambled_constants)));
    CUDA_CHECK(cudaMemcpy(d_scrambled_constants, h_scrambled_constants, n_dims * TOTAL_THREADS * sizeof(*d_scrambled_constants), cudaMemcpyHostToDevice));

    setup_randomized_kernel<<<BLOCK_COUNT, THREADS_PER_BLOCK, 0, stream>>>(d_directions, d_scrambled_constants, states);

    constexpr int n_iterations = 1;
    constexpr int n_samples    = 1024;

    for (int i = 0; i < n_iterations; i++) {
        generate_kernel<<<BLOCK_COUNT, THREADS_PER_BLOCK, 0, stream>>>(states, n_samples);
    }

    CUDA_CHECK(cudaFree(states));
    CUDA_CHECK(cudaFree(d_directions));
    CUDA_CHECK(cudaFree(d_scrambled_constants));
}
