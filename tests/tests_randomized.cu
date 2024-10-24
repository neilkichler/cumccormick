#include <cumccormick/cumccormick.cuh>

#include "tests_common.h"
#include "tests_randomized.h"

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

template<typename T>
using mc = cu::mccormick<T>;

using u64       = unsigned long long;
using rng_state = curandStateScrambledSobol64_t;

constexpr int BLOCK_COUNT       = 256;
constexpr int THREADS_PER_BLOCK = 32;
constexpr int TOTAL_THREADS     = BLOCK_COUNT * THREADS_PER_BLOCK;
constexpr int VECTOR_SIZE       = 64;

// we need 5 values: x.box.lb, x.cv, x.cc, x.box.ub, and x_interior
constexpr int n_dims = 5;

__global__ void setup_randomized_kernel(u64 *sobol_directions,
                                        u64 *sobol_scrambled_constants,
                                        rng_state *state)
{
    int idx = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int dim = n_dims * idx;

    for (int i = 0; i < n_dims; i++) {
        curand_init(sobol_directions + VECTOR_SIZE * (dim + i),
                    sobol_scrambled_constants[dim + i],
                    0, // offset
                    &state[dim + i]);
    }
}

template<typename T>
__device__ void check_univariate(mc<T> x, T *interior_samples, int n)
{
    auto contains = [x](mc<T> y_mc, T y, T x_sample) {
        if (!(y_mc.cv <= y && y <= y_mc.cc)) {
            printf("[E] Invalid bounds: y.cv = %.15g, y = %.15g, y.cc = %.15g\n", y_mc.cv, y, y_mc.cc);
            printf("[E] Invalid bounds: y.cv = %a, y = %a, y.cc = %a\n", y_mc.cv, y, y_mc.cc);
            printf("[E] For input: x_sample = %.15g, x = [%.15g, (%.15g, %.15g), %.15g]\n", x_sample, x.box.lb, x.cv, x.cc, x.box.ub);
            printf("[E] For input: x_sample = %a, x = [%a, (%a, %a), %a]\n", x_sample, x.box.lb, x.cv, x.cc, x.box.ub);
        }
        return y_mc.cv <= y && y <= y_mc.cc;
    };

    for (int i = 0; i < n; i++) {
        T x_sample = interior_samples[i];
        assert(contains(pow(x, 1), x_sample, x_sample));
        assert(contains(pow(x, 2), x_sample * x_sample, x_sample));
        assert(contains(pow(x, 3), pow(x_sample, 3), x_sample));
        assert(contains(pow(x, 4), pow(x_sample, 4), x_sample));
        assert(contains(pow(x, 5), pow(x_sample, 5), x_sample));
        assert(contains(abs(x), abs(x_sample), x_sample));
        assert(contains(exp(x), exp(x_sample), x_sample));
        assert(contains(fabs(x), fabs(x_sample), x_sample));
        assert(contains(neg(x), -x_sample, x_sample));
        assert(contains(sqr(x), x_sample * x_sample, x_sample));
        assert(contains(cos(x), cos(x_sample), x_sample));
        // assert(contains(sin(x), sin(x_sample), x_sample));
        assert(contains(tanh(x), tanh(x_sample), x_sample));
        if (inf(x) >= 0) {
            assert(contains(log(x), log(x_sample), x_sample));
            assert(contains(recip(x), __drcp_rn(x_sample), x_sample));
            assert(contains(sqrt(x), sqrt(x_sample), x_sample));
        }
    }
}

template<typename T>
__device__ void check_bivariate(mc<T> x, mc<T> y, T *interior_x_samples, T *interior_y_samples, int n)
{
    auto contains = [](mc<T> x, T y) {
        return x.cv <= y && y <= x.cc;
    };

    for (int i = 0; i < n; i++) {
        T x_sample = interior_x_samples[i];
        T y_sample = interior_y_samples[i];

        assert(contains(x + y, x_sample + y_sample));
        assert(contains(x - y, x_sample - y_sample));
        assert(contains(x * y, x_sample * y_sample));
        assert(contains(max(x, y), max(x_sample, y_sample)));
        assert(contains(min(x, y), min(x_sample, y_sample)));
    }
}

__global__ void generate_and_check(rng_state *state, int n, u64 offset)
{
    // TODO: relax sampling to allow intervals that are tighter than the McCormick bound.

    auto random_mccormick = [](rng_state *state, int offset) {
        mc<double> x;
        x.cv     = curand_uniform_double(&state[offset + 0]);
        x.cc     = x.cv + (1.0 - x.cv) * curand_uniform_double(&state[offset + 1]);
        x.box.lb = x.cv * curand_uniform_double(&state[offset + 2]);
        x.box.ub = x.cc + (1.0 - x.cc) * curand_uniform_double(&state[offset + 3]);
        return x;
    };

    auto random_interior_samples = [](rng_state *state, mc<double> x, double *samples, int n, int key) {
        for (int j = 0; j < n; j++) {
            samples[j] = x.cv + (x.cc - x.cv) * curand_uniform_double(&state[key]);
        }
    };

    int x_idx = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int x_dim = n_dims * x_idx;

    // reuse rng state of x for y
    int y_idx = x_idx;
    int y_dim = n_dims * y_idx;

    for (int i = 0; i < n_dims; i++) {
        skipahead<rng_state *>(offset, &state[x_dim + i]);
    }

    constexpr int n_interior_samples = 8;

    for (int i = 0; i < n; i++) {
        mc<double> x = random_mccormick(state, x_dim);
        double interior_x_samples[n_interior_samples];
        random_interior_samples(state, x, interior_x_samples, n_interior_samples, x_dim + 4);
        check_univariate(x, interior_x_samples, n_interior_samples);

        mc<double> y = random_mccormick(state, y_dim);
        double interior_y_samples[n_interior_samples];
        random_interior_samples(state, y, interior_y_samples, n_interior_samples, y_dim + 4);
        check_bivariate(x, y, interior_x_samples, interior_y_samples, n_interior_samples);
    }
}

void tests_randomized(cuda_streams streams)
{

    rng_state *states;
    curandDirectionVectors64_t *h_directions;

    u64 *h_scrambled_constants;
    u64 *d_directions;
    u64 *d_scrambled_constants;

    CUDA_CHECK(cudaMalloc(&states, TOTAL_THREADS * n_dims * sizeof(*states)));

    CURAND_CHECK(curandGetDirectionVectors64(&h_directions, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
    CURAND_CHECK(curandGetScrambleConstants64(&h_scrambled_constants));

    CUDA_CHECK(cudaMalloc(&d_directions, n_dims * TOTAL_THREADS * VECTOR_SIZE * sizeof(*d_directions)));
    CUDA_CHECK(cudaMemcpy(d_directions, h_directions, n_dims * TOTAL_THREADS * VECTOR_SIZE * sizeof(*d_directions), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_scrambled_constants, n_dims * TOTAL_THREADS * sizeof(*d_scrambled_constants)));
    CUDA_CHECK(cudaMemcpy(d_scrambled_constants, h_scrambled_constants, n_dims * TOTAL_THREADS * sizeof(*d_scrambled_constants), cudaMemcpyHostToDevice));

    constexpr int n_iterations = 8;
    constexpr int n_samples    = 16;

    setup_randomized_kernel<<<BLOCK_COUNT, THREADS_PER_BLOCK, 0, streams[0]>>>(d_directions, d_scrambled_constants, states);
    cudaStreamSynchronize(streams[0]);

    for (u64 i = 0; i < n_iterations; i++) {
        u64 offset = i * (n_dims * TOTAL_THREADS);
        generate_and_check<<<BLOCK_COUNT, THREADS_PER_BLOCK, 0, streams[i % streams.size()]>>>(states, n_samples, offset);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(states));
    CUDA_CHECK(cudaFree(d_directions));
    CUDA_CHECK(cudaFree(d_scrambled_constants));
}
