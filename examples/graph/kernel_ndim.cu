#include <array>
#include <cstdio>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

template<typename T>
__device__ T model(T *x, int n_vars)
{
    T res {};

    // Slow for loop on purpose to simulate large model
    for (int i = 0; i < n_vars; i++) {
        auto x_i = x[i];
        res += sqr(cos(x_i - 1)) + pow(x_i - 1, 3) + pow(x_i - 1, 4);
    }

    return res - 1;
}

// x   of size n_elems * n_vars
// res of size n_elems
__global__ void k_model(auto *x, auto *res, int n_elems, int n_vars)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    //     res[i] = model(x[i], y[i]);
    // }

    // printf("blockDim.x: %d, gridDim.x: %d\n", blockDim.x, gridDim.x);
    // printf("Hello from outside\n");
#if 1
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n_elems; i += blockDim.x * gridDim.x) {
        // printf("blockIdx.x = %d\n", blockIdx.x);
        // printf("[i]: %d %f %f %f\n", i, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = model(&x[i * n_vars], n_vars);
        // printf("[i]: %d %f %f %f\n", i, res[i].cv, res[i].cc, res[i].box.ub);
    }
#endif
}

int main()
{
    constexpr int n_blocks            = 1024;
    constexpr int n_threads_per_block = 512;
    constexpr int n_elems             = 1024;
    constexpr int n_vars              = 10000;

    using T = mc<double>;

    mc<double> *d_xs, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*d_xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n_elems * sizeof(*d_res)));

    T *xs;
    T *res;

    CUDA_CHECK(cudaMallocHost(&xs, n_elems * n_vars * sizeof(*xs)));
    CUDA_CHECK(cudaMallocHost(&res, n_elems * sizeof(*res)));

    for (int i = 0; i < n_elems; i++) {
        double v = i;
        for (int j = 0; j < n_vars; j++) {
            xs[i * n_vars + j] = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
        }
        // double v = 0;
        // xs[i]    = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } };
        // printf("setting xs[i]: %d %f %f %f\n", i, xs[i].cv, xs[i].cc, xs[i].box.ub);
    }

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*d_xs), cudaMemcpyHostToDevice));
    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_res, n_elems, n_vars);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__,
                __FILE__, __LINE__, cudaGetErrorString(err),
                cudaGetErrorName(err), err);
    }

    CUDA_CHECK(cudaMemcpy(res, d_res, n_elems * sizeof(*res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto r = res[0];
    printf("custom([0, (0, 0), 0]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    // Second run
    xs[0] = { .cv = 0.5, .cc = 1.0, .box = { .lb = 0.0, .ub = 2.0 } };

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * sizeof(*d_xs), cudaMemcpyHostToDevice));

    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_res, n_elems, n_vars);

    CUDA_CHECK(cudaMemcpy(res, d_res, n_elems * sizeof(*res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    r = res[0];
    printf("custom([0.0, (0.5, 1.0), 2.0]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));
}
