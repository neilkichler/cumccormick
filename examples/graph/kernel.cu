#include <array>
#include <cstdio>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

__device__ auto rosenbrock(auto x, auto y)
{
    double a = 1.0;
    double b = 100.0;
    return pow(a - x, 2) + b * pow((y - pow(x, 2)), 2);
}

__device__ auto beale(auto x, auto y)
{
    return pow(1.5 - x * (1 - y), 2)
        + pow(2.25 - x * (1 - sqr(y)), 2)
        + pow(2.625 - x * (1 - pow(y, 3)), 2);
}

__device__ auto model(auto x, auto y)
{
    return beale(x, y);
    // return rosenbrock(x, y);
}

__global__ void k_model(auto *x, auto *y, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    //     res[i] = model(x[i], y[i]);
    // }

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("Hello");
        // printf("[i]: %d %f %f %f\n", i, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = model(x[i], y[i]);
    }
}

// #define FIXED 1
int main()
{
    constexpr int n_blocks            = 1024;
    constexpr int n_threads_per_block = 1024;
    // constexpr int n                   = 32 * 1024 * 1024;
    constexpr int n                   = 100 * 1024;

    using T = mc<double>;

    mc<double> *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*d_xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*d_ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*d_res)));

#ifdef FIXED
    // std::vector<mc<double>> xs {
    //     { .cv = -1.96, .cc = 1.25, .box = { .lb = -2.0, .ub = 2.0 } },
    //     { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.0, .ub = 0.7 } },
    //     { .cv = 7.6, .cc = 7.65, .box = { .lb = 6.1, .ub = 7.7 } },
    //     { .cv = 50.6, .cc = 100.65, .box = { .lb = 50.0, .ub = 100.7 } },
    //     { .cv = 3.6, .cc = 3.85, .box = { .lb = -4.1, .ub = 7.7 } },
    //     { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.1, .ub = 0.1 } },
    //     { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.01, .ub = 0.01 } },
    //     { .cv = 10000.01, .cc = 10001.01, .box = { .lb = 0.0, .ub = 100000.0 } },
    //     { .cv = -3.96, .cc = -3.25, .box = { .lb = -4.1, .ub = -3.1 } },
    // };
    //
    // std::vector<mc<double>> ys {
    //     { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    //     { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
    //     { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    //     { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
    //     { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    //     { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
    //     { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    //     { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
    //     { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    // };

    std::vector<mc<double>> xs(n);
    std::vector<mc<double>> ys(n);
    T res[n];

    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i]    = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
        ys[i]    = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
    }

    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    printf("rosenbrok(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    xs[0] = { .cv = -4.96, .cc = 4.25, .box = { .lb = -8.0, .ub = 8.0 } };

    // Second run
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    r = res[0];
    printf("rosenbrok(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

#else
    printf("inside else\n");
    T *xs;
    T *ys;
    T *res;

    CUDA_CHECK(cudaMallocHost(&xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMallocHost(&ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMallocHost(&res, n * sizeof(*res)));

    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i]    = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
        ys[i]    = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
    }

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto r = res[0];
    printf("rosenbrok([0, (0, 0), 0]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    r = res[1];
    printf("rosenbrok([-1, (-1, 1), 1]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    // Second run
    xs[0] = { .cv = -4.96, .cc = 4.25, .box = { .lb = -8.0, .ub = 8.0 } };

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n_blocks, n_threads_per_block>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    r = res[0];
    printf("rosenbrok(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

#endif

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));
}
