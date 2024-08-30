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

__device__ auto model(auto x, auto y)
{
    auto rosen = rosenbrock(x, y);
    auto z     = cos(rosen) - x + x;
    z          = 10.0 * z;
    return z;
}

__global__ void k_model(auto *x, auto *y, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = model(x[i], y[i]);
    }
}

template<typename T>
void single_kernel_model(cuda_ctx &ctx, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, int n)
{
    k_model<<<n, 1>>>(d_xs, d_ys, d_res, n);
}

int main()
{
    constexpr int n = 128;
    using T         = mc<double>;
    T res[n];

    std::vector<mc<double>> xs {
        { .cv = -1.96, .cc = 1.25, .box = { .lb = -2.0, .ub = 2.0 } },
        { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.0, .ub = 0.7 } },
        { .cv = 7.6, .cc = 7.65, .box = { .lb = 6.1, .ub = 7.7 } },
        { .cv = 50.6, .cc = 100.65, .box = { .lb = 50.0, .ub = 100.7 } },
        { .cv = 3.6, .cc = 3.85, .box = { .lb = -4.1, .ub = 7.7 } },
        { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.1, .ub = 0.1 } },
        { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.01, .ub = 0.01 } },
        { .cv = 10000.01, .cc = 10001.01, .box = { .lb = 0.0, .ub = 100000.0 } },
        { .cv = -3.96, .cc = -3.25, .box = { .lb = -4.1, .ub = -3.1 } },
    };

    std::vector<mc<double>> ys {
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    };

    mc<double> *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*d_xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*d_ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*d_res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    printf("rosenbrok(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    xs[0] = { .cv = -4.96, .cc = 4.25, .box = { .lb = -8.0, .ub = 8.0 } };

    // Second run
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), n * sizeof(*d_xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), n * sizeof(*d_ys), cudaMemcpyHostToDevice));

    k_model<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    r = res[0];
    printf("rosenbrok(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));
}
