#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

__device__ auto beale(auto x, auto y)
{
    return pow(1.5 - x * (1 - y), 2)
        + pow(2.25 - x * (1 - sqr(y)), 2)
        + pow(2.625 - x * (1 - pow(y, 3)), 2);
}

__global__ void kernel(auto *xs, auto *ys, auto *res, std::integral auto n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = beale(xs[i], ys[i]);
    }
}

int main()
{
    constexpr int n = 256;
    using T = mc<double>;
    T xs[n], ys[n], res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i] = {{ .lb = -v, .cv = -v, .cc = v, .ub = v }};
        ys[i] = {{ .lb = -v, .cv = -v, .cc = v, .ub = v }};
    }

    mc<double> *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*ys), cudaMemcpyHostToDevice));

    kernel<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    printf("beale(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
