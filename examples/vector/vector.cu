#include <array>
#include <cstdio>
#include <iostream>
#include <span>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

#define fn     __device__ auto
#define lambda [] __device__

fn rosenbrock(auto x, auto y)
{
    double a = 1.0;
    double b = 100.0;
    return pow(a - x, 2) + b * pow((y - pow(x, 2)), 2);
}

fn model(auto x, auto y)
{
    auto rosen = rosenbrock(x, y);
    auto z     = cos(rosen) - x + x;
    z          = 10.0 * z;
    return z;
}

__global__ void generic_kernel(auto &&f, mc<double> *xs, mc<double> *ys, mc<double> *res, auto n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        res[i] = f(xs[i], ys[i]);
    }
}

template<typename T>
void launch(auto &&user_kernel, std::span<mc<T>> xs, std::span<mc<T>> ys)
{
    mc<T> *d_xs;
    mc<T> *d_ys;
    mc<T> *d_res;

    auto n        = xs.size();
    auto xs_size  = xs.size_bytes();
    auto ys_size  = ys.size_bytes();
    auto res_size = xs_size;

    CUDA_CHECK(cudaMalloc(&d_xs, xs_size));
    CUDA_CHECK(cudaMalloc(&d_ys, ys_size));
    CUDA_CHECK(cudaMalloc(&d_res, res_size));

    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), xs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), ys_size, cudaMemcpyHostToDevice));

    // TODO: have a global context from which we can make use of the
    // gpu configuration, stream, threadblocks etc.

    // TODO: make kernel variadic in input arguments
    generic_kernel<<<128, 1>>>(user_kernel, d_xs, d_ys, d_res, n);

    std::vector<mc<T>> res(n);
    CUDA_CHECK(cudaMemcpy(res.data(), d_res, res_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));

    printf("Results: \n");
    for (auto r : res) {
        println("{}", r);
    }
}

void user_kernel_multiple_inputs()
{
    // clang-format off
    std::vector<mc<double>> xs {
        //   lb        cv        cc        ub
        {  -2.0,    -1.96,     1.25,      2.0 },
        {   0.0,      0.6,     0.65,      0.7 },
        {   6.1,      7.6,     7.65,      7.7 },
        {  50.0,     50.6,   100.65,    100.7 },
        {  -4.1,      3.6,     3.85,      7.7 },
        {  -0.1,    -0.01,     0.01,      0.1 },
        { -0.01,    -0.01,     0.01,     0.01 },
        {   0.0, 10000.01, 10001.01, 100000.0 },
        {  -4.1,    -3.96,    -3.25,     -3.1 },
    };

    std::vector<mc<double>> ys {
        //  lb    cv   cc   ub
        { -1.0, -0.5, 0.5, 3.0 },
        {  0.0,  0.5, 2.5, 3.0 },
        { -1.0, -0.5, 0.5, 3.0 },
        {  0.0,  0.5, 2.5, 3.0 },
        { -1.0, -0.5, 0.5, 3.0 },
        {  0.0,  0.5, 2.5, 3.0 },
        { -1.0, -0.5, 0.5, 3.0 },
        {  0.0,  0.5, 2.5, 3.0 },
        { -1.0, -0.5, 0.5, 3.0 },
    };
    // clang-format on

    //
    // Using a lambda function as the user kernel
    //
    auto user_kernel = lambda(auto x, auto y)
    {
        return pow(1.0 - x, 2) + 100.0 * pow((y - pow(x, 2)), 2);
    };

    launch<double>(user_kernel, xs, ys);

    //
    // Using a predefined function (must be wrapped in a lambda)
    //
    launch<double>([] __device__(auto x, auto y) { return model(x, y); }, xs, ys);
}

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    std::array<cuda_buffer, n_streams> buffers {};

    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::array<cudaEvent_t, n_streams> events {};
    for (auto &event : events)
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    cuda_ctx ctx { buffers, streams, events };
    user_kernel_multiple_inputs();

    for (auto &event : events)
        CUDA_CHECK(cudaEventDestroy(event));

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
