#include "../common.h"
#include "../tests/tests_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <iomanip>
#include <iostream>
#include <numbers>

template<typename T>
using mc = cu::mccormick<T>;

namespace blackscholes
{

template<typename T>
struct parameters
{
    T r;     // interest rate
    T S0;    // spot price
    T tau;   // time until maturity
    T K;     // strike price
    T sigma; // std. dev. of stock return (i.e., volatility)
};

template<typename T>
constexpr auto call(parameters<T> params)
{
    auto [r, S0, tau, K, sigma] = params;
    assert((S0 > 0.0) && (tau > 0.0) && (sigma > 0.0) && (K > 0.0));

    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;

    auto normcdf = [](auto x) {
        using std::erfc;
        return 0.5 * erfc(-x * 1.0 / std::numbers::sqrt2);
    };

    auto discount_factor = exp(-r * tau);
    auto variance        = sigma * sqrt(tau);
    auto forward_price   = S0 / discount_factor;

    auto dp         = (log(forward_price / K) + 0.5 * pow(sigma, 2) * tau) / variance;
    auto dm         = dp - variance;
    auto call_price = discount_factor * (forward_price * normcdf(dp) - K * normcdf(dm));
    return call_price;
}

}; // namespace blackscholes

__global__ void bs_kernel(auto *ps, auto *res, std::integral auto n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = blackscholes::call(ps[i]);
    }
}

int main()
{
    constexpr int n = 256;

    using T = mc<double>;
    blackscholes::parameters<T> xs[n];
    T res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i + 100;

        xs[i] = {
            .r     = { .cv = 0.01, .cc = 0.01, .box = { .lb = 0.01, .ub = 0.01 } },
            .S0    = { .cv = 99.5, .cc = 100.5, .box = { .lb = 99, .ub = 101 } },
            .tau   = { .cv = 0.01 * v, .cc = 0.01 * v, .box = { .lb = 0.01 * v, .ub = 0.01 * v } },
            .K     = { .cv = 95, .cc = 95, .box = { .lb = 95, .ub = 95 } },
            .sigma = { .cv = 0.5, .cc = 0.5, .box = { .lb = 0.5, .ub = 0.5 } },
        };
    }

    blackscholes::parameters<T> *d_xs;
    T *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    bs_kernel<<<n, 1>>>(d_xs, d_res, n);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    std::cout << "Black Scholes call option price for r=0.01, S0=100, K=95, sigma=0.5, and" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    for (auto i = 0; i < n; i++) {
        auto r = res[i];
        std::cout << "t=" << xs[i].tau.cv << " -> " << r << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
