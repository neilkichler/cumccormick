#include <cumccormick/arithmetic/basic.cuh>

#include "tests_root.h"

#include <numbers>
#include <stdio.h>

__global__ void root_kernel()
{
    solver_options<double> opts {};
    auto tolerance = opts.tolerance;
    auto check     = [tolerance](auto z, auto z_true) {
        auto lb = z_true - tolerance;
        auto ub = z_true + tolerance;
        printf("root x=%.15g is inside [%.15g, %.15g]\n", z, lb, ub);
        assert(z >= lb && z <= ub);
    };
    {
        auto f  = [](double x) { return cos(x); };
        auto df = [](double x) { return -sin(x); };

        double z      = root_safe_newton(f, df, 4.0, 4.0, 2.0 * std::numbers::pi, opts);
        double z_true = 3.0 * (std::numbers::pi / 2.0);

        check(z, z_true);
    }
    {
        auto f  = [](double x) { return pow(x - 1.75, 3); };
        auto df = [](double x) { return 3.0 * pow(x - 1.75, 2); };

        double z      = root_safe_newton(f, df, 1.5, 1.0, 2.0, opts);
        double z_true = 1.75;

        check(z, z_true);
    }
    {
        auto f  = [](double x) { return log(x); };
        auto df = [](double x) { return 1.0 / x; };

        double z      = root_safe_newton(f, df, 7.5, 0.25, 10.0, opts);
        double z_true = 1.0;

        check(z, z_true);
    }
    {
        auto f  = [](double x) { return pow(x, 2) - 5.0*x + 4.0; };
        auto df = [](double x) { return 2.0 * x - 5.0; };

        double z = root_safe_newton(f, df, 0.0, -1.0, 1.0, opts);
        double z_true = 1.0;

        check(z, z_true);
    }
}

void tests_root(cudaStream_t stream, cudaEvent_t event)
{
    root_kernel<<<1, 1, 0, stream>>>();
}
