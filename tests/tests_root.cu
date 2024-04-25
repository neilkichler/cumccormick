#include <cumccormick/arithmetic/basic.cuh>

#include "tests_root.h"

#include <numbers>
#include <stdio.h>

__global__ void root_kernel()
{
    solver_options<double> opts { .maxiter = 100 };
    auto tolerance = opts.atol;
    auto check     = [tolerance](auto z, auto z_true) {
        auto lb = z_true - tolerance;
        auto ub = z_true + tolerance;
        printf("root x=%.15g is inside [%.15g, %.15g]\n", z, lb, ub);
        assert(z >= lb && z <= ub);
    };
    {
        auto f    = [](double x) { return cos(x); };
        auto df   = [](double x) { return -sin(x); };
        auto ddf  = [](double x) { return -cos(x); };
        auto dddf = [](double x) { return sin(x); };

        double z_true = 3.0 * (std::numbers::pi / 2.0);
        {
            printf("Newton Bisection\n");
            double z = root_newton_bisection(f, df, 4.0, 4.0, 2.0 * std::numbers::pi);
            check(z, z_true);
        }
        {
            printf("Halley bisection\n");
            double z = root_halley_bisection(f, df, ddf, 4.0, 4.0, 2.0 * std::numbers::pi);
            check(z, z_true);
        }
        {
            printf("Householder bisection\n");
            double z = root_householder_bisection(f, df, ddf, dddf, 4.0, 4.0, 2.0 * std::numbers::pi);
            check(z, z_true);
        }
    }
    {
        auto f    = [](double x) { return pow(x - 1.75, 3); };
        auto df   = [](double x) { return 3.0 * pow(x - 1.75, 2); };
        auto ddf  = [](double x) { return 6.0 * (x - 1.75); };
        auto dddf = [](double x) { return 6.0; };

        double z_true = 1.75;
        {
            printf("Halley\n");
            double z = root_halley(f, df, ddf, 1.5, 1.0, 2.0, opts);
            check(z, z_true);
        }
        {
            printf("Householder\n");
            double z = root_householder(f, df, ddf, dddf, 1.5, 1.0, 2.0, opts);
            check(z, z_true);
        }
        {
            printf("Newton bisection\n");
            double z = root_newton_bisection(f, df, 1.5, 1.0, 2.0, opts);
            check(z, z_true);
        }
        {
            printf("Halley bisection\n");
            double z = root_halley_bisection(f, df, ddf, 1.5, 1.0, 2.0, opts);
            check(z, z_true);
        }
        {
            printf("Householder bisection\n");
            double z = root_householder_bisection(f, df, ddf, dddf, 1.5, 1.0, 2.0, opts);
            check(z, z_true);
        }
    }
    {
        auto f    = [](double x) { return log(x); };
        auto df   = [](double x) { return 1.0 / x; };
        auto ddf  = [](double x) { return -1.0 / pow(x, 2); };
        auto dddf = [](double x) { return 2.0 / pow(x, 3); };

        double z_true = 1.0;
        {
            printf("Newton bisection\n");
            double z = root_newton_bisection(f, df, 7.5, 0.25, 10.0, opts);
            check(z, z_true);
        }
        {
            printf("Halley bisection\n");
            double z = root_halley_bisection(f, df, ddf, 7.5, 0.25, 10.0, opts);
            check(z, z_true);
        }
        {
            printf("Householder bisection\n");
            double z = root_householder_bisection(f, df, ddf, dddf, 7.5, 0.25, 10.0, opts);
            check(z, z_true);
        }
    }
    {
        auto f    = [](double x) { return pow(x, 2) - 5.0 * x + 4.0; };
        auto df   = [](double x) { return 2.0 * x - 5.0; };
        auto ddf  = [](double x) { return 2.0; };
        auto dddf = [](double x) { return 0.0; };

        double z_true = 1.0;
        {
            printf("Newton bisection\n");
            double z = root_newton_bisection(f, df, 0.0, -1.0, 1.0, opts);
            check(z, z_true);
        }
        {
            printf("Halley bisection\n");
            double z = root_halley_bisection(f, df, ddf, 0.0, -1.0, 1.0, opts);
            check(z, z_true);
        }
        {
            printf("Householder bisection\n");
            double z = root_householder_bisection(f, df, ddf, dddf, 0.0, -1.0, 1.0, opts);
            check(z, z_true);
        }
    }
}

void tests_root(cudaStream_t stream, cudaEvent_t event)
{
    root_kernel<<<1, 1, 0, stream>>>();
}
