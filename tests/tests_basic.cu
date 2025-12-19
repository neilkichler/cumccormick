#include <cumccormick/cumccormick.cuh>

#include "tests_common.h"

#include <stdio.h>

template<typename T>
using mc = cu::mccormick<T>;

template<typename T>
__device__ void print(mc<T> x)
{
    printf("(cv: %.15g, cc: %.15g, box: [%g, %g])\n", x.cv, x.cc, x.box.lb, x.box.ub);
};

template<typename T>
__global__ void basic_kernel()
{
    mc<T> a({ .lb = 0.0, .cv = 1.0, .cc = 2.0, .ub = 3.0 });
    mc<T> b({ .lb = 2.0, .cv = 3.0, .cc = 4.0, .ub = 5.0 });

    print(a);
    print(b);

    T two = 2.0;

    auto c = add(a, b);
    print(c);
    auto d = sub(a, b);
    print(d);
    auto e = mul(two, a);
    print(e);
    mc<T> f = sqr((a + b) - a);
    print(f);
    auto g = div(f, two);
    print(g);
    auto h = exp(a - two);
    print(h);
    auto i = sqrt(a + b);
    print(i);
    auto j = cos(i);
    print(j);
    auto k = sin(j);
    print(k);
    auto l = log(a);
    print(l);
    auto m = log(b);
    print(m);
    auto n = mul(a, b);
    print(n);
    auto o = recip(a);
    print(o);
    auto p = div(a, b);
    print(p);
    auto q = pown(a, 3);
    print(q);
    // auto r = pown(b, 3);
    // print(r);
    // auto s = pown(a - two, 3);
    // print(s);
    auto t = abs(a - two);
    print(t);
    auto u = abs(b);
    print(u);
    auto v = max(a, b);
    print(v);
    auto w = min(a, b);
    print(w);
    auto x = tanh(a);
    print(x);
    auto y = asin(two * (a - two));
    print(y);
    auto z = acos(two * (a - two));
    print(z);
    auto aa = atan(a);
    print(aa);
    auto bb = sinh(a);
    print(bb);
    auto cc = cosh(a);
    print(cc);
    auto dd = asinh(a);
    print(dd);
    auto ee = acosh(b);
    print(ee);
    auto ff = atanh(two * (a - two));
    print(ff);
}

__global__ void pown_kernel()
{
    mc<double> a({ .lb = 0.0, .cv = 1.0, .cc = 2.0, .ub = 3.0 });

    auto c = pow(a, 5);
    print(c);
    auto d = pow(a - 2.0, 5);
    print(d);
    auto e = pow(a, 4);
    print(e);
    auto f = pow(a - 2.0, 4);
    print(f);
    auto g = pow(a - 4.0, 4);
    print(g);
    auto h = pow(a + 0.0001, -4);
    print(h);
    auto i = pow(a + 1.0, -4);
    print(i);
    auto j = pow(a - 4.0, -4);
    print(j);
    auto k = pow(a - 1.5, -4);
    print(k);
    auto l = pow(a + 2.0, -5);
    print(l);
    auto m = pow(a, -5);
    print(m);
    auto n = pow(a - 4.0, -5);
    print(n);
    auto o = pow(a + 0.0001, -5);
    print(o);
}

template<typename T>
constexpr bool within_ulps(T x, T y, std::size_t n)
{
    if (x == y) {
        return true;
    }

    for (int i = 0; i < n; ++i) {
        x = std::nextafter(x, y);

        if (x == y) {
            return true;
        }
    }

    return false;
}

constexpr auto ackley(auto x, auto y)
{
    using std::numbers::e;
    using std::numbers::pi;
    return -20.0 * exp(-0.2 * sqrt(0.5 * (sqr(x) + sqr(y))))
        - exp(0.5 * (cos(2.0 * pi * x) + cos(2.0 * pi * y))) + e + 20.0;
}

constexpr auto griewank2d(auto x, auto y)
{
    return 1.0 + (sqr(x) + sqr(y)) / 4000.0 - cos(x / sqrt(1.0)) * cos(y / sqrt(2.0));
}

constexpr auto rastrigin2d(auto x, auto y)
{
    using std::numbers::pi;
    return 10.0 * 2.0 + sqr(x) + sqr(y) - 10.0 * cos(2.0 * pi * x) + -10.0 * cos(2.0 * pi * y);
}

template<typename T>
__global__ void contains_samples_check_univariate(mc<T> *xs, int n_x, std::integral auto n)
{
    // Check that a range of samples are all contained in the mccormick bound
    int i = blockIdx.x;
    int j = threadIdx.x;

    auto contains = [i](mc<T> y_mc, T y) {
        if (!(y_mc.cv <= y && y <= y_mc.cc)) {
            printf("[E][%d] Invalid bounds: y.cv = %.15g, y = %.15g, y.cc = %.15g\n", i, y_mc.cv, y, y_mc.cc);
            printf("[E][%d] Invalid bounds: y.cv = %a, y = %a, y.cc = %a\n", i, y_mc.cv, y, y_mc.cc);
        }
        return y_mc.cv <= y && y <= y_mc.cc;
    };

    if (i < n) {
        mc<T> x    = xs[j];
        T x_sample = x.cv + static_cast<T>(i) * (x.cc - x.cv) / static_cast<T>(n);
        assert(contains(pow(x, 1), x_sample));
        assert(contains(pow(x, 2), x_sample * x_sample));
        assert(contains(pow(x, 3), pow(x_sample, 3)));
        assert(contains(pow(x, 4), pow(x_sample, 4)));
        assert(contains(pow(x, 5), pow(x_sample, 5)));
        assert(contains(abs(x), abs(x_sample)));
        assert(contains(exp(x), exp(x_sample)));
        assert(contains(abs(x), abs(x_sample)));
        assert(contains(fabs(x), fabs(x_sample)));
        assert(contains(neg(x), -x_sample));
        assert(contains(sqr(x), x_sample * x_sample));
        assert(contains(cos(x), cos(x_sample)));
        assert(contains(sin(x), sin(x_sample)));
        assert(contains(tanh(x), tanh(x_sample)));
        assert(contains(atan(x), atan(x_sample)));
        assert(contains(asinh(x), asinh(x_sample)));
        if (inf(x) >= 0) {
            assert(contains(log(x), log(x_sample)));
            assert(contains(recip(x), __drcp_rn(x_sample)));
            assert(contains(sqrt(x), sqrt(x_sample)));

            if (inf(x) >= 1) {
                assert(contains(acosh(x), acosh(x_sample)));
            }
        }

        if (inf(x) >= -1.0 and sup(x) <= 1.0) {
            assert(contains(asin(x), asin(x_sample)));
            assert(contains(acos(x), acos(x_sample)));
            assert(contains(atanh(x), atanh(x_sample)));
            assert(contains(sinh(x), sinh(x_sample)));
            assert(contains(cosh(x), cosh(x_sample)));
        }
    }
}

template<typename T>
__global__ void contains_samples_check_bivariate(mc<T> *xs, mc<T> *ys, std::integral auto n)
{
    // Check that a range of samples are all contained in the mccormick bound
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;

    auto contains = [](mc<T> x, T y) {
        return x.cv <= y && y <= x.cc;
    };

    if (i < n) {
        mc<T> x = xs[j];
        mc<T> y = ys[k];

        T x_sample = x.cv + static_cast<T>(i) * (x.cc - x.cv) / static_cast<T>(n);
        T y_sample = y.cv + static_cast<T>(i) * (y.cc - y.cv) / static_cast<T>(n);
        assert(contains(x + y, x_sample + y_sample));
        assert(contains(x - y, x_sample - y_sample));
        assert(contains(x * y, x_sample * y_sample));
        assert(contains(max(x, y), max(x_sample, y_sample)));
        assert(contains(min(x, y), min(x_sample, y_sample)));
    }
}

__global__ void test_fn_kernel()
{
    mc<double> x({ .lb = 1.0, .cv = 1.5, .cc = 1.5, .ub = 2.0 });
    mc<double> y({ .lb = 0.5, .cv = 0.6, .cc = 0.65, .ub = 0.7 });
    mc<double> z({ .lb = -1.0, .cv = 0.2, .cc = 1.0, .ub = 2.0 });

    auto xy = x * y;
    assert(within_ulps(xy.cv, 0.85, 1));
    assert(within_ulps(xy.cc, 1.0, 1));

    auto xz = x * z;
    assert(within_ulps(xz.cv, -0.3, 1));
    assert(within_ulps(xz.cc, 2.0, 1));

    auto yz = y * z;
    assert(within_ulps(yz.cc, 0.8, 1));

    auto xexp = x * exp(-pow(x, 2));
    assert(within_ulps(xexp.cv, 0x1.75bb077991bc3p-4, 1));
    assert(within_ulps(xexp.cc, 0x1.9fea64b7c3615p-2, 1));

    auto sincospow = sin(pown(y, -3)) * cos(pown(y, 2)); // -9.358968236779348e-01, 6.095699354841704e-01
    assert(inf(sincospow) > -1.0);
    assert(sincospow.cv > -1.0);
    assert(sincospow.cc < 1.0);
    assert(sup(sincospow) < 1.0);

    mc<double> ack_x({ .cv = 0.0, .cc = 0.0, .box = { { .lb = -42.0, .ub = 42.0 } } });
    mc<double> ack_y({ .cv = 0.0, .cc = 0.0, .box = { { .lb = -42.0, .ub = 42.0 } } });
    auto ack = ackley(ack_x, ack_y);
    assert(ack.cv > 0.0 - 1e-12);
    assert(inf(ack) > 0.0 - 1e-12);

    mc<double> gw_x({ .cv = 0.0, .cc = 0.0, .box = { { .lb = -42.0, .ub = 42.0 } } });
    mc<double> gw_y({ .cv = 0.0, .cc = 0.0, .box = { { .lb = -42.0, .ub = 42.0 } } });
    auto gw = griewank2d(gw_x, gw_y);
    assert(within_ulps(gw.cv, 0.0, 1));
    assert(within_ulps(inf(gw), 0.0, 1));

    mc<double> ras_x({ .cv = 1e-6, .cc = 1e-6, .box = { { .lb = -100.0, .ub = 100.0 } } });
    mc<double> ras_y({ .cv = 1e-6, .cc = 1e-6, .box = { { .lb = -100.0, .ub = 100.0 } } });
    auto ras = rastrigin2d(ras_x, ras_y);
    assert(ras.cv >= 0.0);
    assert(inf(ras) >= 0.0);

    {
        mc<double> a({ .lb = 0.7042016756583301, .cv = 0.7042016756583301, .cc = 0.7042016756583301, .ub = 0.7238093671679029 });
        mc<double> b(0.6617671655226747);
        auto c = a * b;
        assert(c.cv <= c.cc);
        c = sin(a);
        assert(c.cv <= c.cc);
    }

    { // unit test recip
        auto infty = std::numeric_limits<double>::infinity();
        {
            mc<double> a({ .lb = -1.0, .cv = 0.0, .cc = 0.0, .ub = 1.0 });
            auto b = recip(a);
            assert(inf(b) == -infty);
            assert(cv(b) == -infty);
            assert(cc(b) == +infty);
            assert(sup(b) == +infty);
        }
        {
            mc<double> a({ .lb = -1.0, .cv = 0.0, .cc = 0.0, .ub = 0.0 });
            auto b = recip(a);
            assert(inf(b) == -infty);
            assert(cv(b) == -infty);
            assert(cc(b) == -infty); // see implementation for explanation
            assert(sup(b) == -1.0);
        }
        {
            mc<double> a({ .lb = -1.0, .cv = -0.5, .cc = -0.5, .ub = 0.0 });
            auto b = recip(a);
            assert(inf(b) == -infty);
            assert(cv(b) == -infty);
            assert(cc(b) == -2.0);
            assert(sup(b) == -1.0);
        }
        {
            mc<double> a({ .lb = 0.0, .cv = 0.0, .cc = 0.0, .ub = 1.0 });
            auto b = recip(a);
            assert(inf(b) == 1.0);
            assert(cv(b) == infty);
            assert(cc(b) == infty);
            assert(sup(b) == infty);
        }
        {
            mc<double> a({ .lb = 0.0, .cv = 0.5, .cc = 0.5, .ub = 1.0 });
            auto b = recip(a);
            assert(inf(b) == 1.0);
            assert(cv(b) == 2.0);
            assert(cc(b) == infty);
            assert(sup(b) == infty);
        }
    }
}

void test_bounds([[maybe_unused]] cudaStream_t stream)
{
    constexpr int n_samples = 512;
    constexpr int n_xs      = 15;

    mc<double> xs[n_xs] = {
        //                  lb,                  cv,                   cc,                   ub
        {                  0.0,                 0.6,                 0.65,                  0.7 },
        {                  6.1,                 7.6,                 7.65,                  7.7 },
        {                 50.0,                50.6,               100.65,                100.7 },
        {                 -4.1,                 3.6,                 3.85,                  7.7 },
        {                 -0.1,               -0.01,                 0.01,                  0.1 },
        {                -0.01,               -0.01,                 0.01,                 0.01 },
        {                  0.0,            10000.01,             10001.01,             100000.0 },
        {                 -4.1,               -3.96,                -3.25,                 -3.1 },
        {                0.875,               0.875,                0.875,                0.875 },
        {                  0.5,                 0.5,                  0.5,                  0.5 },
        {          0x1.eb12p-2,         0x1.eb12p-1,          0x1.eb12p-1,          0x1.eb12p-1 },
        {                 -1.0,                 0.3,                  0.5,                  4.0 },
        {                 -4.0,                 0.3,                  0.5,                  1.0 },
        { 0x1.6636b09e7047p-33, 0x1.b6b00005212bp-1, 0x1.b6b0000580578p-1, 0x1.b6b00005a1b54p-1 },
        {   0.7042016756583301,  0.7042016756583301,   0.7042016756583301,   0.7238093671679029 },
    };

    mc<double> *d_xs;
    CUDA_CHECK(cudaMalloc(&d_xs, n_xs * sizeof(mc<double>)));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_xs * sizeof(mc<double>), cudaMemcpyHostToDevice));

    contains_samples_check_univariate<<<n_samples, n_xs>>>(d_xs, n_xs, n_samples);

    mc<double> *d_ys;
    constexpr int n_ys  = 3;
    mc<double> ys[n_ys] = {
        { -1.0, -0.5, 0.5, 3.0 },
        { 0.0, 0.5, 2.5, 3.0 },
        { 0.6617671655226747, 0.6617671655226747, 0.6617671655226747, 0.6617671655226747 },
    };

    CUDA_CHECK(cudaMalloc(&d_ys, n_ys * sizeof(mc<double>)));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n_ys * sizeof(mc<double>), cudaMemcpyHostToDevice));

    dim3 blocks(n_xs, n_ys);
    contains_samples_check_bivariate<<<n_samples, blocks>>>(d_xs, d_ys, n_samples);

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
}

void test_basic(cudaStream_t stream)
{
    basic_kernel<float><<<1, 1, 0, stream>>>();
    basic_kernel<double><<<1, 1, 0, stream>>>();
}

void test_pown(cudaStream_t stream)
{
    pown_kernel<<<1, 1, 0, stream>>>();
}

void test_fn(cudaStream_t stream)
{
    test_fn_kernel<<<1, 1, 0, stream>>>();
}
