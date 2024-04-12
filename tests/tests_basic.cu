#include <cumccormick/arithmetic/basic.cuh>

#include <stdio.h>

__device__ void print(mc<double> x)
{
    printf("(cv: %.15g, cc: %.15g, box: [%g, %g])\n", x.cv, x.cc, x.box.lb, x.box.ub);
};

__global__ void basic_kernel()
{
    mc<double> a { .cv = 1.0, .cc = 2.0, .box = { .lb = 0.0, .ub = 3.0 } };
    mc<double> b { .cv = 3.0, .cc = 4.0, .box = { .lb = 2.0, .ub = 5.0 } };

    print(a);
    print(b);

    auto c = add(a, b);
    print(c);
    auto d = sub(a, b);
    print(d);
    auto e = mul(2.0, a);
    print(e);
    auto f = sqr((a + b) - a);
    print(f);
    auto g = div(f, 2.0);
    print(g);
    auto h = exp(a - 1.5);
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
    auto r = pown(b, 3);
    print(r);
    auto s = pown(a - 2.0, 3);
    print(s);
    auto t = abs(a - 2.0);
    print(t);
    auto u = abs(b);
    print(u);
    auto v = max(a, b);
    print(v);
    auto w = min(a, b);
    print(w);
}

__global__ void test_pown()
{
    mc<double> a { .cv = 1.0, .cc = 2.0, .box = { .lb = 0.0, .ub = 3.0 } };

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
__device__ bool within_ulps(T x, T y, std::size_t n)
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

__device__ auto ackley(auto x, auto y)
{
    using std::numbers::e;
    using std::numbers::pi;
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x * x + y * y)))
        - exp(0.5 * (cos(2.0 * pi * x) + cos(2.0 * pi * y))) + e + 20.0;
}

template<typename T>
__global__ void contains_samples_check_univariate(mc<T> x, std::integral auto n)
{
    // Check that a range of samples are all contained in the mccormick bound
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    auto contains = [](mc<T> x, T y) {
        return x.cv <= y && y <= x.cc;
    };

    if (i < n) {
        T x_sample = x.cv + static_cast<T>(i) * (x.cc - x.cv) / static_cast<T>(n);
        assert(contains(pow(x, 1), pow(x_sample, 1)));
        assert(contains(pow(x, 2), pow(x_sample, 2)));
        assert(contains(pow(x, 3), pow(x_sample, 3)));
        assert(contains(pow(x, 4), pow(x_sample, 4)));
        assert(contains(pow(x, 5), pow(x_sample, 5)));
        assert(contains(abs(x), abs(x_sample)));
        assert(contains(exp(x), exp(x_sample)));
        assert(contains(fabs(x), fabs(x_sample)));
        assert(contains(log(x), log(x_sample)));
        assert(contains(recip(x), pow(x_sample, -1)));
        assert(contains(sqr(x), pow(x_sample, 2)));
        assert(contains(sqrt(x), sqrt(x_sample)));
        assert(contains(cos(x), cos(x_sample)));
        assert(contains(sin(x), sin(x_sample)));
    }
}

__global__ void test_fn_kernel()
{
    mc<double> x { .cv = 1.5, .cc = 1.5, .box = { .lb = 1.0, .ub = 2.0 } };
    mc<double> y { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.5, .ub = 0.7 } };
    mc<double> z { .cv = 0.2, .cc = 1.0, .box = { .lb = -1.0, .ub = 2.0 } };

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

    // auto sincospow = sin(pown(y, -3)) * cos(pown(y, 2)); // -9.358968236779348e-01, 6.095699354841704e-01
    // printf("sincospow.cv: %a, %.15f\n", sincospow.cv, sincospow.cv);
    // printf("sincospow.cc: %a, %.15f\n", sincospow.cc, sincospow.cc);

    auto ack = ackley(x, y);
    // printf("ack.cv: %a, %.15f\n", ack.cv, ack.cv);
    // printf("ack.cc: %a, %.15f\n", ack.cc, ack.cc);
}

void bounds_kernel(cudaStream_t stream)
{
    mc<double> x { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.0, .ub = 0.7 } };

    constexpr int n_samples = 512;
    contains_samples_check_univariate<<<n_samples, 1>>>(x, n_samples);
}

void basic_kernel(cudaStream_t stream)
{
    basic_kernel<<<1, 1, 0, stream>>>();
}

void pown_kernel(cudaStream_t stream)
{
    test_pown<<<1, 1, 0, stream>>>();
}

void fn_kernel(cudaStream_t stream)
{
    test_fn_kernel<<<1, 1, 0, stream>>>();
}
