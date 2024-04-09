#include <cumccormick/arithmetic/basic.cuh>

#include <stdio.h>

__global__ void basic_kernel()
{
    auto print = [](mc<double> x) { printf("(cv: %.15g, cc: %.15g, box: [%g, %g])\n", x.cv, x.cc, x.box.lb, sup(x)); };
    mc<double> a = { .cv = 1.0, .cc = 2.0, .box = { .lb = 0.0, .ub = 3.0 } };
    mc<double> b = { .cv = 3.0, .cc = 4.0, .box = { .lb = 2.0, .ub = 5.0 } };

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
}

__global__ void test_pown()
{
    mc<double> a = { .cv = 1.0, .cc = 2.0, .box = { .lb = 0.0, .ub = 3.0 } };

    auto print = [](mc<double> x) { printf("(cv: %.15g, cc: %.15g, box: [%g, %g])\n", x.cv, x.cc, x.box.lb, sup(x)); };

    auto c = pown(a, 5);
    print(c);
    auto d = pown(a - 2.0, 5);
    print(d);
    auto e = pown(a, 4);
    print(e);
    auto f = pown(a - 2.0, 4);
    print(f);
    auto g = pown(a - 4.0, 4);
    print(g);
    auto h = pown(a, -4);
    print(h);
    auto i = pown(a + 1.0, -4);
    print(i);
    auto j = pown(a - 4.0, -4);
    print(j);
    auto k = pown(a - 1.5, -4);
    print(k);
    auto l = pown(a + 2.0, -5);
    print(l);
    auto m = pown(a, -5);
    print(m);
    auto n = pown(a - 4.0, -5);
    print(n);
}

__global__ void test_fn_kernel()
{
    mc<double> x = { .cv = 1.5, .cc = 1.5, .box = { .lb = 1.0, .ub = 2.0 } };
    mc<double> y = { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.5, .ub = 0.7 } };
    mc<double> z = { .cv = 0.2, .cc = 1.0, .box = { .lb = -1.0, .ub = 2.0 } };

    auto xy = x * y;
    assert(nextafter(xy.cv, 1) == 0.85);
    assert(nextafter(xy.cc, 0) == 1.0);

    auto xz = x * z;
    assert(nextafter(xz.cv, 1) == -0.3);
    assert(xz.cc == 2.0);

    auto yz = y * z;
    assert(nextafter(yz.cc, 1) == 0.8);

    auto xexp = x * exp(-pown(x, 2)); // correct cv,cc 9.124281806826588e-02, 4.061675774727018e-01
    printf("xexp.cv: %a, %.15f\n", xexp.cv, xexp.cv);
    printf("xexp.cc: %a, %.15f\n", xexp.cc, xexp.cc);

    // auto sincospow = sin(pown(y, -3)) * cos(pown(y, 2)); // -9.358968236779348e-01, 6.095699354841704e-01
    // printf("sincospow.cv: %a, %.15f\n", sincospow.cv, sincospow.cv);
    // printf("sincospow.cc: %a, %.15f\n", sincospow.cc, sincospow.cc);
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
