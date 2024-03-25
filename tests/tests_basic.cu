#include <cumccormick/arithmetic/basic.cuh>

#include <stdio.h>

__global__ void basic_kernel()
{
    mc<double> a = { .cv = 1.0, .cc = 2.0, .box = { .lb = 0.0, .ub = 3.0 } };
    mc<double> b = { .cv = 3.0, .cc = 4.0, .box = { .lb = 2.0, .ub = 5.0 } };

    auto print = [](mc<double> x) { printf("(cv: %.15g, cc: %.15g, box: [%g, %g])\n", x.cv, x.cc, x.box.lb, sup(x)); };

    // auto c = add(a, b);
    // print(c);
    auto d = sub(a, b);
    print(d);
    // auto e = mul(2.0, a);
    // print(e);
    auto f = sqr(d);
    print(f);
    // auto g = div(f, 2.0);
    // print(g);
    // auto h = exp(a);
    // print(h);
    // auto i = sqrt(b);
    // print(i);
    // auto j = cos(i);
    // print(j);
    // auto k = sin(j);
    // print(k);
}

void basic_kernel(cudaStream_t stream)
{
    basic_kernel<<<1, 1, 0, stream>>>();
}
