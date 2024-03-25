#include <cumccormick/arithmetic/basic.cuh>

__global__ void basic_kernel()
{
    mc<double> a = { .cv = 1.0, .cc = 2.0 };
    mc<double> b = { .cv = 3.0, .cc = 4.0 };
    auto c = add(a, b);
    auto d = sub(a, b);
    auto e = mul(2.0, a);
    auto f = sqr(e);
    auto g = div(f, 2.0);
    auto h = exp(g);
    auto i = sqrt(h);
    auto j = cos(i);
    auto k = sin(j);
}

void basic_kernel(cudaStream_t stream)
{
    basic_kernel<<<1, 1, 0, stream>>>();
}
