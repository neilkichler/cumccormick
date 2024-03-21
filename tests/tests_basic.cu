#include <cumccormick/arithmetic/basic.cuh>

__global__ void basic_kernel()
{
    mc<double> a = { .cv = 1.0, .cc = 2.0 };
    mc<double> b = { .cv = 3.0, .cc = 4.0 };
    mc<double> c = add(a, b);
    mc<double> d = sub(a, b);
    mc<double> e = mul(2.0, a);
}

void basic_kernel(cudaStream_t stream)
{
    basic_kernel<<<1, 1, 0, stream>>>();
}
