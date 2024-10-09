#ifndef CUMCCORMICK_ARITHMETIC_COMPARE_CUH
#define CUMCCORMICK_ARITHMETIC_COMPARE_CUH

#include <cuinterval/cuinterval.h>
#include <cumccormick/mccormick.h>

namespace cu
{

template<typename T>
using mc = mccormick<T>;

#define cuda_fn inline constexpr __device__

template<typename T>
cuda_fn bool operator==(mc<T> a, mc<T> b)
{
    return a.cv == b.cv && a.cc == b.cc && a.box == b.box;
}

template<typename T>
cuda_fn bool operator!=(mc<T> a, mc<T> b)
{
    return a.cv != b.cv || a.cc != b.cc || a.box != b.box;
}

template<typename T>
cuda_fn bool operator>(mc<T> a, auto b)
{
    return a.box.lb > b;
}

template<typename T>
cuda_fn bool operator<(mc<T> a, auto b)
{
    return a.box.ub < b;
}

template<typename T>
cuda_fn bool operator>=(mc<T> a, mc<T> b)
{
    return inf(a) >= sup(b);
}

template<typename T>
cuda_fn bool operator<=(mc<T> a, mc<T> b)
{
    return inf(b) >= sup(a);
}

#undef cuda_fn

} // namespace cu
#endif // CUMCCORMICK_ARITHMETIC_COMPARE_CUH
