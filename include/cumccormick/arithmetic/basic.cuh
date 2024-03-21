#ifndef CUMCCORMICK_ARITHMETIC_BASIC_CUH
#define CUMCCORMICK_ARITHMETIC_BASIC_CUH

#include <cuinterval/arithmetic/intrinsic.cuh>
#include <cuinterval/arithmetic/basic.cuh>

#include "mccormick.h"

template<typename T>
using mc = mccormick<T>;

template<typename T>
inline __device__ mc<T> add(mc<T> a, mc<T> b)
{
    return { .cv  = intrinsic::add_down(a.cv, b.cv),
             .cc  = intrinsic::add_up(a.cc, b.cc),
             .box = a.box + b.box };
}

template<typename T>
inline __device__ mc<T> sub(mc<T> a, mc<T> b)
{
    return { .cv  = intrinsic::sub_down(a.cv, b.cc),
             .cc  = intrinsic::sub_up(a.cc, b.cv),
             .box = a.box - b.box };
}

template<typename T>
inline __device__ mc<T> mul(T a, mc<T> b)
{
    bool is_neg = a < static_cast<T>(0);
    return { .cv  = intrinsic::mul_down(a, is_neg ? b.cc : b.cv),
             .cc  = intrinsic::mul_up(a, is_neg ? b.cv : b.cc),
             .box = a * b.box };
}

template<typename T>
inline __device__ mc<T> operator+(mc<T> a, mc<T> b)
{
    return add(a, b);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a, mc<T> b)
{
    return sub(a, b);
}

template<typename T>
inline __device__ mc<T> operator*(T a, mc<T> b)
{
    return mul(a, b);
}

template<typename T>
inline __device__ mc<T> operator*(mc<T> a, T b)
{
    return mul(b, a);
}

#endif // CUMCCORMICK_ARITHMETIC_BASIC_CUH
