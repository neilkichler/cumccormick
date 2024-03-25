#ifndef CUMCCORMICK_ARITHMETIC_BASIC_CUH
#define CUMCCORMICK_ARITHMETIC_BASIC_CUH

#include <cuinterval/arithmetic/basic.cuh>
#include <cuinterval/arithmetic/intrinsic.cuh>

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
inline __device__ mc<T> div(mc<T> a, T b)
{
    bool is_neg = b < static_cast<T>(0);
    return { .cv  = intrinsic::div_down(is_neg ? a.cc : a.cv, b),
             .cc  = intrinsic::div_up(is_neg ? a.cv : a.cc, b),
             .box = a.box / b };
}

template<typename T>
inline __device__ T inf(mc<T> x)
{
    return inf(x.box);
}

template<typename T>
inline __device__ T sup(mc<T> x)
{
    return sup(x.box);
}

template<typename T>
inline __device__ mc<T> sqr(mc<T> x)
{
    // since sqr is convex we do not have to find the midpoints.
    // return { .cv  = sqr(x.cv, x.cv),
    //          .cc  = (inf(x.box) + sup(x.box)) * x.cc - inf(x.box) * sup(x.box),
    //          .box = sqr(x.box) };

    using namespace intrinsic;

    T cc = sub_up(mul_up(add_up(inf(x), sup(x)), x.cc), mul_up(inf(x), sup(x)));
    return { .cv  = mul_down(x.cv, x.cv),
             .cc  = cc,
             .box = sqr(x.box) };
}

template<typename T>
inline __device__ mc<T> exp(mc<T> x)
{
    using namespace intrinsic;

    // We are computing a secant here
    T r  = is_singleton(x.box)
         ? static_cast<T>(0)
         : div_up(sub_up(exp(sup(x)), exp(inf(x))), (sub_down(sup(x), inf(x))));
    T cc = add_up(exp(inf(x)), mul_up(r, exp(x.cc)));

    return { .cv  = intrinsic::next_after(exp(x.cv), static_cast<T>(0)),
             .cc  = cc,
             .box = exp(x.box) };
}

template<typename T>
inline __device__ mc<T> sqrt(mc<T> x)
{
    using namespace intrinsic;

    T r  = is_singleton(x.box)
         ? static_cast<T>(0)
         : div_up(sub_up(sqrt(sup(x)), sqrt(inf(x))), (sub_down(sup(x), inf(x))));
    T cv = add_down(sqrt_down(inf(x)), mul_down(r, sqrt_down(x.cv)));

    return { .cv  = cv,
             .cc  = intrinsic::sqrt_up(x.cc),
             .box = sqrt(x.box) };
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

template<typename T>
inline __device__ mc<T> operator/(mc<T> a, T b)
{
    return div(a, b);
}

#endif // CUMCCORMICK_ARITHMETIC_BASIC_CUH
