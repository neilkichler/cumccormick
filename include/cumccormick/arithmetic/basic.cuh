#ifndef CUMCCORMICK_ARITHMETIC_BASIC_CUH
#define CUMCCORMICK_ARITHMETIC_BASIC_CUH

#include <cuinterval/arithmetic/basic.cuh>
#include <cuinterval/arithmetic/intrinsic.cuh>

#include <algorithm>
#include <cmath>

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
inline __device__ mc<T> add(T a, mc<T> b)
{
    return { .cv  = intrinsic::add_down(a, b.cv),
             .cc  = intrinsic::add_up(a, b.cc),
             .box = a + b.box };
}

template<typename T>
inline __device__ mc<T> sub(mc<T> a, mc<T> b)
{
    return { .cv  = intrinsic::sub_down(a.cv, b.cc),
             .cc  = intrinsic::sub_up(a.cc, b.cv),
             .box = a.box - b.box };
}

template<typename T>
inline __device__ mc<T> sub(T a, mc<T> b)
{
    return { .cv  = intrinsic::sub_down(a, b.cc),
             .cc  = intrinsic::sub_up(a, b.cv),
             .box = a - b.box };
}

template<typename T>
inline __device__ mc<T> sub(mc<T> a, T b)
{
    return { .cv  = intrinsic::sub_down(a.cv, b),
             .cc  = intrinsic::sub_up(a.cc, b),
             .box = a.box - b };
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
inline __device__ T mid(T v, T lb, T ub)
{
    return std::clamp(v, lb, ub);
}

template<typename T>
inline __device__ mc<T> sqr(mc<T> x)
{
    using namespace intrinsic;

    T zmin;
    T zmax;

    T midcv;
    T midcc;
    constexpr auto zero = static_cast<T>(0);

    // TODO: maybe we should use x.cv and x.cc in the if statement?
    if (sup(x) < zero) {
        // zmin = sup(x);
        // zmax = inf(x);
        midcv = x.cc;
        midcc = x.cv;
    } else if (inf(x) > zero) {
        midcv = x.cv;
        midcc = x.cc;
        // zmin = inf(x);
        // zmax = sup(x);
    } else {
        midcv = mid(zero, x.cv, x.cc);
        midcc = (abs(inf(x)) >= abs(sup(x))) ? x.cv : x.cc;
        // zmin = zero;
        // zmax = (abs(inf(x)) >= abs(sup(x))) ? inf(x) : sup(x);
    }

    // T midcv = mid(zmin, x.cv, x.cc);
    // T midcc = mid(zmax, x.cv, x.cc);

    T cc = sub_up(mul_up(add_up(inf(x), sup(x)), midcc), mul_up(inf(x), sup(x)));
    return { .cv  = mul_down(midcv, midcv),
             .cc  = cc,
             .box = sqr(x.box) };
}

template<typename T>
inline __device__ mc<T> exp(mc<T> x)
{
    using namespace intrinsic;

    // computing secant over interval endpoints
    T r  = is_singleton(x.box)
         ? static_cast<T>(0)
         : div_up(sub_up(exp(sup(x)), exp(inf(x))), (sub_down(sup(x), inf(x))));

    T cc = add_up(exp(sup(x)), mul_up(r, sub_up(x.cc, sup(x))));

    return { .cv  = intrinsic::next_after(exp(x.cv), static_cast<T>(0)),
             .cc  = cc,
             .box = exp(x.box) };
}

template<typename T>
inline __device__ mc<T> sqrt(mc<T> x)
{
    using namespace intrinsic;

    // computing secant over interval endpoints
    T r  = is_singleton(x.box)
         ? static_cast<T>(0)
         : div_down(sub_down(sqrt(sup(x)), sqrt(inf(x))), (sub_down(sup(x), inf(x))));

    T midcv = mid(inf(x), x.cv, x.cc);
    T midcc = mid(sup(x), x.cv, x.cc);
    T cv = add_down(sqrt_down(inf(x)), mul_down(r, sub_down(midcv, inf(x))));

    return { .cv  = cv,
             .cc  = intrinsic::sqrt_up(midcc),
             .box = sqrt(x.box) };
}

template<typename T>
inline __device__ mc<T> operator+(mc<T> a, mc<T> b)
{
    return add(a, b);
}

template<typename T>
inline __device__ mc<T> operator+(T a, mc<T> b)
{
    return add(a, b);
}

template<typename T>
inline __device__ mc<T> operator+(mc<T> a, T b)
{
    return add(b, a);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a, mc<T> b)
{
    return sub(a, b);
}

template<typename T>
inline __device__ mc<T> operator-(T a, mc<T> b)
{
    return sub(a, b);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a, T b)
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

template<typename T>
inline __device__ mc<T> cos(mc<T> x)
{
    using namespace intrinsic;

    interval<T> cos_box = cos(x.box);

    return { .cv  = inf(cos_box),
             .cc  = sup(cos_box),
             .box = cos_box };
}

template<typename T>
inline __device__ mc<T> sin(mc<T> x)
{
    using namespace intrinsic;

    return cos(x - static_cast<T>(M_PI_2));
}

#endif // CUMCCORMICK_ARITHMETIC_BASIC_CUH
