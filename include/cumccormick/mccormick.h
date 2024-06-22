#ifndef CUMCCORMICK_ARITHMETIC_MCCORMICK_H
#define CUMCCORMICK_ARITHMETIC_MCCORMICK_H

#include <cuinterval/interval.h>

#include <compare>

template<typename T, typename I = interval<T>>
struct mccormick
{
    T cv;   // convex underestimation
    T cc;   // concave overestimation
    I box;  // interval bounds

    constexpr auto operator<=>(const mccormick&) const = default; 
};

#endif // CUMCCORMICK_ARITHMETIC_MCCORMICK_H
