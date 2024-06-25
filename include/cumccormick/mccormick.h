#ifndef CUMCCORMICK_ARITHMETIC_MCCORMICK_H
#define CUMCCORMICK_ARITHMETIC_MCCORMICK_H

#include <cuinterval/interval.h>

#include <compare>

namespace cu
{

template<typename T, typename I = cu::interval<T>>
struct mccormick
{
    T cv;  // convex underestimation
    T cc;  // concave overestimation
    I box; // interval bounds

    constexpr auto operator<=>(const mccormick &) const = default;
};

} // namespace cu

#endif // CUMCCORMICK_ARITHMETIC_MCCORMICK_H
