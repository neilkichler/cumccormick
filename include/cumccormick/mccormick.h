#ifndef CUMCCORMICK_ARITHMETIC_MCCORMICK_H
#define CUMCCORMICK_ARITHMETIC_MCCORMICK_H

#include <cuinterval/interval.h>

namespace cu
{

template<typename T, typename I = cu::interval<T>>
struct mccormick
{
    using value_type = T;

    T cv;  // convex underestimation
    T cc;  // concave overestimation
    I box; // interval bounds

    constexpr bool operator==(const mccormick &) const = default;

    constexpr mccormick &operator=(T value)
    {
        cv  = value;
        cc  = value;
        box = { value, value };
        return *this;
    }
};

} // namespace cu

#endif // CUMCCORMICK_ARITHMETIC_MCCORMICK_H
