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

    // support designated initializer construction
    struct initializer
    {
        T lb;
        T cv;
        T cc;
        T ub;
    };

    struct init_with_box
    {
        T cv;
        T cc;
        I box;
    };

    constexpr mccormick() = default;
    constexpr mccormick(T p)                    : cv(p)       , cc(p)       , box(p)                { }
    constexpr mccormick(T cv, T cc)             : cv(cv)      , cc(cc)      , box(cv, cc)           { }
    constexpr mccormick(T x, I box)             : cv(x)       , cc(x)       , box(box)              { }
    constexpr mccormick(T cv, T cc, I box)      : cv(cv)      , cc(cc)      , box(box)              { }
    constexpr mccormick(T lb, T cv, T cc, T ub) : cv(cv)      , cc(cc)      , box(lb, ub)           { }
    constexpr mccormick(initializer init)       : cv(init.cv) , cc(init.cc) , box(init.lb, init.ub) { }
    constexpr mccormick(init_with_box init)     : cv(init.cv) , cc(init.cc) , box(init.box)         { }

    constexpr bool operator==(const mccormick &) const = default;
};

} // namespace cu

#endif // CUMCCORMICK_ARITHMETIC_MCCORMICK_H
