#ifndef CUMCCORMICK_ARITHMETIC_BASIC_CUH
#define CUMCCORMICK_ARITHMETIC_BASIC_CUH

#include <cuinterval/cuinterval.h>
#include <cumccormick/mccormick.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>

namespace cu
{

template<typename T>
using mc = mccormick<T>;

template<typename T>
concept Number = std::is_arithmetic_v<T>;

#define cuda_fn inline constexpr __device__

cuda_fn auto value(Number auto x) { return x; }

// This functions clips the convex and concave relaxation to the interval.
// Sometimes the McCormick relaxation turns out worse than the interval bounds.
//
// E.g.,
//          f(x,y) = cos(x)*cos(y) with x,y=[-42, (0, 0), 42]
//
//          returns     f(x,y)  = [-1, (-3, 3), 1],
//          whereas cut(f(x,y)) = [-1, (-1, 1), 1].
//
template<typename T, bool CutActive = true>
cuda_fn mc<T> cut(mc<T> a)
{
    if constexpr (CutActive) {
        using std::max;
        using std::min;

        return { { .cv  = max(a.cv, inf(a)),
                   .cc  = min(a.cc, sup(a)),
                   .box = a.box } };
    } else {
        return a;
    }
}

template<typename T>
cuda_fn T mid(T v, T lb, T ub)
{
    using std::clamp;

    return clamp(v, lb, ub);
}

template<typename T>
cuda_fn mc<T> pos(mc<T> x)
{
    return x;
}

template<typename T>
cuda_fn mc<T> neg(mc<T> x)
{
    return { { .cv  = -x.cc,
               .cc  = -x.cv,
               .box = -x.box } };
}

template<typename T>
cuda_fn mc<T> add(mc<T> a, mc<T> b)
{
    return { { .cv  = intrinsic::add_down(a.cv, b.cv),
               .cc  = intrinsic::add_up(a.cc, b.cc),
               .box = a.box + b.box } };
}

template<typename T>
cuda_fn mc<T> add(T a, mc<T> b)
{
    return { { .cv  = intrinsic::add_down(a, b.cv),
               .cc  = intrinsic::add_up(a, b.cc),
               .box = a + b.box } };
}

template<typename T>
cuda_fn mc<T> sub(mc<T> a, mc<T> b)
{
    return { { .cv  = intrinsic::sub_down(a.cv, b.cc),
               .cc  = intrinsic::sub_up(a.cc, b.cv),
               .box = a.box - b.box } };
}

template<typename T>
cuda_fn mc<T> sub(T a, mc<T> b)
{
    return { { .cv  = intrinsic::sub_down(a, b.cc),
               .cc  = intrinsic::sub_up(a, b.cv),
               .box = a - b.box } };
}

template<typename T>
cuda_fn mc<T> sub(mc<T> a, T b)
{
    return { { .cv  = intrinsic::sub_down(a.cv, b),
               .cc  = intrinsic::sub_up(a.cc, b),
               .box = a.box - b } };
}

template<typename T>
cuda_fn mc<T> mul(T a, mc<T> b)
{
    bool is_neg = a < static_cast<T>(0);
    return { { .cv  = intrinsic::mul_down(a, is_neg ? b.cc : b.cv),
               .cc  = intrinsic::mul_up(a, is_neg ? b.cv : b.cc),
               .box = a * b.box } };
}

template<typename T>
cuda_fn mc<T> mul_nearest_even_rounding(mc<T> a, mc<T> b)
{
    using namespace intrinsic;

    T alpha1 = min(inf(b) * a.cv, inf(b) * a.cc);
    T alpha2 = min(inf(a) * b.cv, inf(a) * b.cc);
    T beta1  = min(sup(b) * a.cv, sup(b) * a.cc);
    T beta2  = min(sup(a) * b.cv, sup(a) * b.cc);

    T gamma1 = max(inf(b) * a.cv, inf(b) * a.cc);
    T delta2 = max(inf(a) * b.cv, inf(a) * b.cc);
    T delta1 = max(sup(b) * a.cv, sup(b) * a.cc);
    T gamma2 = max(sup(a) * b.cv, sup(a) * b.cc);

    T cv = max(alpha1 + alpha2 - inf(a) * inf(b),
               beta1 + beta2 - sup(a) * sup(b));

    T cc = min(gamma1 + gamma2 - sup(a) * inf(b),
               delta1 + delta2 - inf(a) * sup(b));

    return { { .cv  = cv,
               .cc  = cc,
               .box = mul(a.box, b.box) } };
}

template<typename T>
cuda_fn mc<T> mul(mc<T> a, mc<T> b)
{
    using namespace intrinsic;

    T alpha1 = min(mul_down(inf(b), a.cv), mul_down(inf(b), a.cc));
    T alpha2 = min(mul_down(inf(a), b.cv), mul_down(inf(a), b.cc));
    T beta1  = min(mul_down(sup(b), a.cv), mul_down(sup(b), a.cc));
    T beta2  = min(mul_down(sup(a), b.cv), mul_down(sup(a), b.cc));

    T gamma1 = max(mul_up(inf(b), a.cv), mul_up(inf(b), a.cc));
    T gamma2 = max(mul_up(sup(a), b.cv), mul_up(sup(a), b.cc));
    T delta1 = max(mul_up(sup(b), a.cv), mul_up(sup(b), a.cc));
    T delta2 = max(mul_up(inf(a), b.cv), mul_up(inf(a), b.cc));

    // T cv = max(sub_down(add_down(alpha1, alpha2), mul_down(inf(a), inf(b))),
    //            sub_down(add_down(beta1, beta2), mul_down(sup(a), sup(b))));
    //
    // T cc = min(sub_up(add_up(gamma1, gamma2), mul_up(sup(a), inf(b))),
    //            sub_up(add_up(delta1, delta2), mul_up(inf(a), sup(b))));

    // using fmas we have:
    T cv = max(fma_down(-inf(a), inf(b), add_down(alpha1, alpha2)),
               fma_down(-sup(a), sup(b), add_down(beta1, beta2)));

    T cc = min(fma_up(-sup(a), inf(b), add_up(gamma1, gamma2)),
               fma_up(-inf(a), sup(b), add_up(delta1, delta2)));

    return cut<T>({ cv, cc, mul(a.box, b.box) });
}

template<typename T>
cuda_fn T chord_of_concave(T x, T lb, T ub, T f_lb, T f_ub)
{
    // computing chord (line segment) over interval endpoints

    // NOTE: Only x should be an active variable in the function when using forward-mode AD.
    return [lb = value(lb), ub = value(ub), f_lb = value(f_lb), f_ub = value(f_ub)](T x) {
        using namespace intrinsic;
        using value_type = decltype(lb);

        value_type slope = lb == ub ? static_cast<value_type>(0)
                                    : div_down(sub_down(f_ub, f_lb), (sub_down(ub, lb)));

        return add_down(f_lb, mul_down(slope, sub_down(x, lb)));
    }(x);
}

template<typename T>
cuda_fn T chord_of_convex(T x, T lb, T ub, T f_lb, T f_ub)
{
    // computing chord (line segment) over interval endpoints

    // NOTE: Only x should be an active variable in the function when using forward-mode AD.
    return [lb = value(lb), ub = value(ub), f_lb = value(f_lb), f_ub = value(f_ub)](T x) {
        using namespace intrinsic;
        using value_type = decltype(lb);

        value_type slope = lb == ub ? static_cast<value_type>(0)
                                    : div_up(sub_up(f_ub, f_lb), (sub_up(ub, lb)));

        return add_up(f_ub, mul_up(slope, sub_up(x, ub)));
    }(x);
}

template<typename T>
cuda_fn mc<T> recip(mc<T> x)
{
    using namespace intrinsic;

    constexpr auto zero = static_cast<T>(0);

    T cv;
    T cc;
    T midcv = mid(sup(x), x.cv, x.cc);
    T midcc = mid(inf(x), x.cv, x.cc);

    if (contains(x.box, zero)) {
        if (inf(x) < zero && zero == sup(x)) {
            cv = intrinsic::neg_inf<T>();
            cc = rcp_up(midcc);
        } else if (inf(x) == zero && zero < sup(x)) {
            cv = rcp_down(midcv);
            cc = intrinsic::pos_inf<T>();
        } else if (inf(x) < zero && zero < sup(x)) {
            return { intrinsic::neg_inf<T>(),
                     intrinsic::pos_inf<T>(),
                     entire<T>() };
        } else if (inf(x) == zero && zero == sup(x)) {
            // NOTE: Alternatively, we could return nans.
            return {
                { .cv  = intrinsic::neg_inf<T>(),
                  .cc  = intrinsic::pos_inf<T>(),
                  .box = entire<T>() }
            };
        }
    } else if (sup(x) < zero) {
        // for x < 0, recip is concave
        cv = chord_of_concave(midcv, inf(x), sup(x), rcp_down(inf(x)), rcp_up(sup(x)));
        cc = rcp_up(midcc);
    } else { // inf(x) > zero
        // for x > 0, recip is convex
        cc = chord_of_convex(midcc, inf(x), sup(x), rcp_down(inf(x)), rcp_up(sup(x)));
        cv = rcp_down(midcv);
    }

    // TODO: relaxation could be more efficient if we embed the IA into the different cases.
    return { { .cv  = cv,
               .cc  = cc,
               .box = recip(x.box) } };
}

template<typename T>
cuda_fn mc<T> div(mc<T> a, T b)
{
    using namespace intrinsic;

    bool is_neg = b < static_cast<T>(0);

    return { { .cv  = div_down(is_neg ? a.cc : a.cv, b),
               .cc  = div_up(is_neg ? a.cv : a.cc, b),
               .box = a.box / b } };
}

template<typename T>
cuda_fn mc<T> div(mc<T> a, mc<T> b)
{
    // TODO: implement tighter relaxation
    return mul(a, recip(b));
}

template<typename T>
cuda_fn T inf(mc<T> x)
{
    return inf(x.box);
}

template<typename T>
cuda_fn T sup(mc<T> x)
{
    return sup(x.box);
}

template<typename T>
cuda_fn mc<T> sqr(mc<T> x)
{
    using namespace intrinsic;
    using std::abs;

    T midcv;
    T midcc;
    constexpr auto zero = static_cast<T>(0);

    // TODO: maybe we should use x.cv and x.cc in the if statement?
    if (sup(x) <= zero) {
        midcv = x.cc;
        midcc = x.cv;
    } else if (inf(x) >= zero) {
        midcv = x.cv;
        midcc = x.cc;
    } else {
        midcv = mid(zero, x.cv, x.cc);
        midcc = (abs(inf(x)) >= abs(sup(x))) ? x.cv : x.cc;
    }

    T cc = sub_up(mul_up(add_up(inf(x), sup(x)), midcc), mul_down(inf(x), sup(x)));
    return { { .cv  = mul_down(midcv, midcv),
               .cc  = cc,
               .box = sqr(x.box) } };
}

template<typename T>
cuda_fn mc<T> abs(mc<T> x)
{
    using std::abs;

    T xmin  = mid(static_cast<T>(0), inf(x), sup(x));
    T midcv = mid(xmin, x.cv, x.cc);

    T xmax  = abs(inf(x)) >= abs(sup(x)) ? inf(x) : sup(x);
    T midcc = mid(xmax, x.cv, x.cc);

    T cc = chord_of_convex(midcc, inf(x), sup(x), abs(inf(x)), abs(sup(x)));
    return { { .cv  = abs(midcv),
               .cc  = cc,
               .box = abs(x.box) } };
}

template<typename T>
cuda_fn mc<T> fabs(mc<T> x)
{
    return abs(x);
}

template<typename T>
cuda_fn mc<T> exp(mc<T> x)
{
    using namespace intrinsic;
    using std::exp;

    // TODO: error in exp not accounted for in secant computation
    T cc = exp(sup(x)) == intrinsic::pos_inf<T>()
        ? intrinsic::pos_inf<T>()
        : chord_of_convex(x.cc, inf(x), sup(x), exp(inf(x)), exp(sup(x)));

    return { { .cv  = next_after(exp(x.cv), static_cast<T>(0)),
               .cc  = cc,
               .box = exp(x.box) } };
}

template<typename T>
cuda_fn mc<T> sqrt(mc<T> x)
{
    using namespace intrinsic;
    T midcv = mid(inf(x), x.cv, x.cc);
    T midcc = mid(sup(x), x.cv, x.cc);
    T cv    = chord_of_concave(midcv, inf(x), sup(x), sqrt_down(inf(x)), sqrt_up(sup(x)));

    return { { .cv  = cv,
               .cc  = sqrt_up(midcc),
               .box = sqrt(x.box) } };
}

template<typename T>
cuda_fn mc<T> pown_even(mc<T> x, Number auto n)
{
    using namespace intrinsic;
    using std::abs;

    auto pow = [](T x, std::integral auto n) -> T {
        // The default std::pow implementation returns a double for std::pow(float, int). We want a float.
        if constexpr (std::is_same_v<T, float>) {
            return powf(x, n);
        } else {
            using std::pow;
            return pow(x, n);
        }
    };

    T midcv;
    T midcc;
    constexpr auto zero = static_cast<T>(0);

    T cc;
    // TODO: Why do we check against sup(x) and inf(x) instead of x.cc and x.cv?
    if (n > 0) {
        if (sup(x) <= zero) {
            midcv = x.cc;
            midcc = x.cv;
        } else if (inf(x) >= zero) {
            midcv = x.cv;
            midcc = x.cc;
        } else {
            midcv = mid(zero, x.cv, x.cc);
            midcc = (abs(inf(x)) >= abs(sup(x))) ? x.cv : x.cc;
        }
    } else {
        if (sup(x) <= zero) {
            midcv = x.cv;
            midcc = x.cc;
        } else if (inf(x) >= zero) {
            midcv = x.cc;
            midcc = x.cv;
        } else {
            midcv = (abs(inf(x)) >= abs(sup(x))) ? x.cv : x.cc;
            return {
                { .cv  = pow(midcv, n),
                  .cc  = intrinsic::pos_inf<T>(),
                  .box = pown(x.box, n) }
            };
        }
    }

    // TODO: floating point error in pow not accounted for
    cc = chord_of_convex(midcc, inf(x), sup(x), pow(inf(x), n), pow(sup(x), n));

    return { { .cv  = pow(midcv, n),
               .cc  = cc,
               .box = pown(x.box, n) } };
}

template<typename T>
cuda_fn mc<T> pown(mc<T> x, Number auto n)
{
    using namespace intrinsic;

    auto pow = [](T x, std::integral auto n) -> T {
        // The default std::pow implementation returns a double for std::pow(float, int). We want a float.
        if constexpr (std::is_same_v<T, float>) {
            return powf(x, n);
        } else {
            using std::pow;
            return pow(x, n);
        }
    };

    T cv;
    T cc;

    if (n == 0) {
        constexpr auto one = static_cast<T>(1);
        return { one };
    } else if (n == 1) {
        return x;
    } else if (n == 2) {
        return sqr(x); // TODO: could be merged with even power case
    }

    // TODO: n < 0 not considered yet
    if (n % 2) { // odd power
        constexpr auto zero = static_cast<T>(0);

        // TODO: not accounting for pow(x,n) error (2 ulps)
        if (sup(x) <= zero) {
            // for x < 0, pown(x,n_odd) is concave
            cv = chord_of_concave(n > 0 ? x.cv : x.cc, inf(x), sup(x), pow(inf(x), n), pow(sup(x), n));
            cc = pow(n > 0 ? x.cc : x.cv, n);
        } else if (inf(x) >= zero) {
            // for x > 0, pown(x,n_odd) is convex
            cv = pow(n > 0 ? x.cv : x.cc, n);
            cc = chord_of_convex(n > 0 ? x.cc : x.cv, inf(x), sup(x), pow(inf(x), n), pow(sup(x), n));
        } else {
            // for 0 in x, pown(x,n_odd) is concavoconvex
            if (n > 0) {
                // differentiable variant
                cv = pow(inf(x), n) * ((sup(x) - x.cv) / (sup(x) - inf(x))) + pow(max(zero, x.cv), n);
                cc = pow(sup(x), n) * ((x.cc - inf(x)) / (sup(x) - inf(x))) + pow(min(zero, x.cc), n);
            } else {
                return {
                    { .cv  = intrinsic::pos_inf<T>(),
                      .cc  = intrinsic::neg_inf<T>(),
                      .box = entire<T>() }
                };
            }
        }
    } else { // even power
        return pown_even(x, n);
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = pown(x.box, n) } };
}

template<typename T>
cuda_fn mc<T> pow(mc<T> x, Number auto n)
{
    return pown(x, n);
}

template<typename T>
cuda_fn mc<T> operator+(mc<T> x)
{
    return x;
}

template<typename T>
cuda_fn mc<T> operator+(mc<T> a, mc<T> b)
{
    return add(a, b);
}

template<typename T>
cuda_fn mc<T> operator+(T a, mc<T> b)
{
    return add(a, b);
}

template<typename T>
cuda_fn mc<T> operator+(Number auto a, mc<T> b)
{
    return add(static_cast<T>(a), b);
}

template<typename T>
cuda_fn mc<T> operator+(mc<T> a, T b)
{
    return add(b, a);
}

template<typename T>
cuda_fn mc<T> operator+(mc<T> a, Number auto b)
{
    return add(static_cast<T>(b), a);
}

template<typename T>
cuda_fn mc<T> operator-(mc<T> a)
{
    return neg(a);
}

template<typename T>
cuda_fn mc<T> operator-(mc<T> a, mc<T> b)
{
    return sub(a, b);
}

template<typename T>
cuda_fn mc<T> operator-(T a, mc<T> b)
{
    return sub(a, b);
}

template<typename T>
cuda_fn mc<T> operator-(Number auto a, mc<T> b)
{
    return sub(static_cast<T>(a), b);
}

template<typename T>
cuda_fn mc<T> operator-(mc<T> a, T b)
{
    return sub(a, b);
}

template<typename T>
cuda_fn mc<T> operator-(mc<T> a, Number auto b)
{
    return sub(a, static_cast<T>(b));
}

template<typename T>
cuda_fn mc<T> operator*(T a, mc<T> b)
{
    return mul(a, b);
}

template<typename T>
cuda_fn mc<T> operator*(Number auto a, mc<T> b)
{
    return mul(static_cast<T>(a), b);
}

template<typename T>
cuda_fn mc<T> operator*(mc<T> a, T b)
{
    return mul(b, a);
}

template<typename T>
cuda_fn mc<T> operator*(mc<T> a, Number auto b)
{
    return mul(static_cast<T>(b), a);
}

template<typename T>
cuda_fn mc<T> operator*(mc<T> a, mc<T> b)
{
    return mul(a, b);
}

template<typename T>
cuda_fn mc<T> operator/(mc<T> a, T b)
{
    return div(a, b);
}

template<typename T>
cuda_fn mc<T> operator/(mc<T> a, Number auto b)
{
    return div(a, static_cast<T>(b));
}

template<typename T>
cuda_fn mc<T> operator/(mc<T> a, mc<T> b)
{
    return div(a, b);
}

template<typename T>
cuda_fn mc<T> &operator+=(mc<T> &a, auto b)
{
    a = a + b;
    return a;
}

template<typename T>
cuda_fn mc<T> &operator-=(mc<T> &a, auto b)
{
    a = a - b;
    return a;
}

template<typename T>
cuda_fn mc<T> &operator*=(mc<T> &a, auto b)
{
    a = a * b;
    return a;
}

template<typename T>
cuda_fn mc<T> &operator/=(mc<T> &a, auto b)
{
    a = a / b;
    return a;
}

template<typename T>
struct solver_options
{
    int maxiter { 50 };
    T atol { 1e-10 };
    T rtol { 0.0 };
};

template<typename T>
struct root_solver_state
{
    T x;
    T lb;
    T ub;
    T y;
    T dydx;
};

template<typename T>
cuda_fn T root(auto &&f, auto &&step, T x0, T lb, T ub, solver_options<T> options = {})
{
    using std::abs;
    using std::signbit;

    if (f(lb) * f(ub) > 0.0) {
        return x0;
    }

    T x       = mid(x0, lb, ub);
    T delta_x = intrinsic::pos_inf<T>();

    auto terminate = [options](auto f_error, auto x, auto x_prev, Number auto i) {
        auto [maxiter, atol, rtol] = options;
        auto scaled_tol            = atol + rtol * abs(x); // alternative: max(atol, rtol * abs(x));
        bool x_small               = abs(x - x_prev) < scaled_tol;
        bool f_small               = abs(f_error) < atol;
        return (x_small && f_small) || i >= maxiter;
    };

    for (int i = 0;; i++) {
        auto state                      = root_solver_state<T> { x, lb, ub };
        auto [x_new, lb_, ub_, y, dydx] = step(state, delta_x);

        // If the derivative step wants to push us in the direction of the boundary
        // we are currently at (instead of into the interval), we can directly stop the search.
        //
        // E.g.: x = 1.0
        //       y = 42.0
        //       dydx = -1.0
        //       bounds [0.0, 1.0]
        //
        //       would lead to x_new = x + 42 = 1.0 (bounded) in the next step.
        if ((x == lb && signbit(y) == signbit(dydx)) || (x == ub && signbit(y) != signbit(dydx))) {
            return x;
        }

        lb    = lb_;
        ub    = ub_;
        x_new = mid(x_new, lb, ub);

        if (terminate(y, x_new, x, i)) {
            return x_new;
        }

        // tighten bounds of the root (used in bisection)
        if (signbit(f(lb)) == signbit(f(x_new))) {
            // lb and x_new have same signs
            lb = x_new;
        } else {
            ub = x_new;
        }

        delta_x = abs(x_new - x);
        x       = x_new;
    }
    return x;
}

template<typename T>
cuda_fn auto derivative_or_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&step_fn, T epsilon = 1e-30)
{
    using std::abs;

    auto [x, lb, ub, y, dydx] = state;

    y    = f(x);
    dydx = df(x);

    bool too_slow_progress = abs(2.0 * y) >= abs(delta_x * dydx);

    T x_new;
    if (abs(dydx) < epsilon || too_slow_progress) {
        // perform bisection (twice) if dydx is close to zero.
        auto f_lb = f(lb);
        auto c    = lb + 0.5 * (ub - lb); // is eqv. to (lb + ub) / 2.0 with potentially better roundoff.
        auto f_c  = f(c);

        if (f_lb * f_c > 0.0) { // f_lb and f_c have same signs
            lb = c;
        } else { // f_ub and f_c have same signs
            ub = c;
        }

        x_new = lb + 0.5 * (ub - lb);
    } else {
        // perform update with bounded newton step
        T step = step_fn(x, y);
        x_new  = x - step;
        x_new  = mid(x_new, lb, ub);
    }

    return root_solver_state { x_new, lb, ub, y, dydx };
}

cuda_fn auto newton_step(auto x, auto y, auto &&df)
{
    return y / df(x);
}

template<typename T>
cuda_fn auto newton_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df)
{
    auto [x, lb, ub, y, dydx] = state;
    y                         = f(x);
    x                         = x - newton_step(x, y, df);
    return root_solver_state { x, lb, ub, y };
}

cuda_fn auto halley_step(auto x, auto y, auto &&df, auto &&ddf)
{
    using std::pow;

    return (2.0 * y * df(x)) / (2.0 * pow(df(x), 2) - y * ddf(x));
}

template<typename T>
cuda_fn auto halley_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df, auto &&ddf)
{
    auto [x, lb, ub, y, dydx] = state;
    y                         = f(x);
    x                         = x - halley_step(x, y, df, ddf);
    return root_solver_state { x, lb, ub, y };
}

cuda_fn auto householder_step(auto x, auto y, auto &&df, auto &&ddf, auto &&dddf)
{
    using std::pow;

    auto dy   = df(x);
    auto ddy  = ddf(x);
    auto dddy = dddf(x);
    return (6.0 * y * pow(dy, 2) - 3.0 * pow(y, 2) * ddy)
        / (6.0 * pow(dy, 3) - 6.0 * y * dy * ddy + pow(y, 2) * dddy);
}

template<typename T>
cuda_fn auto householder_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df, auto &&ddf, auto &&dddf)
{
    auto [x, lb, ub, y, dydx] = state;
    y                         = f(x);
    x                         = x - householder_step(x, y, df, ddf, dddf);
    return root_solver_state { x, lb, ub, y };
}

template<typename T>
cuda_fn auto newton_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df)
{
    auto step_fn = [df](auto x, auto y) { return newton_step(x, y, df); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<typename T>
cuda_fn auto halley_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&ddf)
{
    auto step_fn = [df, ddf](auto x, auto y) { return halley_step(x, y, df, ddf); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<typename T>
cuda_fn auto householder_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&ddf, auto &&dddf)
{
    auto step_fn = [df, ddf, dddf](auto x, auto y) { return householder_step(x, y, df, ddf, dddf); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<typename T>
cuda_fn T root_newton(auto &&f, auto &&df, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df](auto x, auto delta_x) { return newton_step(x, delta_x, f, df); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn T root_newton_bisection(auto &&f, auto &&df, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df](auto x, auto delta_x) { return newton_bisection_step(x, delta_x, f, df); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn T root_halley(auto &&f, auto &&df, auto &&ddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf](auto x, auto delta_x) { return halley_step(x, delta_x, f, df, ddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn T root_halley_bisection(auto &&f, auto &&df, auto &&ddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf](auto x, auto delta_x) { return halley_bisection_step(x, delta_x, f, df, ddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn T root_householder(auto &&f, auto &&df, auto &&ddf, auto &&dddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf, dddf](auto x, auto delta_x) { return householder_step(x, delta_x, f, df, ddf, dddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn T root_householder_bisection(auto &&f, auto &&df, auto &&ddf, auto &&dddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf, dddf](auto x, auto delta_x) { return householder_bisection_step(x, delta_x, f, df, ddf, dddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
cuda_fn mc<T> cos(mc<T> x)
{
    using namespace intrinsic;
    using std::abs;
    using std::ceil;
    using std::cos;
    using std::floor;
    using std::sin;

    // TODO: use rounded ops

    T argmin      = {};
    T argmax      = {};
    T pi          = std::numbers::pi;
    T k           = ceil(-0.5 - inf(x) / (2.0 * pi));
    T two_pi_k_lb = 2.0 * pi * k;
    T x_lb        = inf(x);
    T x_ub        = sup(x);
    T one { 1.0 };

    // We center the x around the interval [-pi, pi]
    T x_lb_centered = x_lb + two_pi_k_lb;
    T x_ub_centered = x_ub + two_pi_k_lb;

    // find argmin and argmax for midcc/midcv calculcation
    if (x_lb_centered <= 0) {
        // cos increases
        if (x_ub_centered <= 0) {
            // cos monotonically increases in interval
            argmin = x_lb;
            argmax = x_ub;
        } else if (x_ub_centered >= pi) {
            // more than one period
            argmin = pi - two_pi_k_lb;
            argmax = -two_pi_k_lb;
        } else {
            // increasing then decreasing
            argmin = (cos(x_lb_centered) <= cos(x_ub_centered)) ? x_lb : x_ub; // take smallest of the two endpoints
            argmax = -two_pi_k_lb;                                             // peak at period of cos and thus cos(argmax)=1
        }
    } else {
        // cos decreases
        if (x_ub_centered <= pi) {
            // cos monotonically decreases in interval
            argmin = x_ub;
            argmax = x_lb;
        } else if (x_ub_centered >= 2.0 * pi) {
            // more than one period
            argmin = pi - two_pi_k_lb;
            argmax = 2.0 * pi * (1.0 - k);
        } else {
            // decreasing then increasing
            argmin = pi - two_pi_k_lb;
            argmax = (cos(x_lb_centered) >= cos(x_ub_centered)) ? x_lb : x_ub;
        }
    }

    T midcv = mid(argmin, x.cv, x.cc);
    T midcc = mid(argmax, x.cv, x.cc);

    auto cv_cos = [x_lb, x_ub, pi, one](T x_cv, T x_cv_lb, T x_cv_ub) {
        T k                              = ceil(-0.5 - x_cv_lb / (2.0 * pi));
        T two_pi_k_lb                    = 2.0 * pi * k;
        auto cv_cos_nonconvex_nonconcave = [one](T x, T lb, T ub) {
            // We require that the slope of the connection line is equal to the slope of the
            // function, i.e., find point x in [a, b] s.t.
            //
            //          (f(x) - f(a)) / (x - a) = f'(x)
            //
            // So,
            //          f(x) - f(a) - (x - a) * f'(x) = 0
            //
            // Which can be solved via Newton's method, or any other root finding method.
            // For more details, see p.16 of McCormick's paper [1].
            //
            // [1] https://link.springer.com/article/10.1007/BF01580665

            bool left;
            T x0;
            T xm;

            if (abs(lb) <= abs(ub)) {
                left = false;
                x0   = ub;
                xm   = lb;
            } else {
                left = true;
                x0   = lb;
                xm   = ub;
            }

            // NOTE: We could potentially use the Interval Newton method in IA instead.

            // NOTE: Analytic: We know the taylor/pade approximation of
            //                 (x - a) * sin(x) + cos(x) - cos(a)
            //                 Make use of it for solving for the roots directly?

            // NOTE: Maybe we can skip the rootfind if xj = xm?
            // T xj = xm;

            auto f    = [xm](T x) { return (x - xm) * sin(x) + cos(x) - cos(xm); };
            auto df   = [xm](T x) { return (x - xm) * cos(x); };
            auto ddf  = [xm](T x) { return (xm - x) * sin(x) + cos(x); };
            auto dddf = [xm](T x) { return (xm - x) * cos(x) - 2.0 * sin(x); };

            T xj = root_householder_bisection(f, df, ddf, dddf, x0, lb, ub);

            if ((left && x <= xj) || (!left && x >= xj)) {
                return next_after(next_after(cos(x), -one), -one);
            } else {
                return next_after(next_after(chord_of_concave(x, xj, xm, cos(xj), cos(xm)), -one), -one);
            }

            return cos(x);
        };

        T cv;
        if (x_cv <= (pi * (1.0 - 2.0 * k))) {
            T x_cv_ub_1 = min(x_cv_ub + two_pi_k_lb, pi);
            T x_cv_lb_1 = x_cv_lb + two_pi_k_lb;

            if (x_cv_lb_1 >= 0.5 * pi) {
                // convex region
                cv = next_after(cos(x_cv), -one); // TODO: might need another rounding here
            } else if (x_cv_lb_1 >= -0.5 * pi && x_cv_ub_1 <= 0.5 * pi) {
                // concave region
                cv = chord_of_concave(x_cv, x_cv_lb, x_cv_ub, cos(x_cv_lb), cos(x_cv_ub));
            } else {
                // nonconvex and nonconcave region
                cv = cv_cos_nonconvex_nonconcave(x_cv + two_pi_k_lb, x_cv_lb_1, x_cv_ub_1);
            }
        } else {
            T k_upper     = floor(0.5 - x_cv_ub / (2.0 * pi));
            T two_pi_k_ub = 2.0 * pi * k_upper;
            if (x_cv >= (pi * (-1.0 - 2.0 * k_upper))) {
                T x_cv_ub_2 = x_cv_ub + two_pi_k_ub;
                if (x_cv_ub_2 <= -0.5 * pi) {
                    cv = next_after(next_after(next_after(cos(x_cv), -one), -one), -one);
                } else {
                    // nonconvex and nonconcave region
                    cv = cv_cos_nonconvex_nonconcave(x_cv + two_pi_k_ub,
                                                     max(x_cv_lb + two_pi_k_ub, -pi), x_cv_ub_2);
                }
            } else {
                cv = -1.0;
            }
        }
        return cv;
    };

    auto x_cv_lb = x_lb;
    auto x_cv_ub = x_ub;
    auto x_cv    = midcv;
    T cv         = cv_cos(x_cv, x_cv_lb, x_cv_ub);

    auto x_cc    = sub_down(midcc, pi);
    auto x_cc_lb = sub_down(x_lb, pi);
    auto x_cc_ub = sub_down(x_ub, pi);

    T cc = -cv_cos(x_cc, x_cc_lb, x_cc_ub);

    return { { .cv  = cv,
               .cc  = cc,
               .box = cos(x.box) } };
}

template<typename T>
cuda_fn mc<T> cos_box(mc<T> x)
{
    using namespace intrinsic;

    interval<T> cos_box = cos(x.box);

    return { { .cv  = inf(cos_box),
               .cc  = sup(cos_box),
               .box = cos_box } };
}

template<typename T>
cuda_fn mc<T> sin(mc<T> x)
{
    constexpr mc<T> pi_2 { 0x1.921fb54442d17p+0, 0x1.921fb54442d19p+0 };
    return cos(x - pi_2);
}

template<typename T>
cuda_fn mc<T> sinh(mc<T> x)
{
    using std::cosh;
    using std::sinh;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // convex region
        cv = sinh(midcv);
        cc = chord_of_convex(midcc, lb, ub, sinh(lb), sinh(ub));
    } else if (ub <= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, sinh(lb), sinh(ub));
        cc = sinh(midcc);
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // for cc:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // where f(x) = sinh(x) and f'(x) = cosh(x).

        auto dsinh = [](T x) { using std::cosh; return cosh(x); };

        {
            // cv:
            auto a   = lb;
            auto f   = [&](T x) { return sinh(x) - sinh(a) - (x - a) * dsinh(x); };
            auto df  = [&](T x) { return (a - x) * sinh(x); };
            auto ddf = [&](T x) { return (a - x) * cosh(x) - sinh(x); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcv >= ub_of_secant) {
                cv = sinh(midcv);
            } else {
                cv = chord_of_concave(midcv, lb, ub_of_secant, sinh(lb), sinh(ub_of_secant));
            }
        }
        {
            // cc:
            auto b   = ub;
            auto f   = [&](T x) { return sinh(b) - sinh(x) - (b - x) * dsinh(x); };
            auto df  = [&](T x) { return (x - b) * sinh(x); };
            auto ddf = [&](T x) { return (x - b) * cosh(x) + sinh(x); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcc < lb_of_secant) {
                cc = sinh(midcc);
            } else {
                cc = chord_of_convex(midcc, lb_of_secant, ub, sinh(lb_of_secant), sinh(ub));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = sinh(x.box) } };
}

template<typename T>
cuda_fn mc<T> cosh(mc<T> x)
{
    using namespace intrinsic;
    using std::abs;
    using std::cosh;

    constexpr auto zero = static_cast<T>(0);

    T midcv;
    T midcc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (ub <= zero) {
        midcv = x.cc;
        midcc = x.cv;
    } else if (lb >= zero) {
        midcv = x.cv;
        midcc = x.cc;
    } else {
        midcv = mid(zero, x.cv, x.cc);
        midcc = (abs(lb) >= abs(ub)) ? x.cv : x.cc;
    }

    return { { .cv  = cosh(midcv),
               .cc  = chord_of_convex(midcc, lb, ub, cosh(lb), cosh(ub)),
               .box = cosh(x.box) } };
}

template<typename T>
cuda_fn mc<T> tanh(mc<T> x)
{
    using std::tanh;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, tanh(lb), tanh(ub));
        cc = tanh(midcc);
    } else if (ub <= zero) {
        // convex region
        cv = tanh(midcv);
        cc = chord_of_convex(midcc, lb, ub, tanh(lb), tanh(ub));
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // for cc:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // where f(x) = tanh(x) and f'(x) = 1 - tanh(x)^2.

        auto dtanh = [](T x) { using std::pow; return 1 - pow(tanh(x), 2); };

        {
            // cv:
            auto b   = ub;
            auto f   = [&](T x) { return tanh(b) - tanh(x) - (b - x) * dtanh(x); };
            auto df  = [&](T x) { return 2.0 * (b - x) * dtanh(x) * tanh(x); };
            auto ddf = [&](T x) { return 2.0 * dtanh(x) * (tanh(x) * (2.0 * (x - b) * tanh(x) - 1.0) - (x - b) * dtanh(x)); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcv <= lb_of_secant) {
                cv = tanh(midcv);
            } else {
                cv = chord_of_concave(midcv, lb_of_secant, ub, tanh(lb_of_secant), tanh(ub));
            }
        }
        {
            // cc:
            auto a   = lb;
            auto f   = [&](T x) { return tanh(x) - tanh(a) - (x - a) * dtanh(x); };
            auto df  = [&](T x) { return 2.0 * (x - a) * dtanh(x) * tanh(x); };
            auto ddf = [&](T x) { return 2.0 * dtanh(x) * (tanh(x) * (2.0 * (x - a) * tanh(x) - 1.0) - (x - a) * dtanh(x)); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcc > ub_of_secant) {
                cc = tanh(midcc);
            } else {
                cc = chord_of_convex(midcc, lb, ub_of_secant, tanh(lb), tanh(ub_of_secant));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = tanh(x.box) } };
}

template<typename T>
cuda_fn mc<T> asin(mc<T> x)
{
    using std::asin;
    using std::pow;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // convex region
        cv = asin(midcv);
        cc = chord_of_convex(midcc, lb, ub, asin(lb), asin(ub));
    } else if (ub <= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, asin(lb), asin(ub));
        cc = asin(midcc);
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // for cc:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // where f(x) = asin(x) and f'(x) = 1 / sqrt(1 - x^2).

        auto dasin = [](T x) { using std::sqrt; using std::pow; return 1 / sqrt(1 - pow(x, 2)); };

        {
            // cv:
            auto a   = lb;
            auto f   = [&](T x) { return asin(x) - asin(a) - (x - a) * dasin(x); };
            auto df  = [&](T x) { return x * (a - x) / pow((1 - pow(x, 2)), 3. / 2.); }; // TODO: check if 1-x^2 >= 0 then x^(3/2) = x*sqrt(x)
            auto ddf = [&](T x) { return -(pow(x, 3) - 2 * a * pow(x, 2) + 2 * x - a) / pow((1 - pow(x, 2)), 5. / 2.); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcv >= ub_of_secant) {
                cv = asin(midcv);
            } else {
                cv = chord_of_concave(midcv, lb, ub_of_secant, asin(lb), asin(ub_of_secant));
            }
        }
        {
            // cc:
            auto b   = ub;
            auto f   = [&](T x) { return asin(b) - asin(x) - (b - x) * dasin(x); };
            auto df  = [&](T x) { return x * (x - b) / pow((1 - pow(x, 2)), 3. / 2.); }; // TODO: check if 1-x^2 >= 0 then x^(3/2) = x*sqrt(x)
            auto ddf = [&](T x) { return (pow(x, 3) - 2 * b * pow(x, 2) + 2 * x - b) / pow((1 - pow(x, 2)), 5. / 2.); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcc < lb_of_secant) {
                cc = asin(midcc);
            } else {
                cc = chord_of_convex(midcc, lb_of_secant, ub, asin(lb_of_secant), asin(ub));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = asin(x.box) } };
}

template<typename T>
cuda_fn mc<T> acos(mc<T> x)
{
    constexpr mc<T> pi_2 { 0x1.921fb54442d17p+0, 0x1.921fb54442d19p+0 };
    return asin(-x) + pi_2;
}

template<typename T>
cuda_fn mc<T> atan(mc<T> x)
{
    using std::atan;
    using std::pow;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, atan(lb), atan(ub));
        cc = atan(midcc);
    } else if (ub <= zero) {
        // convex region
        cv = atan(midcv);
        cc = chord_of_convex(midcc, lb, ub, atan(lb), atan(ub));
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // for cc:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // where f(x) = atan(x) and f'(x) = 1 / (x^2 + 1).

        auto datan = [](T x) { using std::pow; return 1.0 / (pow(x, 2) + 1.0); };

        {
            // cv:
            auto b   = ub;
            auto f   = [&](T x) { return atan(b) - atan(x) - (b - x) * datan(x); };
            auto df  = [&](T x) { return -2.0 * x * (x - b) * pow(datan(x), 2); };
            auto ddf = [&](T x) { return 2.0 * (2.0 * pow(x, 3) - 3.0 * b * pow(x, 2) - 2 * x + b) * pow(datan(x), 3); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcv <= lb_of_secant) {
                cv = atan(midcv);
            } else {
                cv = chord_of_concave(midcv, lb_of_secant, ub, atan(lb_of_secant), atan(ub));
            }
        }
        {
            // cc:
            auto a   = lb;
            auto f   = [&](T x) { return atan(x) - atan(a) - (x - a) * datan(x); };
            auto df  = [&](T x) { return 2.0 * x * (x - a) * pow(datan(x), 2); };
            auto ddf = [&](T x) { return -2.0 * (2.0 * pow(x, 3) - 3.0 * a * pow(x, 2) - 2 * x + a) * pow(datan(x), 3); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcc > ub_of_secant) {
                cc = atan(midcc);
            } else {
                cc = chord_of_convex(midcc, lb, ub_of_secant, atan(lb), atan(ub_of_secant));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = atan(x.box) } };
}

template<typename T>
cuda_fn mc<T> asinh(mc<T> x)
{
    using std::asinh;
    using std::pow;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, asinh(lb), asinh(ub));
        cc = asinh(midcc);
    } else if (ub <= zero) {
        // convex region
        cv = asinh(midcv);
        cc = chord_of_convex(midcc, lb, ub, asinh(lb), asinh(ub));
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // for cc:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // where f(x) = asinh(x) and f'(x) = 1 / sqrt(1 + x^2).

        auto dasinh = [](T x) { using std::sqrt; using std::pow; return 1 / sqrt(1 + pow(x, 2)); };

        {
            // cv:
            auto b   = ub;
            auto f   = [&](T x) { return asinh(b) - asinh(x) + (x - b) * dasinh(x); };
            auto df  = [&](T x) { return -x * (x - b) * pow(dasinh(x), 3); };
            auto ddf = [&](T x) { return (pow(x, 3) - 2.0 * b * pow(x, 2) - 2 * x + b) / pow(dasinh(x), 5); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcv <= lb_of_secant) {
                cv = asinh(midcv);
            } else {
                cv = chord_of_concave(midcv, lb_of_secant, ub, asinh(lb_of_secant), asinh(ub));
            }
        }
        {
            // cc:
            auto a   = lb;
            auto f   = [&](T x) { return asinh(x) - asinh(a) + (a - x) * dasinh(x); };
            auto df  = [&](T x) { return x * (x - a) * pow(dasinh(x), 3); };
            auto ddf = [&](T x) { return -(pow(x, 3) - 2.0 * a * pow(x, 2) - 2 * x + a) / pow(dasinh(x), 5); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcc > ub_of_secant) {
                cc = asinh(midcc);
            } else {
                cc = chord_of_convex(midcc, lb, ub_of_secant, asinh(lb), asinh(ub_of_secant));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = asinh(x.box) } };
}

template<typename T>
cuda_fn mc<T> acosh(mc<T> x)
{
    using namespace intrinsic;

    using std::acosh;

    T midcv = mid(inf(x), x.cv, x.cc);
    T midcc = mid(sup(x), x.cv, x.cc);
    T cv    = chord_of_concave(midcv, inf(x), sup(x), acosh(inf(x)), acosh(sup(x)));

    return { { .cv  = cv,
               .cc  = acosh(midcc),
               .box = acosh(x.box) } };
}

template<typename T>
cuda_fn mc<T> atanh(mc<T> x)
{
    using std::atanh;
    using std::pow;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // convex region
        cv = atanh(midcv);
        cc = chord_of_convex(midcc, lb, ub, atanh(lb), atanh(ub));
    } else if (ub <= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, atanh(lb), atanh(ub));
        cc = atanh(midcc);
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // for cc:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // where f(x) = atanh(x) and f'(x) = 1 / (1 - x^2).

        auto datanh = [](T x) { using std::pow; return 1 / (1 - pow(x, 2)); };

        {
            // cv:
            auto a   = lb;
            auto f   = [&](T x) { return atanh(x) - atanh(a) + (a - x) * datanh(x); };
            auto df  = [&](T x) { return 2 * x * (a - x) * datanh(x); };
            auto ddf = [&](T x) { return 2 * (2 * pow(x, 3) - 3 * a * pow(x, 2) + 2 * x - a) / pow(pow(x, 2) - 1, 3); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcv >= ub_of_secant) {
                cv = atanh(midcv);
            } else {
                cv = chord_of_concave(midcv, lb, ub_of_secant, atanh(lb), atanh(ub_of_secant));
            }
        }
        {
            // cc:
            auto b   = ub;
            auto f   = [&](T x) { return atanh(b) - atanh(x) + (x - b) * datanh(x); };
            auto df  = [&](T x) { return 2 * x * (x - b) / pow((pow(x, 2) - 1), 2); };
            auto ddf = [&](T x) { return -2 * (2 * pow(x, 3) - 3 * b * pow(x, 2) + 2 * x - b) / pow(pow(x, 2) - 1, 3); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcc < lb_of_secant) {
                cc = atanh(midcc);
            } else {
                cc = chord_of_convex(midcc, lb_of_secant, ub, atanh(lb_of_secant), atanh(ub));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = atanh(x.box) } };
}

template<typename T>
cuda_fn mc<T> log(mc<T> x)
{
    using namespace intrinsic;
    using std::log;

    // since log is monotonically increasing:
    //
    //      mid(inf(x), x.cv, x.cc) = x.cv
    //      mid(sup(x), x.cv, x.cc) = x.cc
    //
    T midcv = x.cv;
    T midcc = x.cc;

    // TODO: error in log not accounted for
    T cv = chord_of_concave(midcv, inf(x), sup(x), log(inf(x)), log(sup(x)));

    if (inf(x) <= static_cast<T>(0)) {
        cv = neg_inf<T>();
    }

    return { { .cv  = cv,
               .cc  = log(midcc),
               .box = log(x.box) } };
}

template<typename T>
cuda_fn mc<T> erf(mc<T> x)
{
    using std::erf;
    using std::exp;
    using std::pow;
    using std::sqrt;

    constexpr auto zero = static_cast<T>(0);

    T midcv = x.cv;
    T midcc = x.cc;

    T cv;
    T cc;

    auto lb = inf(x);
    auto ub = sup(x);

    if (lb >= zero) {
        // concave region
        cv = chord_of_concave(midcv, lb, ub, erf(lb), erf(ub));
        cc = erf(midcc);
    } else if (ub <= zero) {
        // convex region
        cv = erf(midcv);
        cc = chord_of_convex(midcc, lb, ub, erf(lb), erf(ub));
    } else {
        // nonconvex and nonconcave region

        // We need to find the point x in [a, b] s.t.
        //
        // for cv:
        //          (f(b) - f(x)) / (b - x) = f'(x)
        //
        // for cc:
        //          (f(x) - f(a)) / (x - a) = f'(x)
        //
        // where f(x) = erf(x) and f'(x) = 2 / sqrt(pi) * exp(-x^2)

        auto derf = [](T x) {
            using std::sqrt;
            using std::exp;
            using std::pow;
            return 2 * exp(-pow(x, 2)) / sqrt(std::numbers::pi);
        };

        {
            // cv:
            auto b   = ub;
            auto f   = [&](T x) { return erf(b) - erf(x) + (x - b) * derf(x); };
            auto df  = [&](T x) { return -2 * x * (x - b) * derf(x); };
            auto ddf = [&](T x) { return (8 * pow(x, 3) - 8 * b * pow(x, 2) - 8 * x + 4 * b) * exp(-pow(x, 2)) / sqrt(std::numbers::pi); };

            T x0           = lb;
            T lb_of_secant = root_halley_bisection(f, df, ddf, x0, lb, zero);

            if (midcv <= lb_of_secant) {
                cv = erf(midcv);
            } else {
                cv = chord_of_concave(midcv, lb_of_secant, ub, erf(lb_of_secant), erf(ub));
            }
        }
        {
            // cc:
            auto a   = lb;
            auto f   = [&](T x) { return erf(x) - erf(a) + (a - x) * derf(x); };
            auto df  = [&](T x) { return 2 * x * (x - a) * derf(x); };
            auto ddf = [&](T x) { return (8 * pow(x, 3) - 8 * a * pow(x, 2) - 8 * x + 4 * a) * exp(-pow(x, 2)) / sqrt(std::numbers::pi); };

            T x0           = ub;
            T ub_of_secant = root_halley_bisection(f, df, ddf, x0, zero, ub);

            if (midcc > ub_of_secant) {
                cc = erf(midcc);
            } else {
                cc = chord_of_convex(midcc, lb, ub_of_secant, erf(lb), erf(ub_of_secant));
            }
        }
    }

    return { { .cv  = cv,
               .cc  = cc,
               .box = erf(x.box) } };
}

template<typename T>
cuda_fn mc<T> erfc(mc<T> x)
{
    using std::erfc;

    return 1.0 - erf(x);
}

template<typename T>
cuda_fn mc<T> max(mc<T> a, mc<T> b)
{
    using std::max;

    T cc {};

    if (sup(a) <= inf(b)) {
        cc = b.cc;
    } else if (sup(b) <= inf(a)) {
        cc = a.cc;
    } else {
        // TODO: think about adding max of multivariate mccormick

        // TODO: check if optimizer removes all the unnecessary cv and IA operations
        cc = 0.5 * (a + b + abs(a - b)).cc; // uses the fact that max(a,b)= (a + b + abs(a - b)) / 2
    }

    return { { .cv  = max(a.cv, b.cv), // max is a convex function
               .cc  = cc,
               .box = max(a.box, b.box) } };
}

template<typename T>
cuda_fn mc<T> min(mc<T> a, mc<T> b)
{
    using std::min;

    T cv {};

    if (sup(a) <= inf(b)) {
        cv = a.cv;
    } else if (sup(b) <= inf(a)) {
        cv = b.cv;
    } else {
        // TODO: think about adding min of multivariate mccormick
        cv = 0.5 * (a + b - abs(a - b)).cv; // uses the fact that min(a,b)= (a + b - abs(a - b)) / 2
    }

    return { { .cv  = cv,
               .cc  = min(a.cc, b.cc),
               .box = min(a.box, b.box) } };
}

template<typename T>
cuda_fn T width(mc<T> x)
{
    return width(x.box);
}

template<typename T>
cuda_fn mc<T> hull(mc<T> a, mc<T> b)
{
    return { { .cv  = min(a, b).cv,
               .cc  = max(a, b).cc,
               .box = convex_hull(a.box, b.box) } };
}

#undef cuda_fn

} // namespace cu
#endif // CUMCCORMICK_ARITHMETIC_BASIC_CUH
