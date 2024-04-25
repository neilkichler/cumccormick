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
inline __device__ T mid(T v, T lb, T ub)
{
    return std::clamp(v, lb, ub);
}

template<typename T>
inline __device__ mc<T> neg(mc<T> x)
{
    return { .cv  = -x.cc,
             .cc  = -x.cv,
             .box = -x.box };
}

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
inline __device__ mc<T> mul_nearest_even_rounding(mc<T> a, mc<T> b)
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

    return { .cv  = cv,
             .cc  = cc,
             .box = mul(a.box, b.box) };
}

template<typename T>
inline __device__ mc<T> mul(mc<T> a, mc<T> b)
{
    using namespace intrinsic;

    T alpha1 = min(mul_down(inf(b), a.cv), mul_down(inf(b), a.cc));
    T alpha2 = min(mul_down(inf(a), b.cv), mul_down(inf(a), b.cc));
    T beta1  = min(mul_down(sup(b), a.cv), mul_down(sup(b), a.cc));
    T beta2  = min(mul_down(sup(a), b.cv), mul_down(sup(a), b.cc));

    T gamma1 = max(mul_up(inf(b), a.cv), mul_up(inf(b), a.cc));
    T delta2 = max(mul_up(inf(a), b.cv), mul_up(inf(a), b.cc));
    T delta1 = max(mul_up(sup(b), a.cv), mul_up(sup(b), a.cc));
    T gamma2 = max(mul_up(sup(a), b.cv), mul_up(sup(a), b.cc));

    T cv = max(sub_down(add_down(alpha1, alpha2), mul_down(inf(a), inf(b))),
               sub_down(add_down(beta1, beta2), mul_down(sup(a), sup(b))));

    T cc = min(sub_up(add_up(gamma1, gamma2), mul_up(sup(a), inf(b))),
               sub_up(add_up(delta1, delta2), mul_up(inf(a), sup(b))));

    return { .cv  = cv,
             .cc  = cc,
             .box = mul(a.box, b.box) };
}

template<typename T, typename F>
inline __device__ T secant_of_concave(T x, T lb, T ub, F &&f)
{
    // TODO: We could also just pass in the f(ub) and f(lb)
    //       and not the function.

    // TODO: Does not consider rounding of f
    using namespace intrinsic;

    // computing secant over interval endpoints
    T r = lb == ub
        ? static_cast<T>(0)
        : div_down(sub_down(f(ub), f(lb)), (sub_down(ub, lb)));

    return add_down(f(lb), mul_down(r, sub_down(x, lb)));
}

template<typename T, typename F>
inline __device__ T secant_of_convex(T x, T lb, T ub, F &&f)
{
    // TODO: Does not consider rounding of f
    using namespace intrinsic;

    // computing secant over interval endpoints
    T r = lb == ub
        ? static_cast<T>(0)
        : div_up(sub_up(f(ub), f(lb)), (sub_up(ub, lb)));

    return add_up(f(ub), mul_up(r, sub_up(x, ub)));
}

template<typename T>
inline __device__ mc<T> recip(mc<T> x)
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
            return { .cv  = intrinsic::neg_inf<T>(),
                     .cc  = intrinsic::pos_inf<T>(),
                     .box = entire<T>() };
        } else if (inf(x) == zero && zero == sup(x)) {
            // NOTE: Alternatively, we could return nans.
            return { .cv  = intrinsic::neg_inf<T>(),
                     .cc  = intrinsic::pos_inf<T>(),
                     .box = entire<T>() };
        }
    } else if (sup(x) < zero) {
        // for x < 0, recip is concave
        cv = secant_of_concave(midcv, inf(x), sup(x), rcp_down<T>);
        cc = rcp_up(midcc);
    } else { // inf(x) > zero
        // for x > 0, recip is convex
        cc = secant_of_convex(midcc, inf(x), sup(x), rcp_up<T>);
        cv = rcp_down(midcv);
    }

    // TODO: relaxation could be more efficient if we embed the IA into the different cases.
    return { .cv  = cv,
             .cc  = cc,
             .box = recip(x.box) };
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
inline __device__ mc<T> div(mc<T> a, mc<T> b)
{
    // TODO: implement tighter relaxation
    return mul(a, recip(b));
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
    using namespace intrinsic;

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
    return { .cv  = mul_down(midcv, midcv),
             .cc  = cc,
             .box = sqr(x.box) };
}

template<typename T>
inline __device__ mc<T> abs(mc<T> x)
{
    T xmin  = mid(static_cast<T>(0), inf(x), sup(x));
    T midcv = mid(xmin, x.cv, x.cc);

    T xmax  = abs(inf(x)) >= abs(sup(x)) ? inf(x) : sup(x);
    T midcc = mid(xmax, x.cv, x.cc);

    T cc = secant_of_convex(midcc, inf(x), sup(x), [](T x) { return abs(x); });
    return { .cv  = abs(midcv),
             .cc  = cc,
             .box = abs(x.box) };
}

template<typename T>
inline __device__ mc<T> fabs(mc<T> x)
{
    return abs(x);
}

template<typename T>
inline __device__ mc<T> exp(mc<T> x)
{
    using namespace intrinsic;
    // TODO: error in exp not accounted for in secant computation
    T cc = exp(sup(x)) == intrinsic::pos_inf<T>()
        ? intrinsic::pos_inf<T>()
        : secant_of_convex(x.cc, inf(x), sup(x), [](T x) { return exp(x); });

    return { .cv  = next_after(exp(x.cv), static_cast<T>(0)),
             .cc  = cc,
             .box = exp(x.box) };
}

template<typename T>
inline __device__ mc<T> sqrt(mc<T> x)
{
    using namespace intrinsic;
    T midcv = mid(inf(x), x.cv, x.cc);
    T midcc = mid(sup(x), x.cv, x.cc);
    T cv    = secant_of_concave(midcv, inf(x), sup(x), [](T x) { return sqrt(x); });

    return { .cv  = cv,
             .cc  = intrinsic::sqrt_up(midcc),
             .box = sqrt(x.box) };
}

template<typename T>
inline __device__ mc<T> pown_even(mc<T> x, std::integral auto n)
{
    using namespace intrinsic;

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
            return { .cv  = pow(midcv, n),
                     .cc  = intrinsic::pos_inf<T>(),
                     .box = pown(x.box, n) };
        }
    }

    // TODO: floating point error in pow not accounted for
    cc = secant_of_convex(midcc, inf(x), sup(x), [n](T x) { return pow(x, n); });

    return { .cv  = pow(midcv, n),
             .cc  = cc,
             .box = pown(x.box, n) };
}

template<typename T>
inline __device__ mc<T> pown(mc<T> x, std::integral auto n)
{
    using namespace intrinsic;

    T cv;
    T cc;

    if (n == 0) {
        constexpr auto one = static_cast<T>(1);
        return { .cv  = one,
                 .cc  = one,
                 .box = { one, one } };
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
            cv = secant_of_concave(n > 0 ? x.cv : x.cc, inf(x), sup(x), [n](T x) { return pow(x, n); });
            cc = pow(n > 0 ? x.cc : x.cv, n);
        } else if (inf(x) >= zero) {
            // for x > 0, pown(x,n_odd) is convex
            cv = pow(n > 0 ? x.cv : x.cc, n);
            cc = secant_of_convex(n > 0 ? x.cc : x.cv, inf(x), sup(x), [n](T x) { return pow(x, n); });
        } else {
            // for 0 in x, pown(x,n_odd) is concavoconvex
            if (n > 0) {
                // differentiable variant
                cv = pow(inf(x), n) * ((sup(x) - x.cv) / (sup(x) - inf(x))) + pow(max(zero, x.cv), n);
                cc = pow(sup(x), n) * ((x.cc - inf(x)) / (sup(x) - inf(x))) + pow(min(zero, x.cc), n);
            } else {
                return { .cv  = intrinsic::pos_inf<T>(),
                         .cc  = intrinsic::neg_inf<T>(),
                         .box = entire<T>() };
            }
        }
    } else { // even power
        return pown_even(x, n);
    }

    return { .cv  = cv,
             .cc  = cc,
             .box = pown(x.box, n) };
}

template<typename T>
inline __device__ mc<T> pow(mc<T> x, std::integral auto n)
{
    return pown(x, n);
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
inline __device__ mc<T> operator+(std::integral auto a, mc<T> b)
{
    return add(static_cast<T>(a), b);
}

template<typename T>
inline __device__ mc<T> operator+(mc<T> a, T b)
{
    return add(b, a);
}

template<typename T>
inline __device__ mc<T> operator+(mc<T> a, std::integral auto b)
{
    return add(static_cast<T>(b), a);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a)
{
    return neg(a);
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
inline __device__ mc<T> operator-(std::integral auto a, mc<T> b)
{
    return sub(static_cast<T>(a), b);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a, T b)
{
    return sub(a, b);
}

template<typename T>
inline __device__ mc<T> operator-(mc<T> a, std::integral auto b)
{
    return sub(a, static_cast<T>(b));
}

template<typename T>
inline __device__ mc<T> operator*(T a, mc<T> b)
{
    return mul(a, b);
}

template<typename T>
inline __device__ mc<T> operator*(std::integral auto a, mc<T> b)
{
    return mul(static_cast<T>(a), b);
}

template<typename T>
inline __device__ mc<T> operator*(mc<T> a, T b)
{
    return mul(b, a);
}

template<typename T>
inline __device__ mc<T> operator*(mc<T> a, std::integral auto b)
{
    return mul(static_cast<T>(b), a);
}

template<typename T>
inline __device__ mc<T> operator*(mc<T> a, mc<T> b)
{
    return mul(b, a);
}

template<typename T>
inline __device__ mc<T> operator/(mc<T> a, T b)
{
    return div(a, b);
}

template<typename T>
inline __device__ mc<T> operator/(mc<T> a, std::integral auto b)
{
    return div(a, static_cast<T>(b));
}

template<typename T>
inline __device__ mc<T> operator/(mc<T> a, mc<T> b)
{
    return div(a, b);
}

template<std::floating_point T>
struct solver_options
{
    int maxiter { 50 };
    T atol { 1e-10 };
    T rtol { 0.0 };
};

template<std::floating_point T>
struct root_solver_state
{
    T x;
    T lb;
    T ub;
    T y;
};

template<typename T>
inline __device__ T root(auto &&f, auto &&step, T x0, T lb, T ub, solver_options<T> options = {})
{
    assert(f(lb) * f(ub) <= 0.0 && "sign must be different for f(lb) and f(ub)");

    T x       = mid(x0, lb, ub);
    T delta_x = intrinsic::pos_inf<T>();

    auto terminate = [options](auto f_error, auto x, auto x_prev, std::integral auto i) {
        auto [maxiter, atol, rtol] = options;
        auto scaled_tol            = atol + rtol * abs(x); // alternative: max(atol, rtol * abs(x));
        bool x_small               = abs(x - x_prev) < scaled_tol;
        bool f_small               = abs(f_error) < atol;
        return (x_small && f_small) || i >= maxiter;
    };

    for (int i = 0;; i++) {
        auto state                = root_solver_state<T> { x, lb, ub };
        auto [x_new, lb_, ub_, y] = step(state, delta_x);
        lb                        = lb_;
        ub                        = ub_;
        x_new                     = mid(x_new, lb, ub);

        if (terminate(y, x_new, x, i)) {
            return x_new;
        }

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

template<std::floating_point T>
inline __device__ auto derivative_or_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&step_fn, T epsilon=1e-30)
{
    auto [x, lb, ub, _] = state;

    T y    = f(x);
    T dydx = df(x);

    bool too_slow_progress = abs(2.0 * y) > abs(delta_x * dydx);

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

    return root_solver_state { x_new, lb, ub, y };
}

inline __device__ auto newton_step(auto x, auto y, auto &&df)
{
    return y / df(x);
}

template<typename T>
inline __device__ auto newton_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df)
{
    auto [x, lb, ub, _] = state;
    T y                 = f(x);
    x                   = x - newton_step(x, y, df);
    return root_solver_state { x, lb, ub, y };
}

inline __device__ auto halley_step(auto x, auto y, auto &&df, auto &&ddf)
{
    return (2.0 * y * df(x)) / (2.0 * pow(df(x), 2) - y * ddf(x));
}

template<typename T>
inline __device__ auto halley_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df, auto &&ddf)
{
    auto [x, lb, ub, _] = state;
    T y                 = f(x);
    x                   = x - halley_step(x, y, df, ddf);
    return root_solver_state { x, lb, ub, y };
}

inline __device__ auto householder_step(auto x, auto y, auto &&df, auto &&ddf, auto &&dddf)
{
    auto dy   = df(x);
    auto ddy  = ddf(x);
    auto dddy = dddf(x);
    return (6 * y * pow(dy, 2) - 3 * pow(y, 2) * ddy)
        / (6 * pow(dy, 3) - 6 * y * dy * ddy + pow(y, 2) * dddy);
}

template<typename T>
inline __device__ auto householder_step(root_solver_state<T> state, auto delta_x, auto &&f, auto &&df, auto &&ddf, auto &&dddf)
{
    auto [x, lb, ub, _] = state;
    T y                 = f(x);
    x                   = x - householder_step(x, y, df, ddf, dddf);
    return root_solver_state { x, lb, ub, y };
}

template<std::floating_point T>
inline __device__ auto newton_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df)
{
    auto step_fn = [df](auto x, auto y) { return newton_step(x, y, df); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<std::floating_point T>
inline __device__ auto halley_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&ddf)
{
    auto step_fn = [df, ddf](auto x, auto y) { return halley_step(x, y, df, ddf); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<std::floating_point T>
inline __device__ auto householder_bisection_step(root_solver_state<T> state, T delta_x, auto &&f, auto &&df, auto &&ddf, auto &&dddf)
{
    auto step_fn = [df, ddf, dddf](auto x, auto y) { return householder_step(x, y, df, ddf, dddf); };
    return derivative_or_bisection_step(state, delta_x, f, df, step_fn);
}

template<typename T>
inline __device__ T root_newton(auto &&f, auto &&df, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df](auto x, auto delta_x) { return newton_step(x, delta_x, f, df); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ T root_newton_bisection(auto &&f, auto &&df, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df](auto x, auto delta_x) { return newton_bisection_step(x, delta_x, f, df); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ T root_halley(auto &&f, auto &&df, auto &&ddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf](auto x, auto delta_x) { return halley_step(x, delta_x, f, df, ddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ T root_halley_bisection(auto &&f, auto &&df, auto &&ddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf](auto x, auto delta_x) { return halley_bisection_step(x, delta_x, f, df, ddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ T root_householder(auto &&f, auto &&df, auto &&ddf, auto &&dddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf, dddf](auto x, auto delta_x) { return householder_step(x, delta_x, f, df, ddf, dddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ T root_householder_bisection(auto &&f, auto &&df, auto &&ddf, auto &&dddf, T x0, T lb, T ub, solver_options<T> options = {})
{
    auto step_fn = [f, df, ddf, dddf](auto x, auto delta_x) { return householder_bisection_step(x, delta_x, f, df, ddf, dddf); };
    return root(f, step_fn, x0, lb, ub, options);
}

template<typename T>
inline __device__ mc<T> cos(mc<T> x)
{
    using namespace intrinsic;

    // TODO: use rounded ops

    T argmin      = {};
    T argmax      = {};
    T pi          = std::numbers::pi;
    T k           = std::ceil(-0.5 - inf(x) / (2.0 * pi));
    T two_pi_k_lb = 2.0 * pi * k;
    T x_lb        = inf(x);
    T x_ub        = sup(x);

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
            return { .cv = -1.0, .cc = 1.0, .box = { -1.0, 1.0 } };
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
            return { .cv = -1.0, .cc = 1.0, .box = { -1.0, 1.0 } };
        } else {
            // decreasing then increasing
            argmin = pi * (1.0 - 2.0 * k);
            argmax = (cos(x_lb_centered) >= cos(x_ub_centered)) ? x_lb : x_ub;
        }
    }

    T midcv = mid(argmin, x.cv, x.cc);
    T midcc = mid(argmax, x.cv, x.cc);

    auto cv_cos = [x_lb, x_ub, pi](T x_cv, T x_cv_lb, T x_cv_ub) {
        T k                              = std::ceil(-0.5 - x_cv_lb / (2.0 * pi));
        T two_pi_k_lb                    = 2.0 * pi * k;
        auto cv_cos_nonconvex_nonconcave = [](T x, T lb, T ub) {
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

            // NOTE: We could potentially use the Interval Newton method instead.
            //       Or a better root finding method: Halley's method or Brent's method.

            // NOTE: Analytic: We know the taylor/pade approximation of
            //                 (x - a) * sin(x) + cos(x) - cos(a)
            //                 Make use of it for solving for the roots directly?

            // NOTE: Maybe we can skip the rootfind if xj = xm?
            // T xj = xm;

            auto f  = [xm](T x) { return (x - xm) * sin(x) + cos(x) - cos(xm); };
            auto df = [xm](T x) { return (x - xm) * cos(x); };
            T xj    = root_newton_bisection(f, df, x0, lb, ub);

            if (left && x <= xj || !left && x >= xj) {
                return next_after(cos(x), -1.0);
            } else {
                return secant_of_concave(x, xj, xm, [](T x) { return cos(x); });
            }

            return cos(x);
        };

        T cv;
        if (x_cv <= (pi * (1.0 - 2.0 * k))) {
            T x_cv_ub_1 = min(x_cv_ub + two_pi_k_lb, pi);
            T x_cv_lb_1 = x_cv_lb + two_pi_k_lb;

            if (x_cv_lb_1 >= 0.5 * pi) {
                // convex region
                cv = cos(x_cv);
            } else if (x_cv_lb_1 >= -0.5 * pi && x_cv_ub_1 <= 0.5 * pi) {
                // concave region
                cv = secant_of_concave(x_cv, x_cv_lb, x_cv_ub, [](T x) { return cos(x); });
            } else {
                // nonconvex and nonconcave region
                cv = cv_cos_nonconvex_nonconcave(x_cv + two_pi_k_lb, x_cv_lb_1, x_cv_ub_1);
            }
        } else {
            T k_upper     = std::floor(0.5 - x_cv_ub / (2.0 * pi));
            T two_pi_k_ub = 2.0 * pi * k_upper;
            if (x_cv >= (pi * (-1.0 - 2.0 * k_upper))) {
                T x_cv_ub_2 = x_cv_ub + two_pi_k_ub;
                if (x_cv_ub_2 <= -0.5 * pi) {
                    cv = next_after(next_after(next_after(cos(x_cv), -1.0), -1.0), -1.0);
                } else {
                    // nonconvex and nonconcave region
                    cv = cv_cos_nonconvex_nonconcave(x_cv + two_pi_k_ub,
                                                     max(two_pi_k_ub, -pi), x_cv_ub_2);
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

    return { .cv  = cv,
             .cc  = cc,
             .box = cos(x.box) };
}

template<typename T>
inline __device__ mc<T> cos_box(mc<T> x)
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

    return cos(x - std::numbers::pi / static_cast<T>(2.0));
}

template<typename T>
inline __device__ mc<T> log(mc<T> x)
{
    using namespace intrinsic;

    // since log is monotonically increasing:
    //
    //      mid(inf(x), x.cv, x.cc) = x.cv
    //      mid(sup(x), x.cv, x.cc) = x.cc
    //
    T midcv = x.cv;
    T midcc = x.cc;

    // TODO: error in log not accounted for
    T cv = secant_of_concave(midcv, inf(x), sup(x), [](T x) { return log(x); });

    if (inf(x) <= static_cast<T>(0)) {
        cv = neg_inf<T>();
    }

    return { .cv  = cv,
             .cc  = log(midcc),
             .box = log(x.box) };
}

template<typename T>
inline __device__ mc<T> max(mc<T> a, mc<T> b)
{
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

    return { .cv  = max(a.cv, b.cv), // max is a convex function
             .cc  = cc,
             .box = max(a.box, b.box) };
}

template<typename T>
inline __device__ mc<T> min(mc<T> a, mc<T> b)
{
    T cv {};

    if (sup(a) <= inf(b)) {
        cv = a.cv;
    } else if (sup(b) <= inf(a)) {
        cv = b.cv;
    } else {
        // TODO: think about adding min of multivariate mccormick
        cv = 0.5 * (a + b - abs(a - b)).cv; // uses the fact that min(a,b)= (a + b - abs(a - b)) / 2
    }

    return { .cv  = cv,
             .cc  = min(a.cc, b.cc),
             .box = min(a.box, b.box) };
}

#endif // CUMCCORMICK_ARITHMETIC_BASIC_CUH
