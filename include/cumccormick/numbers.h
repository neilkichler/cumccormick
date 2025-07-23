#ifndef CUMCCORMICK_NUMBERS_H
#define CUMCCORMICK_NUMBERS_H

#include <cuinterval/numbers.h>
#include <cumccormick/mccormick.h>

#include <numbers>

// Explicit specialization of math constants is allowed for custom types.
// See https://eel.is/c++draft/numbers#math.constants-2.
namespace std::numbers
{

// The enclosure is chosen to be the smallest representable floating point
// interval which still contains the real value. The cv and cc values are
// equal to the lower and upper bound, respectively.

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> e_v<cu::mccormick<T, I>> = { e_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> log2e_v<cu::mccormick<T, I>> = { log2e_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> log10e_v<cu::mccormick<T, I>> = { log10e_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> pi_v<cu::mccormick<T, I>> = { pi_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> inv_pi_v<cu::mccormick<T, I>> = { inv_pi_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> inv_sqrtpi_v<cu::mccormick<T, I>> = { inv_sqrtpi_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> ln2_v<cu::mccormick<T, I>> = { ln2_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> ln10_v<cu::mccormick<T, I>> = { ln10_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> sqrt2_v<cu::mccormick<T, I>> = { sqrt2_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> sqrt3_v<cu::mccormick<T, I>> = { sqrt3_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> inv_sqrt3_v<cu::mccormick<T, I>> = { inv_sqrt3_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> egamma_v<cu::mccormick<T, I>> = { egamma_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> phi_v<cu::mccormick<T, I>> = { phi_v<I> };

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::e_v;
using std::numbers::egamma_v;
using std::numbers::inv_pi_v;
using std::numbers::inv_sqrt3_v;
using std::numbers::inv_sqrtpi_v;
using std::numbers::ln10_v;
using std::numbers::ln2_v;
using std::numbers::log10e_v;
using std::numbers::log2e_v;
using std::numbers::phi_v;
using std::numbers::pi_v;
using std::numbers::sqrt2_v;
using std::numbers::sqrt3_v;

template<typename T, typename I>
inline constexpr mccormick<T, I> pi_2_v<mccormick<T, I>> = { pi_2_v<I> };

template<typename T, typename I>
inline constexpr cu::mccormick<T, I> tau_v<cu::mccormick<T, I>> = { tau_v<I> };

} // namespace cu

#endif // CUMCCORMICK_NUMBERS_H
