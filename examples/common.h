#ifndef CUMCCORMICK_EXAMPLES_COMMON_H
#define CUMCCORMICK_EXAMPLES_COMMON_H

// #define MCCORMICK_FORMAT "[lb: %g (cv: %g, cc: %g), ub: %g]"

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

template<typename T>
using mc = cu::mccormick<T>;

// printing utils
// NOTE: This is built-in if you use c++23

#include <format>
#include <string_view>

// send to stdout by default
constexpr void print(const std::string_view str, auto &&...args)
{
    fputs(std::vformat(str, std::make_format_args(args...)).c_str(), stdout);
}

// send to stdout by default
constexpr void println(const std::string_view str, auto &&...args)
{
    print(str, args...);
    fputs("\n", stdout);
}

#endif // CUMCCORMICK_EXAMPLES_COMMON_H
