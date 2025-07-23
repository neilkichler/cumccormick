#ifndef CUMCCORMICK_FORMAT_H
#define CUMCCORMICK_FORMAT_H

#include <cumccormick/cumccormick.h>

#include <format>
#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, cu::mccormick<T> x)
{
    return os << "[lb: " << x.box.lb << ", (cv: " << x.cv << ", cc: " << x.cc << "), ub: " << x.box.ub << "]";
}

} // namespace cu

template<typename T>
struct std::formatter<cu::mccormick<T>> : std::formatter<T>
{
    auto format(const cu::mccormick<T> &x, std::format_context &ctx) const
    {
        auto out = ctx.out();

        out = std::format_to(out, "[");
        out = std::formatter<T>::format(x.box.lb, ctx);
        out = std::format_to(out, ", (");
        out = std::formatter<T>::format(x.cv, ctx);
        out = std::format_to(out, ", ");
        out = std::formatter<T>::format(x.cc, ctx);
        out = std::format_to(out, "), ");
        out = std::formatter<T>::format(x.box.ub, ctx);
        return std::format_to(out, "]");
    }
};

#endif
