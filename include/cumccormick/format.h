#ifndef CUMCCORMICK_FORMAT_H
#define CUMCCORMICK_FORMAT_H

#include <cumccormick/cumccormick.h>

#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, cu::mccormick<T> x)
{
    return os << "[lb: " << x.box.lb << ", (cv: " << x.cv << ", cc: " << x.cc << "), ub: " << x.box.ub << "]";
}

} // namespace cu

#endif
