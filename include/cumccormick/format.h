#ifndef CUMCCORMICK_FORMAT_H
#define CUMCCORMICK_FORMAT_H

#include <cumccormick/cumccormick.h>

#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, cu::mccormick<T> x)
{
    return os << "[lb: " << inf(x) << ", (cv: " << x.cv << ", cc: " << x.cc << "), ub: " << sup(x) << "]";
}

} // namespace cu

#endif
