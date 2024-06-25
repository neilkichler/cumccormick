#ifndef CUMCCORMICK_EXAMPLES_COMMON_H
#define CUMCCORMICK_EXAMPLES_COMMON_H

#define MCCORMICK_FORMAT "[lb: %g (cv: %g, cc: %g), ub: %g]"

#include <cumccormick/cumccormick.cuh>

template<typename T>
using mc = cu::mccormick<T>;

#endif // CUMCCORMICK_EXAMPLES_COMMON_H
