#include "tests_basic.h"

void tests_basic(cudaStream_t stream, cudaEvent_t event)
{
    basic_kernel(stream);
}

void tests_pown(cudaStream_t stream, cudaEvent_t event)
{
    pown_kernel(stream);
}

void tests_fn(cudaStream_t stream, cudaEvent_t event)
{
    fn_kernel(stream);
}

void tests_bounds(cudaStream_t stream, cudaEvent_t event)
{
    bounds_kernel(stream);
}
