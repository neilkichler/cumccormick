#pragma once

#include <cuda_runtime.h>

void basic_kernel(cudaStream_t stream);

void tests_basic(cudaStream_t stream, cudaEvent_t event);

