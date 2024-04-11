#pragma once

#include <cuda_runtime.h>

void basic_kernel(cudaStream_t stream);

void pown_kernel(cudaStream_t stream);

void fn_kernel(cudaStream_t stream);

void tests_pown(cudaStream_t stream, cudaEvent_t event);

void tests_basic(cudaStream_t stream, cudaEvent_t event);

void tests_fn(cudaStream_t stream, cudaEvent_t event);

