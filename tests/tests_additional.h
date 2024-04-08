#pragma once

#include "tests_common.h"
#include "tests_basic.h"

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams, cuda_events events)
{
    tests_basic(streams[0], events[0]);
    tests_pown(streams[1], events[1]);
}
