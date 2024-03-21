#pragma once

#include "tests_common.h"
#include "tests_basic.h"

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams, cuda_events events)
{
    tests_basic(streams[0], events[0]);
}
