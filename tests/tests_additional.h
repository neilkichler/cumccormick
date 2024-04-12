#pragma once

#include "tests_common.h"
#include "tests_basic.h"

template<typename T>
void tests_additional(cuda_ctx ctx)
{
    tests_basic(ctx.streams[0], ctx.events[0]);
    tests_pown(ctx.streams[1], ctx.events[1]);
    tests_fn(ctx.streams[2], ctx.events[2]);
    tests_bounds(ctx.streams[3], ctx.events[3]);
}
