#pragma once

#include "tests_common.h"
#include "tests_basic.h"
#include "tests_randomized.h"
#include "tests_root.h"

template<typename T>
void tests_additional(cuda_ctx ctx)
{
    constexpr auto n = ctx.buffers.size();
    tests_basic(ctx.streams[0 % n], ctx.events[0 % n]);
    tests_pown(ctx.streams[1 % n], ctx.events[1 % n]);
    tests_fn(ctx.streams[2 % n], ctx.events[2 % n]);
    tests_bounds(ctx.streams[3 % n], ctx.events[3 % n]);
    tests_root(ctx.streams[4 % n], ctx.events[4 % n]);
    tests_randomized(ctx.streams[5 % n], ctx.events[5 % n]);
}
