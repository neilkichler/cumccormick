#pragma once

#include "tests_common.h"
#include "tests_basic.h"
#include "tests_randomized.h"
#include "tests_root.h"

template<typename T>
void tests_additional(cuda_ctx ctx)
{
    constexpr auto n = ctx.buffers.size();
    test_basic(ctx.streams[0 % n]);
    test_pown(ctx.streams[1 % n]);
    test_fn(ctx.streams[2 % n]);
    test_bounds(ctx.streams[3 % n]);
    test_root(ctx.streams[4 % n]);
    tests_randomized(ctx.streams);
}
