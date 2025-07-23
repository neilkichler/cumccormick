#include <array>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

template<typename T>
__device__ T model(T *x, int n_vars)
{
    T res {};

    // Slow for loop on purpose to simulate large model
    for (int i = 0; i < n_vars; i++) {
        auto x_i = x[i];
        res += sqr(cos(x_i - 1)) + pow(x_i - 1, 3) + pow(x_i - 1, 4);
    }

    return res - 1;
}

__global__ void k_cos(auto *x, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = cos(x[i]);
        auto r = res[i];
        // printf("[%d] cos is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_sqr(auto *x, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = sqr(x[i]);
        auto r = res[i];
        // printf("[%d] sqr is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_exp(auto *x, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = exp(x[i]);
        auto r = res[i];
        // printf("[%d] exp is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_pow(auto *x, int pow_n, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] pow_n is: %d\n", i, pow_n);
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        res[i] = pow(x[i], pow_n);
        auto r = res[i];
        // printf("[%d] pow is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_sub(auto x_const, auto *y, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);
        res[i] = sub(x_const, y[i]);
        auto r = res[i];
        // printf("[%d] sub const is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_sub(auto *x, auto y_const, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);
        res[i] = sub(x[i], y_const);
        auto r = res[i];
        // printf("[%d] sub const is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_sub(auto *x, auto *y, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x : [%f, %f] ", i, x[i].box.lb, x[i].box.ub);
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);
        res[i] = sub(x[i], y[i]);
        auto r = res[i];
        // printf("[%d] sub res is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

__global__ void k_add(auto *x, auto *y, auto *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x is: " MCCORMICK_FORMAT "\n", i, x[i].box.lb, x[i].cv, x[i].cc, x[i].box.ub);
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);
        res[i] = add(x[i], y[i]);
        auto r = res[i];
        // printf("[%d] add res is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

template<typename T>
__global__ void k_mul(mc<T> *x, mc<T> *y, mc<T> *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x_const is %f\n", i, x_const);
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);
        res[i] = mul(x[i], y[i]);
        auto r = res[i];
        // printf("[%d] mul res is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

template<typename T>
__global__ void k_mul(T x_const, mc<T> *y, mc<T> *res, int n)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i < n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] x_const is %f\n", i, x_const);
        // printf("[%d] y is: " MCCORMICK_FORMAT "\n", i, y[i].box.lb, y[i].cv, y[i].cc, y[i].box.ub);

        res[i] = mul(x_const, y[i]);
        // auto r = res[i];

        // for (int j = 0; j < n_vars; j++) {
        //     res[i * n_vars + j] = mul(x_const, y[j]);
        // }

        // for (int i = 0; i < n_vars; i++) {
        //     auto x_i = x[i];
        //     res += sqr(cos(x_i - 1)) + pow(x_i - 1, 3) + pow(x_i - 1, 4);
        // }

        // printf("[%d] mul res is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        // printf("[%d] -------------\n", i);
    }
}

template<typename T>
__global__ void k_print(mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // printf("[%d] res is: " MCCORMICK_FORMAT "\n", i, res[i].box.lb, res[i].cv, res[i].cc, res[i].box.ub);
        printf("%s", std::format("[{}] res is: {}\n", res[i]).c_str());

        // auto r = res[i];

        // for (int j = 0; j < n_vars; j++) {
        //     res[i * n_vars + j] = mul(x_const, y[j]);
        // }

        // for (int i = 0; i < n_vars; i++) {
        //     auto x_i = x[i];
        //     res += sqr(cos(x_i - 1)) + pow(x_i - 1, 3) + pow(x_i - 1, 4);
        // }

        // printf("[%d] mul res is: " MCCORMICK_FORMAT "\n", i, r.box.lb, r.cv, r.cc, r.box.ub);
        printf("[%d] -------------\n", i);
    }
}

// Performs a reduction step and updates numTotal with how many are remaining
template<typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads)
{
    return cg::reduce(threads, in, cg::plus<T>());
}

template<class T>
__global__ void cg_reduce(T *g_idata, T *g_odata, unsigned int n)
{
    // Shared memory for intermediate steps
    T *sdata = SharedMemory<T>();
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    unsigned int ctaSize     = cta.size();
    unsigned int numCtas     = gridDim.x;
    unsigned int threadRank  = cta.thread_rank();
    unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

    T threadVal = 0;
    {
        unsigned int i           = threadIndex;
        unsigned int indexStride = (numCtas * ctaSize);
        while (i < n) {
            threadVal += g_idata[i];
            i += indexStride;
        }
        sdata[threadRank] = threadVal;
    }

    // Wait for all tiles to finish and reduce within CTA
    {
        unsigned int ctaSteps = tile.meta_group_size();
        unsigned int ctaIndex = ctaSize >> 1;
        while (ctaIndex >= 32) {
            cta.sync();
            if (threadRank < ctaIndex) {
                threadVal += sdata[threadRank + ctaIndex];
                sdata[threadRank] = threadVal;
            }
            ctaSteps >>= 1;
            ctaIndex >>= 1;
        }
    }
    // Shuffle redux instead of smem redux
    {
        cta.sync();
        if (tile.meta_group_rank() == 0) {
            threadVal = cg_reduce_n(threadVal, tile);
        }
    }

    if (threadRank == 0)
        g_odata[blockIdx.x] = threadVal;
}

constexpr int n_blocks  = 1024;
constexpr int n_threads = 512;

// constexpr int n_blocks  = 4;
// constexpr int n_threads = 32;

template<typename T>
void model_captured(cuda_ctx &ctx, mc<T> *d_xs, mc<T> *d_res, mc<T> *d_tmp, int n_elems, int n_vars)
{
    T one = 1.0;

    auto s = ctx.streams[0];

    // for (int i = 0; i < n_vars; i++) {
    //     auto x_i = x[i];
    //     res += sqr(cos(x_i - 1)) + pow(x_i - 1, 3) + pow(x_i - 1, 4);
    // }
    //
    // d_res[i] = d_res - 1;

    mc<T> *d_res_accum;
    CUDA_CHECK(cudaMallocAsync(&d_res_accum, n_elems * sizeof(*d_res_accum), s));
    CUDA_CHECK(cudaMemsetAsync(d_res_accum, 0, n_elems * sizeof(*d_res_accum), s));

    for (int j = 0; j < n_vars; j++) {
        // v0 = x
        auto v0 = &d_xs[j];
        // v1 = x - 1
        k_sub<<<n_blocks, n_threads, 0, s>>>(v0, one, d_res, n_elems);
        auto v1 = d_res;
        // v2 = cos(v1)
        k_cos<<<n_blocks, n_threads, 0, s>>>(v1, d_res, n_elems);
        auto v2 = d_res;
        // v3 = sqr(v2) // tmp
        k_sqr<<<n_blocks, n_threads, 0, s>>>(v2, d_tmp, n_elems);
        auto v3 = d_tmp;
        // v4 = x - 1
        k_sub<<<n_blocks, n_threads, 0, s>>>(v0, one, d_res, n_elems);
        auto v4 = d_res;
        // v5 = pow(v4, 3)
        k_pow<<<n_blocks, n_threads, 0, s>>>(v4, 3, d_res, n_elems);
        auto v5 = d_res;
        // v6 = v3 + v5 // tmp
        k_add<<<n_blocks, n_threads, 0, s>>>(v3, v5, d_tmp, n_elems);
        auto v6 = d_tmp;
        // v7 = x - 1
        k_sub<<<n_blocks, n_threads, 0, s>>>(v0, one, d_res, n_elems);
        auto v7 = d_res;
        // v8 = pow(v6, 4)
        k_pow<<<n_blocks, n_threads, 0, s>>>(v7, 4, d_res, n_elems);
        auto v8 = d_res;
        // v9 = v6 + v8
        k_add<<<n_blocks, n_threads, 0, s>>>(v6, v8, d_res, n_elems);
        auto v9 = d_res;

        k_add<<<n_blocks, n_threads, 0, s>>>(d_res_accum, v9, d_res_accum, n_elems);
    }

    k_sub<<<n_blocks, n_threads, 0, s>>>(d_res_accum, one, d_res_accum, n_elems);
    // k_print<<<n_blocks, n_threads, 0, s>>>(d_res_accum, n_elems);

    CUDA_CHECK(cudaMemcpyAsync(d_res, d_res_accum, n_elems * sizeof(*d_res), cudaMemcpyDeviceToDevice, s));
}
// int main()
// {
//
//     constexpr int n_elems = 1024;
//     constexpr int n_vars  = 10000;
//
//     using T = mc<double>;
//
//     mc<double> *d_xs, *d_res;
//     CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*d_xs)));
//     CUDA_CHECK(cudaMalloc(&d_res, n_elems * sizeof(*d_res)));
//
//     T *xs;
//     T *res;
//
//     CUDA_CHECK(cudaMallocHost(&xs, n_elems * n_vars * sizeof(*xs)));
//     CUDA_CHECK(cudaMallocHost(&res, n_elems * sizeof(*res)));
//
//     for (int i = 0; i < n_elems; i++) {
//         double v = i;
//         for (int j = 0; j < n_vars; j++) {
//             xs[i * n_vars + j] = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
//         }
//         // double v = 0;
//         // xs[i]    = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } };
//         // printf("setting xs[i]: %d %f %f %f\n", i, xs[i].cv, xs[i].cc, xs[i].box.ub);
//     }
//
//     CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*d_xs), cudaMemcpyHostToDevice));
//     // k_model<<<n_blocks, n_threads>>>(d_xs, d_res, n_elems, n_vars);
//     model_captured<<<n_blocks, n_threads>>>(d_xs, d_res, n_elems, n_vars);
//
//     auto err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__,
//                 __FILE__, __LINE__, cudaGetErrorString(err),
//                 cudaGetErrorName(err), err);
//     }
//
//     CUDA_CHECK(cudaMemcpy(res, d_res, n_elems * sizeof(*res), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaDeviceSynchronize());
//
//     auto r = res[0];
//     printf("custom([0, (0, 0), 0]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
//
//     // Second run
//     xs[0] = { .cv = 0.5, .cc = 1.0, .box = { .lb = 0.0, .ub = 2.0 } };
//
//     CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * sizeof(*d_xs), cudaMemcpyHostToDevice));
//
//     model_captured<<<n_blocks, n_threads>>>(d_xs, d_res, n_elems, n_vars);
//
//     CUDA_CHECK(cudaMemcpy(res, d_res, n_elems * sizeof(*res), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaDeviceSynchronize());
//
//     r = res[0];
//     printf("custom([0.0, (0.5, 1.0), 2.0]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
//
//     CUDA_CHECK(cudaFree(d_xs));
//     CUDA_CHECK(cudaFree(d_res));
// }

void streaming_example(cuda_ctx ctx)
{
    constexpr int n_elems = 1024;
    constexpr int n_vars  = 10000;
    constexpr int n       = n_elems * n_vars;

    using T = double;
    mc<T> *xs;
    mc<T> *res;

    CUDA_CHECK(cudaMallocHost(&xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMallocHost(&res, n * sizeof(*res)));

    for (int i = 0; i < n_elems; i++) {
        double v = i;
        for (int j = 0; j < n_vars; j++) {
            xs[i * n_vars + j] = { -v, v };
        }
    }

    mc<T> *d_xs;
    mc<T> *d_res;
    mc<T> *d_tmp;

    const int n_xs = n;
    // const int n_ys = n;

    // const int n_xs     = xs.size();
    // const int n_ys     = xs.size();
    const int xs_size  = n_xs * sizeof(mc<T>);
    const int res_size = xs_size;

    CUDA_CHECK(cudaMalloc(&d_xs, xs_size));
    CUDA_CHECK(cudaMalloc(&d_res, res_size));
    CUDA_CHECK(cudaMalloc(&d_tmp, res_size));
    // CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), xs_size, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), ys_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, xs_size, cudaMemcpyHostToDevice));

    cudaGraph_t graph;

    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    model_captured(ctx, d_xs, d_res, d_tmp, n_elems, n_vars);
    CUDA_CHECK(cudaStreamEndCapture(ctx.streams[0], &graph));

    cudaGraphNode_t *nodes = nullptr;
    size_t n_nodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &n_nodes));
    println("Stream capture generated {} nodes for the graph.\n", n_nodes);

    cudaGraphExec_t graph_exe;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exe, graph, nullptr, nullptr, 0));
    cudaStream_t g_stream = ctx.streams[1];
    CUDA_CHECK(cudaGraphLaunch(graph_exe, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    // std::vector<mc<T>> res(n_xs);
    // CUDA_CHECK(cudaMemcpy(res.data(), d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res, d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    println("Results (1st Capture): ");
    // for (auto r : res) {
    //     printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    // }
    auto r = res[0];
    println("{}", r);
    // r = res[1];
    // printf("rosenbrok([-1, (-1, 1), 1]) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

#if 1
    // reset result buffers
    // res.clear();
    // res.resize(n_xs);
    CUDA_CHECK(cudaMemset(d_res, 0, n_xs * sizeof(mc<T>)));
    CUDA_CHECK(cudaMemset(d_tmp, 0, n_xs * sizeof(mc<T>)));

    // CUDA_CHECK(cudaGraphDebugDotPrint(graph, "stream_capture_1.dot", 0));

    //
    // Second capture
    //

    // we can reuse the same graph for different inputs
    xs[0] = { 0.0, 0.5, 1.0, 2.0 };

    // CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), xs_size, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), ys_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, xs_size, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys, ys_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    // rosenbrock_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
    // beale_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
    model_captured(ctx, d_xs, d_res, d_tmp, n_elems, n_vars);
    CUDA_CHECK(cudaStreamEndCapture(ctx.streams[0], &graph));

    cudaGraphExecUpdateResultInfo update_info;
    if (cudaGraphExecUpdate(graph_exe, graph, &update_info) != cudaSuccess) {
        // graph update failed -> create a new one
        printf("Failed to update the graph, creating a new one.\n");
        CUDA_CHECK(cudaGraphExecDestroy(graph_exe));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exe, graph, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(graph_exe, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    // CUDA_CHECK(cudaMemcpy(res.data(), d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res, d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));

    println("Results (2nd Capture): ");
    // for (auto r : res) {
    //     printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    // }
    r = res[0];
    println("{}", r);

    // CUDA_CHECK(cudaGraphDebugDotPrint(graph, "stream_capture_2.dot", 0));

#endif
    CUDA_CHECK(cudaGraphExecDestroy(graph_exe));
    CUDA_CHECK(cudaGraphDestroy(graph));

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));
}

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    std::size_t n_bytes = 16 * 1024 * 2 * sizeof(double);
    std::array<cuda_buffer, n_streams> buffers {};

    char *host_backing_buffer;
    char *device_backing_buffer;
    CUDA_CHECK(cudaMallocHost(&host_backing_buffer, buffers.size() * n_bytes));
    CUDA_CHECK(cudaMalloc(&device_backing_buffer, buffers.size() * n_bytes));

    std::size_t offset = 0;
    for (auto &buffer : buffers) {
        buffer.host   = host_backing_buffer + offset;
        buffer.device = device_backing_buffer + offset;
        offset += n_bytes;
    }

    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::array<cudaEvent_t, n_streams> events {};
    for (auto &event : events)
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    cuda_ctx ctx { buffers, streams, events };
    streaming_example(ctx);

    for (auto &event : events)
        CUDA_CHECK(cudaEventDestroy(event));

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(device_backing_buffer));
    CUDA_CHECK(cudaFreeHost(host_backing_buffer));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
