#include <array>
#include <cstdio>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

__device__ auto rosenbrock(auto x, auto y)
{
    double a = 1.0;
    double b = 100.0;
    return pow(a - x, 2) + b * pow((y - pow(x, 2)), 2);
}

__device__ auto model(auto x, auto y)
{
    auto rosen = rosenbrock(x, y);
    auto z     = cos(rosen) - x + x;
    z          = 10.0 * z;
    return z;
}

__global__ void k_cos(auto *x, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cos(x[i]);
    }
}

__global__ void k_exp(auto *x, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp(x[i]);
    }
}

__global__ void k_pow(auto *x, int pow_n, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pow(x[i], pow_n);
    }
}

__global__ void k_sub(auto x_const, auto *y, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sub(x_const, y[i]);
    }
}

__global__ void k_sub(auto *x, auto *y, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sub(x[i], y[i]);
    }
}

__global__ void k_add(auto *x, auto *y, auto *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = add(x[i], y[i]);
    }
}

template<typename T>
__global__ void k_mul(T x_const, mc<T> *y, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mul(x_const, y[i]);
    }
}

template<typename T>
void multiple_kernels_model(cuda_ctx &ctx, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, int n)
{
    T a = 1.0;
    T b = 100.0;

    auto s = ctx.streams[0];

    // DAG manual serialization of model (see above).

    // v0 = x
    auto v0 = d_xs;
    // v1 = y
    auto v1 = d_ys;
    // v2 = a - v0
    k_sub<<<128, 1, 0, s>>>(a, v0, d_res, n);
    auto v2 = d_res;
    // v3 = pow(v2, 2)
    k_pow<<<128, 1, 0, s>>>(v2, 2, d_res, n);
    auto v3 = d_res;
    // v4 = pow(v0, 2)
    k_pow<<<128, 1, 0, s>>>(v0, 2, d_res, n);
    auto v4 = d_res;
    // v5 = v1 - v4
    k_sub<<<128, 1, 0, s>>>(v1, v4, d_res, n);
    auto v5 = d_res;
    // v6 = pow(v5, 2)
    k_pow<<<128, 1, 0, s>>>(v5, 2, d_res, n);
    auto v6 = d_res;
    // v7 = b * v6
    k_mul<<<128, 1, 0, s>>>(b, v6, d_res, n);
    auto v7 = d_res;
    // v8 = v3 + v7
    k_add<<<128, 1, 0, s>>>(v3, v7, d_res, n);
    auto v8 = d_res;
    // v9 = cos(v8);
    k_cos<<<128, 1, 0, s>>>(v8, d_res, n);
    auto v9 = d_res;
    // v10 = v9 - v0
    k_sub<<<128, 1, 0, s>>>(v9, v0, d_res, n);
    auto v10 = d_res;
    // v11 = v10 + v0
    k_add<<<128, 1, 0, s>>>(v10, v0, d_res, n);
    auto v11 = d_res;
    // v12 = 10.0 * v10
    k_mul<<<128, 1, 0, s>>>(10.0, v10, d_res, n);
    auto v12 = d_res;
}

void streaming_example(cuda_ctx ctx)
{
    std::vector<mc<double>> xs {
        { .cv = -1.96, .cc = 1.25, .box = { .lb = -2.0, .ub = 2.0 } },
        { .cv = 0.6, .cc = 0.65, .box = { .lb = 0.0, .ub = 0.7 } },
        { .cv = 7.6, .cc = 7.65, .box = { .lb = 6.1, .ub = 7.7 } },
        { .cv = 50.6, .cc = 100.65, .box = { .lb = 50.0, .ub = 100.7 } },
        { .cv = 3.6, .cc = 3.85, .box = { .lb = -4.1, .ub = 7.7 } },
        { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.1, .ub = 0.1 } },
        { .cv = -0.01, .cc = 0.01, .box = { .lb = -0.01, .ub = 0.01 } },
        { .cv = 10000.01, .cc = 10001.01, .box = { .lb = 0.0, .ub = 100000.0 } },
        { .cv = -3.96, .cc = -3.25, .box = { .lb = -4.1, .ub = -3.1 } },
    };

    std::vector<mc<double>> ys {
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
        { .cv = 0.5, .cc = 2.5, .box = { .lb = 0.0, .ub = 3.0 } },
        { .cv = -0.5, .cc = 0.5, .box = { .lb = -1.0, .ub = 3.0 } },
    };

    using T = double;
    mc<T> *d_xs;
    mc<T> *d_ys;
    mc<T> *d_res;

    const int n_xs     = xs.size();
    const int n_ys     = xs.size();
    const int xs_size  = n_xs * sizeof(mc<T>);
    const int ys_size  = n_ys * sizeof(mc<T>);
    const int res_size = xs_size;

    CUDA_CHECK(cudaMalloc(&d_xs, xs_size));
    CUDA_CHECK(cudaMalloc(&d_ys, ys_size));
    CUDA_CHECK(cudaMalloc(&d_res, res_size));
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), xs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), ys_size, cudaMemcpyHostToDevice));

    cudaGraph_t graph;

    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    multiple_kernels_model(ctx, d_xs, d_ys, d_res, n_xs);
    CUDA_CHECK(cudaStreamEndCapture(ctx.streams[0], &graph));

    cudaGraphNode_t *nodes = nullptr;
    size_t n_nodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &n_nodes));
    printf("Stream capture generated %zu nodes for the graph.\n\n", n_nodes);

    cudaGraphExec_t graph_exe;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exe, graph, nullptr, nullptr, 0));
    cudaStream_t g_stream = ctx.streams[1];
    CUDA_CHECK(cudaGraphLaunch(graph_exe, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    std::vector<mc<T>> res(n_xs);
    CUDA_CHECK(cudaMemcpy(res.data(), d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));

    printf("Results (1st Capture): \n");
    for (auto r : res) {
        printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    }

    // reset result buffers
    res.clear();
    res.resize(n_xs);
    CUDA_CHECK(cudaMemset(d_res, 0, n_xs * sizeof(mc<T>)));

    CUDA_CHECK(cudaGraphDebugDotPrint(graph, "stream_capture_1.dot", 0));


    //
    // Second capture
    //

    // we can reuse the same graph for different inputs
    xs[0] = { .cv = -4.96, .cc = 4.25, .box = { .lb = -8.0, .ub = 8.0 } };

    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), xs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), ys_size, cudaMemcpyHostToDevice));

    
    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    multiple_kernels_model(ctx, d_xs, d_ys, d_res, n_xs);
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
    CUDA_CHECK(cudaMemcpy(res.data(), d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));

    printf("Results (2nd Capture): \n");
    for (auto r : res) {
        printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    }

    CUDA_CHECK(cudaGraphDebugDotPrint(graph, "stream_capture_2.dot", 0));

    CUDA_CHECK(cudaGraphExecDestroy(graph_exe));
    CUDA_CHECK(cudaGraphDestroy(graph));

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
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
