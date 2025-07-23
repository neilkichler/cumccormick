#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

__global__ void k_cos(auto *x, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = cos(x[i]);
        auto r = res[i];
    }
}

__global__ void k_exp(auto *x, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = exp(x[i]);
        auto r = res[i];
    }
}

__global__ void k_pow(auto *x, int pow_n, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = pow(x[i], pow_n);
        auto r = res[i];
    }
}

__global__ void k_sub(auto x_const, auto *y, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = sub(x_const, y[i]);
        auto r = res[i];
    }
}

__global__ void k_sub(auto *x, auto *y, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = sub(x[i], y[i]);
        auto r = res[i];
    }
}

__global__ void k_add(auto *x, auto *y, auto *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = add(x[i], y[i]);
        auto r = res[i];
    }
}

template<typename T>
__global__ void k_mul(mc<T> *x, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = mul(x[i], y[i]);
        auto r = res[i];
    }
}

template<typename T>
__global__ void k_mul(T x_const, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = mul(x_const, y[i]);
        auto r = res[i];
    }
}

constexpr int n_blocks            = 1024;
constexpr int n_threads_per_block = 1024;

template<typename T>
void rosenbrock_captured(cuda_ctx &ctx, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, mc<T> *d_tmp, int n)
{
    T a = 1.0;
    T b = 100.0;

    auto s = ctx.streams[0];

    // DAG manual serialization of rosenbrock function.

    // v0 = x
    auto v0 = d_xs;
    // v1 = y
    auto v1 = d_ys;
    // v2 = a - v0
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(a, v0, d_res, n);
    auto v2 = d_res;
    // v3 = pow(v2, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v2, 2, d_tmp, n);
    auto v3 = d_tmp; // TODO: this will be overriden!!!
    // v4 = pow(v0, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v0, 2, d_res, n);
    auto v4 = d_res;
    // v5 = v1 - v4
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(v1, v4, d_res, n);
    auto v5 = d_res;
    // v6 = pow(v5, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v5, 2, d_res, n);
    auto v6 = d_res;
    // v7 = b * v6
    k_mul<<<n_blocks, n_threads_per_block, 0, s>>>(b, v6, d_res, n);
    auto v7 = d_res;
    // v8 = v3 + v7
    k_add<<<n_blocks, n_threads_per_block, 0, s>>>(v3, v7, d_res, n);
}

template<typename T>
void beale_captured(cuda_ctx &ctx, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, mc<T> *d_tmp, int n)
{
    T one = 1.0;
    T a   = 1.5;
    T b   = 2.25;
    T c   = 2.625;

    auto s = ctx.streams[0];

    // DAG manual serialization of beale function as defined below:

    // __device__ auto beale(auto x, auto y)
    // {
    //     return pow(1.5 - x * (1 - y), 2)
    //         + pow(2.25 - x * (1 - sqr(y)), 2)
    //         + pow(2.625 - x * (1 - pow(y, 3)), 2);
    // }

    // v0 = x
    auto v0 = d_xs;
    // v1 = y
    auto v1 = d_ys;
    // v2 = 1 - v1
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(one, v1, d_res, n);
    auto v2 = d_res;
    // v3 = v0 * v2
    k_mul<<<n_blocks, n_threads_per_block, 0, s>>>(v0, v2, d_res, n);
    auto v3 = d_res;
    // v4 = 1.5 - v3
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(a, v3, d_res, n);
    auto v4 = d_res;
    // v5 = pow(v4, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v4, 2, d_tmp, n);
    auto v5 = d_tmp;

    // v6 = pow(v1, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v1, 2, d_res, n);
    auto v6 = d_res;
    // v7 = v0 * v6
    k_mul<<<n_blocks, n_threads_per_block, 0, s>>>(v0, v6, d_res, n);
    // v8 = 2.25 - v7
    auto v7 = d_res;
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(b, v7, d_res, n);
    auto v8 = d_res;
    // v9 = pow(v8, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v8, 2, d_res, n);
    auto v9 = d_res;

    // v9a = v5 + v9
    k_add<<<n_blocks, n_threads_per_block, 0, s>>>(v5, v9, d_tmp, n);
    auto v9a = d_tmp;

    // v10 = pow(v1, 3)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v1, 3, d_res, n);
    auto v10 = d_res;
    // v11 = 1 - v10
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(one, v10, d_res, n);
    auto v11 = d_res;
    // v12 = v0 * v11
    k_mul<<<n_blocks, n_threads_per_block, 0, s>>>(v0, v11, d_res, n);
    auto v12 = d_res;
    // v13 = 2.625 - v12
    k_sub<<<n_blocks, n_threads_per_block, 0, s>>>(c, v12, d_res, n);
    auto v13 = d_res;
    // v14 = pow(v13, 2)
    k_pow<<<n_blocks, n_threads_per_block, 0, s>>>(v13, 2, d_res, n);
    auto v14 = d_res;
    // v15 = v9a + v14
    k_add<<<n_blocks, n_threads_per_block, 0, s>>>(v9a, v14, d_res, n);
    auto v15 = d_res;
}

void streaming_example(cuda_ctx ctx)
{
    // constexpr int n                   = 32 * 1024 * 1024;
    constexpr int n = 100 * 1024;

    using T = double;
    mc<T> *xs;
    mc<T> *ys;
    mc<T> *res;

    CUDA_CHECK(cudaMallocHost(&xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMallocHost(&ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMallocHost(&res, n * sizeof(*res)));

    for (int i = 0; i < n; i++) {
        double v = i; // + 1;
        // double v = i + 1;
        xs[i] = {{ .lb = -v, .cv = -v, .cc = v, .ub = v }};
        ys[i] = {{ .lb = -v, .cv = -v, .cc = v, .ub = v }};
    }

    mc<T> *d_xs;
    mc<T> *d_ys;
    mc<T> *d_res;
    mc<T> *d_tmp;

    const int n_xs     = n;
    const int n_ys     = n;
    const int xs_size  = n_xs * sizeof(mc<T>);
    const int ys_size  = n_ys * sizeof(mc<T>);
    const int res_size = xs_size;

    CUDA_CHECK(cudaMalloc(&d_xs, xs_size));
    CUDA_CHECK(cudaMalloc(&d_ys, ys_size));
    CUDA_CHECK(cudaMalloc(&d_res, res_size));
    CUDA_CHECK(cudaMalloc(&d_tmp, res_size));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, xs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, ys_size, cudaMemcpyHostToDevice));

    cudaGraph_t graph;

    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    // rosenbrock_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
    beale_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
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

    CUDA_CHECK(cudaMemcpy(res, d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    println("Results (1st Capture): ");
    auto r = res[0];
    println("{}", r);
    r = res[1];
    println("rosenbrok([-1, (-1, 1), 1]) = {}", r);

    CUDA_CHECK(cudaMemset(d_res, 0, n_xs * sizeof(mc<T>)));
    CUDA_CHECK(cudaMemset(d_tmp, 0, n_xs * sizeof(mc<T>)));

    // CUDA_CHECK(cudaGraphDebugDotPrint(graph, "stream_capture_1.dot", 0));

    //
    // Second capture
    //

    // we can reuse the same graph for different inputs
    xs[0] = {{ .lb = -8.0, .cv = -4.96, .cc = 4.25, .ub = 8.0 }};

    CUDA_CHECK(cudaMemcpy(d_xs, xs, xs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, ys_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    // rosenbrock_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
    beale_captured(ctx, d_xs, d_ys, d_res, d_tmp, n_xs);
    CUDA_CHECK(cudaStreamEndCapture(ctx.streams[0], &graph));

    cudaGraphExecUpdateResultInfo update_info;
    if (cudaGraphExecUpdate(graph_exe, graph, &update_info) != cudaSuccess) {
        // graph update failed -> create a new one
        println("Failed to update the graph, creating a new one.");
        CUDA_CHECK(cudaGraphExecDestroy(graph_exe));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exe, graph, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(graph_exe, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    CUDA_CHECK(cudaMemcpy(res, d_res, n_xs * sizeof(mc<T>), cudaMemcpyDeviceToHost));

    println("Results (2nd Capture): ");
    r = res[0];
    println("{}", r);

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
