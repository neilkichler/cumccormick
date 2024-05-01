#include <cumccormick/arithmetic/basic.cuh>

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

template<typename T>
__global__ void k_cos(mc<T> *x, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cos(x[i]);
    }
}

template<typename T>
__global__ void k_exp(mc<T> *x, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp(x[i]);
    }
}

template<typename T>
__global__ void k_pow(mc<T> *x, int pow_n, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pow(x[i], pow_n);
    }
}

template<typename T>
__global__ void k_sub_const(T x_const, mc<T> *y, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sub(x_const, y[i]);
    }
}

template<typename T>
__global__ void k_sub(mc<T> *x, mc<T> *y, mc<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sub(x[i], y[i]);
    }
}

template<typename T>
__global__ void k_add(mc<T> *x, mc<T> *y, mc<T> *res, int n)
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
cudaGraph_t construct_graph(cuda_ctx &ctx, mc<T> *xs, mc<T> *ys, mc<T> *res, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, int n)
{
    T a = 1.0;
    T b = 100.0;

    auto v0 = d_xs;
    auto v1 = d_ys;

    cudaGraph_t graph;
    cudaStream_t g_stream = ctx.streams[0];

    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    std::vector<cudaGraphNode_t> dependencies;

    using node = cudaGraphNode_t;
    node node_memcpy, node_kernel;

    constexpr int n_blocks            = 128;
    constexpr int n_threads_per_block = 1;

    int n_k = n;

    {
        // copy xs to device buffer d_xs
        cudaMemcpy3DParms memcpy_params = {
            .srcArray = NULL,
            .srcPos   = make_cudaPos(0, 0, 0),
            .srcPtr   = make_cudaPitchedPtr(xs, n * sizeof(*xs), n, 1),
            .dstArray = NULL,
            .dstPos   = make_cudaPos(0, 0, 0),
            .dstPtr   = make_cudaPitchedPtr(d_xs, n * sizeof(*d_xs), n, 1),
            .extent   = make_cudaExtent(n * sizeof(*d_xs), 1, 1),
            .kind     = cudaMemcpyHostToDevice
        };

        CUDA_CHECK(cudaGraphAddMemcpyNode(&node_memcpy, graph, NULL, 0, &memcpy_params));
        dependencies.push_back(node_memcpy);
    }
    {
        // copy ys to device buffer d_ys
        cudaMemcpy3DParms memcpy_params = {
            .srcArray = NULL,
            .srcPos   = make_cudaPos(0, 0, 0),
            .srcPtr   = make_cudaPitchedPtr(ys, n * sizeof(*ys), n, 1),
            .dstArray = NULL,
            .dstPos   = make_cudaPos(0, 0, 0),
            .dstPtr   = make_cudaPitchedPtr(d_ys, n * sizeof(*d_ys), n, 1),
            .extent   = make_cudaExtent(n * sizeof(*d_ys), 1, 1),
            .kind     = cudaMemcpyHostToDevice
        };
        CUDA_CHECK(cudaGraphAddMemcpyNode(&node_memcpy, graph, NULL, 0, &memcpy_params));
        dependencies.push_back(node_memcpy);
    }

    cudaKernelNodeParams kernel_params {
        .func           = nullptr,
        .gridDim        = dim3(n_blocks, 1, 1),
        .blockDim       = dim3(n_threads_per_block, 1, 1),
        .sharedMemBytes = 0,
        .kernelParams   = nullptr,
        .extra          = nullptr
    };

    {
        void *params[4]            = { &a, &v0, &d_res, &n_k };
        kernel_params.func         = (void *)k_sub_const<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    int pow_n = 2;
    auto v2   = d_res;
    {
        void *params[4] { &v2, &pow_n, &d_res, &n_k };
        kernel_params.func         = (void *)k_pow<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v3 = d_res;
    {
        void *params[4] { &v0, &pow_n, &d_res, &n_k };
        kernel_params.func         = (void *)k_pow<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v4 = d_res;
    {
        void *params[4] { &v1, &v4, &d_res, &n };
        kernel_params.func         = (void *)k_sub<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v5 = d_res;
    {
        void *params[4] { &v5, &pow_n, &d_res, &n };
        kernel_params.func         = (void *)k_pow<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v6 = d_res;
    {
        void *params[4] { &b, &v6, &d_res, &n };
        kernel_params.func         = (void *)k_mul<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v7 = d_res;
    {
        void *params[4] { &v3, &v7, &d_res, &n };
        kernel_params.func         = (void *)k_add<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v8 = d_res;
    {
        void *params[3] { &v8, &d_res, &n };
        kernel_params.func         = (void *)k_cos<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v9 = d_res;
    {
        void *params[4] { &v9, &v0, &d_res, &n };
        kernel_params.func         = (void *)k_sub<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v10 = d_res;
    {
        void *params[4] { &v10, &v0, &d_res, &n };
        kernel_params.func         = (void *)k_add<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v11 = d_res;
    {
        double ten = 10.0;
        void *params[4] { &ten, &v10, &d_res, &n };
        kernel_params.func         = (void *)k_mul<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v12 = d_res;
    {
        // copy result from device buffer d_res to res
        cudaMemcpy3DParms memcpy_params = {
            .srcArray = NULL,
            .srcPos   = make_cudaPos(0, 0, 0),
            .srcPtr   = make_cudaPitchedPtr(d_res, n * sizeof(*d_res), n, 1),
            .dstArray = NULL,
            .dstPos   = make_cudaPos(0, 0, 0),
            .dstPtr   = make_cudaPitchedPtr(res, n * sizeof(*res), n, 1),
            .extent   = make_cudaExtent(n * sizeof(*d_res), 1, 1),
            .kind     = cudaMemcpyDeviceToHost
        };
        CUDA_CHECK(cudaGraphAddMemcpyNode(&node_memcpy, graph, dependencies.data(), dependencies.size(), &memcpy_params));
    }

    return graph;
}

void graph_example(cuda_ctx ctx)
{
    using T = double;

    std::vector<mc<T>> xs {
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

    std::vector<mc<T>> ys {
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

    const int n    = xs.size();
    const int size = n * sizeof(mc<T>);

    std::vector<mc<T>> res(n); // TODO: use pinned memory

    mc<T> *d_xs;
    mc<T> *d_ys;
    mc<T> *d_res;

    CUDA_CHECK(cudaMalloc(&d_xs, size));
    CUDA_CHECK(cudaMalloc(&d_ys, size));
    CUDA_CHECK(cudaMalloc(&d_res, size));

    cudaGraph_t graph = construct_graph(ctx, xs.data(), ys.data(), res.data(), d_xs, d_ys, d_res, n);

    cudaGraphNode_t *nodes = nullptr;
    size_t n_nodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &n_nodes));
    printf("CUDA manually created %zu nodes for the graph.\n\n", n_nodes);

    cudaGraphExec_t graph_exec;
    cudaStream_t g_stream = ctx.streams[0];
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    printf("Results (1st Capture): \n");
    for (auto r : res) {
        printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    }

    //
    // Second capture
    //

    // we can reuse the same graph for different inputs
    xs[0] = { .cv = -4.96, .cc = 4.25, .box = { .lb = -8.0, .ub = 8.0 } };
    CUDA_CHECK(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    printf("Results (2nd Capture): \n");
    for (auto r : res) {
        printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    }

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
    graph_example(ctx);

    for (auto &event : events)
        CUDA_CHECK(cudaEventDestroy(event));

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(device_backing_buffer));
    CUDA_CHECK(cudaFreeHost(host_backing_buffer));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
