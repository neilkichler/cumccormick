#include <array>
#include <cstdio>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "../tests/tests_common.h"

template<typename T>
__global__ void k_cos(mc<T> *x, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = cos(x[i]);
    }
}

template<typename T>
__global__ void k_exp(mc<T> *x, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = exp(x[i]);
    }
}

template<typename T>
__global__ void k_pow(mc<T> *x, int pow_n, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = pow(x[i], pow_n);
    }
}

template<typename T>
__global__ void k_sub_const(T x_const, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = sub(x_const, y[i]);
    }
}

template<typename T>
__global__ void k_sub(mc<T> *x, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = sub(x[i], y[i]);
    }
}

template<typename T>
__global__ void k_add(mc<T> *x, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = add(x[i], y[i]);
    }
}

template<typename T>
__global__ void k_mul(T x_const, mc<T> *y, mc<T> *res, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = mul(x_const, y[i]);
    }
}

template<typename T>
cudaGraph_t construct_graph(mc<T> *xs, mc<T> *ys, mc<T> *res, mc<T> *d_xs, mc<T> *d_ys, mc<T> *d_res, mc<T> *d_tmp, int n)
{
    T a = 1.0;
    T b = 100.0;

    auto v0 = d_xs;
    auto v1 = d_ys;

    cudaGraph_t graph;

    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    std::vector<cudaGraphNode_t> dependencies;

    using node = cudaGraphNode_t;
    node node_memcpy, node_kernel;

    constexpr int n_blocks            = 1024;
    constexpr int n_threads_per_block = 1024;

    int n_k = n;

    // Implements hardcoded rosenbrock function in terms of CUDA graph nodes.
    {
        // copy xs to device buffer d_xs
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&node_memcpy, graph, NULL, 0,
                                            d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
        dependencies.push_back(node_memcpy);
    }
    {
        // copy ys to device buffer d_ys
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&node_memcpy, graph, NULL, 0,
                                            d_ys, ys, n * sizeof(*xs), cudaMemcpyHostToDevice));
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
        void *params[4] { &v2, &pow_n, &d_tmp, &n_k };
        kernel_params.func         = (void *)k_pow<T>;
        kernel_params.kernelParams = params;

        CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
        dependencies.clear();
        dependencies.push_back(node_kernel);
    }
    auto v3 = d_tmp;
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
    {
        // copy result from device buffer d_res to res
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&node_memcpy, graph, dependencies.data(), dependencies.size(),
                                            res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));
    }

    return graph;
}

void graph_example(cuda_ctx ctx)
{
    using T = double;

    mc<T> *xs;
    mc<T> *ys;
    mc<T> *res;

    constexpr int n = 100 * 1024;
    // constexpr int n                   = 32 * 1024 * 1024;

    CUDA_CHECK(cudaMallocHost(&xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMallocHost(&ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMallocHost(&res, n * sizeof(*res)));

    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i]    = { .cv = -v, .cc = v, .box = {{ .lb = -v, .ub = v }} };
        ys[i]    = { .cv = -v, .cc = v, .box = {{ .lb = -v, .ub = v }} };
    }

    const int size = n * sizeof(mc<T>);

    mc<T> *d_xs;
    mc<T> *d_ys;
    mc<T> *d_res;
    mc<T> *d_tmp;

    CUDA_CHECK(cudaMalloc(&d_xs, size));
    CUDA_CHECK(cudaMalloc(&d_ys, size));
    CUDA_CHECK(cudaMalloc(&d_res, size));
    CUDA_CHECK(cudaMalloc(&d_tmp, size));

    cudaGraph_t graph = construct_graph(xs, ys, res, d_xs, d_ys, d_res, d_tmp, n);

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
    auto r = res[0];
    printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    r = res[1];
    printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

    //
    // Second capture
    //

    // we can reuse the same graph for different inputs
    xs[0] = { .cv = -4.96, .cc = 4.25, .box = {{ .lb = -8.0, .ub = 8.0 }} };
    CUDA_CHECK(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    printf("Results (2nd Capture): \n");
    r = res[0];
    printf(MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);

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
