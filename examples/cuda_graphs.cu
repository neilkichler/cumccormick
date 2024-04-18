#include <cumccormick/arithmetic/basic.cuh>

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../tests/tests_common.h"

__device__ auto rosenbrock(auto x, auto y)
{
    double a = 1.0;
    double b = 100.0;
    return pow(a - x, 2) + b * pow((y - pow(x, 2)), 2);
}

__global__ void series_of_function_calls()
{
    mc<double> x { .cv = -1.96, .cc = 1.25, .box = { .lb = -2.0, .ub = 2.0 } };
    mc<double> y { .cv = -0.55, .cc = 2.50, .box = { .lb = -1.0, .ub = 3.0 } };
    auto rosen = rosenbrock(x, y);
    auto z = cos(rosen);

    printf("Intermediate output is (%.15g, %.15g)\n", y.cv, y.cc);
    printf("Final output is (%.15g, %.15g)\n", z.cv, z.cc);
}

void streaming_example(cuda_ctx ctx)
{
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(ctx.streams[0], cudaStreamCaptureModeGlobal));
    series_of_function_calls<<<1, 1, 0, ctx.streams[0]>>>();
    CUDA_CHECK(cudaStreamEndCapture(ctx.streams[0], &graph));

    cudaGraphNode_t *nodes = nullptr;
    size_t n_nodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &n_nodes));
    printf("Stream capture generated %zu nodes for the graph\n", n_nodes);

    cudaGraphExec_t graph_exe;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exe, graph, nullptr, nullptr, 0));
    cudaStream_t g_stream = ctx.streams[1];
    CUDA_CHECK(cudaGraphLaunch(graph_exe, g_stream)); // NOTE: The cuda graph stream can differ from
    CUDA_CHECK(cudaStreamSynchronize(g_stream));      //       the stream used for recording.

    CUDA_CHECK(cudaGraphExecDestroy(graph_exe));
    CUDA_CHECK(cudaGraphDestroy(graph));
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
