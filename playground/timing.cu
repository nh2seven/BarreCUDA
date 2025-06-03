#include <iostream>
#include <cuda_runtime.h>

__global__ void dummy_kernel(float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = idx * 2.0f;
}

int main()
{
    const int N = 1024;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start and stop events around the kernel launch
    cudaEventRecord(start);
    dummy_kernel<<<4, 256>>>(d_out);
    cudaEventRecord(stop);

    // Wait for the kernel to finish
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;

    // Clean up
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}