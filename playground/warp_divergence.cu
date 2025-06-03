#include <iostream>
#include <cuda_runtime.h>

// Kernel to demonstrate warp divergence; threads in the warp take different execution paths based on a condition
__global__ void warp_diverge(float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0;

    // Simulate warp divergence by using a condition that depends on the thread index
    // Threads with even indices will execute one path, while threads with odd indices will execute another
    if (idx % 2 == 0)
        x = idx * 2.0f;
    else
        x = idx * 0.5f;

    __syncthreads();
    out[idx] = x;
}

int main()
{
    const int N = 64;
    float *d_out;
    float h_out[N];

    // Allocate device memory
    cudaMalloc(&d_out, N * sizeof(float));
    warp_diverge<<<1, N>>>(d_out);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << "out[" << i << "] = " << h_out[i] << "\n";

    cudaFree(d_out);
    return 0;
}