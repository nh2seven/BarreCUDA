#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel to add two integers
__global__ void add(int* a, int* b, int* c) // __global__ indicates that this is a CUDA kernel
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

// __managed__ indicates that the memory can be managed by CUDA or the host, without needing to copy it explicitly
__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main()
{
    printf("CUDA version: %d\n", CUDART_VERSION);
    
    for (int i = 0; i < 256; i++)
    {
        vector_a[i] = i;
        vector_b[i] = 256 - i;
    }
    
    printf("Adding vectors...\n");

    // <<<1 block, 256 threads>>> to be launched on the GPU
    add<<<1, 256>>>(vector_a, vector_b, vector_c);

    cudaDeviceSynchronize(); // Wait for the GPU to finish executing

    int result = 0;

    // Check the result on the host
    for (int i = 0; i < 256; i++)
    {
        result += vector_c[i];
    }

    printf("Result: %d\n", result);

    return 0;
}