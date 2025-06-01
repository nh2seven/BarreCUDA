#include <iostream>
#include <cuda_runtime.h>

#define N 256
#define BLOCK_SIZE 256

// Global memory for threads in a grid
__global__ void add_global(float *a, float *b, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If index is within array bounds
    if (idx < N)
    {
        out[idx] = a[idx] + b[idx];
        printf("Global - Thread ID: %d, Block ID: %d, Operation: %.1f + %.1f = %.1f\n", 
               idx, blockIdx.x, a[idx], b[idx], out[idx]);
    }
}

// Shared memory between threads in a block
__global__ void add_shared(float *a, float *b, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_a[BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE];

    // If index is within array bounds
    if (idx < N)
    {
        // Load from global to shared
        s_a[threadIdx.x] = a[idx];
        s_b[threadIdx.x] = b[idx];

        // Synchronize all threads within the block
        __syncthreads(); // Ensures that all threads reach the same execution point before continuing

        // Compute using shared memory
        out[idx] = s_a[threadIdx.x] + s_b[threadIdx.x];
        printf("Shared - Thread ID: %d, Block ID: %d, Local Thread ID: %d, Operation: %.1f + %.1f = %.1f\n", 
               idx, blockIdx.x, threadIdx.x, s_a[threadIdx.x], s_b[threadIdx.x], out[idx]);
    }
}

// Function to check the addition result
void check_result(float *a, float *b, float *out)
{
    for (int i = 0; i < N; ++i)
    {
        float expected = a[i] + b[i];

        // Check if the result matches the expected value
        if (fabs(out[i] - expected) > 1e-5)
        {
            std::cerr << "Mismatch at " << i << ": got " << out[i] << ", expected " << expected << std::endl;
            return;
        }
    }
    std::cout << "Result verified. The computation works as expected." << std::endl;
}

// Main function to run the kernels
int main()
{
    // Host pointers for arrays in global memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];

    // Device pointers for arrays in global memory
    float *d_a;
    float *d_b;
    float *d_out;

    // Init host arrays with some values for demonstration
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions in 3D space
    dim3 block(BLOCK_SIZE);                       // 256 threads per block
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // 1 block for 256 elements, 4 blocks for 1024 elements, etc.

    // Launch global memory kernel with specified grid and block dimensions
    std::cout << "\nRunning global memory kernel:\n";
    add_global<<<grid, block>>>(d_a, d_b, d_out);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure all threads have completed before checking results
    check_result(h_a, h_b, h_out);

    // Launch shared memory kernel with specified grid and block dimensions
    std::cout << "\nRunning shared memory kernel:\n";
    add_shared<<<grid, block>>>(d_a, d_b, d_out);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure all threads have completed before checking results
    check_result(h_a, h_b, h_out);

    // Print some final results
    std::cout << "\nSome final results:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    // Free device memory and delete host arrays (cleanup)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;

    return 0;
}
