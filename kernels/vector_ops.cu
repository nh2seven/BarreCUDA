#ifndef VECTOR_OPS
#define VECTOR_OPS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper operations -------------------------------------------------------------
// Vector initialization with constant value
__global__ void v_fill(float *vec, float value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        vec[idx] = value;
}

// Copy vector from source to destination
__global__ void v_copy(const float *src, float *dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        dst[idx] = src[idx];
}

// Arithmetic operations -------------------------------------------------------------
// Vector addition: c = a + b
__global__ void v_add(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Vector subtraction: c = a - b
__global__ void v_sub(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        c[idx] = a[idx] - b[idx];
}

// Scalar multiplication: c = a * scalar
__global__ void v_sMul(const float *a, float scalar, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        c[idx] = a[idx] * scalar;
}

// Element-wise multiplication: c = a * b
__global__ void v_mul(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread safety check
    if (idx < n)
        c[idx] = a[idx] * b[idx];
}

// Reduction operations -------------------------------------------------------------
// Dot product using reduction
__global__ void v_dotPartial(const float *a, const float *b, float *partial_sums, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n)
        sdata[tid] = a[idx] * b[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// Sum reduction kernel
__global__ void v_reduceSum(const float *input, float *output, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

#endif