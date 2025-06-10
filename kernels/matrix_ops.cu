#ifndef MATRIX_OPS
#define MATRIX_OPS

#include <matrix_ops.h>

// Matrix-vector operations -------------------------------------------------------------
// Matrix-vector multiplication: y = A * x; A: m x n, x: n-vector, y: m-vector
__global__ void mv_mul(const float *A, const float *x, float *y, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m)
    {
        float sum = 0.0f;
        for (int col = 0; col < n; col++)
            sum += A[row * n + col] * x[col];

        y[row] = sum;
    }
}

// Matrix-vector multiplication with bias: y = A * x + b; A: m x n, x: n-vector, b: m-vector, y: m-vector
__global__ void mv_mulBias(const float *A, const float *x, const float *b, float *y, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m)
    {
        float sum = 0.0f;
        for (int col = 0; col < n; col++)
            sum += A[row * n + col] * x[col];

        y[row] = sum + b[row];
    }
}

// Matrix transpose-vector multiplication: result = A^T * y; A: m x n, y: m-vector, result: n-vector
__global__ void mv_transMul(const float *A, const float *y, float *result, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        float sum = 0.0f;
        for (int row = 0; row < m; row++) {
            sum += A[row * n + col] * y[row];
        }
        result[col] = sum;
    }
}

// Matrix-matrix operations -------------------------------------------------------------
// Matrix-matrix multiplication: C = A * B; A: m x k, B: k x n, C: m x n
__global__ void mm_mul(const float *A, const float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }
}

// Matrix-matrix multiplication with shared memory optimization; C = A * B; A: m x k, B: k x n, C: m x n
__global__ void mm_mulShared(const float *A, const float *B, float *C, int m, int k, int n)
{
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of A and B and compute partial sums
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        // Load tile of A into shared memory
        if (row < m && tile * TILE_SIZE + threadIdx.x < k)
            As[threadIdx.y][threadIdx.x] = A[row * k + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < n && tile * TILE_SIZE + threadIdx.y < k)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for resultant tile
        for (int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // Write the result to C
    if (row < m && col < n)
        C[row * n + col] = sum;
}

// Outer product: C = a * b^T; a: m-vector, b: n-vector, C: m x n matrix
__global__ void mm_outer(const float *a, const float *b, float *C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        C[row * n + col] = a[row] * b[col];
}

// Matrix-only operations -------------------------------------------------------------
// Matrix addition: C = A + B; A: m x n, B: m x n, C: m x n
__global__ void m_add(const float *A, const float *B, float *C, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = m * n;

    if (idx < total_elements)
        C[idx] = A[idx] + B[idx];
}

// Matrix scalar multiplication: B = scalar * A; A: m x n, B: m x n
__global__ void mm_scalarMul(const float *A, float scalar, float *B, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = m * n;

    if (idx < total_elements)
        B[idx] = scalar * A[idx];
}

// Matrix transpose: B = A^T; A: m x n, B: n x m
__global__ void m_transpose(const float *A, float *B, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        B[col * m + row] = A[row * n + col];
}

#endif