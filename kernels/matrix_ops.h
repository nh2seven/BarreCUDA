#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mv_mul(const float *A, const float *x, float *y, int m, int n);
__global__ void mv_mulBias(const float *A, const float *x, const float *b, float *y, int m, int n);
__global__ void mm_mul(const float *A, const float *B, float *C, int m, int k, int n);
__global__ void mm_mulShared(const float *A, const float *B, float *C, int m, int k, int n);
__global__ void mm_outer(const float *a, const float *b, float *C, int m, int n);
__global__ void m_add(const float *A, const float *B, float *C, int m, int n);
__global__ void mm_scalarMul(const float *A, float scalar, float *B, int m, int n);
__global__ void m_transpose(const float *A, float *B, int m, int n);

#endif