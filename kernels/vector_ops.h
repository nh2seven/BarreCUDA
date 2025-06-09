#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel declarations
__global__ void v_fill(float *vec, float value, int n);
__global__ void v_copy(const float *src, float *dst, int n);
__global__ void v_add(const float *a, const float *b, float *c, int n);
__global__ void v_sAdd(const float *a, float scalar, float *c, int n);
__global__ void v_sub(const float *a, const float *b, float *c, int n);
__global__ void v_sMul(const float *a, float scalar, float *c, int n);
__global__ void v_mul(const float *a, const float *b, float *c, int n);
__global__ void v_dotPartial(const float *a, const float *b, float *partial_sums, int n);
__global__ void v_reduceSum(const float *input, float *output, int n);

#endif