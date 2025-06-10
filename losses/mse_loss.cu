#ifndef MSE_LOSS
#define MSE_LOSS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// MSE loss computation; loss = (1/2n) * sum((y_pred - y_true)^2)
__global__ void mse_loss(const float *y_pred, const float *y_true, float *losses, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float diff = y_pred[idx] - y_true[idx];
        losses[idx] = 0.5f * diff * diff;
    }
}

// MSE loss gradient computation; grad = (y_pred - y_true) / n
__global__ void mse_grad(const float *y_pred, const float *y_true, float *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        grad[idx] = (y_pred[idx] - y_true[idx]) / n;
}

// Combined MSE forward and backward pass
__global__ void mse_fb(const float *y_pred, const float *y_true, float *losses, float *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float diff = y_pred[idx] - y_true[idx];
        losses[idx] = 0.5f * diff * diff;
        grad[idx] = diff / n;
    }
}

#endif
