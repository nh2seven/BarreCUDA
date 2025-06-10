#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mse_loss(const float *y_pred, const float *y_true, float *losses, int n);
__global__ void mse_grad(const float *y_pred, const float *y_true, float *grad, int n);
__global__ void mse_fb(const float *y_pred, const float *y_true, float *losses, float *grad, int n);

#endif
