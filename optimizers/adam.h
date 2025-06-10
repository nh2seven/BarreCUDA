#ifndef ADAM_OPTIM_H
#define ADAM_OPTIM_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void adam_update(float *params, const float *grads, float *m, float *v,
                            float lr, float b1, float b2, float E,
                            float b1_t, float b2_t, int n);

#endif