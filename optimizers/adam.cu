#ifndef ADAM_OPTIM
#define ADAM_OPTIM

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/*
m = beta1 * m + (1 - beta1) * gradients
v = beta2 * v + (1 - beta2) * gradients^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
params = params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
*/

// Adam optimizer kernel to update parameters based on gradients
__global__ void adam_update(float *params, const float *grads, float *m, float *v,
                            float lr, float b1, float b2, float E,
                            float b1_t, float b2_t, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float grad = grads[idx];

        // Update biased first moment estimate
        m[idx] = b1 * m[idx] + (1.0f - b1) * grad;

        // Update biased second raw moment estimate
        v[idx] = b2 * v[idx] + (1.0f - b2) * grad * grad;

        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - b1_t);

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - b2_t);

        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + E);
    }
}

#endif
