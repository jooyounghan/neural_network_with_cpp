#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>

__global__ void matMul(float* mat_result, float* mat_in1, float* mat_in2, const int& k_size, const int& col_size){
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * blockDim.x + col;
    float s_result = 0;

    for (int k = 0; k < k_size; k += 1)
    {
        s_result += mat_in1[k_size * row + k] * mat_in2[col_size * k + col];
    }
    mat_result[idx] = s_result;
}

__global__ void initialize(float* mat_result) {
    printf("Hello from threadIdx.x %d, threadIdx.y %d\n", threadIdx.x, threadIdx.y);
    int row = threadIdx.y;
    int col = threadIdx.x;
    printf("Hello from threadIdx.x %d, threadIdx.y %d\n", threadIdx.x, threadIdx.y);
    int idx = row * blockDim.x + col;
    curandState s;
    float rand_float = curand_uniform(&s);
    mat_result[idx] = 1;
}