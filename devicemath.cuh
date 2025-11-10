//
// Created by sam on 11/10/25.
//

#ifndef CUDA_RAYTRACING_DEVICEMATH_CUH
#define CUDA_RAYTRACING_DEVICEMATH_CUH

__global__ void Divide(float const* a, float const* b, float* c, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] / b[i];
}

#endif //CUDA_RAYTRACING_DEVICEMATH_CUH