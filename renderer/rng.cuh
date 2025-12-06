#ifndef CUDA_RAYTRACING_RNG_CUH
#define CUDA_RAYTRACING_RNG_CUH
#include <functional>
#include <thread>
#include <curand_kernel.h>

// Return a random float in the range [0, 1) using the cuRAND library.
__device__ inline float RandomFloat(curandStateXORWOW_t *rng) {
    constexpr float rmax = 0x1.fffffep-1;
    const float r = static_cast<float>(curand(rng)) * 0x1p-32f;
    return r < rmax ? r : rmax;
}

#endif //CUDA_RAYTRACING_RNG_CUH