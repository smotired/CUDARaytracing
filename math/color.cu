#include "color.cuh"

__device__ color exp(const color col) {
    return color(expf(col.x), expf(col.y), expf(col.z));
}

__global__ void ConvertColors(color const* in, Color24 *out, const size_t N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = Color24(in[i]);
}
