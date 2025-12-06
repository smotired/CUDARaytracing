#include "color.cuh"

__device__ color exp(const color col) {
    return color(expf(col.x), expf(col.y), expf(col.z));
}

__device__ inline float LinearTosRGB( const float ch ) {
    // stolen from cem
    return ch < 0.0031308f ? ch * 12.92f : powf(ch, 0.41666f) * 1.055f - 0.055f;
}

__device__ inline color LinearTosRGB( const color c ) {
    // stolen from cem
    return color(LinearTosRGB(c.x),
                 LinearTosRGB(c.y),
                 LinearTosRGB(c.z));
}

__global__ void ConvertColors(color const* in, Color24 *out, const size_t N, const bool sRGB, const float passMultiplier) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = Color24(sRGB ? LinearTosRGB(in[i] * passMultiplier) : in[i] * passMultiplier);
}
