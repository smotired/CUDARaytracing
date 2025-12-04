#ifndef CUDA_RAYTRACING_TEXTURE_CUH
#define CUDA_RAYTRACING_TEXTURE_CUH
#include "color.cuh"

class Texture {
private:
    Color24* data;
public:
    unsigned int width;
    unsigned int height;

    Texture(color const& solidColor) : width(1), height(1) {
        const Color24 transformed(solidColor);
        cudaMalloc(&data, sizeof(Color24));
        cudaMemcpy(data, &transformed, sizeof(Color24), cudaMemcpyHostToDevice);
    }

    Texture(color const* source, unsigned int width, unsigned int height);

    void Free() const { cudaFree(data); }

    __device__ color Eval(float3 const& uvw) const;

    __device__ color EvalEnvironment( float3 const &dir ) const;
};

#endif //CUDA_RAYTRACING_TEXTURE_CUH