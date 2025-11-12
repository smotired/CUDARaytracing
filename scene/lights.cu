/// Implementations for light classes.
#include "lights.cuh"

#include "renderer.cuh"

__device__ color AmbientLight::Illuminate(const Hit &hit, float3 &dir) const {
    return intensity;
}

__device__ color DirectionalLight::Illuminate(const Hit &hit, float3 &dir) const {
    dir = -direction;
    return intensity;
}

__device__ color PointLight::Illuminate(const Hit &hit, float3 &dir) const {
    dir = asNorm(position - hit.pos);
    return intensity;
}