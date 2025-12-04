/// Implementations for light classes.
#include "lights.cuh"
#include "trace.cuh"

__device__ color AmbientLight::Illuminate(const Hit &hit, float3 &dir) const {
    return intensity;
}

__device__ color DirectionalLight::Illuminate(const Hit &hit, float3 &dir) const {
    dir = -direction;

    // Trace a shadow ray
    ShadowRay ray(hit.pos, dir);
    const bool obstructed = TraceShadowRay(ray, hit.n);

    return obstructed ? BLACK : intensity;
}

__device__ color PointLight::Illuminate(const Hit &hit, float3 &dir) const {
    dir = asNorm(position - hit.pos);

    // Trace a shadow ray
    ShadowRay ray(hit.pos, dir);
    const bool obstructed = TraceShadowRay(ray, hit.n, length(position - hit.pos));

    // Add attenuation and radiance
    const color radiance = intensity * (1.0f / (4 * M_PI * size * size));
    const float attenuation = 1.0f / lengthsq(position - hit.pos);

    return obstructed ? BLACK : radiance * attenuation;
}