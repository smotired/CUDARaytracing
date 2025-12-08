#include "vector.cuh"

__device__ void orthonormals(const float3 n, float3& x, float3& y) {
    if (fabs(n.x) > fabs(n.z))
        set(x, -n.y, n.x, 0.0f);
    else
        set(x, 0.0f, -n.z, n.y);
    doNorm(x);
    const auto [cx, cy, cz] = cross(n, x);
    set(y, cx, cy, cz);
}

__device__ float3 reflect(const float3 incident, const float3 normal) {
    return 2 * (incident % normal) * normal - incident;
}

__device__ float3 transmit(const float3 incident, const float3 normal, const float outerIor, const float innerIor, bool& totalInnerReflection) {
    const float cos_theta = incident % normal;

    // Calculate eta, the ratio of IORs, from snell's law. Equal to sin phi / sin theta.
	const float eta = outerIor / innerIor;
    const float cos_phi_squared = 1 - eta * eta * (1 - cos_theta * cos_theta);

    // If that was negative, this is undergoing total internal reflection.
    if (cos_phi_squared < 0) {
        totalInnerReflection = true;
        return reflect(incident, normal);
    }

    totalInnerReflection = false;
    return -normal * sqrtf(cos_phi_squared) + eta * (cos_theta * normal - incident);
}

__device__ float3 glossyNormal(const float3 normal, const float glossiness, curandStateXORWOW_t *rng) {
    // Calculate random variables for the CDF and for phi (yaw angle)
    const float x = RandomFloat(rng);
    const float phi = RandomFloat(rng) * M_PI * 2.0f;

    // Calculate cos_theta from the cdf
    const float cos_theta = powf(1 - x, 1.0f / (glossiness + 1.0f));

    // Calculate the half vector from cos_theta and the normals
    float3 u, v;
    orthonormals(normal, u, v);

    const float sin_theta = sqrtf(1 - cos_theta * cos_theta);
    const float cos_phi = cosf(phi);
    const float sin_phi = sinf(phi);

    // Return half vector as normal vector
    return normal * cos_theta + u * sin_theta * cos_phi + v * sin_theta * sin_phi;
}