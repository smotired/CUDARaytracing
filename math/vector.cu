#include "vector.cuh"

__device__ void orthonormals(const float3 n, float3& x, float3& y) {
    // Copied from Cem's code
    if (n.z >= n.y) {
        const float a =  1.0f / (1.0f + n.z);
        const float b = -n.x * n.y * a;
        set(x, 1.0f - n.x * n.x * a, b, -n.x);
        set(y, b, 1 - n.y * n.y * a, -n.y);
    } else {
        const float a =  1.0f / (1.0f + n.y);
        const float b = -n.x * n.z * a;
        set(x, b, -n.z, 1 - n.y * n.y * a );
        set(y, 1.0f - n.x * n.x * a, -n.x, b);
    }
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