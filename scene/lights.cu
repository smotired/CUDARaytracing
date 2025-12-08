/// Implementations for light classes.
#include "lights.cuh"
#include "trace.cuh"
#include "vector.cuh"
#include "rays.cuh"

constexpr float OVERPI = 1.0f / M_PI;
constexpr float F_PI = static_cast<float>(M_PI);

__device__ bool PointLight::GenerateSample(float3 const& v, Hit const &hit, float3 &dir, curandStateXORWOW_t *rng, SampleInfo &info) const {
    // Pick a random point on the visible half of the sphere
    const float3 L = asNorm(hit.pos - position);
    const float x = RandomFloat(rng);
    const float phi = RandomFloat(rng) * 2 * F_PI;
    const float cos_theta = 1 - x;
    const float sin_theta = sqrtf(1 - cos_theta * cos_theta);

    float3 a, b;
    orthonormals(L, a, b);
    const float3 n = L * cos_theta + a * sin_theta * cosf(phi) + b * sin_theta * sinf(phi);

    // Set up direction ot that point
    const float3 target = position + n * size;
    dir = asNorm(target - hit.pos);
    const float d2 = lengthsq(target - hit.pos);
    const float cos_theta_l = dir % -asNorm(target - position);

    // Set up the info
    info.prob = d2 / (2 * F_PI * size * size * cos_theta_l);
    info.mult = Radiance();
    info.dist = sqrtf(d2);

    return true;
}

__device__ bool PointLight::Intersect(Ray const &ray, Hit &hit) const {
    // Mostly copied from sphere: intersectRay
    const float3 o = ray.pos - position; // local ray origin

    // Ray is not transformed into light's space so we must include the radiance and position offset
    const float a = ray.dir % ray.dir;
    const float b = (2 * ray.dir) % o;
    const float c = o % o - size * size;

    // There is a real solution to the quadratic formula if this is positive.
    // If this is negative, the solution(s) are negative and there is no intersection.
    const float determinant = b * b - 4 * a * c;
    if (determinant < 0) return false;

    // Calculate the solutions to the quadratic formula.
    const  float sqrtDeterminant = sqrt(determinant);
    const float front = (-b - sqrtDeterminant) / (2 * a);
    const float back = (-b + sqrtDeterminant) / (2 * a);

    // Check front hit
    if (front >= 0) {
        if (front >= hit.z) return false;

        hit.z = front;
        const float3 normPos = asNorm(o);

        hit.pos = ray.pos + front * ray.dir;
        hit.n = normPos;
        hit.uvw = float3( // Spherical mapping
            0.5f * OVERPI * atan2(normPos.y, normPos.x) + 0.5f,
            OVERPI * asin(fmin(fmax(normPos.z, -1.0f), 1.0f)) + 0.5f,
            0);
        hit.front = true;
        hit.hitLight = true;

        return true;
    }

    // Check back hit
    if (back < 0) return false;
    if (back >= hit.z) return false;
    hit.z = back;
    const float3 normPos = asNorm(hit.pos);
    hit.pos = ray.pos + back * ray.dir;
    hit.uvw = float3( // Spherical mapping
        0.5f * OVERPI * atan2(normPos.y, normPos.x) + 0.5f,
        OVERPI * asin(fmin(fmax(normPos.z, -1.0f), 1.0f)) + 0.5f,
        0);
    hit.n = normPos;
    hit.front = false;
    hit.hitLight = true;

    return true;
}