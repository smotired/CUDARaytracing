/// Raycasting info
#pragma once
#include "settings.cuh"
#include "float3.cuh"
#include "color.cuh"

struct Node;
class Material;

// Hit sides
#define HIT_NONE 0
#define HIT_FRONT 1
#define HIT_BACK 2
#define HIT_FRONT_AND_BACK (HIT_FRONT | HIT_BACK)

struct Hit {
    // Position of the hit in world space
    float3 pos;

    // Distance of the hit in world space
    float z;

    // Normal vector at hit point
    float3 n;

    // If we hit the front of the object
    bool front;

    // Pointer to the node that we hit
    Node* node;

    // Initialize with default values
    __host__ __device__ void Init() {
        pos = F3_ZERO;
        z = BIGFLOAT;
        n = F3_UP;
        front = true;
        node = nullptr;
    }

    // Initialize with default values
    __host__ __device__ Hit() {
        pos = F3_ZERO;
        z = BIGFLOAT;
        n = F3_UP;
        front = true;
        node = nullptr;
    }
};

struct Ray {
    // Origin of the ray
    float3 pos;

    // Direction of the ray
    float3 dir;

    // Hit position of the ray
    Hit hit;

    // Pixel index (y * width + x) of the ray
    unsigned int pixel = 0;

    // Current bounce index of the ray
    unsigned int bounce = 0;

    // Contribution of the ray to the final color
    color contribution = WHITE;

    // Multiplier for contribution based on distance (for absorption)
    color absorption = BLACK;

    __host__ __device__ Ray(const float3 pos, const float3 dir, const unsigned int pixel, const unsigned int bounce = 0, const color contribution = WHITE, const color absorption = BLACK) :
        pos(pos), dir(dir), pixel(pixel), bounce(bounce), contribution(contribution), absorption(absorption) {
        hit.Init();
    }

    __device__ inline bool IsPrimary() { return bounce == 0; }

    __device__ inline bool CanBounce() { return bounce + 1 < BOUNCES; }
};

struct ShadowRay {
    // Origin of the ray
    float3 pos;

    // Direction of the ray
    float3 dir;

    // Initialize a ray
    __host__ __device__ void Init(const float3 p, const float3 d) {
        pos = p;
        dir = d;
    }
};

/// <summary>
/// Minimum and maximum corners of a node
/// </summary>
struct Box {
    // Minimum positions of the box
    float3 pmin;

    // Maximum positions of the box
    float3 pmax;

    // Initializes the box, such that there exists no point inside the box (i.e. it is empty).
    void Init() { pmin = float3(BIGFLOAT,BIGFLOAT,BIGFLOAT); pmax = float3(-BIGFLOAT,-BIGFLOAT,-BIGFLOAT); }

    // Returns true if the box is empty; otherwise, returns false.
    [[nodiscard]] bool IsEmpty() const { return pmin.x>pmax.x || pmin.y>pmax.y || pmin.z>pmax.z; }

    // Expand the box to include a point
    void operator+=(const float3 pos) {
        if (pmin.x > pos.x) pmin.x = pos.x;
        if (pmin.y > pos.y) pmin.y = pos.y;
        if (pmin.z > pos.z) pmin.z = pos.z;
        if (pmax.x < pos.x) pmax.x = pos.x;
        if (pmax.y < pos.y) pmax.y = pos.y;
        if (pmax.z < pos.z) pmax.z = pos.z;
    }

    // Use the slab method to determine if the ray intersects with the box
    __host__ __device__ bool IntersectRay(const Ray& ray, float& dist, const float t_max = BIGFLOAT) const {
	    const float3 inv = float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);

        const float3 tLow = (pmin - ray.pos) * inv;
        const float3 tHigh = (pmax - ray.pos) * inv;

        const float3 tClose(std::fmin(tLow.x, tHigh.x),
                      std::fmin(tLow.y, tHigh.y),
                      std::fmin(tLow.z, tHigh.z));
        const float3 tFar(std::fmax(tLow.x, tHigh.x),
                    std::fmax(tLow.y, tHigh.y),
                    std::fmax(tLow.z, tHigh.z));

        const float tEnter = std::fmax(tClose.x, std::fmax(tClose.y, tClose.z));
        const float tExit = std::fmin(tFar.x, std::fmin(tFar.y, tFar.z));

        // If it actually enters, the box, return the intersection distance
        if ((tEnter >= 0 || tExit >= 0) && tEnter <= tExit && tEnter <= t_max) {
            dist = tEnter >= 0 ? tEnter : tExit;
            return true;
        }

        return false;
    }
};