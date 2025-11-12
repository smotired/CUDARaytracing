/// Header files for different types of objects
#pragma once
#include "../math/float3.cuh"
#include "rays.cuh"
#include <cuda/std/variant>

#define HAS_OBJ(objptr) cuda::std::visit([](const auto& ptr){ return ptr != nullptr; }, objptr)

// All objects must have the following methods:
// __device__ bool IntersectRay(Ray &ray, int hitSide) const;
// __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
// __host__ __device__ Box GetBoundBox() const { return Box(-F3_ONE, F3_ONE); };
// void ViewportDisplay( Material const* material ) const;
// void Load( Loader const& loader );

// -------- Sphere

class Sphere {
public:
    __device__ bool IntersectRay(Ray &ray, int hitSide) const;
    __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
    __host__ __device__ Box GetBoundBox() const { return Box(-F3_ONE, F3_ONE); };
    void ViewportDisplay( /*Material const* material*/ ) const;
    void Load( Loader const &loader ) {}
};

// -------- Object union

using ObjectPtr = cuda::std::variant<Sphere*>;