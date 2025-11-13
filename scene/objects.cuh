/// Header files for different types of objects
#pragma once
#include "../math/float3.cuh"
#include "rays.cuh"
#include <cuda/std/variant>

// All objects must have the following methods:
// __device__ bool IntersectRay(Ray const &ray, Hit& hit, int hitSide) const;
// __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
// __host__ __device__ Box GetBoundBox() const { return Box(-F3_ONE, F3_ONE); };
// void ViewportDisplay( Material const* material ) const;
// void Load( Loader const& loader );

// -------- Sphere

class Sphere {
public:
    __device__ bool IntersectRay(Ray const &ray, Hit& hit, int hitSide) const;
    __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
    __host__ __device__ Box GetBoundBox() const { return Box(-F3_ONE, F3_ONE); };
    void ViewportDisplay( Material const* material ) const;
    void Load( Loader const &loader ) {}
};

// -------- Plane

class Plane {
public:
    __device__ bool IntersectRay(Ray const &ray, Hit& hit, int hitSide) const;
    __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
    __host__ __device__ Box GetBoundBox() const { return Box(float3(-1,-1,0), float3(1,1,0)); };
    void ViewportDisplay( Material const* material ) const;
    void Load( Loader const &loader ) {}
};

// -------- Object union

using ObjectPtr = cuda::std::variant<Sphere*,Plane*>;

#define HAS_OBJ(objptr) cuda::std::visit([](const auto& ptr){ return ptr != nullptr; }, objptr)
#define OBJ_INTERSECT(obj, ray, hit, hitSide) cuda::std::visit([&ray, &hit, hitSide](const auto& object){ return object.IntersectRay(ray, hit, hitSide); }, obj);
#define OBJ_INTSHADOW(obj, ray, tMax, hitSide) cuda::std::visit([&ray, tMax, hitSide](const auto& object){ return object.IntersectShadowRay(ray, tMax, hitSide); }, obj);
#define OBJ_BOUNDBOX(obj) cuda::std::visit([](const auto& object){ return object.GetBoundBox(); }, obj);
#define OBJ_VIEWPORT(obj, mtlptr) cuda::std::visit([mtlptr](const auto& object){ object.ViewportDisplay(mtlptr); }, obj);
#define OBJ_LOAD(obj, loader) cuda::std::visit([&loader](auto& object){ object.Load(loader); }, obj);
