/// Header files for different types of objects
#pragma once
#include "../math/float3.cuh"
#include "rays.cuh"
#include <cuda/std/variant>
#include "mesh.cuh"

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

// -------- MeshObject

class MeshObject : public Mesh {
protected:
    __device__ bool IntersectTriangle(Ray const& ray, Hit& hit, int hitSide, unsigned int faceID) const;
    __device__ bool IntersectShadowTriangle(ShadowRay const& ray, float tMax, int hitSide, unsigned int faceID) const;
public:
    __device__ bool IntersectRay(Ray const &ray, Hit& hit, int hitSide) const;
    __device__ bool IntersectShadowRay(const ShadowRay &ray, float tMax, int hitSide) const;
    __host__ __device__ Box GetBoundBox() const { return Box(float3(-1,-1,0), float3(1,1,0)); };
    void ViewportDisplay( Material const* material ) const;
    void Load( Loader const &loader ) {}
};

// -------- Object union

using ObjectPtr = cuda::std::variant<Sphere*,Plane*,MeshObject*>;

#define HAS_OBJ(objptr) cuda::std::visit([](const auto& ptr){ return ptr != nullptr; }, objptr)
#define OBJ_INTERSECT(objptr, ray, hit, hitSide) cuda::std::visit([&ray, &hit, hitSide](const auto& object){ return object->IntersectRay(ray, hit, hitSide); }, objptr)
#define OBJ_INTSHADOW(objptr, ray, tMax, hitSide) cuda::std::visit([&ray, tMax, hitSide](const auto& object){ return object->IntersectShadowRay(ray, tMax, hitSide); }, objptr)
#define OBJ_BOUNDBOX(objptr) cuda::std::visit([](const auto& object){ return object->GetBoundBox(); }, objptr)
#define OBJ_VIEWPORT(objptr, mtlptr) cuda::std::visit([mtlptr](const auto& object){ object->ViewportDisplay(mtlptr); }, objptr)
#define OBJ_LOAD(objptr, loader) cuda::std::visit([&loader](auto& object){ object->Load(loader); }, objptr)
