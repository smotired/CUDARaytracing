/// Header files for different types of lights
#pragma once
#include "../math/float3.cuh"
#include "xmlload.cuh"
#include <cuda/std/variant>
#include "sampler.cuh"

// All lights must have the following methods:
// __device__ bool IsAmbient() const;
// Returns a color for how much the hit point is illuminated by this light source.
// __device__ color Illuminate(const Hit &hit, float3& dir) const;
// void SetViewportLight( int lightID ) const;
// void Load( Loader const &loader );

// -------- GLLight for viewport

class GLLight {
public:
    void SetViewportParam( int lightID, color const &ambient, color const &intensity, float4 const &pos ) const;
};


// -------- Ambient

class AmbientLight : public GLLight {
    color intensity = BLACK;
public:
    __device__ bool IsAmbient() const { return true; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    __device__ color Radiance() const { return intensity; };
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, intensity, BLACK, make_float4(0,0,0,1)); }
    void Load( Loader const &loader );

    __device__ bool Intersect(Ray const& ray, Hit &hit) const { return false; }
    __device__ bool GenerateSample(float3 const& v, Hit const& hit, float3& dir, SampleInfo& info) const { return false; };
};

// -------- Directional

class DirectionalLight : public GLLight {
    color intensity = BLACK;
    float3 direction = F3_UP;
public:
    __device__ bool IsAmbient() const { return false; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    __device__ color Radiance() const { return intensity; };
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, BLACK, intensity, make_float4(-direction.x, -direction.y, -direction.z, 0.0f)); };
    void Load( Loader const &loader );

    __device__ bool Intersect(Ray const& ray, Hit &hit) const { return false; }
    __device__ bool GenerateSample(float3 const& v, Hit const& hit, float3& dir, SampleInfo& info) const { return false; };
};

// -------- Point

class PointLight : public GLLight {
    color intensity = BLACK;
    float3 position = F3_ZERO;
    float size = 1;
public:
    __device__ bool IsAmbient() const { return false; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    __device__ color Radiance() const { return intensity * (1.0f / (M_PI * size * size)); };
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, BLACK, intensity, make_float4(position.x, position.y, position.z,1.0f)); };
    void Load( Loader const &loader );

    __device__ bool Intersect(Ray const& ray, Hit &hit) const;
    __device__ bool GenerateSample(float3 const& v, Hit const& hit, float3& dir, SampleInfo& info) const;
};

// -------- Light union

using Light = cuda::std::variant<AmbientLight, DirectionalLight, PointLight>;
// TODO: Do this for shapes too
// Visit variants of light to call a property
#define LIGHT_CALL(lightptr, body, ...) cuda::std::visit([__VA_ARGS__](const auto& light)body, *lightptr)
#define LIGHT_CALL_NCONST(lightptr, body, ...) cuda::std::visit([__VA_ARGS__](auto& light)body, *lightptr)
#define LIGHT_ISAMBIENT(lightptr) LIGHT_CALL(lightptr, { return light.IsAmbient(); })
#define LIGHT_RADIANCE(lightptr) LIGHT_CALL(lightptr, { return light.Radiance(); })
#define LIGHT_ILLUMINATE(lightptr, hit, l) LIGHT_CALL(lightptr, { return light.Illuminate(hit, l); }, &hit, &l)
#define LIGHT_SETVIEWPORTLIGHT(lightptr, lightID) LIGHT_CALL(lightptr, { light.SetViewportLight(lightID); }, lightID)
#define LIGHT_LOAD(lightptr, loader) LIGHT_CALL_NCONST(lightptr, { light.Load(loader); }, &loader)
#define LIGHT_GENSAMPLE(lightptr, v, hit, dir, info) LIGHT_CALL(lightptr, { return light.GenerateSample(v, hit, dir, info); }, &v, &hit, &dir, &info)
#define LIGHT_INTERSECT(lightptr, ray, hit) LIGHT_CALL(lightptr, { return light.Intersect(ray, hit); }, &ray, &hit)