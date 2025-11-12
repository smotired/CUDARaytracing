/// Header files for different types of lights
#pragma once
#include "../math/float3.cuh"
#include "scene.cuh"
#include <cuda/std/variant>

// All lights must have the following methods:
// __device__ bool IsAmbient()
// Returns a color for how much the hit point is illuminated by this light source.
// __device__ color Illuminate(const Hit &hit, float3& dir) const;
// void SetViewportLight( int lightID ) const;
// void Load( Loader const &loader );

// -------- GLLight for viewport

class GLLight {
protected:
    void SetViewportParam( int lightID, color const &ambient, color const &intensity, float4 const &pos ) const;
};


// -------- Ambient

class AmbientLight : public GLLight {
    color intensity = BLACK;
public:
    __device__ bool IsAmbient() { return true; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, intensity, BLACK, make_float4(0,0,0,1)); }
    void Load( Loader const &loader );
};

// -------- Directional

class DirectionalLight : public GLLight {
    color intensity = BLACK;
    float3 direction = F3_UP;
public:
    __device__ bool IsAmbient() { return true; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, BLACK, intensity, make_float4(-direction.x, -direction.y, -direction.z, 0.0f)); };
    void Load( Loader const &loader );
};

// -------- Point

class PointLight : public GLLight {
    color intensity = BLACK;
    float3 position = F3_ZERO;
public:
    __device__ bool IsAmbient() { return true; }
    __device__ color Illuminate(const Hit &hit, float3& dir) const;
    void SetViewportLight( int lightID ) const { SetViewportParam(lightID, BLACK, intensity, make_float4(position.x, position.y, position.z,1.0f)); };
    void Load( Loader const &loader );
};

// -------- Light union

using Light = cuda::std::variant<AmbientLight, DirectionalLight, PointLight>;
// TODO: Do this for shapes too
// Visit variants of light to call a property
#define LIGHT_CALL(lightptr, body, ...) cuda::std::visit([__VA_ARGS__](const auto& light)body, *lightptr)
#define LIGHT_ISAMBIENT(lightptr) LIGHT_CALL(lightptr, { return light.IsAmbient(); })
#define LIGHT_ILLUMINATE(lightptr, hit, l) LIGHT_CALL(lightptr, { return light.Illuminate(hit, l); }, hit, l)