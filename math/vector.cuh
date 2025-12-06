/// Methods specifically for vector float3s
#pragma once
#include "float3.cuh"
#include "rng.cuh"

/// <summary>
/// Get two unit vectors orthogonal to another normal vector, forming an orthonormal basis.
/// </summary>
__device__ void orthonormals(float3 n, float3& x, float3& y);

/// <summary>
/// Get the incident vector reflected by a normal
/// </summary>
__device__ float3 reflect(float3 incident, float3 normal);

/// <summary>
/// Get the incident vector transmitted through a surface
/// </summary>
__device__ float3 transmit(float3 incident, float3 normal, float outerIor, float innerIor, bool& totalInnerReflection);

/// <summary>
/// Compute a random normal vector
/// </summary>
__device__ float3 glossyNormal(float3 normal, float glossiness, RNG& rng);