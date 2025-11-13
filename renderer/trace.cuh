/// Kernel functions for tracing rays
#pragma once
#include "rays.cuh"

// Fire primary rays from the camera origin to the center of the plane
__global__ void DispatchPrimaryRays();

// Trace a ray through the scene and calculate the returned color
__device__ void TraceRay(Ray& ray, int hitSide = HIT_FRONT_AND_BACK);

// Trace a ray through the scene and calculate a shadow hit. Return true if it hits an object.
__device__ bool TraceShadowRay(ShadowRay& ray, float3 n, float tMax = BIGFLOAT, int hitSide = HIT_FRONT_AND_BACK);