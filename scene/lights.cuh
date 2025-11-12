/// Header files for different types of lights
#pragma once
#include "../math/float3.cuh"
#include "scene.cuh"
#include <cuda/std/variant>

// All lights must have the following methods:
// Returns a color for how much the hit point is illuminated by this light source.
// __device__ color Illuminate(const Hit &hit, float3* dir) const;
// void ViewportDisplay( Material const* material ) const;

// -------- Light union

using Light = cuda::std::variant<>;