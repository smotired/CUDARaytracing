/// Color methods
#pragma once
#include <cstdint>

#include "float3.cuh"

// Typedef of a Color as a float3

/// <summary>
/// A float3 that represents a color
/// </summary>
typedef float3 color;

// Shorthand for make_float3
#define color(x, y, z) make_float3(x, y, z)
#define copycolor(from) make_float3(from.x, from.y, from.z)

// Default colors

#define BLACK color(0.0f, 0.0f, 0.0f)
#define WHITE color(1.0f, 1.0f, 1.0f)

// Methods on colors

__device__ color exp(color col); // e^col

// Integer version from 0-255 that freeglut uses

struct Color24 {
    uint8_t r; uint8_t g; uint8_t b;
    __host__ __device__ explicit Color24(const color& col) {
        r = ClampToInt(col.x * 256);
        g = ClampToInt(col.y * 256);
        b = ClampToInt(col.z * 256);
    }
    Color24() { r = 0; g = 0; b = 0; }
private:
    __host__ __device__ static uint8_t ClampToInt(float f) { int v = (int)f; return v<0 ? 0 : (v>255 ? 255 : static_cast<uint8_t>(v)); }
};

// Kernel to convert colors to the final Color24. Will do gamma correction here too
__global__ void ConvertColors(color const* in, Color24 *out, size_t N);