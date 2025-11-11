/// Vector operations callable on device.
#pragma once
#include <cuda_runtime.h>

// Default vectors
__managed__ const float3 F3_ZERO = make_float3(1.0f, 0.0f, 0.0f);
__managed__ const float3 F3_RIGHT = make_float3(1.0f, 0.0f, 0.0f);
__managed__ const float3 F3_UP = make_float3(0.0f, 1.0f, 0.0f);
__managed__ const float3 F3_FORWARD = make_float3(0.0f, 0.0f, 1.0f);

// Typedef of a Color as a float3

/// <summary>
/// A float3 that represents a color
/// </summary>
typedef float3 color;

// Default colors

__managed__ const color BLACK = make_float3(0.0f, 0.0f, 0.0f);
__managed__ const color WHITE = make_float3(1.0f, 1.0f, 1.0f);

// Shorthand for make_float3
__host__ __device__ inline float3 Float3(const float x, const float y, const float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 Float3(const float3 from) { return make_float3(from.x, from.y, from.z); }
__host__ __device__ inline float3 Color(const float x, const float y, const float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 Color(const color from) { return make_float3(from.x, from.y, from.z); }

// Operator overloads

/// <summary>
/// Multiply a vector by a scalar
/// </summary>
/// <param name="a">The vector</param>
/// <param name="s">The scalar</param>
/// <returns>The scaled vector</returns>
__host__ __device__ inline float3 operator*(const float3 a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

/// <summary>
/// Multiply a vector by a scalar
/// </summary>
/// <param name="s">The scalar</param>
/// <param name="a">The vector</param>
/// <returns>The scaled vector</returns>
__host__ __device__ inline float3 operator*(const float s, const float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

/// <summary>
/// Multiply an existing vector by a scalar
/// </summary>
/// <param name="a">The vector</param>
/// <param name="s">The scalar</param>
/// <returns>The scaled vector</returns>
__host__ __device__ inline void operator*=(float3& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
}

/// <summary>
/// Add two vectors together.
/// </summary>
/// <param name="a">The first vector.</param>
/// <param name="b">The second vector.</param>
/// <returns>The sum of the two vectors.</returns>
__host__ __device__ inline float3 operator+(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/// <summary>
/// Add a vector to another vector.
/// </summary>
/// <param name="a">The vector to add to.</param>
/// <param name="b">The second vector.</param>
__host__ __device__ inline void operator+=(float3& a, const float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

/// <summary>
/// Subtract two vectors.
/// </summary>
/// <param name="a">The sum vector.</param>
/// <param name="b">The vector to subtract.</param>
/// <returns>The difference of the two vectors.</returns>
__host__ __device__ inline float3 operator-(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/// <summary>
/// Subtract a vector from another vector.
/// </summary>
/// <param name="a">The vector to add to.</param>
/// <param name="b">The second vector.</param>
__host__ __device__ inline void operator-=(float3& a, const float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

/// <summary>
/// Multiply two vectors component-wise
/// </summary>
/// <param name="a">The first vector.</param>
/// <param name="b">The second vector.</param>
/// <returns>The product vector of the components of the two vectors.</returns>
__host__ __device__ inline float3 operator*(const float3 a, const float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

/// <summary>
/// Computes the dot product of two vectors.
/// </summary>
/// <param name="a">The first vector.</param>
/// <param name="b">The second vector.</param>
/// <returns>The dot product of the two vectors.</returns>
__host__ __device__ inline float operator%(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// <summary>
/// Computes the cross product of two vectors.
/// </summary>
/// <param name="a">The first vector.</param>
/// <param name="b">The second vector.</param>
/// <returns>The cross product of the two vectors.</returns>
__host__ __device__ inline float3 cross(const float3 a, const float3 b) {
    return RIGHT * (a.y * b.z - b.y * a.z) + UP * (a.x * b.z - b.x * a.z) + FORWARD * (a.x * b.y - b.x * a.y);
}

/// <summary>
/// Calculate the squared length of a vector
/// </summary>
/// <param name="a">The vector.</param>
/// <returns>The squared length of the vector.</returns>
__host__ __device__ inline float lengthsq(const float3 a) {
    return a % a;
}

/// <summary>
/// Calculate the length of a vector
/// </summary>
/// <param name="a">The vector.</param>
/// <returns>The length of the vector.</returns>
__host__ __device__ inline float length(const float3 a) {
    return sqrtf(a % a);
}

/// <summary>
/// Calculate the normalzed form of a vector
/// </summary>
/// <param name="a">The vector.</param>
/// <returns>The length of the vector.</returns>
__host__ __device__ inline float3 norm(const float3 a) {
    float scale = 1.0f / sqrtf(a % a);
    return scale * a;
}

/// <summary>Set values on a float3.</summary>
__host__ __device__ inline float3 set(float3& a, const float x, const float y, const float z) {
    a.x = x; a.y = y; a.z = z;
}

/// <summary>
/// Get two unit vectors orthogonal to another normal vector, forming an orthonormal basis.
/// </summary>
__host__ __device__ inline float3 orthonormals(const float3 n, float3& x, float3& y) {
    // Copied from Cem's code
    if (n.z >= n.y) {
        const float a =  1.0f / (1.0f + n.z);
        const float b = -n.x * n.y * a;
        set(x, 1.0f - n.x * n.x * a, b, -n.x);
        set(y, b, 1 - n.y * n.y * a, -n.y);
    } else {
        const float a =  1.0f / (1.0f + n.y);
        const float b = -n.x * n.z * a;
        set(x, b, -n.z, 1 - n.y * n.y * a );
        set(y, 1.0f - n.x * n.x * a, -n.x, b);
    }
}