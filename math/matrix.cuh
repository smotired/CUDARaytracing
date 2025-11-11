/// A 3x4 matrix for use in transformations
#pragma once
#include "float3.cuh"

#define DEG2RAD 0.017453292519943296f

/// <summary>
/// A 3x4 matrix for use in transformations. Implicit fourth row is [0, 0, 0, 1].
/// </summary>
class Matrix {
public:
    // Elements in column-major order
    float cells[12]{};

    // Construct the identity matrix
    Matrix() {
        cells[0] = 1.0f;
        cells[4] = 1.0f;
        cells[8] = 1.0f;
    }

    /// <summary>
    /// Get the inverse of this matrix, representing the opposite transformation direction.
    /// </summary>
    [[nodiscard]] Matrix GetInverse() const {
        // Copied shamelessly from Cem's code
        //  (4 8 - 5 7)    (5 6 - 3 8)    (3 7 - 4 6)    (3 (8 10 - 7 11) + 4 (6 11 - 8  9) + 5 (7  9 - 6 10))
        //  (2 7 - 1 8)    (0 8 - 2 6)    (1 6 - 0 7)    (0 (7 11 - 8 10) + 1 (8  9 - 6 11) + 2 (6 10 -  7 9))    / det
        //  (1 5 - 2 4)    (2 3 - 0 5)    (0 4 - 1 3)    (0 (5 10 - 4 11) + 1 (3 11 - 5  9) + 2 (4  9 - 3 10))
        Matrix inverse{};

        const float data_8_10_7_11 = cells[8] * cells[10] - cells[7] * cells[11];
        const float data_6_11_8_9 = cells[6] * cells[11] - cells[8] * cells[ 9];
        const float data_7_9_6_10 = cells[7] * cells[ 9] - cells[6] * cells[10];

        inverse.cells[ 0] = (cells[4]*cells[8] - cells[5]*cells[7]);
        inverse.cells[ 1] = (cells[2]*cells[7] - cells[1]*cells[8]);
        inverse.cells[ 2] = (cells[1]*cells[5] - cells[2]*cells[4]);

        inverse.cells[ 3] = (cells[5]*cells[6] - cells[3]*cells[8]);
        inverse.cells[ 4] = (cells[0]*cells[8] - cells[2]*cells[6]);
        inverse.cells[ 5] = (cells[2]*cells[3] - cells[0]*cells[5]);

        inverse.cells[ 6] = (cells[3]*cells[7] - cells[4]*cells[6]);
        inverse.cells[ 7] = (cells[1]*cells[6] - cells[0]*cells[7]);
        inverse.cells[ 8] = (cells[0]*cells[4] - cells[1]*cells[3]);

        inverse.cells[ 9] = cells[3] * data_8_10_7_11 + cells[4] * data_6_11_8_9 + cells[5] * data_7_9_6_10;
        inverse.cells[10] = cells[0] *-data_8_10_7_11 + cells[1] *-data_6_11_8_9 + cells[2] *-data_7_9_6_10;
        inverse.cells[11] = cells[0] * (cells[5] * cells[10] - cells[4] * cells[11]) +
                           cells[1] * (cells[3] * cells[11] - cells[5] * cells[ 9]) +
                           cells[2] * (cells[4] * cells[ 9] - cells[3] * cells[10]);

        const float det = cells[0] * inverse.cells[0] + cells[1] * inverse.cells[3] + cells[2] * inverse.cells[6];
        return inverse * (1.0f / det);
    }

    /// <summary>
    /// Multiply a matrix by a scalar
    /// </summary>
    inline Matrix operator*(const float s) const {
        Matrix product{};
        for (int i = 0; i < 12; i++)
            product.cells[i] = cells[i] * s;
        return product;
    }

    /// <summary>
    /// Multiply the matrix by the vector. Reperesents a position transformation.
    /// </summary>
    /// <param name="v">The vector to transform.</param>
    /// <returns>The transformed vector.</returns>
    __host__ __device__ inline float3 operator*(const float3 v) const {
        return Float3(cells[0] * v.x + cells[3] * v.y + cells[6] * v.z + cells[9],
                      cells[1] * v.x + cells[4] * v.y + cells[7] * v.z + cells[10],
                      cells[2] * v.x + cells[5] * v.y + cells[8] * v.z + cells[11]);
    }

    /// <summary>
    /// Multiply the matrix by the vector, excluding the fourth column. Reperesents a direction transformation.
    /// </summary>
    /// <param name="v">The vector to transform.</param>
    /// <returns>The transformed vector.</returns>
    __host__ __device__ inline float3 operator%(const float3 v) const {
        return Float3(cells[0] * v.x + cells[3] * v.y + cells[6] * v.z,
                      cells[1] * v.x + cells[4] * v.y + cells[7] * v.z,
                      cells[2] * v.x + cells[5] * v.y + cells[8] * v.z);
    }

    /// <summary>
    /// Convert to a 4x4 matrix for use in the OpenGL preview
    /// </summary>
    inline void As4x4(float out[16]) const {
        std::copy(cells, cells + 12, out);
        out[12] = out[13] = out[14] = 0.0f;
        out[15] = 1.0f;
    }

    /// <summary>
    /// Multiply by another matrix. The other matrix goes FIRST in this calculation
    /// </summary>
    Matrix operator*(const Matrix& first) const {
        Matrix product{};
        product.cells[0] = first.cells[0] * cells[0] + first.cells[3] * cells[1] + first.cells[6] * cells[2];
        product.cells[1] = first.cells[1] * cells[0] + first.cells[4] * cells[1] + first.cells[7] * cells[2];
        product.cells[2] = first.cells[2] * cells[0] + first.cells[5] * cells[1] + first.cells[8] * cells[2];
        product.cells[3] = first.cells[0] * cells[3] + first.cells[3] * cells[4] + first.cells[6] * cells[5];
        product.cells[4] = first.cells[1] * cells[3] + first.cells[4] * cells[4] + first.cells[7] * cells[5];
        product.cells[5] = first.cells[2] * cells[3] + first.cells[5] * cells[4] + first.cells[8] * cells[5];
        product.cells[6] = first.cells[0] * cells[6] + first.cells[3] * cells[7] + first.cells[6] * cells[8];
        product.cells[7] = first.cells[1] * cells[6] + first.cells[4] * cells[7] + first.cells[7] * cells[8];
        product.cells[8] = first.cells[2] * cells[6] + first.cells[5] * cells[7] + first.cells[8] * cells[8];
        product.cells[9] = first.cells[0] * cells[9] + first.cells[3] * cells[10] + first.cells[6] * cells[11];
        product.cells[10] = first.cells[1] * cells[9] + first.cells[4] * cells[10] + first.cells[7] * cells[11];
        product.cells[11] = first.cells[2] * cells[9] + first.cells[5] * cells[10] + first.cells[8] * cells[11];
        return product;
    }

    /// <summary>
    /// Multiply this matrix by another matrix. The other matrix would go first in the multiplication
    /// </summary>
    void operator*=(const Matrix& first) {
        Matrix product = first * *this;
        std::copy(product.cells, product.cells + 12, cells);
    }

    // Transformation Methods

    static Matrix Translation(const float3 v) {
        Matrix m{};
        m.cells[9] += v.x;
        m.cells[10] += v.y;
        m.cells[11] += v.z;
        return m;
    }

    static Matrix Rotation(const float3 axis, float degrees) {
        // This is also from Cem's code
        Matrix m{};

        const float rad = DEG2RAD * degrees;
        const float sina = sinf(rad);
        const float cosa = cosf(rad);

        const float t = 1.0f - cosa;
        const float3 a = t * axis;
        const float txy = a.x * axis.y;
        const float txz = a.x * axis.z;
        const float tyz = a.y * axis.z;
        float3 s = sina * axis;

        m.cells[ 0] = a.x * axis.x + cosa;
        m.cells[ 1] = txy + s.z;
        m.cells[ 2] = txz - s.y;
        m.cells[ 3] = txy - s.z;
        m.cells[ 4] = a.y * axis.y + cosa;
        m.cells[ 5] = tyz + s.x;
        m.cells[ 6] = txz + s.y;
        m.cells[ 7] = tyz - s.x;
        m.cells[ 8] = a.z * axis.z + sina;

        return m;
    }

    static Matrix Scale(const float3 s) {
        Matrix m{};
        m.cells[0] = s.x;
        m.cells[4] = s.y;
        m.cells[8] = s.z;
        return m;
    }
};