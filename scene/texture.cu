#include <cmath>

#include "texture.cuh"

Texture::Texture(color const* source, const unsigned int width, const unsigned int height) : width(width), height(height) {
    Color24* transformed = new Color24[width * height];
    for (unsigned int i = 0; i < width * height; ++i)
        transformed[i] = Color24(source[i]);

    cudaMalloc(&data, width * height * sizeof(Color24));
    cudaMemcpy(data, transformed, width * height * sizeof(Color24), cudaMemcpyHostToDevice);
}

__device__ float3 TileClamp( float3 const &uvw )
{
    float3 u;
    u.x = uvw.x - (int) uvw.x;
    u.y = uvw.y - (int) uvw.y;
    u.z = uvw.z - (int) uvw.z;
    if ( u.x < 0 ) u.x += 1;
    if ( u.y < 0 ) u.y += 1;
    if ( u.z < 0 ) u.z += 1;
    return u;
}

__device__ color Texture::Eval(float3 const& uvw) const {
    // yoinked from cem again
    if ( width + height == 0 ) return BLACK;
    float3 u = TileClamp(uvw);

    const float x = width * u.x;
    const float y = height * u.y;
    int ix = (int)x;
    int iy = (int)y;
    const float fx = x - ix;
    const float fy = y - iy;

    if ( ix < 0 ) ix -= (ix/width - 1)*width;
    if ( ix >= width ) ix -= (ix/width)*width;
    int ixp = ix+1;
    if ( ixp >= width ) ixp -= width;

    if ( iy < 0 ) iy -= (iy/height - 1)*height;
    if ( iy >= height ) iy -= (iy/height)*height;
    int iyp = iy+1;
    if ( iyp >= height ) iyp -= height;

    return	data[iy *width+ix ].ToColor() * ((1-fx)*(1-fy)) +
            data[iy *width+ixp].ToColor() * (   fx *(1-fy)) +
            data[iyp*width+ix ].ToColor() * ((1-fx)*   fy ) +
            data[iyp*width+ixp].ToColor() * (   fx *   fy );
}

__device__ color Texture::EvalEnvironment(const float3 &dir) const
{
    const float3 d = asNorm(dir);
    const float phi = atan2f(d.x, d.z);
    const float theta = acosf(fminf(fmaxf(d.y, -1.0f), 1.0f)); // clamp avoids nans

    const float u = (phi + M_PI) * (0.5f / M_PI);
    const float v = theta * (1.0f / M_PI);
    return Eval(make_float3(u, v, 0.0f));
}