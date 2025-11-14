/// Triangular mesh class
/// This is heavily based on Cem's cyTriMesh
#pragma once
#include "rays.cuh"

class Mesh {
protected:
    float3 *v;  // Vertices
    uint3 *f;   // Faces
    float3 *vn; // Vertex normals
    uint3 *fn;  // Normal faces
    float3 *vt; // Texture vertices
    uint3 *ft;  // Texture faces

    size_t nv;  // Vertex count
    size_t nf;  // Face count
    size_t nvn; // Vertex normal count
    size_t nvt; // Texture vertex count

    Box box;    // outer bounding box

    void Clear() {
        nv = 0;
        nf = 0;
        nvn = 0;
        nvt = 0;
        box.Init();

        cudaFree(v);
        cudaFree(f);
        cudaFree(vn);
        cudaFree(fn);
        cudaFree(vt);
        cudaFree(ft);
    }
public:
    Mesh() : v(nullptr), f(nullptr), vn(nullptr), fn(nullptr), vt(nullptr), ft(nullptr),
            nv(0), nf(0), nvn(0), nvt(0), box() { }
    Mesh( Mesh const &t ) : v(nullptr), f(nullptr), vn(nullptr), fn(nullptr), vt(nullptr), ft(nullptr),
            nv(0), nf(0), nvn(0), nvt(0), box() { *this = t; }
    virtual ~Mesh() { Clear(); }

    [[nodiscard]] __device__ bool HasNormals() const { return nvn > 0; }
    [[nodiscard]] __device__ bool HasTextureVertices() const { return nvt > 0; }

    [[nodiscard]] Box const& GetBoundingBox() const { return box; }
    void ComputeBoundingBox();                 // Compute the bounding box from vertices

    // Load from a file
    bool LoadFromFileObj( char const* filename );
};
