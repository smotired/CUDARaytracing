/// Structs that are part of the scene
#pragma once
#include "../math/float3.cuh"
#include "../math/color.cuh"
#include "../math/matrix.cuh"
#include "../lib/xmlload.cuh"
#include "rays.cuh"
#include "objects.cuh"
#include "lights.cuh"
#include "rng.cuh"

// Overview objects
struct RenderInfo;
struct Scene;
struct Camera;

// Properties of nodes or the scene
struct Box;
struct Node;

/// <summary>
/// The RenderInfo contains information about the final image, as well as references to
/// the actual image arrays.
/// </summary>
struct RenderInfo {
    // Width of the image in pixels
    unsigned int width;

    // Height of the image in pixels
    unsigned int height;

    // Results of the render for each pixel
    color* results;

    // Extra information for render
    float3* normals;
    color* albedos;

    // Size of a single pixel in world space
    float pixelSize;

    // Unit vectors for camera.
    float3 cX, cY, cZ;

    // Image plane. X and Y are width and height, Z is distance from camera along cZ.
    float3 plane;

    // World space coordinates of the pixel in the top left corner. Used to calculate offset of remaining pixels.
    float3 topLeftPixel;

    // Load the RenderInfo
    void Load( Loader const &loader );
};

/// <summary>
/// The Camera defines the origin and direction of primary rays.
/// </summary>
struct Camera {
    // Position of the camera
    float3 position;

    // Target position of the camera, in world space. If DOF is present, this will be exactly in focus.
    float3 target;

    // Up vector of the camera.
    float3 up;

    // Field of view in degrees
    float fov;

    // Depth of field in world space units
    float dof;

    // If it uses sRGB color space
    bool sRGB;

    // Load the camera
    void Load( Loader const &loader );
};

/// <summary>
/// The scene holds all the information about the render and objects.
/// </summary>
struct Scene {
    // The information about the current render
    RenderInfo render;

    // The information about the camera
    Camera camera;

    // The nodes in the tree, in depth-first order
    Node* nodes;

    // The amount of nodes in the scene.
    size_t nodeCount;

    // The list of lights in the scene
    Light* lights;

    // How many lights are present in the scene
    size_t lightCount;

    // The list of materials in the scene
    Material* materials;

    // How many materials are in the scene
    size_t materialCount;

    // The environment texture
    Texture* env;

    // Load the scene
    void Load( Loader const &loader );
};

class Material {
public:
    Texture* diffuse = nullptr;  // How much light is scattered
    Texture* specular = nullptr; // How much light is reflected
    float glossiness = 20;  // Smoothness of the surface

    Texture* reflection = nullptr; // How much light is reflected (will be removed after path tracing)
    Texture* refraction = nullptr; // How much light is directly refracted
    color absorption = BLACK; // How much light is absorbed
    float ior = 1;

    __device__ void Shade(Ray const& ray, Hit const& hit) const; // Legacy
    void SetViewportMaterial( int mtlID=0 ) const {} // used for OpenGL display (unused though i think)
    void Load( Loader const &loader ) { /* Will do something later */ }

    __device__ bool GenerateSample(float3 const& v, Hit const& hit, float3& dir, curandStateXORWOW_t *rng, SampleInfo& info) const;
    __device__ void GetSampleInfo(float3 const& v, Hit const& hit, float3 const& dir, SampleInfo &info) const;
};

/// <summary>
/// A node represents an object in the scene.
/// </summary>
struct Node {
    // The amount of children on this node. The node list is in depth-first order.
    size_t childCount;

    // The object associated with this node
    ObjectPtr object;

    // The material associated with this node
    const Material* material;

    // The bounding box of this object.
    Box boundingBox{};

    // Transformation matrix from world space to local space
    Matrix tm;

    // Transformation matrix from local space to world space
    Matrix itm;

    // Default constructor
    Node() : childCount(0), object(static_cast<Sphere *>(nullptr)), material(nullptr), boundingBox(Box()), tm(Matrix()), itm(Matrix()) {
    }

    // Transform a ray from world space to local space
    __host__ __device__ void ToLocal(Ray& ray) const {
        itm.TransformPosition(ray.pos);
        itm.TransformDirection(ray.dir);
    }

    // Transform a ray from local space to world space
    __host__ __device__ void FromLocal(Ray& ray) const {
        tm.TransformPosition(ray.pos);
        tm.TransformDirection(ray.dir);
    }

    // Transform a shadow ray from world space to local space
    __host__ __device__ void ToLocal(ShadowRay& ray) const {
        itm.TransformPosition(ray.pos);
        itm.TransformDirection(ray.dir);
    }

    // Transform a shadow ray from local space to world space
    __host__ __device__ void FromLocal(ShadowRay& ray) const {
        tm.TransformPosition(ray.pos);
        tm.TransformDirection(ray.dir);
    }

    // Transform a ray's hit from local space to world space
    __host__ __device__ void FromLocal(Hit& hit) {
        tm.TransformPosition(hit.pos);
        itm.TransformNormal(hit.n);
        hit.node = this;
    }

    // Create a tight AABB in world space around the possibly-not-aligned box in object space.
    Box FromLocal(const Box& box) const {
        Box transformed;
        transformed.Init();

        // Calculate and transform each corner of the original box, and add it to the new box.
        float3 x0y0z0 = box.pmin;
        tm.TransformPosition(x0y0z0);
        transformed += x0y0z0;
        float3 x0y0z1 = float3(box.pmin.x, box.pmin.y, box.pmax.z);
        tm.TransformPosition(x0y0z1);
        transformed += x0y0z1;

        float3 x0y1z0 = float3(box.pmin.x, box.pmax.y, box.pmin.z);
        tm.TransformPosition(x0y1z0);
        transformed += x0y1z0;
        float3 x0y1z1 = float3(box.pmin.x, box.pmax.y, box.pmax.z);
        tm.TransformPosition(x0y1z1);
        transformed += x0y1z1;

        float3 x1y0z0 = float3(box.pmax.x, box.pmin.y, box.pmin.z);
        tm.TransformPosition(x1y0z0);
        transformed += x1y0z0;
        float3 x1y0z1 = float3(box.pmax.x, box.pmin.y, box.pmax.z);
        tm.TransformPosition(x1y0z1);
        transformed += x1y0z1;

        float3 x1y1z0 = float3(box.pmax.x, box.pmax.y, box.pmin.z);
        tm.TransformPosition(x1y1z0);
        transformed += x1y1z0;
        float3 x1y1z1 = box.pmax;
        tm.TransformPosition(x1y1z1);
        transformed += x1y1z1;

        return transformed;
    }

    // Calculate the bounding box of this node, assuming bounding boxes of child nodes have been calculated.
    void CalculateBoundingBox(const Node* nodeList, const unsigned int nodeID) {
        boundingBox.Init();

        // If this node has an object, add its bounding box
        if (HAS_OBJ(object)) boundingBox += FromLocal(OBJ_BOUNDBOX(object));

        // Recurse to children
        unsigned int children = childCount;
        unsigned int skip = 0;
        for (unsigned int i = nodeID + 1; children > 0; i++) {
            // If we should skip this node, also add its children as skippable
            if (skip > 0) {
                skip--;
                skip += nodeList[i].childCount;
            }
            // Otherwise, this is a direct child, so add its bounding box and skip its children
            else {
                boundingBox += nodeList[i].boundingBox;
                skip = nodeList[i].childCount;
                children--;
            }
        }
    }

    // Load the node into the scene
    void Load( Loader const &loader );
};
