/// Structs that are part of the scene
#pragma once
#include "../math/float3.cuh"
#include "../math/color.cuh"
#include "../math/matrix.cuh"
#include "../lib/xmlload.cuh"
#include "rays.cuh"
#include "objects.cuh"
#include "lights.cuh"

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

    // If the camera renders in sRGB
    // bool sRGB;

    // Results of the render for each pixel
    color* results;

    // Minimum distance in world space traveled by primary rays from each pixel
    float* zBuffer;

    // Primary ray counts per pixel
    // unsigned int* sampleCounts;

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
    // float dof;

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

    // Load the scene
    void Load( Loader const &loader );
};

class Material {
public:
    color diffuse = BLACK;  // How much light is scattered
    color specular = WHITE; // How much light is reflected
    float glossiness = 512;              // Smoothness of the surface

    [[nodiscard]] color Shade(Ray const& ray) const { return ray.hit.n; };
    void SetViewportMaterial( int mtlID=0 ) const {} // used for OpenGL display
    void Load( Loader const &loader ) { /* Will do something later */ }
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
    Material* material;

    // The bounding box of this object.
    Box boundingBox{};

    // Transformation matrix from world space to local space
    Matrix tm;

    // Transformation matrix from local space to world space
    Matrix itm;

    // Default constructor
    Node() {
        object = nullptr;
        material = nullptr;
        tm = Matrix();
        itm = Matrix();
        boundingBox = Box();
        childCount = 0;
    }

    // Construct a node
    Node(ObjectPtr obj, Material* mat, const Matrix &transformation) {
        object = obj;
        material = mat;
        tm = transformation;
        itm = transformation.GetInverse();
        Box objectBoundingBox = HAS_OBJ(obj) ? cuda::std::visit([](const auto &object) { return object->GetBoundBox(); }, object) : Box();
        boundingBox = Box(tm * objectBoundingBox.pmin, tm * objectBoundingBox.pmax);
        childCount = 0;
    }

    // Transform a ray from world space to local space
    void ToLocal(Ray& ray) const {
        ray.Transform(tm);
    }

    // Transform a ray from local space to world space
    void FromLocal(Ray& ray) const {
        ray.Transform(itm);
    }

    // Load the node into the scene
    void Load( Loader const &loader );
};