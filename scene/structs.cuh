/// Structs that are part of the scene
#pragma once
#include "structs.cuh"
#include "../math/float3.cuh"
#include "../math/color.cuh"
#include "../math/matrix.cuh"
#include "../lib/xmlload.cuh"

#define BIGFLOAT std::numeric_limits<float>::max()

// Overview objects
struct RenderInfo;
struct Scene;
struct Camera;

// Raycasting objects
struct Hit;
struct Ray;

// Properties of nodes or the scene
struct Box;
struct Node;
class Object;
class Material; // All materials use our blinn-phong like material
class Light; // will eventually be replaced with emissive material

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

// Hit sides
#define HIT_NONE 0
#define HIT_FRONT 1
#define HIT_BACK 2
#define HIT_FRONT_AND_BACK HIT_FRONT | HIT_BACK

struct Hit {
    // Position of the hit in world space
    float3 pos;

    // Distance of the hit in world space
    float z;

    // Normal vector at hit point
    float3 n;

    // If we hit the front of the object
    bool front;

    // Pointer to the node that we hit
    Node* node;

    // Initialize with default values
    void Init() {
        pos = F3_ZERO;
        z = BIGFLOAT;
        n = F3_UP;
        front = true;
        node = nullptr;
    }

    // Transform a hit with a matrix
    __host__ __device__ void Transform(const Matrix& tm) {
        pos = tm * pos;
        n = norm(tm % n);
    }
};

struct Ray {
    // Origin of the ray
    float3 pos;

    // Direction of the ray
    float3 dir;

    // Hit position of the ray
    Hit hit;

    // Pixel index (y * width + x) of the ray
    unsigned int pixel;

    // Contribution of the ray to the final color
    color contribution;

    // Initialize a ray
    void Init(const float3 p, const float3 d, const unsigned int pI, const color cont = WHITE) {
        pos = p;
        dir = d;
        pixel = pI;
        contribution = cont;
        hit.Init();
    }

    // Transform a ray with a matrix
    __host__ __device__ void Transform(const Matrix& tm) {
        pos = tm * pos;
        dir = tm % dir;
        hit.Transform(tm);
    }
};

/// <summary>
/// Represents the actual body that goes with a node
/// </summary>
class Object {
public:
    virtual bool IntersectRay(Ray& ray, int hitSide) const = 0;
    virtual Box GetBoundBox() const = 0;
    virtual void ViewportDisplay( /*Material const* material*/ ) const {} // used for OpenGL preview
    virtual void Load( Loader const &loader ) {} // Used for things with special load requirements like meshes
};

/// <summary>
/// Represents a temporary light in the scene
/// </summary>
class Light {
    virtual color Illuminate(Ray const& ray, float3 &dir) const = 0; // Returns intensity and direction
    virtual bool IsAmbient() const { return false; }
    virtual void SetViewportLight( int lightID ) const {} // used for OpenGL preview
    virtual void Load( Loader const &loader ) {}
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
/// Minimum and maximum corners of a node
/// </summary>
struct Box {
    // Minimum positions of the box
    float3 pmin;

    // Maximum positions of the box
    float3 pmax;

    // Initializes the box, such that there exists no point inside the box (i.e. it is empty).
    void Init() { pmin = float3(BIGFLOAT,BIGFLOAT,BIGFLOAT); pmax = float3(-BIGFLOAT,-BIGFLOAT,-BIGFLOAT); }

    // Returns true if the box is empty; otherwise, returns false.
    [[nodiscard]] bool IsEmpty() const { return pmin.x>pmax.x || pmin.y>pmax.y || pmin.z>pmax.z; }

    // Expand the box to include a point
    void operator+=(const float3 pos) {
        if (pmin.x > pos.x) pmin.x = pos.x;
        if (pmin.y > pos.y) pmin.y = pos.y;
        if (pmin.z > pos.z) pmin.z = pos.z;
        if (pmax.x < pos.x) pmax.x = pos.x;
        if (pmax.y < pos.y) pmax.y = pos.y;
        if (pmax.z < pos.z) pmax.z = pos.z;
    }

    // Use the slab method to determine if the ray intersects with the box
    __host__ __device__ bool IntersectRay(const Ray& ray, float& dist, const float t_max = BIGFLOAT) const {
	    const float3 inv = float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);

        const float3 tLow = (pmin - ray.pos) * inv;
        const float3 tHigh = (pmax - ray.pos) * inv;

        const float3 tClose(std::fmin(tLow.x, tHigh.x),
                      std::fmin(tLow.y, tHigh.y),
                      std::fmin(tLow.z, tHigh.z));
        const float3 tFar(std::fmax(tLow.x, tHigh.x),
                    std::fmax(tLow.y, tHigh.y),
                    std::fmax(tLow.z, tHigh.z));

        const float tEnter = std::fmax(tClose.x, std::fmax(tClose.y, tClose.z));
        const float tExit = std::fmin(tFar.x, std::fmin(tFar.y, tFar.z));

        // If it actually enters, the box, return the intersection distance
        if ((tEnter >= 0 || tExit >= 0) && tEnter <= tExit && tEnter <= t_max) {
            dist = tEnter >= 0 ? tEnter : tExit;
            return true;
        }

        return false;
    }
};

/// <summary>
/// A node represents an object in the scene.
/// </summary>
struct Node {
    // The amount of children on this node. The node list is in depth-first order.
    size_t childCount;

    // The object associated with this node
    Object* object;

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
    Node(Object* obj, Material* mat, const Matrix &transformation) {
        object = obj;
        material = mat;
        tm = transformation;
        itm = transformation.GetInverse();
        Box objectBoundingBox = object->GetBoundBox();
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