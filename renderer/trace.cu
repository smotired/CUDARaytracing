#include "trace.cuh"

__global__ void TracePrimaryRays() {
    // Each thread is responsible for 1 pixel in the block, across each iteration.
    // Define coords of starting pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x; // We only have 1 block right now
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;
    const float3 pixelCoords = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // If the start pixel is already out of bounds, we can just quit.
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    Ray ray(theScene.camera.position, pixelCoords - theScene.camera.position, pI);
    TraceRay(ray);
}

__device__ void TraceRay(Ray &ray, int hitSide) {
    // Add some bias
    ray.pos += ray.dir * BIAS;

    // Initialize a hit
    Hit hit;

    // Primary rays should only check front hits
    if (ray.IsPrimary() && hitSide & HIT_FRONT) hitSide = HIT_FRONT;

    // Loop through the object list
    bool hitAnything = false;
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;
        if (HAS_OBJ(node->object)) {
            // Trace a ray
            node->ToLocal(ray);

            // Check for intersection and transform hit
            if (OBJ_INTERSECT(node->object, ray, hit, hitSide)) {
                hitAnything = true;
                node->FromLocal(hit);
            }
            node->FromLocal(ray);
        }
    }

    // Shade, or add color from environment. Assume no hit means no absorption.
    if (hitAnything) hit.node->material->Shade(ray, hit);
    else theScene.render.results[ray.pixel] += ray.contribution * color(0, 0, 0.1f);

    // If this is a primary ray, update the Z buffer
    if (ray.IsPrimary() && hitAnything)
        theScene.render.zBuffer[ray.pixel] = fmin(theScene.render.zBuffer[ray.pixel], hit.z);
}

__device__ bool TraceShadowRay(ShadowRay& ray, const float3 n, const float tMax, const int hitSide) {
    // Add some bias
    ray.pos += ray.dir * BIAS + n * BIAS;

    // Loop through the object list
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;
        if (HAS_OBJ(node->object)) {
            // Trace a ray
            node->ToLocal(ray);

            if (OBJ_INTSHADOW(node->object, ray, tMax, hitSide))
                return true;

            node->FromLocal(ray);
        }
    }

    return false;
}
