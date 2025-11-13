#include "trace.cuh"
#include "renderer.cuh"

__global__ void DispatchPrimaryRays() {
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;

    // Return if we are out of bounds
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Get the world-space coordinates of this pixel on the screen
    const float3 pixelCoords = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // Create and cast a ray
    Ray ray(theScene.camera.position, pixelCoords - theScene.camera.position, pI);
    TraceRay(ray, HIT_FRONT);
}

__device__ void TraceRay(Ray &ray, int hitSide) {
    // Add some bias
    ray.pos += ray.dir * BIAS;

    // Loop through the object list
    bool hitAnything = false;
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;
        if (HAS_OBJ(node->object)) {
            // Trace a ray
            node->ToLocal(ray);
            const bool hit = cuda::std::visit(
                [&ray, hitSide](const auto &object) { return object->IntersectRay(ray, hitSide); }, node->object);

            if (hit) {
                // Apply the node
                ray.hit.node = node;
                hitAnything = true;
            }
            node->FromLocal(ray);
        }
    }

    // Shade, or add color from environment. Assume no hit means no absorption.
    if (hitAnything) ray.hit.node->material->Shade(ray);
    else theScene.render.results[ray.pixel] += ray.contribution * color(0, 0, 0.1f);

    // If this is a primary ray, update the Z buffer
    if (ray.IsPrimary()) theScene.render.zBuffer[ray.pixel] = fmin(theScene.render.zBuffer[ray.pixel], ray.hit.z);
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

            const bool hit = cuda::std::visit(
                [&ray, tMax, hitSide](const auto &object) { return object->IntersectShadowRay(ray, tMax, hitSide); }, node->object);
            if (hit) return true;

            node->FromLocal(ray);
        }
    }

    return false;
}