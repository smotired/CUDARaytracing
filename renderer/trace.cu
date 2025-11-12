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
    const color sample = TraceRay(ray);

    // TODO: When we have a ray queue they should handle this with their contributions
    theScene.render.results[pI] = sample;
    theScene.render.zBuffer[pI] = ray.hit.z;
}

__device__ color TraceRay(Ray &ray, int hitSide) {
    // Loop through the object list
    color col = color(0, 0, 0.1); // Initialize dark blue so we can tell it works
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;
        if (HAS_OBJ(node->object)) {
            // Trace a ray
            ray.Transform(node->itm);
            const bool hit = cuda::std::visit(
                [&ray, hitSide](const auto &object) { return object->IntersectRay(ray, hitSide); }, node->object);

            if (hit) {
                // Apply the node, and return white
                ray.hit.node = node;
                col = WHITE;
            }
            ray.Transform(node->tm);
        }
    }
    return col;
}