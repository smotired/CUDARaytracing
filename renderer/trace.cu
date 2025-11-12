#include "trace.cuh"
#include "renderer.cuh"

__global__ void DispatchPrimaryRays() {
    DEBUG_PRINT("START DISPATCH\n");
    const unsigned int pX = DEBUG ? DEBUG_X : blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = DEBUG ? DEBUG_Y : blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;

    // Return if we are out of bounds
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Get the world-space coordinates of this pixel on the screen
    const float3 pixelCoords = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // Create and cast a ray
    DEBUG_PRINT("pI: %d, pCoords: %.2f,%.2f,%.2f\n", pI, pixelCoords.x, pixelCoords.y, pixelCoords.z);
    DEBUG_PRINT("START CAST\n");
    Ray ray(theScene.camera.position, pixelCoords - theScene.camera.position, pI);
    ray.pos += ray.dir * BIAS;
    DEBUG_PRINT("RayO: %.2f,%.2f,%.2f, RayD: %.2f,%.2f,%.2f\n", ray.pos.x, ray.pos.y, ray.pos.z, ray.dir.x, ray.dir.y, ray.dir.z);
    const bool hit = TraceRay(ray, HIT_FRONT);
    DEBUG_PRINT("DID HIT: %d\n", hit);
    if (hit) {
        DEBUG_PRINT("HIT POS: %.2f,%.2f,%.2f\n", ray.hit.pos.x, ray.hit.pos.y, ray.hit.pos.z);
        DEBUG_PRINT("HIT Z: %.2f\n", ray.hit.z);
        DEBUG_PRINT("HIT N: %.2f,%.2f,%.2f\n", ray.hit.n.x, ray.hit.n.y, ray.hit.n.z);;
        DEBUG_PRINT("HIT FRONT: %d\n", ray.hit.front);
    }

    DEBUG_PRINT("START SHADE\n");
    color sample = color(0, 0, 0.1f); // Dark blue if we hit nothing
    if (hit) {
        DEBUG_PRINT("MATERIAL PTR: %p\n", ray.hit.node->material);
        const Material& mtl = *ray.hit.node->material;
        DEBUG_PRINT("DIFFUSE: %.2f,%.2f,%.2f, SPECULAR: %.2f,%.2f,%.2f, GLOSSINESS: %.2f\n", mtl.diffuse.x, mtl.diffuse.y, mtl.diffuse.z, mtl.specular.x, mtl.specular.y, mtl.specular.z, mtl.glossiness);
        sample = ray.hit.node->material->Shade(ray);
    }

    DEBUG_PRINT("FINAL COLOR: %.2f,%.2f,%.2f, FINAL Z: %.2f\n", sample.x, sample.y, sample.z, ray.hit.z);
    theScene.render.results[pI] = sample;
    theScene.render.zBuffer[pI] = ray.hit.z;
}

__device__ bool TraceRay(Ray &ray, int hitSide) {
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
    return hitAnything;
}