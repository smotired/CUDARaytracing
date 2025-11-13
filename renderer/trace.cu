#include "trace.cuh"

__managed__ RayQueue rayQueue;

__global__ void DispatchRows() {
    // Each thread is responsible for 1 pixel in the block, across each iteration.
    // Define coords of starting pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x; // We only have 1 block right now
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int fpI = pY * theScene.render.width + pX;
    const float3 firstPixel = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // If the start pixel is already out of bounds, we can just quit. Only happens on the far right and bottom edge of the images.
    bool finished = false;
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        finished = true; // We can't just return because of control flow requirements with __syncthreads().

    // Find out just how many pixels we are responsible for
    const size_t colCount = (theScene.render.width + RAY_ITERSIZE - 1) / RAY_ITERSIZE;
    const size_t rowCount = (theScene.render.height + RAY_ITERSIZE - 1) / RAY_ITERSIZE;

    // Cast rays for each row and column
    for (int row = 0; row < rowCount; row++) {
        for (int col = 0; col < colCount; col++) {
            // If this pixel is out of bounds, we're done.
            if ((pX + RAY_ITERSIZE * col) >= theScene.render.width || (pY + RAY_ITERSIZE * row) >= theScene.render.height)
                finished = true;

            // Calculate index and coordinates of the pixel by adding BLOCKDIM * BLOCKCOUNT rows
            const unsigned int pI = (row * RAY_ITERSIZE * theScene.render.width) + (col * RAY_ITERSIZE) + fpI;
            const float3 pixel = firstPixel +theScene.render.pixelSize * RAY_ITERSIZE * (col * theScene.render.cX - row * theScene.render.cY);

            // Enqueue the primary ray and set up z buffer
            if (!finished) {
                rayQueue.Enqueue(blockIdx, Ray(theScene.camera.position, pixel - theScene.camera.position, pI));
                theScene.render.zBuffer[pI] = BIGFLOAT;
            }

            while (true) {
                // Wait for all the threads to be ready
                __syncthreads();

                // If this is the first thread in the block, swap the queues and reset
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    rayQueue.SwapQueues(blockIdx);
                }

                // Wait for all the threads to be ready
                __syncthreads();

                // If the read queue is empty, all threads can break out of this loop.
                if (rayQueue.IsEmpty(blockIdx)) break;

                // All threads must finish that check before we can start dequeuing.
                __syncthreads();

                // If we aren't offscreen, dequeue rays until the queue is exhausted
                while (!finished) {
                    Ray* ray = rayQueue.Dequeue(blockIdx);
                    if (ray == nullptr) break;
                    TraceRay(blockIdx, *ray); // Will enqueue rays as needed
                }
                // The queue is exhausted, so loop back until all threads acknowledge that.
            }
            // Move on to the next iteration
        }
    }
}

__device__ void TraceRay(const uint3 blockIdx, Ray &ray, int hitSide) {
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
    if (hitAnything) ray.hit.node->material->Shade(blockIdx, ray);
    else theScene.render.results[ray.pixel] += ray.contribution * color(0, 0, 0.1f);

    // If this is a primary ray, update the Z buffer
    if (ray.IsPrimary() && hitAnything) {
        theScene.render.zBuffer[ray.pixel] = fmin(theScene.render.zBuffer[ray.pixel], ray.hit.z);
    }
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