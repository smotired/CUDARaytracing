#include "renderer.cuh"

__managed__ Scene theScene;

/**
// Temporary implementation of BeginRendering: Launch all the threads and have them just return pixel coords as a color
__global__ void RenderPixels() {
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;

    // Return if we are out of bounds
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Use x for r and y for g, and then y for z buffer as well
    const float dX = pX / (float)theScene.render.width;
    const float dY = pY / (float)theScene.render.height;
    theScene.render.results[pI] = color(dX, dY, 0);
    theScene.render.zBuffer[pI] = pY;
} **/

__device__ color TraceRay(Ray &ray, int hitSide = HIT_FRONT_AND_BACK) {
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

__global__ void RenderPixels() {
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;

    // Return if we are out of bounds
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Set up camera stuff (TODO: Precompute these on the RenderInfo)
    const float aspectRatio = (float)theScene.render.width / (float)theScene.render.height;
    const float fovRad = theScene.camera.fov * DEG2RAD;
    const float focaldist = length(theScene.camera.target - theScene.camera.position);
    const float planeHeight = 2 * focaldist * tanf(fovRad * 0.5f); // TODO: store that on camera too as we will need it later
    const float planeWidth = aspectRatio * planeHeight;
    const float3 cZ = norm(theScene.camera.target - theScene.camera.position);
    const float3 cY = norm(theScene.camera.up);
    const float3 cX = cross(cZ, cY);

    // Find info about pixel array
    const float pixelSize = planeHeight / (float)theScene.render.height;
    const float3 planeCenter = theScene.camera.position + focaldist * cZ;
    const float3 topLeftCorner = planeCenter - (planeWidth * 0.5f * cX) + (planeHeight * 0.5f * cY);
    const float3 topLeftPixel = topLeftCorner + pixelSize * 0.5f * (cX - cY);

    // Get the world-space coordinates of this pixel on the screen
    const float3 pixelCoords = topLeftPixel + pixelSize * (pX * cX - pY * cY);

    // Create and cast a ray
    Ray ray(theScene.camera.position, pixelCoords - theScene.camera.position, pI);
    const color sample = TraceRay(ray);

    // TODO: When we have a ray queue they should handle this with their contributions
    theScene.render.results[pI] = sample;
    theScene.render.zBuffer[pI] = ray.hit.z;
}

void Renderer::BeginRendering() {
    rendering = true;

    // Allocate memory for the rendered image
    unsigned int size = theScene.render.width * theScene.render.height;
    cudaMalloc(&theScene.render.results, sizeof(color) * size);
    cudaMalloc(&theScene.render.zBuffer, sizeof(float) * size);

    // Set up the thread count
    unsigned int xBlocks = (theScene.render.width + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    unsigned int yBlocks = (theScene.render.height + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    dim3 numBlocks(xBlocks, yBlocks);
    dim3 threadsPerBlock(RAY_THREADS_PER_BLOCK_X, RAY_THREADS_PER_BLOCK_X);

    // Launch kernel
    Scene& scene = theScene; // for debugging
    RenderPixels<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    // Copy z buffer back and free
    cudaMemcpy(image.zBuffer, theScene.render.zBuffer, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFree(theScene.render.zBuffer);

    // Convert results to image format
    Color24 *converted;
    cudaMalloc(&converted, sizeof(Color24) * size);

    unsigned int convBlocks = (size + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    unsigned int convThreads = RAY_THREADS_PER_BLOCK_X * RAY_THREADS_PER_BLOCK_X;
    ConvertColors<<<convBlocks, convThreads>>>(theScene.render.results, converted, size);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    // Bring results to host
    cudaMemcpy(image.pixels, converted, sizeof(Color24) * size, cudaMemcpyDeviceToHost);

    // Free remaining cuda memory
    cudaFree(converted);
    cudaFree(theScene.render.results);

    // We are done
    rendering = false;

    ComputeZBufferImage();
    image.SaveImage("output.png");
    image.SaveZBufferImage("outputZ.png");
}

void Renderer::StopRendering() {
    rendering = false;
}
