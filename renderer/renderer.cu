#include "renderer.cuh"

__managed__ Scene theScene;

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
    RenderPixels<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

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
