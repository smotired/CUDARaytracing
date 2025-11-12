#include "renderer.cuh"
#include "trace.cuh"

// The scene that we are rendering
__managed__ Scene theScene;

void Renderer::BeginRendering() {
    rendering = true;

    CLERR();
    // printf("Starting render...\n");

    // Allocate memory for the rendered image
    unsigned int size = theScene.render.width * theScene.render.height;
    CERR(cudaMalloc(&theScene.render.results, sizeof(color) * size));
    CERR(cudaMalloc(&theScene.render.zBuffer, sizeof(float) * size));

    // Set up the thread count
    unsigned int xBlocks = (theScene.render.width + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    unsigned int yBlocks = (theScene.render.height + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    dim3 numBlocks(xBlocks, yBlocks);
    dim3 threadsPerBlock(RAY_THREADS_PER_BLOCK_X, RAY_THREADS_PER_BLOCK_X);

    // Launch kernel
    // printf("Casting primary rays...\n");
    DispatchPrimaryRays<<<numBlocks, threadsPerBlock>>>();
    // DispatchPrimaryRays<<<1, 1>>>();
    CLERR();
    CERR(cudaDeviceSynchronize());
    // printf("Primary rays finished.\n");

    // Copy z buffer back and free
    cudaMemcpy(image.zBuffer, theScene.render.zBuffer, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFree(theScene.render.zBuffer);

    // Convert results to image format
    Color24 *converted;
    CERR(cudaMalloc(&converted, sizeof(Color24) * size));

    unsigned int convBlocks = (size + RAY_THREADS_PER_BLOCK_X - 1) / RAY_THREADS_PER_BLOCK_X;
    unsigned int convThreads = RAY_THREADS_PER_BLOCK_X * RAY_THREADS_PER_BLOCK_X;
    // printf("Converting colors...\n");
    ConvertColors<<<convBlocks, convThreads>>>(theScene.render.results, converted, size);
    CLERR();
    CERR(cudaDeviceSynchronize());
    // printf("Color conversion finished.\n");

    // Bring results to host
    CERR(cudaMemcpy(image.pixels, converted, sizeof(Color24) * size, cudaMemcpyDeviceToHost));

    // Free remaining cuda memory
    CERR(cudaFree(converted));
    CERR(cudaFree(theScene.render.results));

    // We are done
    rendering = false;
}

void Renderer::StopRendering() {
    rendering = false;
}
