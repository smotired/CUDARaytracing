#include "trace.cuh"

// The scene that we are rendering
__managed__ Scene theScene;

void Renderer::BeginRendering() {
    rendering = true;

    CLERR();
    printf("Starting render...\n");

    // Allocate memory for the rendered image
    const unsigned int size = theScene.render.width * theScene.render.height;
    CERR(cudaMalloc(&theScene.render.results, sizeof(color) * size));
    CERR(cudaMalloc(&theScene.render.zBuffer, sizeof(float) * size));

    // Set up the thread count
    dim3 numBlocks((theScene.render.width + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM, (theScene.render.width + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM);
    dim3 threadsPerBlock(RAY_BLOCKDIM, RAY_BLOCKDIM);

    // Increase stack size on device
    size_t originalStackSize;
    CERR(cudaDeviceGetLimit(&originalStackSize, cudaLimitStackSize));
    CERR(cudaDeviceSetLimit(cudaLimitStackSize, RAY_STACK_KB * 1024));

    // Launch kernel
    printf("Launching kernel...\n");
    TracePrimaryRays<<<numBlocks, threadsPerBlock>>>();
    CLERR();
    CERR(cudaDeviceSynchronize());
    printf("Primary rays finished.\n");

    // Copy z buffer back and free
    cudaMemcpy(image.zBuffer, theScene.render.zBuffer, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFree(theScene.render.zBuffer);

    // Revert stack size on device
    CERR(cudaDeviceSetLimit(cudaLimitStackSize, originalStackSize));

    // Convert results to image format
    Color24 *converted;
    CERR(cudaMalloc(&converted, sizeof(Color24) * size));

    unsigned int convBlocks = (size + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM;
    unsigned int convThreads = RAY_BLOCKDIM * RAY_BLOCKDIM;
    printf("Converting colors...\n");
    ConvertColors<<<convBlocks, convThreads>>>(theScene.render.results, converted, size);
    CLERR();
    CERR(cudaDeviceSynchronize());
    printf("Color conversion finished.\n");

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
