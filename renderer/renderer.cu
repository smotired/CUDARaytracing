#include "trace.cuh"

// The scene that we are rendering
__managed__ Scene theScene;

void Renderer::BeginRendering() {
    rendering = true;

    auto mainThread = std::thread(&Renderer::DoRendering, this);

    if (mainThread.joinable())
        mainThread.detach();
}

void Renderer::DoRendering() {
    CLERR();
    printf("Starting render...\n");

    // Allocate memory for the rendering image
    const unsigned int size = theScene.render.width * theScene.render.height;
    CERR(cudaMalloc(&theScene.render.results, sizeof(color) * size));

    // And the final converted image
    Color24 *converted;
    CERR(cudaMalloc(&converted, sizeof(Color24) * size));

    // Set up the thread count
    dim3 numBlocks((theScene.render.width + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM, (theScene.render.width + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM);
    dim3 threadsPerBlock(RAY_BLOCKDIM, RAY_BLOCKDIM);
    unsigned int convBlocks = (size + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM;
    unsigned int convThreads = RAY_BLOCKDIM * RAY_BLOCKDIM;

    image.passes = 0;

    // Loop for each pass
    for (int pass = 0; pass < PASSES; pass++) {
        printf("================================\nStarting pass %d.\n", pass);

        // Launch kernel
        printf("Launching kernel...\n");
        TracePrimaryRays<<<numBlocks, threadsPerBlock>>>(pass);
        CLERR();
        CERR(cudaDeviceSynchronize());
        printf("Primary rays finished.\n");

        // Convert results to image format
        printf("Converting colors...\n");
        ConvertColors<<<convBlocks, convThreads>>>(theScene.render.results, converted, size, theScene.camera.sRGB, 1.0f / static_cast<float>(pass + 1));
        CLERR();
        CERR(cudaDeviceSynchronize());
        printf("Color conversion finished.\n");

        // Bring results to host
        CERR(cudaMemcpy(image.pixels, converted, sizeof(Color24) * size, cudaMemcpyDeviceToHost));

        // Refresh display
        image.passes++;
    }

    // Free remaining cuda memory
    printf("================================\nFreeing final memory.\n");
    CERR(cudaFree(converted));
    CERR(cudaFree(theScene.render.results));

    // We are done
    rendering = false;
    printf("Render complete!\n");
}


void Renderer::StopRendering() {
    rendering = false;
}
