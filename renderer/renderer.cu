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
    dim3 numBlocks(RAY_BLOCKCOUNT, RAY_BLOCKCOUNT);
    dim3 threadsPerBlock(RAY_BLOCKDIM, RAY_BLOCKDIM);

    // Allocate enough space for the ray queues. We will have 2 queues, each with enough room for RAY_QUEUE_SIZE rays per pixel going at once.
    printf("Allocating queue...\n");
    cudaMalloc(&rayQueue.rays, sizeof(Ray) * 2 * SINGLE_QUEUE_SIZE * RAY_BLOCKCOUNT * RAY_BLOCKCOUNT);
    cudaMallocManaged(&rayQueue.readIdxs, sizeof(unsigned int) * RAY_BLOCKCOUNT * RAY_BLOCKCOUNT);
    cudaMallocManaged(&rayQueue.writeIdxs, sizeof(unsigned int) * RAY_BLOCKCOUNT * RAY_BLOCKCOUNT);
    cudaMallocManaged(&rayQueue.endIdxs, sizeof(unsigned int) * RAY_BLOCKCOUNT * RAY_BLOCKCOUNT);
    cudaMallocManaged(&rayQueue.maxIdxs, sizeof(unsigned int) * RAY_BLOCKCOUNT * RAY_BLOCKCOUNT);

    // Setup iteration variables for the ray queue
    printf("Initializing blocks...\n");
    for (unsigned int x = 0; x < RAY_BLOCKCOUNT; x++)
        for (unsigned int y = 0; y < RAY_BLOCKCOUNT; y++)
            rayQueue.Init(make_uint3(x, y, 0));

    // Launch kernel
    printf("Launching kernel...\n");
    DispatchRows<<<numBlocks, threadsPerBlock>>>();
    CLERR();
    CERR(cudaDeviceSynchronize());
    printf("Primary rays finished.\n");

    // Copy z buffer back and free
    cudaMemcpy(image.zBuffer, theScene.render.zBuffer, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFree(theScene.render.zBuffer);

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
