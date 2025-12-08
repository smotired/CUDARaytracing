#include "trace.cuh"
#include <OpenImageDenoise/oidn.h>

// The scene that we are rendering
__managed__ Scene theScene;

void Renderer::BeginRendering(const bool wait) {
    rendering = true;
    std::thread t(&Renderer::DoRendering, this);
    if (t.joinable()) {
        if (wait == true) t.join();
        else t.detach();
    }
}

void Renderer::DoRendering() {
    CLERR();
    printf("Starting render...\n");

    // Allocate memory for the rendering image
    const unsigned int size = theScene.render.width * theScene.render.height;
    CERR(cudaMalloc(&theScene.render.results, sizeof(color) * size));
    CERR(cudaMalloc(&theScene.render.normals, sizeof(float3) * size));
    CERR(cudaMalloc(&theScene.render.albedos, sizeof(color) * size));

    // And the final converted image
    Color24 *converted;
    CERR(cudaMalloc(&converted, sizeof(Color24) * size));

    // Set up the thread count
    dim3 numBlocks((theScene.render.width + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM, (theScene.render.height + RAY_BLOCKDIM - 1) / RAY_BLOCKDIM);
    dim3 threadsPerBlock(RAY_BLOCKDIM, RAY_BLOCKDIM);
    unsigned int convBlocks = (size + RAY_BLOCKDIM * RAY_BLOCKDIM - 1) / (RAY_BLOCKDIM * RAY_BLOCKDIM);
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
    printf("================================\n");

    //*
    // WITHOUT DENOISER

    // Free remaining cuda memory
    printf("Freeing final memory.\n");
    CERR(cudaFree(converted));
    CERR(cudaFree(theScene.render.results));
    CERR(cudaFree(theScene.render.normals));
    CERR(cudaFree(theScene.render.albedos));
    image.passes++;

    //*/

    /*
    // WITH DENOISER

    // Format colors, normals, albedo into a format processable by OIDN.
    printf("Preparing for denoise.\n");
    float *prepColors;
    CERR(cudaMalloc(&prepColors, size * 3 * sizeof(float)));
    float *prepNormals;
    CERR(cudaMalloc(&prepNormals, size * 3 * sizeof(float)));
    float *prepAlbedos;
    CERR(cudaMalloc(&prepAlbedos, size * 3 * sizeof(float)));

    PrepareForDenoise<<<convBlocks, convThreads>>>(
        theScene.render.results, theScene.render.normals, theScene.render.albedos,
        prepColors, prepNormals, prepAlbedos,
        size, 1.0f / static_cast<float>(PASSES));
    CLERR();
    CERR(cudaDeviceSynchronize());
    CERR(cudaFree(theScene.render.results));
    CERR(cudaFree(theScene.render.normals));
    CERR(cudaFree(theScene.render.albedos));

    printf("Preparing OIDN device and filter\n");
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    OIDNBuffer colorBuffer = oidnNewBuffer(device, size * 3 * sizeof(float));
    OIDNBuffer normalBuffer = oidnNewBuffer(device, size * 3 * sizeof(float));
    OIDNBuffer albedoBuffer = oidnNewBuffer(device, size * 3 * sizeof(float));

    // TODO: Filter creation is supposedly expensive so might do it elsewhere
    OIDNFilter filter = oidnNewFilter(device, "RT");
    oidnSetFilterImage(filter, "color", colorBuffer, OIDN_FORMAT_FLOAT3, theScene.render.width, theScene.render.height, 0, 0, 0);
    oidnSetFilterImage(filter, "normal", normalBuffer, OIDN_FORMAT_FLOAT3, theScene.render.width, theScene.render.height, 0, 0, 0);
    oidnSetFilterImage(filter, "albedo", albedoBuffer, OIDN_FORMAT_FLOAT3, theScene.render.width, theScene.render.height, 0, 0, 0);
    oidnSetFilterImage(filter, "output", colorBuffer, OIDN_FORMAT_FLOAT3, theScene.render.width, theScene.render.height, 0, 0, 0);
    oidnCommitFilter(filter);

    // Fill input image buffers
    printf("Filling buffers\n");
    CERR(cudaMemcpy(oidnGetBufferData(colorBuffer), prepColors, size * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CERR(cudaMemcpy(oidnGetBufferData(normalBuffer), prepNormals, size * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CERR(cudaMemcpy(oidnGetBufferData(albedoBuffer), prepAlbedos, size * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    CERR(cudaFree(prepColors));
    CERR(cudaFree(prepNormals));
    CERR(cudaFree(prepAlbedos));

    // Filter the beauty image
    printf("Executing denoiser\n");
    oidnExecuteFilter(filter);
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
        printf("Error: %s\n", errorMessage);

    // Convert colors
    printf("Saving colors.\n");
    float *filteredColors;
    CERR(cudaMalloc(&filteredColors, size * 3 * sizeof(float)));
    CERR(cudaMemcpy(filteredColors, oidnGetBufferData(colorBuffer), size * 3 * sizeof(float), cudaMemcpyHostToDevice));

    printf("Denoiser cleanup.\n");
    oidnReleaseBuffer(colorBuffer);
    oidnReleaseBuffer(normalBuffer);
    oidnReleaseBuffer(albedoBuffer);
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);

    printf("Converting colors.\n");
    ConvertColors<<<convBlocks, convThreads>>>(filteredColors, converted, size, theScene.camera.sRGB);
    CLERR();
    CERR(cudaDeviceSynchronize());
    printf("Color conversion finished.\n");

    // Bring results to host
    CERR(cudaMemcpy(image.pixels, converted, sizeof(Color24) * size, cudaMemcpyDeviceToHost));
    CERR(cudaFree(converted));
    image.passes++;
    //*/

    // We are done
    rendering = false;
    printf("Render complete!\n");
}