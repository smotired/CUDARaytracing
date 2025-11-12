/// The actual renderer logic
#pragma once
#include <string>

#include "../lib/lodepng.cuh"
#include "../math/float3.cuh"
#include "../math/color.cuh"
#include "../scene/scene.cuh"
#include "../scene/objects.cuh"

// Rendering macros
#define BIAS 0.0002f
#define BOUNCES 8
// #define SAMPLE_MIN 4
// #define SAMPLE_MAX 64

// Program macros
#define RAY_THREADS_PER_BLOCK_X 16
inline cudaError_t err = cudaSuccess; // Global variable to ensure these macros always work.
// Check a specific call for an error
#define CERR(fn) \
    err = fn; \
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err))
// Check the last error
#define CLERR() CERR(cudaGetLastError())
// Target pixels for debugging
 #define DEBUG_X (-1)
 #define DEBUG_Y (-1)
// #define DEBUG_X 651
// #define DEBUG_Y 150
#define DEBUG ((DEBUG_X | DEBUG_Y) >= 0)
#define DEBUG_PRINT(...) if (DEBUG) printf(__VA_ARGS__)
#define DEBUG_KERNEL(blocks, threads, kernel, ...) \
    if (!DEBUG) kernel<<<blocks, threads>>>(__VA_ARGS__); \
    else kernel<<<1, 1>>>(__VA_ARGS__)

//-------------------------------------------------------------------

// Information about the completed image -- host only

struct RenderedImage {
    unsigned int width; unsigned int height;
    Color24* pixels;
    float* zBuffer;
    uint8_t* zBufferImg;

    void Init(const int w, const int h) {
        width = w;
        height = h;
        pixels = new Color24[w * h];
        zBuffer = new float[w * h];
        zBufferImg = new uint8_t[w * h];
    }

    bool SaveImage(char const* filename) const {
        std::vector<Color24> converted(pixels, pixels+width*height);
        return lodepng::encode(filename, &converted[0].r, width, height, LCT_RGB, 8) == 0;
    }
    bool SaveZBufferImage(char const* filename) const { return lodepng::encode(filename, zBufferImg, width, height, LCT_RGB, 8) == 0; }
};

//-------------------------------------------------------------------

__managed__ extern Scene theScene;

/// <summary>
/// The renderer is in charge of actually rendering the scene
/// </summary>
class Renderer {
    // Name of the scene file
    char const* sceneFile;

    // The image that we rendered
    RenderedImage image;

    // If we are currently rendering
    bool rendering = false;
public:
    // Load the scene with this filename
    bool LoadScene(const char* filename);

    // Get a reference to the image file
    [[nodiscard]] const RenderedImage& GetImage() const { return image; }

    // Get the scene filename
    [[nodiscard]] std::string SceneFileName() const { return sceneFile; }

    [[nodiscard]] bool IsRendering() const { return rendering; }
    void BeginRendering();
    void StopRendering();

    // Convert the RenderImage's z buffer to a black and white image for display
    void ComputeZBufferImage() { ComputeImage<float,true>( image.zBuffer, image.zBufferImg, BIGFLOAT ); }
private:
    // Compute a black and white image
    template <typename T, bool invert>
    T ComputeImage(T const* in, uint8_t *out, T skipv) {
        size_t size = theScene.render.width * theScene.render.height;
        // Get minimum and maximum values
        T vmin = std::numeric_limits<T>::max(), vmax=T(0);
        for (int i = 0; i < size; i++) {
            if (in[i] == skipv) continue;
            if (in[i] < vmin) vmin = in[i];
            if (in[i] > vmax) vmax = in[i];
        }
        // Create image based on float difference
        for ( int i=0; i<size; i++ ) {
            if ( in[i] == skipv ) out[i] = 0;
            else {
                float f = float(in[i]-vmin)/float(vmax-vmin);
                if constexpr ( invert ) f = 1 - f;
                int c = int(f * 255);
                out[i] = c < 0 ? 0 : ( c > 255 ? 255 : c );
            }
        }
        return vmax;
    }
};
