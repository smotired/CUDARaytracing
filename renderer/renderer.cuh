/// The actual renderer logic
#pragma once
#include <string>

#include "../lib/lodepng.cuh"
#include "../math/float3.cuh"
#include "../math/color.cuh"
#include "../scene/structs.cuh"
#include "../scene/objects.cuh"

// Program macros
#define RAY_THREADS_PER_BLOCK_X 16

// Rendering macros
#define BIAS 0.0001f;
#define BOUNCES 8;
// #define SAMPLE_MIN 4;
// #define SAMPLE_MAX 64;

//-------------------------------------------------------------------

// Information about the completed image (probably host only?)

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

class Renderer {
    char const* sceneFile;
    RenderedImage image;

    bool rendering = false;

public:
    bool LoadScene(const char* filename);

    [[nodiscard]] const RenderedImage& GetImage() const { return image; }
    [[nodiscard]] std::string SceneFileName() const { return sceneFile; }

    [[nodiscard]] bool IsRendering() const { return rendering; }
    void BeginRendering();
    void StopRendering();

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
