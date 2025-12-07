/// sequence.h -- Sam Hill
/// Created for CS 5620 - Rendering with Ray Tracing
/// Fall 2025 semester, University of Utah
///
/// You will likely need to adjust the import path to your renderer, and trigger the right class below.
/// You will also need to adjust your renderer's implementation slightly. This assumes BeginRendering
/// takes in a boolean parameter called `wait`, and if that is set to true, the method does not exit
/// until the render is fully completed (i.e. joining the thread instead of detaching it).
///
/// To use this file, include this header in your project. Instantiate this class and run it like this:
/// ```
/// const Sequence s("path/to/scene/folder");
/// s.DoRender();
/// ```
/// where `path/to/scenes/folder` is the path from the working directory to the directory containing your
/// scene files. All xml files in this folder will be rendered in an arbitrary order and saved to the
/// `output` folder, where a scene at `scenes/example/frame1.xml` would be saved as `output/frame.png`.
///
/// It's recommended to name your scene files alphabetically because they will be rendered in an arbirtrary
/// order, so naming them alphabetically makes it easier to convert them to a gif or video using online tools.
///
/// This could be sped up a bit by adding a Reset method to the renderer so that a new one doesn't have to be instantiated
/// every time. Be careful if you're aware of any memory leaks, as they will be much more severe with a lot of frames.

#pragma once
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

#include "renderer.cuh"

class Sequence {
    size_t sceneCount = 0;
    std::string directory;
    std::string *scenes = nullptr;

    static void ReportTime(std::chrono::high_resolution_clock::time_point const start, std::chrono::high_resolution_clock::time_point const end, char const* name) {
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        const int h = static_cast<int>(duration / 3600000000);
        const int m = static_cast<int>((duration % 3600000000) / 60000000);
        const int s = static_cast<int>((duration % 60000000) / 1000000);
        const int mus = static_cast<int>(duration % 1000000);
        printf("\n%s render time is %d:%02d:%02d.%06d.\n", name, h, m, s, mus);
    }

public:
    explicit Sequence(const std::string& path) {
        directory = path;

        // Count files
        for (const auto & entry : fs::directory_iterator(directory))
            sceneCount++;

        // Add to list
        scenes = new std::string[sceneCount];
        size_t i = 0;
        for (const auto & entry : fs::directory_iterator(directory)) {
            scenes[i++] = entry.path().string();
        }
    }

    void DoRender() const {
        // Initialize output dierctory
        fs::create_directory("output");
        for (const auto& entry : std::filesystem::directory_iterator("output")) {
            std::filesystem::remove_all(entry.path());
        }

        // Start time of rendering
        const auto start = std::chrono::high_resolution_clock::now();

        // Render each frame from scratch.
        for (size_t i = 0; i < sceneCount; i++) {
            printf("\n================================================================\nRENDERING SCENE %lu / %lu: %s\n================================================================\n", i + 1, sceneCount, scenes[i].c_str());
            const auto frame= std::chrono::high_resolution_clock::now();

            // Do the render
            Renderer renderer;
            renderer.LoadScene(scenes[i].c_str());
            renderer.BeginRendering(true);

            // Save the image
            const std::string scene_name = scenes[i].substr(directory.size() + 1, scenes[i].size() - directory.size() - 5);
            const std::string outputFilename = std::string("output/") + scene_name + ".png";
            printf("Saving image to %s\n", outputFilename.c_str());
            const RenderedImage &image = renderer.GetImage();
            const bool success = image.SaveImage(outputFilename.c_str());
            printf("Image save successful: %s\n", success ? "true" : "false");

            // Report time
            const auto end = std::chrono::high_resolution_clock::now();
            ReportTime(frame, end, "Frame");
        }

        printf("\n================================================================\nALL SCENES RENDERED\n================================================================\n");
        const auto end = std::chrono::high_resolution_clock::now();
        ReportTime(start, end, "Total");
    }
};