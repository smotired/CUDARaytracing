#include <iostream>

#include "lib/viewport.cuh"
#include "math/float3.cuh"
#include "renderer/renderer.cuh"

int main () {
    Renderer renderer;
    renderer.LoadScene("scenes/proj4/box.xml");
    ShowViewport(&renderer, false);

    return 0;
}