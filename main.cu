#include <iostream>

#include "lib/viewport.cuh"
#include "math/float3.cuh"
#include "renderer/renderer.cuh"

int main () {
    Renderer renderer;
    renderer.LoadScene("scenes/proj13/box.xml");
    ShowViewport(&renderer, false);

    return 0;
}