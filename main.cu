#include <iostream>

#include "sequence.h"
#include "lib/viewport.cuh"
#include "math/float3.cuh"
#include "renderer/renderer.cuh"

int main () {
    Renderer renderer;
    renderer.LoadScene("scenes/proj13/box.xml");
    ShowViewport(&renderer, false);

    /*
    const Sequence s("scenes/testseq");
    s.DoRender();
    */

    return 0;
}