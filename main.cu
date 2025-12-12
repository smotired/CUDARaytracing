#include <iostream>

#include "sequence.h"
#include "lib/viewport.cuh"
#include "math/float3.cuh"
#include "renderer/renderer.cuh"

int main () {
    //*
    Renderer renderer;
    renderer.LoadScene("scenes/final/frame0042.xml");
    ShowViewport(&renderer, false);
    //*/

    /*
    const Sequence s("scenes/final");
    s.DoRender();
    //*/

    return 0;
}