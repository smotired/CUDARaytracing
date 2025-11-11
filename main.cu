#include <iostream>
#include "math/float3.cuh"

int main () {
    std::cout << "Hello World!" << std::endl;

    float3 a = make_float3(1.0f, 2.0f, 3.0f);
    float3 b = make_float3(1.0f, 2.0f, 3.0f);
    float3 c = make_float3(1.0f, 2.0f, 3.0f);

    std::cout << length(a % b * c) << std::endl;

    return 0;
}