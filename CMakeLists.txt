cmake_minimum_required(VERSION 4.0)
project(cuda_raytracing CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_raytracing main.cu
        math/float3.cuh
)

set_target_properties(cuda_raytracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
