cmake_minimum_required(VERSION 4.0)
project(cuda_raytracing CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_raytracing main.cu
        math/float3.cuh
        math/color.cuh
        math/matrix.cuh
        scene/structs.cuh
        scene/objects.cu
        scene/objects.cuh
        lib/xmlload.cu
        lib/xmlload.cuh
        lib/viewport.cu
        lib/viewport.cuh
        lib/tinyxml2.cu
        lib/tinyxml2.cuh
        lib/lodepng.cu
        lib/lodepng.cuh
        renderer/renderer.cu
        renderer/renderer.cuh
        math/color.cu
        lib/lodepng.cuh
)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
)

set_target_properties(cuda_raytracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
