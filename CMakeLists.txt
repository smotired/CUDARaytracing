cmake_minimum_required(VERSION 4.0)
project(cuda_raytracing CUDA)

set(CMAKE_CUDA_STANDARD 20)

# --------------------------------
# Sections
# --------------------------------
set(SRC_MAIN main.cu)
set(SRC_IO lib/lodepng.cu lib/tinyxml2.cu lib/viewport.cu lib/xmlload.cu)
set(SRC_MATH math/color.cu)
set(SRC_RDR renderer/renderer.cu renderer/trace.cu)
set(SRC_SCN scene/objects.cu)

# --------------------------------
# Final executable
# --------------------------------
add_executable(${PROJECT_NAME} ${SRC_MAIN} ${SRC_IO} ${SRC_MATH} ${SRC_RDR} ${SRC_SCN})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE lib math renderer scene)

# --------------------------------
# Link FreeGLUT
# --------------------------------
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
)

# Ensure device functions are callable from anywhere
set_target_properties(cuda_raytracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
