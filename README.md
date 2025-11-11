# CUDA Raytracing

A ray tracing program that uses CUDA to render on the GPU.

Loosely based on Prof. Cem Yuksel's *CS 6620: Rendering with Raytracing* course.

## File Structure

The main directory should only include `main.cpp`.

### `lib` directory

This contains code related to the OpenGL viewport and parsing XML which I do not care about implementing
myself at this time.

### `math` directory

Contains header and source files related to the various math functions. We use CUDA's float3 as a
stand-in for both Vec3f and Color, and this directory contains all the math used on these objects.

### `scene` directory

Contains information and structs for the scene, objects, and materials

### `renderer` directory

Contains information about the renderer.

### `scenes` directory

Contains the scene files.

### `models` directory

Contains OBJ files.

### `textures` directory

Contains PNG files as textures.