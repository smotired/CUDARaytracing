/// Header files for different types of objects
#pragma once
#include "structs.cuh"

// -------- Sphere

class Sphere : public Object {
public:
    bool IntersectRay(Ray &ray, int hitSide) const override;
    Box GetBoundBox() const override { return Box(Float3(-1, -1, -1), Float3(1, 1, 1)); };
    void ViewportDisplay( /*Material const* material*/ ) const override;
};