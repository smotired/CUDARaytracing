/// Header files for different types of objects
#pragma once
#include "structs.cuh"

// -------- Sphere

class Sphere : public Object {
public:
    bool IntersectRay(Ray &ray, int hitSide) const override;
    Box GetBoundBox() const override { return Box(-F3_ONE, F3_ONE); };
    void ViewportDisplay( /*Material const* material*/ ) const override;
};