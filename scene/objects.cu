#include "scene.cuh"
#include "objects.cuh"

#define EPS 0.000001f
#define OVERPI 0.318309f // 1 / pi

__device__ bool Sphere::IntersectRay(Ray &ray, const int hitSide = HIT_FRONT_AND_BACK) const {
	// Assume that if HIT_NONE is provided we should always return false.
	if (hitSide == HIT_NONE) return false;

	// Optimization: If the dot product of the direction and the position difference
	// is less than zero, then if t exists, t<0, so we can discard.
	// NOTE: This is only true for spheres.
	if (ray.dir % ray.pos >= 0)
		return false;

	// A: d^2                   d^2
	// B: 2d * (p - q)          2d * p
	// C: (p - q)^2 - r^2       p^2 - 1
	const float a = ray.dir % ray.dir;
	const float b = ray.dir % ray.pos * 2;
	const float c = ray.pos % ray.pos - 1;

	// There is a real solution to the quadratic formula if this is positive.
	// If this is negative, the solution(s) are negative and there is no intersection.
	const float determinant = b * b - 4 * a * c;
	if (determinant < 0) return false;

	// Calculate the solutions to the quadratic formula.
	const float sqrtDeterminant = sqrtf(determinant);
	const float front = (-b - sqrtDeterminant) / (2 * a);
	const float back = (-b + sqrtDeterminant) / (2 * a);

	// If we should calculate front hits
	if (hitSide & HIT_FRONT) {
		// Don't render if the front intersection is behind the camera, unless we should check for a back hit.
		if (front >= 0) {
			// If this is less close than the existing hit, return false (the back hit also will be)
			if (front >= ray.hit.z) return false;

			// Set the value for t into the z buffer.
			ray.hit.z = front;

			// Calculate hit position and normal vector, in object space.
			ray.hit.pos = ray.pos + front * ray.dir;
			ray.hit.n = ray.hit.pos; // unit sphere is a special case.
			ray.hit.front = true;

			return true;
		}
	}

	// If we should calculate back hits
	if (hitSide & HIT_BACK) {
		// Same logic as before.
		if (back < 0) return false;
		if (back >= ray.hit.z) return false;
		ray.hit.z = back;
		ray.hit.pos = ray.pos + back * ray.dir;
		ray.hit.n = ray.hit.pos;
		ray.hit.front = false;

		return true;
	}

	return false;
}

__device__ bool Sphere::IntersectShadowRay(const ShadowRay &ray, const float tMax = BIGFLOAT, const int hitSide = HIT_FRONT_AND_BACK) const {
	// Mostly the same logic as above, except that we don't care to calculate where exactly it hits.
	if (hitSide == HIT_NONE) return false;
	if (ray.dir % ray.pos >= 0)
		return false;
	const float a = ray.dir % ray.dir;
	const float b = ray.dir % ray.pos * 2;
	const float c = ray.pos % ray.pos - 1;
	const float determinant = b * b - 4 * a * c;
	if (determinant < 0) return false;
	// Definitely hits the sphere at some point

	// Calculate the solutions to the quadratic formula.
	const float sqrtDeterminant = sqrtf(determinant);
	const float front = (-b - sqrtDeterminant) / (2 * a);
	const float back = (-b + sqrtDeterminant) / (2 * a);

	// Calculate hits for other sides
	if (hitSide & HIT_FRONT && front >= 0 && front <= tMax) return true;
	if (hitSide & HIT_BACK && back >= 0 && back <= tMax) return true;
	return false;
}