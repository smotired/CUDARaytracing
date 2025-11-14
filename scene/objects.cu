#include "scene.cuh"
#include "objects.cuh"

#define EPS 0.000001f
#define OVERPI 0.318309f // 1 / pi

__device__ bool Sphere::IntersectRay(Ray const &ray, Hit& hit, const int hitSide = HIT_FRONT_AND_BACK) const {
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
			if (front >= hit.z) return false;

			// Set the value for t into the z buffer.
			hit.z = front;

			// Calculate hit position and normal vector, in object space.
			hit.pos = ray.pos + front * ray.dir;
			hit.n = hit.pos; // unit sphere is a special case.
			hit.front = true;

			return true;
		}
	}

	// If we should calculate back hits
	if (hitSide & HIT_BACK) {
		// Same logic as before.
		if (back < 0) return false;
		if (back >= hit.z) return false;
		hit.z = back;
		hit.pos = ray.pos + back * ray.dir;
		hit.n = hit.pos;
		hit.front = false;

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

__device__ bool Plane::IntersectRay(Ray const& ray, Hit& hit, const int hitSide = HIT_FRONT_AND_BACK) const {
	// Assume that if HIT_NONE is provided we should always return false.
	if (hitSide == HIT_NONE) return false;

	// If ray's z direction is 0, the ray is parallel to the plane.
	if (ray.dir.z == 0) return false;

	// Find the single point at which the ray intersects the plane
	const float t = -ray.pos.z / ray.dir.z; // kinda crazy how simple this gets in local space

	// Check that it's visible and the closest hit so far
	if (t < 0 || t >= hit.z)
		return false;

	// Check that it intersects the correct side
	if (!(ray.dir.z < 0 && (hitSide & HIT_FRONT)) && !(ray.dir.z > 0 && (hitSide & HIT_BACK)))
		return false;

	// Check that it's within the bounds of the plane
	const float3 hitPos = ray.pos + t * ray.dir;
	if (std::abs(hitPos.x) > 1 || std::abs(hitPos.y) > 1)
		return false;

	// Set up the hit info
	hit.z = t;
	hit.pos = hitPos;
	hit.n = F3_FORWARD; // the plane is z up
	hit.front = ray.dir.z < 0;
	return true;
}

__device__ bool Plane::IntersectShadowRay(const ShadowRay &ray, const float tMax = BIGFLOAT, const int hitSide = HIT_FRONT_AND_BACK) const {
	// Assume that if HIT_NONE is provided we should always return false.
	if (hitSide == HIT_NONE) return false;

	// If ray's z direction is 0, the ray is parallel to the plane.
	if (ray.dir.z == 0) return false;

	// Find the single point at which the ray intersects the plane
	const float t = -ray.pos.z / ray.dir.z; // kinda crazy how simple this gets in local space

	// Check that it's visible and the closest hit so far
	if (t < 0 || t >= tMax)
		return false;

	// Check that it intersects the correct side
	if (!(ray.dir.z < 0 && (hitSide & HIT_FRONT)) && !(ray.dir.z > 0 && (hitSide & HIT_BACK)))
		return false;

	// Check that it's within the bounds of the plane
	const float3 hitPos = ray.pos + t * ray.dir;
	if (std::abs(hitPos.x) > 1 || std::abs(hitPos.y) > 1)
		return false;
}

__device__ bool MeshObject::IntersectRay(Ray const &ray, Hit &hit, const int hitSide) const {
	// Assume that if HIT_NONE is provided we should always return false.
	if (hitSide == HIT_NONE) return false;

	// Just check every single triangle
	bool hitAnyTriangle = false;
	for (int i = 0; i < nf; i++)
		if (IntersectTriangle(ray, hit, hitSide, i))
			hitAnyTriangle = true;
	return hitAnyTriangle;
}

__device__ bool MeshObject::IntersectShadowRay(ShadowRay const &ray, const float tMax, const int hitSide) const {
	// Assume that if HIT_NONE is provided we should always return false.
	if (hitSide == HIT_NONE) return false;

	// Just check every single triangle
	for (int i = 0; i < nf; i++)
		if (IntersectShadowTriangle(ray, tMax, hitSide, i))
			return true;
	return false;
}

__device__ bool MeshObject::IntersectTriangle(Ray const& ray, Hit& hit, const int hitSide, const unsigned int faceID) const {
	// Uses the Moller-Trumbore method.
	// Get basic info about the triangle.
	const uint3 face = f[faceID];
	const float3 p0 = v[face.x];
	const float3 p1 = v[face.y];
	const float3 p2 = v[face.z];

	// Get edges and ensure ray intersects the face's plane
	const float3 edge1 = p1 - p0;
	const float3 edge2 = p2 - p0;
	const float3 rayXEdge2 = cross(ray.dir, edge2);
	const float determinant = edge1 % rayXEdge2;

	if (determinant > -EPS && determinant < EPS) return false;

	// Get first barycentric coordinate
	const float inverseDeterminant = 1 / determinant;
	const float3 offset = ray.pos - p0;
	const float b1 = inverseDeterminant * (offset % rayXEdge2);

	if (b1 <= EPS || b1 >= 1 + EPS) return false;

	// Get the second barycentric coordinate
	const float3 offsetXEdge1 = cross(offset, edge1);
	const float b2 = inverseDeterminant * (ray.dir % offsetXEdge1);

	if (b2 <= -EPS || (b1 + b2) >= 1 + EPS) return false;

	// At this point, the ray definitely intersects the triangle.
	// Find t, return if it's invalid.
	const float t = inverseDeterminant * (edge2 % offsetXEdge1);
	if (t <= EPS || t >= hit.z + EPS) return false;

	// Check that we're hitting the correct side of the triangle
	const float ndot = cross(edge1, edge2) % ray.dir;
	if (!(ndot < EPS && (hitSide & HIT_FRONT)) && !(ndot > -EPS && (hitSide & HIT_BACK))) return false;

	// Get remaining barycentric coordinate and interpolate normals.
	const float b0 = 1 - b2 - b1;
	const float3 barycentric(b0, b1, b2);
	const float3 gn = asNorm(cross(edge1, edge2));

	const float3 n = !HasNormals() ? gn : vn[fn[faceID].x] * barycentric.x + vn[fn[faceID].y] * barycentric.y + vn[fn[faceID].z] * barycentric.z;

	// Set up hit
	hit.z = t;
	hit.n = n;
	hit.pos = ray.pos + t * ray.dir;
	hit.front = ndot < 0;
	return true;
}

__device__ bool MeshObject::IntersectShadowTriangle(ShadowRay const& ray, const float tMax, const int hitSide, const unsigned int faceID) const {
	// Uses the Moller-Trumbore method.
	// Get basic info about the triangle.
	const uint3 face = f[faceID];
	const float3 p0 = v[face.x];
	const float3 p1 = v[face.y];
	const float3 p2 = v[face.z];

	// Get edges and ensure ray intersects the face's plane
	const float3 edge1 = p1 - p0;
	const float3 edge2 = p2 - p0;
	const float3 rayXEdge2 = cross(ray.dir, edge2);
	const float determinant = edge1 % rayXEdge2;

	if (determinant > -EPS && determinant < EPS) return false;

	// Get first barycentric coordinate
	const float inverseDeterminant = 1 / determinant;
	const float3 offset = ray.pos - p0;
	const float b1 = inverseDeterminant * (offset % rayXEdge2);

	if (b1 <= EPS || b1 >= 1 + EPS) return false;

	// Get the second barycentric coordinate
	const float3 offsetXEdge1 = cross(offset, edge1);
	const float b2 = inverseDeterminant * (ray.dir % offsetXEdge1);

	if (b2 <= -EPS || (b1 + b2) >= 1 + EPS) return false;

	// At this point, the ray definitely intersects the triangle.
	// Find t, return if it's invalid.
	const float t = inverseDeterminant * (edge2 % offsetXEdge1);
	if (t <= EPS || t >= tMax + EPS) return false;

	// Check that we're hitting the correct side of the triangle
	const float ndot = cross(edge1, edge2) % ray.dir;
	if (!(ndot < EPS && (hitSide & HIT_FRONT)) && !(ndot > -EPS && (hitSide & HIT_BACK))) return false;

	// This should cast a shadow
	return true;
}