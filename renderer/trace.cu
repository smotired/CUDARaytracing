#include "trace.cuh"
#include <curand_kernel.h>

constexpr float invertedSampleCount = 1.0f / static_cast<float>(SAMPLES);
constexpr float F_PI = static_cast<float>(M_PI);

__global__ void TracePrimaryRays(const int passId) {

    // Each thread is responsible for 1 pixel in the block, across each iteration.
    // Define coords of starting pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;
    const float3 pixelCoords = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // If the start pixel is already out of bounds, we can just quit.
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Set up random number generation
    curandStateXORWOW_t rng;
    curand_init(pI, passId, 0, &rng);

    for (int i = 0; i < SAMPLES; i++) {
        // Randomly offset start position by DOF
        const float dof_r = sqrtf(RandomFloat(&rng));
        const float dof_theta = RandomFloat(&rng) * 2 * F_PI;
        const float dof_x = dof_r * cosf(dof_theta);
        const float dof_y = dof_r * sinf(dof_theta);
        const float3 rayStartPos = theScene.camera.position + theScene.camera.dof * (dof_x * theScene.render.cX - dof_y * theScene.render.cY);

        // Randomly offset camera plane position for antialiasing
        const float aa_x = RandomFloat(&rng) - 0.5f;
        const float aa_y = RandomFloat(&rng) - 0.5f;
        const float3 rayEndPos = pixelCoords + theScene.render.pixelSize * (aa_x * theScene.render.cX - aa_y * theScene.render.cY);

        Ray ray(rayStartPos, rayEndPos - rayStartPos, pI, 0, WHITE * invertedSampleCount);
        TracePath(ray, theScene.render.results + pI, theScene.render.normals + pI, theScene.render.albedos + pI, &rng);
    }
}

// Trace a ray through the scene and return true if it hits anything
__device__ bool TraceThroughScene(Ray& ray, Hit& hit) {
    // Add some bias
    ray.pos += ray.dir * BIAS;
    const int hitSide = ray.IsPrimary() ? HIT_FRONT : HIT_FRONT_AND_BACK;

    // Loop through the object list
    bool hitAnything = false;
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;

        if (HAS_OBJ(node->object)) {
            // Check for intersection with its bounding box
            float boxZ = BIGFLOAT;
            const bool hitBox = node->boundingBox.IntersectRay(ray, boxZ);

            if (hitBox && boxZ < hit.z) {
                // Check for intersection with the actual object, and transform hit
                node->ToLocal(ray);
                if (OBJ_INTERSECT(node->object, ray, hit, hitSide)) {
                    hitAnything = true;
                    node->FromLocal(hit);
                }
                node->FromLocal(ray);
            }
        }
    }

    // Loop through the light list too
    for (int i = 0; i < theScene.lightCount; i++) {
        Light* thisLight = theScene.lights + i;
        if (LIGHT_INTERSECT(thisLight, ray, hit)) {
            hitAnything = true;
            hit.light = thisLight;
        }
    }

    return hitAnything;
}

__device__ void TracePath(Ray const& origin, color* target, float3* normal, color* albedo, curandStateXORWOW_t *rng) {
    Ray ray = origin;
    while (ray.bounce < BOUNCES) {
        const float3 v = -asNorm(ray.dir);

        // Initialize a hit and trace the ray through the scene
        Hit hit;
        const bool hitAnything = TraceThroughScene(ray, hit);
        if (!hitAnything) {
            *target += ray.contribution * theScene.env->EvalEnvironment(ray.dir);
            if (ray.bounce == 0) {
                *normal += ray.contribution * asNorm(ray.dir);
                *albedo += ray.contribution * theScene.env->EvalEnvironment(ray.dir);
            }
            break;
        }

        // If we hit a light, just end at the light's radiance.
        if (hit.hitLight) {
            Light* light = hit.light;
            const color rad = LIGHT_RADIANCE(light);
            *target += ray.contribution * rad;
            if (ray.bounce == 0) {
                *normal += ray.contribution * hit.n;
                *albedo += ray.contribution * rad;
            }
            break;
        }

        // Pick a random light, and get the material from the hit
        const int lightId = static_cast<int>(RandomFloat(rng) * static_cast<float>(theScene.lightCount));
        const Light* light = theScene.lights + lightId;
        const Material* mtl = hit.node->material;

        // For primary rays, add normal/albedo colors.
        if (ray.bounce == 0) {
            *normal += ray.contribution * hit.n;
            *albedo += ray.contribution * hit.Eval(mtl->diffuse) + hit.Eval(mtl->emission);
        }

        // Add emission color
        *target += ray.contribution * hit.Eval(mtl->emission);

        // Generate a sample for the light
        float3 lDir;
        SampleInfo lInfo;
        LIGHT_GENSAMPLE(light, v, hit, lDir, rng, lInfo);

        // Generate probability from BRDF
        SampleInfo brdf;
        mtl->GetSampleInfo(v, hit, lDir, brdf);

        // Trace a shadow ray, and add estimation if not occluded.
        ShadowRay shadowRay(hit.pos, lDir);
        if (!TraceShadowRay(shadowRay, hit.n, lInfo.dist)) {
            const float probDenom = lInfo.prob * lInfo.prob + brdf.prob * brdf.prob;
            if (probDenom > F_EPS) {
                const float powerOverProb = lInfo.prob / probDenom; // for efficiency we do not square numerator since we would just divide by it later
                *target += ray.contribution * lInfo.mult * brdf.mult * powerOverProb;
            }
        }

        // Set up the next ray and recurse if it doesn't die
        float3 nDir;
        SampleInfo nInfo;
        if (!mtl->GenerateSample(v, hit, nDir, rng, nInfo))
            break;

        ray.pos = hit.pos;
        ray.dir = nDir;
        if (nInfo.prob <= F_EPS)
            break;

        ray.contribution *= nInfo.mult * (1.0f / nInfo.prob);
    }
}

__device__ bool TraceShadowRay(ShadowRay& ray, const float3 n, const float tMax, const int hitSide) {
    // Add some bias
    ray.pos += ray.dir * BIAS + n * BIAS;

    // Loop through the object list
    unsigned int skip = 0;
    for (int i = 0; i < theScene.nodeCount; i++) {
        Node* node = theScene.nodes + i;

        // If we missed a parent's bounding box, we can skip all descendants
        // Unfortunately this trick actually slows it down a lot for sampling rays.
        if (skip > 0) {
            skip--;
            skip += node->childCount;
            continue;
        }

        if (HAS_OBJ(node->object)) {
            float dist;
            const bool hitBox = node->boundingBox.IntersectShadowRay(ray, dist, tMax);

            // If we don't collide with parent bounding box, we don't collide with any child either.
            if (!hitBox) {
                skip = node->childCount;
                continue;
            }

            // Trace a ray
            node->ToLocal(ray);
            if (OBJ_INTSHADOW(node->object, ray, tMax, hitSide))
                return true;
            node->FromLocal(ray);
        }
    }

    return false;
}
