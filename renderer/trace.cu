#include "trace.cuh"

constexpr float invertedSampoles = 1.0f / static_cast<float>(SAMPLES);

__global__ void TracePrimaryRays() {
    // Each thread is responsible for 1 pixel in the block, across each iteration.
    // Define coords of starting pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x; // We only have 1 block right now
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pI = pY * theScene.render.width + pX;
    const float3 pixelCoords = theScene.render.topLeftPixel
        + theScene.render.pixelSize * (pX * theScene.render.cX - pY * theScene.render.cY);

    // If the start pixel is already out of bounds, we can just quit.
    if (pX >= theScene.render.width || pY >= theScene.render.height)
        return;

    // Initialize Z buffer
    theScene.render.zBuffer[pI] = BIGFLOAT;

    for (int i = 0; i < SAMPLES; i++) {
        Ray ray(theScene.camera.position, pixelCoords - theScene.camera.position, pI, 0, WHITE * invertedSampoles);
        TracePath(ray, theScene.render.results + pI);
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

__device__ void TracePath(Ray const& origin, color* target) {
    Ray ray = origin;
    while (ray.bounce < BOUNCES) {
        const float3 v = -asNorm(ray.dir);

        // Initialize a hit and trace the ray through the scene
        Hit hit;
        const bool hitAnything = TraceThroughScene(ray, hit);
        if (!hitAnything) {
            *target += ray.contribution * theScene.env->EvalEnvironment(ray.dir);
            break;
        }

        // If we hit a light, just end at the light's radiance.
        if (hit.hitLight) {
            Light* light = hit.light;
            *target += ray.contribution * LIGHT_RADIANCE(light);
            break;
        }

        // Update minimum Z on a primary ray

        // Pick a random light, and get the material from the hit
        const int lightId = static_cast<int>(theScene.rng->RandomFloat() * theScene.lightCount);
        const Light* light = theScene.lights + lightId;
        const Material* mtl = hit.node->material;


        // Generate a sample for the light
        float3 lDir;
        SampleInfo lInfo;
        LIGHT_GENSAMPLE(light, v, hit, lDir, lInfo);

        // Generate probability from BRDF
        SampleInfo brdf;
        mtl->GetSampleInfo(v, hit, lDir, brdf);

        // Trace a shadow ray, and add estimation if not occluded.
        ShadowRay shadowRay(hit.pos, lDir);
        if (!TraceShadowRay(shadowRay, hit.n, lInfo.dist)) {
            const float powerOverProb = lInfo.prob /  (lInfo.prob * lInfo.prob + brdf.prob * brdf.prob); // for efficiency we do not square numerator since we would just divide
            *target += ray.contribution * lInfo.mult * brdf.mult * powerOverProb;
        }

        // Set up the next ray and recurse if it doesn't die
        float3 nDir;
        SampleInfo nInfo;
        if (!mtl->GenerateSample(v, hit, nDir, nInfo))
            break;

        ray.pos = hit.pos;
        ray.dir = nDir;
        ray.contribution *= nInfo.mult * (1.0f / nInfo.prob);
    }
}

// Legacy
__device__ void TraceRay(Ray &ray, int hitSide) {
    // Add some bias
    ray.pos += ray.dir * BIAS;

    // Initialize a hit
    Hit hit;

    // Primary rays should only check front hits
    if (ray.IsPrimary() && hitSide & HIT_FRONT) hitSide = HIT_FRONT;

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

    // Shade, or add color from environment. Assume no hit means no absorption.
    if (hitAnything) hit.node->material->Shade(ray, hit);
    else theScene.render.results[ray.pixel] += ray.contribution * theScene.env->EvalEnvironment(ray.dir);

    // If this is a primary ray, update the Z buffer
    if (ray.IsPrimary() && hitAnything)
        theScene.render.zBuffer[ray.pixel] = fmin(theScene.render.zBuffer[ray.pixel], hit.z);
}

// Not legacy
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
