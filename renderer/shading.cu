#include <algorithm>

#include "renderer.cuh"
#include "trace.cuh"
#include "vector.cuh"
#include "../scene/scene.cuh"

__device__ void Material::Shade(const uint3 blockIdx, Ray const& ray) const {
    float3 v = asNorm(-ray.dir); // View direction
    const Hit &hit = ray.hit; // Reference to the hit so we can pass it to lights

    // Calculate color contribution from direct lighting
    color direct = BLACK;

    // Loop through lights in the light list
    for (int i = 0; i < theScene.lightCount; i++) {
        float3 l = F3_ZERO; // Direction to the light source
        const Light* light = theScene.lights + i;
        // Get intensity and direction
        color intensity = LIGHT_ILLUMINATE(light, hit, l);
        if (intensity == BLACK) continue;

        // Ambient lights
        if (LIGHT_ISAMBIENT(light)) {
            direct += diffuse * intensity;
        }

        // Non-ambient lights
        else {
            // Blinn shading: Compute half vector
            const float3 h = asNorm(l + v);

            // Compute cos(theta) (angle between light source and normal)
            // and cos(phi) (angle between normal vector and half vector)
            const float cos_theta = l % hit.n;
            const float cos_phi = h % hit.n;

            // Add colors
            const color diff = cos_theta >= 0 ? (intensity * diffuse * cos_theta) : BLACK; // clamp to 0
            const color spec = intensity * specular * powf(cos_phi, glossiness);

            direct += diff + spec;
        }
    }

    // Absorption -- On a front hit, use the ray's absorption from the previous medium. On a back hit, use the absorpt ion from the material we hit.
    const float distance = length(ray.hit.pos - ray.pos);
    const color relevantAbsorption = hit.front ? ray.absorption : absorption;
    // This isn't perfect and only works if all objects are enclosed entirely

    // Atomically the ray's contribution from direct lighting to the color. Eventually this will ONLY be the material's emission.
    const color contribution = direct * exp(relevantAbsorption * -distance * 0.5f) * ray.contribution; // No idea if 0.5f for absorption is right
    atomicAdd(&theScene.render.results[ray.pixel].x, contribution.x);
    atomicAdd(&theScene.render.results[ray.pixel].y, contribution.y);
    atomicAdd(&theScene.render.results[ray.pixel].z, contribution.z);


    // Trace indirect light
    if (!ray.CanBounce()) return;

    // Reflections -- must trace if either reflection or refraction is enabled, because of fresnel.
    color reflectionContribution = reflection;

    // Calculate refraction and add contribution from Fresnel
    if (refraction != BLACK) {
        // Arrange material. Outer IOR shouldn't always be 1 but I don't know how to do a medium stack.
        const float outerIor = hit.front ? 1 : ior;
        const float innerIor = hit.front ? ior : 1;
        const float3 norm = hit.front ? hit.n : -hit.n; // Use inverted normal on back hits.

        // Calculate transmittance direction and if TIR occurs
        bool tir = false;
        const float3 t = transmit(v, norm, outerIor, innerIor, tir);

        // If TIR happens, this all gets reflected.
        if (tir) reflectionContribution += refraction;

        // Otherwise, calculate Fresnel coefficient and trace the transmitted ray.
        else {
            const float sqrt_f_0 = (outerIor - innerIor) / (outerIor + innerIor);
            const float f_0 = sqrt_f_0 * sqrt_f_0;
            const float cos_theta = v % norm;
            float f_theta = f_0 + (1 - f_0) * powf(1 - cos_theta, 5);
            f_theta = fmin(fmax(f_theta, 0.0f), 1.0f); // funky normals can mess this up

            // Add fresnel contribution to reflection ray
            reflectionContribution += refraction * f_theta;

            // Enqueue a refraction ray with absorption
            const color refractionContribution = refraction * (1 - f_theta);
            Ray refractionRay(hit.pos, t, ray.pixel, ray.bounce + 1, ray.contribution * refractionContribution, absorption);
            rayQueue.Enqueue(blockIdx, refractionRay);
        }
    }

    // Trace the reflection ray
    if (reflectionContribution != BLACK) {
        const float3 r = reflect(v, hit.n);
        Ray reflectionRay(hit.pos, r, ray.pixel, ray.bounce + 1, ray.contribution * reflectionContribution);
        rayQueue.Enqueue(blockIdx, reflectionRay);
    }
}