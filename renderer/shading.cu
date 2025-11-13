#include "renderer.cuh"
#include "trace.cuh"
#include "vector.cuh"
#include "../scene/scene.cuh"

__device__ color Material::Shade(Ray const& ray) const {
    color total = BLACK;
    float3 v = asNorm(-ray.dir); // View direction
    const Hit &hit = ray.hit; // Reference to the hit so we can pass it to lights

    // Loop through lights in the light list
    for (int i = 0; i < theScene.lightCount; i++) {
        float3 l = F3_ZERO; // Direction to the light source
        const Light* light = theScene.lights + i;
        // Get intensity and direction
        color intensity = LIGHT_ILLUMINATE(light, hit, l);
        DEBUG_PRINT("Light %d intensity: %.2f,%.2f,%.2f, direction: %.2f,%.2f,%.2f\n", i, intensity.x, intensity.y, intensity.z, l.x, l.y, l.z);
        if (intensity == BLACK) continue;

        // Ambient lights
        if (LIGHT_ISAMBIENT(light)) {
            color amb = diffuse * intensity;
            DEBUG_PRINT("Light %d Contribution: %.2f,%.2f,%.2f (ambient)\n", i, amb.x, amb.y, amb.z);
            total += amb;
            DEBUG_PRINT("Current total: %.2f,%.2f,%.2f\n", total.x, total.y, total.x);
        }

        // Non-ambient lights
        else {
            // Blinn shading: Compute half vector
            float3 h = asNorm(l + v);
            DEBUG_PRINT("v: %.2f,%.2f,%.2f\n", v.x, v.y, v.z);
            DEBUG_PRINT("h: %.2f,%.2f,%.2f\n", h.x, h.y, h.z);

            // Compute cos(theta) (angle between light source and normal)
            // and cos(phi) (angle between normal vector and half vector)
            const float cos_theta = l % hit.n;
            const float cos_phi = h % hit.n;
            DEBUG_PRINT("cos_theta: %.2f, cos_phi:%.2f\n", cos_theta, cos_phi);

            // Add colors
            color diff = cos_theta >= 0 ? (intensity * diffuse * cos_theta) : BLACK; // clamp to 0
            color spec = intensity * specular * powf(cos_phi, glossiness);
            DEBUG_PRINT("Light %d Contribution: %.2f,%.2f,%.2f (diffuse), %.2f,%.2f,%.2f (specular)\n", i, diff.x, diff.y, diff.z, spec.x, spec.y, spec.z);

            total += diff + spec;
            DEBUG_PRINT("Current total: %.2f,%.2f,%.2f\n", total.x, total.y, total.x);
        }
    }

    // Reflections -- must do if either reflection or refraction is enabled, because of fresnel.
    color reflected = BLACK;
    /*
    if (reflection != BLACK || refraction != BLACK) {
        // Trace a ray in the viewing direction.
        float3 r = reflect(v, hit.n);
        Ray reflectionRay(hit.pos, r, ray.pixel);
        if (TraceRay(reflectionRay)) reflected = reflectionRay.hit.node->material->Shade(reflectionRay);
        else reflected = color(0, 0, 0.1);

        total += reflection * reflected;
    }
    */

    return total;
}