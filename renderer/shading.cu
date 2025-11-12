#include "renderer.cuh"
#include "../scene/scene.cuh"

__device__ color Material::Shade(Ray const& ray) const {
    color total = BLACK;
    float3 l; // Direction to some light source
    float3 v = asNorm(ray.dir); // View direction
    const Hit &hit = ray.hit; // Reference to the hit so we can pass it to lights

    // Loop through lights in the light list
    for (int i = 0; i < theScene.lightCount; i++) {
        const Light* light = theScene.lights + i;

        // Ambient lights
        if (LIGHT_ISAMBIENT(light)) {
            total += diffuse * LIGHT_ILLUMINATE(light, hit, l);
        }

        // Non-ambient lights
        else {
            // Get intensity and direction
            color intensity = LIGHT_ILLUMINATE(light, hit, l);

            // Blinn shading: Compute half vector
            float3 h = asNorm(l + v);

            // Compute cos(theta) (angle between light source and normal)
            // and cos(phi) (angle between normal vector and half vector)
            float cos_theta = l % hit.n;
            float cos_phi = h % hit.n;

            // Add colors if the surface is lit
            if (cos_theta > 0) {
                color thisDiffuse = intensity * diffuse * cos_theta;
                color thisSpecular = intensity * specular * powf(cos_phi, glossiness);

                total += thisDiffuse + thisSpecular;
            }
        }
    }

    return total;
}