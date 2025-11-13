#include "renderer.cuh"
#include "trace.cuh"
#include "vector.cuh"
#include "../scene/scene.cuh"

__device__ void Material::Shade(Ray const& ray) const {
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
            float3 h = asNorm(l + v);

            // Compute cos(theta) (angle between light source and normal)
            // and cos(phi) (angle between normal vector and half vector)
            const float cos_theta = l % hit.n;
            const float cos_phi = h % hit.n;

            // Add colors
            color diff = cos_theta >= 0 ? (intensity * diffuse * cos_theta) : BLACK; // clamp to 0
            color spec = intensity * specular * powf(cos_phi, glossiness);

            direct += diff + spec;
        }
    }

    // Absorption -- On a front hit, use the ray's absorption from the previous medium. On a back hit, use the absorption from the material we hit.
    const float distance = length(ray.hit.pos - ray.pos);
    const color mediumAbsorption = hit.front ? ray.absorption : absorption;
    direct *= exp(mediumAbsorption * -distance);
    // This isn't perfect and only works if all objects are enclosed entirely

    // Add the ray's contribution from direct lighting to the color. Eventually this will ONLY be the material's emission.
    theScene.render.results[ray.pixel] += direct * ray.contribution;

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
}