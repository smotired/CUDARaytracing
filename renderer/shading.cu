#include <algorithm>

#include "renderer.cuh"
#include "trace.cuh"
#include "vector.cuh"
#include "../scene/scene.cuh"

constexpr float OVERPI = 1.0f / M_PI;
constexpr float F_PI = static_cast<float>(M_PI);

__device__ bool Material::GenerateSample(float3 const& v, Hit const &hit, float3 &dir, curandStateXORWOW_t *rng, SampleInfo &info) const {
    // Calculate colors at this point
    const color kD = hit.Eval(diffuse);
    const color kS = hit.Eval(specular);
    const color kT = hit.Eval(refraction);

    // Calculate probabilities of each bounce type from average colors of material properties
	constexpr float oneThird = 1.0f / 3.0f;
	float pD = (kD.x + kD.y + kD.z) * oneThird;
	float pS = (kS.x + kS.y + kS.z) * oneThird;
	float pT = (kT.x + kT.y + kT.z) * oneThird;

	// Normalize probabilities to top out below 1
	if (pD + pS + pT > MAX_SURVIVAL) {
		const float normalization = MAX_SURVIVAL / (pD + pS + pT);
		pD *= normalization;
		pS *= normalization;
		pT *= normalization;
	}

	// Calculate absorption on a back hit
	color kA = WHITE;
	if (!hit.front)
		kA = exp(absorption * -hit.z);
	// multiply all mult by kA before returning

	// Decide how the photon should bounce
	const float p = RandomFloat(rng);

	if (p < pD) {
		// Bounce diffusely. Pick a random direction from the hemisphere
		const float x = RandomFloat(rng);
		const float phi = RandomFloat(rng) * 2 * F_PI;
		const float cos_theta = 1 - x;
		const float sin_theta = sqrtf(1 - cos_theta * cos_theta);

		// Set up the direction
		float3 u, v;
		orthonormals(hit.n, u, v);
		dir = hit.n * cos_theta + u * sin_theta * cosf(phi) + v * sin_theta * sinf(phi);

		// Set up the Info struct
		info.prob = pD * OVERPI;
		info.mult = kA * kD * OVERPI * (hit.n % dir);

		return true;
	}

	// Generate a glossy normal for reflections or refractions.
	const float3 gln = glossyNormal(hit.n, glossiness, rng);

	if (p < pD + pS) {
		// Bounce specularly. Reflect off the random normal.
		dir = reflect(v, gln);
		info.prob = pS * (glossiness + 1) * 0.5f * OVERPI * powf(gln % hit.n, glossiness + 1);
		info.mult = kA * kS * (glossiness + 2) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);

		return true;
	}

	if (p < pD + pS + pT) {
		// Transmit into the material
		// Pick eta values and normal based on front/back hit
		const float eta_i = hit.front ? 1 : ior;
		const float eta_o = hit.front ? ior : 1;
		const float3 h = hit.front ? gln : -gln;

		// Calculate transmittance direction and if TIR occurs
		bool tir = false;
		dir = transmit(v, h, eta_i, eta_o, tir);

		// Calculate fresnel
		const float f_0 = powf((eta_i - eta_o) / (eta_i + eta_o), 2);
		const float cos_theta = v % h; // invert normal on back hits
		float f_theta = f_0 + (1 - f_0) * powf(1 - cos_theta, 5);
		f_theta = fmin(fmax(f_theta, 0.0f), 1.0f); // Funky normals can sometimes mess with this

		// If TIR does not happen, decide between fresnel or not
		if (!tir) {
			if (RandomFloat(rng) >= f_theta) {
				info.prob = pT * (1 - f_theta); // * (glossiness + 1) * 0.5f * OVERPI * powf(gln % hit.n, glossiness + 1);
				info.mult = kA * kT * (1 - f_theta); // * (glossiness + 2) * 0.5f * OVERPI * powf(gln % hit.n, glossiness);
			} else {
				dir = reflect(v, h);
				info.prob = pT * f_theta; // * (glossiness + 1) * 0.5f * OVERPI * powf(gln % hit.n, glossiness + 1);
				info.mult = kA * kT * f_theta; // * (glossiness + 2) * 0.5f * OVERPI * powf(gln % hit.n, glossiness);
			}
		}

		// Otherwise bounce specularly
		else {
			info.prob = pT;
			info.mult = kA * kT;
		}

		return true;
	}

	// No reflectance
	info.prob = 1 - pT - pS - pD;
	info.mult = BLACK;
	return false;
}


__device__ void Material::GetSampleInfo(float3 const& v, Hit const &hit, float3 const &dir, SampleInfo &info) const {
	// Calculate required colors at this point
	const color kD = hit.Eval(diffuse);
	const color kS = hit.Eval(specular);
	const color kT = hit.Eval(refraction);

	// Calculate probabilities of each bounce type from average colors of material properties
	constexpr float oneThird = 1.0f / 3.0f;
	float pD = (kD.x + kD.y + kD.z) * oneThird;
	float pS = (kS.x + kS.y + kS.z) * oneThird;
	float pT = (kT.x + kT.y + kT.z) * oneThird;

	// Normalize probabilities to top out below 1
	if (pD + pS + pT > MAX_SURVIVAL) {
		const float normalization = MAX_SURVIVAL / (pD + pS + pT);
		pD *= normalization;
		pS *= normalization;
		pT *= normalization;
	}

	if (hit.n % dir >= 0) {
		// Find probability of sampling in this direction on a diffuse bounce
		// Currently diffuse samples are generated uniformly so this is just pD * 1/pi.
		info.prob += pD * (hit.n % dir > 0 ? OVERPI : 0);
		info.mult += kD * OVERPI * (hit.n % dir);

		// Probability of sampling this direction on a specular bounce
		// = probability of choosing the half-vector as a glossy normal
		const float3 rn = asNorm(v + dir);
		info.prob += pS * (glossiness + 1) * 0.5f * OVERPI * powf(hit.n % rn, glossiness + 1);
		info.mult += kS * (glossiness + 2) * 0.125f * OVERPI * powf(hit.n % rn, glossiness);
	}

	// should include refraction but idk how lmao
}

// Legacy
__device__ void Material::Shade(Ray const& ray, Hit const& hit) const {
    float3 v = asNorm(-ray.dir); // View direction

    // Calculate color contribution from direct lighting
    color direct = BLACK;

    // Calculate hit colors
    const color kD = hit.Eval(diffuse);
    const color kS = hit.Eval(specular);
    const color kR = hit.Eval(reflection);
    const color kT = hit.Eval(refraction);

    // Loop through lights in the light list
    for (int i = 0; i < theScene.lightCount; i++) {
        float3 l = F3_ZERO; // Direction to the light source
        const Light* light = theScene.lights + i;
        // Get intensity and direction
        color intensity = LIGHT_ILLUMINATE(light, hit, l);
        if (intensity == BLACK) continue;

        // Ambient lights
        if (LIGHT_ISAMBIENT(light)) {
            direct += kD * intensity;
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
            const color diff = cos_theta >= 0 ? (intensity * kD * cos_theta) : BLACK; // clamp to 0
            const color spec = intensity * kS * powf(cos_phi, glossiness);

            direct += diff + spec;
        }
    }

    // Absorption -- On a front hit, use the ray's absorption from the previous medium. On a back hit, use the absorpt ion from the material we hit.
    const float distance = length(hit.pos - ray.pos);
    const color relevantAbsorption = hit.front ? ray.absorption : absorption;
    // This isn't perfect and only works if all objects are enclosed entirely

    // Atomically the ray's contribution from direct lighting to the color. Eventually this will ONLY be the material's emission.
    const color contribution = direct * exp(relevantAbsorption * -distance * 0.5f) * ray.contribution; // No idea if 0.5f for absorption is right
    atomicAdd(&theScene.render.results[ray.pixel].x, contribution.x);
    atomicAdd(&theScene.render.results[ray.pixel].y, contribution.y);
    atomicAdd(&theScene.render.results[ray.pixel].z, contribution.z);

    // Everything after this requires a bounce
    if (!ray.CanBounce()) return;

    // Reflections -- must trace if either reflection or refraction is enabled, because of fresnel.
    color reflectionContribution = kR;

    // Calculate refraction and add contribution from Fresnel
    if (kT != BLACK) {
        // Arrange material. Outer IOR shouldn't always be 1 but I don't know how to do a medium stack.
        const float outerIor = hit.front ? 1 : ior;
        const float innerIor = hit.front ? ior : 1;
        const float3 norm = hit.front ? hit.n : -hit.n; // Use inverted normal on back hits.

        // Calculate transmittance direction and if TIR occurs
        bool tir = false;
        const float3 t = transmit(v, norm, outerIor, innerIor, tir);

        // If TIR happens, this all gets reflected.
        if (tir) reflectionContribution += kT;

        // Otherwise, calculate Fresnel coefficient and trace the transmitted ray.
        else {
            const float sqrt_f_0 = (outerIor - innerIor) / (outerIor + innerIor);
            const float f_0 = sqrt_f_0 * sqrt_f_0;
            const float cos_theta = v % norm;
            float f_theta = f_0 + (1 - f_0) * powf(1 - cos_theta, 5);
            f_theta = fmin(fmax(f_theta, 0.0f), 1.0f); // funky normals can mess this up

            // Add fresnel contribution to reflection ray
            reflectionContribution += kT * f_theta;

            // Enqueue a refraction ray with absorption
            const color refractionContribution = kT * (1 - f_theta);
            Ray refractionRay(hit.pos, t, ray.pixel, ray.bounce - 1, ray.contribution * refractionContribution, absorption);
            TraceRay(refractionRay);
        }
    }

    // Trace the reflection ray
    if (reflectionContribution != BLACK) {
        const float3 r = reflect(v, hit.n);
        Ray reflectionRay(hit.pos, r, ray.pixel, ray.bounce - 1, ray.contribution * reflectionContribution);
        TraceRay(reflectionRay);
    }
}
