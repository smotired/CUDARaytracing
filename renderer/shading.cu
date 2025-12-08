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
		info.prob = pS * (glossiness + 1) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
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
				info.prob = pT * (1 - f_theta); // * (glossiness + 1) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
				info.mult = kA * kT * (1 - f_theta); // * (glossiness + 2) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
			} else {
				dir = reflect(v, h);
				info.prob = pT * f_theta; // * (glossiness + 1) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
				info.mult = kA * kT * f_theta; // * (glossiness + 2) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
			}
		}

		// Otherwise bounce specularly
		else {
			info.prob = pT; // * (glossiness + 1) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
			info.mult = kA * kT; // * (glossiness + 1) * 0.125f * OVERPI * powf(gln % hit.n, glossiness);
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