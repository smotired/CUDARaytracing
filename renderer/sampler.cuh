#ifndef CUDA_RAYTRACING_SAMPLER_CUH
#define CUDA_RAYTRACING_SAMPLER_CUH

struct Ray;
struct Hit;

struct SampleInfo {
    // Estimator
    color mult;

    // Probability of generating this sample
    float prob;

    // Distance to the sample intersection point
    float dist;

    __device__ explicit SampleInfo(const color mult = BLACK, const float prob = 0.0f, const float dist = BIGFLOAT) : mult(mult), prob(prob), dist(dist) {}
};

#endif //CUDA_RAYTRACING_SAMPLER_CUH