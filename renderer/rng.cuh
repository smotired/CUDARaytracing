#ifndef CUDA_RAYTRACING_RNG_CUH
#define CUDA_RAYTRACING_RNG_CUH
#include <functional>
#include <thread>

// Random Number Generator using the PCG algorithm
class RNG
{
public:
    // Constructors
    RNG() { SetSequence( static_cast<__uint32_t>( std::hash<std::thread::id>{}( std::this_thread::get_id() ) ) ); }
    RNG( __uint64_t sequenceIndex )                { SetSequence(sequenceIndex); }
    RNG( __uint64_t sequenceIndex, __uint64_t seed ) { SetSequence(sequenceIndex,seed); }

    // Selects a sequence with the given index and seed.
    void SetSequence( __uint64_t sequenceIndex ) { SetSequence( sequenceIndex, MixBits(sequenceIndex) ); }
    void SetSequence( __uint64_t sequenceIndex, __uint64_t seed )
    {
        state = 0u;
        inc = (sequenceIndex << 1u) | 1u;
        RandomInt();
        state += seed;
        RandomInt();
    }

    // Returns a random integer.
    __host__ __device__ __uint32_t RandomInt()
    {
        // minimal PCG32 / (c) 2014 M.E. O'Neill / pcg-random.org
        __uint64_t oldstate = state;
        state = oldstate * PCG32_Mult() + inc;
        __uint32_t xorshifted = (__uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        __uint32_t rot = (__uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    // Returns a random float.
    __host__ __device__ float RandomFloat()
    {
        const float rmax = 0x1.fffffep-1;
        float r = RandomInt() * 0x1p-32f;
        return r < rmax ? r : rmax;
    }

    // Advances the random sequence by the given offset, which can be positive or negative.
    __host__ __device__ void Advance( int64_t offset )
    {
        uint64_t curMult = PCG32_Mult(), curPlus = inc, accMult = 1u;
        uint64_t accPlus = 0u, delta = offset;
        while ( delta > 0 ) {
            if (delta & 1) {
                accMult *= curMult;
                accPlus = accPlus * curMult + curPlus;
            }
            curPlus = (curMult + 1) * curPlus;
            curMult *= curMult;
            delta /= 2;
        }
        state = accMult * state + accPlus;
    }


private:
    __uint64_t state, inc;

    __host__ __device__ constexpr inline __uint64_t PCG32_Mult() const { return 0x5851f42d4c957f2dULL; }

    inline __uint64_t MixBits(__uint64_t v)
    {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44d;
        v ^= (v >> 33);
        return v;
    }
};

#endif //CUDA_RAYTRACING_RNG_CUH