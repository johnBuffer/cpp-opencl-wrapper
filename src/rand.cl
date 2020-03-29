typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant float normalizer = 1.0f / 4294967296.0f;


float rand(__global int32_t* seed, int32_t index)
{
    const int32_t a = 16807;
    const int32_t m = 2147483647;

    return (seed[index] / (float)m);
}


uint rand_xorshift(uint32_t* state)
{
    // Xorshift algorithm from George Marsaglia's paper
	uint32_t s = *state;
    s ^= (s << 13);
    s ^= (s >> 17);
    s ^= (s << 5);
	*state = s;
    return s;
}


uint wang_hash(uint32_t seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}


__kernel void work_id_output(
	__global float* result
)
{
	const int32_t gid = get_global_id(0);
	uint32_t rng_state = wang_hash(gid);
	result[gid] = rand_xorshift(&rng_state) * normalizer;
}
