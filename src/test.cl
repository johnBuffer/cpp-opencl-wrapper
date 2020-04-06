typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


float3 normalFromNumber(const float number)
{
	const int32_t n = fabs(number);
	return sign(number) * (float3)(n & 1, (float)((n >> 1) & 1), (n >> 2u) & 1);
}


float normalToNumber(const float3 normal)
{
	return normal.x + 2.0f * normal.y + 4.0f * normal.z;
}


__kernel void test(
    global float* data
)
{
    const int32_t test_n = 2;
    const int gid = get_global_id(0);
    const float3 n = (float3)(0.0f, -1.0f, 0.0f);
    const float number = normalToNumber(n);
    data[0] = ((test_n >> 1u) & 2u);
    const float3 n_r = normalFromNumber(number);

    data[1] = n_r.x;
    data[2] = n_r.y;
    data[3] = n_r.z;
}