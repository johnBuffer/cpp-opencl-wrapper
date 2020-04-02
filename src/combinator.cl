typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__kernel void combine(
		__global float* albedo,
		__global float* shadow,
		__global float* lighting
	)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
	const uint32_t light_downscale = 1;
	const uint32_t index_light = gid.x/light_downscale + gid.y/light_downscale * screen_size.x/light_downscale;

	const float light_intensity = fmin(1.0f, shadow[index] + lighting[4 * index_light]);

	albedo[4*index + 0] *= light_intensity;
	albedo[4*index + 1] *= light_intensity;
	albedo[4*index + 2] *= light_intensity;
}
