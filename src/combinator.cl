typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;


__kernel void combine(
		__global float* albedo,
		__global float* shadow,
		image2d_t lighting
	)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float2 pxl_position = (float2)(gid.x / (float)screen_size.x, gid.y / (float)screen_size.y);
	const uint32_t index = gid.x + gid.y * screen_size.x;
	//const float light_scale = 0.5f;
	//const uint8_t light_scale = 2;
	//const uint32_t index_light = (gid.x/ light_scale + gid.y/ light_scale * screen_size.x / light_scale);

	const float light_intensity = 255.0f * read_imagef(lighting, tex_sampler, pxl_position).x;
	//const float light_intensity = fmin(1.0f, shadow[index] + lighting[4 * index]);
	
	albedo[4*index + 0] = light_intensity;
	albedo[4*index + 1] = light_intensity;
	albedo[4*index + 2] = light_intensity;
	albedo[4*index + 3] = 255;
}
