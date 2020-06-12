typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR;


__kernel void combine(
		__global float* albedo,
		image2d_t gi,
		image2d_t shadows
	)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
	const float2 tex_coords = (float2)(gid.x, gid.y) / (float2)(screen_size.x, screen_size.y);
	
	const float4 gi_value = read_imagef(gi, tex_sampler, gid);
	const float3 shadows_value = read_imagef(shadows, tex_sampler, gid).xyz;
	const float3 light_intensity = shadows_value + gi_value.xyz;

	albedo[4*index + 0] *= fmin(1.0f, light_intensity.x);
	albedo[4*index + 1] *= fmin(1.0f, light_intensity.y);
	albedo[4*index + 2] *= fmin(1.0f, light_intensity.z);
	// albedo[4*index + 0] = 255.0f * fmin(1.0f, light_intensity.x);
	// albedo[4*index + 1] = 255.0f * fmin(1.0f, light_intensity.y);
	// albedo[4*index + 2] = 255.0f * fmin(1.0f, light_intensity.z);
	
	albedo[4*index + 3] = 255;
}
