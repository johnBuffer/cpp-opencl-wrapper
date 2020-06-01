typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;


__kernel void combine(
		__global float* albedo,
		__global float* shadow,
		image2d_t lighting
	)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
	//const float light_scale = 0.5f;
	//const uint8_t light_scale = 2;
	//const uint32_t index_light = (gid.x/ light_scale + gid.y/ light_scale * screen_size.x / light_scale);

	const float4 gi_value = read_imagef(lighting, tex_sampler, gid);
	const float3 gi_intensity = gi_value.xyz;
	const float3 light_intensity = gi_intensity * (float3)(fmax(0.0f, shadow[index]));
	//const float3 light_intensity = gi_intensity;

	albedo[4*index + 0] *= fmin(1.0f, light_intensity.x);
	albedo[4*index + 1] *= fmin(1.0f, light_intensity.y);
	albedo[4*index + 2] *= fmin(1.0f, light_intensity.z);

	// albedo[4*index + 0] = 255.0f * fmin(1.0f, light_intensity.x);
	// albedo[4*index + 1] = 255.0f * fmin(1.0f, light_intensity.y);
	// albedo[4*index + 2] = 255.0f * fmin(1.0f, light_intensity.z);

	albedo[4*index + 3] = 255;
}
