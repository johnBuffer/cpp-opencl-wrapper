typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
constant float ACC_COUNT = 32.0f;
constant float NEAR = 0.5f;
constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
constant sampler_t exact_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;


// Utils functions
float3 preMultVec3Mat3(float3 v, constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[3] + v.z * mat[6],
		v.x * mat[1] + v.y * mat[4] + v.z * mat[7],
		v.x * mat[2] + v.y * mat[5] + v.z * mat[8]
	);
}

float2 projectPoint(const float3 in)
{
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float aspect_ratio = (screen_size.x) / (float)(screen_size.y);
    const float inv_z = 1.0f / in.z;
	const float2 offset = (float2)(0.5f + 0.5f / (float)screen_size.x, 0.5f + 0.5f / (float)screen_size.y);

	return (float2) ((NEAR * in.x * inv_z), (aspect_ratio * NEAR * in.y * inv_z)) + offset;
}

float4 getOldValue(image2d_t temporal_acc, image2d_t last_frame_depth, __constant float* last_view_matrix, const float3 last_position, const float4 intersection)
{
    const float3 last_view_pos = preMultVec3Mat3(intersection.xyz - last_position, last_view_matrix);
    const float2 last_screen_pos = projectPoint(last_view_pos);

	const float4 last_color = read_imagef(temporal_acc, tex_sampler, last_screen_pos);
	const float2 last_depth = read_imagef(last_frame_depth, tex_sampler, last_screen_pos).xy;
	
	float acc = 0.0f;
	const float accuracy_threshold = 0.25f;
	const float far_threshold = 0.1f;
	if (fabs(1.0f - length(last_view_pos) / last_depth.x) < accuracy_threshold && (last_depth.y == intersection.w || last_depth.x > far_threshold)) {
		return last_color;
	}
	
	return (float4)((float3)(0.0f), 0.0f);
}

float3 normalFromNumber(const float number)
{
	const int32_t n = fabs(number);
	return sign(number) * (float3)(n & 1, (n >> 1u) & 1, (n >> 2u) & 1);
}


__kernel void temporal(
    write_only image2d_t result
	, read_only image2d_t temporal_acc
	, read_only image2d_t raw_input
	, constant float* last_view_matrix
    , float3 last_position
	, read_only image2d_t depth
	, read_only image2d_t last_depth
	, read_only image2d_t ss_position
)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));

	float3 color = 1.0f;
	float acc = 1.0f;

	const float4 intersection = read_imagef(ss_position, exact_sampler, gid);
	if (intersection.w) {
		const float4 acc_value = getOldValue(temporal_acc, last_depth, last_view_matrix, last_position, intersection);
		const float3 new_value = read_imagef(raw_input, exact_sampler, gid).xyz;
		acc = acc_value.w + 1.0f;
		if (acc < ACC_COUNT) {
			color = new_value + acc_value.xyz;
		} else {
			acc = ACC_COUNT - 1.0f;
			const float conservation_coef = acc / ACC_COUNT;
			color = conservation_coef * (new_value + acc_value.xyz);
		}
	}
	
	write_imagef(result, gid, (float4)(color, acc));
}

__kernel void normalizer(
	read_only image2d_t input,
    write_only image2d_t output
) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const float4 in_color = read_imagef(input, exact_sampler, gid);
	write_imagef(output, gid, (float4)(fmax(0.0f, in_color.xyz / in_color.w), in_color.w));
}
