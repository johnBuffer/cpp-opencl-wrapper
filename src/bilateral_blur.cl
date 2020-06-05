typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
__constant sampler_t tex_position_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float KERNEL[5][5] = {
    {0.003765f,	0.015019f,	0.023792f,	0.015019f,	0.003765},
    {0.015019f,	0.059912f,	0.094907f,	0.059912f,	0.015019},
    {0.023792f,	0.094907f,	0.150342f,	0.094907f,	0.023792},
    {0.015019f,	0.059912f,	0.094907f,	0.059912f,	0.015019},
    {0.003765f,	0.015019f,	0.023792f,	0.015019f,	0.003765}
};

__constant float THRESHOLD = 51.0f;


__kernel void blur(
        read_only image2d_t input,
        read_only image2d_t screen_space_positions,
        write_only image2d_t output
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
    const float acc = read_imagef(input, tex_sampler, gid).w;

    const float3 current_position = read_imagef(screen_space_positions, tex_position_sampler, gid).xyz;

    float3 color = (0.0f);
    float sum = 0.0f;
    const int32_t width = 2;
    for (int32_t x = -width; x < width + 1; ++x) {
        for (int32_t y = -width; y < width + 1; ++y) {
            const int2 coords = gid + (int2)(x, y);
            const float3 point_position = read_imagef(screen_space_positions, tex_position_sampler, coords).xyz;
            if ((point_position.x == current_position.x || point_position.y == current_position.y || point_position.z == current_position.z)) {
                const float kernel_val = KERNEL[x + 2][y + 2];
                const float4 point_color = read_imagef(input, tex_sampler, coords);
                sum += kernel_val * (point_color.w);
                color += kernel_val * point_color.xyz * (point_color.w);
            }
        }
    }

    color /= sum;
    write_imagef(output, gid, (float4)(color, 1.0f));
}
