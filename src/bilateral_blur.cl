typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
__constant sampler_t tex_position_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float KERNEL[3][3] = {
    {0.077847f, 0.123317f, 0.077847f},
    {0.123317f, 0.195346f, 0.123317f},
    {0.077847f, 0.123317f, 0.077847f}
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

    if (acc < THRESHOLD) {
        const float3 current_position = read_imagef(screen_space_positions, tex_position_sampler, gid).xyz;

        float3 color = (0.0f);
        float sum = 0.0f;
        const int32_t min_width = 3;
        const int32_t max_width = 12;
        const int32_t width = min_width + (max_width - min_width) * (1.0f - acc/THRESHOLD);
        for (int32_t x = -width; x < width + 1; ++x) {
            for (int32_t y = -width; y < width + 1; ++y) {
                const int2 coords = gid + (int2)(x, y);
                const float3 point_position = read_imagef(screen_space_positions, tex_position_sampler, coords).xyz;

                if ((point_position.x == current_position.x || point_position.y == current_position.y || point_position.z == current_position.z)) {
                    const float4 point_color = read_imagef(input, tex_sampler, coords);
                    //const float kernel_val = KERNEL[x + 1][y + 1];
                    sum += point_color.w;
                    color += point_color.w * point_color.xyz;
                }
            }
        }

        color /= sum;
        write_imagef(output, gid, (float4)(color, 1.0f));
    }
    else {
        write_imagef(output, gid, read_imagef(input, tex_sampler, gid));
    }
}
