typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
__constant sampler_t tex_position_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

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
    const float4 current_color = read_imagef(input, tex_sampler, gid);

    if (current_color.w == 1.0f) {
        const float3 current_position = read_imagef(screen_space_positions, tex_position_sampler, gid).xyz;
        float3 color = (0.0f);
        float sum = 0.0f;
        const int32_t width = 2;
        for (int32_t x = -width; x < width + 1; ++x) {
            for (int32_t y = -width; y < width + 1; ++y) {
                const int2 coords = gid + (int2)(x, y);
                const float4 point_color = read_imagef(input, tex_sampler, coords);
                sum += pow(point_color.w, 4.0f);
                color += point_color.xyz * pow(point_color.w, 4.0f);
            }
        }

        color /= sum;
        write_imagef(output, gid, (float4)(color, 1.0f));
    } else {
        write_imagef(output, gid, current_color);
    }
}
