typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_MIRRORED_REPEAT;
__constant sampler_t tex_position_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float KERNEL[3] = { 3.0f/8.0f, 1.0f/4.0f, 1.0f/16.0f };


__kernel void blur(
        write_only image2d_t output,
        read_only image2d_t input,
        read_only image2d_t ss_positions,
        uint8_t iteration
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;

    const float4 current_position = read_imagef(ss_positions, tex_position_sampler, gid);

    float3 color = 0.0f;
    float sum = 0.0f;

    const uint32_t spacing = 2 * iteration + 1;
    
    const int32_t width = 2;
    for (int32_t x = -width; x < width + 1; ++x) {
        for (int32_t y = -width; y < width + 1; ++y) {
            const int2 coord_off = (int2)(spacing * x, spacing * y);
            const float4 other_position = read_imagef(ss_positions, tex_position_sampler, gid + coord_off);
            if (other_position.x == current_position.x || other_position.y == current_position.y || other_position.z == current_position.z) {
                const float kernel_val_x = KERNEL[abs(x)];
                const float kernel_val_y = KERNEL[abs(y)];
                const float kernel_val = (kernel_val_x + kernel_val_y) * 0.5f;
                color += kernel_val * read_imagef(input, tex_sampler, gid + coord_off).xyz;
                sum += kernel_val;
            }
        }
    }

    color /= sum;
    write_imagef(output, gid, (float4)(color, 1.0f));
}
