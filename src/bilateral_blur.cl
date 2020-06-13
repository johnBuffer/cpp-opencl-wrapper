typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR  | CLK_ADDRESS_MIRRORED_REPEAT;
__constant sampler_t exact_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_MIRRORED_REPEAT;

__constant float KERNEL[3] = { 3.0f/8.0f, 1.0f/4.0f, 1.0f/16.0f };


// Utils
float3 normalFromNumber(const float number)
{
	const int32_t n = fabs(number);
	return sign(number) * (float3)(n & 1, (n >> 1u) & 1, (n >> 2u) & 1);
}


__kernel void blur(
        write_only image2d_t output,
        read_only image2d_t color_input,
        read_only image2d_t ss_positions,
        uint8_t iteration
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));

    const float4 temporal_data = read_imagef(color_input, linear_sampler, gid);
    if (temporal_data.w) {
        const float3 current_color    = temporal_data.xyz;
        const float4 current_data     = read_imagef(ss_positions, exact_sampler, gid);
        const float3 current_position = current_data.xyz;
        const float3 current_normal   = current_data.w;

        float3 color = 0.0f;
        float sum = 0.0f;

        const uint32_t spacing = 2 * iteration - 1;
        
        const int32_t width = 2;
        float3 tmp;
        float dist2;
        for (int32_t x = -width; x < width + 1; ++x) {
            for (int32_t y = -width; y < width + 1; ++y) {
                // Offset
                const int2 coord_off = (int2)(spacing * x, spacing * y);
                // Retrieving other data
                const float4 other_data   = read_imagef(ss_positions, exact_sampler, gid + coord_off);
                const float3 other_position = other_data.xyz;
                const float other_normal = other_data.w;
                if (other_position.x == current_position.x || other_position.y == current_position.y || other_position.z == current_position.z) {
                    const float4 other_color  = read_imagef(color_input, linear_sampler, gid + coord_off);
                    // Kernel value
                    const float kernel_val_x = KERNEL[abs(x)];
                    const float kernel_val_y = KERNEL[abs(y)];
                    const float kernel_val = (kernel_val_x + kernel_val_y) * 0.5f;
                    // Sum
                    const float weight = other_color.w * kernel_val + 0.0001f;
                    color += other_color.xyz * weight;
                    sum += weight;
                }
            }
        }

        color /= sum;
        write_imagef(output, gid, (float4)(color, 1.0f));
    } else {
        write_imagef(output, gid, temporal_data);
    }
}

__kernel void gradient(
        write_only image2d_t output,
        read_only image2d_t input
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const float depth = 512.0f * read_imagef(input, exact_sampler, gid).x;

    float sum = 0.0f;
    for (int32_t x = -1; x < 2; ++x) {
        for (int32_t y = -1; y < 2; ++y) {
            const int2 coord_off = (int2)(x, y);
            sum += depth - 512.0f * read_imagef(input, exact_sampler, gid + coord_off).x;
        }
    }

    write_imagef(output, gid, (float4)(sum / 8.0f));
}
