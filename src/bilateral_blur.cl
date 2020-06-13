typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR  | CLK_ADDRESS_MIRRORED_REPEAT;
__constant sampler_t exact_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_MIRRORED_REPEAT;

// From SVGF
__constant float C_PHI_REF = 4.0f;
__constant float N_PHI_REF = 128.0f;
__constant float P_PHI_REF = 1.0f;

__constant float KERNEL[3] = { 1.0f/2.0f, 1.0f/4.0f, 1.0f/8.0f };


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
        read_only image2d_t depth_gradient,
        uint8_t iteration
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));

    // const float4 temporal_data = read_imagef(color_input, linear_sampler, gid);
    // if (temporal_data.w) {
    //     const float3 current_color    = temporal_data.xyz;
    //     const float4 current_data     = read_imagef(ss_positions, exact_sampler, gid);
    //     const float3 current_position = current_data.xyz;
    //     const float3 current_normal   = normalFromNumber(current_data.w);

    //     float3 color = 0.0f;
    //     float sum = 0.0f;

    //     const uint32_t spacing = 1 << iteration;
    //     const float c_phi = C_PHI_REF / (float)(spacing);
    //     const float p_phi = P_PHI_REF / (float)(spacing);
    //     const float n_phi = N_PHI_REF / (float)(spacing);
        
    //     const int32_t width = 2;
    //     float3 tmp;
    //     float dist2;
    //     for (int32_t x = -width; x < width + 1; ++x) {
    //         for (int32_t y = -width; y < width + 1; ++y) {
    //             // Kernel value
    //             const float kernel_val_x = KERNEL[abs(x)];
    //             const float kernel_val_y = KERNEL[abs(y)];
    //             const float kernel_val = (kernel_val_x + kernel_val_y) * 0.5f;
    //             // Offset
    //             const int2 coord_off = (int2)(spacing * x, spacing * y);
    //             // Retrieving other data
    //             const float3 other_color = read_imagef(color_input, linear_sampler, gid + coord_off).xyz;
    //             const float4 other_data = read_imagef(ss_positions, exact_sampler, gid + coord_off);
    //             const float3 other_position = other_data.xyz;
    //             const float3 other_normal   = other_data.w;
    //             // Color
    //             //tmp = current_color - other_color;
    //             //dist2 = dot(tmp, tmp);
    //             //const float w_c = min(exp(-dist2/c_phi), 1.0f);
    //             // Normal
    //             const float w_n = max(dot(current_normal, other_normal), 0.0f);
    //             // Position
    //             //tmp = current_position - other_position;
    //             //dist2 = dot(tmp, tmp);
    //             //const float w_p = min(exp(-dist2/p_phi), 1.0f);
    //             // Sum
    //             const float weight = w_n + 0.0001f;
    //             color += other_color * (kernel_val * weight);
    //             sum += kernel_val * weight;
    //         }
    //     }

    //     color /= sum;
    //     write_imagef(output, gid, (float4)(color, 1.0f));
    // } else {
    //     write_imagef(output, gid, temporal_data);
    // }

    write_imagef(output, gid, read_imagef(depth_gradient, exact_sampler, gid));
}

__kernel void gradient(
        write_only image2d_t output,
        read_only image2d_t input
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const float depth = read_imagef(input, exact_sampler, gid).x;

    float sum = 0.0f;
    for (int32_t x = -1; x < 2; ++x) {
        for (int32_t y = -1; y < 2; ++y) {
            const int2 coord_off = (int2)(x, y);
            sum += depth - read_imagef(input, exact_sampler, gid + coord_off).x;
        }
    }

    write_imagef(output, gid, (float4)(sum / 8.0f));
}
