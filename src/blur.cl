typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float KERNEL[] = {0.382925f, 0.24173f, 0.060598f, 0.005977f, 0.000229f, 0.000003f};

__constant float THRESHOLD = 10.0f;


uint32_t getIndexFromCoords(const int2 coords)
{
    return coords.x + coords.y * get_global_size(0);
}


void colorToResultBuffer(float3 color, uint32_t index, __global float* buffer)
{
    buffer[4 * index + 0] = color.x;
	buffer[4 * index + 1] = color.y;
	buffer[4 * index + 2] = color.z;
}


__kernel void blur_v(
        read_only image2d_t input,
        write_only image2d_t output
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    
    const float4 current_color = read_imagef(input, tex_sampler, gid);

    if (current_color.w < THRESHOLD) {
        float3 color = (float3)(0.0f);
        float sum = 0.0f;

        for (int i=-5; i<6; ++i) {
            const float kernel_value = KERNEL[abs(i)];
            sum += kernel_value;
            const float4 pxl_color = read_imagef(input, tex_sampler, gid + (int2)(0, i));
            color += (pxl_color.xyz / pxl_color.w) * kernel_value;        }
        color /= sum;

        write_imagef(output, gid, (float4)(color * current_color.w, current_color.w));
    } else {
        write_imagef(output, gid, current_color);
    }
}

__kernel void blur_h(
        read_only image2d_t input,
        write_only image2d_t output
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    
    const float4 current_color = read_imagef(input, tex_sampler, gid);

    if (current_color.w < THRESHOLD) {
        float3 color = (float3)(0.0f);
        float sum = 0.0f;

        for (int i=-5; i<6; ++i) {
            const float kernel_value = KERNEL[abs(i)];
            sum += kernel_value;
            const float4 pxl_color = read_imagef(input, tex_sampler, gid + (int2)(i, 0));
            color += (pxl_color.xyz / pxl_color.w) * kernel_value;
        }
        color /= sum;

        write_imagef(output, gid, (float4)(color * current_color.w, current_color.w));
    } else {
        write_imagef(output, gid, current_color);
    }
}
