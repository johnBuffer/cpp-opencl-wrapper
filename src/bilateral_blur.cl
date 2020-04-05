typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float KERNEL[3][3] = {
    {0.077847f, 0.123317f, 0.077847f},
    {0.123317f, 0.195346f, 0.123317f},
    {0.077847f, 0.123317f, 0.077847f}
};


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


__kernel void blur(
        read_only image2d_t input,
        read_only image2d_t depth,
        write_only image2d_t output
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
    const float acc = read_imagef(input, tex_sampler, gid).w;

    if (acc < 10.0f) {
        const float current_depth = read_imagef(depth, tex_sampler, gid).y;

        float3 color = (0.0f);
        float sum = 0.0f;
        for (int32_t x = -1; x < 2; ++x) {
            for (int32_t y = -1; y < 2; ++y) {
                const int2 coords = gid + (int2)(x, y);
                const float point_depth = read_imagef(depth, tex_sampler, gid + (int2)(x, y)).y;

                if (fabs(current_depth - point_depth) < 0.125f * 0.125f * 0.125f) {
                    const float4 point_color = read_imagef(input, tex_sampler, coords);
                    const float kernel_val = KERNEL[x + 1][y + 1];
                    sum += kernel_val;
                    color += kernel_val * point_color.xyz / point_color.w;
                }
            }
        }

        color /= sum;
        write_imagef(output, gid, (float4)(color, acc));
    }
    else {
        write_imagef(output, gid, read_imagef(input, tex_sampler, gid));
    }
}
