typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


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
	buffer[4 * index + 3] = 255.0f;
}


__kernel void blur(
        __global float* frame_color,
        __global float* depth,
        __global float* output
    )
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const uint32_t index = gid.x + gid.y * screen_size.x;
    const float current_normal = depth[2 * index];
    const float current_depth = depth[2 * index + 1];

    float color = 0.0f;
    float sum = 0.0f;
    for (int32_t x = -1; x < 2; ++x) {
        for (int32_t y = -1; y < 2; ++y) {
            const int2 coords = gid + (int2)(x, y);
            if (coords.x >= 0 && coords.x < screen_size.x && coords.y >= 0 && coords.y < screen_size.y) {
                const uint32_t index_2 = getIndexFromCoords(coords);
                const float normal = depth[2 * index_2];
                const float point_depth = depth[2 * index_2 + 1];

                if (normal == current_normal && fabs(current_depth - point_depth) < 0.125f * 0.125f * 0.125f) {
                    const float kernel_val = KERNEL[x + 1][y + 1];
                    sum += kernel_val;
                    color += kernel_val * frame_color[4 * index_2];
                }
            }
        }
    }

    color /= sum;
    colorToResultBuffer((float3)color, index, output);
}
