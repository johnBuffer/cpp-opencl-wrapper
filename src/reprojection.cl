typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant float NEAR = 0.5f;


/*float3 multVec3Mat3(float3 v, __constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[1] + v.z * mat[2],
		v.x * mat[3] + v.y * mat[4] + v.z * mat[5],
		v.x * mat[6] + v.y * mat[7] + v.z * mat[8]
	);
}*/

float3 preMultVec3Mat3(float3 v, __constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[3] + v.z * mat[6],
		v.x * mat[1] + v.y * mat[4] + v.z * mat[7],
		v.x * mat[2] + v.y * mat[5] + v.z * mat[8]
	);
}


int2 projectPoint(const float3 in, const int2 screen_size)
{
	const float aspect_ratio = screen_size.x / (float)screen_size.y;

    float2 out_f;
    const float inv_z = 1.0f / in.z;

	out_f.x = (NEAR * in.x * inv_z) + 0.5f;
	out_f.y = (aspect_ratio * NEAR * in.y * inv_z) + 0.5f;

    int2 out_i;
    out_i.x = out_f.x * (screen_size.x + 1);
    out_i.y = out_f.y * (screen_size.y + 1);

	return out_i;
}


void colorToResultBuffer(float3 color, uint32_t index, __global float* buffer)
{
    buffer[4 * index + 0] = color.x;
	buffer[4 * index + 1] = color.y;
	buffer[4 * index + 2] = color.z;
	buffer[4 * index + 3] = 255.0f;
}

float4 getFloat4FromBuffer(__global float* buffer, uint32_t index)
{
    return (float4)(buffer[4 * index], buffer[4*index + 1], buffer[4*index + 2], buffer[4*index + 3]);
}


__kernel void reproject(
    __global float* last_frame_points,
    __global float* last_frame_color,
    __constant float* view_matrix,
    float3 camera_position,
    __global float* output
)
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const uint32_t index = gid.x + gid.y * get_global_size(0);
    const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 0.5f);

    colorToResultBuffer((float3)(0.0f), index, output);

    const float4 buffer_point = getFloat4FromBuffer(last_frame_points, index);
    if (buffer_point.w) {
        const float3 view_point = multVec3Mat3(buffer_point.xyz - camera_position, view_matrix);
        const int2 projected_point = projectPoint(view_point, screen_size);

        const int32_t point_index = projected_point.x + projected_point.y * screen_size.x;
        if (projected_point.x >= 0 && projected_point.x < screen_size.x && projected_point.y >= 0 && projected_point.y < screen_size.y) {
            const float dist = output[4 * point_index + 3];

            if (view_point.z < dist) {
                output[4 * point_index + 0] = last_frame_color[4 * index + 0];
                output[4 * point_index + 1] = last_frame_color[4 * index + 1];
                output[4 * point_index + 2] = last_frame_color[4 * index + 2];
                output[4 * point_index + 3] = view_point.z;
            }
        }
    }
}
