typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


float3 multVec3Mat3(float3 v, __constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[1] + v.z * mat[2],
		v.x * mat[3] + v.y * mat[4] + v.z * mat[5],
		v.x * mat[6] + v.y * mat[7] + v.z * mat[8]
	);
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

    output[]
}