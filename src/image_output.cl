
__kernel void work_id_output(
	__read_only const uint2 size,
	__global unsigned char* result)
{
	const int2 gid = (get_global_id(0), get_global_id(1));
	const unsigned int index = 4 * (gid.x + gid.y * size.x);
	result[index + 0] = gid.y;// *gid.x / size.x;
	result[index + 1] = 0;// *gid.y / size.y;
	result[index + 2] = 0;
	result[index + 3] = 255;
}
