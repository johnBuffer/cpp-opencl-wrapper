
__kernel void work_id_output(__global unsigned char* result)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const unsigned int index = gid.x + gid.y * get_global_size(0);
	result[4 * index + 0] = 255 * (gid.x / (float)get_global_size(0));
	result[4 * index + 1] = 255 * (gid.y / (float)get_global_size(1));
	result[4 * index + 2] = 0;
	result[4 * index + 3] = 255;
}
