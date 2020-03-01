
__kernel void work_id_output(__global uint* result)
{
	const int2 gid = (get_global_id(0), get_global_id(1));
	const unsigned int index = gid.x + gid.y * get_global_size(0);
	result[index] = gid.x;
}
