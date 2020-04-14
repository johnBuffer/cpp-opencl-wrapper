typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

__constant uint32_t GRID_SIZE = 256u; 


typedef struct HitPoint
{
	char     hit; // Could be removed because normal can do this
	float3   position;
	float2   tex_coords;
	float    distance;
	float3   normal;
	bool     water;
	bool     emissive;
	uint16_t complexity;
} HitPoint;


void castRay(const float3 start, const float3 direction, __global uint8_t* grid)
{
    HitPoint point;
    point.hit = false;

	// Compute how much (in units of t) we can move along the ray
	// before reaching the cell's width and height
	const float t_dx = fabs(1.0f / direction.x);
	const float t_dy = fabs(1.0f / direction.y);
	const float t_dz = fabs(1.0f / direction.z);

	// step_x and step_y describe if cell_x and cell_y
	// are incremented or decremented during iterations
	const int32_t step_x = direction.x < 0 ? -1 : 1;
	const int32_t step_y = direction.y < 0 ? -1 : 1;
	const int32_t step_z = direction.z < 0 ? -1 : 1;

	// Compute the value of t for first intersection in x and y
	const int32_t dir_x = step_x > 0 ? 1 : 0;
	const int32_t dir_y = step_y > 0 ? 1 : 0;
	const int32_t dir_z = step_z > 0 ? 1 : 0;

	// cell_x and cell_y are the starting voxel's coordinates
	int32_t cell_x = position.x;
	int32_t cell_y = position.y;
	int32_t cell_z = position.z;

	float t_max_x = ((cell_x + dir_x) - position.x) / direction.x;
	float t_max_y = ((cell_y + dir_y) - position.y) / direction.y;
	float t_max_z = ((cell_z + dir_z) - position.z) / direction.z;

	uint8_t hit_side;

	const uint32_t max_iter = 128;
	uint32_t iter = 0U;
	while (cell_x >= 0 && cell_y >= 0 && cell_z >= 0 && cell_x < GRID_SIZE && cell_y < GRID_SIZE && cell_z < GRID_SIZE && iter < max_iter) {
		float t_max_min;
		++iter;
		if (t_max_x < t_max_y) {
			if (t_max_x < t_max_z) {
				t_max_min = t_max_x;
				t_max_x += t_dx;
				cell_x += step_x;
				hit_side = 0;
			}
			else {
				t_max_min = t_max_z;
				t_max_z += t_dz;
				cell_z += step_z;
				hit_side = 2;
			}
		}
		else {
			if (t_max_y < t_max_z) {
				t_max_min = t_max_y;
				t_max_y += t_dy;
				cell_y += step_y;
				hit_side = 1;
			}
			else {
				t_max_min = t_max_z;
				t_max_z += t_dz;
				cell_z += step_z;
				hit_side = 2;
			}
		}

		if (cell_x >= 0 && cell_y >= 0 && cell_z >= 0 && cell_x < GRID_SIZE && cell_y < GRID_SIZE && cell_z < GRID_SIZE) {
            const uint32_t cell_index = cell_x + cell_y * GRID_SIZE + cell_z * GRID_SIZE * GRID_SIZE;
			if (grid[cell_index]) {
				float hit_x = position.x + t_max_min * direction.x;
				float hit_y = position.y + t_max_min * direction.y;
				float hit_z = position.z + t_max_min * direction.z;

				point.position = (float3)(hit_x, hit_y, hit_z);

				if (hit_side == 0) {
					point.normal = (float3)(-step_x, 0.0f, 0.0f);
					//point.voxel_coord = glm::vec2(1.0f-frac(hit_z), frac(hit_y));
				} else if (hit_side == 1) {
					point.normal = (float3)(0.0f, -step_y, 0.0f);
					//point.voxel_coord = glm::vec2(frac(hit_x), frac(hit_z));
				}else if (hit_side == 2) {
					point.normal = (float3)(0.0f, 0.0f, -step_z);
					//point.voxel_coord = glm::vec2(frac(hit_x), frac(hit_y));
				}

				point.distance = t_max_min;

				break;
			}
		}
	}

	return point;
}


__kernel void test(
    __global uint8_t* grid,
    constant float3 position,
    constant float3* view_matrix
)
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const uint32_t index = gid.x + gid.y * get_global_size(0);
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 0.5f);
	// Cast ray
	const float3 d = normalize(multVec3Mat3(screen_position, view_matrix));
}
