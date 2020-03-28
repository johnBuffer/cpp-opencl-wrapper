typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
//__constant float3 light_position = (float3)(0.0f, 1.0f, 0.0f);
__constant float EPS = 0x1.fffffep-1f;
__constant float NORMAL_EPS = 0.0078125f * 0.0078125f * 0.0078125f;
__constant float AMBIENT = 0.05f;
__constant float SUN_INTENSITY = 10.0f;
__constant float3 SKY_COLOR = (float3)(51.0f, 204.0f, 255.0f);
__constant float3 WATER_COLOR = (float3)(0.0f, 100.0f / 255.0f, 150.0f / 255.0f);


// Structs declaration
typedef struct Node
{
    uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
	uint8_t reflective_mask;
	uint8_t padding;
} Node;

typedef struct OctreeStack
{
	uint32_t parent_index;
	float t_max;
} OctreeStack;

typedef struct HitPoint
{
	char     hit; // Could be removed because normal can do this
	float3   position;
	float2   tex_coords;
	float    distance;
	char3    normal;
	bool     water;
	uint16_t complexity;
} HitPoint;


// Utils functions
float3 multVec3Mat3(float3 v, __constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[1] + v.z * mat[2],
		v.x * mat[3] + v.y * mat[4] + v.z * mat[5],
		v.x * mat[6] + v.y * mat[7] + v.z * mat[8]
	);
}

float frac(float x)
{
	return fmin(x - floor(x), EPS);
}

float rand(__global int32_t* seed, int32_t index)
{
    const int32_t a = 16807;
    const int32_t m = 2147483647;

    seed[index] = ((long)(seed[index] * a))%m;
    return (seed[index] / (float)m);
}

float3 getRandomizedNormal(float3 normal, __global int32_t* seed, int32_t index)
{
	const float range = 1.0f;
	const float coord_1 = range * rand(seed, index);
	const float coord_2 = range * rand(seed, index);
	if (normal.x) {
		return normalize((float3)(normal.x, coord_1, coord_2));
	}
	else if (normal.y) {
		return normalize((float3)(coord_1, normal.y, coord_2));
	}
	else if (normal.z) {
		return normalize((float3)(coord_1, coord_2, normal.z));
	}

	return (float3)(0.0f);
}

// Raytracing functions
HitPoint castRay(__global Node* svo_data, float3 position, float3 d)
{
	HitPoint result;
	result.hit = 0;
	result.complexity = 0;

	const float EPS = 1.0f / (float)(1 << SVO_MAX_DEPTH);
	
	// Initialize stack
	OctreeStack stack[23];
	// Check octant mask and modify ray accordingly
	if (fabs(d.x) < EPS) { d.x = copysign(EPS, d.x); }
	if (fabs(d.y) < EPS) { d.y = copysign(EPS, d.y); }
	if (fabs(d.z) < EPS) { d.z = copysign(EPS, d.z); }
	const float3 t_coef = -1.0f / fabs(d);
	float3 t_offset = position * t_coef;
	uint8_t mirror_mask = 7u;
	if (d.x > 0.0f) { mirror_mask ^= 1u, t_offset.x = 3.0f * t_coef.x - t_offset.x; }
	if (d.y > 0.0f) { mirror_mask ^= 2u, t_offset.y = 3.0f * t_coef.y - t_offset.y; }
	if (d.z > 0.0f) { mirror_mask ^= 4u, t_offset.z = 3.0f * t_coef.z - t_offset.z; }
	// Initialize t_span
	float t_min = fmax(2.0f * t_coef.x - t_offset.x, fmax(2.0f * t_coef.y - t_offset.y, 2.0f * t_coef.z - t_offset.z));
	float t_max = fmin(       t_coef.x - t_offset.x, fmin(       t_coef.y - t_offset.y,        t_coef.z - t_offset.z));
	float h = t_max;
	t_min = fmax(0.0f, t_min);
	t_max = fmin(1.0f, t_max);
	// Init current voxel
	uint32_t parent_id = 0u;
	uint8_t child_offset = 0u;
	int8_t scale = SVO_MAX_DEPTH - 1u;
	float3 pos = (float3)(1.0f);
	float scale_f = 0.5f;
	// Initialize child position
	if (1.5f * t_coef.x - t_offset.x > t_min) { child_offset ^= 1u, pos.x = 1.5f; }
	if (1.5f * t_coef.y - t_offset.y > t_min) { child_offset ^= 2u, pos.y = 1.5f; }
	if (1.5f * t_coef.z - t_offset.z > t_min) { child_offset ^= 4u, pos.z = 1.5f; }
	uint8_t normal = 0u;
	uint16_t child_infos = 0u;
	// Explore octree
	while (scale < SVO_MAX_DEPTH) {
		++result.complexity;
		const Node parent_ref = svo_data[parent_id];
		// Compute new T span
		const float3 t_corner = (float3)(pos.x * t_coef.x - t_offset.x, pos.y * t_coef.y - t_offset.y, pos.z * t_coef.z - t_offset.z);
		const float tc_max = fmin(t_corner.x, fmin(t_corner.y, t_corner.z));
		// Check if child exists here
		const uint8_t child_shift = child_offset ^ mirror_mask;
		const uint8_t child_mask = parent_ref.child_mask >> child_shift;
		if ((child_mask & 1u) && (t_min <= t_max || true)) {
			const float tv_max = fmin(t_max, tc_max);
			const float half_scale = scale_f * 0.5f;
			const float3 t_half = half_scale * t_coef + t_corner;
			if (t_min <= tv_max || true) {
				const uint8_t leaf_mask = parent_ref.leaf_mask >> child_shift;
				// We hit a leaf
				if (leaf_mask & 1u) {
					result.hit = 1u;
					// Could use mirror mask
					result.normal = -convert_char3(sign(d)) * (char3)(normal & 1u, normal & 2u, normal & 4u);
					result.distance = t_min;
					result.water = (parent_ref.reflective_mask >> child_shift) & 1u;

					if ((mirror_mask & 1) == 0) pos.x = 3.0f - scale_f - pos.x;
					if ((mirror_mask & 2) == 0) pos.y = 3.0f - scale_f - pos.y;
					if ((mirror_mask & 4) == 0) pos.z = 3.0f - scale_f - pos.z;
					result.position.x = fmin(fmax(position.x + t_min * d.x, pos.x + EPS), pos.x + scale_f - EPS);
					result.position.y = fmin(fmax(position.y + t_min * d.y, pos.y + EPS), pos.y + scale_f - EPS);
					result.position.z = fmin(fmax(position.z + t_min * d.z, pos.z + EPS), pos.z + scale_f - EPS);

					const float tex_scale = (float)(1 << (SVO_MAX_DEPTH - scale));
					if (result.normal.x) {
						result.tex_coords = (float2)(frac(result.position.z * tex_scale), frac(result.position.y * tex_scale));
					}
					else if (result.normal.y) {
						result.tex_coords = (float2)(frac(result.position.x * tex_scale), frac(result.position.z * tex_scale));
					}
					else if (result.normal.z) {
						result.tex_coords = (float2)(frac(result.position.x * tex_scale), frac(result.position.y * tex_scale));
					}
					break;
				}
				// Eventually add parent to the stack
				if (tc_max < h) {
					stack[scale].parent_index = parent_id;
					stack[scale].t_max = t_max;
				}
				h = tc_max;
				// Update current voxel
				parent_id += parent_ref.child_offset + child_shift;
				child_offset = 0u;
				--scale;
				scale_f = half_scale;
				if (t_half.x > t_min) { child_offset ^= 1u, pos.x += scale_f; }
				if (t_half.y > t_min) { child_offset ^= 2u, pos.y += scale_f; }
				if (t_half.z > t_min) { child_offset ^= 4u, pos.z += scale_f; }
				t_max = tv_max;
				continue;
			}
		} // End of depth exploration

		uint32_t step_mask = 0u;
		if (t_corner.x <= tc_max) { step_mask ^= 1u, pos.x -= scale_f; }
		if (t_corner.y <= tc_max) { step_mask ^= 2u, pos.y -= scale_f; }
		if (t_corner.z <= tc_max) { step_mask ^= 4u, pos.z -= scale_f; }

		t_min = tc_max;
		child_offset ^= step_mask;
		normal = step_mask;

		if (child_offset & step_mask) {
			uint32_t differing_bits = 0u;
			const int32_t ipos_x = as_int(pos.x);
			const int32_t ipos_y = as_int(pos.y);
			const int32_t ipos_z = as_int(pos.z);
			if (step_mask & 1u) differing_bits |= (ipos_x ^ as_int(pos.x + scale_f));
			if (step_mask & 2u) differing_bits |= (ipos_y ^ as_int(pos.y + scale_f));
			if (step_mask & 4u) differing_bits |= (ipos_z ^ as_int(pos.z + scale_f));
			scale = (as_int((float)differing_bits) >> SVO_MAX_DEPTH) - 127u;
			scale_f = as_float((scale - SVO_MAX_DEPTH + 127u) << SVO_MAX_DEPTH);
			const OctreeStack entry = stack[scale];
			parent_id = entry.parent_index;
			t_max = entry.t_max;
			const uint32_t shx = ipos_x >> scale;
			const uint32_t shy = ipos_y >> scale;
			const uint32_t shz = ipos_z >> scale;
			pos.x = as_float(shx << scale);
			pos.y = as_float(shy << scale);
			pos.z = as_float(shz << scale);
			child_offset = (shx & 1u) | ((shy & 1u) << 1u) | ((shz & 1u) << 2u);
			h = 0.0f;
		}
	}

	return result;
}

float3 getColorFromNormal(char3 normal)
{
	if (normal.x) {
		return (float3)(255.0f, 0.0f, 0.0f);
	}
	if (normal.y) {
		return (float3)(0.0f, 255.0f, 0.0f);
	}
	if (normal.z) {
		return (float3)(0.0f, 0.0f, 255.0f);
	}
}

float getGlobalIllumination(__global Node* svo_data, const float3 position, const float3 normal, float3 light_position, __global int32_t* seed, int32_t index)
{
	float gi_add = 0.0f;
	const float range = 100.0f;
	const uint32_t ray_count = 1u;
	const float ray_contrib = 1.0f / (float)ray_count;
	for (uint32_t i = ray_count; i--;) {
		// First bounce
		const float3 noise_normal = getRandomizedNormal(normal, seed, index);
		const HitPoint gi_intersection = castRay(svo_data, position, noise_normal);
		if (gi_intersection.hit) {
			const float3 gi_normal = convert_float3(gi_intersection.normal);
			const float3 gi_light_start = gi_intersection.position + NORMAL_EPS * gi_normal;
			const float3 gi_light_direction = normalize(light_position - gi_light_start);
			const HitPoint gi_light_intersection = castRay(svo_data, gi_light_start, gi_light_direction);
			if (!gi_light_intersection.hit) {
				gi_add += SUN_INTENSITY * fmax(AMBIENT, dot(gi_normal, gi_light_direction)) * ray_contrib;
			}

			/*const float3 noise_normal2 = getRandomizedNormal(gi_normal, seed, index);
			const HitPoint gi_intersection2 = castRay(svo_data, gi_light_start, noise_normal2);
			if (gi_intersection2.hit) {
				const float3 gi_normal2 = convert_float3(gi_intersection2.normal);
				const float3 gi_light_start2 = gi_intersection2.position + NORMAL_EPS * gi_normal2;
				const float3 gi_light_direction2 = normalize(light_position - gi_light_start2);
				const HitPoint gi_light_intersection2 = castRay(svo_data, gi_light_start2, gi_light_direction2);
				if (!gi_light_intersection2.hit) {
					gi_add += SUN_INTENSITY * fmax(AMBIENT, dot(gi_normal2, gi_light_direction2)) * ray_contrib;
				}
			}*/
		}
	}
	
	return gi_add;
}

float get_ambient_occlusion(__global Node* svo_data, const float3 position, const float3 normal, __global int32_t* seed, int32_t index)
{
	float acc = 1.0f;
	const float range = 1.0f;
	const uint32_t ray_count = 4u;
	const float ray_contrib = 1.0f / (float)ray_count;
	for (uint32_t i = ray_count; i--;) {
		const float3 noise_normal = getRandomizedNormal(normal, seed, index);
		const HitPoint ao_intersection = castRay(svo_data, position, noise_normal);
		if (ao_intersection.hit) {
			acc -= ray_contrib;
		}
	}
	
	return acc;
}

float3 reflect(float3 v, float3 normal){
	return v - 2.0f * dot(v, normal) * normal;
}

float3 refract(float3 v, float3 normal, float index)
{
	const float cos_i = -dot(normal, v);
	const float cos_t2 = 1.0f - index * index * (1.0f - cos_i * cos_i);
	return (index * v) + (index * cos_i - sqrt( cos_t2 )) * normal;
}

void colorToResultBuffer(float3 color, uint32_t index, __global float* buffer)
{
    buffer[4 * index + 0] = color.x;
	buffer[4 * index + 1] = color.y;
	buffer[4 * index + 2] = color.z;
	buffer[4 * index + 3] = 255.0f;
}


float3 getColorFromIntersection(HitPoint intersection, image2d_t top_image, image2d_t side_image)
{
	float3 color = SKY_COLOR;
	if (intersection.normal.y) {
		color = convert_float3(read_imagei(top_image, tex_sampler, intersection.tex_coords).xyz);
	}
	else if (intersection.normal.x || intersection.normal.z) {
		color = convert_float3(read_imagei(side_image, tex_sampler, intersection.tex_coords).xyz);
	}

	//const float3 light_start = intersection.position + intersection.normal * NORMAL_EPS;
	//const HitPoint light_intersection = castRay(svo_data, light_start, )

	return color;
}


// Kernels
__kernel void albedo(
    __global Node* svo_data,
    __global float* albedo_result,
    //__global float* shadow_result,
	float3 position,
	__constant float* view_matrix,
	image2d_t top_image,
	image2d_t side_image,
	uint8_t mode,
	float time
)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const uint32_t index = gid.x + gid.y * get_global_size(0);
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 0.8f);

	const float3 d = normalize(multVec3Mat3(screen_position, view_matrix));

	HitPoint intersection = castRay(svo_data, position, d);

	float3 color = SKY_COLOR;
	if (intersection.hit) {
		if (intersection.water) {
			const float distortion_strength = 0.01f;
			float3 normal_distortion = (float3)(0.0f);
			float3 mirror_color = (float3)(1.0f);
			if (intersection.normal.y) {
				normal_distortion = (float3)(distortion_strength * sin((intersection.position.z + 0.01f * time) * 2400.0f), 0.0f, distortion_strength * sin((intersection.position.x + 0.01f * time) * 2400.0f));
				mirror_color = WATER_COLOR;
			}
			const float3 normal = normalize(convert_float3(intersection.normal) + normal_distortion);
			const float3 reflection_start = intersection.position + NORMAL_EPS * normal;
			const float3 reflection_d = reflect(d, normal);
			HitPoint reflection = castRay(svo_data, reflection_start, reflection_d);
			if (reflection.hit) {
				if (reflection.water) {
					float3 normal_distortion_2 = (float3)(0.0f);
					if (reflection.normal.y) {
						normal_distortion = (float3)(distortion_strength * sin((intersection.position.z + 0.1f * time) * 600.0f), 0.0f, distortion_strength * sin((intersection.position.x + 0.1f * time) * 600.0f));
					}
					const float3 normal_2 = normalize(convert_float3(reflection.normal) + normal_distortion);
					const float3 reflection_start_2 = reflection.position + NORMAL_EPS * normal_2;
					HitPoint reflection_2 = castRay(svo_data, reflection_start_2, reflect(reflection_d, normal_2));
					if (reflection_2.hit) {
						color =  mirror_color * WATER_COLOR * getColorFromIntersection(reflection_2, top_image, side_image);
					}
					else {
						color *= WATER_COLOR * WATER_COLOR;
					}
				}
				else {
					color = mirror_color * getColorFromIntersection(reflection, top_image, side_image);
				}
			}
			else {
				color *= WATER_COLOR;
			}
		}
		else {
			color = getColorFromIntersection(intersection, top_image, side_image);
		}
	}

	// Color output
	if (mode == 0) {
	    const uint16_t complexity = intersection.complexity > 255 ? 255 : intersection.complexity;
		color = convert_float3(complexity);
	}

    colorToResultBuffer(color, index, albedo_result);
}


__kernel void lighting(
    __global Node* svo_data,
    __global float* result,
	float3 position,
	__constant float* view_matrix,
	__global int32_t* rand_seed,
	float time,
	__global float* depth

) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const uint32_t index = gid.x + gid.y * get_global_size(0);
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 0.8f);

	const float time_su = 0.0f;
	const float time_of = -1.5f;
	const float light_radius = 6.0f;
	const float3 light_position = (float3)(light_radius * cos(time_su * time + time_of) + 1.5f, -1.0f, light_radius * sin(time_su * time + time_of) + 1.5f);

	const float3 d = normalize(multVec3Mat3(screen_position, view_matrix));

	const HitPoint intersection = castRay(svo_data, position, d);
	float3 color = (float3)255.0f;
	float light_intensity = SUN_INTENSITY;
	if (intersection.hit) {
		const float3 normal = normalize(convert_float3(intersection.normal));
		const float3 hit_start = intersection.position + NORMAL_EPS * normal;
		const float3 shadow_ray = normalize(light_position - hit_start);
		const HitPoint light_intersection = castRay(svo_data, hit_start, shadow_ray);

		if (light_intersection.hit) {
			light_intensity *= AMBIENT;
		}
		else {
			light_intensity *= fmax(AMBIENT, dot(shadow_ray, normal));
		}

		if (!intersection.water) {
			light_intensity *= fmax(0.3f, fmin(1.0f, get_ambient_occlusion(svo_data, hit_start, normal, rand_seed, index)));
		}
		
		color *= light_intensity;
	}

	const float conservation_coef = 0.5f;
	const float new_contribution_coef = 1.0f - conservation_coef;

	// Color output
	const float final_color = fmin(255.0f, color.x) * new_contribution_coef + result[4 * index ] * conservation_coef;
	result[4 * index + 0] = final_color;
	result[4 * index + 1] = final_color;
	result[4 * index + 2] = final_color;
	result[4 * index + 3] = 255;
}
