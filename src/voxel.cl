typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
__constant float3 light_position = (float3)(0.0f, 1.0f, 0.0f);
__constant float EPS = 0x1.fffffep-1f;


// Structs declaration
typedef struct Node
{
    uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
	uint16_t padding;
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
	uint16_t complexity;
} HitPoint;


// Utils functions
float3 mult_vec3_mat3(float3 v, __constant float* mat)
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

HitPoint cast_ray(__global Node* svo_data, float3 position, float3 d)
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
	float t_max = fmin(t_coef.x - t_offset.x, fmin(t_coef.y - t_offset.y, t_coef.z - t_offset.z));
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
		if ((child_mask & 1u) && t_min <= t_max) {
			const float tv_max = fmin(t_max, tc_max);
			const float half_scale = scale_f * 0.5f;
			const float3 t_half = half_scale * t_coef + t_corner;
			if (t_min <= tv_max) {
				const uint8_t leaf_mask = parent_ref.leaf_mask >> child_shift;
				// We hit a leaf
				if (leaf_mask & 1u) {
					result.hit = 1;
					// Could use mirror mask
					result.normal = -convert_char3(sign(d)) * (char3)(normal & 1u, normal & 2u, normal & 4u);

					result.distance = t_min;
					//result.position = position + t_min * d;
					
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

char3 getColorFromNormal(char3 normal)
{
	if (normal.x) {
		return (char3)(255, 0, 0);
	}
	if (normal.y) {
		return (char3)(0, 255, 0);
	}
	if (normal.z) {
		return (char3)(0, 0, 255);
	}
}


// Kernel's main
__kernel void raytracer(
    __global Node* svo_data,
    __global uint8_t* result,
	float3 position,
	__constant float* view_matrix,
	image2d_t top_image,
	image2d_t side_image
) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 1.0f);

	float3 d = normalize(mult_vec3_mat3(screen_position, view_matrix));

	HitPoint intersection = cast_ray(svo_data, position, d);

	float light_intensity = 1.0f;
	if (intersection.hit)
	{
		const float3 normal = normalize(convert_float3(intersection.normal));
		const float3 shadow_start = intersection.position + 0.125f * normal;
		const float3 shadow_ray   = normalize(light_position - shadow_start);
		const HitPoint light_intersection = cast_ray(svo_data, shadow_start, shadow_ray);
		if (light_intersection.hit) {
			light_intensity = 0.25f;
		}
		else {
			light_intensity = fmax(0.25f, dot(normal, shadow_ray));
		}
	}

	// Color output
	const unsigned int index = gid.x + gid.y * get_global_size(0);
	float3 color = (float3)0.0f;
	if (intersection.normal.y)
	{
		color = convert_float3(read_imagei(top_image, tex_sampler, intersection.tex_coords).xyz);
	}
	else if (intersection.normal.x || intersection.normal.z)
	{
		color = convert_float3(read_imagei(side_image, tex_sampler, intersection.tex_coords).xyz);
	}

	color *= light_intensity;

	const char3 final_color = convert_char3(color);
	result[4 * index + 0] = final_color.x;
	result[4 * index + 1] = final_color.y;
	result[4 * index + 2] = final_color.z;
	result[4 * index + 3] = 255;
}
