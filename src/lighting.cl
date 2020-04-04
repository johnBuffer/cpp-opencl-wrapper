typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant float NORMALIZER = 1.0f / 4294967296.0f;
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
__constant sampler_t noise_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
__constant float EPS = 0x1.fffffep-1f;
__constant float NORMAL_EPS = 0.0078125f * 0.0078125f * 0.0078125f;
__constant float AMBIENT = 0.0f;
__constant float SUN_INTENSITY = 10.0f;
//__constant float3 SKY_COLOR = (float3)(153.0f, 223.0f, 255.0f);
__constant float3 SKY_COLOR = (float3)(255.0f);
__constant float time_su = 0.0f;
__constant float NEAR = 0.5f;
__constant float GOLDEN_RATIO = 1.61803398875f;


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
	float3   normal;
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


float3 preMultVec3Mat3(float3 v, __constant float* mat)
{
	return (float3)(
		v.x * mat[0] + v.y * mat[3] + v.z * mat[6],
		v.x * mat[1] + v.y * mat[4] + v.z * mat[7],
		v.x * mat[2] + v.y * mat[5] + v.z * mat[8]
	);
}


float frac(float x)
{
	return fmin(x - floor(x), EPS);
}


float3 getRandomizedNormal(float3 normal, image2d_t noise, uint32_t frame_count)
{
	const int2 tex_coords = (int2)(get_global_id(0) % 512u, get_global_id(1) % 512u);
	const float3 noise_value = convert_float3(read_imagei(noise, noise_sampler, tex_coords).xyz) / 255.0f;

	const float range = 10.0f;
	const float coord_1 = range * (fmod(noise_value.x + GOLDEN_RATIO * ((frame_count) % 1000), 1.0f) - 0.5f);
	const float coord_2 = range * (fmod(noise_value.y + (GOLDEN_RATIO-0.1f) * ((frame_count) % 1000), 1.0f) - 0.5f);
	//const float coord_3 = range * (fmod(noise_value.y + 1.0f * GOLDEN_RATIO * (frame_count % 100), 1.0f));
	if (normal.x) {
		return normalize((float3)(normal.x, coord_1, coord_2));
	}
	
	if (normal.y) {
		return normalize((float3)(coord_1, normal.y, coord_2));
	}
	
	if (normal.z) {
		return normalize((float3)(coord_1, coord_2, normal.z));
	}

	return (float3)(0.0f);
}


float3 getColorFromIntersection(HitPoint intersection)
{
	const float x = 255.0f * (1.0f - (intersection.position.x - 1.0f));
	const float y = 255.0f * (1.0f - (intersection.position.y - 1.0f));
	const float z = 255.0f * (1.0f - (intersection.position.z - 1.0f));

	const float r = (uint8_t)(y) % 5 < 2 ? 1.0f : 0.0f;
	const float g = (uint8_t)(x) % 5 < 2 ? 1.0f : 0.0f;

	return (float3)(r, g, 1.0f - r);
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
		const uint8_t child_mask = (parent_ref.child_mask >> child_shift) & 1u;
		if (child_mask) {
			const float tv_max = fmin(t_max, tc_max);
			const float half_scale = scale_f * 0.5f;
			const float3 t_half = half_scale * t_coef + t_corner;
			const uint8_t leaf_mask = (parent_ref.leaf_mask >> child_shift) & 1u;
			const uint8_t watr_mask = (parent_ref.reflective_mask >> child_shift) & 1u;
			// We hit a leaf
			if (leaf_mask) {
				result.hit = 1u;
				// Could use mirror mask
				result.normal = -sign(d) * (float3)(normal & 1u, normal & 2u, normal & 4u);
				result.distance = t_min;
				result.water = watr_mask;

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

	result.distance = t_min;
	return result;
}


float3 getGlobalIllumination(__global Node* svo_data, const float3 position, const float3 normal, const float3 light_position, image2d_t noise, uint32_t frame_count)
{
	float3 gi_add = (float3)0.0f;
    // First bounce
    const float3 noise_normal = getRandomizedNormal(normal, noise, frame_count);
    const HitPoint gi_intersection = castRay(svo_data, position, noise_normal);
    if (gi_intersection.hit) {
        const float3 gi_normal = gi_intersection.normal;
        const float3 gi_light_start = gi_intersection.position + NORMAL_EPS * gi_normal;
        const float3 gi_light_direction = normalize(light_position - gi_light_start);
        const HitPoint gi_light_intersection = castRay(svo_data, gi_light_start, gi_light_direction);
        if (!gi_light_intersection.hit) {
            gi_add += 2.0f * getColorFromIntersection(gi_intersection);
        }
    } else {
        gi_add += 0.2f * SKY_COLOR / 255.0f;
    }
	
	return gi_add;
}


float2 projectPoint(const float3 in, const int2 screen_size)
{
	const float aspect_ratio = (screen_size.x) / (float)(screen_size.y);
    const float inv_z = 1.0f / in.z;
	const float2 offset = (float2)(0.5f + 0.5f / (float)screen_size.x, 0.5f + 0.5f / (float)screen_size.y);

	return (float2) ((NEAR * in.x * inv_z), (aspect_ratio * NEAR * in.y * inv_z)) + offset;
}


float3 getOldValue(image2d_t last_frame_color, __constant float* last_view_matrix, const float3 last_position, const float3 position, const int2 screen_size)
{
    const float3 last_view_pos = preMultVec3Mat3(position - last_position, last_view_matrix);
    const float2 last_screen_pos = projectPoint(last_view_pos, screen_size);

	const float4 last_color = read_imagef(last_frame_color, tex_sampler, last_screen_pos);
	/*if (fabs(length(last_view_pos) - last_color.w) < 0.001f) {
		return last_color.x;
	}*/
    
	return last_color.xyz;
	//return 0.0f;
}


__kernel void lighting(
    global Node* svo_data,
    write_only image2d_t result,
	float3 position,
	constant float* view_matrix,
	read_only image2d_t noise,
	float time,
	read_only image2d_t last_frame_color,
	constant float* last_view_matrix,
    float3 last_position,
    global float* depth,
	uint32_t frame_count
) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const uint32_t index = gid.x + gid.y * get_global_size(0);
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.y) / (float)(screen_size.x);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, (gid.y / (float)screen_size.y - 0.5f) * screen_ratio, NEAR);

	const float time_of = -1.5f;
	const float light_radius = 6.0f;
	const float3 light_position = (float3)(light_radius * cos(time_su * time + time_of) + 1.5f, -1.0f, light_radius * sin(time_su * time + time_of) + 1.5f);

	const float3 d = normalize(multVec3Mat3(screen_position, view_matrix));

	float3 color = 1.0f;

	const HitPoint intersection = castRay(svo_data, position, d);
	if (intersection.hit) {
		if (!intersection.water) {
			const float3 gi_start = intersection.position + intersection.normal * NORMAL_EPS;
			const float3 gi = getGlobalIllumination(svo_data, gi_start, intersection.normal, light_position, noise, frame_count);
            // Accumulation
            const float conservation_coef = 0.0f;
            const float new_contribution_coef = 1.0f - conservation_coef;
            const float3 old = getOldValue(last_frame_color, last_view_matrix, last_position, intersection.position, screen_size);
            color = gi * new_contribution_coef + old * conservation_coef;

            depth[2 * index + 0] = fabs(intersection.normal.x + 2.0f * intersection.normal.y + 3.0f * intersection.normal.z);
            depth[2 * index + 1] = intersection.distance;
		}
	} else {
        depth[index] = 4.0f;
    }

	const float4 out_color = (float4)(color, intersection.distance);
	/*const float3 noise_value = convert_float3(read_imagei(noise, noise_sampler, (int2)(gid.x % 512, gid.y % 512)).xyz);
	const float4 out_color = (float4)(noise_value.x + fmod(noise_value.x + GOLDEN_RATIO * (frame_count % 100), 1.0f),
	                                  noise_value.y + fmod(noise_value.x + GOLDEN_RATIO * (frame_count % 100), 1.0f),
									  noise_value.z + fmod(noise_value.x + GOLDEN_RATIO * (frame_count % 100), 1.0f), 255);*/
	write_imagef(result, gid, out_color);
}