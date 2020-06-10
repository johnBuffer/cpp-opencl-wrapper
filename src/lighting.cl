typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant float NORMALIZER = 1.0f / 4294967296.0f;
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
__constant sampler_t exact_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
__constant float EPS = 0x1.fffffep-1f;
__constant float NORMAL_EPS = 0.0078125f * 0.0078125f * 0.0078125f;
__constant float3 SKY_COLOR = (float3)(153.0f, 223.0f, 255.0f);
//constant float3 SKY_COLOR = (float3)(0.0f);
__constant float time_su = 0.0f;
__constant float NEAR = 0.5f;
__constant float GOLDEN_RATIO = 1.61803398875f;
__constant float G = 1.0f / 1.22074408460575947536f;
__constant float PI = 3.141592653f;
__constant float ACC_COUNT = 32.0f;


// Structs declaration
typedef struct Node
{
    uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
	uint8_t  reflective_mask;
	uint8_t  emissive;
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
	bool     emissive;
	uint16_t complexity;
} HitPoint;

typedef struct  __attribute__ ((packed)) _SceneSettings
{
	float3 camera_position;
	float3 light_position;
	float light_intensity;
	float light_radius;
	float time;
} SceneSettings;


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
	const float3 noise_value = convert_float3(read_imagei(noise, exact_sampler, tex_coords).xyz) / 255.0f;

	const float coord_1 = (fmod(noise_value.x + G       * (frame_count % 1000), 1.0f) - 0.5f);
	const float coord_2 = (fmod(noise_value.y + (G*G)   * (frame_count % 1000), 1.0f) - 0.5f);
	const float coord_3 = (fmod(noise_value.z + (G*G*G) * (frame_count % 1000), 0.5f));
	if (normal.x) {
		return normalize((float3)(sign(normal.x) * coord_3, coord_1, coord_2));
	}
	
	if (normal.y) {
		return normalize((float3)(coord_1, sign(normal.y) * coord_3, coord_2));
	}
	
	if (normal.z) {
		return normalize((float3)(coord_1, coord_2, sign(normal.z) * coord_3));
	}

	return (float3)(0.0f);
}


// Raytracing functions
HitPoint castRay(__global Node* svo_data, float3 position, float3 d, float max_dist, float ray_coef, float ray_bias)
{
	HitPoint result;
	result.hit = 0;
	result.complexity = 0;

	const float EPS = 1.0f / (float)(1 << SVO_MAX_DEPTH);
	
	// Initialize stack
	OctreeStack stack[11];
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
	while (scale < SVO_MAX_DEPTH && t_min < max_dist) {
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
			if (leaf_mask || tc_max * ray_coef + ray_bias >= scale_f) {
				result.hit = 1u;
				// Could use mirror mask
				result.normal = -sign(d) * (float3)(normal & 1u, normal & 2u, normal & 4u);
				result.distance = t_min;
				result.water = watr_mask;
				result.emissive = (parent_ref.emissive >> child_shift) & 1u;

				if ((mirror_mask & 1) == 0) pos.x = 3.0f - scale_f - pos.x;
				if ((mirror_mask & 2) == 0) pos.y = 3.0f - scale_f - pos.y;
				if ((mirror_mask & 4) == 0) pos.z = 3.0f - scale_f - pos.z;
				result.position = fmin(fmax(position + t_min * d, pos + (float3)EPS), pos + (float3)(scale_f - EPS));
				
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
				stack[scale-12].parent_index = parent_id;
				stack[scale-12].t_max = t_max;
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
			const OctreeStack entry = stack[scale-12];
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

float3 getGlobalIllumination(__global Node* svo_data, const float3 position, const float3 normal, const float3 light_position, const float light_intensity, image2d_t noise, uint32_t frame_count)
{
	float3 gi_add = (float3)0.0f;
    // First bounce
    const float3 noise_normal = getRandomizedNormal(normal, noise, frame_count);
    const HitPoint gi_intersection = castRay(svo_data, position, noise_normal, 2.0f, 0.2f, 0.0f);
    if (gi_intersection.hit) {
		const float3 gi_normal = gi_intersection.normal;
		const float3 gi_light_start = gi_intersection.position + NORMAL_EPS * gi_normal;
		const float3 gi_light_direction = normalize(light_position - gi_light_start);
		const HitPoint gi_light_intersection = castRay(svo_data, gi_light_start, gi_light_direction, 2.0f, 0.2f, 0.0f);
		if (!gi_light_intersection.hit) {
			gi_add += 0.5f * fmax(0.0f, light_intensity * dot(gi_light_direction, gi_normal));
		}
    } else if (noise_normal.y < 0.0f) {
        gi_add += SKY_COLOR / 255.0f;
    }
	
	return gi_add / PI;
}

float2 projectPoint(const float3 in)
{
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float aspect_ratio = (screen_size.x) / (float)(screen_size.y);
    const float inv_z = 1.0f / in.z;
	const float2 offset = (float2)(0.5f + 0.5f / (float)screen_size.x, 0.5f + 0.5f / (float)screen_size.y);

	return (float2) ((NEAR * in.x * inv_z), (aspect_ratio * NEAR * in.y * inv_z)) + offset;
}


float3 normalFromNumber(const float number)
{
	const int32_t n = fabs(number);
	return sign(number) * (float3)(n & 1, (n >> 1u) & 1, (n >> 2u) & 1);
}


float4 getOldValue(image2d_t last_frame_color, image2d_t last_frame_depth, __constant float* last_view_matrix, const float3 last_position, const float4 intersection)
{
    const float3 last_view_pos = preMultVec3Mat3(intersection.xyz - last_position, last_view_matrix);
    const float2 last_screen_pos = projectPoint(last_view_pos);

	const float4 last_color = read_imagef(last_frame_color, tex_sampler, last_screen_pos);
	const float2 last_depth = read_imagef(last_frame_depth, tex_sampler, last_screen_pos).xy;
	float acc = 0.0f;

	const float accuracy_threshold = 0.05f;
	if (fabs(1.0f - length(last_view_pos) / last_depth.x) < accuracy_threshold && last_depth.y == intersection.w) {
		return last_color;
	}
	
	return (float4)(0.0f);
}


float getLightIntensity(__global Node* svo_data, const float3 position, const float3 normal, const float3 light_position, const float light_radius, const float light_intensity, image2d_t noise, uint32_t frame_count)
{
	const int2 tex_coords = (int2)(get_global_id(0) % 512u, get_global_id(1) % 512u);
	const float3 noise_value = convert_float3(read_imagei(noise, exact_sampler, tex_coords).xyz) / 255.0f;

	const float3 ray_start = position + normal * NORMAL_EPS;

	const float light_offset_1 = (fmod(noise_value.x + G         * (frame_count%10000), 1.0f) - 0.5f);
	const float light_offset_2 = (fmod(noise_value.y + G * G     * (frame_count%10000), 1.0f) - 0.5f);
	const float light_offset_3 = (fmod(noise_value.z + G * G * G * (frame_count%10000), 1.0f) - 0.5f);
	const float3 shadow_ray = normalize(light_position + light_radius * (float3)(light_offset_1, light_offset_2, light_offset_3) - position);
	const HitPoint light_intersection = castRay(svo_data, ray_start, shadow_ray, 2.0f, 0.02f, 0.0f);
	if (light_intersection.hit) {
		return 0.0f;
	}
	
	return light_intensity * fmin(1.0f, dot(normal, shadow_ray));
}


__kernel void lighting(
    global Node* svo_data
	, read_only SceneSettings scene
    , write_only image2d_t result
	, read_only image2d_t noise
	, read_only image2d_t last_frame_color
	, constant float* last_view_matrix
    , float3 last_position
	, uint32_t frame_count
	, read_only image2d_t depth
	, read_only image2d_t last_depth
	, read_only image2d_t screen_space_positions
) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	const float3 light_position = scene.light_position;

	float3 color = 1.0f;
	float acc = 1.0f;

	const float4 intersection = read_imagef(screen_space_positions, exact_sampler, gid);
	if (intersection.w) {
		const float3 normal = normalFromNumber(intersection.w);
		const float3 gi_start = intersection.xyz + normal * NORMAL_EPS;

		const float4 old_gi = getOldValue(last_frame_color, last_depth, last_view_matrix, last_position, intersection);
		const float3 new_gi = getGlobalIllumination(svo_data, gi_start, normal, light_position, scene.light_intensity, noise, frame_count);
		const float light_intensity = getLightIntensity(svo_data, intersection.xyz, normal, light_position, scene.light_radius, scene.light_intensity, noise, frame_count);

		acc = old_gi.w + 1.0f;
		if (acc < ACC_COUNT) {
			color = ((float3)(light_intensity) + new_gi) + old_gi.xyz;
		} else {
			const float conservation_coef = 1.0f - 1.0f / ACC_COUNT;
			acc = ACC_COUNT;
			color = ((float3)(light_intensity) + new_gi) + old_gi.xyz * conservation_coef;
		}
		//acc = 1.0f;
		//color = (float3)(light_intensity) + new_gi;
	}

	const float4 out_color = (float4)(color, acc);

	write_imagef(result, gid, out_color);
}

__kernel void normalizer(
	read_only image2d_t input,
    write_only image2d_t output
) 
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const float4 in_color = read_imagef(input, exact_sampler, gid);
	write_imagef(output, gid, (float4)(fmax(0.0f, in_color.xyz / in_color.w), in_color.w));
}
