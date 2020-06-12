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
__constant float ACC_COUNT = 64.0f;
__constant uint32_t LEVELS[11] = {0u, 8u, 72u, 584u, 4680u, 37448u, 299592u, 2396744u, 19173960u, 153391688u, 1227133512u};


typedef struct OctreeStack
{
	uint32_t parent_index;
	uint32_t parent_level_index;
	float t_max;
} OctreeStack;

typedef struct __attribute__ ((packed)) _HitPoint
{
	float3   position;
	float3   normal;
	float    distance;
	uint8_t  cell_type;
	float2   tex_coords;
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


float3 getRandomizedNormal(const float3 normal, image2d_t noise, uint32_t frame_count)
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
HitPoint castRay(global uint8_t* svo_data, float3 position, float3 d)
{
	HitPoint result;
	result.cell_type = 0;

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
	uint32_t node_id = 0u;
	uint32_t parent_id = 0u;
	uint8_t child_offset = 0u;
	int8_t scale = SVO_MAX_DEPTH - 1u;
	
	uint32_t level_index = 0u;

	float3 pos = (float3)(1.0f);
	float scale_f = 0.5f;
	// Initialize child position
	if (1.5f * t_coef.x - t_offset.x > t_min) { child_offset ^= 1u, pos.x = 1.5f; }
	if (1.5f * t_coef.y - t_offset.y > t_min) { child_offset ^= 2u, pos.y = 1.5f; }
	if (1.5f * t_coef.z - t_offset.z > t_min) { child_offset ^= 4u, pos.z = 1.5f; }
	uint8_t normal = 0u;
	
	// Explore octree
	while (scale < SVO_MAX_DEPTH) {
		const uint8_t node = svo_data[node_id];
		// Compute new T span
		const float3 t_corner = (float3)(pos.x * t_coef.x - t_offset.x, pos.y * t_coef.y - t_offset.y, pos.z * t_coef.z - t_offset.z);
		const float tc_max = fmin(t_corner.x, fmin(t_corner.y, t_corner.z));
		// Check if child exists here
		const uint8_t child_shift = child_offset ^ mirror_mask;
		const uint8_t child_mask = (node >> child_shift) & 1u;
		if (child_mask) {
			const float tv_max = fmin(t_max, tc_max);
			const float half_scale = scale_f * 0.5f;
			const float3 t_half = half_scale * t_coef + t_corner;
			// We hit a leaf
			if (scale == SVO_MAX_DEPTH - 9) {
				// Could use mirror mask
				result.normal = -sign(d) * (float3)(normal & 1u, (normal>>1u) & 1u, (normal>>2u) & 1u);
				result.distance = t_min;
				
				const uint32_t current_index = parent_id * 8 + child_shift;
				node_id = LEVELS[SVO_MAX_DEPTH - scale - 1] + current_index + 1;
				result.cell_type = svo_data[node_id];

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
				stack[scale-12].parent_index = node_id;
				stack[scale-12].parent_level_index = parent_id;
				stack[scale-12].t_max = t_max;
			}
			h = tc_max;
			// Update current voxel
			const uint32_t current_index = parent_id * 8 + child_shift;
			parent_id = current_index;
			node_id = LEVELS[SVO_MAX_DEPTH - scale - 1] + current_index + 1;
			child_offset = 0u;
			--scale;
			// Need to fix LEVELS (+1 everywhere)
			scale_f = half_scale;
			if (t_half.x > t_min) { child_offset ^= 1u, pos.x += scale_f; }
			if (t_half.y > t_min) { child_offset ^= 2u, pos.y += scale_f; }
			if (t_half.z > t_min) { child_offset ^= 4u, pos.z += scale_f; }
			t_max = tv_max;
			continue;
			
		} // End of depth exploration

		// Maybe remove the ifs ?
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
			node_id = entry.parent_index;
			parent_id = entry.parent_level_index;
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


float isInShadow(global uint8_t* svo_data, const HitPoint point, const float3 light_position)
{
	const float3 ray_start = point.position + point.normal * NORMAL_EPS;
	const float3 ray_direction = normalize(light_position - ray_start);
	const HitPoint light_intersection = castRay(svo_data, ray_start, ray_direction);
	if (light_intersection.cell_type) {
		return 0.0f;
	}
	return max(0.0f, dot(point.normal, ray_direction));
}


float3 getGlobalIllumination(global uint8_t* svo_data, const float3 position, const float3 normal, const float3 light_position, const float light_intensity, image2d_t noise, uint32_t frame_count)
{
	float3 gi_add = (float3)0.0f;
    // First bounce
    float3 noise_normal = getRandomizedNormal(normal, noise, frame_count);
    HitPoint gi_intersection = castRay(svo_data, position, noise_normal);
    if (gi_intersection.cell_type) {
		if (gi_intersection.cell_type == 2) {
			gi_add += 16.0f;
		}
		else {
			// Sun contribution
			gi_add += light_intensity * isInShadow(svo_data, gi_intersection, light_position);
			float3 gi_normal    = gi_intersection.normal;
			float3 gi_start     = gi_intersection.position + NORMAL_EPS * gi_normal;
			float3 gi_direction = getRandomizedNormal(gi_normal, noise, frame_count);
			gi_intersection = castRay(svo_data, gi_start, gi_direction);
			if (gi_intersection.cell_type) {
				if (gi_intersection.cell_type == 2) {
					gi_add += 16.0f;
				} /*else {
					gi_add += 0.5f * light_intensity * isInShadow(svo_data, gi_intersection, light_position);
					gi_normal = gi_intersection.normal;
					gi_direction = getRandomizedNormal(gi_normal, noise, frame_count);
					gi_start = gi_intersection.position + NORMAL_EPS * gi_normal;
					gi_intersection = castRay(svo_data, gi_start, gi_direction);
					if (!gi_intersection.cell_type) {
						gi_add += SKY_COLOR / 255.0f;
					}
				}*/
			} else {
				gi_add += SKY_COLOR / 255.0f;
			}
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


float getLightIntensity(global uint8_t* svo_data, const float3 position, const float3 normal, const SceneSettings scene, image2d_t noise, uint32_t frame_count)
{
	const int2 tex_coords = (int2)(get_global_id(0) % 512u, get_global_id(1) % 512u);
	const float3 noise_value = convert_float3(read_imagei(noise, exact_sampler, tex_coords).xyz) / 255.0f;
	const float light_offset_1 = (fmod(noise_value.x + G         * (frame_count%10000), 1.0f) - 0.5f);
	const float light_offset_2 = (fmod(noise_value.y + G * G     * (frame_count%10000), 1.0f) - 0.5f);
	const float light_offset_3 = (fmod(noise_value.z + G * G * G * (frame_count%10000), 1.0f) - 0.5f);

	const float3 ray_start = position + normal * NORMAL_EPS;
	const float3 shadow_ray = normalize(scene.light_position + scene.light_radius * (float3)(light_offset_1, light_offset_2, light_offset_3) - position);
	const HitPoint light_intersection = castRay(svo_data, ray_start, shadow_ray);
	if (light_intersection.cell_type) {
		return 0.0f;
	}
	
	return fmin(1.0f, dot(normal, shadow_ray));
}


__kernel void lighting(
    global uint8_t* svo_data
	, read_only SceneSettings scene
    , write_only image2d_t result_gi
    , write_only image2d_t result_shadows
	, read_only image2d_t noise
	, uint32_t frame_count
	, read_only image2d_t depth
	, read_only image2d_t ss_position
)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));

	const float4 intersection = read_imagef(ss_position, exact_sampler, gid);
	if (intersection.w) {
		const float3 normal = normalFromNumber(intersection.w);
		const float3 gi_start = intersection.xyz + normal * NORMAL_EPS;
		const float3 gi = getGlobalIllumination(svo_data, gi_start, normal, scene.light_position, scene.light_intensity, noise, frame_count);
		const float light_intensity = getLightIntensity(svo_data, intersection.xyz, normal, scene, noise, frame_count);
		write_imagef(result_gi, gid, (float4)(gi, 0.0f));
		write_imagef(result_shadows, gid, (float4)(light_intensity));
	}
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
