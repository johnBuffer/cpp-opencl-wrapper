typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
__constant sampler_t exact_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
__constant float EPS = 0x1.fffffep-1f;
__constant float NORMAL_EPS = 0.0078125f * 0.0078125f * 0.0078125f;
__constant float3 SKY_COLOR = (float3)(153.0f, 223.0f, 255.0f);
//constant float3 SKY_COLOR = (float3)(0.0f);
__constant float NEAR = 0.5f;
//__constant float GOLDEN_RATIO = 1.61803398875f;
__constant float G1 = 0.819172513f;
__constant float G2 = 0.671043607f;
__constant float G3 = 0.549700478f;
__constant float G4 = 0.450299522;

__constant float PI = 3.141592653f;
__constant uint32_t LEVELS[11] = {0u, 8u, 72u, 584u, 4680u, 37448u, 299592u, 2396744u, 19173960u, 153391688u, 1227133512u};
__constant float EMISSIVE_INTENSITY = 16.0f;
__constant float INV_255 = 1.0f / 255.0f;


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

typedef struct GiBounce
{
	HitPoint point;
	float3 acc;
	float intensity;
} GiBounce;


// Utils functions
float frac(float x)
{
	return fmin(x - floor(x), EPS);
}


float3 getRandomizedNormal(const float3 normal, image2d_t noise, float off)
{
	const int2 tex_coords = (int2)(get_global_id(0) % 512u, get_global_id(1) % 512u);
	const float3 noise_value = convert_float3(read_imagei(noise, exact_sampler, tex_coords).xyz) * INV_255;

	const float coord_1 = frac(noise_value.x + off * G1) - 0.5f;
	const float coord_2 = frac(noise_value.y + off * G2) - 0.5f;
	const float coord_3 = frac(noise_value.z + off * G3) * 0.5f;
	if (normal.x) {
		return normalize((float3)(coord_3 * normal.x, coord_1, coord_2));
	}
	if (normal.y) {
		return normalize((float3)(coord_1, coord_3 * normal.y, coord_2));
	}
	return normalize((float3)(coord_1, coord_2, coord_3 * normal.z));
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


float getSunContribution(global uint8_t* svo_data, const HitPoint point, const float3 light_position)
{
	const float3 ray_start = point.position + point.normal * NORMAL_EPS;
	const float3 ray_direction = normalize(light_position - ray_start);
	const HitPoint light_intersection = castRay(svo_data, ray_start, ray_direction);
	if (light_intersection.cell_type) {
		return 0.0f;
	}
	return max(0.0f, dot(point.normal, ray_direction));
}


GiBounce bounceOnce(global uint8_t* svo_data, global float3* blocks_data, const SceneSettings scene, image2d_t noise, float off, const HitPoint start)
{
	GiBounce result;
	result.acc = 0.0f;
	// Launch random ray from surface
	const float3 gi_start = start.position + NORMAL_EPS * start.normal;
	const float3 gi_direction = getRandomizedNormal(start.normal, noise, off);
	result.point = castRay(svo_data, gi_start, gi_direction);
	result.intensity = max(0.0f, dot(gi_direction, start.normal));
	if (result.point.cell_type) {
		// Check for sun
		result.acc = (scene.light_intensity * getSunContribution(svo_data, result.point, scene.light_position)) * blocks_data[result.point.cell_type];
		// Check for emissive
		//result.acc += (result.point.cell_type == 2) * EMISSIVE_INTENSITY * block_color;
	} else {
		// Sky contribution
		result.acc += (INV_255) * SKY_COLOR;
	}

	result.acc *= result.intensity;

	return result;
}


float3 getGlobalIllumination(global uint8_t* svo_data, global float3* blocks_data, const float3 position, const float3 normal, const SceneSettings scene, image2d_t noise, uint32_t frame_count)
{
	HitPoint start;
	start.position = position;
	start.normal = normal;

	const GiBounce bounce_1 = bounceOnce(svo_data, blocks_data, scene, noise, frame_count, start);
	float3 gi_acc = bounce_1.acc;
	// Eventually second bounce
	if (bounce_1.point.cell_type) {
		const GiBounce bounce_2 = bounceOnce(svo_data, blocks_data, scene, noise, frame_count * G4, bounce_1.point);
		gi_acc += 0.5f * bounce_1.intensity * bounce_2.acc;
	}
	
	return gi_acc;
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
	const float off = frame_count%10000;
	const float light_offset_1 = (frac(noise_value.x + G1 * off) - 0.5f);
	const float light_offset_2 = (frac(noise_value.y + G2 * off) - 0.5f);
	const float light_offset_3 = (frac(noise_value.z + G3 * off) - 0.5f);

	const float3 ray_start = position + normal * NORMAL_EPS;
	const float3 shadow_ray = normalize(scene.light_position + scene.light_radius * (float3)(light_offset_1, light_offset_2, light_offset_3) - position);
	const HitPoint light_intersection = castRay(svo_data, ray_start, shadow_ray);
	if (light_intersection.cell_type) {
		return 0.0f;
	}
	
	return fmin(1.0f, dot(normal, shadow_ray));
}


__kernel void lighting(
    global uint8_t* svo_data,
    global float3* blocks_data,
	read_only SceneSettings scene,
    write_only image2d_t result_gi,
    write_only image2d_t result_shadows,
	read_only image2d_t noise,
	uint32_t frame_count,
	read_only image2d_t depth,
	read_only image2d_t ss_position
)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));

	const float4 intersection = read_imagef(ss_position, exact_sampler, gid);
	if (intersection.w) {
		
		const float3 normal = normalFromNumber(intersection.w);
		const float3 gi_start = intersection.xyz + normal * NORMAL_EPS;
		const float3 gi = getGlobalIllumination(svo_data, blocks_data, gi_start, normal, scene, noise, frame_count);		
		const float light_intensity = getLightIntensity(svo_data, intersection.xyz, normal, scene, noise, frame_count);
		write_imagef(result_gi, gid, (float4)(gi, 0.0f));
		write_imagef(result_shadows, gid, (float4)(light_intensity));
	}
	else {
		write_imagef(result_gi, gid, (float4)(0.0f));
	}
}
