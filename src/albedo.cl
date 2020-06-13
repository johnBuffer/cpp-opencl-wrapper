typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;

// Const values
__constant float NORMALIZER = 1.0f / 4294967296.0f;
__constant uint8_t SVO_MAX_DEPTH = 23u;
__constant sampler_t tex_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
//__constant float3 light_position = (float3)(0.0f, 1.0f, 0.0f);
__constant float EPS = 0x1.fffffep-1f;
__constant float NORMAL_EPS = 0.0078125f * 0.0078125f * 0.0078125f;
__constant float AMBIENT = 0.2f;
__constant float3 SKY_COLOR = (float3)(153.0f, 223.0f, 255.0f);
//__constant float3 SKY_COLOR = (float3)(41.0f, 66.0f, 107.0f);
//__constant float3 SKY_COLOR = (float3)(0.0f);
//__constant float3 WATER_COLOR = (float3)(1.0f);
__constant float3 WATER_COLOR = (float3)(28.0f / 255.0f, 194.0f / 255.0f, 255.0f / 255.0f);
//__constant float3 WATER_COLOR = (float3)(128.0f / 255.0f, 194.0f / 255.0f, 255.0f / 255.0f);
__constant float R0 = 0.0204f;
__constant float time_su = 0.0f;
__constant uint32_t LEVELS[11] = {0u, 8u, 72u, 584u, 4680u, 37448u, 299592u, 2396744u, 19173960u, 153391688u, 1227133512u};


// Structs declaration
typedef uint8_t Node;

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

float frac(float x)
{
	return fmin(x - floor(x), EPS);
}

// Raytracing functions
HitPoint castRay(global Node* svo_data, float3 position, float3 d, bool in_water)
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

void colorToResultBuffer(float3 color, uint32_t index, __global float* buffer)
{
    buffer[4 * index + 0] = color.x;
	buffer[4 * index + 1] = color.y;
	buffer[4 * index + 2] = color.z;
	buffer[4 * index + 3] = 255.0f;
}

float3 getColorFromIntersection(HitPoint intersection, global float3* blocks_data)
{
	// Uncomment to enable textures
	//const float scale = 256.0f;
	//const float color = convert_float3(read_imagei(top_image, tex_sampler, intersection.tex_coords).x); 
	return 255.0f * blocks_data[intersection.cell_type].xyz;

	// Uncomment to disable textures
	//return intersection.complexity;
}

float3 getColorAndLightFromIntersection(HitPoint intersection, global float3* blocks_data, __global Node* svo_data, float3 light_position, bool under_water)
{
	const float3 ray_start = intersection.position + intersection.normal * NORMAL_EPS;

	const float3 shadow_ray = normalize(light_position - ray_start);
	const HitPoint light_intersection = castRay(svo_data, ray_start, shadow_ray, under_water);
	float light_intensity = AMBIENT;
	if (!light_intersection.cell_type) {
		light_intensity = 1.0f;//max(0.0f, dot(intersection.normal, shadow_ray));
	}
	
	return light_intensity * getColorFromIntersection(intersection, blocks_data); 
}

float normalToNumber(const float3 normal)
{
	return normal.x + 2.0f * normal.y + 4.0f * normal.z;
}


// Kernels
__kernel void albedo(
    global uint8_t* svo_data,
    global float3* blocks_data,
	SceneSettings scene,
    write_only image2d_t albedo_result,
	constant float* view_matrix,
	image2d_t top_image,
	image2d_t side_image,
	write_only image2d_t depth,
	write_only image2d_t ss_position
)
{
	// Initialization
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
	const uint32_t index = gid.x + gid.y * get_global_size(0);
	const int2 screen_size = (int2)(get_global_size(0), get_global_size(1));
	const float screen_ratio = (float)(screen_size.x) / (float)(screen_size.y);
	const float3 screen_position = (float3)(gid.x / (float)screen_size.x - 0.5f, gid.y / (float)screen_size.x - 0.5f / screen_ratio, 0.5f);
	// Light
	const float current_time_value = scene.time * time_su;
	// Cast ray
	const float3 d = normalize(multVec3Mat3(screen_position, view_matrix));
	const HitPoint intersection = castRay(svo_data, scene.camera_position, d, false);
	// Result
	float3 color = 0.5f * SKY_COLOR;
	if (intersection.cell_type) {		
		color = getColorFromIntersection(intersection, blocks_data);
		write_imagef(ss_position, gid, (float4)(intersection.position, normalToNumber(intersection.normal)));
		write_imagef(depth, gid, (float4)(intersection.distance, normalToNumber(intersection.normal), intersection.cell_type, 0.0f));
	}
	else {
		write_imagef(ss_position, gid, (float4)(0.0f));
		const float to_sun_intensity = dot(d, normalize(scene.light_position - scene.camera_position));
		const float primary_intensity = pow(to_sun_intensity, 511.0f);
		const float secondary_intensity = 0.2f * pow(to_sun_intensity, 1.0f);
		color = max((float3)(24.0f, 59.0f, 75.0f), min((float3)(255.0f * (primary_intensity + secondary_intensity)) + color, (float3)(255.0f)));
	}

	write_imagef(albedo_result, gid, (float4)(color, 0.0f));
}
