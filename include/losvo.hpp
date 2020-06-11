#pragma once

#include <vector>
#include "volumetric.hpp"
#include <algorithm>
#include "utils.hpp"

#pragma pack(push, 1)
struct Mutation
{
	cl_uchar  needed;
	cl_uchar  value;
	cl_uint   node_id;

	Mutation()
		: needed(0)
		, value(0)
		, node_id(0)
	{}

	Mutation(cl_uchar needed_, cl_uchar value_, cl_uint node_id_)
		: needed(needed_)
		, value(value_)
		, node_id(node_id_)
	{}
};
#pragma pack(pop)


struct Losvo
{
	Losvo(const uint32_t levels_count)
		: levels(levels_count)
		, world_size(as<float>(1 << levels_count))
	{
		// Last level is for leafs
		const uint64_t nodes_count = static_cast<uint64_t>(std::pow(8, levels_count + 1) - 1) / 7;
		node_data.resize(nodes_count);

		for (uint8_t& node : node_data) {
			node = 0;
		}

		std::cout << "Allocation done (" << node_data.size() << " Nodes)" << std::endl;
	}

	void setCell(uint32_t x, uint32_t y, uint32_t z, uint8_t cell_type) {
		// Value == 0 is empty
		if (!cell_type) {
			setEmpty(x, y, z);
		}
		else {
			setType(x, y, z, cell_type);
		}
	}

	void setEmpty(uint32_t x, uint32_t y, uint32_t z) {
		
	}

	static uint8_t getSubIndex(const glm::uvec3& v) {
		return v.x + v.y * 2u + v.z * 4u;
	}

	void setType(uint32_t x, uint32_t y, uint32_t z, uint8_t cell_type) {
		uint32_t side_size = static_cast<uint32_t>(std::pow(2, levels));
		// Update hierarchy
		glm::uvec3 current_position(x, y, z);
		uint32_t level_index = 0;
		uint32_t parent_index = 0;
		uint8_t parent_sub_index = 0;

		for (uint32_t current_level(0); current_level < levels; ++current_level) {
			// Level consts
			const uint32_t level_size = 1 << (3 * current_level);
			uint32_t half_size = side_size >> 1u;
			const glm::uvec3 sub_position   = current_position / half_size;
			const uint8_t sub_index = getSubIndex(sub_position);
			// Checking current node
			const uint32_t current_index = parent_index * 8 + parent_sub_index;
			const uint32_t node_index = level_index + current_index;
			node_data[node_index] |= (1 << sub_index);
			parent_index = current_index;
			parent_sub_index = sub_index;
			level_index += level_size;
			// Updating for next step
			current_position -= sub_position * half_size;
			side_size = half_size;
		}
		const glm::uvec3 level_position = current_position / side_size;
		const uint32_t leaf_computed_index = level_index + parent_index * 8 + parent_sub_index;
		// Set the actual leaf
		node_data[leaf_computed_index] = cell_type;
	}

	std::vector<Mutation> addCell(uint32_t x, uint32_t y, uint32_t z, uint8_t cell_type) {
		std::vector<Mutation> mutations(levels + 1);
		for (Mutation& m : mutations) {
			m.needed = false;
		}

		uint32_t side_size = static_cast<uint32_t>(std::pow(2, levels));
		// Update hierarchy
		glm::uvec3 current_position(x, y, z);
		uint32_t level_index = 0;
		uint32_t parent_index = 0;
		uint8_t parent_sub_index = 0;

		for (uint32_t current_level(0); current_level < levels; ++current_level) {
			// Level consts
			const uint32_t level_size = 1 << (3 * current_level);
			uint32_t half_size = side_size >> 1u;
			const glm::uvec3 sub_position = current_position / half_size;
			const uint8_t sub_index = getSubIndex(sub_position);
			// Checking current node
			const uint32_t current_index = parent_index * 8 + parent_sub_index;
			const uint32_t node_index = level_index + current_index;

			uint8_t& current_node = node_data[node_index];
			// If the current node doesn't have sub where we need it
			if (!((current_node >> sub_index) & 1)) {
				// Add mutation
				current_node |= (1 << sub_index);
				mutations[current_level] = Mutation(true, current_node, node_index);
			}
			
			parent_index = current_index;
			parent_sub_index = sub_index;
			level_index += level_size;
			// Updating for next step
			current_position -= sub_position * half_size;
			side_size = half_size;
		}
		const glm::uvec3 level_position = current_position / side_size;
		const uint32_t leaf_computed_index = level_index + parent_index * 8 + parent_sub_index;
		// Set the actual leaf
		node_data[leaf_computed_index] = cell_type;
		mutations[levels] = Mutation(true, cell_type, leaf_computed_index);

		return mutations;
	}

	std::vector<Mutation> removeCell(uint32_t x, uint32_t y, uint32_t z) {
		std::vector<Mutation> mutations(levels + 1);
		std::vector<uint8_t> sub_indexes(levels + 1);
		for (Mutation& m : mutations) {
			m.needed = false;
		}

		uint32_t side_size = static_cast<uint32_t>(std::pow(2, levels));
		// Update hierarchy
		glm::uvec3 current_position(x, y, z);
		uint32_t level_index = 0;
		uint32_t parent_index = 0;
		uint8_t parent_sub_index = 0;

		for (uint32_t current_level(0); current_level < levels; ++current_level) {
			// Level consts
			const uint32_t level_size = 1 << (3 * current_level);
			uint32_t half_size = side_size >> 1u;
			const glm::uvec3 sub_position = current_position / half_size;
			const uint8_t sub_index = getSubIndex(sub_position);
			// Checking current node
			const uint32_t current_index = parent_index * 8 + parent_sub_index;
			const uint32_t node_index = level_index + current_index;

			uint8_t& current_node = node_data[node_index];
			// If the current node doesn't have sub where we need it
			// Add mutation
			
			mutations[current_level] = Mutation(false, current_node, node_index);
			sub_indexes[current_level] = sub_index;

			parent_index = current_index;
			parent_sub_index = sub_index;
			level_index += level_size;
			// Updating for next step
			current_position -= sub_position * half_size;
			side_size = half_size;
		}

		const glm::uvec3 level_position = current_position / side_size;
		const uint32_t leaf_computed_index = level_index + parent_index * 8 + parent_sub_index;
		// Set the actual leaf
		node_data[leaf_computed_index] = 0;
		mutations[levels] = Mutation(true, 0, leaf_computed_index);

		// Remove empty cells
		for (uint8_t i(levels); i--;) {
			std::cout << "Level " << int(i) << " " << int(mutations[i].value) << " prev val " << int(mutations[i + 1].value) << std::endl;
			if (mutations[i + 1].needed && !(mutations[i + 1].value)) {
				std::cout << "Previous node now empty, updating..." << std::endl;
				mutations[i].needed = true;
				node_data[mutations[i].node_id] ^= (1 << sub_indexes[i]);
				mutations[i].value = node_data[mutations[i].node_id];
			}
			else {
				break;
			}
		}

		return mutations;
	}

	HitPoint castRay(glm::vec3 world_space_position, glm::vec3 d) const
	{
		const uint32_t LEVELS[11] = { 0u, 8u, 72u, 584u, 4680u, 37448u, 299592u, 2396744u, 19173960u, 153391688u, 1227133512u };
		glm::vec3 position = world_space_position / world_size + glm::vec3(1.0f);
		HitPoint result;
		// Const values
		constexpr uint8_t SVO_MAX_DEPTH = 23u;
		constexpr float EPS = 1.0f / float(1 << SVO_MAX_DEPTH);
		// Initialize stack
		OctreeStack stack[23];
		// Check octant mask and modify ray accordingly
		if (std::abs(d.x) < EPS) { d.x = copysign(EPS, d.x); }
		if (std::abs(d.y) < EPS) { d.y = copysign(EPS, d.y); }
		if (std::abs(d.z) < EPS) { d.z = copysign(EPS, d.z); }
		const glm::vec3 t_coef = -1.0f / glm::abs(d);
		glm::vec3 t_offset = position * t_coef;
		uint8_t mirror_mask = 7u;
		if (d.x > 0.0f) { mirror_mask ^= 1u, t_offset.x = 3.0f * t_coef.x - t_offset.x; }
		if (d.y > 0.0f) { mirror_mask ^= 2u, t_offset.y = 3.0f * t_coef.y - t_offset.y; }
		if (d.z > 0.0f) { mirror_mask ^= 4u, t_offset.z = 3.0f * t_coef.z - t_offset.z; }
		// Initialize t_span
		float t_min = std::max(2.0f * t_coef.x - t_offset.x, std::max(2.0f * t_coef.y - t_offset.y, 2.0f * t_coef.z - t_offset.z));
		float t_max = std::min(t_coef.x - t_offset.x, std::min(t_coef.y - t_offset.y, t_coef.z - t_offset.z));
		float h = t_max;
		t_min = std::max(0.0f, t_min);
		t_max = std::min(1.0f, t_max);
		// Init current voxel
		uint32_t node_id = 0u;
		uint32_t parent_id = 0u;
		uint8_t child_offset = 0u;
		int8_t scale = SVO_MAX_DEPTH - 1u;
		glm::vec3 pos(1.0f);
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
			const uint8_t node = node_data[node_id];
			// Compute new T span
			const glm::vec3 t_corner(pos.x * t_coef.x - t_offset.x, pos.y * t_coef.y - t_offset.y, pos.z * t_coef.z - t_offset.z);
			const float tc_max = std::min(t_corner.x, std::min(t_corner.y, t_corner.z));
			// Check if child exists here
			const uint8_t child_shift = child_offset ^ mirror_mask;
			const uint8_t child_mask = node >> child_shift;
			if (child_mask & 1u) {
				const float tv_max = std::min(t_max, tc_max);
				const float half = scale_f * 0.5f;
				const glm::vec3 t_half = half * t_coef + t_corner;
				// We hit a leaf
				if (scale == SVO_MAX_DEPTH - levels) {
					result.hit = true;
					result.global_index = node_id;
					result.child_index = child_shift;
					break;
				}
				// Eventually add parent to the stack
				if (tc_max < h) {
					stack[scale].node_index = node_id;
					stack[scale].parent_index = parent_id;
					stack[scale].t_max = t_max;
				}
				h = tc_max;
				// Update current voxel
				const uint32_t current_index = parent_id * 8 + child_shift;
				parent_id = current_index;
				node_id = LEVELS[SVO_MAX_DEPTH - scale - 1] + current_index + 1;
				child_offset = 0u;
				--scale;
				scale_f = half;
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
				const int32_t ipos_x = floatAsInt(pos.x);
				const int32_t ipos_y = floatAsInt(pos.y);
				const int32_t ipos_z = floatAsInt(pos.z);
				if (step_mask & 1u) differing_bits |= (ipos_x ^ floatAsInt(pos.x + scale_f));
				if (step_mask & 2u) differing_bits |= (ipos_y ^ floatAsInt(pos.y + scale_f));
				if (step_mask & 4u) differing_bits |= (ipos_z ^ floatAsInt(pos.z + scale_f));
				scale = (floatAsInt((float)differing_bits) >> SVO_MAX_DEPTH) - 127u;
				scale_f = intAsFloat((scale - SVO_MAX_DEPTH + 127u) << SVO_MAX_DEPTH);
				const OctreeStack entry = stack[scale];
				node_id = entry.node_index;
				parent_id = entry.parent_index;
				t_max = entry.t_max;
				const uint32_t shx = ipos_x >> scale;
				const uint32_t shy = ipos_y >> scale;
				const uint32_t shz = ipos_z >> scale;
				pos.x = intAsFloat(shx << scale);
				pos.y = intAsFloat(shy << scale);
				pos.z = intAsFloat(shz << scale);
				child_offset = (shx & 1u) | ((shy & 1u) << 1u) | ((shz & 1u) << 2u);
				h = 0.0f;
			}
		}

		if (result.hit) {
			result.normal = glm::sign(d) * glm::vec3(float(normal & 1u), float((normal & 2u) >> 1), float((normal & 4u) >> 2));

			if ((mirror_mask & 1) == 0) pos.x = 3.0f - scale_f - pos.x;
			if ((mirror_mask & 2) == 0) pos.y = 3.0f - scale_f - pos.y;
			if ((mirror_mask & 4) == 0) pos.z = 3.0f - scale_f - pos.z;

			result.distance = t_min * world_size;
			result.position.x = std::min(std::max(position.x + t_min * d.x, pos.x + EPS), pos.x + scale_f - EPS);
			result.position.y = std::min(std::max(position.y + t_min * d.y, pos.y + EPS), pos.y + scale_f - EPS);
			result.position.z = std::min(std::max(position.z + t_min * d.z, pos.z + EPS), pos.z + scale_f - EPS);

			result.position = world_size - (result.position - glm::vec3(1.0f)) * world_size;
		}

		return result;
	}

	const uint32_t levels;
	const float world_size;
	std::vector<uint8_t> node_data;
};

