#pragma once

#include <vector>
#include "utils.hpp"


struct _Node {
	uint8_t node_mask;
};

struct _Leaf {
	uint8_t type;
};

struct Losvo
{
	Losvo(const uint32_t levels_count)
		: levels(levels_count)
	{
		// Last level is for leafs
		const uint64_t nodes_count = as<uint64_t>((std::pow(8, levels_count + 1) - 1) / 7) - 1;
		node_data.resize(nodes_count);

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
		const uint32_t side_size = as<uint32_t>(std::pow(2, levels));
		// Update hierarchy
		glm::uvec3 current_position(x, y, z);
		uint32_t level_index = 0;
		uint32_t parent_index = 0;
		uint32_t half_size = side_size >> 1u;
		for (uint32_t current_level(0); current_level < levels - 1; ++current_level) {
			const glm::uvec3 level_position = current_position / half_size;
			const uint8_t sub_index = getSubIndex(level_position);
			// Checking current node
			const uint32_t level_size = 1 << ( 3 * current_level);
			std::cout << level_size << std::endl;
			const uint32_t current_index = parent_index * level_size + sub_index;
			node_data[level_index + current_index].node_mask |= (1 << sub_index);
			parent_index = current_index;
			level_index += 8 * level_size;
			// Updating for next step
			current_position -= level_position * half_size;
			half_size >>= 1u;
		}

		const glm::uvec3 level_position = current_position / half_size;
		const uint8_t sub_index = getSubIndex(level_position);
		const uint32_t level_size = 1 << (3 * (levels - 2));
		const uint32_t leaf_computed_index = level_index + parent_index * level_size + sub_index;

		// Set the actual leaf
		std::cout << "Leaf index: " << leaf_computed_index << std::endl;
	}

	const uint32_t levels;
	std::vector<_Node> node_data;
};

