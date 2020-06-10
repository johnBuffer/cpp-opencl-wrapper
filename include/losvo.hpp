#pragma once

#include <vector>


struct Losvo
{
	Losvo(const uint32_t levels_count)
		: levels(levels_count)
	{
		// Last level is for leafs
		const uint64_t nodes_count = static_cast<uint64_t>(std::pow(8, levels_count + 1) - 1) / 7;
		node_data.resize(nodes_count);

		for (uint8_t& node : node_data) {
			node = 0;
		}

		std::cout << "Allocation done (" << node_data.size() << " Nodes)" << std::endl;
	}

	static std::string vecToStr(const glm::uvec3 & v)
	{
		std::stringstream sx;
		sx << "(" << v.x << ", " << v.y << ", " << v.z << ")";
		return sx.str();
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
		//std::cout << "Adding " << vecToStr(current_position) << std::endl;
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
			//std::cout << "Side size " << side_size << " Current index " << current_index << " Sub position " << vecToStr(sub_position) << std::endl;
			const uint32_t node_index = level_index + current_index;
			node_data[node_index] |= (1 << sub_index);
			parent_index = current_index;
			parent_sub_index = sub_index;
			//std::cout << level_index << std::endl;
			level_index += level_size;
			// Updating for next step
			current_position -= sub_position * half_size;
			side_size = half_size;
		}
		//std::cout << "Level index: " << level_index << std::endl;
		//std::cout << "Parent index: " << parent_index << std::endl;

		const glm::uvec3 level_position = current_position / side_size;
		const uint32_t leaf_computed_index = level_index + parent_index * 8 + parent_sub_index;

		// Set the actual leaf
		//std::cout << "Leaf index: " << leaf_computed_index << std::endl;
		//node_data[leaf_computed_index] = cell_type;
	}

	const uint32_t levels;
	std::vector<uint8_t> node_data;
};

