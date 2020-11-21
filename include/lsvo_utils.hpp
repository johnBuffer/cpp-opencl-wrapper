#pragma once

#include "svo.hpp"
#include <vector>


struct LSVONode
{
	LSVONode()
		: child_mask(0U)
		, leaf_mask(0U)
		, child_offset(0U)
	{}

	uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
};


struct OctreeStack
{
	uint32_t parent_index;
	float t_max;
};


void compileSVO_rec(const Node* node, std::vector<LSVONode>& data, const uint64_t node_index, uint64_t& max_offset);


std::vector<LSVONode> compileSVO(const SVO& svo);