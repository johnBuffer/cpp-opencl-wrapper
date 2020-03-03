#pragma once

#include "svo.hpp"
#include <vector>


struct LSVONode
{
	LSVONode()
		: child_mask(0U)
		, leaf_mask(0U)
		, child_offset(0U)
		, padding(0u)
	{}

	uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
	uint16_t padding;
};


void compileSVO_rec(const Node* node, std::vector<LSVONode>& data, const uint32_t node_index, uint32_t& max_offset);


std::vector<LSVONode> compileSVO(const SVO& svo);