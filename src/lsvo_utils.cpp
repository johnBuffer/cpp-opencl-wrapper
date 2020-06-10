#include "lsvo_utils.hpp"


void compileSVO_rec(const Node* node, std::vector<LSVONode>& data, const uint64_t node_index, uint64_t& max_offset)
{
	if (node) {
		const uint64_t child_pos = data.size();
		const uint64_t offset = child_pos - node_index;
		max_offset = offset > max_offset ? offset : max_offset;
		data[node_index].child_offset = static_cast<uint32_t>(offset);

		bool empty = true;
		for (uint8_t x(0U); x < 2; ++x) {
			for (uint8_t y(0U); y < 2; ++y) {
				for (uint8_t z(0U); z < 2; ++z) {
					if (node->sub[x][y][z]) {
						empty = false;
						break;
					}
				}
			}
		}

		if (!empty) {
			for (uint8_t i(8U); i--;) {
				data.emplace_back();
			}

			for (uint8_t x(0U); x < 2; ++x) {
				for (uint8_t y(0U); y < 2; ++y) {
					for (uint8_t z(0U); z < 2; ++z) {
						const Node* sub_node = node->sub[x][y][z];
						if (sub_node) {
							const uint8_t sub_index = z * 4 + y * 2 + x;
							data[node_index].child_mask |= (1U << sub_index);
							if (!(sub_node->leaf)) {
								compileSVO_rec(sub_node, data, child_pos + sub_index, max_offset);
							}
							else {
								if (sub_node->cell.type == Cell::Mirror) {
									data[node_index].reflective_mask |= (1U << sub_index);
								}

								if (sub_node->cell.type != Cell::Empty) {
									data[node_index].leaf_mask |= (1U << sub_index);
								}
							}
						}
					}
				}
			}
		}
	}
}

std::vector<LSVONode> compileSVO(const SVO & svo)
{
	std::vector<LSVONode> data;
	data.push_back(LSVONode());

	uint64_t max_offset = 0U;
	compileSVO_rec(svo.m_root, data, 0, max_offset);

	return data;
}


