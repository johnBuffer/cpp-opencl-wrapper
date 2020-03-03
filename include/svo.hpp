#pragma once

#include "volumetric.hpp"


struct Node
{
	Node()
		: leaf(false)
	{
		for (int x(0); x < 2; ++x) {
			for (int y(0); y < 2; ++y) {
				for (int z(0); z < 2; ++z) {
					sub[x][y][z] = nullptr;
				}
			}
		}
	}

	Node* sub[2][2][2];
	bool leaf;
	Cell cell;
};


class SVO
{
public:
	template<uint8_t>
	friend struct LSVO;

	SVO(uint8_t max_depth)
		: m_max_depth(max_depth)
	{
		m_root = new Node();
	}

	~SVO()
	{
		if (m_root) {
			clear(m_root);
		}
	}

	void clear(Node* node)
	{
		for (int x(0); x < 2; ++x) {
			for (int y(0); y < 2; ++y) {
				for (int z(0); z < 2; ++z) {
					if (node->sub[x][y][z]) {
						clear(node->sub[x][y][z]);
					}
				}
			}
		}

		delete node;
	}

	void setCell(Cell::Type type, Cell::Texture texture, uint32_t x, uint32_t y, uint32_t z)
	{
		const uint32_t max_size = uint32_t(std::pow(2, m_max_depth));
		rec_setCell(type, texture, x, y, z, m_root, max_size);
	}

	Node* m_root;

private:
	const uint8_t m_max_depth;
	void rec_setCell(Cell::Type type, Cell::Texture texture, uint32_t x, uint32_t y, uint32_t z, Node* node, uint32_t size)
	{
		if (!node) {
			return;
		}

		if (size == 1) {
			node->cell.type = type;
			node->cell.texture = texture;
			node->leaf = true;
			return;
		}

		const uint32_t sub_size = size / 2;
		const uint32_t cell_x = x / sub_size;
		const uint32_t cell_y = y / sub_size;
		const uint32_t cell_z = z / sub_size;

		if (!node->sub[cell_x][cell_y][cell_z]) {
			node->sub[cell_x][cell_y][cell_z] = new Node();
		}

		rec_setCell(type, texture, x - cell_x * sub_size, y - cell_y * sub_size, z - cell_z * sub_size, node->sub[cell_x][cell_y][cell_z], sub_size);
	}
};
