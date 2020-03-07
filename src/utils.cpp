#include "utils.hpp"
#include <iostream>
#include "ocl_wrapper.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>


oclw::Context createDefaultContext(oclw::Wrapper& wrapper)
{
	auto platforms = wrapper.getPlatforms();
	if (platforms.empty()) {
		return nullptr;
	}
	// Trying to create GPU context
	std::cout << "Creating context on GPU..." << std::endl;
	oclw::Context context = wrapper.createContext(platforms.front(), oclw::GPU);
	if (!context) {
		// If not available try on CPU
		std::cout << "Creating context on CPU..." << std::endl;
		context = wrapper.createContext(platforms.front(), oclw::CPU);
		if (!context) {
			std::cout << "Cannot create context." << std::endl;
			return nullptr;
		}
	}
	std::cout << "Done." << std::endl;
	return context;
}

std::vector<LSVONode> generateSVO(uint8_t max_depth)
{
	const uint32_t size = 1 << max_depth;
	const uint32_t grid_size_x = size;
	const uint32_t grid_size_y = size;
	const uint32_t grid_size_z = size;
	using Volume = SVO;
	Volume* volume_raw = new Volume(max_depth);

	FastNoise myNoise;
	myNoise.SetNoiseType(FastNoise::SimplexFractal);
	for (uint32_t x = 1; x < grid_size_x - 1; x++) {
		for (uint32_t z = 1; z < grid_size_z - 1; z++) {
			int32_t max_height = grid_size_y;
			float amp_x = x - grid_size_x * 0.5f;
			float amp_z = z - grid_size_z * 0.5f;
			float ratio = std::pow(1.0f - sqrt(amp_x * amp_x + amp_z * amp_z) / (10.0f * grid_size_x), 256.0f);
			int32_t height = int32_t(128.0f * myNoise.GetNoise(float(0.5f * x), float(0.5f * z)) + 32);

			volume_raw->setCell(Cell::Solid, Cell::Grass, x, 0, z);

			for (int y(1); y < std::min(max_height, height); ++y) {
				volume_raw->setCell(Cell::Solid, Cell::Grass, x, y, z);
			}
		}
	}

	uint32_t b_start_x = 200;
	uint32_t b_start_y = 1;
	uint32_t b_start_z = 200;

	uint32_t b_size = 20;

	for (uint32_t x = 0; x < b_size; x++) {
		for (uint32_t y = 0; y < b_size; y++) {
			for (uint32_t z = 0; z < b_size; z++) {
				if (x == 0 && y == b_size / 2 && z == b_size / 2) {
					continue;
				}
				if (x == b_size / 2 && y == b_size-1 && z == b_size / 2) {
					continue;
				}
				if (x == b_size / 2 -1 && y == b_size - 1 && z == b_size / 2) {
					continue;
				}
				if (x == b_size / 2 + 1 && y == b_size - 1 && z == b_size / 2) {
					continue;
				}
				if (x == b_size / 2 + 1 && y == b_size - 1 && z == b_size / 2 + 1) {
					continue;
				}
				if (x == b_size / 2 && y == b_size - 1 && z == b_size / 2 + 1) {
					continue;
				}
				if (x == 0 || x == b_size - 1 ||
					y == 0 || y == b_size - 1 ||
					z == 0 || z == b_size - 1 )
				volume_raw->setCell(Cell::Solid, Cell::Grass, x + b_start_x, y + b_start_y, z + b_start_z);
			}
		}
	}

	auto result = compileSVO(*volume_raw);

	delete volume_raw;

	return result;
}

glm::mat3 generateRotationMatrix(const glm::vec2& angle)
{
	const glm::mat4 rx = glm::rotate(glm::mat4(1.0f), -angle.x, glm::vec3(0.0f, 1.0f, 0.0f));
	const glm::mat4 ry = glm::rotate(glm::mat4(1.0f), -angle.y, glm::vec3(1.0f, 0.0f, 0.0f));

	return ry * rx;
}
