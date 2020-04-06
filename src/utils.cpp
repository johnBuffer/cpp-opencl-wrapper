#include "utils.hpp"
#include <iostream>
#include "ocl_wrapper.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <cmath>


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


void generateSVO(uint8_t max_depth, SVO& svo)
{
	const uint32_t size = 1 << max_depth;
	const uint32_t grid_size_x = size;
	const uint32_t grid_size_y = size;
	const uint32_t grid_size_z = size;
	using Volume = SVO;
	Volume* volume_raw = &svo;

	sf::Image terrain_height_map;
	terrain_height_map.loadFromFile("../height_map.png");
	//terrain_height_map.loadFromFile("../terrain_3.jpg");

	FastNoise myNoise;
	myNoise.SetNoiseType(FastNoise::SimplexFractal);
	for (uint32_t x = 1; x < grid_size_x - 1; x++) {
		for (uint32_t z = 1; z < grid_size_z - 1; z++) {
			int32_t max_height = grid_size_y;

			float amp_x = x - grid_size_x * 0.5f;
			float amp_z = z - grid_size_z * 0.5f;
			float ratio = std::pow(1.0f - sqrt(amp_x * amp_x + amp_z * amp_z) / (10.0f * grid_size_x), 256.0f);
			int32_t height = int32_t(64.0f * myNoise.GetNoise(float(0.75f * x), float(0.75f * z)) + 32);

			volume_raw->setCell(Cell::Solid, Cell::Grass, x, 0, z);

			for (int y(0); y < 10; ++y) {
				volume_raw->setCell(Cell::Mirror, Cell::Grass, x, y, z);
				
				for (int y(0); y < std::min(max_height, height); ++y) {
					volume_raw->setCell(Cell::Solid, Cell::Grass, x, y, z);
				}
			}
		}
	}

	for (uint32_t u(1); u < grid_size_y; ++u) {
		//volume_raw->setCell(Cell::Solid, Cell::Grass, 255, u, 255);
	}

	uint32_t b_start_x = 200;
	uint32_t b_start_y = 1;
	uint32_t b_start_z = 200;

	uint32_t b_size = 20;

	/*for (uint32_t x = 0; x < b_size + 10; x++) {
		for (uint32_t y = 0; y < b_size + 20; y++) {
			if (x > 0 && y > 0 && x < b_size + 9 && y < b_size + 19)
				volume_raw->setCell(Cell::Mirror, Cell::Grass, x + b_start_x - 5, y + b_start_y, b_start_z - 5);
			else
				volume_raw->setCell(Cell::Solid, Cell::Grass, x + b_start_x - 5, y + b_start_y, b_start_z - 5);
		}
	}*/

	for (uint32_t x = 220; x < 230; x++) {
		volume_raw->setCell(Cell::Solid, Cell::Grass, x, 1, 250);
	}

	for (uint32_t x = 220; x < 230; x++) {
		volume_raw->setCell(Cell::Solid, Cell::Grass, x, 1, 240);
		volume_raw->setCell(Cell::Solid, Cell::Grass, x, 2, 240);
	}
}


glm::mat4 generateRotationMatrix(const glm::vec2& angle)
{
	const glm::mat4 rx = glm::rotate(glm::mat4(1.0f), -angle.x, glm::vec3(0.0f, 1.0f, 0.0f));
	const glm::mat4 ry = glm::rotate(glm::mat4(1.0f), -angle.y, glm::vec3(1.0f, 0.0f, 0.0f));

	return ry * rx;
}


float frac(float f)
{
	float whole;
	return std::modf(f, &whole);
}


uint32_t floatAsInt(const float f)
{
	return *(uint32_t*)(&f);
}


float intAsFloat(const uint32_t i)
{
	return *(float*)(&i);
}


std::string vecToString(const glm::vec3 & v)
{
	std::stringstream sx;
	sx << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return sx.str();
}


glm::vec3 projVec(const glm::vec3& in)
{
	const float screen_size_x = 800;
	const float screen_size_y = 450;
	const float aspect_ratio = 16.0f / 9.0f;
	const float near = 0.5f;

	glm::vec3 out(0.0f);

	out.x = near * in.x / (in.z);
	out.y = near * in.y / (in.z);
	out.z = near;

	out.x = int((out.x + 0.5f) * (screen_size_x + 1));
	out.y = int((out.y * aspect_ratio + 0.5f) * (screen_size_y + 1));

	return out;
}

