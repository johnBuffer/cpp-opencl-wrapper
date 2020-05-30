#include "utils.hpp"
#include <iostream>
#include "ocl_wrapper.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <cmath>
#include <fstream>
#include <string>


float getValueAt(const std::string& str, uint32_t start, uint32_t end)
{
	return 0.0f;
}


void generateSVO(uint8_t max_depth, SVO& svo)
{
	const uint32_t size = 1 << max_depth;
	const uint32_t grid_size_x = size;
	const int32_t grid_size_y = size;
	const uint32_t grid_size_z = size;
	using Volume = SVO;
	Volume* volume_raw = &svo;

	//std::ifstream data_file("../res/Pointcloud_2m/tq2575_DSM_2M.asc");
	//std::ifstream data_file("../res/Pointcloud_50cm/tq3580_DSM_50CM.asc");
	std::ifstream data_file("../res/cloud.xyz");
	if (data_file.is_open()) {
		std::string line;

		uint32_t current_coord = 0;
		uint32_t current_line = 0;
		// Skip header
		while (current_line++ < 6) {
			std::getline(data_file, line);
		}

		const float scale = 1.0f;
		const float no_value_value = -9999;
		uint32_t valid_coords_count = 0;

		std::vector<float> coords;

		while (std::getline(data_file, line)) {
			const uint32_t line_size = line.size();
			uint32_t start_position = 0;
			for (uint32_t i(0); i < line_size; ++i) {
				if (line[i] == ' ' || i == line_size-1) {
					const uint32_t offset = (i == line_size - 1) ? 1 : 0;
					const std::string value_str = line.substr(start_position, (i + offset) - start_position);
					const float value = std::max(1.0f, std::min(scale * (std::stof(value_str)), float(grid_size_x - 2)));
					start_position = ++i;
					++valid_coords_count;

					coords.push_back(value);
				}
			}
		}

		for (uint32_t x(1); x < (grid_size_x/2000)*2000; ++x) {
			for (uint32_t z(1); z < (grid_size_x / 2000) * 2000; ++z) {
				volume_raw->setCell(Cell::Solid, Cell::Grass, x, 1, z);
			}
		}

		data_file.close();
		const uint32_t coords_count = coords.size();
		for (uint32_t i(0); i < coords_count; ++i) {
			float height = coords[i];
			float x = (i % 2000);
			float z = i / 2000.0f;
			if (x > 1.0f && x < grid_size_x - 1 && z > 1.0f && z < grid_size_x - 1) {
				volume_raw->setCell(Cell::Solid, Cell::Grass, x, height, z);
			}
		}

		std::cout << valid_coords_count << std::endl;
		//exit(0);
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

