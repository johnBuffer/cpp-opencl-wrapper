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


float readFloat()
{
	float f;
	std::ifstream fin("male_16_down.bin", std::ios::binary);
	while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
		std::cout << f << '\n';

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

	/*FastNoise myNoise;
	myNoise.SetNoiseType(FastNoise::SimplexFractal);
	for (uint32_t x = 1; x < grid_size_x - 1; x++) {
		for (uint32_t z = 1; z < grid_size_z - 1; z++) {
			int32_t max_height = grid_size_y;

			float amp_x = x - grid_size_x * 0.5f;
			float amp_z = z - grid_size_z * 0.5f;
			float ratio = std::pow(1.0f - sqrt(amp_x * amp_x + amp_z * amp_z) / (10.0f * grid_size_x), 256.0f);
			int32_t height = int32_t(128.0f * myNoise.GetNoise(float(0.75f * x), float(0.75f * z)) + 32);

			volume_raw->setCell(Cell::Solid, Cell::Grass, x, 0, z);

			for (int y(0); y < std::min(max_height, height); ++y) {
				volume_raw->setCell(Cell::Solid, Cell::Grass, x, y, z);
			}
		}
	}*/

	//std::ifstream data_file("../res/Pointcloud_2m/tq2575_DSM_2M.asc");
	//std::ifstream data_file("../res/Pointcloud_50cm/tq3580_DSM_50CM.asc");
	std::ifstream data_file("../res/cloud.bin", std::ios::binary | std::ios::ate);
	if (data_file.is_open()) {
		std::streamsize size = data_file.tellg();
		data_file.seekg(0, std::ios::beg);
		std::cout << "File size " << size << std::endl;
		std::vector<glm::vec3> points;
		std::vector<float> buffer(size/4);
		if (data_file.read((char*)buffer.data(), size)) {
			std::cout << "Load complete" << std::endl;
		}
		else {
			std::cout << "Load failed" << std::endl;
		}

		const uint64_t points_count = buffer.size() / 3;
		const uint32_t skip = 2;
		points.resize(points_count);
		for (uint32_t i(0); i < points_count; i+=skip) {
			points[i].x = buffer[3 * i];
			points[i].y = buffer[3 * i + 1];
			points[i].z = buffer[3 * i + 2];
		}
		buffer.clear();
		std::cout << "Conversion complete" << std::endl;

		glm::vec3 min_val(0.0f);
		glm::vec3 max_val(0.0f);
		for (const glm::vec3 pt : points) {
			min_val.x = std::min(min_val.x, pt.x);
			min_val.y = std::min(min_val.y, pt.y);
			min_val.z = std::min(min_val.z, pt.z);

			max_val.x = std::max(max_val.x, pt.x);
			max_val.y = std::max(max_val.y, pt.y);
			max_val.z = std::max(max_val.z, pt.z);
		}

		std::cout << min_val.z << " " << max_val.z << std::endl;

		const float scale = 8.0f;
		for (const glm::vec3 pt_raw : points) {
			const glm::vec3 pt = (pt_raw - min_val) * scale;
			const float x = std::max(1.0f, std::min(pt.x, float(grid_size_x - 1)));
			const float y = std::max(1.0f, std::min(pt.y, float(grid_size_x - 1)));
			const float z = std::max(1.0f, std::min(pt.z, float(grid_size_x - 1)));
			volume_raw->setCell(Cell::Solid, Cell::Grass, x, z, y);
		}

		data_file.close();
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

