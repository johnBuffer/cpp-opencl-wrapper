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

	FastNoise myNoise;
	myNoise.SetNoiseType(FastNoise::SimplexFractal);
	for (uint32_t x = 1; x < grid_size_x - 1; x++) {
		for (uint32_t z = 1; z < grid_size_z - 1; z++) {
			int32_t max_height = grid_size_y;

			float amp_x = x - grid_size_x * 0.5f;
			float amp_z = z - grid_size_z * 0.5f;
			float ratio = std::pow(1.0f - sqrt(amp_x * amp_x + amp_z * amp_z) / (10.0f * grid_size_x), 256.0f);
			int32_t height = int32_t(92.0f * myNoise.GetNoise(float(0.75f * x), float(0.75f * z)) + 32);

			for (int y(0); y < 30; ++y) {
				volume_raw->setCell(Cell::Mirror, Cell::Grass, x, y, z);
			}

			for (int y(0); y < std::min(max_height, height); ++y) {
				volume_raw->setCell(Cell::Solid, Cell::Grass, x, y, z);
			}
		}
	}
}


void loadPointCloud(const std::string& filename, uint8_t svo_level, SVO& svo)
{
	const uint32_t size = 1 << svo_level;
	const uint32_t grid_size_x = size;
	const int32_t grid_size_y = size;
	const uint32_t grid_size_z = size;
	using Volume = SVO;

	Volume* volume_raw = &svo;
	std::ifstream data_file(filename, std::ios::binary | std::ios::ate);
	if (data_file.is_open()) {
		std::streamsize size = data_file.tellg();
		data_file.seekg(0, std::ios::beg);
		std::cout << "File size " << size << std::endl;
		std::vector<glm::vec3> points;
		std::vector<float> buffer(size / 4);
		if (data_file.read((char*)buffer.data(), size)) {
			std::cout << "Load complete" << std::endl;
		}
		else {
			std::cout << "Load failed" << std::endl;
		}

		const uint64_t points_count = buffer.size() / 3;
		const uint32_t skip = 1;
		points.resize(points_count);
		for (uint32_t i(0); i < points_count; i += skip) {
			points[i].x = buffer[3 * i];
			points[i].y = buffer[3 * i + 1];
			points[i].z = buffer[3 * i + 2];
		}
		buffer.clear();

		std::cout << "Conversion complete, " << points_count << " points" << std::endl;

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

		uint32_t done = 0u;
		uint32_t last_progress = 0u;
		
		const float scale = std::pow(2.0f, float(svo_level - 8));
		for (const glm::vec3 pt_raw : points) {
			const glm::vec3 pt = (pt_raw - min_val) * scale;
			const float x = std::max(1.0f, std::min(pt.x, float(grid_size_x - 1)));
			const float y = std::max(1.0f, std::min(pt.y, float(grid_size_x - 1)));
			const float z = std::max(1.0f, std::min(pt.z, float(grid_size_x - 1)));
			volume_raw->setCell(Cell::Solid, Cell::Grass, as<uint32_t>(x), as<uint32_t>(z), as<uint32_t>(y));
			++done;
			const uint32_t progress = as<uint32_t>(100 * (done / float(points_count)));
			if (progress % 10 == 0 && last_progress != progress) {
				std::cout << "Importation... " << progress << "%" << std::endl;
				last_progress = progress;
			}
		}

		data_file.close();
	}
	else {
		std::cout << "Cannot find '" << filename << "'" << std::endl;
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

