#include "svo_builder.hpp"
#include "fastnoise/FastNoise.h"


void generateSVO(uint8_t max_depth, Losvo& svo) {
	const uint32_t size = 1 << max_depth;
	const uint32_t grid_size_x = size;
	const int32_t grid_size_y = size;
	const uint32_t grid_size_z = size;

	FastNoise myNoise;
	myNoise.SetNoiseType(FastNoise::SimplexFractal);
	for (uint32_t x = 1; x < grid_size_x - 1; x++) {
		for (uint32_t z = 1; z < grid_size_z - 1; z++) {
			int32_t max_height = grid_size_y;

			float amp_x = x - grid_size_x * 0.5f;
			float amp_z = z - grid_size_z * 0.5f;
			float ratio = std::pow(1.0f - sqrt(amp_x * amp_x + amp_z * amp_z) / (10.0f * grid_size_x), 256.0f);
			const float freq = 0.75f;
			int32_t height = int32_t(64.0f * myNoise.GetNoise(float(freq * x), float(freq * z)) + 32);

			for (int y(1); y < 2; ++y) {
				svo.setCell(x, y, z, 1);
			}

			for (int y(2); y < height; ++y) {
				svo.setCell(x, y, z, 1);
			}
		}
	}
}


void loadPointCloud(const std::string& filename, uint8_t svo_level, Losvo& svo) {

}

