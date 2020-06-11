#pragma once

#include "losvo.hpp"


void generateSVO(uint8_t max_depth, Losvo& svo);


void loadPointCloud(const std::string& filename, uint8_t svo_level, Losvo& svo);