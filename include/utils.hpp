#pragma once

#include "ocl_wrapper.hpp"
#include "fastnoise/FastNoise.h"
#include "lsvo_utils.hpp"
#include <vector>


oclw::Context createDefaultContext(oclw::Wrapper& wrapper);


void generateSVO(uint8_t max_depth, SVO& svo);


glm::mat3 generateRotationMatrix(const glm::vec2& angle);


float frac(float f);


uint32_t floatAsInt(const float f);


float intAsFloat(const uint32_t i);