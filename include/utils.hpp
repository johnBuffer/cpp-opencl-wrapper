#pragma once

#include "ocl_wrapper.hpp"
#include "fastnoise/FastNoise.h"
#include "lsvo_utils.hpp"
#include <vector>


oclw::Context createDefaultContext(oclw::Wrapper& wrapper);


std::vector<LSVONode> generateSVO();


glm::mat3 generateRotationMatrix(const glm::vec2& angle);
