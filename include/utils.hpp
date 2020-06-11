#pragma once

#include "ocl_wrapper.hpp"
#include "lsvo_utils.hpp"
#include <vector>


glm::mat4 generateRotationMatrix(const glm::vec2& angle);


float frac(float f);


uint32_t floatAsInt(const float f);


float intAsFloat(const uint32_t i);


std::string vecToString(const glm::vec3& v);


std::string vecToString(const glm::uvec3& v);


template<typename U, typename T>
U as(const T& obj) {
	return static_cast<U>(obj);
}

