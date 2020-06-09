#pragma once
#include "ocl_wrapper.hpp"


struct SceneSettings
{
	cl_float3 camera_position;
	cl_float3 light_position;
	cl_float light_intensity;
	cl_float light_radius;
	cl_float time;
};
