#pragma once

#include <CL/cl.hpp>


cl::Platform getDefaultPlatform();


cl::Device getDefaultDevice(const cl::Platform& paltform);


cl::Context createDefaultContext();
