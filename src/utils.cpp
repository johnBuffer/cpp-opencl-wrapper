#include "utils.hpp"
#include <iostream>
#include "ocl_wrapper.hpp"


cl::Platform getDefaultPlatform()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
		throw oclw::Exception(-1, "Cannot get platforms");
	}
	return platforms.front();
}

cl::Device getDefaultDevice(const cl::Platform& platform)
{
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.empty()) {
		throw oclw::Exception(-1, "Cannot get devices");
	}
	return devices.front();
}

cl::Context createDefaultContext()
{
	cl_int err_num;
	cl::Context context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &err_num);
	oclw::checkError(err_num, "Cannot create context");
	return context;
}