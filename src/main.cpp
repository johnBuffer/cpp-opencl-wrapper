#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>

cl_context createDefaultContext(oclw::Wrapper& wrapper)
{
	auto platforms = wrapper.getPlatforms();
	if (platforms.empty()) {
		return nullptr;
	}
	// Trying to create GPU context
	std::cout << "Creating context on GPU..." << std::endl;
	cl_context context = wrapper.createContext(platforms.front(), oclw::Wrapper::GPU);
	if (!context) {
		// If not available try on CPU
		std::cout << "Creating context on CPU..." << std::endl;
		context = wrapper.createContext(platforms.front(), oclw::Wrapper::CPU);
		if (!context) {
			std::cout << "Cannot create context." << std::endl;
			return nullptr;
		}
	}
	std::cout << "Done." << std::endl;
	return context;
}


int main()
{
	oclw::Wrapper wrapper;
	wrapper.fetchPlatforms(1);

	cl_context context = createDefaultContext(wrapper);

	return 0;
}
