#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>

oclw::Context createDefaultContext(oclw::Wrapper& wrapper)
{
	auto platforms = wrapper.getPlatforms();
	if (platforms.empty()) {
		return nullptr;
	}
	// Trying to create GPU context
	std::cout << "Creating context on GPU..." << std::endl;
	oclw::Context context = wrapper.createContext(platforms.front(), oclw::GPU);
	if (!context) {
		// If not available try on CPU
		std::cout << "Creating context on CPU..." << std::endl;
		context = wrapper.createContext(platforms.front(), oclw::CPU);
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
	try
	{
		oclw::Wrapper wrapper;
		wrapper.fetchPlatforms(1);

		// Retrieve context
		oclw::Context context = createDefaultContext(wrapper);

		// Get devices
		auto& devices_list = context.getDevices();
		cl_device_id device = devices_list.front();

		cl_command_queue command_queue = context.createQueue(device);

		// Create OpenCL program from HelloWorld.cl kernel source
		oclw::Program program = context.createProgram(device, "../src/kernel.cu");

		// Create OpenCL kernel
		cl_kernel kernel = program.createKernel("hello_kernel");

		// Create memory objects that will be used as arguments to kernel
		constexpr uint64_t ARRAY_SIZE = 128;
		float result[ARRAY_SIZE];
		std::vector<float> a(ARRAY_SIZE);
		std::vector<float> b(ARRAY_SIZE);
		for (int i = 0; i < ARRAY_SIZE; i++)
		{
			a[i] = (float)i;
			b[i] = (float)(i * 2);
		}

		oclw::MemoryObject buff_a = context.createMemoryObject(a, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
		oclw::MemoryObject buff_b = context.createMemoryObject(b, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
		oclw::MemoryObject buff_c = context.createMemoryObject<float>(ARRAY_SIZE, CL_MEM_READ_WRITE);

	}
	catch (const oclw::WrapperException& error)
	{
		std::cout << "Error: " << error.what() << std::endl;
	}

	return 0;
}
