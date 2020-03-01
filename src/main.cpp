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
		// Initialize wrapper
		oclw::Wrapper wrapper;
		// Retrieve context
		oclw::Context context = createDefaultContext(wrapper);
		// Get devices
		auto& devices_list = context.getDevices();
		cl_device_id device = devices_list.front();
		// Create command queue
		oclw::CommandQueue command_queue = context.createQueue(device);
		// Create OpenCL program from HelloWorld.cl kernel source
		oclw::Program program = context.createProgram(device, "../src/kernel.cl");
		// Create OpenCL kernel
		oclw::Kernel kernel = program.createKernel("hello_kernel");
		// Create memory objects that will be used as arguments to kernel
		constexpr uint64_t ARRAY_SIZE = 8u;
		std::vector<float> result(ARRAY_SIZE);
		std::vector<float> a(ARRAY_SIZE);
		std::vector<float> b(ARRAY_SIZE);
		for (int i = 0; i < ARRAY_SIZE; i++) {
			a[i] = (float)i;
			b[i] = (float)(i * 2);
		}
		oclw::MemoryObject buff_a = context.createMemoryObject(a, oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject buff_b = context.createMemoryObject(b, oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject buff_result = context.createMemoryObject<float>(ARRAY_SIZE, oclw::ReadWrite);
		// Set kernel's args
		kernel.setArgument(0, buff_a);
		kernel.setArgument(1, buff_b);
		kernel.setArgument(2, buff_result);
		// Queue the kernel up for execution across the array
		size_t globalWorkSize[1] = { ARRAY_SIZE };
		size_t localWorkSize[1] = { 1 };
		command_queue.addKernel(kernel, 1, NULL, globalWorkSize, localWorkSize);
		command_queue.readMemoryObject(buff_result, true, result);
		// Output the result buffer
		for (uint32_t i = 0; i < ARRAY_SIZE; i++)
		{
			std::cout << result[i] << " ";
		}
		std::cout << std::endl;
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.what() << std::endl;
	}

	return 0;
}
