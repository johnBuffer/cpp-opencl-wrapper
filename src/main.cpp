#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>


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


template<typename T>
void print(const std::vector<T>& vec)
{
	for (const T& obj : vec) {
		std::cout << obj << " ";
	}
	std::cout << std::endl;
}


template<typename T>
float mean(const std::vector<T>& vec)
{
	T sum = 0.0;
	for (const T& obj : vec) {
		sum += obj;
	}
	return sum / (double)(vec.size());
}



int main()
{
	try
	{
		srand(time(NULL));
		oclw::Wrapper wrapper;
		oclw::Context context = createDefaultContext(wrapper);
		auto& devices_list = context.getDevices();
		cl_device_id device = devices_list.front();
		// Create command queue
		oclw::CommandQueue command_queue = context.createQueue(device);
		// Create OpenCL program from HelloWorld.cl kernel source
		oclw::Program program = context.createProgram(device, "../src/image_output.cl");
		oclw::Kernel kernel = program.createKernel("work_id_output");

		constexpr uint8_t thread_count(256);
		std::vector<float> result(thread_count);

		oclw::MemoryObject results_buff = context.createMemoryObject<float>(thread_count, oclw::MemoryObjectReadMode::WriteOnly);

		const size_t globalWorkSize = thread_count;
		const size_t localWorkSize = 1;
		kernel.setArgument(0, results_buff);
		command_queue.addKernel(kernel, 1, NULL, &globalWorkSize, &localWorkSize);
		command_queue.readMemoryObject(results_buff, true, result);

		std::cout << mean(result) << std::endl;
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.getReadableError() << std::endl;
	}

	return 0;
}
