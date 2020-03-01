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


int main()
{
	constexpr uint32_t WIN_WIDTH = 640u;
	constexpr uint32_t WIN_HEIGHT = 480u;

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
		oclw::Program program = context.createProgram(device, "../src/image_output.cl");
		// Create OpenCL kernel
		oclw::Kernel kernel = program.createKernel("work_id_output");
		// Create memory objects that will be used as arguments to kernel
		std::vector<uint8_t> result(WIN_WIDTH * WIN_HEIGHT * 4);
		
		oclw::MemoryObject buff_result = context.createMemoryObject<uint8_t>(result.size(), oclw::ReadWrite);
		// Set kernel's args
		kernel.setArgument(0, buff_result);
		// Queue the kernel up for execution across the array
		size_t globalWorkSize[] = { WIN_WIDTH, WIN_HEIGHT };
		size_t localWorkSize[] = { 1, 1 };
		command_queue.addKernel(kernel, 2, NULL, globalWorkSize, localWorkSize);
		command_queue.readMemoryObject(buff_result, true, result);

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML");

		sf::Image ocl_result;
		ocl_result.create(WIN_WIDTH, WIN_HEIGHT);

		while (window.isOpen())
		{
			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
			}

			for (uint32_t x(0); x < WIN_WIDTH; ++x) {
				for (uint32_t y(0); y < WIN_HEIGHT; ++y) {
					uint32_t index = 4 * (x + y * WIN_WIDTH);
					uint8_t r = result[index + 0];
					uint8_t g = result[index + 1];
					uint8_t b = result[index + 2];
					uint8_t a = result[index + 3];
					ocl_result.setPixel(x, y, sf::Color(r, g, b, a));
				}
			}

			window.clear();

			sf::Texture tex;
			tex.loadFromImage(ocl_result);
			window.draw(sf::Sprite(tex));

			window.display();
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.getReadableError() << std::endl;
	}

	return 0;
}
