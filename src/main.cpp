#include <iostream>
#include <CL/cl.hpp>
#include "utils.hpp"
#include "ocl_wrapper.hpp"
#include <SFML/Graphics.hpp>


int main()
{
	constexpr uint32_t WIN_WIDTH = 640u;
	constexpr uint32_t WIN_HEIGHT = 480u;

	try
	{
		// OpenCL initialization
		constexpr uint64_t buffer_size = 8u;
		cl::Platform platform = getDefaultPlatform();
		cl::Device device = getDefaultDevice(platform);
		cl::Context context = createDefaultContext();
		cl_int err_num;
		cl::Buffer buffer(context, oclw::WriteOnly, buffer_size, nullptr, &err_num);
		
		const std::string source = oclw::loadSourceFromFile("../src/image_output.cl");
		cl::Program program(source, true, &err_num);
		oclw::checkError(err_num, "Cannot build program");

		cl::Kernel kernel(program, "image_output.cl", &err_num);
		oclw::checkError(err_num, "Cannot create kernel");

		cl::CommandQueue queue(context, device, 0, &err_num);
		oclw::checkError(err_num, "Cannot create queue");


		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML");

		sf::Image ocl_result;
		ocl_result.create(WIN_WIDTH, WIN_HEIGHT);

		std::vector<uint8_t> image_result;

		while (window.isOpen())
		{
			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
			}

			window.clear();

			/*for (uint32_t x(0); x < WIN_WIDTH; ++x) {
				for (uint32_t y(0); y < WIN_HEIGHT; ++y) {
					uint32_t index = 4 * (x + y * WIN_WIDTH);
					uint8_t r = image_result[index + 0];
					uint8_t g = image_result[index + 1];
					uint8_t b = image_result[index + 2];
					uint8_t a = image_result[index + 3];
					ocl_result.setPixel(x, y, sf::Color(r, g, b, a));
				}
			}*/

			window.display();
		}
		
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error " << error.getErrorCode() << ": " << error.getMessage() << std::endl;
	}

	return 0;
}
