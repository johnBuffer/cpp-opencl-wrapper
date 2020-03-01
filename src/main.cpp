#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>
#include <event_manager.hpp>
#include "utils.hpp"


int main()
{
	constexpr uint32_t WIN_WIDTH = 1600u;
	constexpr uint32_t WIN_HEIGHT = 900u;

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
		oclw::Program program = context.createProgram(device, "../src/julia.cl");
		// Create OpenCL kernel
		oclw::Kernel kernel = program.createKernel("julia");
		// Create memory objects that will be used as arguments to kernel
		std::vector<uint8_t> result(WIN_WIDTH * WIN_HEIGHT * 4);
		oclw::MemoryObject buff_result = context.createMemoryObject<uint8_t>(result.size(), oclw::ReadWrite);
		// Set kernel's args
		float zoom = 0.5f;
		cl_float2 position = { 0.0f, 0.0f };
		cl_float2 params = { -0.8f, 0.0f};
		const float speed = 0.01f;

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML");

		sfev::EventManager event_manager(window);
		event_manager.addEventCallback(sf::Event::Closed, [&](sfev::CstEv) {window.close(); });
		event_manager.addKeyPressedCallback(sf::Keyboard::Z, [&](sfev::CstEv) {position.y -= speed / zoom; });
		event_manager.addKeyPressedCallback(sf::Keyboard::S, [&](sfev::CstEv) {position.y += speed / zoom; });
		event_manager.addKeyPressedCallback(sf::Keyboard::Q, [&](sfev::CstEv) {position.x -= speed / zoom; });
		event_manager.addKeyPressedCallback(sf::Keyboard::D, [&](sfev::CstEv) {position.x += speed / zoom; });
		event_manager.addKeyPressedCallback(sf::Keyboard::Space, [&](sfev::CstEv) {zoom *= 1.2f; });
		event_manager.addKeyPressedCallback(sf::Keyboard::LShift, [&](sfev::CstEv) {zoom /= 1.2f; });

		event_manager.addKeyPressedCallback(sf::Keyboard::Up, [&](sfev::CstEv) {params.y -= speed; });
		event_manager.addKeyPressedCallback(sf::Keyboard::Down, [&](sfev::CstEv) {params.y += speed; });
		event_manager.addKeyPressedCallback(sf::Keyboard::Left, [&](sfev::CstEv) {params.x -= speed; });
		event_manager.addKeyPressedCallback(sf::Keyboard::Right, [&](sfev::CstEv) {params.x += speed; });

		sf::Image ocl_result;
		ocl_result.create(WIN_WIDTH, WIN_HEIGHT);

		while (window.isOpen())
		{
			event_manager.processEvents();

			kernel.setArgument(0, zoom);
			kernel.setArgument(1, position);
			kernel.setArgument(2, params);
			kernel.setArgument(3, buff_result);
			// Queue the kernel up for execution across the array
			size_t globalWorkSize[] = { WIN_WIDTH, WIN_HEIGHT };
			size_t localWorkSize[] = { 1, 1 };
			command_queue.addKernel(kernel, 2, NULL, globalWorkSize, localWorkSize);
			command_queue.readMemoryObject(buff_result, true, result);

			/*for (uint32_t x(0); x < WIN_WIDTH; ++x) {
				for (uint32_t y(0); y < WIN_HEIGHT; ++y) {
					uint32_t index = 4 * (x + y * WIN_WIDTH);
					uint8_t r = result[index + 0];
					uint8_t g = result[index + 1];
					uint8_t b = result[index + 2];
					uint8_t a = result[index + 3];
					ocl_result.setPixel(x, y, sf::Color(r, g, b, a));
				}
			}*/

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
