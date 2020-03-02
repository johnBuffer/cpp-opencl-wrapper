#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>
#include <event_manager.hpp>
#include "utils.hpp"
#include "fly_controller.hpp"


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
		oclw::Program program = context.createProgram(device, "../src/voxel.cl");
		// Create OpenCL kernel
		oclw::Kernel kernel = program.createKernel("raytracer");
		// Create memory objects that will be used as arguments to kernel
		std::vector<LSVONode> svo = generateSVO();
		oclw::MemoryObject svo_data = context.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		std::vector<uint8_t> result(WIN_WIDTH * WIN_HEIGHT * 4);
		oclw::MemoryObject buff_result = context.createMemoryObject<uint8_t>(result.size(), oclw::WriteOnly);
		kernel.setArgument(0, svo_data);
		kernel.setArgument(1, buff_result);
		// Problem dimensions
		size_t globalWorkSize[] = { WIN_WIDTH, WIN_HEIGHT };
		size_t localWorkSize[] = { 20, 20 };

		const float speed = 0.01f;

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML");

		sfev::EventManager event_manager(window);
		event_manager.addEventCallback(sf::Event::Closed, [&](sfev::CstEv) {window.close(); });

		sf::Texture tex;
		sf::Image ocl_result;
		ocl_result.create(WIN_WIDTH, WIN_HEIGHT);

		// Camera
		Camera camera;
		camera.position = glm::vec3(256, 200, 256);
		camera.view_angle = glm::vec2(0.0f);
		camera.fov = 1.0f;
		event_manager.addKeyPressedCallback(sf::Keyboard::Space, [&](sfev::CstEv) { camera.position.y -= 2.0f; });

		const float scale = 1.0f / 512.0f;

		FlyController controller;

		while (window.isOpen())
		{
			event_manager.processEvents();

			const cl_float3 camera_position  = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };
			//const cl_float3 camera_position  = { 1.5f, 1.5f, 1.5f };
			const cl_float3 camera_direction = { camera.position.x * scale, camera.position.y * scale , camera.position.z * scale };
			kernel.setArgument(2, camera_position);
			command_queue.addKernel(kernel, 2, NULL, globalWorkSize, localWorkSize);
			command_queue.readMemoryObject(buff_result, true, result);

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
