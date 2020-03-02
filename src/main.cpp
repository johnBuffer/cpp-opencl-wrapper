#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>
#include "event_manager.hpp"
#include <swarm.hpp>

#include "utils.hpp"
#include "fly_controller.hpp"


int main()
{
	constexpr uint32_t WIN_WIDTH = 1920u;
	constexpr uint32_t WIN_HEIGHT = 1080u;

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
		sf::Image image_side, image_top;
		image_side.loadFromFile("../res/grass_side_16x16.bmp");
		image_top.loadFromFile("../res/grass_top_16x16.bmp");

		std::vector<LSVONode> svo = generateSVO();
		oclw::MemoryObject svo_data = context.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject buff_view_matrix = context.createMemoryObject<float>(9, oclw::ReadOnly);
		std::vector<uint8_t> result(WIN_WIDTH * WIN_HEIGHT * 4);
		oclw::MemoryObject buff_result = context.createMemoryObject<uint8_t>(result.size(), oclw::WriteOnly);

		oclw::MemoryObject buff_image_top = context.createImage2D(image_top.getSize().x, image_top.getSize().y, (void*)image_top.getPixelsPtr(), oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject buff_image_side = context.createImage2D(image_side.getSize().x, image_side.getSize().y, (void*)image_side.getPixelsPtr(), oclw::ReadOnly | oclw::CopyHostPtr);

		kernel.setArgument(0, svo_data);
		kernel.setArgument(1, buff_result);
		kernel.setArgument(4, buff_image_top);
		kernel.setArgument(5, buff_image_side);


		// Problem dimensions
		size_t globalWorkSize[] = { WIN_WIDTH, WIN_HEIGHT };
		size_t localWorkSize[] = { 20, 20 };

		const float speed = 0.01f;

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML");
		window.setMouseCursorVisible(false);
		const uint32_t thread_count = 16U;
		const uint32_t area_count = uint32_t(sqrt(thread_count));
		swrm::Swarm swarm(thread_count);

		EventManager event_manager(window);

		sf::Texture tex;
		sf::Image ocl_result;
		ocl_result.create(WIN_WIDTH, WIN_HEIGHT);

		// Camera
		Camera camera;
		camera.position = glm::vec3(256, 200, 256);
		camera.view_angle = glm::vec2(0.0f);
		camera.fov = 1.0f;

		const float scale = 1.0f / 1024.0f;

		FlyController controller;

		while (window.isOpen())
		{
			sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
			event_manager.processEvents(controller, camera);

			if (event_manager.mouse_control) {
				sf::Mouse::setPosition(sf::Vector2i(WIN_WIDTH / 2, WIN_HEIGHT / 2), window);
				const float mouse_sensitivity = 0.0015f;
				controller.updateCameraView(mouse_sensitivity * glm::vec2(mouse_pos.x - WIN_WIDTH * 0.5f, (WIN_HEIGHT  * 0.5f) - mouse_pos.y), camera);
			}
			const glm::mat3 view_matrix = camera.rot_mat;

			const cl_float3 camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };
			const cl_float2 camera_direction = { camera.view_angle.x, camera.view_angle.y };
			command_queue.writeInMemoryObject(buff_view_matrix, true, &view_matrix[0]);
			kernel.setArgument(2, camera_position);
			kernel.setArgument(3, buff_view_matrix);
			command_queue.addKernel(kernel, 2, NULL, globalWorkSize, localWorkSize);
			command_queue.readMemoryObject(buff_result, true, result);

			// Computing some constants, could be done outside main loop
			const uint32_t area_width = WIN_WIDTH / area_count;
			const uint32_t area_height = WIN_HEIGHT / area_count;
			auto group = swarm.execute([&](uint32_t thread_id, uint32_t max_thread) {
				const uint32_t start_x = thread_id % 4;
				const uint32_t start_y = thread_id / 4;
				for (uint32_t x(start_x * area_width); x < (start_x + 1) * area_width; ++x) {
					for (uint32_t y(start_y * area_height); y < (start_y + 1) * area_height; ++y) {
						// Computing ray coordinates in 'lens' space ie in normalized screen space
						uint32_t index = 4 * (x + y * WIN_WIDTH);
						uint8_t r = result[index + 0];
						uint8_t g = result[index + 1];
						uint8_t b = result[index + 2];
						uint8_t a = result[index + 3];
						ocl_result.setPixel(x, y, sf::Color(r, g, b, a));
					}
				}
			});
			// Wait for threads to terminate
			group.waitExecutionDone();

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
