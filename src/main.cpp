
#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>
#include "event_manager.hpp"
#include <swarm.hpp>

#include "utils.hpp"
#include "fly_controller.hpp"
#include "fps_controller.hpp"
#include "ocl_raytracer.hpp"
#include "dynamic_blur.hpp"
#include "lsvo.hpp"



int main()
{
	constexpr uint32_t WIN_WIDTH = 1600;
	constexpr uint32_t WIN_HEIGHT = 900;

	try
	{
		const float lighting_quality = 1.0f;

		const uint8_t max_depth = 9;
		SVO* builder = new SVO(max_depth);
		generateSVO(max_depth, *builder);
		LSVO svo(*builder, max_depth);
		delete builder;

		Raytracer raytracer(WIN_WIDTH, WIN_HEIGHT, max_depth, svo.data, lighting_quality);

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML", sf::Style::Default);
		window.setMouseCursorVisible(false);

		EventManager event_manager(window);

		sf::Texture tex_albedo;

		sf::Vector2f sun(0.0f, 0.0f);
		SceneSettings scene;
		scene.light_intensity = 1.5f;
		scene.light_position = { 0.0f, 0.0f, 0.0f };
		scene.light_radius = 0.0f;

		// Camera
		Camera camera;
		camera.position = glm::vec3(345.317f, 322.447f, 432.477f);
		camera.last_move = glm::vec3(0.0f);
		camera.view_angle = glm::vec2(3.61429, -0.72);
		camera.fov = 1.0f;

		sf::Mouse::setPosition(sf::Vector2i(WIN_WIDTH / 2, WIN_HEIGHT / 2), window);
		FpsController controller;

		while (window.isOpen())
		{
			sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
			event_manager.processEvents(controller, camera, svo, scene, sun);

			if (event_manager.mouse_control) {
				sf::Mouse::setPosition(sf::Vector2i(WIN_WIDTH / 2, WIN_HEIGHT / 2), window);
				const float mouse_sensitivity = 0.0015f;
				controller.updateCameraView(mouse_sensitivity * glm::vec2(mouse_pos.x - WIN_WIDTH * 0.5f, (WIN_HEIGHT  * 0.5f) - mouse_pos.y), camera);
			}

			const float sun_trajectory_radius = 2.0f;
			scene.light_position = { 1.5f + sun_trajectory_radius * cos(sun.x), sun.y, 1.5f + sun_trajectory_radius * sin(sun.x) };

			//std::cout << camera.view_angle.x << " " << camera.view_angle.y << std::endl;
			//std::cout << camera.position.x << " " << camera.position.y << " " << camera.position.z << std::endl;

			if (event_manager.mutate_waiting) {
				event_manager.mutate_waiting = false;
				raytracer.mutate(event_manager.index, event_manager.child_index, event_manager.value);
			}

			raytracer.updateKernelArgs(camera, scene);
			raytracer.render();

			window.clear();

			tex_albedo.loadFromImage(raytracer.getAlbedo());
			sf::Sprite albedo_sprite(tex_albedo);

			window.draw(albedo_sprite);

			/*const float aim_size = 2.0f;
			sf::RectangleShape aim(sf::Vector2f(aim_size, aim_size));
			aim.setOrigin(aim_size * 0.5f, aim_size * 0.5f);
			aim.setPosition(WIN_WIDTH*0.5f, WIN_HEIGHT*0.5f);
			aim.setFillColor(sf::Color::Green);
			window.draw(aim);*/

			window.display();
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.what() << std::endl;
	}

	return 0;
}