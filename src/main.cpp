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
	constexpr uint32_t WIN_WIDTH = 1280;
	constexpr uint32_t WIN_HEIGHT = 720;

	try
	{
		const float lighting_quality = 1.0f;

		const uint8_t max_depth = 7;
		SVO* builder = new SVO(max_depth);
		generateSVO(max_depth, *builder);
		LSVO svo(*builder, max_depth);
		delete builder;

		Raytracer raytracer(10, 10);

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML", sf::Style::Default);
		//window.setMouseCursorVisible(false);

		EventManager event_manager(window);

		sf::Texture tex_lighting;
		sf::Texture tex_albedo;
		sf::RenderTexture lighting_render, tex_lighting_upscale;
		lighting_render.create(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality);
		lighting_render.setSmooth(true);
		tex_lighting_upscale.create(WIN_WIDTH, WIN_HEIGHT);
		tex_lighting_upscale.setSmooth(true);
		Blur blur(WIN_WIDTH, WIN_HEIGHT, 1.0f);

		// Camera
		Camera camera;
		camera.position = glm::vec3(68.7249f, 60, 60);
		camera.last_move = glm::vec3(0.0f);
		//camera.view_angle = glm::vec2(0.395287f, 0.00f);
		camera.view_angle = glm::vec2(0.0f);
		camera.fov = 1.0f;

		FpsController controller;
		controller.updateCameraView(glm::vec2(0.0f), camera);

		while (window.isOpen())
		{
			sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);

			event_manager.processEvents();

			raytracer.render();

			window.clear();

			tex_albedo.loadFromImage(raytracer.getResult());
			sf::Sprite albedo_sprite(tex_albedo);

			window.draw(albedo_sprite);

			const float aim_size = 2.0f;
			sf::RectangleShape aim(sf::Vector2f(aim_size, aim_size));
			aim.setOrigin(aim_size * 0.5f, aim_size * 0.5f);
			aim.setPosition(WIN_WIDTH*0.5f, WIN_HEIGHT*0.5f);
			aim.setFillColor(sf::Color::Green);
			window.draw(aim);

			window.display();
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.what() << std::endl;
	}

	return 0;
}
