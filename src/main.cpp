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
		const float lighting_quality = 0.5f;

		const uint8_t max_depth = 8;
		SVO* builder = new SVO(max_depth);
		generateSVO(max_depth, *builder);
		LSVO svo(*builder, max_depth);
		delete builder;

		Raytracer raytracer(WIN_WIDTH, WIN_HEIGHT, max_depth, svo.data, lighting_quality);

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML", sf::Style::Default);
		window.setMouseCursorVisible(false);

		EventManager event_manager(window);

		sf::Texture tex_lighting;
		sf::Texture tex_albedo;
		sf::RenderTexture lighting_render, tex_lighting_upscale;
		lighting_render.create(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality);
		tex_lighting_upscale.create(WIN_WIDTH, WIN_HEIGHT);
		Blur blur(WIN_WIDTH, WIN_HEIGHT, 1.0f);

		sf::Shader median; 
		median.loadFromFile("../res/median.frag", sf::Shader::Fragment);

		// Camera
		Camera camera;
		camera.position = glm::vec3(100, 20, 100);
		camera.view_angle = glm::vec2(0.0f);
		camera.fov = 1.0f;

		FpsController controller;

		while (window.isOpen())
		{
			sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
			event_manager.processEvents(controller, camera, svo, raytracer.render_mode);

			if (event_manager.mouse_control) {
				sf::Mouse::setPosition(sf::Vector2i(WIN_WIDTH / 2, WIN_HEIGHT / 2), window);
				const float mouse_sensitivity = 0.0015f;
				controller.updateCameraView(mouse_sensitivity * glm::vec2(mouse_pos.x - WIN_WIDTH * 0.5f, (WIN_HEIGHT  * 0.5f) - mouse_pos.y), camera);
			}

			raytracer.updateKernelArgs(camera);
			raytracer.render();

			window.clear();

			tex_lighting.loadFromImage(raytracer.getLighting());
			tex_albedo.loadFromImage(raytracer.getAlbedo());

			sf::Sprite lighting_sprite(tex_lighting);
			lighting_sprite.setScale(1.0f / lighting_quality, 1.0f / lighting_quality);
			tex_lighting_upscale.draw(lighting_sprite);
			tex_lighting_upscale.display();
			sf::Sprite lighting_sprite_upscale(blur.apply(tex_lighting_upscale.getTexture(), 1));

			sf::Sprite albedo_sprite(tex_albedo);

			window.draw(albedo_sprite);
			if (raytracer.render_mode == 1) {
				window.draw(lighting_sprite_upscale, sf::BlendMultiply);
			}
			//window.draw(lighting_sprite_upscale);


			window.display();
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.getReadableError() << std::endl;
	}

	return 0;
}
