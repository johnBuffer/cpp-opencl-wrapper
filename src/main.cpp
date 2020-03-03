#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>
#include <SFML/Graphics.hpp>
#include "event_manager.hpp"
#include <swarm.hpp>

#include "utils.hpp"
#include "fly_controller.hpp"
#include "ocl_raytracer.hpp"
#include "dynamic_blur.hpp"


int main()
{
	constexpr uint32_t WIN_WIDTH = 1920u;
	constexpr uint32_t WIN_HEIGHT = 1080u;

	try
	{
		const float lighting_quality = 0.5f;
		Raytracer raytracer(WIN_WIDTH, WIN_HEIGHT, 9, lighting_quality);

		// Main loop
		sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "OpenCL and SFML", sf::Style::Fullscreen);
		window.setMouseCursorVisible(false);

		EventManager event_manager(window);

		sf::Texture tex_lighting;
		sf::Texture tex_albedo;
		sf::RenderTexture lighting_render, lighting_render2;
		lighting_render.create(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality);
		lighting_render2.create(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality);
		Blur blur(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality, 1.0f);

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

			raytracer.updateKernelArgs(camera);
			raytracer.render();

			window.clear();

			tex_lighting.loadFromImage(raytracer.getLighting());
			tex_albedo.loadFromImage(raytracer.getAlbedo());

			// Add some persistence to reduce the noise
			const float old_value_conservation = 0.5f;
			const float c1 = 255 * old_value_conservation;
			const float c2 = 255 * (1.0f - old_value_conservation);
			sf::RectangleShape cache1(sf::Vector2f(WIN_WIDTH * lighting_quality, WIN_HEIGHT * lighting_quality));
			cache1.setFillColor(sf::Color(c1, c1, c1, 255));
			sf::RectangleShape cache2 = cache1;
			cache2.setFillColor(sf::Color(c2, c2, c2, 255));
			// Draw image to final render texture
			lighting_render2.draw(sf::Sprite(lighting_render.getTexture()));
			lighting_render2.draw(cache1, sf::BlendMultiply);
			lighting_render2.display();

			lighting_render.draw(sf::Sprite(tex_lighting));
			lighting_render.draw(cache2, sf::BlendMultiply);
			lighting_render.draw(sf::Sprite(lighting_render2.getTexture()), sf::BlendAdd);
			lighting_render.display();

			sf::Sprite lighting_sprite(lighting_render.getTexture());
			sf::Sprite albedo_sprite(tex_albedo);
			lighting_sprite.setScale(1.0f / lighting_quality, 1.0f / lighting_quality);

			window.draw(albedo_sprite);
			window.draw(lighting_sprite, sf::BlendMultiply);
			//window.draw(lighting_sprite);


			window.display();
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.getReadableError() << std::endl;
	}

	return 0;
}
