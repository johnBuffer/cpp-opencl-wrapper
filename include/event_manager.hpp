#pragma once

#include <SFML/Graphics.hpp>
#include "camera_controller.hpp"
#include "scene.hpp"
#include "losvo.hpp"


struct EventManager
{
	EventManager(sf::RenderWindow& window_)
		: window(window_)
		, forward(false)
		, left(false)
		, right(false)
		, up(false)
		, backward(false)
		, mouse_control(true)
		, boost(false)
		, mutate_waiting(false)
		, mutate(false)
		, mutate_ready(true)
		, block_type(1)
		, moving_sun(0)
	{
	}

	void processEvents(CameraController& controller, Camera& camera, Losvo& svo, SceneSettings& scene, sf::Vector2f& sun)
	{
		glm::vec3 move = glm::vec3(0.0f);
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			else if (event.type == sf::Event::MouseButtonPressed) {
				mutate = true;
				if (event.mouseButton.button == sf::Mouse::Left) {
					value = 0;
				} else if (event.mouseButton.button == sf::Mouse::Right) {
					value = 1;
				}
			}
			else if (event.type == sf::Event::MouseButtonReleased) {
				mutate = false;
				mutate_ready = true;
			}

			if (event.type == sf::Event::KeyPressed) {
				switch (event.key.code) {
				case sf::Keyboard::Escape:
					window.close();
					break;
				case sf::Keyboard::Z:
					forward = true;
					break;
				case sf::Keyboard::S:
					backward = true;
					break;
				case sf::Keyboard::P:
					scene.light_radius += 0.02f;
					break;
				case sf::Keyboard::O:
					scene.light_radius -= 0.02f;
					scene.light_radius = std::max(0.0f, scene.light_radius);
					break;
				case sf::Keyboard::Num1:
					block_type = 1;
					break;
				case sf::Keyboard::Num2:
					block_type = 2;
					break;
				case sf::Keyboard::Num3:
					block_type = 3;
					break;
				case sf::Keyboard::Num4:
					block_type = 4;
					break;
				case sf::Keyboard::Num5:
					block_type = 5;
					break;
				case sf::Keyboard::Num6:
					block_type = 6;
					break;
				case sf::Keyboard::Num7:
					block_type = 7;
					break;
				case sf::Keyboard::Num8:
					block_type = 8;
					break;
				case sf::Keyboard::Numpad4:
					sun.x += 0.2f;
					break;
				case sf::Keyboard::Numpad6:
					sun.x -= 0.2f;
					break;
				case sf::Keyboard::Numpad8:
					sun.y -= 0.2f;
					break;
				case sf::Keyboard::Numpad2:
					sun.y += 0.2f;
					break;
				case sf::Keyboard::Numpad7:
					scene.light_intensity -= 0.5f;
					scene.light_intensity = std::max(0.0f, scene.light_intensity);
					break;
				case sf::Keyboard::Numpad9:
					scene.light_intensity += 0.5f;
					break;
				case sf::Keyboard::F:
				{
					moving_sun = !moving_sun;
					break;
				}
				case sf::Keyboard::Q:
					left = true;
					break;
				case sf::Keyboard::D:
					right = true;
					break;
				case sf::Keyboard::Space:
					up = true;
					controller.jump();
					break;
				case sf::Keyboard::E:
					mouse_control = !mouse_control;
					window.setMouseCursorVisible(!mouse_control);
					break;
				case sf::Keyboard::Up:
					break;
				case sf::Keyboard::Down:
					break;
				case sf::Keyboard::Right:
					camera.aperture += 0.1f;
					break;
				case sf::Keyboard::Left:
					camera.aperture -= 0.1f;
					if (camera.aperture < 0.0f) {
						camera.aperture = 0.0f;
					}
					break;
				case sf::Keyboard::R:
					break;
				case sf::Keyboard::G:
					break;
				case sf::Keyboard::H:
					break;
				case sf::Keyboard::LShift:
					boost = true;
					break;
				default:
					break;
				}
			}
			else if (event.type == sf::Event::KeyReleased) {
				switch (event.key.code) {
				case sf::Keyboard::Z:
					forward = false;
					break;
				case sf::Keyboard::S:
					backward = false;
					break;
				case sf::Keyboard::Q:
					left = false;
					break;
				case sf::Keyboard::D:
					right = false;
					break;
				case sf::Keyboard::F:
				{
					break;
				}
				case sf::Keyboard::Space:
					up = false;
					break;
				case sf::Keyboard::LShift:
					boost = false;
					break;
				default:
					break;
				}
			}
		}

		const float boost_value = boost ? 10.0f : 5.0f;
		if (forward) {
			move += camera.camera_vec;
		}
		else if (backward) {
			move -= camera.camera_vec;
		}

		if (left) {
			move += glm::vec3(-camera.camera_vec.z, 0.0f, camera.camera_vec.x);
		}
		else if (right) {
			move -= glm::vec3(-camera.camera_vec.z, 0.0f, camera.camera_vec.x);
		}

		if (up) {
			move += glm::vec3(0.0f, -1.0f, 0.0f);
		}

		controller.move(move, camera, svo, boost);

		if (mutate && mutate_ready) {
			mutate_ready = false;
			const uint32_t svo_size = 1 << svo.levels;
			const glm::vec3 ray = camera.camera_vec;
			const HitPoint point = svo.castRay(camera.position, ray);
			if (point.hit) {
				mutate_waiting = true;
				if (value) {
					// Add
					const glm::ivec3 hit_position = point.position;
					const glm::uvec3 add_position = hit_position + glm::ivec3(point.normal);
					mutations = svo.addCell(add_position.x, add_position.y, add_position.z, block_type);
				}
				else {
					// Remove
					const glm::ivec3 hit_position = point.position;
					mutations = svo.removeCell(hit_position.x, hit_position.y, hit_position.z);
				}
			}
		}
	}

	sf::Clock clock;
	bool forward, left, right, up, backward, boost, mouse_control;
	bool mutate_waiting;

	bool moving_sun;
	int8_t block_type;
	uint8_t value;
	std::vector<Mutation> mutations;

	bool mutate, mutate_ready;

private:
	sf::RenderWindow& window;
};