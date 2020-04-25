#pragma once

#include <SFML/Graphics.hpp>
#include "camera_controller.hpp"


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
		, emissive(false)
	{
	}

	void processEvents(CameraController& controller, Camera& camera, LSVO& svo, uint8_t& render_mode)
	{
		glm::vec3 move = glm::vec3(0.0f);
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();

			else if (event.type == sf::Event::MouseButtonPressed) {
				emissive = true;
				if (event.mouseButton.button == sf::Mouse::Left) {
					value = 1;
				} else if (event.mouseButton.button == sf::Mouse::Right) {
					value = 0;
				}
			}
			else if (event.type == sf::Event::MouseButtonReleased) {
				emissive = false;
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
				case sf::Keyboard::F:
				{
					const uint32_t svo_size = 1 << svo.max_depth;
					const glm::vec3 ray = camera.camera_vec;
					const HitPoint point = svo.castRay(camera.position, ray);
					if (point.hit) {
						mutate_waiting = true;
						index = point.global_index;
						child_index = point.child_index;
					}
					break;
				}
				case sf::Keyboard::Q:
					left = true;
					break;
				case sf::Keyboard::D:
					right = true;
					break;
				case sf::Keyboard::O:
					render_mode = !render_mode;
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

		if (emissive) {
			const uint32_t svo_size = 1 << svo.max_depth;
			const glm::vec3 ray = camera.camera_vec;
			const HitPoint point = svo.castRay(camera.position, ray);
			if (point.hit) {
				mutate_waiting = true;
				index = point.global_index;
				child_index = point.child_index;
			}
		}
	}

	sf::Clock clock;
	bool forward, left, right, up, backward, boost, mouse_control;
	bool mutate_waiting;
	uint32_t index;
	uint8_t child_index;
	uint8_t value;

	bool emissive;

private:
	sf::RenderWindow& window;
};