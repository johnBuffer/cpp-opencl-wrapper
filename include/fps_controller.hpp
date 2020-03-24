#pragma once
#include "camera_controller.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>


struct FpsController : public CameraController
{
	void move(const glm::vec3& move_vector, Camera& camera, const LSVO& svo) override
	{
		const float body_height = 1.8f;
		const float body_radius = 0.5f;

		const float elapsed_time = clock.restart().asSeconds();
		v += elapsed_time * g;
		camera.position += glm::vec3(move_vector.x, 0.0f, move_vector.z) * movement_speed * elapsed_time;
		camera.position.y += v * elapsed_time;

		const HitPoint yp_ray = svo.castRay(camera.position, glm::vec3(0.0f, 1.0f, 0.0f));
		if (yp_ray.cell) {
			if (yp_ray.distance < body_height) {
				camera.position.y = yp_ray.position.y - body_height;
				v = 0.0f;
			}
		}

		const HitPoint xp_ray = svo.castRay(camera.position + glm::vec3(0.0f, 1.5f, 0.0f) , glm::vec3(1.0f, 0.0f, 0.0f));
		if (xp_ray.cell) {
			if (xp_ray.distance < body_radius) {
				camera.position.x = xp_ray.position.x - body_radius;
			}
		}

		const HitPoint xn_ray = svo.castRay(camera.position + glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
		if (xn_ray.cell) {
			if (xn_ray.distance < body_radius) {
				camera.position.x = xn_ray.position.x + body_radius;
			}
		}

		const HitPoint zp_ray = svo.castRay(camera.position + glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		if (zp_ray.cell) {
			if (zp_ray.distance < body_radius) {
				camera.position.z = zp_ray.position.z - body_radius;
			}
		}

		const HitPoint zn_ray = svo.castRay(camera.position + glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		if (zn_ray.cell) {
			if (zn_ray.distance < body_radius) {
				camera.position.z = zn_ray.position.z + body_radius;
			}
		}
	}

	void forward()
	{
	}

	void jump()
	{
		v = -5.0f;
	}

	const float movement_speed = 7.0f;
	float v = 0.0f;
	const float g = 9.81;
	sf::Clock clock;
};