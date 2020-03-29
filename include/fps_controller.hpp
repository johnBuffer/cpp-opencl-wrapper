#pragma once
#include "camera_controller.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include "sound_player.hpp"


struct FpsController : public CameraController
{
	FpsController()
		: can_jump(false)
	{
		steps.push_back(SoundPlayer::registerSound("../res/Sounds/step1.flac"));
		steps.push_back(SoundPlayer::registerSound("../res/Sounds/step2.flac"));
		steps.push_back(SoundPlayer::registerSound("../res/Sounds/step3.flac"));
		steps.push_back(SoundPlayer::registerSound("../res/Sounds/step4.flac"));

		fall = SoundPlayer::registerSound("../res/Sounds/fall.flac");
		start_jump = SoundPlayer::registerSound("../res/Sounds/jump.flac");
	}

	void move(const glm::vec3& move_vector, Camera& camera, const LSVO& svo, bool boost) override
	{
		const float body_height = 1.8f;
		const float body_radius = 0.2f;
		const float feet_eps = 0.05f;

		const float elapsed_time = clock.restart().asSeconds();
		

		if (boost) {
			camera.position += 10.0f * move_vector * movement_speed * elapsed_time;
		}
		else {
			v += elapsed_time * g;
			camera.position += glm::vec3(move_vector.x, 0.0f, move_vector.z) * movement_speed * elapsed_time;
			camera.position.y += v * elapsed_time;
		}

		// Ground check
		const HitPoint yp_ray = svo.castRay(camera.position, glm::vec3(0.0f, 1.0f, 0.0f));
		if (yp_ray.hit) {
			if (yp_ray.distance < body_height) {
				can_jump = true;
				camera.position.y = yp_ray.position.y - body_height;
				if (v * elapsed_time > 0.1f && yp_ray.cell.type != Cell::Mirror) {
					//SoundPlayer::playInstanceOf(fall);
				}
				v = 0.0f;
			}
		}

		if (!v) {
			const float dist = glm::distance(camera.position, last_position);
			if (dist >= step_size && yp_ray.cell.type != Cell::Mirror) {
				last_position = camera.position;
				//SoundPlayer::playInstanceOf(steps[rand() % steps.size()]);
			}
		}

		const glm::vec3 feet_offset(0.0f, body_height - feet_eps, 0.0f);
		const glm::vec3 feet_position = camera.position + feet_offset;

		const HitPoint xp_ray = svo.castRay(feet_position, glm::vec3(1.0f, 0.0f, 0.0f));
		if (xp_ray.hit) {
			if (xp_ray.distance < body_radius) {
				camera.position.x = xp_ray.position.x - body_radius;
			}
		}

		const HitPoint xn_ray = svo.castRay(feet_position, glm::vec3(-1.0f, 0.0f, 0.0f));
		if (xn_ray.hit) {
			if (xn_ray.distance < body_radius) {
				camera.position.x = xn_ray.position.x + body_radius;
			}
		}

		const HitPoint zp_ray = svo.castRay(feet_position, glm::vec3(0.0f, 0.0f, 1.0f));
		if (zp_ray.hit) {
			if (zp_ray.distance < body_radius) {
				camera.position.z = zp_ray.position.z - body_radius;
			}
		}

		const HitPoint zn_ray = svo.castRay(feet_position, glm::vec3(0.0f, 0.0f, -1.0f));
		if (zn_ray.hit) {
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
		if (can_jump) {
			//SoundPlayer::playInstanceOf(start_jump);
			can_jump = false;
			v = -8.5f;
		}
	}

	const float movement_speed = 7.0f;

	std::vector<size_t> steps;
	size_t fall;
	size_t start_jump;

	bool can_jump;
	float v = 0.0f;
	const float g = 24.0f;
	sf::Clock clock;

	const float step_size = 3.0f;
	glm::vec3 last_position;
};