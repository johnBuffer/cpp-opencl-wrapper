#pragma once

#include "camera_controller.hpp"


struct FlyController : public CameraController
{
	void move(const glm::vec3& move_vector, Camera& camera, const Losvo& svo, bool boost) override
	{
		camera.position += move_vector;
	}
};
