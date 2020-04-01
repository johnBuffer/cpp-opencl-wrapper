#pragma once

#include <glm/glm.hpp>
#include "utils.hpp"
#include "lsvo.hpp"
#include <glm/gtx/transform.hpp>


constexpr float PI = 3.141592653f;


struct CameraRay
{
	glm::vec3 ray;
	glm::vec3 world_rand_offset;
};

struct Camera
{
	glm::vec3 position;
	glm::vec2 view_angle;
	glm::vec3 camera_vec;
	glm::mat3 rot_mat;

	float aperture = 0.0f;
	float focal_length = 1.0f;
	float fov = 1.0f;

	void setViewAngle(const glm::vec2& angle)
	{
		view_angle = angle;
		rot_mat = generateRotationMatrix(view_angle);
		camera_vec = viewToWorld(glm::vec3(0.0f, 0.0f, 1.0f));
	}

	CameraRay getRay(const glm::vec2& lens_position)
	{
		const glm::vec3 screen_position = glm::vec3(lens_position, fov);
		const glm::vec3 ray = glm::normalize(screen_position);

		CameraRay result;
		result.ray = viewToWorld(ray);

		return result;
	}

	glm::vec3 viewToWorld(const glm::vec3& v) const
	{
		return v * rot_mat;
	}

	const glm::mat4 getViewMatrix() const
	{
		const glm::vec3 up_vector = glm::vec3(0, 1, 0);
		glm::mat4 camera = glm::rotate(glm::mat4(), view_angle.x, up_vector);
		const glm::vec3 pitch_vector = glm::vec3(1, 0, 0);
		camera = glm::rotate(camera, view_angle.y, pitch_vector);
		return glm::inverse(camera);
	}
};


struct CameraController
{
	virtual void updateCameraView(const glm::vec2& d_view_angle, Camera& camera)
	{
		glm::vec2 new_angle = camera.view_angle + d_view_angle;
		glm::clamp(new_angle.y, -PI * 0.5f, PI * 0.5f);

		camera.setViewAngle(new_angle);
	}

	virtual void move(const glm::vec3& move_vector, Camera& camera, const LSVO& svo, bool boost) = 0;

	virtual void forward() = 0;
	virtual void jump() = 0;

	float movement_speed = 0.25f;
};
