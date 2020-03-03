#pragma once

#include <SFML/Graphics.hpp>
#include "ocl_wrapper.hpp"
#include "utils.hpp"
#include "camera_controller.hpp"


class Raytracer
{
public:
	Raytracer(uint32_t render_width, uint32_t render_height, uint8_t max_depth, float lighting_quality = 1.0f)
		: m_render_dimension(render_width, render_height)
		, m_lighting_quality(lighting_quality)
		, m_wrapper()
		, m_max_depth(max_depth)
	{
		m_context = createDefaultContext(m_wrapper);
		initialize(max_depth);
	}

	void updateKernelArgs(const Camera& camera)
	{
		const float scale = 1.0f / (1 << m_max_depth);
		const cl_float3 camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };
		const cl_float2 camera_direction = { camera.view_angle.x, camera.view_angle.y };
		m_command_queue.writeInMemoryObject(m_buff_view_matrix, true, &camera.rot_mat[0]);
		m_albedo.setArgument(2, camera_position);
		m_albedo.setArgument(3, m_buff_view_matrix);
	}

	void render()
	{
		renderAlbedo();
	}

	void renderAlbedo()
	{
		const size_t globalWorkSize[2] = { m_render_dimension.x, m_render_dimension.y };
		const size_t localWorkSize[2] = { 20, 20 };
		m_command_queue.addKernel(m_albedo, 2, NULL, globalWorkSize, localWorkSize);
		m_command_queue.readMemoryObject(m_buff_result, true, m_result);
	}

	const std::vector<uint8_t>& getResult() const
	{
		return m_result;
	}

private:
	// Conf
	sf::Vector2u m_render_dimension;
	float m_lighting_quality;
	const uint8_t m_max_depth;
	// OpenCL
	oclw::Wrapper m_wrapper;
	oclw::Context m_context;
	oclw::CommandQueue m_command_queue;
	oclw::Program m_program;
	oclw::Kernel m_albedo;
	oclw::Kernel m_lighting;
	// Resources
	sf::Image m_image_side, m_image_top;
	// Buffers
	oclw::MemoryObject m_buff_svo;
	oclw::MemoryObject m_buff_view_matrix;
	oclw::MemoryObject m_buff_result;
	oclw::MemoryObject m_buff_image_top;
	oclw::MemoryObject m_buff_image_side;
	std::vector<uint8_t> m_result;

private:
	void initialize(uint8_t max_depth)
	{
		// Get devices
		auto& devices_list = m_context.getDevices();
		cl_device_id device = devices_list.front();
		// Create command queue
		m_command_queue = m_context.createQueue(device);
		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_context.createProgram(device, "../src/voxel.cl");
		// Create memory objects that will be used as arguments to kernel
		loadImagesToDevice();
		std::vector<LSVONode> svo = generateSVO(max_depth);
		m_result.resize(m_render_dimension.x * m_render_dimension.y * 4);
		m_buff_svo = m_context.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		m_buff_view_matrix = m_context.createMemoryObject<float>(9, oclw::ReadOnly);
		m_buff_result = m_context.createMemoryObject<uint8_t>(m_result.size(), oclw::WriteOnly);
		// Kernels initialization
		m_albedo = m_program.createKernel("albedo");
		m_albedo.setArgument(0, m_buff_svo);
		m_albedo.setArgument(1, m_buff_result);
		m_albedo.setArgument(4, m_buff_image_top);
		m_albedo.setArgument(5, m_buff_image_side);
	}

	void loadImagesToDevice()
	{
		m_image_side.loadFromFile("../res/grass_side_16x16.bmp");
		m_image_top.loadFromFile("../res/grass_top_16x16.bmp");
		const sf::Vector2u top_size = m_image_top.getSize();
		m_buff_image_top = imageToDevice(m_image_top);
		m_buff_image_side = imageToDevice(m_image_side);
	}

	oclw::MemoryObject imageToDevice(const sf::Image& image)
	{
		const sf::Vector2u image_size = image.getSize();
		return m_context.createImage2D(image_size.x, image_size.y, (void*)image.getPixelsPtr(), oclw::ReadOnly | oclw::CopyHostPtr);
	}
};
