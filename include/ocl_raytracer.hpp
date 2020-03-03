#pragma once

#include <SFML/Graphics.hpp>
#include "ocl_wrapper.hpp"
#include "utils.hpp"
#include "camera_controller.hpp"


constexpr uint32_t CPU_THREADS = 16u;

class Raytracer
{
public:
	Raytracer(uint32_t render_width, uint32_t render_height, uint8_t max_depth, float lighting_quality = 1.0f)
		: m_render_dimension(render_width, render_height)
		, m_lighting_quality(lighting_quality)
		, m_wrapper()
		, m_max_depth(max_depth)
		, m_swarm(CPU_THREADS)
		, m_time(0.0f)
	{
		m_context = createDefaultContext(m_wrapper);
		initialize(max_depth);
	}

	void updateKernelArgs(const Camera& camera)
	{
		m_time += 0.001f;
		const float scale = 1.0f / (1 << m_max_depth);
		const cl_float3 camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };
		const cl_float2 camera_direction = { camera.view_angle.x, camera.view_angle.y };
		m_command_queue.writeInMemoryObject(m_buff_view_matrix, true, &camera.rot_mat[0]);
		m_albedo.setArgument(2, camera_position);
		m_albedo.setArgument(3, m_buff_view_matrix);

		m_lighting.setArgument(2, camera_position);
		m_lighting.setArgument(3, m_buff_view_matrix);
		m_lighting.setArgument(5, m_time);
	}

	void render()
	{
		const uint32_t area_count = sqrt(CPU_THREADS);
		const uint32_t area_width = m_render_dimension.x / area_count;
		const uint32_t area_height = m_render_dimension.y / area_count;
		// Run albedo kernel
		renderAlbedo();
		auto group_albedo = m_swarm.execute([&](uint32_t thread_id, uint32_t max_thread) {
			const uint32_t start_x = thread_id % 4;
			const uint32_t start_y = thread_id / 4;
			for (uint32_t x(start_x * area_width); x < (start_x + 1) * area_width; ++x) {
				for (uint32_t y(start_y * area_height); y < (start_y + 1) * area_height; ++y) {
					// Computing ray coordinates in 'lens' space ie in normalized screen space
					const uint32_t index = 4 * (x + y * m_render_dimension.x);
					uint8_t r = m_result_albedo[index + 0];
					uint8_t g = m_result_albedo[index + 1];
					uint8_t b = m_result_albedo[index + 2];
					m_output_albedo.setPixel(x, y, sf::Color(r, g, b, 255));
				}
			}
		});
		group_albedo.waitExecutionDone();
		// Run lighting kernel
		renderLighting();
		const uint32_t light_area_width = area_width * m_lighting_quality;
		const uint32_t light_area_height = area_height * m_lighting_quality;
		auto group_lighting = m_swarm.execute([&](uint32_t thread_id, uint32_t max_thread) {
			const uint32_t start_x = thread_id % 4;
			const uint32_t start_y = thread_id / 4;
			for (uint32_t x(start_x * light_area_width); x < (start_x + 1) * light_area_width; ++x) {
				for (uint32_t y(start_y * light_area_height); y < (start_y + 1) * light_area_height; ++y) {
					// Computing ray coordinates in 'lens' space ie in normalized screen space
					const uint32_t index = 4 * (x + y * m_render_dimension.x * m_lighting_quality);
					uint8_t r = m_result_lighting[index + 0];
					uint8_t g = m_result_lighting[index + 1];
					uint8_t b = m_result_lighting[index + 2];
					m_output_lighting.setPixel(x, y, sf::Color(r, g, b, 255));
				}
			}
		});
		// Wait for threads to terminate
		group_lighting.waitExecutionDone();
	}

	void renderAlbedo()
	{
		const size_t globalWorkSize[2] = { m_render_dimension.x, m_render_dimension.y };
		const size_t localWorkSize[2] = { 20, 20 };
		m_command_queue.addKernel(m_albedo, 2, NULL, globalWorkSize, localWorkSize);
		m_command_queue.readMemoryObject(m_buff_result_albedo, true, m_result_albedo);
	}

	void renderLighting()
	{
		const size_t globalWorkSize[2] = { m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality };
		const size_t localWorkSize[2] = { 10, 10 };
		m_command_queue.addKernel(m_lighting, 2, NULL, globalWorkSize, localWorkSize);
		m_command_queue.readMemoryObject(m_buff_result_lighting, true, m_result_lighting);
	}

	const sf::Image& getAlbedo() const
	{
		return m_output_albedo;
	}

	const sf::Image& getLighting() const
	{
		return m_output_lighting;
	}

private:
	// Conf
	sf::Vector2u m_render_dimension;
	float m_lighting_quality;
	const uint8_t m_max_depth;
	float m_time;
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
	oclw::MemoryObject m_buff_result_albedo;
	oclw::MemoryObject m_buff_result_lighting;
	oclw::MemoryObject m_buff_image_top;
	oclw::MemoryObject m_buff_image_side;
	oclw::MemoryObject m_buff_seeds;

	std::vector<uint8_t> m_result_albedo;
	std::vector<uint8_t> m_result_lighting;
	std::vector<int32_t> m_seeds;
	// Ouput images
	swrm::Swarm m_swarm;
	sf::Image m_output_albedo;
	sf::Image m_output_lighting;

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
		m_buff_svo = m_context.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		m_buff_view_matrix = m_context.createMemoryObject<float>(9, oclw::ReadOnly);
		initializeSeeds();
		m_buff_seeds = m_context.createMemoryObject(m_seeds, oclw::ReadWrite | oclw::CopyHostPtr);
		// Create output buffers
		initializeOutputImages();
		m_result_albedo.resize(m_render_dimension.x * m_render_dimension.y * 4);
		m_buff_result_albedo = m_context.createMemoryObject<uint8_t>(m_result_albedo.size(), oclw::WriteOnly);
		m_result_lighting.resize(m_render_dimension.x * m_render_dimension.y * m_lighting_quality * 4);
		m_buff_result_lighting = m_context.createMemoryObject<uint8_t>(m_result_lighting.size(), oclw::WriteOnly);
		// Kernels initialization
		m_albedo = m_program.createKernel("albedo");
		m_albedo.setArgument(0, m_buff_svo);
		m_albedo.setArgument(1, m_buff_result_albedo);
		m_albedo.setArgument(4, m_buff_image_top);
		m_albedo.setArgument(5, m_buff_image_side);

		m_lighting = m_program.createKernel("lighting");
		m_lighting.setArgument(0, m_buff_svo);
		m_lighting.setArgument(1, m_buff_result_lighting);
		m_lighting.setArgument(4, m_buff_seeds);
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

	void initializeSeeds()
	{
		m_seeds.resize(m_render_dimension.x * m_render_dimension.y * m_lighting_quality);
		for (int32_t& s : m_seeds) {
			s = rand();
		}
	}

	void initializeOutputImages()
	{
		m_output_albedo.create(m_render_dimension.x, m_render_dimension.y);
		m_output_lighting.create(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality);
	}
};
