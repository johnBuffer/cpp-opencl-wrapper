#pragma once

#include <SFML/Graphics.hpp>
#include "ocl_wrapper.hpp"
#include "utils.hpp"
#include "camera_controller.hpp"


constexpr uint32_t CPU_THREADS = 16u;

class Raytracer
{
public:
	Raytracer(uint32_t render_width, uint32_t render_height, uint8_t max_depth, std::vector<LSVONode>& svo_data, float lighting_quality = 1.0f)
		: m_render_dimension(render_width, render_height)
		, m_lighting_quality(lighting_quality)
		, m_wrapper()
		, m_max_depth(max_depth)
		, m_swarm(CPU_THREADS)
		, m_time(0.0f)
		, m_current_lighting_buffer(0)
	{
		m_context = createDefaultContext(m_wrapper);
		initialize(max_depth, svo_data);
	}

	void updateKernelArgs(const Camera& camera)
	{
		// Swap lighting buffers
		/*if (!render_mode) {
		}*/
		//m_current_lighting_buffer = !m_current_lighting_buffer;

		m_time += 0.001f;
		const float scale = 1.0f / (1 << m_max_depth);
		const cl_float3 camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };

		if (first) {
			first = false;
			old_view = camera.rot_mat;
			old_pos = camera_position;
		}

		//m_command_queue.writeInMemoryObject(m_buff_view_matrix_old, true, &old_view[0]);
		m_command_queue.writeInMemoryObject(m_buff_view_matrix, true, &camera.rot_mat[0]);

		m_albedo.setArgument(2, camera_position);
		m_albedo.setArgument(3, m_buff_view_matrix);
		m_albedo.setArgument(6, render_mode);
		m_albedo.setArgument(7, m_time);

		m_lighting.setArgument(1, m_buff_result_lighting[m_current_lighting_buffer]);
		m_lighting.setArgument(2, camera_position);
		m_lighting.setArgument(3, m_buff_view_matrix);
		m_lighting.setArgument(5, m_time);
		m_lighting.setArgument(6, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_lighting.setArgument(7, m_buff_view_matrix_old);
		m_lighting.setArgument(8, old_pos);
		m_lighting.setArgument(9, m_buff_depth);

		m_combinator.setArgument(2, m_buff_result_lighting[!m_current_lighting_buffer]);

		old_view = camera.rot_mat;
		old_pos = camera_position;
	}

	void render()
	{
		const uint32_t area_count = static_cast<uint32_t>(sqrt(CPU_THREADS));
		const uint32_t area_width = m_render_dimension.x / area_count;
		const uint32_t area_height = m_render_dimension.y / area_count;
		// Run albedo kernel
		renderAlbedo();
		// Run lighting kernel
		//renderLighting();
		//biblur();
		//combine();

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

		// To Be Removed
		/*const float scale = 1.0f / (1 << m_max_depth);
		const glm::vec3 camera_position = { camera_ptr->position.x * scale + 1.0f, camera_ptr->position.y * scale + 1.0f, camera_ptr->position.z * scale + 1.0f };
		const uint32_t pt_x(400);
		const uint32_t pt_y(225);
		const uint32_t index = pt_x + pt_y * 800;
		const glm::vec3 point(m_result_points[4 * index + 0], m_result_points[4 * index + 1], m_result_points[4 * index + 2]);
		const glm::vec3 view_point_1 = camera_ptr->rot_mat * (point - camera_position);
		const glm::vec3 view_point_2 = (point - camera_position) * camera_ptr->rot_mat;
		std::cout << "Camera pos   " << camera_position.x << ", " << camera_position.y << ", " << camera_position.z << std::endl;
		std::cout << "View point 1 " << vecToString(view_point_1) << std::endl;
		std::cout << "Test       1 " << vecToString(projVec(view_point_1)) << std::endl << std::endl;*/

		// Wait for threads to terminate
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
		const size_t work_gorup_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_gorup_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);
		//std::cout << work_gorup_width << " " << work_gorup_height << std::endl;
		const size_t globalWorkSize[2] = { work_gorup_width, work_gorup_height };
		const size_t localWorkSize[2] = { 20, 20 };
		m_command_queue.addKernel(m_lighting, 2, NULL, globalWorkSize, localWorkSize);
		m_command_queue.readMemoryObject(m_buff_result_lighting[m_current_lighting_buffer], true, m_result_lighting);
	}

	void biblur()
	{
		const size_t work_gorup_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_gorup_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);
		const size_t globalWorkSize[2] = { work_gorup_width, work_gorup_height };
		const size_t localWorkSize[2] = { 20, 20 };

		m_biblur.setArgument(0, m_buff_result_lighting[m_current_lighting_buffer]);
		m_biblur.setArgument(2, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_command_queue.addKernel(m_biblur, 2, NULL, globalWorkSize, localWorkSize);

		/*m_biblur.setArgument(0, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_biblur.setArgument(2, m_buff_result_lighting[m_current_lighting_buffer]);
		m_command_queue.addKernel(m_biblur, 2, NULL, globalWorkSize, localWorkSize);

		m_biblur.setArgument(0, m_buff_result_lighting[m_current_lighting_buffer]);
		m_biblur.setArgument(2, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_command_queue.addKernel(m_biblur, 2, NULL, globalWorkSize, localWorkSize);

		m_biblur.setArgument(0, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_biblur.setArgument(2, m_buff_result_lighting[m_current_lighting_buffer]);
		m_command_queue.addKernel(m_biblur, 2, NULL, globalWorkSize, localWorkSize);

		m_biblur.setArgument(0, m_buff_result_lighting[m_current_lighting_buffer]);
		m_biblur.setArgument(2, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_command_queue.addKernel(m_biblur, 2, NULL, globalWorkSize, localWorkSize);*/
		//m_command_queue.readMemoryObject(m_buff_result_lighting[m_current_lighting_buffer], true, m_result_lighting);
	}

	void combine()
	{
		const size_t globalWorkSize[2] = { m_render_dimension.x, m_render_dimension.y };
		const size_t localWorkSize[2] = { 40, 20 };
		m_command_queue.addKernel(m_combinator, 2, NULL, globalWorkSize, localWorkSize);
		m_command_queue.readMemoryObject(m_buff_result_albedo, true, m_result_albedo);
	}

	const sf::Image& getAlbedo() const
	{
		return m_output_albedo;
	}

	const sf::Image& getLighting() const
	{
		return m_output_lighting;
	}

	uint8_t render_mode = 1;

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
	oclw::Program m_program_gi;
	oclw::Program m_program_biblur;
	oclw::Program m_program_combinator;
	// Kernels
	oclw::Kernel m_albedo;
	oclw::Kernel m_lighting;
	oclw::Kernel m_biblur;
	oclw::Kernel m_combinator;
	// Resources
	sf::Image m_image_side, m_image_top;
	// Buffers
	oclw::MemoryObject m_buff_svo;
	oclw::MemoryObject m_buff_view_matrix;
	oclw::MemoryObject m_buff_view_matrix_old;
	oclw::MemoryObject m_buff_result_albedo;
	oclw::MemoryObject m_buff_result_lighting[2];
	oclw::MemoryObject m_buff_depth;
	oclw::MemoryObject m_buff_image_top;
	oclw::MemoryObject m_buff_image_side;
	oclw::MemoryObject m_buff_seeds;
	oclw::MemoryObject m_buff_shadow;
	bool m_current_lighting_buffer;

	std::vector<float> m_result_albedo;
	std::vector<float> m_result_lighting;
	std::vector<int32_t> m_seeds;
	// Ouput images
	swrm::Swarm m_swarm;
	sf::Image m_output_albedo;
	sf::Image m_output_lighting;

	// Dev
	//const Camera* camera_ptr;
	bool first = true;
	glm::mat3 old_view;
	cl_float3 old_pos;


private:
	void initialize(uint8_t max_depth, std::vector<LSVONode>& svo)
	{
		// Get devices
		auto& devices_list = m_context.getDevices();
		cl_device_id device = devices_list.front();
		// Create command queue
		m_command_queue = m_context.createQueue(device);
		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_context.createProgram(device, "../src/albedo.cl");
		m_program_gi = m_context.createProgram(device, "../src/lighting.cl");
		m_program_biblur = m_context.createProgram(device, "../src/bilateral_blur.cl");
		m_program_combinator = m_context.createProgram(device, "../src/combinator.cl");
		// Create memory objects that will be used as arguments to kernel
		loadImagesToDevice();
		m_buff_svo = m_context.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		m_buff_view_matrix = m_context.createMemoryObject<float>(9, oclw::ReadOnly);
		m_buff_view_matrix_old = m_context.createMemoryObject<float>(9, oclw::ReadOnly);
		initializeSeeds();
		m_buff_seeds = m_context.createMemoryObject(m_seeds, oclw::ReadWrite | oclw::CopyHostPtr);
		// Create output buffers
		initializeOutputImages();
		const uint64_t albedo_render_pxl_count = m_render_dimension.x * m_render_dimension.y;
		const uint64_t light_render_pxl_count = albedo_render_pxl_count * m_lighting_quality;
		m_result_albedo.resize(albedo_render_pxl_count * 4);
		m_buff_result_albedo = m_context.createMemoryObject<float>(m_result_albedo.size(), oclw::ReadWrite);
		m_buff_shadow = m_context.createMemoryObject<float>(albedo_render_pxl_count, oclw::ReadWrite);
		m_result_lighting.resize(light_render_pxl_count * 4);
		m_buff_result_lighting[0] = m_context.createMemoryObject<float>(m_result_lighting.size(), oclw::ReadWrite);
		m_buff_result_lighting[1] = m_context.createMemoryObject<float>(m_result_lighting.size(), oclw::ReadWrite);
		m_buff_depth = m_context.createMemoryObject<float>(2 * light_render_pxl_count, oclw::ReadWrite);
		// Kernels initialization
		m_albedo = m_program.createKernel("albedo");
		m_albedo.setArgument(0, m_buff_svo);
		m_albedo.setArgument(1, m_buff_result_albedo);
		m_albedo.setArgument(4, m_buff_image_top);
		m_albedo.setArgument(5, m_buff_image_side);
		m_albedo.setArgument(8, m_buff_shadow);

		m_lighting = m_program_gi.createKernel("lighting");
		m_lighting.setArgument(0, m_buff_svo);
		m_lighting.setArgument(4, m_buff_seeds);

		m_biblur = m_program_biblur.createKernel("blur");
		m_biblur.setArgument(1, m_buff_depth);

		m_combinator = m_program_combinator.createKernel("combine");
		m_combinator.setArgument(0, m_buff_result_albedo);
		m_combinator.setArgument(1, m_buff_shadow);
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
