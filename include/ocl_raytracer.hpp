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
		, m_wrapper(oclw::GPU)
		, m_max_depth(max_depth)
		, m_swarm(CPU_THREADS)
		, m_time(0.0f)
		, m_current_lighting_buffer(0)
		, frame_count(0)
		, m_current_final_buffer(0)
	{
		initialize(max_depth, svo_data);
	}

	void updateKernelArgs(const Camera& camera)
	{
		// Swap lighting buffers
		m_current_lighting_buffer = !m_current_lighting_buffer;
		++frame_count;

		m_time += 0.001f;
		const float scale = 1.0f / (1 << m_max_depth);
		const cl_float3 camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };

		if (first) {
			first = false;
			old_view = camera.rot_mat;
			old_pos = camera_position;
		}

		m_wrapper.writeInMemoryObject(m_buff_view_matrix_old, &old_view[0], true);
		m_wrapper.writeInMemoryObject(m_buff_view_matrix, &camera.rot_mat[0], true);

		uint32_t albedo_index_count = 0u;
		m_albedo.setArgument(albedo_index_count++, m_buff_svo);
		m_albedo.setArgument(albedo_index_count++, m_buff_result_albedo);
		m_albedo.setArgument(albedo_index_count++, camera_position);
		m_albedo.setArgument(albedo_index_count++, m_buff_view_matrix);
		m_albedo.setArgument(albedo_index_count++, m_buff_image_top);
		m_albedo.setArgument(albedo_index_count++, m_buff_image_side);
		m_albedo.setArgument(albedo_index_count++, render_mode);
		m_albedo.setArgument(albedo_index_count++, m_time);
		m_albedo.setArgument(albedo_index_count++, m_buff_shadow);
		m_albedo.setArgument(albedo_index_count++, m_buff_position);
		m_albedo.setArgument(albedo_index_count++, m_buff_depth[m_current_lighting_buffer]);

		uint32_t gi_index_count = 0u;
		m_lighting.setArgument(gi_index_count++, m_buff_svo);
		m_lighting.setArgument(gi_index_count++, m_buff_result_lighting[m_current_lighting_buffer]);
		m_lighting.setArgument(gi_index_count++, m_buff_noise);
		m_lighting.setArgument(gi_index_count++, m_time);
		m_lighting.setArgument(gi_index_count++, m_buff_result_lighting[!m_current_lighting_buffer]);
		m_lighting.setArgument(gi_index_count++, m_buff_view_matrix_old);
		m_lighting.setArgument(gi_index_count++, old_pos);
		m_lighting.setArgument(gi_index_count++, frame_count);
		m_lighting.setArgument(gi_index_count++, m_buff_depth[m_current_lighting_buffer]);
		m_lighting.setArgument(gi_index_count++, m_buff_depth[!m_current_lighting_buffer]);
		m_lighting.setArgument(gi_index_count++, m_buff_position);

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
		renderLighting();
		normalize();
		biblur();
		median();
		//blur();
		combine();

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
	}

	void renderAlbedo()
	{
		m_wrapper.runKernel(m_albedo, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(20, 20));
	}

	void renderLighting()
	{
		const size_t work_group_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_group_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);
		//std::cout << work_gorup_width << " " << work_gorup_height << std::endl;

		m_wrapper.runKernel(m_lighting, oclw::Size(work_group_width, work_group_height), oclw::Size(20, 20));
	}

	void biblur()
	{
		m_biblur.setArgument(1, m_buff_position);
		const size_t work_group_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_group_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);

		bool current_buffer = 0;
		for (uint8_t i(1); i--;) {
			m_biblur.setArgument(0, m_buff_final_lighting[m_current_final_buffer]);
			m_biblur.setArgument(2, m_buff_final_lighting[!m_current_final_buffer]);
			swapFinalBuffers();
			m_wrapper.runKernel(m_biblur, oclw::Size(work_group_width, work_group_height), oclw::Size(20, 20));

			current_buffer = !current_buffer;
		}
	}

	void blur()
	{
		const size_t work_group_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_group_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);

		for (uint8_t i(3); i--;) {
			m_blur_v.setArgument(0, m_buff_result_lighting[m_current_lighting_buffer]);
			m_blur_v.setArgument(1, m_buff_result_lighting[!m_current_lighting_buffer]);
			m_wrapper.runKernel(m_blur_v, oclw::Size(work_group_width, work_group_height), oclw::Size(10, 10));

			m_blur_h.setArgument(0, m_buff_result_lighting[!m_current_lighting_buffer]);
			m_blur_h.setArgument(1, m_buff_result_lighting[m_current_lighting_buffer]);
			m_wrapper.runKernel(m_blur_h, oclw::Size(work_group_width, work_group_height), oclw::Size(10, 10));
		}
	}

	void median()
	{
		//m_median.setArgument(1, m_buff_position);
		const size_t work_group_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_group_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);
		
		m_median.setArgument(0, m_buff_final_lighting[m_current_final_buffer]);
		m_median.setArgument(1, m_buff_final_lighting[!m_current_final_buffer]);
		swapFinalBuffers();
		m_wrapper.runKernel(m_median, oclw::Size(work_group_width, work_group_height), oclw::Size(10, 10));
	}

	void combine()
	{
		m_combinator.setArgument(2, m_buff_final_lighting[m_current_final_buffer]);
		const size_t globalWorkSize[2] = { m_render_dimension.x, m_render_dimension.y };
		const size_t localWorkSize[2] = { 20, 20 };
		m_wrapper.runKernel(m_combinator, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(20, 20));
		m_wrapper.readMemoryObject(m_buff_result_albedo, m_result_albedo, true);
	}

	void normalize()
	{
		m_normalizer.setArgument(0, m_buff_result_lighting[m_current_lighting_buffer]);
		m_normalizer.setArgument(1, m_buff_final_lighting[m_current_final_buffer]);
		m_wrapper.runKernel(m_normalizer, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(20, 20));
	}

	const sf::Image& getAlbedo() const
	{
		return m_output_albedo;
	}

	const sf::Image& getLighting() const
	{
		return m_output_lighting;
	}

	void mutate(const uint32_t index, uint8_t child_index, uint8_t value)
	{
		m_mutator.setArgument(1, index);
		m_mutator.setArgument(2, child_index);
		m_mutator.setArgument(3, value);
		m_wrapper.runKernel(m_mutator, oclw::Size(1), oclw::Size(1));
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
	oclw::Program m_program;
	oclw::Program m_program_gi;
	oclw::Program m_program_biblur;
	oclw::Program m_program_combinator;
	oclw::Program m_program_blur;
	oclw::Program m_program_median;
	oclw::Program m_program_mutator;
	// Kernels
	oclw::Kernel m_albedo;
	oclw::Kernel m_lighting;
	oclw::Kernel m_biblur;
	oclw::Kernel m_combinator;
	oclw::Kernel m_blur_v;
	oclw::Kernel m_blur_h;
	oclw::Kernel m_median;
	oclw::Kernel m_normalizer;
	oclw::Kernel m_mutator;
	// Resources
	sf::Image m_image_side, m_image_top;
	// Buffers
	oclw::MemoryObject m_buff_svo;
	oclw::MemoryObject m_buff_view_matrix;
	oclw::MemoryObject m_buff_view_matrix_old;
	oclw::MemoryObject m_buff_result_albedo;
	oclw::MemoryObject m_buff_result_lighting[2];
	oclw::MemoryObject m_buff_final_lighting[2];
	oclw::MemoryObject m_buff_depth[2];
	oclw::MemoryObject m_buff_position;
	oclw::MemoryObject m_buff_image_top;
	oclw::MemoryObject m_buff_image_side;
	oclw::MemoryObject m_buff_noise;
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
	uint32_t frame_count;
	bool m_current_final_buffer;


private:
	void initialize(uint8_t max_depth, std::vector<LSVONode>& svo)
	{
		/*oclw::Program test_prog = m_context.createProgram(device, "../src/test.cl");
		oclw::Kernel test_kernel = test_prog.createKernel("test");
		std::vector<float> data(4);
		oclw::MemoryObject mem_object = m_context.createMemoryObject<float>(4, oclw::WriteOnly);
		test_kernel.setArgument(0, mem_object);
		const size_t globalWorkSize = 1;
		const size_t localWorkSize = 1;
		m_command_queue.addKernel(test_kernel, 1, NULL, &globalWorkSize, &localWorkSize);
		m_command_queue.readMemoryObject(mem_object, true, data);
		std::cout << "lol : " << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;
		exit(0);*/


		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_wrapper.createProgramFromFile("../src/albedo.cl");
		m_program_gi = m_wrapper.createProgramFromFile("../src/lighting.cl");
		m_program_biblur = m_wrapper.createProgramFromFile("../src/bilateral_blur.cl");
		m_program_combinator = m_wrapper.createProgramFromFile("../src/combinator.cl");
		m_program_blur = m_wrapper.createProgramFromFile("../src/blur.cl");
		m_program_median = m_wrapper.createProgramFromFile("../src/median.cl");
		m_program_mutator = m_wrapper.createProgramFromFile("../src/mutator.cl");

		// Create memory objects that will be used as arguments to kernel
		loadImagesToDevice();
		m_buff_svo = m_wrapper.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		m_buff_view_matrix = m_wrapper.createMemoryObject<float>(9, oclw::ReadOnly);
		m_buff_view_matrix_old = m_wrapper.createMemoryObject<float>(9, oclw::ReadOnly);
		initializeSeeds();
		m_buff_seeds = m_wrapper.createMemoryObject(m_seeds, oclw::ReadWrite | oclw::CopyHostPtr);
		// Create OpenCL buffers
		initializeOutputImages();
		const uint64_t albedo_render_pxl_count = m_render_dimension.x * m_render_dimension.y;
		const uint64_t light_render_pxl_count = albedo_render_pxl_count * m_lighting_quality * m_lighting_quality;
		m_result_albedo.resize(albedo_render_pxl_count * 4);
		m_buff_result_albedo = m_wrapper.createMemoryObject<float>(m_result_albedo.size(), oclw::ReadWrite);
		m_buff_shadow = m_wrapper.createMemoryObject<float>(albedo_render_pxl_count, oclw::ReadWrite);
		m_result_lighting.resize(light_render_pxl_count * 4);
		m_buff_result_lighting[0] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_result_lighting[1] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_final_lighting[0] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_final_lighting[1] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_depth[0] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RG, oclw::Float);
		m_buff_depth[1] = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RG, oclw::Float);
		m_buff_position = m_wrapper.getContext().createImage2D(m_render_dimension.x * m_lighting_quality, m_render_dimension.y * m_lighting_quality, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);

		// Kernels initialization
		m_albedo = m_program.createKernel("albedo");

		m_lighting = m_program_gi.createKernel("lighting");
		m_normalizer = m_program_gi.createKernel("normalizer");

		m_biblur = m_program_biblur.createKernel("blur");
		//m_biblur.setArgument(1, m_buff_depth);

		m_combinator = m_program_combinator.createKernel("combine");
		m_combinator.setArgument(0, m_buff_result_albedo);
		m_combinator.setArgument(1, m_buff_shadow);

		m_blur_v = m_program_blur.createKernel("blur_v");
		m_blur_h = m_program_blur.createKernel("blur_h");

		m_median = m_program_median.createKernel("median");

		m_mutator = m_program_mutator.createKernel("mutate");
		m_mutator.setArgument(0, m_buff_svo);
	}

	void swapFinalBuffers()
	{
		m_current_final_buffer = !m_current_final_buffer;
	}

	void loadImagesToDevice()
	{
		m_image_side.loadFromFile("../res/grass_side_16x16.bmp");
		m_image_top.loadFromFile("../res/grass_top_16x16.bmp");

		sf::Image noise_image;
		noise_image.loadFromFile("../res/noise.png");

		m_buff_image_top = imageToDevice(m_image_top);
		m_buff_image_side = imageToDevice(m_image_side);
		m_buff_noise = imageToDevice(noise_image);
	}

	oclw::MemoryObject imageToDevice(const sf::Image& image)
	{
		const sf::Vector2u image_size = image.getSize();
		return m_wrapper.getContext().createImage2D(image_size.x, image_size.y, (void*)image.getPixelsPtr(), oclw::ReadOnly | oclw::CopyHostPtr, oclw::RGBA, oclw::Unsigned_INT8);
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
