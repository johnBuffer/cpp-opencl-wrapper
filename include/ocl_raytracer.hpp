#pragma once

#include <SFML/Graphics.hpp>
#include "ocl_wrapper.hpp"
#include "utils.hpp"
#include "camera_controller.hpp"
#include "scene.hpp"
#include "double_buffer.hpp"
#include "denoiser.hpp"


constexpr uint32_t CPU_THREADS = 16u;


#pragma pack(push, 1)
typedef struct cl_HitPoint
{
	cl_float3 position;
	cl_float3 normal;
	cl_float  distance;
	cl_char   cell_type;
	cl_float2 tex_coords;
} cl_HitPoint;
#pragma pack(pop)


class Raytracer
{
public:
	Raytracer(uint32_t render_width, uint32_t render_height, uint8_t max_depth, std::vector<uint8_t>& svo_data, float lighting_quality = 1.0f)
		: m_render_dimension(render_width, render_height)
		, m_lighting_quality(lighting_quality)
		, m_wrapper(oclw::GPU)
		, m_max_depth(max_depth)
		, m_swarm(CPU_THREADS)
		, m_time(0.0f)
		, frame_count(0)
		, m_gi_denoiser(m_wrapper, { render_width, render_height }, 2)
		, m_shadows_denoiser(m_wrapper, { render_width, render_height }, 1)
	{
		initialize(max_depth, svo_data);
	}

	void updateKernelArgs(const Camera& camera, SceneSettings scene)
	{
		m_buff_depths.swap();
		++frame_count;

		m_time += 0.001f;
		const float scale = 1.0f / (1 << m_max_depth);
		camera_position = { camera.position.x * scale + 1.0f, camera.position.y * scale + 1.0f, camera.position.z * scale + 1.0f };

		if (first) {
			first = false;
			old_view = camera.rot_mat;
			old_pos = camera_position;
		}

		m_wrapper.writeInMemoryObject(m_buff_view_matrix, &camera.rot_mat[0], true);

		scene.camera_position = camera_position;
		scene.time = m_time;

		uint32_t albedo_index_count = 0u;
		m_albedo.setArgument(albedo_index_count++, m_buff_svo);
		m_albedo.setArgument(albedo_index_count++, scene);
		m_albedo.setArgument(albedo_index_count++, m_buff_albedo);
		m_albedo.setArgument(albedo_index_count++, m_buff_view_matrix);
		m_albedo.setArgument(albedo_index_count++, m_buff_image_top);
		m_albedo.setArgument(albedo_index_count++, m_buff_image_side);
		m_albedo.setArgument(albedo_index_count++, m_buff_depths.getCurrent());
		m_albedo.setArgument(albedo_index_count++, m_buff_ss_positions);

		uint32_t gi_index_count = 0u;
		m_lighting.setArgument(gi_index_count++, m_buff_svo);
		m_lighting.setArgument(gi_index_count++, scene);
		m_lighting.setArgument(gi_index_count++, m_buff_result_gi);
		m_lighting.setArgument(gi_index_count++, m_buff_result_shadows);
		m_lighting.setArgument(gi_index_count++, m_buff_noise);
		m_lighting.setArgument(gi_index_count++, frame_count);
		m_lighting.setArgument(gi_index_count++, m_buff_depths.getCurrent());
		m_lighting.setArgument(gi_index_count++, m_buff_ss_positions);

		cur_view = camera.rot_mat;
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

		m_gi_denoiser.execute(old_view, old_pos, m_buff_result_gi, m_buff_ss_positions, m_buff_depths);
		m_shadows_denoiser.execute(old_view, old_pos, m_buff_result_shadows, m_buff_ss_positions, m_buff_depths);

		combine();

		auto group_albedo = m_swarm.execute([&](uint32_t thread_id, uint32_t max_thread) {
			const uint32_t start_x = thread_id % 4;
			const uint32_t start_y = thread_id / 4;
			for (uint32_t x(start_x * area_width); x < (start_x + 1) * area_width; ++x) {
				for (uint32_t y(start_y * area_height); y < (start_y + 1) * area_height; ++y) {
					const uint32_t index = 4 * (x + y * m_render_dimension.x);
					uint8_t r = as<uint8_t>(m_result_albedo[index + 0]);
					uint8_t g = as<uint8_t>(m_result_albedo[index + 1]);
					uint8_t b = as<uint8_t>(m_result_albedo[index + 2]);
					m_output_albedo.setPixel(x, y, sf::Color(r, g, b, 255));
				}
			}
		});
		group_albedo.waitExecutionDone();

		old_view = cur_view;
		old_pos = camera_position;
	}

	void renderAlbedo()
	{
		m_wrapper.runKernel(m_albedo, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(work_group_size, work_group_size));
	}

	void renderLighting()
	{
		const size_t work_group_width = static_cast<size_t>(m_render_dimension.x * m_lighting_quality);
		const size_t work_group_height = static_cast<size_t>(m_render_dimension.y * m_lighting_quality);
		m_wrapper.runKernel(m_lighting, oclw::Size(work_group_width, work_group_height), oclw::Size(work_group_size, work_group_size));
	}

	void combine()
	{
		m_combinator.setArgument(2, m_gi_denoiser.getResult());
		m_combinator.setArgument(3, m_shadows_denoiser.getResult());
		m_wrapper.runKernel(m_combinator, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(work_group_size, work_group_size));
		m_wrapper.readImageObject(m_buff_final_image, m_result_albedo, true);
	}

	const sf::Image& getAlbedo() const
	{
		return m_output_albedo;
	}

	void mutate(const std::vector<Mutation>& mutations)
	{
		m_wrapper.writeInMemoryObject(m_buff_mutations, mutations.data(), true);
		m_mutator.setArgument(1, m_buff_mutations);
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
	oclw::Program m_program_combinator;
	oclw::Program m_program_mutator;
	// Kernels
	oclw::Kernel m_albedo;
	oclw::Kernel m_lighting;
	oclw::Kernel m_combinator;
	oclw::Kernel m_mutator;
	// Resources
	sf::Image m_image_side, m_image_top;
	// Buffers
	oclw::MemoryObject m_buff_svo;
	oclw::MemoryObject m_buff_mutations;
	oclw::MemoryObject m_buff_view_matrix;
	oclw::MemoryObject m_buff_albedo;
	oclw::MemoryObject m_buff_result_shadows;
	oclw::MemoryObject m_buff_result_gi;
	oclw::Image m_buff_final_image;

	DoubleBuffer m_buff_depths;
	oclw::Image m_buff_ss_positions;
	oclw::Image m_buff_image_top;
	oclw::Image m_buff_image_side;
	oclw::Image m_buff_noise;

	Denoiser m_gi_denoiser;
	Denoiser m_shadows_denoiser;

	std::vector<float> m_result_albedo;
	std::vector<int32_t> m_seeds;

	// Ouput images
	swrm::Swarm m_swarm;
	sf::Image m_output_albedo;

	// Dev
	bool first = true;

	glm::mat3 cur_view;
	glm::mat3 old_view;
	cl_float3 old_pos;

	uint32_t frame_count;
	bool m_current_final_buffer;
	cl_float3 camera_position;

	const uint32_t work_group_size = 20;


private:
	void initialize(uint8_t max_depth, std::vector<uint8_t>& svo)
	{
		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_wrapper.createProgramFromFile("../src/albedo.cl");
		m_program_gi = m_wrapper.createProgramFromFile("../src/lighting.cl");
		m_program_combinator = m_wrapper.createProgramFromFile("../src/combinator.cl");
		m_program_mutator = m_wrapper.createProgramFromFile("../src/mutator.cl");

		// Create memory objects that will be used as arguments to kernel
		loadImagesToDevice();
		m_buff_svo = m_wrapper.createMemoryObject(svo, oclw::ReadOnly | oclw::CopyHostPtr);
		m_buff_mutations = m_wrapper.createMemoryObject<Mutation>(10, oclw::ReadOnly);
		m_buff_view_matrix = m_wrapper.createMemoryObject<float>(9, oclw::ReadOnly);
		// Create OpenCL buffers
		initializeOutputImages();
		const uint64_t albedo_render_pxl_count = m_render_dimension.x * m_render_dimension.y;
		const uint32_t lighting_render_width = as<uint32_t>(m_render_dimension.x * m_lighting_quality);
		const uint32_t lighting_render_height = as<uint32_t>(m_render_dimension.y * m_lighting_quality);

		std::cout << "Render size: " << m_render_dimension.x << " x " << m_render_dimension.y << std::endl;
		std::cout << "Light  size: " << lighting_render_width << " x " << lighting_render_height << std::endl;
		
		m_result_albedo.resize(albedo_render_pxl_count * 4);
		//m_buff_result_albedo = m_wrapper.createMemoryObject<float>(m_result_albedo.size(), oclw::ReadWrite);
		m_buff_final_image = m_wrapper.getContext().createImage2D(m_render_dimension.x, m_render_dimension.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_albedo = m_wrapper.getContext().createImage2D(m_render_dimension.x, m_render_dimension.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_result_gi = m_wrapper.getContext().createImage2D(lighting_render_width, lighting_render_height, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_result_shadows = m_wrapper.getContext().createImage2D(lighting_render_width, lighting_render_height, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		m_buff_depths.create(m_wrapper.getContext(), lighting_render_width, lighting_render_height, nullptr, oclw::ReadWrite, oclw::RG, oclw::Float);
		m_buff_ss_positions = m_wrapper.getContext().createImage2D(lighting_render_width, lighting_render_height, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);

		// Kernels initialization
		m_albedo = m_program.createKernel("albedo");

		m_lighting = m_program_gi.createKernel("lighting");
		
		m_combinator = m_program_combinator.createKernel("combine");
		m_combinator.setArgument(0, m_buff_final_image);
		m_combinator.setArgument(1, m_buff_albedo);

		m_mutator = m_program_mutator.createKernel("mutate");
		m_mutator.setArgument(0, m_buff_svo);
	}

	void swapFinalBuffers()
	{
		m_current_final_buffer = !m_current_final_buffer;
	}

	void loadImagesToDevice()
	{
		//m_image_side.loadFromFile("../res/grass.jpg");
		//m_image_top.loadFromFile("../res/grass.jpg");

		m_image_side.loadFromFile("../res/grass_side_16x16.bmp");
		m_image_top.loadFromFile("../res/grass_top_16x16.bmp");

		sf::Image noise_image;
		noise_image.loadFromFile("../res/noise.png");

		m_buff_image_top = imageToDevice(m_image_top);
		m_buff_image_side = imageToDevice(m_image_side);
		m_buff_noise = imageToDevice(noise_image);
	}

	oclw::Image imageToDevice(const sf::Image& image)
	{
		const sf::Vector2u image_size = image.getSize();
		return m_wrapper.getContext().createImage2D(image_size.x, image_size.y, (void*)image.getPixelsPtr(), oclw::ReadOnly | oclw::CopyHostPtr, oclw::RGBA, oclw::Unsigned_INT8);
	}

	void initializeOutputImages()
	{
		m_output_albedo.create(m_render_dimension.x, m_render_dimension.y);
	}
};
