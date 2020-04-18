#pragma once

#include <SFML/Graphics.hpp>
#include "ocl_wrapper.hpp"
#include "utils.hpp"
#include "camera_controller.hpp"


constexpr uint32_t CPU_THREADS = 16u;

class Raytracer
{
public:
	Raytracer(uint32_t render_width, uint32_t render_height)
		: m_render_dimension(render_width, render_height)
		, m_wrapper(oclw::GPU)
		, m_time(0.0f)
	{
		initialize();
	}

	void updateKernels(const Camera& camera)
	{
		const glm::mat3& view_matrix = camera.getViewMatrix();
	}

	void render()
	{
		compute();

		for (uint32_t x(0); x < m_render_dimension.x; ++x) {
			for (uint32_t y(0); y < m_render_dimension.y; ++y) {
				// Computing ray coordinates in 'lens' space ie in normalized screen space
				const uint32_t index = 4 * (x + y * m_render_dimension.x);
				uint8_t r = m_result_data[index + 0];
				uint8_t g = m_result_data[index + 1];
				uint8_t b = m_result_data[index + 2];
				m_output_image.setPixel(x, y, sf::Color(r, g, b, 255));
			}
		}
	}

	const sf::Image getResult() const
	{
		return m_output_image;
	}

	void compute()
	{
		m_wrapper.runKernel(m_kernel, oclw::Size(m_render_dimension.x, m_render_dimension.y), oclw::Size(10, 10));
	}

private:
	// Conf
	sf::Vector2u m_render_dimension;
	float m_time;
	// OpenCL
	oclw::Wrapper m_wrapper;
	oclw::Program m_program;
	// Kernels
	oclw::Kernel m_kernel;
	// Buffers
	oclw::MemoryObject m_buffer;

	std::vector<float> m_result_data;
	// Ouput images
	sf::Image m_output_image;


private:
	void initialize()
	{
		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_wrapper.createProgram("../src/image_3d.cl");
		m_kernel = m_program.createKernel("test");
		// Create OpenCL buffers
		m_result_data.resize(m_render_dimension.x * m_render_dimension.y * 4);
		initializeOutputImages();
		initializeData();
	}

	void initializeOutputImages()
	{
		m_output_image.create(m_render_dimension.x, m_render_dimension.y);
	}

	void initializeData()
	{
		const uint64_t cube_size = 16u;
		std::vector<float> data(4 * cube_size * cube_size * cube_size);
		/*for (float& f : data) {
			f = (rand() % 100) * 0.01f;
		}*/
		m_buffer = m_wrapper.getContext().createImage3D(cube_size, cube_size, cube_size, data.data(), oclw::MemoryObjectReadMode::CopyHostPtr | oclw::MemoryObjectReadMode::ReadWrite, oclw::ImageFormat::RGBA, oclw::ChannelDatatype::Float);
	}
};
