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
		, m_wrapper()
		, m_time(0.0f)
	{
		m_context = createDefaultContext(m_wrapper);
		initialize();
	}

	void render()
	{
		//compute();

		for (uint32_t x(0); x < 10; ++x) {
			for (uint32_t y(0); y < 10; ++y) {
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
		const size_t globalWorkSize[2] = { 10, 10 };
		const size_t localWorkSize[2] = { 1, 1 };
		m_command_queue.addKernel(m_kernel, 2, NULL, globalWorkSize, localWorkSize);
	}

private:
	// Conf
	sf::Vector2u m_render_dimension;
	float m_time;
	// OpenCL
	oclw::Wrapper m_wrapper;
	oclw::Context m_context;
	oclw::CommandQueue m_command_queue;
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
		// Get devices
		auto& devices_list = m_context.getDevices();
		cl_device_id device = devices_list.front();
		// Create command queue
		m_command_queue = m_context.createQueue(device);

		// Create OpenCL program from HelloWorld.cl kernel source
		m_program = m_context.createProgram(device, "../src/image_3d.cl");
		m_kernel = m_program.createKernel("test");
		
		// Create memory objects that will be used as arguments to kernel
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
		const uint64_t cube_size = 512u;
		std::vector<float> data(4 * cube_size * cube_size * cube_size);
		for (float& f : data) {
			f = (rand() % 100) * 0.01f;
		}
		m_buffer = m_context.createImage3D(512, 512, 512, data.data(), oclw::MemoryObjectReadMode::CopyHostPtr | oclw::MemoryObjectReadMode::ReadWrite, oclw::ImageFormat::RGBA, oclw::ChannelDatatype::Float);
	}
};
