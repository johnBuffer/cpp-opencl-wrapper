#pragma once
#include <glm/glm.hpp>
#include "ocl_wrapper.hpp"
#include "double_buffer.hpp"



struct Denoiser
{
	const uint8_t local_size = 20u;

	Denoiser(oclw::Wrapper& wrapper_, const glm::uvec2 render_size_, const uint8_t blur_passes_, bool diso = false)
		: wrapper(wrapper_)
		, render_size(render_size_)
		, blur_passes(blur_passes_)
		, diso_handling(diso)
	{
		initialize();
	}

	void execute(const glm::mat3& last_view_matrix, cl_float3 last_position, oclw::MemoryObject& raw_lighting, oclw::MemoryObject& ss_positions, DoubleBuffer& depths)
	{
		swapBuffers();
		execute_temporal(last_view_matrix, last_position, raw_lighting, ss_positions, depths);
		execute_normalize();
		if (diso_handling) {
			//execute_median();
			//execute_blur_diso(ss_positions);
		}
		execute_blur(ss_positions);
	}

	void swapBuffers()
	{
		buff_temporal.swap();
	}

	oclw::MemoryObject& getResult()
	{
		return buff_denoised.getCurrent();
	}

	oclw::Wrapper& wrapper;
	const glm::uvec2 render_size;
	const uint8_t blur_passes;
	const bool diso_handling;

	oclw::Program temporal_program;
	oclw::Program median_program;
	oclw::Program blur_program;
	oclw::Program blur_diso_program;

	oclw::Kernel temporal;
	oclw::Kernel normalier;
	oclw::Kernel median;
	oclw::Kernel gradient;
	oclw::Kernel blur;
	oclw::Kernel blur_diso;

	// Buffers
	DoubleBuffer buff_temporal;
	DoubleBuffer buff_denoised;
	oclw::Image buff_depth_gradient;
	oclw::MemoryObject buff_view_mat;

private:
	void initialize()
	{
		// Temporal kernels
		temporal_program = wrapper.createProgramFromFile("../src/temporal.cl");
		temporal = temporal_program.createKernel("temporal");
		normalier = temporal_program.createKernel("normalizer");
		// Spacial kernels
		blur_program = wrapper.createProgramFromFile("../src/bilateral_blur.cl");
		blur_diso_program = wrapper.createProgramFromFile("../src/bilateral_blur_diso.cl");
		blur = blur_program.createKernel("blur");
		blur_diso = blur_diso_program.createKernel("blur");
		median_program = wrapper.createProgramFromFile("../src/median.cl");
		median = median_program.createKernel("median");
		// Initialise buffers
		buff_view_mat = wrapper.createMemoryObject<float>(9, oclw::ReadOnly);
		buff_temporal.create(wrapper.getContext(), render_size.x, render_size.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		buff_denoised.create(wrapper.getContext(), render_size.x, render_size.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		buff_depth_gradient = wrapper.getContext().createImage2D(render_size.x, render_size.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
	}

	// Kernels executions
	void execute_temporal(const glm::mat3& last_view_matrix, cl_float3 last_position, oclw::MemoryObject& raw_lighting, oclw::MemoryObject& ss_positions, DoubleBuffer& depths)
	{
		// Update buffer
		wrapper.writeInMemoryObject(buff_view_mat, &last_view_matrix[0], true);
		// Set args
		uint32_t args_c = 0u;
		temporal.setArgument(args_c++, buff_temporal.getCurrent());
		temporal.setArgument(args_c++, buff_temporal.getLast());
		temporal.setArgument(args_c++, raw_lighting);
		temporal.setArgument(args_c++, buff_view_mat);
		temporal.setArgument(args_c++, last_position);
		temporal.setArgument(args_c++, depths.getLast());
		temporal.setArgument(args_c++, ss_positions);
		wrapper.runKernel(temporal, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}

	void execute_normalize()
	{
		normalier.setArgument(0, buff_temporal.getCurrent());
		normalier.setArgument(1, buff_denoised.getCurrent());
		wrapper.runKernel(normalier, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}

	void execute_median()
	{
		buff_denoised.swap();
		median.setArgument(0, buff_denoised.getCurrent());
		median.setArgument(1, buff_denoised.getLast());
		wrapper.runKernel(median, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}

	void execute_blur_diso(oclw::MemoryObject& ss_positions)
	{
		blur_diso.setArgument(2, ss_positions);
		for (uint8_t i(3); i--;) {
			buff_denoised.swap();
			blur_diso.setArgument(0, buff_denoised.getCurrent());
			blur_diso.setArgument(1, buff_denoised.getLast());
			wrapper.runKernel(blur_diso, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
		}
	}

	void execute_gradient(oclw::MemoryObject& ss_positions)
	{
		gradient.setArgument(0, buff_depth_gradient);
		gradient.setArgument(1, ss_positions);
		wrapper.runKernel(gradient, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}

	void execute_blur(oclw::MemoryObject& ss_positions)
	{
		blur.setArgument(2, ss_positions);
		for (uint8_t i(0); i < blur_passes; ++i) {
			buff_denoised.swap();
			blur.setArgument(0, buff_denoised.getCurrent());
			blur.setArgument(1, buff_denoised.getLast());
			blur.setArgument(3, i);
			wrapper.runKernel(blur, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
		}

		const uint8_t pass_id = 0;
		for (uint8_t i(0); i < 1; ++i) {
			buff_denoised.swap();
			blur.setArgument(0, buff_denoised.getCurrent());
			blur.setArgument(1, buff_denoised.getLast());
			blur.setArgument(3, pass_id);
			wrapper.runKernel(blur, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
		}
	}
};
