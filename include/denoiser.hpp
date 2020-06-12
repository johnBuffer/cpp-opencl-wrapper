#pragma once
#include <glm/glm.hpp>
#include "ocl_wrapper.hpp"
#include "double_buffer.hpp"



struct Denoiser
{
	const uint8_t local_size = 20u;

	Denoiser(oclw::Wrapper& wrapper_, const glm::uvec2 render_size_)
		: wrapper(wrapper_)
		, render_size(render_size_)
	{
		initialize();
	}

	void execute(const glm::mat3& last_view_matrix, cl_float3 last_position, oclw::MemoryObject& raw_lighting, oclw::MemoryObject& ss_positions, DoubleBuffer& depths)
	{
		swapBuffers();
		execute_temporal(last_view_matrix, last_position, raw_lighting, ss_positions, depths);
		execute_normalize();
		execute_blur(ss_positions);
	}

	void swapBuffers()
	{
		buff_temporal.swap();
		//buff_denoised.swap();
	}

	oclw::MemoryObject& getResult()
	{
		return buff_denoised.getCurrent();
	}

	oclw::Wrapper& wrapper;
	const glm::uvec2 render_size;

	oclw::Program temporal_program;
	oclw::Program blur_program;

	oclw::Kernel temporal;
	oclw::Kernel normalier;
	oclw::Kernel blur;

	// Buffers
	DoubleBuffer buff_temporal;
	DoubleBuffer buff_denoised;
	oclw::MemoryObject buff_view_mat;

private:
	void initialize()
	{
		temporal_program = wrapper.createProgramFromFile("../src/temporal.cl");
		temporal = temporal_program.createKernel("temporal");
		normalier = temporal_program.createKernel("normalizer");

		blur_program = wrapper.createProgramFromFile("../src/bilateral_blur.cl");
		blur = blur_program.createKernel("blur");

		buff_view_mat = wrapper.createMemoryObject<float>(9, oclw::ReadOnly);

		buff_temporal.create(wrapper.getContext(), render_size.x, render_size.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
		buff_denoised.create(wrapper.getContext(), render_size.x, render_size.y, nullptr, oclw::ReadWrite, oclw::RGBA, oclw::Float);
	}

	void execute_temporal(const glm::mat3& last_view_matrix, cl_float3 last_position, oclw::MemoryObject& raw_lighting, oclw::MemoryObject& ss_positions, DoubleBuffer& depths)
	{
		wrapper.writeInMemoryObject(buff_view_mat, &last_view_matrix[0], true);

		uint32_t args_c = 0u;
		temporal.setArgument(args_c++, buff_temporal.getCurrent());
		temporal.setArgument(args_c++, buff_temporal.getLast());
		temporal.setArgument(args_c++, raw_lighting);
		temporal.setArgument(args_c++, buff_view_mat);
		temporal.setArgument(args_c++, last_position);
		temporal.setArgument(args_c++, depths.getCurrent());
		temporal.setArgument(args_c++, depths.getLast());
		temporal.setArgument(args_c++, ss_positions);

		wrapper.runKernel(temporal, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}

	void execute_normalize()
	{
		normalier.setArgument(0, buff_temporal.getCurrent());
		normalier.setArgument(1, buff_denoised.getCurrent());
		wrapper.runKernel(normalier, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
		buff_denoised.swap();
	}

	void execute_blur(oclw::MemoryObject& ss_positions)
	{
		blur.setArgument(0, buff_denoised.getCurrent());
		blur.setArgument(1, buff_denoised.getLast());
		blur.setArgument(2, ss_positions);
		wrapper.runKernel(blur, oclw::Size(render_size.x, render_size.y), oclw::Size(local_size, local_size));
	}
};
