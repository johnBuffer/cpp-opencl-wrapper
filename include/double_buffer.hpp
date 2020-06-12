#pragma once
#include "ocl_wrapper.hpp"
#include <functional>


struct DoubleBuffer
{
	DoubleBuffer()
		: current_buffer(0u)
	{
	}

	void create(oclw::Context& context, uint32_t width, uint32_t height, void* data, int32_t mode, oclw::ImageFormat format, oclw::ChannelDatatype datatype)
	{
		for (uint8_t i(2); i--;) {
			buffers[i] = context.createImage2D(width, height, data, mode, format, datatype);
		}
	}

	void swap()
	{
		current_buffer = !current_buffer;
	}

	oclw::MemoryObject& getCurrent()
	{
		return buffers[current_buffer];
	}

	oclw::MemoryObject& getLast()
	{
		return buffers[!current_buffer];
	}

	uint8_t current_buffer;
	oclw::MemoryObject buffers[2];
};