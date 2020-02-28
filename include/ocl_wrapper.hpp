#pragma once

#include <CL/opencl.h>
#include <vector>


namespace oclw
{
	class Wrapper
	{
	public:
		enum PlatformType {
			CPU = CL_DEVICE_TYPE_CPU,
			GPU = CL_DEVICE_TYPE_GPU
		};

		Wrapper() = default;

		void fetchPlatforms(const uint32_t num = 1u)
		{
			m_platforms.resize(num);
			cl_int err_num = clGetPlatformIDs(1, m_platforms.data(), &m_platforms_count);
		}

		uint32_t getPlatformCount() const
		{
			return m_platforms_count;
		}

		const std::vector<cl_platform_id>& getPlatforms() const
		{
			return m_platforms;
		}

		cl_context createContext(cl_platform_id platform_id, PlatformType type) const
		{
			cl_context context = nullptr;
			cl_int errNum;
			cl_context_properties contextProperties[] = {
				CL_CONTEXT_PLATFORM,
				(cl_context_properties) platform_id,
				0
			};
			
			context = clCreateContextFromType(contextProperties, type, NULL, NULL, &errNum);
			if (errNum != CL_SUCCESS)
			{
				return nullptr;
			}

			return context;
		}

	private:
		uint32_t m_platforms_count;
		std::vector<cl_platform_id> m_platforms;
	};
}