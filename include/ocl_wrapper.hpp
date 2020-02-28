#pragma once

#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>


namespace oclw
{
	enum PlatformType {
		CPU = CL_DEVICE_TYPE_CPU,
		GPU = CL_DEVICE_TYPE_GPU
	};

	enum MemoryObjectReadMode {
		ReadOnly = CL_MEM_READ_ONLY,
		ReadWrite = CL_MEM_READ_WRITE
	};

	class MemoryObject
	{
	public:
		template<typename T>
		MemoryObject(cl_context context, std::vector<T>& data, int32_t mode)
		{
			const uint64_t element_count = data.size();
			m_memory_object = clCreateBuffer(context, mode, sizeof(T) * element_count, data.data(), NULL);
		}

		MemoryObject(cl_context context, uint32_t element_size, uint64_t element_count, int32_t mode)
		{
			m_memory_object = clCreateBuffer(context, mode, element_size * element_count, NULL, NULL);
		}

		operator bool() const
		{
			return m_memory_object;
		}

	private:
		cl_mem m_memory_object;
	};


	class Kernel
	{

	};


	class Program
	{
	public:
		Program(cl_program program = nullptr)
			: m_program(program)
		{

		}

		Program(cl_context context, const std::string& source, cl_device_id device)
			: m_program(nullptr)
		{
			int32_t err_num;
			cl_program program = nullptr;
			const char *src_str = source.c_str();
			program = clCreateProgramWithSource(context, 1, (const char**)&src_str, NULL, NULL);
			if (program == NULL) {
				return;
			}

			err_num = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
			if (err_num != CL_SUCCESS)
			{
				// Determine the reason for the error
				char buildLog[16384];
				clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
					sizeof(buildLog), buildLog, NULL);
				std::cerr << "Error in kernel: " << std::endl;
				std::cerr << buildLog << std::endl;
				clReleaseProgram(program);
				return;
			}

			m_program = program;
		}

		operator bool() const
		{
			return m_program;
		}

		cl_kernel createKernel(const std::string& kernel_name)
		{
			return clCreateKernel(m_program, kernel_name.c_str(), NULL);
		}

	private:
		cl_program m_program;
	};


	class Context
	{
	public:
		Context(cl_context raw_context = nullptr)
			: m_context(raw_context)
		{}

		operator bool() const
		{
			return m_context;
		}

		Context(cl_platform_id platform_id, PlatformType type)
		{
			cl_int errNum;
			cl_context context = nullptr;
			cl_context_properties contextProperties[] = {
				CL_CONTEXT_PLATFORM,
				(cl_context_properties)platform_id,
				0
			};

			context = clCreateContextFromType(contextProperties, type, NULL, NULL, &errNum);
			if (errNum == CL_SUCCESS) {
				m_context = context;
			}
		}

		const std::vector<cl_device_id>& getDevices()
		{
			m_devices.clear();
			cl_int err_num;
			std::size_t device_buffer_size = 0;
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, NULL, &device_buffer_size);
			// Check everything is OK
			if (err_num != CL_SUCCESS) {
				return m_devices;
			}
			if (device_buffer_size <= 0) {
				return m_devices;
			}
			const uint64_t devices_count = device_buffer_size / sizeof(cl_device_id);
			m_devices.resize(devices_count);
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, device_buffer_size, m_devices.data(), NULL);
			if (err_num != CL_SUCCESS) {
				return m_devices;
			}

			return m_devices;
		}

		cl_command_queue createQueue(cl_device_id device)
		{
			cl_command_queue command_queue = nullptr;			
			command_queue = clCreateCommandQueue(m_context, device, 0, NULL);
			if (!command_queue) {
				return nullptr;
			}
			return command_queue;
		}

		Program createProgram(cl_device_id device, const char* fileName) const
		{
			std::ifstream kernel_file(fileName, std::ios::in);
			if (!kernel_file.is_open())
			{
				return nullptr;
			}
			std::ostringstream oss;
			oss << kernel_file.rdbuf();

			return Program(m_context, oss.str(), device);
		}

		template<typename T>
		MemoryObject createMemoryObject(std::vector<T>& data, int32_t mode = ReadOnly)
		{
			return MemoryObject(m_context, data, mode);
		}

		template<typename T>
		MemoryObject createMemoryObject(uint64_t element_count, int32_t mode = ReadOnly)
		{
			return MemoryObject(m_context, sizeof(T), element_count, mode);
		}

	private:
		cl_context m_context;
		std::vector<cl_device_id> m_devices;
	};


	class Wrapper
	{
	public:
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

		Context createContext(cl_platform_id platform_id, PlatformType type) const
		{
			return Context(platform_id, type);
		}

	private:
		uint32_t m_platforms_count;
		std::vector<cl_platform_id> m_platforms;
	};
}