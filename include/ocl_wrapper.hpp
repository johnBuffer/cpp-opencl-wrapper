#pragma once

#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>


namespace oclw
{
	const std::vector<std::string> cl_errors = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"CL_MISALIGNED_SUB_BUFFER_OFFSET",
		"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
		"CL_COMPILE_PROGRAM_FAILURE",
		"CL_LINKER_NOT_AVAILABLE",
		"CL_LINK_PROGRAM_FAILURE",
		"CL_DEVICE_PARTITION_FAILED",
		"CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
		"CL_INVALID_PROPERTY",
		"CL_INVALID_IMAGE_DESCRIPTOR",
		"CL_INVALID_COMPILER_OPTIONS",
		"CL_INVALID_LINKER_OPTIONS",
		"CL_INVALID_DEVICE_PARTITION_COUNT"
	};

	const std::string& getErrorString(cl_int err_num)
	{
		const int32_t error_code = -(err_num < -19 ? err_num + 10 : err_num);
		return cl_errors[error_code];
	}

	bool checkError(cl_int err_num)
	{
		return err_num != CL_SUCCESS;
	}

	enum PlatformType {
		CPU = CL_DEVICE_TYPE_CPU,
		GPU = CL_DEVICE_TYPE_GPU
	};

	enum MemoryObjectReadMode {
		ReadOnly = CL_MEM_READ_ONLY,
		ReadWrite = CL_MEM_READ_WRITE,
		CopyHostPtr = CL_MEM_COPY_HOST_PTR
	};


	class WrapperException : public std::exception
	{
	public:
		WrapperException(cl_int error, const std::string& message)
			: m_error(error)
			, m_message(message)
		{}

		const char* what() const override
		{
			return m_message.c_str();
		}

		const std::string getReadableError() const
		{
			return (m_message + " [" + getErrorString(m_error) + "]").c_str();
		}

		cl_int getErrorCode() const
		{
			return m_error;
		}

	private:
		cl_int m_error;
		std::string m_message;
	};


	class MemoryObject
	{
	public:
		template<typename T>
		MemoryObject(cl_context context, std::vector<T>& data, int32_t mode)
			: m_element_count(data.size())
			, m_total_size(sizeof(T) * data.size())
		{
			initialize(context, mode, m_total_size, data.data());
		}

		MemoryObject(cl_context context, uint32_t element_size, uint64_t element_count, int32_t mode)
			: m_element_count(element_count)
			, m_total_size(element_size * element_count)
		{
			initialize(context, mode, m_total_size, NULL);
		}

		void initialize(cl_context context, int32_t mode, uint64_t total_size, void* data)
		{
			cl_int err_num;
			m_memory_object = clCreateBuffer(context, mode, m_total_size, data, &err_num);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot create memory object");
			}
		}

		operator bool() const
		{
			return m_memory_object;
		}

		cl_mem& getRaw()
		{
			return m_memory_object;
		}

		std::size_t getBytesSize() const
		{
			return m_total_size;
		}

	private:
		cl_mem m_memory_object;
		const std::size_t m_element_count;
		const std::size_t m_total_size;
	};


	class Kernel
	{
	public:
		Kernel(cl_kernel raw_kernel = nullptr)
			: m_kernel(raw_kernel)
			, m_name("")
		{}

		Kernel(cl_program program, const std::string& name)
			: m_kernel(nullptr)
			, m_name(name)
		{
			cl_int err_num;
			m_kernel = clCreateKernel(program, name.c_str(), &err_num);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot create kernel '" + name + "'");
			}
		}

		void setArgument(uint32_t arg_num, MemoryObject& object)
		{
			int32_t err_num = clSetKernelArg(m_kernel, arg_num, sizeof(cl_mem), &(object.getRaw()));
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot set argument of kernel '" + m_name + "'");
			}
		}

		cl_kernel& getRaw()
		{
			return m_kernel;
		}

		const std::string& getName() const
		{
			return m_name;
		}

	private:
		cl_kernel m_kernel;
		const std::string m_name;
	};


	class Program
	{
	public:
		Program(cl_program program = nullptr)
			: m_program(program)
		{}

		Program(cl_context context, const std::string& source, cl_device_id device)
			: m_program(nullptr)
		{
			int32_t err_num;
			const char *src_str = source.c_str();
			m_program = clCreateProgramWithSource(context, 1, (const char**)&src_str, NULL, &err_num);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot create program");
			}

			err_num = clBuildProgram(m_program, 0, NULL, NULL, NULL, NULL);
			if (err_num != CL_SUCCESS) {
				// Determine the reason for the error
				char buildLog[16384];
				clGetProgramBuildInfo(m_program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
				clReleaseProgram(m_program);
				throw WrapperException(err_num, "Cannot build program: '" + std::string(buildLog) + "'");
			}
		}

		operator bool() const
		{
			return m_program;
		}

		Kernel createKernel(const std::string& kernel_name)
		{
			return Kernel(m_program, kernel_name);
		}

	private:
		cl_program m_program;
	};


	class CommandQueue
	{
	public:
		CommandQueue(cl_command_queue raw_command_queue = nullptr)
			: m_command_queue(raw_command_queue)
		{}

		CommandQueue(cl_context context, cl_device_id device)
			: m_command_queue(nullptr)
		{
			cl_int err_num;
			m_command_queue = clCreateCommandQueue(context, device, 0, &err_num);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot create command queue");
			}
		}

		operator bool() const
		{
			return m_command_queue;
		}

		void addKernel(Kernel& kernel, uint32_t work_dimension, const std::size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size)
		{
			int32_t err_num = clEnqueueNDRangeKernel(m_command_queue, kernel.getRaw(), work_dimension, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
			if (err_num != CL_SUCCESS) {
				throw WrapperException(err_num, "Cannot add kernel '" + kernel.getName() + "' to command queue");
			}
		}

		template<typename T>
		void readMemoryObject(MemoryObject& object, bool blocking_read, std::vector<T>& result)
		{
			int32_t err_num = clEnqueueReadBuffer(m_command_queue, object.getRaw(), blocking_read ? CL_TRUE : CL_FALSE, 0, object.getBytesSize(), result.data(), 0, NULL, NULL);
			if (err_num != CL_SUCCESS) {
				throw WrapperException(err_num, "Cannot read from buffer");
			}
		}

	private:
		cl_command_queue m_command_queue;
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
			cl_int err_num;
			cl_context_properties contextProperties[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
			};

			m_context = clCreateContextFromType(contextProperties, type, NULL, NULL, &err_num);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot create context");
			}
		}

		const std::vector<cl_device_id>& getDevices()
		{
			m_devices.clear();
			cl_int err_num;
			std::size_t device_buffer_size = 0;
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, NULL, &device_buffer_size);
			// Check everything is OK
			if (checkError(err_num) || device_buffer_size <= 0) {
				throw WrapperException(err_num, "Cannot get devices");
			}
			
			const uint64_t devices_count = device_buffer_size / sizeof(cl_device_id);
			m_devices.resize(devices_count);
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, device_buffer_size, m_devices.data(), NULL);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot get devices");
			}

			return m_devices;
		}

		CommandQueue createQueue(cl_device_id device)
		{
			return CommandQueue(m_context, device);
		}

		Program createProgram(cl_device_id device, const std::string& source_filename) const
		{
			std::ifstream kernel_file(source_filename, std::ios::in);
			if (!kernel_file.is_open()) {
				throw WrapperException(-1, "Cannot open source file '" + source_filename + "'");
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
		Wrapper(const uint32_t num = 1u)
		{
			fetchPlatforms(num);
		}

		void fetchPlatforms(const uint32_t num)
		{
			m_platforms.resize(num);
			cl_int err_num = clGetPlatformIDs(num, m_platforms.data(), &m_platforms_count);
			if (checkError(err_num)) {
				throw WrapperException(err_num, "Cannot fetch platforms");
			}
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
