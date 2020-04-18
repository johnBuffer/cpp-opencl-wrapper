#pragma once

#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <CL/cl.hpp>


namespace oclw
{
	struct Size
	{
		Size(std::size_t x)
			: dimension(1u)
			, sizes{x, 0, 0}
		{}
		
		Size(std::size_t x, std::size_t y)
			: dimension(2u)
			, sizes{ x, y, 0 }
		{}

		Size(std::size_t x, std::size_t y, std::size_t z)
			: dimension(3u)
			, sizes{ x, y, z }
		{}

		const uint8_t dimension;
		const std::size_t sizes[3];
	};


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


	const std::string& getErrorString(cl_int err_num);


	const std::string loadSourceFromFile(const std::string& filename);


	void checkError(cl_int err_num, const std::string& err_message);


	enum PlatformType {
		CPU = CL_DEVICE_TYPE_CPU,
		GPU = CL_DEVICE_TYPE_GPU
	};


	enum MemoryObjectReadMode {
		ReadOnly = CL_MEM_READ_ONLY,
		ReadWrite = CL_MEM_READ_WRITE,
		WriteOnly = CL_MEM_WRITE_ONLY,
		CopyHostPtr = CL_MEM_COPY_HOST_PTR
	};


	enum ImageFormat
	{
		Red = CL_R,
		Alpha = CL_A,
		Intensity = CL_INTENSITY,
		Luminance = CL_LUMINANCE,
		RG = CL_RG,
		RA = CL_RA,
		RGB = CL_RGB,
		RGBA = CL_RGBA,
		ARGB = CL_ARGB,
		BGRA = CL_BGRA
	};


	enum ChannelDatatype
	{
		Normalized_INT8  = CL_SNORM_INT8,  // Each channel component is a normalized signed 8 - bit integer value.
		Normalized_INT16 = CL_SNORM_INT16, // Each channel component is a normalized signed 16 - bit integer value.
		Normalized_UINT8 = CL_UNORM_INT8, // Each channel component is a normalized unsigned 8 - bit integer value.
		Normalized_UINT16 = CL_UNORM_INT16, // Each channel component is a normalized unsigned 16 - bit integer value.
		NormalizedShort565 = CL_UNORM_SHORT_565, // Represents a normalized 5 - 6 - 5 3 - channel RGB image.The channel order must be CL_RGB.
		/*CL_UNORM_SHORT_555	Represents a normalized x - 5 - 5 - 5 4 - channel xRGB image.The channel order must be CL_RGB.
		CL_UNORM_INT_101010	Represents a normalized x - 10 - 10 - 10 4 - channel xRGB image.The channel order must be CL_RGB.
		CL_SIGNED_INT8	Each channel component is an unnormalized signed 8 - bit integer value.
		CL_SIGNED_INT16	Each channel component is an unnormalized signed 16 - bit integer value.
		CL_SIGNED_INT32	Each channel component is an unnormalized signed 32 - bit integer value.*/
		Unsigned_INT8 = CL_UNSIGNED_INT8, // Each channel component is an unnormalized unsigned 8 - bit integer value.
		/*CL_UNSIGNED_INT16	Each channel component is an unnormalized unsigned 16 - bit integer value.
		CL_UNSIGNED_INT32	Each channel component is an unnormalized unsigned 32 - bit integer value.
		CL_HALF_FLOAT	Each channel component is a 16 - bit half - float value.*/
		Float = CL_FLOAT
	};


	class Exception : public std::exception
	{
	public:
		Exception(cl_int error, const std::string& message)
			: m_error(error)
			, m_message(message)
		{}

		const char* what() const override
		{
			return m_message.c_str();
		}

		cl_int getErrorCode() const
		{
			return m_error;
		}

	private:
		const cl_int m_error;
		const std::string m_message;
	};


	class MemoryObject
	{
	public:
		MemoryObject(cl_mem buffer = nullptr, uint64_t element_count = 0u, uint64_t element_size = 0u)
			: m_memory_object(buffer)
			, m_element_count(element_count)
			, m_total_size(element_count * element_size)
		{
		}

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

		MemoryObject& operator=(const MemoryObject& other)
		{
			m_memory_object = other.m_memory_object;
			m_element_count = other.m_element_count;
			m_total_size = other.m_total_size;
			cl_int err_num = clRetainMemObject(m_memory_object);
			checkError(err_num, "Cannot retain memory object");
			return *this;
		}

		~MemoryObject()
		{
			checkError(clReleaseMemObject(m_memory_object), "Cannot delete memory object");
		}

		operator bool() const
		{
			return m_memory_object != nullptr;
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
		std::size_t m_element_count;
		std::size_t m_total_size;

		void initialize(cl_context context, int32_t mode, uint64_t total_size, void* data)
		{
			cl_int err_num;
			m_memory_object = clCreateBuffer(context, mode, m_total_size, data, &err_num);
			checkError(err_num, "Cannot create memory object");
		}
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
			checkError(err_num, "Cannot create kernel '" + name + "'");
		}

		void setArgument(uint32_t arg_num, MemoryObject& object)
		{
			std::stringstream ssx;
			ssx << "Cannot set argument [" << arg_num << "] of kernel '" << m_name << "'";
			checkError(clSetKernelArg(m_kernel, arg_num, sizeof(cl_mem), &(object.getRaw())), ssx.str());
		}

		template<typename T>
		void setArgument(uint32_t arg_num, const T& arg_value)
		{
			std::stringstream ssx;
			ssx << "Cannot set argument [" << arg_num << "] of kernel '" << m_name << "'";
			checkError(clSetKernelArg(m_kernel, arg_num, sizeof(T), &arg_value), ssx.str());
		}

		Kernel& operator=(const Kernel& other)
		{
			m_name = other.m_name;
			m_kernel = other.m_kernel;
			checkError(clRetainKernel(m_kernel), "Cannot retain kernel");
			return *this;
		}

		~Kernel()
		{
			cl_int err_num;
			err_num = clReleaseKernel(m_kernel);
			checkError(err_num, "Cannot release kernel");
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
		std::string m_name;
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
			checkError(err_num, "Cannot create program");

			err_num = clBuildProgram(m_program, 0, NULL, NULL, NULL, NULL);
			if (err_num != CL_SUCCESS) {
				// Determine the reason for the error
				char buildLog[32000];
				clGetProgramBuildInfo(m_program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
				clReleaseProgram(m_program);
				checkError(err_num, "Cannot build program: '" + std::string(buildLog) + "'");
			}
		}

		Program& operator=(const Program& other)
		{
			m_program = other.m_program;
			checkError(clRetainProgram(m_program), "Cannot retain program");
		}

		~Program()
		{
			checkError(clReleaseProgram(m_program), "Cannot release program");
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
			checkError(err_num, "Cannot create command queue");
		}

		CommandQueue& operator=(const CommandQueue& other)
		{
			m_command_queue = other.m_command_queue;
			checkError(clRetainCommandQueue(m_command_queue), "Cannot retain command queue");
			return *this;
		}

		~CommandQueue()
		{
			checkError(clReleaseCommandQueue(m_command_queue), "Cannot create command queue");
		}

		operator bool() const
		{
			return m_command_queue;
		}

		void addKernel(Kernel& kernel, uint32_t work_dimension, const std::size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size)
		{
			const int32_t err_num = clEnqueueNDRangeKernel(m_command_queue, kernel.getRaw(), work_dimension, global_work_offset, global_work_size, local_work_size, 0, 0, 0);
			checkError(err_num, "Cannot add kernel '" + kernel.getName() + "' to command queue");
		}

		template<typename T>
		void readMemoryObject(MemoryObject& object, bool blocking_read, std::vector<T>& result)
		{
			int32_t err_num = clEnqueueReadBuffer(m_command_queue, object.getRaw(), blocking_read ? CL_TRUE : CL_FALSE, 0, object.getBytesSize(), result.data(), 0, NULL, NULL);
			checkError(err_num, "Cannot read from buffer");
		}

		template<typename T>
		void writeInMemoryObject(MemoryObject& object, bool blocking_write, const T* data)
		{
			int32_t err_num = clEnqueueWriteBuffer(m_command_queue, object.getRaw(), blocking_write ? CL_TRUE : CL_FALSE, 0, object.getBytesSize(), data, 0, NULL, NULL);
			checkError(err_num, "Cannot write in buffer");
		}

		void waitCompletion()
		{
			clFinish(m_command_queue);
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

		Context(cl_platform_id platform_id, PlatformType type)
		{
			cl_int err_num;
			cl_context_properties contextProperties[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
			};

			m_context = clCreateContextFromType(contextProperties, type, NULL, NULL, &err_num);

			checkError(err_num, "Cannot create context");
		}

		~Context()
		{
			checkError(clReleaseContext(m_context), "Cannot release context");
		}

		Context& operator=(const Context& other)
		{
			m_context = other.m_context;
			checkError(clRetainContext(m_context), "Cannot retain context");
			return *this;
		}

		operator bool() const
		{
			return m_context;
		}

		const std::vector<cl_device_id> getDevices()
		{
			cl_int err_num;
			std::size_t device_buffer_size = 0;
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, NULL, &device_buffer_size);
			checkError(err_num, "Cannot get devices");
			
			const uint64_t devices_count = device_buffer_size / sizeof(cl_device_id);
			std::vector<cl_device_id> devices(devices_count);
			err_num = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, device_buffer_size, devices.data(), NULL);
			checkError(err_num, "Cannot get devices");

			for (cl_device_id id : devices) {
				uint64_t value_size;
				clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &value_size);
				char* value = (char*)malloc(value_size);
				clGetDeviceInfo(id, CL_DEVICE_NAME, value_size, value, NULL);
				printf("Device: %s\n", value);
				free(value);
			}

			return devices;
		}

		CommandQueue createQueue(cl_device_id device)
		{
			return CommandQueue(m_context, device);
		}

		Program createProgram(cl_device_id device, const std::string& source_filename) const
		{
			std::ifstream kernel_file(source_filename, std::ios::in);
			if (!kernel_file.is_open()) {
				throw Exception(-1, "Cannot open source file '" + source_filename + "'");
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

		MemoryObject createImage2D(uint32_t width, uint32_t height, void* data, int32_t mode, ImageFormat format, ChannelDatatype datatype)
		{
			cl_image_format image_format;
			image_format.image_channel_order = format;
			image_format.image_channel_data_type = datatype;
			cl_int err_num;
			cl_mem image = clCreateImage2D(m_context, mode, &image_format, width, height, 0, data, &err_num);
			checkError(err_num, "Cannot create image");
			return MemoryObject(image, width * height, 4u);
		}

		MemoryObject createImage3D(uint32_t width, uint32_t height, uint32_t depth, void* data, int32_t mode, ImageFormat format, ChannelDatatype datatype)
		{
			cl_image_format image_format;
			image_format.image_channel_order = format;
			image_format.image_channel_data_type = datatype;
			cl_int err_num;
			cl_mem image = clCreateImage2D(m_context, mode, &image_format, width, height, 0, data, &err_num);
			checkError(err_num, "Cannot create 2D image");
			return MemoryObject(image, width * height, 4u);
		}

		MemoryObject createImage3D(uint32_t width, uint32_t height, uint32_t depth, ImageFormat format, ChannelDatatype datatype)
		{
			cl_image_format image_format;
			image_format.image_channel_order = format;
			image_format.image_channel_data_type = datatype;
			cl_int err_num;
			cl_mem image = clCreateImage3D(m_context, MemoryObjectReadMode::ReadWrite, &image_format, width, height, depth, 0, 0, nullptr, &err_num);
			checkError(err_num, "Cannot create 3D image");
			return MemoryObject(image, width * height, 4u);
		}

	private:
		cl_context m_context;
	};


	class Wrapper
	{
	public:
		Wrapper()
		{
		}

		Wrapper(PlatformType type)
		{
			initializeContext(type);
		}

		std::vector<cl_platform_id> getPlatforms(const uint32_t num, cl_uint* platforms_count = nullptr)
		{
			std::vector<cl_platform_id> platforms(num);
			cl_int err_num = clGetPlatformIDs(num, platforms.data(), platforms_count);
			checkError(err_num, "Cannot fetch platforms");
			return platforms;
		}

		Context createContext(cl_platform_id platform_id, PlatformType type) const
		{
			return Context(platform_id, type);
		}

		Program createProgram(const std::string& filename)
		{
			return m_context.createProgram(m_device, filename);
		}

		void runKernel(Kernel& kernel, const Size& global_size, const Size& local_size, const std::size_t* global_work_offset = nullptr)
		{
			m_command_queue.addKernel(kernel, global_size.dimension, global_size.sizes, local_size.sizes, global_work_offset);
		}

		Context getContext()
		{
			return m_context;
		}

	private:
		Context m_context;
		cl_device_id m_device;
		CommandQueue m_command_queue;

		void initializeContext(PlatformType type)
		{
			auto platforms = getPlatforms(1u);
			if (!platforms.empty()) {
				m_context = createContext(platforms.front(), type);
				if (!m_context) {
					std::cout << "Cannot create context." << std::endl;
				}
				initializeCommandQueue();
			}
			else {
				std::cout << "Cannot find platform." << std::endl;
			}
		}

		void initializeCommandQueue()
		{
			if (m_context) {
				auto& devices_list = m_context.getDevices();
				m_device = devices_list.front();
				m_command_queue = m_context.createQueue(m_device);
			}
		}
	};
}
