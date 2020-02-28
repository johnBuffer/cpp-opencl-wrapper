#include <iostream>
#include <CL/opencl.h>


cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	// First, select an OpenCL platform to run on.
	// For this example, we simply choose the first available
	// platform. Normally, you would query for all available
	// platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform. Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};

	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}


int main()
{
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;
	// Create an OpenCL context on first available platform
	context = CreateContext();
	if (context == NULL)
	{
	}

	return 0;
}
