#include <iostream>
#include <CL/opencl.h>
#include <ocl_wrapper.hpp>


const std::string program_source = "                                    \
__kernel void test(__global int* a, __global int* b, __global int* c) { \
	const int idx = get_global_id(0);                                   \
	c[idx] = a[idx] + b[idx];                                           \
}";


int main()
{
	try
	{
		// Create a wrapper using GPU
		oclw::Wrapper wrapper(oclw::DeviceType::GPU);
		// Initialize problem's data
		const uint32_t elements_count = 5u;
		std::vector<int> a{ 1, 5, 7, 0, 5 };
		std::vector<int> b{ 4, 3, 1, 1, 4 };
		std::vector<int> c;
		// Create memory objects
		oclw::MemoryObject a_buff = wrapper.createMemoryObject(a, oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject b_buff = wrapper.createMemoryObject(b, oclw::ReadOnly | oclw::CopyHostPtr);
		oclw::MemoryObject c_buff = wrapper.createMemoryObject<int>(elements_count, oclw::WriteOnly);
		// Compile program
		oclw::Program program = wrapper.createProgram(program_source);
		oclw::Kernel kernel = program.createKernel("test");
		// Create kernel
		kernel.setArgument(0, a_buff);
		kernel.setArgument(1, b_buff);
		kernel.setArgument(2, c_buff);
		// Execute the kernel
		wrapper.runKernel(kernel, oclw::Size(elements_count), oclw::Size(1u));
		// Read device buffer c_buff
		wrapper.safeReadMemoryObject(c_buff, c);
		// Print result
		for (int i : c) {
			std::cout << i << std::endl;
		}
	}
	catch (const oclw::Exception& error)
	{
		std::cout << "Error: " << error.what() << std::endl;
	}

	return 0;
}
