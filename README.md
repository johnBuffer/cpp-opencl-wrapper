# opencl-wrapper

This wrapper aims to greatly ease the use of OpenCL providing an easy to use interface.

# Usage

In order to use this wrapper you just need to include `ocl_wrapper.hpp`

```cpp
#include <ocl_wrapper.hpp>
```

# Example
In this example we will create a very simple hello worldish opencl application which sums up two arrays and writes the result in a third one.

### First, some includes
```cpp
#include <iostream>
#include <ocl_wrapper.hpp>
```

### Declare opencl program source
```cpp
const std::string program_source = "                             \
kernel void test(global int* a, global int* b, global int* c) {  \
    const int idx = get_global_id(0);                            \
    c[idx] = a[idx] + b[idx];                                    \
}";
```
Source can also be loaded from a file using `Utils::loadSourceFromFile`

### The main function
```cpp
int main()
{
    try
    {
        // Create a wrapper using GPU
        oclw::Wrapper wrapper(oclw::DeviceType::GPU);
        
        // Initialize problem's data
        const uint32_t elements_count = 5u;
        std::vector<int> a({ 1, 5, 7, 0, 5 });
        std::vector<int> b({ 4, 3, 1, 1, 4 });
        std::vector<int> c;
        
        // Create memory objects
        oclw::MemoryObject a_buff = wrapper.createMemoryObject(a, oclw::ReadOnly | oclw::CopyHostPtr);
        oclw::MemoryObject b_buff = wrapper.createMemoryObject(b, oclw::ReadOnly | oclw::CopyHostPtr);
        oclw::MemoryObject c_buff = wrapper.createMemoryObject<int>(elements_count, oclw::WriteOnly);
        
        // Compile program
        oclw::Program program = wrapper.createProgram(program_source);
        
        // Create kernel
        oclw::Kernel kernel = program.createKernel("test");
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
```

Note that it is also possible to create a program **from a file**. In this case the code becomes
```cpp
oclw::Program program = wrapper.createProgramFromFile("source_file.cl");
```

# Exceptions
When an OpenCL api call fails, an `oclw::Exception` is raised. It contains the error string corresponding to the OpenCL error code.

For example if in the previous example one argument **wasn't set**, the std output would show this
```
Error: Cannot add kernel 'test' to command queue [CL_INVALID_KERNEL_ARGS]
```
