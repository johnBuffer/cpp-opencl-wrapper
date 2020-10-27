cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME opencl_wrapper)
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenCL)

set(SOURCES "src/main.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "include" ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})