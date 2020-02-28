cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME opencl_wrapper)
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)

set(SFML_DIR "" CACHE PATH "SFML lib path")
set(SFML_LIB_DIR "${SFML_DIR}/lib")
set(SFML_INC_DIR "${SFML_DIR}/include")

find_package(OpenGL)

set(SOURCES "src/main.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries(${PROJECT_NAME} ${SFML_LIBS})