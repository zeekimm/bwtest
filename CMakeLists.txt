cmake_minimum_required(VERSION 3.10)

project(bw)

set(CMAKE_CONFIGURATION_TYPES Release RelWithDebInfo CACHE STRING INTERNAL FORCE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=300)

add_executable(bw Bw.cc OpenclRuntime.cc)
target_include_directories(bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bw PRIVATE dl)

