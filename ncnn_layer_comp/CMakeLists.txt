cmake_minimum_required(VERSION 3.10)
project(ncnn_deconv1d_comparison)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ncnn package
find_package(ncnn REQUIRED)

# Set up include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/layer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/layer/arm)

# Add compiler flags for ARM NEON if available
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()

# Add OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_deconv1d_comparison PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create the test executable
add_executable(test_deconv1d_comparison test_deconv1d_comparison.cpp)

# Link with ncnn
target_link_libraries(test_deconv1d_comparison ncnn)

# Add compile definitions
target_compile_definitions(test_deconv1d_comparison PRIVATE __ARM_NEON)

# Set output directory
set_target_properties(test_deconv1d_comparison PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
