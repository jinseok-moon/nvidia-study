cmake_minimum_required(VERSION 3.20)

project(nvidia-study LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures
if(CMAKE_VERSION VERSION_LESS "3.24")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMake < 3.24 detected. Consider setting CMAKE_CUDA_ARCHITECTURES manually (e.g., -DCMAKE_CUDA_ARCHITECTURES=86)")
  endif()
else()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP QUIET)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG)
endif()

# Common compile options
add_library(cuda_common INTERFACE)
target_compile_features(cuda_common INTERFACE cxx_std_20)
# target_compile_options(cuda_common INTERFACE
#   $<$<COMPILE_LANGUAGE:CXX>:-O3>
#   $<$<COMPILE_LANGUAGE:CUDA>:-O3>
# )

# Find CUDA package
set(nvtx3_dir /usr/local/cuda/include/nvtx3)

set(CUDA_LIB cuda ${CUDA_LIBRARIES} cublas)
set(CUDA_FILE_EXTENSION cu)
target_link_libraries(cuda_common INTERFACE ${CUDA_LIB} cublas)

# Common include dirs
set(PROJ_UTILS_DIR ${CMAKE_SOURCE_DIR}/src/cuda/utils)
target_include_directories(cuda_common INTERFACE ${PROJ_UTILS_DIR})
# ----- Auto-discover CUDA samples -----

add_subdirectory(src/cuda)


