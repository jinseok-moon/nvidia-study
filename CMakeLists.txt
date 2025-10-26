cmake_minimum_required(VERSION 3.20)

project(nvidia-study LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
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

# Common compile options
add_library(cuda_common INTERFACE)
target_compile_features(cuda_common INTERFACE cxx_std_17)
target_compile_options(cuda_common INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(cuda_common INTERFACE CUDA::cudart)

# Common include dirs
set(PROJ_UTILS_DIR ${CMAKE_SOURCE_DIR}/src/cuda/utils)
target_include_directories(cuda_common INTERFACE ${PROJ_UTILS_DIR})

# ----- Auto-discover CUDA samples -----

add_subdirectory(src/cuda)


