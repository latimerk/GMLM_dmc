cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


option(WITH_GPU "Enable CUDA code" ON)
option(WITH_DOUBLE_PRECISION "Use double precision for CUDA only" ON)

if(WITH_GPU)
  message("Compiling with GPU code.")
  add_definitions(-DUSE_GPU)
  project(kcGMLMProject LANGUAGES CXX CUDA)
else()
  message("GPU code disabled.")
  project(kcGMLMProject LANGUAGES CXX)
endif()

if(WITH_DOUBLE_PRECISION)
  message("Compiling double precision code.")
else()
  message("Compiling single precision code.")
  add_definitions(-DUSE_SINGLE_PRECISION)
endif()

add_subdirectory(src)
add_subdirectory(Python/src)