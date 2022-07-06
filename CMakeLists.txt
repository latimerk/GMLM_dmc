cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(kcGMLMProject LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()



option(WITH_DOUBLE_PRECISION "Use double precision for CUDA only" ON)
message("Compiling with GPU code.")
add_definitions(-DUSE_GPU)


if(WITH_DOUBLE_PRECISION)
  message("Compiling double precision code.")
else()
  message("Compiling single precision code.")
  add_definitions(-DUSE_SINGLE_PRECISION)
endif()

add_subdirectory(src)