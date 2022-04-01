cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(kcGMLMProject LANGUAGES CXX CUDA)

add_subdirectory(+kgmlm/+CUDAlib/kCUDAsrc)

add_subdirectory(Python/src)