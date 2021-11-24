cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda_kcGMLM LANGUAGES CXX CUDA)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

include(CTest)
 
add_library(kcGMLM STATIC
  kcGMLM.cu
  kcGMLM.hpp
  kcGMLM_computeBlock.cu
  kcGMLM_computeBlock.hpp
  kcGMLM_dataStructures.cu
  kcGMLM_dataStructures.hpp
  kcBase.cu
  kcBase.hpp
  )

add_library(kcGMLMPop STATIC
  kcGMLMPop.cu
  kcGMLMPop.hpp
  kcGMLMPop_computeBlock.cu
  kcGMLMPop_computeBlock.hpp
  kcGMLMPop_dataStructures.cu
  kcGMLMPop_dataStructures.hpp
  kcBase.cu
  kcBase.hpp
  )

add_library(kcGLM STATIC
  kcGLM.cu
  kcGLM.hpp
  kcGLM_computeBlock.cu
  kcGLM_computeBlock.hpp
  kcGLM_dataStructures.cu
  kcGLM_dataStructures.hpp
  kcBase.cu
  kcBase.hpp
  )
# Request that kcGMLM be built with -std=c++11
# As this is a public compile feature anything that links to 
# kcGMLM will also build with -std=c++11
target_compile_features(kcGMLM PUBLIC cxx_std_11)
target_compile_features(kcGMLMPop PUBLIC cxx_std_11)
target_compile_features(kcGLM PUBLIC cxx_std_11)

 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( kcGMLM
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( kcGMLMPop
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( kcGLM
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
 