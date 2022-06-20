
if(WITH_GPU)
  enable_language(CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  include(CTest)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  
  # add_library(kcGMLM STATIC
  #   kcGMLM.cu
  #   kcGMLM_computeBlock.cu
  #   kcGMLMPop_computeBlock.cu
  #   kcGMLM_dataStructures.cu
  #   )

  add_library(kcGMLM STATIC
    kcBase.cu
    kcGLM.cu
    kcGLM_computeBlock.cu
    kcGLM_dataStructures.cu
    kcGMLM.cu
    kcGMLM_computeBlock.cu
    kcGMLMPop_computeBlock.cu
    kcGMLM_dataStructures.cu
    )


  # Request that kcGMLM be built with -std=c++11
  # As this is a public compile feature anything that links to 
  # kcGMLM will also build with -std=c++11
  target_compile_features(kcGMLM PUBLIC cxx_std_11)
  
  # We need to explicitly state that we need all CUDA files in the 
  # particle library to be built with -dc as the member functions 
  # could be called by other libraries and executables
  set_target_properties( kcGMLM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  
  # install(TARGETS kcGMLM kcGMLMPop kcGLM DESTINATION ${kcGMLMProject_SOURCE_DIR}/obj/)
  set_property(TARGET kcGMLM PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${kcGMLMProject_SOURCE_DIR}/obj/)
endif()