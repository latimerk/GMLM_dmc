find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${kcGMLMProject_SOURCE_DIR}/+kgmlm/+CUDAlib/kCUDAsrc)
link_directories(${kcGMLMProject_SOURCE_DIR}/+kgmlm/+CUDAlib/kCUDAobj)

pybind11_add_module(pyGMLM MODULE  kcGMLMPython.cpp kcGMLMPython_glm.cpp)

target_link_libraries(pyGMLM PRIVATE  kcGMLM  -lcuda -lcudadevrt -lcudart  -lcusparse -lcublas)
set_property(TARGET pyGMLM PROPERTY LIBRARY_OUTPUT_DIRECTORY ${kcGMLMProject_SOURCE_DIR}/Python)