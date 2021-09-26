kgmlm.CUDAlib.compileCUDAMexLibs(); %compiles the c++/CUDA code using a direct call to nvcc (note: this code is not matlab dependent)
kgmlm.CUDAlib.compileCUDAMexFunc(); %compiles the MEX API functions that call the GMLM CUDA code
kgmlm.CUDAlib.linkCUDAMex(true);        %links the MEX API to the CUDA library files