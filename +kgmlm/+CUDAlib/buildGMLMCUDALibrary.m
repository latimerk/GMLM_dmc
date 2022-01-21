function [] = buildGMLMCUDALibrary(lib_num)

if(nargin < 1)
    kgmlm.CUDAlib.compileCUDAMexLibs(); %compiles the c++/CUDA code using a direct call to nvcc (note: this code is not matlab dependent)
    kgmlm.CUDAlib.compileCUDAMexFunc(); %compiles the MEX API functions that call the GMLM CUDA code
    kgmlm.CUDAlib.linkCUDAMex([], true);        %links the MEX API to the CUDA library files
else
    kgmlm.CUDAlib.compileCUDAMexLibs(lib_num);
    kgmlm.CUDAlib.compileCUDAMexFunc(lib_num); 
    kgmlm.CUDAlib.linkCUDAMex(lib_num);
end