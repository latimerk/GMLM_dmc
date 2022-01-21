function [ projectHome, CUDAdirectory, CUDAlibSubdirectory, MATLABdirectory, sourceDir, objDir, mexDir ] = myCUDAPaths(  )
%MYPATHS this function contains path information to the CUDA and MATLAB folders.
%   This is used for compiling CUDA files into mex files.

%% 1. Set absolute path to the base directory for this project
projectHome = fileparts(fileparts(fileparts(which('kgmlm.CUDAlib.myCUDAPaths'))));

% check if directory exists
if(~isfolder(projectHome))
    warning(['ERROR: projectHome directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],projectHome);
end

% IF this fails for some reason, specify absolute path to directory containing this function, e.g.:
% projectHome =  '/home/USERNAME/gitCode/GLM/';

objDir    = [ projectHome '/+kgmlm/+CUDAlib/kCUDAobj/'];
mexDir    = [ projectHome '/+kgmlm/+CUDAlib/'];
sourceDir = [ projectHome '/+kgmlm/+CUDAlib/kCUDAsrc/'];

%% 2. Set absolute path for directory where CUDA installation lives:
if(ispc)
    CUDAdirectory   = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/';
    CUDAlibSubdirectory = [CUDAdirectory '/lib/x64/'];
    CUDALibFile = [CUDAlibSubdirectory 'cuda.lib'];
else
    CUDAdirectory   = '/usr/local/cuda/';
    CUDAlibSubdirectory = [CUDAdirectory '/lib64/'];
    CUDALibFile = [CUDAlibSubdirectory 'libcudart.so'];
end
               
% check if directory exists
if(~isfolder(CUDAdirectory))
    warning(['ERROR: CUDAdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDAdirectory);
end
if(~exist(CUDALibFile, "file"))
    error(['ERROR: CUDALibFile does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDALibFile);
end

%% 3. Directory of the MATLAB installation. 
MATLABdirectory = matlabroot;  % this *shouldn't* need adjusting
MATLABdirectory = [MATLABdirectory, '/'];


