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
    CUDAdirectory   = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/';
    CUDAlibSubdirectory = [CUDAdirectory '/lib/x64/'];
else
    CUDAdirectory   = '/usr/local/cuda/';
    CUDAlibSubdirectory = [CUDAdirectory '/lib64/'];
end
               
% check if directory exists
if(~isfolder(CUDAdirectory))
    warning(['ERROR: CUDAdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDAdirectory);
end

%% 3. Directory of the MATLAB installation. 
MATLABdirectory = matlabroot;  % this *shouldn't* need adjusting
MATLABdirectory = [MATLABdirectory, '/'];


