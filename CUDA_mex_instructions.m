% compilation instructions for the CUDA code
% requires CUDA version 11.0

% on Ubuntu or Windows (Linux commands may work similarly on MAC, but never attempted)
% first, check paths in +kgmlm/+CUDAlib/myCUDAPaths.m
% then run
%     kgmlm.CUDAlib.buildGMLMCUDALibrary;
%
%       If a link error occurs, try 'clear mex' and then link again.