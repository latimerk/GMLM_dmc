function [] = linkCUDAMex(lib_num, doReset)
if(nargin < 1 || isempty(lib_num))
    fNames = kgmlm.CUDAlib.compileCUDAMexFunc();
else
    [~,fNames_grp] = kgmlm.CUDAlib.compileCUDAMexFunc();
    fNames = fNames_grp{lib_num};
end

clear mex;
[~, CUDAdirectory, CUDAlibSubdirectory, MATLABdirectory, ~, objDir, mexDir, mexSourceDir] = kgmlm.CUDAlib.myCUDAPaths();


for ii = 1:length(fNames)
    fName = fNames(ii);
    
    if(fName.endsWith(".cpp"))
        fName = fName.split(".cpp");
        fName = fName(1);
    end
    
    if(ispc)
        fileType_obj = '.obj';
        fileType_lib = '.lib';
    else
        fileType_obj = '.o';
        fileType_lib = '.a';
    end

    obFileName = sprintf('%s/%s%s', objDir, fName, fileType_obj);
    if(~exist(obFileName, 'file'))
        error('No object file found for linking! %s\n', obFileName);
    end
    if(~isfolder(mexDir))
        mkdir(mexDir);
        addpath(mexDir);
    end

    % rough way to select which library to use
    if(contains(fName, 'GMLM'))
        libName = 'kcGMLM_lib';
    else
        libName = 'kcGLM_lib';
    end
    
    linkCUDAlibMex    = @(fName) mex('-cxx','-outdir', mexDir, sprintf('-L%s', CUDAlibSubdirectory), '-lcuda', '-lcudadevrt', '-lcudart',  '-lcusparse', '-lcublas', sprintf('-L%s', objDir), obFileName, sprintf('%s/%s%s', objDir, libName, fileType_lib));
    
    
    linkCUDAlibMex(fName);
end

if(nargin > 1 && doReset)
    if(ispc)
        mex('-outdir',mexDir,['-I' CUDAdirectory 'include/'], ... 
            ['-L' CUDAlibSubdirectory], '-lcudart', [mexSourceDir 'kcResetDevices.cpp']);
    else
        mex('-outdir',mexDir,['-I' CUDAdirectory 'include/'], ['-L' MATLABdirectory 'sys/os/glnxa64/'],'-lstdc++',...
            ['-L' CUDAlibSubdirectory], '-lcudart', [mexSourceDir 'kcResetDevices.cpp']);
    end
end


