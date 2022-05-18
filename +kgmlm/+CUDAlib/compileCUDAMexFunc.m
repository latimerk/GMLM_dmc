function [fNames, fNames_grp] = compileCUDAMexFunc(lib_num)

fNames_grp = {["kcGMLM_mex_create.cpp", "kcGMLM_mex_clear.cpp",  "kcGMLM_mex_computeLL.cpp", "kcGMLM_mex_computeLL_async.cpp", "kcGMLM_mex_computeLL_gather.cpp"], ...
              ["kcGLM_mex_create.cpp", "kcGLM_mex_clear.cpp",  "kcGLM_mex_computeLL.cpp"]};

fNames = [fNames_grp{:}];

if(nargout > 0)
    return;
end

if(nargin > 0 && (~isempty(lib_num) && isnumeric(lib_num)))
    fNames = fNames_grp{lib_num};
elseif(nargin > 0)
    fNames = lib_num;
end
      

if(nargin > 0 && numel(fNames) == 1)
    [~, ~, ~, ~, sourceDir, objDir, ~, mexSourceDir] = kgmlm.CUDAlib.myCUDAPaths();
        
    if(~isfolder(mexSourceDir))
        error('CUDA source directory not found! %s\n',mexSourceDir);
    end 
    if(~isfolder(objDir))
        mkdir(objDir);
    end
    
    compileCUDAlibMex = @(fName) mex('-c', sprintf('%s/%s',mexSourceDir,fName), '-outdir', objDir, sprintf('-I%s', sourceDir), sprintf('-I%s', mexSourceDir));%,'COMPFLAGS=$COMPFLAGS /std:c++14'
    
    compileCUDAlibMex(fNames);
else
    for ii = 1:numel(fNames)
        kgmlm.CUDAlib.compileCUDAMexFunc(fNames(ii));
    end
end