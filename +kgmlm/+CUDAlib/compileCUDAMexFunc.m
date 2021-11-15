function [fNames] = compileCUDAMexFunc(fName)


fNames = ["kcGMLM_mex_create.cpp", "kcGMLM_mex_clear.cpp",  "kcGMLM_mex_computeLL.cpp", "kcGMLM_mex_computeLL_async.cpp", "kcGMLM_mex_computeLL_gather.cpp", ...
          "kcGLM_mex_create.cpp", "kcGLM_mex_clear.cpp",  "kcGLM_mex_computeLL.cpp", ...
          "kcGMLMPop_mex_create.cpp", "kcGMLMPop_mex_clear.cpp",  "kcGMLMPop_mex_computeLL.cpp", "kcGMLMPop_mex_computeLL_async.cpp", "kcGMLMPop_mex_computeLL_gather.cpp"];
      
if(nargin > 0 && numel(fName) > 1)
    for ii = 1:numel(fName)
        kgmlm.CUDAlib.compileCUDAMexFunc(fName(ii));
    end
elseif(nargin > 0 && ~isempty(fName))
    if(isnumeric(fName))
        fName = fNames{fName};
    end
    [~, ~, ~, ~, sourceDir, objDir] = kgmlm.CUDAlib.myCUDAPaths();
        
    if(~isfolder(sourceDir))
        error('CUDA source directory not found! %s\n',sourceDir);
    end 
    if(~isfolder(objDir))
        mkdir(objDir);
    end
    
    compileCUDAlibMex = @(fName) mex('-c', sprintf('%s/%s',sourceDir,fName), '-outdir', objDir, sprintf('-I%s', sourceDir));%,'COMPFLAGS=$COMPFLAGS /std:c++14'
    
    compileCUDAlibMex(fName );
elseif((nargin == 0 || isempty(fName)) && nargout == 0)
    for ii = 1:numel(fNames)
        kgmlm.CUDAlib.compileCUDAMexFunc(fNames(ii));
    end
end