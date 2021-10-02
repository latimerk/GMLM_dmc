function [] = compileCUDAMexLibs(idx)

[~, CUDAdirectory, ~, ~, sourceDir, objDir] = kgmlm.CUDAlib.myCUDAPaths();
fNames = {[sourceDir 'kcBase.cu ' sourceDir 'kcGMLM_dataStructures.cu ' sourceDir 'kcGMLM_computeBlock.cu ' sourceDir 'kcGMLM.cu '];
          [sourceDir 'kcBase.cu ' sourceDir 'kcGLM_dataStructures.cu ' sourceDir 'kcGLM_computeBlock.cu ' sourceDir 'kcGLM.cu ' ];
          [sourceDir 'kcBase.cu ' sourceDir 'kcGMLMPop_dataStructures.cu ' sourceDir 'kcGMLMPop_computeBlock.cu ' sourceDir 'kcGMLMPop.cu ']};
outNames = {'kcGMLM_lib', 'kcGLM_lib', 'kcGMLMPop_lib'};

if(nargin > 0)
    fNames = fNames(idx);
    outNames = outNames(idx);
end

if(~isfolder(objDir))
    mkdir(objDir);
end

for ii = 1:numel(fNames)
    outputName = outNames{ii};
    files = fNames{ii};
    
    %% gets CUDA architecture for compiling: auto-detects using first available CUDA device
    if(gpuDeviceCount() > 0)
        d = gpuDevice(1);
        computeLevel = round(str2double(d.ComputeCapability)*10);
        %arch_flag = sprintf('--gpu-architecture sm_%d',computeLevel');
        
        arch_flag = sprintf('-gencode arch=compute_%d,code=sm_%d', computeLevel, computeLevel);
        
        fprintf('Using architecture detected on first GPU: %s\n',arch_flag);
    else
        arch_flag = '';
        fprintf('No GPUs detected: using default architecture\n');
    end
    
    %% gets system command to perform compiling with NVCC (system specific)
    if(ispc)
        fExtension = '.lib';
        compileCommand = ['nvcc -O2 -Xcompiler "/MD" --lib --shared  --machine 64 ' arch_flag ' -I' sourceDir  ' ' files ' -o '  objDir outputName fExtension]; 
    else
        fExtension = '.a';
        extraArgs = '-O3 -Xcompiler -fPIC';
        compileCommand = [CUDAdirectory 'bin/nvcc --lib -shared -m64  ' arch_flag ' ' extraArgs  ' -I' sourceDir  ' ' files ' -o '  objDir outputName fExtension]; 
    end
    
    %% run compiler for library
    system(compileCommand);
end

end