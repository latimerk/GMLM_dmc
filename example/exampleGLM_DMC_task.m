%  Package GMLM_dmc for dimensionality reduction of neural data.
%   
%  References
%   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
%   decisions in parietal cortex reflects long-term training history.
%   bioRxiv
%
%  Copyright (c) 2021 Kenneth Latimer
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
%% load data and get task information
folder = fileparts(which('exampleGLM_DMC_task'));
fname = sprintf('%s/DMC_data/sub_B_data.mat', folder);
load(fname, 'Neurons', 'TaskInfo'); %see exampleGMLM_DMC_task.m for more info on the data

%which parts of the trial we will keep (UNITS ARE BINS)
timeBeforeSample     = 0; %time before sample stimulus on that will be include
timeAfterTestOrResponse = ceil(50/TaskInfo.binSize_ms); %time after response OR test stim off that will be kept

GPU_to_use = 0;
gpuDoublePrecision = true;

%% setup basis functions & task setup
ND = numel(TaskInfo.Directions);
bases = DMC.modelBuilder.setupBasis('bins_post_response', timeAfterTestOrResponse, 'delta_t', TaskInfo.binSize_ms/1e3, 'plotBases', false);
[stimulus_config, stimPrior_setup] = DMC.modelBuilder.getModelSetup(TaskInfo, 'dirTuningType', 'cosine', 'dirSameSampleTest', true, 'includeCategory', true);

%% setup design matrix for ONE cell
cellToFit = 1;
[GLMstructure, trials] = DMC.modelBuilder.constructGLMdesignMatrix(Neurons, TaskInfo, cellToFit, stimulus_config, 'timeBeforeSample_bins', timeBeforeSample, 'timeAfterTestOrResponse_bins', timeAfterTestOrResponse, 'bases', bases);

%% setup prior distribution over filters
[prior_function_stim, prior_function_response, prior_function_spkHist, prior_function_glmComplete, levPrior_setup, spkHistPrior_setup] = DMC.modelBuilder.setupPriorFunctions(stimPrior_setup, bases);

GLMstructure.prior.dim_H          = stimPrior_setup.NH + levPrior_setup.NH + spkHistPrior_setup.NH;
GLMstructure.prior.log_prior_func = prior_function_glmComplete;

%% build the GMLM object
glm = kgmlm.kcGLM(GLMstructure, trials, TaskInfo.binSize_ms./1e3);

%%
return;
%% do MLE then MAP fitting on GLM with hyperparameters set by evidence optimization
%load to GPU
if(gpuDeviceCount() > 0 && ~glm.isOnGPU())
    glm.toGPU(GPU_to_use, 'useDoublePrecision', gpuDoublePrecision);
else
    warning("No GPUs found. CPU computation can be a lot slower!");
end

%run evidence optimization
[params_mle, results_mle] = glm.computeMLE('display', 'off');
[params_map, results_map] = glm.computeMAPevidenceOptimization();

%plot fit
R_sample_stim = stimulus_config(:,:,1);
R_sample_stim_dirOnly = R_sample_stim; % for the cosine tuning model, we can try to decompose category and direction (this piece doesn't make sense for the full tuning model)
R_sample_stim_dirOnly(:,1:4) = 0;
R_sample_stim_catOnly = [1 0 0 0 0 0;
                         0 1 0 0 0 0];
DMC.plottingTools.plotGLMFit(params_mle, R_sample_stim, R_sample_stim_dirOnly, bases, TaskInfo, [], 'MLE');
DMC.plottingTools.plotGLMFit(params_map, R_sample_stim, R_sample_stim_dirOnly, bases, TaskInfo, [], 'MAP fit')

%%
return;
%% cross-validated evidence optimization
%load to GPU
if(gpuDeviceCount() > 0 && ~glm.isOnGPU())
    glm.toGPU(GPU_to_use, 'useDoublePrecision', gpuDoublePrecision);
else
    warning("No GPUs found. CPU computation can be a lot slower!");
end

K = 10;
cvInds =  crossvalind('KFold', Neurons(cellToFit).sampleDir, K); % split folds over sample directions

%run evidence optimization
[paramStruct_xv, results_test_xv, results_train_xv] = glm.computeMAPevidenceOptimization_crossValidated(cvInds);
    %nans in the results_xv indicate that trial was not in training (or test) set

%clear from GPU
glm = glm.freeGPU();

%%
return;
%% Bayesian fitting with MCMC
%load to GPU
if(gpuDeviceCount() > 0 && ~glm.isOnGPU())
    glm.toGPU(GPU_to_use, 'useDoublePrecision', gpuDoublePrecision);
else
    warning("No GPUs found. CPU computation can be a lot slower!");
end

%run evidence optimization
DEBUG   = true;
HMC_settings = glm.setupHMCparams([], [], DEBUG); %note: DEBUG uses few samples just to make sure the code can run
params_init  = glm.getRandomParamStruct();
[samples, summary, HMC_settings, params_final, M] = glm.runHMC_simple( params_init, HMC_settings, "figure", 10);

%clear from GPU
glm = glm.freeGPU();


