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
if(isempty(which('hosvd')))
    error('Please add tensor toolbox to your path for visualizations.')
end

%% load data and get task information
folder = fileparts(which('exampleGMLM_DMC_task'));
fname = sprintf('%s/DMC_data/sub_B_data.mat', folder);
load(fname, 'Neurons', 'TaskInfo');

%examine the data structure here
%Note that Neurons(nn).sampleDir is stored here as an index variable, not an angle, for ease
%the angles are given in TaskInfo.Directions, and the corresponding category numbers for the directions
%are in TaskInfo.Categories
%   Neurons(nn).*Time are all times in units of bins
%   Neurons(nn).Y is the bined spike trains.
%      Neurons(nn).Y( Neurons(nn).sampleTime(mm), mm) == spike count on the bin when the sample stimulus came on
%   To make things quicker, I've made the bin size to 5ms, rather than 1ms. The spike history basis here might need to be changed for 1ms bins (for the refractory period)
%
%Note that this code assumes the sample stimulus presentations are all the same length. The GMLM code
% can handle variable stim presentation lengths, but would require loading up the data differently.

%which parts of the trial we will keep (UNITS ARE BINS)
timeBeforeSample     = 0; %time before sample stimulus on that will be include
timeAfterTestOrLever = ceil(50/TaskInfo.binSize_ms); %time after lever release OR test stim off that will be kept

GPUs_to_use = 0; %the device numbers to use. This can be a vector of GPUs: data will be divided across them.
                 %Setting "GPUs_to_use = 2" will use device number 2 (zero indexed), not two GPUs;
                 %Setting "GPUs_to_use = [0 1]" will use the first 2 GPUs

gpuDoublePrecision = true; %use single or double precision on the GPU

%% setup basis functions & task setup
ND = numel(TaskInfo.Directions);
bases = DMC.modelBuilder.setupBasis('bins_post_lever', timeAfterTestOrLever, 'delta_t', TaskInfo.binSize_ms/1e3, 'plotBases', false);
[stimulus_config, stimPrior_setup] = DMC.modelBuilder.getModelSetup(TaskInfo, 'dirTuningType', 'cosine', 'dirSameSampleTest', true, 'includeCategory', true);

%% parse data
[GMLMstructure, trials] = DMC.modelBuilder.constructGMLMRegressors(Neurons, TaskInfo, stimulus_config, 'timeBeforeSample_bins', timeBeforeSample, 'timeAfterTestOrLever_bins', timeAfterTestOrLever, 'bases', bases);

%% setup prior distributions for model parameters (used for MCMC inference, or MAP if you wish to specificy an init point to the MAP_MLE function that has proper hyperparameters)
[prior_function_stim, prior_function_lever, prior_function_spkHist, prior_function_glmComplete, levPrior_setup, spkHistPrior_setup] = DMC.modelBuilder.setupPriorFunctions(stimPrior_setup, bases);

GMLMstructure.prior.dim_H          = spkHistPrior_setup.NH;
GMLMstructure.prior.log_prior_func = prior_function_spkHist;

GMLMstructure.Groups(1).prior.dim_H          = stimPrior_setup.NH;
GMLMstructure.Groups(1).prior.log_prior_func = prior_function_stim;

GMLMstructure.Groups(2).prior.dim_H          = levPrior_setup.NH;
GMLMstructure.Groups(2).prior.log_prior_func = prior_function_lever;

%% build the GMLM object

gmlm = kgmlm.GMLM(GMLMstructure, trials, TaskInfo.binSize_ms./1e3);
gmlm.setDimR('Stimulus', 7);

return;
%% fit MLE
if(~gmlm.isOnGPU())
    gmlm.toGPU(GPUs_to_use, 'useDoublePrecision', gpuDoublePrecision);
end

[params_mle, results_mle, params_init] = gmlm.computeMLE();

%now fit again by alternating search directions and the same init point
%[params_alt_mle, results_alt_mle] = gmlm.computeMLE(params_init, 'alternating_opt', true, 'max_iters', 10, 'max_quasinewton_steps', 2e2);

%% fit MLE - cross validated
if(~gmlm.isOnGPU())
    gmlm.toGPU(GPUs_to_use);
end

%divides CV sets up by neuron and sample direction
K = 10;
ctr = 0;
trialNeuronSampleDir_ids = nan(gmlm.dim_M, 1);
for ii = 1:numel(Neurons)
    trialNeuronSampleDir_ids(ctr + (1:numel(Neurons(ii).sampleDir))) = Neurons(ii).sampleDir + ii*100;
    ctr = ctr + numel(Neurons(ii).sampleDir);
end
cvInds =  crossvalind('KFold', trialNeuronSampleDir_ids, K);

[params_mle_cv, results_test_mle_cv, results_train_mle_cv, params_init_cv] = gmlm.computeMLE_crossValidated(cvInds);


%% fit MAP estimate
if(~gmlm.isOnGPU())
    gmlm.toGPU(GPUs_to_use);
end

% git init point with hyperparameters set (I'm just going to optimize with the random hyperparams for now)
params_init = gmlm.getRandomParamStruct('includeHyperparameters', true);

[params_map, results_map] = gmlm.computeMAP(params_init);

% now do cross validation
[params_map_cv, results_test_map_cv, results_train_map_cv] = gmlm.computeMAP_crossValidated(cvInds, params_init);

%% Bayesian inference via HMC

if(~gmlm.isOnGPU())
    gmlm.toGPU(GPUs_to_use);
end

%get settings (use DEBUG mode for fewer samples)
DEBUG = true;
HMC_settings = gmlm.setupHMCparams([], [], DEBUG);
params_init  = gmlm.getRandomParamStruct();
[samples, summary, HMC_settings, params_final, M] = gmlm.runHMC_simple( params_init, HMC_settings, "figure", 10);

%MAP estimate using the posterior median hyperparams
params_final.H(:) = median(samples.H(:,(HMC_settings.nWarmup+1):end),2);
for jj = 1:gmlm.dim_J
    params_final.Groups(jj).H(:) = median(samples.Groups(jj).H(:,(HMC_settings.nWarmup+1):end),2);
end
[params_map, results_map] = gmlm.computeMAP(params_final);

%plot MAP fit
R_sample_stim = stimulus_config(:,:,1);
R_sample_stim_dirOnly = R_sample_stim; % for the cosine tuning model, we can try to decompose category and direction (this piece doesn't make sense for the full tuning model)
R_sample_stim_dirOnly(:,1:4) = 0;
R_sample_stim_catOnly = [1 0 0 0 0 0;
                         0 1 0 0 0 0];

DMC.plottingTools.plotGMLMFit(TaskInfo, params_map, bases, R_sample_stim, R_sample_stim_dirOnly, R_sample_stim_catOnly);

%% clear GMLM from gpu
% MATLAB's garbage collector cannot see the internal memory usage in the GMLM when loaded to the GPU. The destructor for GMLM calls this function.
gmlm.freeGPU();

% if something goes wrong with CUDA, the devices can be reset (Clearing all GPU memory) with this function:
% kgmlm.CUDAlib.kcResetDevices(GPUs_to_use);



