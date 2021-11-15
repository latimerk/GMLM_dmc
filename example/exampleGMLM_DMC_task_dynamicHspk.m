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
%% sets up the GMLM with the dynamic spike history terms
if(isempty(which('ktensor')))
    error('Please add tensor toolbox to your path for visualizations.')
end

%% load data and get task information
folder = fileparts(which('exampleGMLM_DMC_task'));
fname = sprintf('%s/DMC_data/sub_B_data.mat', folder);
load(fname, 'Neurons', 'TaskInfo');

%which parts of the trial we will keep (UNITS ARE BINS)
timeBeforeSample     = 0; %time before sample stimulus on that will be include
timeAfterTestOrResponse = ceil(50/TaskInfo.binSize_ms); %time after response OR test stim off that will be kept

GPUs_to_use = 0; %the device numbers to use. This can be a vector of GPUs: data will be divided across them.
                 %Setting "GPUs_to_use = 2" will use device number 2 (zero indexed), not two GPUs;
                 %Setting "GPUs_to_use = [0 1]" will use the first 2 GPUs

gpuDoublePrecision = true; %use single or double precision on the GPU

%% setup basis functions & task setup
ND = numel(TaskInfo.Directions);
bases = DMC.modelBuilder.setupBasis('bins_post_response', timeAfterTestOrResponse, 'delta_t', TaskInfo.binSize_ms/1e3, 'plotBases', false);
[stimulus_config, stimPrior_setup]   = DMC.modelBuilder.getModelSetup(TaskInfo, 'dirTuningType', 'cosine', 'dirSameSampleTest', true, 'includeCategory', true);
[dynHspk_config, dynHspkPrior_setup] = DMC.modelBuilder.getModelSetup(TaskInfo, 'dirTuningType', 'none', 'includeCategory', false);

%% parse data
[GMLMstructure, trials] = DMC.modelBuilder.constructGMLMRegressors(Neurons, TaskInfo, stimulus_config, dynHspk_config, 'includeResponseDynHspk', true, 'timeBeforeSample_bins', timeBeforeSample, 'timeAfterTestOrResponse_bins', timeAfterTestOrResponse, 'bases', bases);

%% setup prior distributions for model parameters (used for MCMC inference, or MAP if you wish to specificy an init point to the MAP_MLE function that has proper hyperparameters)
[prior_function_stim, prior_function_response, prior_function_spkHist, prior_function_glmComplete, responsePrior_setup, spkHistPrior_setup, prior_function_dynHspk] = DMC.modelBuilder.setupPriorFunctions(stimPrior_setup, bases, dynHspkPrior_setup);

GMLMstructure.prior.dim_H          = spkHistPrior_setup.NH;
GMLMstructure.prior.log_prior_func = prior_function_spkHist;

GMLMstructure.Groups(1).prior.dim_H          = stimPrior_setup.NH;
GMLMstructure.Groups(1).prior.log_prior_func = prior_function_stim;

GMLMstructure.Groups(2).prior.dim_H          = responsePrior_setup.NH;
GMLMstructure.Groups(2).prior.log_prior_func = prior_function_response;

% the dynamic hspk terms
    % stimulus
GMLMstructure.Groups(3).prior.dim_H          = dynHspkPrior_setup.NH;
GMLMstructure.Groups(3).prior.log_prior_func = prior_function_dynHspk;

    % touch-bar
GMLMstructure.Groups(4).prior.dim_H          = responsePrior_setup.NH;
GMLMstructure.Groups(4).prior.log_prior_func = prior_function_response;



MH_scaleSettings.sig =  0.2;
MH_scaleSettings.N   = 5; % I used to sample 10 because I could, but 5 ought to be more than enough
MH_scaleSettings.sample_every = 1;
GMLMstructure.Groups(1).gibbs_step.dim_H = 0;
GMLMstructure.Groups(1).gibbs_step.sample_func = @(gmlm, params, optStruct, sampleNum, groupNum, opts, results) DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, stimPrior_setup, opts, results);
GMLMstructure.Groups(2).gibbs_step.dim_H = 0;
GMLMstructure.Groups(2).gibbs_step.sample_func = @(gmlm, params, optStruct, sampleNum, groupNum, opts, results) DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, responsePrior_setup, opts, results);
GMLMstructure.Groups(3).gibbs_step.dim_H = 0;
GMLMstructure.Groups(3).gibbs_step.sample_func = @(gmlm, params, optStruct, sampleNum, groupNum, opts, results) DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, stimPrior_setup, opts, results);
GMLMstructure.Groups(4).gibbs_step.dim_H = 0;
GMLMstructure.Groups(4).gibbs_step.sample_func = @(gmlm, params, optStruct, sampleNum, groupNum, opts, results) DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, responsePrior_setup, opts, results);
%% build the GMLM object

gmlm = kgmlm.GMLM(GMLMstructure, trials, TaskInfo.binSize_ms./1e3);
gmlm.setDimR('Stimulus', 7);
gmlm.setDimR('Response', 3);
gmlm.setDimR('StimDynamicSpikeHistory', 1);
gmlm.setDimR('ResponseDynamicSpikeHistory', 1);

return;

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

%% plot dynamic spike history components

% normalize components, standardize signs (keeps coefficient tensor the same)

dynHspk_stim.timing = bases.stim.B * params_map.Groups(3).T{2};
dynHspk_stim.kernel = bases.spkHist.B * params_map.Groups(3).T{1};
dynHspk_stim.sample_weighting = params_map.Groups(3).T{3}(1,:);
dynHspk_stim.test_weighting   = params_map.Groups(3).T{3}(2,:);
dynHspk_stim.neuron_loading_weights   = params_map.Groups(3).V;
        
mm = sign(dynHspk_stim.sample_weighting(1,:));
dynHspk_stim.sample_weighting = dynHspk_stim.sample_weighting.*mm;
dynHspk_stim.timing   = dynHspk_stim.timing./mm;

mm = mode(sign(dynHspk_stim.neuron_loading_weights), 1);
dynHspk_stim.neuron_loading_weights = dynHspk_stim.neuron_loading_weights.*mm;
dynHspk_stim.timing   = dynHspk_stim.timing./mm;

mm = sign(dynHspk_stim.kernel(1,:));
dynHspk_stim.kernel = dynHspk_stim.kernel.*mm;
dynHspk_stim.timing   = dynHspk_stim.timing./mm;

mm = sqrt(sum(dynHspk_stim.kernel.^2, 1)) ;
dynHspk_stim.kernel = dynHspk_stim.kernel./mm;
dynHspk_stim.neuron_loading_weights   = dynHspk_stim.neuron_loading_weights.*mm;

mm = sqrt(sum(dynHspk_stim.timing.^2, 1)) ;
dynHspk_stim.timing = dynHspk_stim.timing./mm;
dynHspk_stim.neuron_loading_weights   = dynHspk_stim.neuron_loading_weights.*mm;

mm = sqrt(sum(dynHspk_stim.sample_weighting.^2, 1)) ;
dynHspk_stim.sample_weighting = dynHspk_stim.sample_weighting./mm;
dynHspk_stim.test_weighting = dynHspk_stim.test_weighting./mm;
dynHspk_stim.neuron_loading_weights   = dynHspk_stim.neuron_loading_weights.*mm;

figure(11);
clf;

subplot(2, 2, 1);
hold on
plot([0 bases.stim.tts_0(end)], [0 0], 'k:', 'linewidth', 0.5, 'handlevisibility', 'off');
plot(bases.stimBasis_tts * TaskInfo.binSize_ms, dynHspk_stim.timing);
xlabel('time from sample onset (ms)');
ylabel('weight (normalized)');
title('dynamic spike history: stimulus-timing kernel');

subplot(2, 2, 2);
hold on
plot([0 bases.spkHist.tts_(end)], [0 0], 'k:', 'linewidth', 0.5, 'handlevisibility', 'off');
plot(bases.spkHistBasis_tts * TaskInfo.binSize_ms, dynHspk_stim.kernel);
xlabel('time from previous spike (ms)');
ylabel('weight (normalized)');
title('dynamic hspk kernel');

subplot(2, 2, 3);
histogram(dynHspk_stim.neuron_loading_weights);
ylabel('n. cells');
xlabel('weights');
title('neuron loading weights');

% mean spike history at different times
subplot(2, 2, 4);
hold on

plot([0 bases.spkHist.tts_0(end)], [0 0], 'k:', 'linewidth', 0.5, 'handlevisibility', 'off');

PS = [100 1100];
B_spk = bases.spkHist.B * params_map.B; % per-neuron fixed hspk
color_ps = [0 0 0;
            1 0 0];
labels = cell(numel(PS),1);
for pp = 1:numel(PS)
    % get timepoint in basis closest to PS(pp)
    [~,tt] = min(abs(PS(pp) - bases.stim.tts_0));
    
    
    hspk_c = B_spk + double(ktensor(dynHspk_stim.sample_weighting(:), {dynHspk_stim.kernel, dynHspk_stim.neuron_loading_weights, dynHspk_stim.timing(tt,:)}));
    
    y = mean(hspk_c, 2);
    sem = std(hspk_c, [], 2) ./ sqrt(size(hspk_c,2));
    eb_1 = y - 2 * sem;
    eb_2 = y + 2 * sem;
    
    DMC.plottingTools.plotLineWithErrorRegion(gca, bases.spkHist.tts_0, y, eb_1, eb_2, color_ps(pp,:));
    
    labels{pp} = sprintf("%d ms after sample", PS(pp));
end
legend(labels);
title("mean effective spike history");
xlabel('time from previous spike (ms)');
ylabel('weight (log gain per spike)');



%% clear GMLM from gpu
% MATLAB's garbage collector cannot see the internal memory usage in the GMLM when loaded to the GPU. The destructor for GMLM calls this function.
gmlm.freeGPU();

% if something goes wrong with CUDA, the devices can be reset (Clearing all GPU memory) with this function:
% kgmlm.CUDAlib.kcResetDevices(GPUs_to_use);



