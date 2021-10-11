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
function [GLMstructure, trials] = constructGLMdesignMatrix(Neurons, TaskInfo, cellToFit, stimulusConfig, varargin)
%   Neurons (struct)  - the data structure holding the trial information for each cell
%   TaskInfo (struct) - the data structure holding the task information (directions and categories of the stimuli, bin size)
%   cellToFit (integer) - which cell in Neurons to fit
%   stimulusConfig (array, ND x P x 2) - the regressor setup for the stimuli. ND = number of unique directions used in the task, P is number of parameters, and
%                                        the 2 is for the sample/test stimuli (see getModelSetup which creates this array)
%
%   Parameters:
%       timeBeforeSample_bins (default in getDefaultTrialSettings) - number of bins before sample onset to include in trial window for analysis 
%       timeAfterTestOrLever_bins (default in getDefaultTrialSettings) - number of bins before after touch-bar release OR test stimulus offset to include in trial window for analysis 
%       bases (struct) - the basis sets to use for the temporal filters (see setupBasis)


% process args
ND = numel(TaskInfo.Directions);

[timeBeforeSample_default, timeAfterTestOrLever_default, delta_t_default] = DMC.modelBuilder.getDefaultTrialSettings(TaskInfo, false, false, false);
bases_default = DMC.modelBuilder.setupBasis('delta_t', delta_t_default, 'bins_post_lever', timeAfterTestOrLever_default);

p = inputParser;
p.CaseSensitive = false;

p.addRequired('Neurons'     ,    @isstruct);
p.addRequired('TaskInfo'     ,     @isstruct);
p.addRequired('cellToFit'     ,     @(aa) isnumeric(aa) & fix(aa) == aa & aa > 0 & aa <= numel(Neurons));
p.addRequired('stimulusConfig',   @(sc) isnumeric(sc) & size(sc, 1) == ND  & size(sc, 3) == 2);

p.addParameter('timeBeforeSample_bins',     timeBeforeSample_default,     @(aa) isnumeric(aa) & isscalar(aa) & fix(aa) == aa);
p.addParameter('timeAfterTestOrLever_bins', timeAfterTestOrLever_default, @(aa) isnumeric(aa) & isscalar(aa) & fix(aa) == aa);
p.addParameter('bases'  ,  bases_default,    @isstruct);

parse(p, Neurons, TaskInfo, cellToFit, stimulusConfig, varargin{:});

timeBeforeSample_bins     = p.Results.timeBeforeSample_bins;
timeAfterTestOrLever_bins = p.Results.timeAfterTestOrLever_bins;
bases     = p.Results.bases;


%%
fprintf('Parsing data from cell %d... ', cellToFit);

totalTrials = numel(Neurons(cellToFit).sampleDir);

N_stim_filts = size(stimulusConfig, 2) ;
N_stim_cov   = N_stim_filts * size(bases.stim.B, 2);
N_lev_cov    = size(bases.response.B, 2);
N_hspk_cov   = size(bases.spkHist.B, 2);

if(N_hspk_cov > 0)
    GLMstructure.group_names = ["stim", "lever", "hspk", "const"];
    GLMstructure.dim_Ks = [N_stim_cov, N_lev_cov, N_hspk_cov, 1];
else
    GLMstructure.group_names = ["stim", "lever",  "const"];
    GLMstructure.dim_Ks = [N_stim_cov, N_lev_cov, 1];
end

% setup spike history
if(N_hspk_cov > 0)
    paddedSpikeHist = [zeros(size(bases.spkHist.B) + [1 0]); bases.spkHist.B];%zero padding is to make convolutions easier
    Y_c = zeros([size(Neurons(cellToFit).Y) size(bases.spkHist.B,2)]);
    for bb = 1:size(bases.spkHist.B, 2)
        Y_c(:, :, bb) = conv2(Neurons(cellToFit).Y, paddedSpikeHist(:, bb), 'same');
    end
end

trials = struct('X', cell(totalTrials,1), 'Y', []);

% parse out all trials
for mm = 1:totalTrials
    trStart = Neurons(cellToFit).sampleTime(mm) - timeBeforeSample_bins;
    if(~isnan(Neurons(cellToFit).leverTime(mm)))
        %if lever released
        trEnd = Neurons(cellToFit).leverTime(mm) + timeAfterTestOrLever_bins;
    else
        %if lever not released and entire test stim viewed
        trEnd = Neurons(cellToFit).testTime(mm) + ceil(TaskInfo.StimLength_ms/TaskInfo.binSize_ms) + timeAfterTestOrLever_bins;
    end
    trLength = trEnd - trStart + 1;
    
    tt_idx = (1:trLength) + Neurons(cellToFit).sampleTime(mm) - 1 + timeBeforeSample_bins;

    if(N_hspk_cov > 0)
        trials(mm).X = cell(1,4);
    else
        trials(mm).X = cell(1,3);
    end
    
    %% get direction stimulus info
    sampleStimRegressors = zeros(trLength, N_stim_cov);
    testStimRegressors   = zeros(trLength, N_stim_cov);

    %sample stim direction
    sampleStimDirection  = Neurons(cellToFit).sampleDir(mm);
    sampleFilterLoadings = stimulusConfig(sampleStimDirection, :, 1);

    %sample stim timing & filter setup
    sampleBases_tts = (1:trLength) - timeBeforeSample_bins; %NOTE: all timings here will be relative to this sample stim onset time
    valid_tts = sampleBases_tts > 0 & sampleBases_tts <= size(bases.stim.B, 1);
    sampleStimRegressors(valid_tts, :) = kron(sampleFilterLoadings, bases.stim.B( sampleBases_tts(valid_tts), :)); 

    %test stim direction
    testStimDirection  = Neurons(cellToFit).testDir(mm);
    testFilterLoadings = stimulusConfig(testStimDirection, :, 2);

    %test stim timing & filter setup
    testBases_tts = (1:trLength) - timeBeforeSample_bins - (Neurons(cellToFit).testTime(mm) - Neurons(cellToFit).sampleTime(mm)); %NOTE: all timings here will be relative to this sample stim onset time
    valid_tts = testBases_tts > 0 & testBases_tts <= size(bases.stim.B, 1);
    testStimRegressors(valid_tts, :) = kron(testFilterLoadings, bases.stim.B( testBases_tts(valid_tts), :)); 

    trials(mm).X{1} = sampleStimRegressors + testStimRegressors;
    
    %% get lever timing info
    trials(mm).X{2} = zeros(trLength, N_lev_cov);
    if(~isnan(Neurons(cellToFit).leverTime(mm))) %if lever release happens
        lever_tts = (1:trLength) - timeBeforeSample_bins - (Neurons(cellToFit).leverTime(mm) - Neurons(cellToFit).sampleTime(mm)) - bases.response.tts(1);
        %the lever timing from the bases is important here: activity reflects lever release BEFORE it happens (filter is acausal)
        %I ignored it for the stimulus timing because I know how those filters were setup
        valid_tts = lever_tts > 0 & lever_tts <= size(bases.stim.B, 1);
        trials(mm).X{2}( valid_tts, :) =  bases.response.B( lever_tts(valid_tts), :); 
    end

    %% get spike hist
    if(N_hspk_cov > 0)
        trials(mm).X{3} = squeeze(Y_c(tt_idx, mm, :));
    end
    
    %% get spike observations
    trials(mm).Y = Neurons(cellToFit).Y(tt_idx, mm);

    %% add constant term
    trials(mm).X{end} = ones(numel(trials(mm).Y), 1);

end


fprintf('done.\n');