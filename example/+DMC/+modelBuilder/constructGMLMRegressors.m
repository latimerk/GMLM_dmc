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
function [GMLMstructure, trials] = constructGMLMRegressors(Neurons, TaskInfo, stimulusConfig, varargin)
% Inputs:
%   Neurons (struct)  - the data structure holding the trial information for each cell
%   TaskInfo (struct) - the data structure holding the task information (directions and categories of the stimuli, bin size)
%   stimulusConfig (array, ND x P x 2) - the regressor setup for the stimuli. ND = number of unique directions used in the task, P is number of parameters, and
%                                        the 2 is for the sample/test stimuli (see getModelSetup which creates this array)
%   dynHspkConfig (optional, array; default = EMPTY) - same as stimulusConfig, but for the stimulus regressors of the stimulus-linked dynamic spike history term
%                                                      If EMPTY, no stimulus-timed dynamic spike history is included
%
%   Parameters:
%       includeResponseDynHspk (default = FALSE) - include the response-timed dynamic spike history components
%       timeBeforeSample_bins (default in getDefaultTrialSettings) - number of bins before sample onset to include in trial window for analysis 
%       timeAfterTestOrResponse_bins (default in getDefaultTrialSettings) - number of bins before after touch-bar release OR test stimulus offset to include in trial window for analysis 
%       bases (struct) - the basis sets to use for the temporal filters (see setupBasis)

%  NOTE: this current function assumes cells may be recorded individually.
%        A more efficient tool for simultaneously recorded populations (GMLM_pop)
%        will be included in a different example.

% process args
ND = numel(TaskInfo.Directions);

[timeBeforeSample_default, timeAfterTestOrResponse_default, delta_t_default] = DMC.modelBuilder.getDefaultTrialSettings(TaskInfo, false, false, false);
bases_default = DMC.modelBuilder.setupBasis('delta_t', delta_t_default, 'bins_post_response', timeAfterTestOrResponse_default);

p = inputParser;
p.CaseSensitive = false;

p.addRequired('Neurons'       , @isstruct);
p.addRequired('TaskInfo'      , @isstruct);
p.addRequired('stimulusConfig', @(sc) iscell(sc) | (isnumeric(sc) & size(sc, 1) == ND  & size(sc, 3) == 2));
p.addOptional('dynHspkConfig' , [], @(dc) isempty(dc) | (isnumeric(dc) & size(dc, 1) == ND  & size(dc, 3) == 2));

p.addParameter('includeResponseDynHspk' , false, @islogical);

p.addParameter('timeBeforeSample_bins',     timeBeforeSample_default,     @(aa) isnumeric(aa) & isscalar(aa) & fix(aa) == aa);
p.addParameter('timeAfterTestOrResponse_bins', timeAfterTestOrResponse_default, @(aa) isnumeric(aa) & isscalar(aa) & fix(aa) == aa);
p.addParameter('bases'  ,  bases_default,    @isstruct);
p.addParameter('silent'  ,  false,    @islogical);

parse(p, Neurons, TaskInfo, stimulusConfig, varargin{:});
dynHspkConfig     = p.Results.dynHspkConfig;

includeResponseDynHspk = p.Results.includeResponseDynHspk;

timeBeforeSample_bins     = p.Results.timeBeforeSample_bins;
timeAfterTestOrResponse_bins = p.Results.timeAfterTestOrResponse_bins;
bases     = p.Results.bases;
silent     = p.Results.silent;

includeStimDynHspk = ~isempty(dynHspkConfig);

if(~iscell(stimulusConfig))
    stimulusConfig = {stimulusConfig};
end

NS = numel(stimulusConfig);

%%

%get total number of trials
totalTrials = 0;
for nn = 1:numel(Neurons)
    NT = numel(Neurons(nn).sampleDir);
    totalTrials = totalTrials + NT;
end


stimLength_bins = ceil(TaskInfo.StimLength_ms/TaskInfo.binSize_ms);
paddedSpikeHistBasis = [zeros(size(bases.spkHist.B) + [1 0]); bases.spkHist.B];%zero padding is to make convolutions easier

N_tensor_groups = NS + 1;
if(includeStimDynHspk)
    N_tensor_groups = N_tensor_groups + 1;
end
if(includeResponseDynHspk)
    N_tensor_groups = N_tensor_groups + 1;
end

%sets up main structure
GMLMstructure.dim_B = size(bases.spkHist.B, 2); % specify size of the linear term
GMLMstructure.Groups = struct('X_shared', cell(N_tensor_groups,1), 'dim_R_max', [], 'dim_A', [], 'name', [], 'dim_names', []); % tensor coefficient groups

for ss = 1:NS
    if(NS > 1)
        GMLMstructure.Groups(ss).name = sprintf("Stimulus_%d", ss);
    else
        GMLMstructure.Groups(ss).name = "Stimulus";
    end
    GMLMstructure.Groups(ss).dim_names = ["timing", "xstim"]; %dimensions of the tensor
    GMLMstructure.Groups(ss).dim_A = 2;      % number of events in group: 2 for sample and test stimuli
    GMLMstructure.Groups(ss).dim_R_max = 16; % max allocated space for rank
    GMLMstructure.Groups(ss).X_shared{1} = bases.stim.B; %shared regressors for each factor/dimension (here factor and dimension are the same)
    GMLMstructure.Groups(ss).X_shared{2} = [stimulusConfig{ss}(:, :, 1); stimulusConfig{ss}(:, :, 2)]; % stacks the sample and test
    GMLMstructure.Groups(ss).dim_T = [size(bases.stim.B, 2) size(stimulusConfig{ss},2)]; % I require specifying the dimensions of each part of the tensor to make sure everything is correct
    GMLMstructure.Groups(ss).factor_idx = 1:2; %factor setup: here all dims are their own factor
end

GMLMstructure.Groups(NS+1).name = "Response";
GMLMstructure.Groups(NS+1).dim_names = "timing";
GMLMstructure.Groups(NS+1).dim_A = 1;
GMLMstructure.Groups(NS+1).dim_R_max = 8; % max allocated space for rank
GMLMstructure.Groups(NS+1).X_shared{1} = bases.response.B;
GMLMstructure.Groups(NS+1).dim_T = size(bases.response.B, 2);
GMLMstructure.Groups(NS+1).factor_idx = 1;

jj = NS+2;
if(includeStimDynHspk)
    jj = jj + 1;
    GMLMstructure.Groups(jj).name = "StimDynamicSpikeHistory";
    GMLMstructure.Groups(jj).dim_names = ["kernel", "timing", "xstim" ];
    GMLMstructure.Groups(jj).dim_A = 2;
    GMLMstructure.Groups(jj).dim_R_max = 3; % max allocated space for rank
    GMLMstructure.Groups(jj).X_shared = {[], bases.stim.B, [dynHspkConfig(:, :, 1); dynHspkConfig(:, :, 2)];};
    GMLMstructure.Groups(jj).dim_T = [size(bases.spkHist.B, 2) size(GMLMstructure.Groups(jj).X_shared{2}, 2) size(GMLMstructure.Groups(jj).X_shared{3}, 2) ];
    GMLMstructure.Groups(jj).factor_idx = 1:3;
end
if(includeResponseDynHspk)
    jj = jj + 1;
    GMLMstructure.Groups(jj).name = "ResponseDynamicSpikeHistory";
    GMLMstructure.Groups(jj).dim_names = ["kernel", "timing"];
    GMLMstructure.Groups(jj).dim_A = 1;
    GMLMstructure.Groups(jj).dim_R_max = 1; % max allocated space for rank
    GMLMstructure.Groups(jj).X_shared = {[], bases.response.B};
    GMLMstructure.Groups(jj).isCompleteTensor = false; % true means full tensor structure is given, not just multilinear form (note that this is always going to be true for this order of tensor)
    GMLMstructure.Groups(jj).isShared = [false true];
    GMLMstructure.Groups(jj).dim_T = [size(bases.spkHist.B, 2) size(GMLMstructure.Groups(jj).X_shared{2}, 2)];
    GMLMstructure.Groups(jj).factor_idx = 1:2;
end

%% setup rescaling MH step
%parameters for an MH step to quickly traverse the scaler part of each component of tensor
MH_scaleSettings.sig =  0.2;
MH_scaleSettings.N   = 5; % I used to sample 10 because I could, but 5 ought to be more than enough
MH_scaleSettings.sample_every = 1;

for jj = 1:numel(GMLMstructure.Groups)
    GMLMstructure.Groups(jj).gibbs_step.dim_H = 0; 
    GMLMstructure.Groups(jj).gibbs_step.sample_func = @(gmlm, params, optStruct, sampleNum, groupNum, optStruct_empty, resultStruct_empty) DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum,  MH_scaleSettings, optStruct_empty, resultStruct_empty);
end

%%

%loop over neurons
tr_idx  = 1; %running index of current trial
trials = struct('Y', cell(totalTrials,1), 'X_lin', [], 'neuron', [], 'Groups', []);

for nn = 1:numel(Neurons)
    if(~silent)
        fprintf('setting up Neuron %d / %d\n', nn, numel(Neurons));
    end
    %set initial index of current neuron
    
    NT = numel(Neurons(nn).sampleDir);
    %% setup spike history
    Y_c = zeros([size(Neurons(nn).Y) size(bases.spkHist.B,2)]);
    for bb = 1:size(bases.spkHist.B, 2)
        Y_c(:, :, bb) = conv2(Neurons(nn).Y, paddedSpikeHistBasis(:, bb), 'same');
    end
    
    %% loop over trials
    for mm = 1:NT
        %set initial index of current trial
        trials(tr_idx).neuron = nn;
    
        %get current trial length
        trStart = Neurons(nn).sampleTime(mm) - timeBeforeSample_bins;%NOTE: all timings here will be relative to this sample stim onset time
        if(~isnan(Neurons(nn).leverTime(mm)))
            %if response released
            trEnd = Neurons(nn).leverTime(mm) + timeAfterTestOrResponse_bins;
        else
            %if response not released and entire test stim viewed
            trEnd = Neurons(nn).testTime(mm) + stimLength_bins + timeAfterTestOrResponse_bins;
        end
        trLength = trEnd - trStart + 1;
        tt_idx_local = (1:trLength) + Neurons(nn).sampleTime(mm) - 1 + timeBeforeSample_bins;
        
        %% setup stim & response groups for the trial
        trials(tr_idx).Groups = struct('X_local', cell(N_tensor_groups,1), 'iX_shared', []);
        
        for jj = 1:N_tensor_groups
            dim_S = numel(GMLMstructure.Groups(jj).dim_T);
            trials(tr_idx).Groups(jj).X_local   = cell(1, dim_S);
            trials(tr_idx).Groups(jj).iX_shared = cell(1, dim_S);
            for ss = 1:dim_S
                if(isempty(GMLMstructure.Groups(jj).X_shared(ss)))
                    trials(tr_idx).Groups(jj).X_local{ss}   = zeros(trLength, GMLMstructure.Groups(jj).dim_T(ss), GMLMstructure.Groups(jj).dim_A); % regressors are filled below
                    trials(tr_idx).Groups(jj).iX_shared{ss} = []; % leave indices blank
                else
                    trials(tr_idx).Groups(jj).X_local{ss} = []; %not a dense regressor, so leave empty
                    trials(tr_idx).Groups(jj).iX_shared{ss} =  zeros(trLength, GMLMstructure.Groups(jj).dim_A); %will fill in these indices later
                end
            end
        end
        
        %% get direction stimulus info 
        for ss = 1:NS
            %stim timings
            trials(tr_idx).Groups(ss).iX_shared{1}(:, :) = getEventTimingsInBasis(trLength, [Neurons(nn).sampleTime(mm) Neurons(nn).testTime(mm)] - trStart, bases.stim.tts);
                %entries here are an index into a row of StimRegressors.timings (okay if out of bounds, GMLM code pads with 0's)

            %sample stim direction
            trials(tr_idx).Groups(ss).iX_shared{2}(:, 1) = Neurons(nn).sampleDir(mm);
                %entries here are an index into a row of StimRegressors.xstim

            %test stim direction
            trials(tr_idx).Groups(ss).iX_shared{2}(:, 2) = Neurons(nn).testDir(mm) + ND; %test cat info should be in a the second block of the xstim regressors (hence the + ND)

            vv = trials(tr_idx).Groups(ss).iX_shared{1}(:, :) > 0 & trials(tr_idx).Groups(ss).iX_shared{1}(:, :) <= size(bases.stim.B,1);
            trials(tr_idx).Groups(ss).iX_shared{2}(~vv) = 0;
        end
        
        %% get response timing info
        trials(tr_idx).Groups(NS+1).iX_shared{1}(:, 1) = getEventTimingsInBasis(trLength, Neurons(nn).leverTime(mm)  - trStart, bases.response.tts);
        %the response timing from the bases is important here: activity reflects response release BEFORE it happens (filter is acausal)

        %% get spike hist
        trials(tr_idx).X_lin = squeeze(Y_c(tt_idx_local, mm, :));
        
        %% get any dynamic hspk terms
        
        jj = NS+2;
        if(includeStimDynHspk)
            jj = jj + 1;
            trials(tr_idx).Groups(jj).X_local{1}      = trials(tr_idx).X_lin;
            trials(tr_idx).Groups(jj).iX_shared{2}(:, 1) = getEventTimingsInBasis(trLength, Neurons(nn).sampleTime(mm) - trStart, bases.stim.tts);
            trials(tr_idx).Groups(jj).iX_shared{2}(:, 2) = getEventTimingsInBasis(trLength, Neurons(nn).testTime(mm)   - trStart, bases.stim.tts);
            trials(tr_idx).Groups(jj).iX_shared{3}(:, 1) = Neurons(nn).sampleDir(mm);
            trials(tr_idx).Groups(jj).iX_shared{3}(:, 2) = Neurons(nn).testDir(mm) + ND; 

            %vv = trials(tr_idx).Groups(jj).iX_shared{2}(:, :) > 0 & trials(tr_idx).Groups(jj).iX_shared{2}(:, :) <= size(bases.stim.B,1);
            %trials(tr_idx).Groups(jj).iX_shared{3}(~vv) = 0;
        end
        if(includeResponseDynHspk)
            jj = jj + 1;
            trials(tr_idx).Groups(jj).X_local{1}      = trials(tr_idx).X_lin;
            trials(tr_idx).Groups(jj).iX_shared{2}(:, 1) = getEventTimingsInBasis(trLength, Neurons(nn).leverTime(mm) - trStart, bases.response.tts);
        end
        
        %% get spike observations
        trials(tr_idx).Y = Neurons(nn).Y(tt_idx_local, mm);
        
        %% add trial length to running index
        tr_idx  = tr_idx  + 1;
    end
end
if(~silent)
    fprintf('Done setting up data\n');
end

end

%% gets the timing indexes for a point event
%    All inputs are in units of bins
% 
%         trialLength (scalar: positive integer)
%
%         eventTimes  (array: integers or nans)  time (in trial) of one or more events
%
%         basis_tts   (array: length of basis functions) time of the basis function points relative to event onset
%                     assumes basis_tts is a range with increment +1.    eg., 1:20,   -10:32
function [iX] = getEventTimingsInBasis(trialLength, eventTimes, basis_tts)
    A = numel(eventTimes);
    
    iX = repmat((1:trialLength)', [1 A]) - eventTimes(:)' - basis_tts(1);
    
    iX(isnan(iX)) = -1;
end