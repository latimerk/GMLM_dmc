% Runs the HMC algorithm for the GMLM
% Computes PSISLOO and WAIC on the samples.
%
% NOTE: uses HMC_settings.samplesFile to store some partial results. This file can get big, but is deleted after running is complete.
%
%  Inputs:
%      params_init  : the initial parameters for sample = 1
%      HMC_settings : HMC settings struct (see setupHMCparams)
% 
%    Optional (key/value pairs)
%       figure                (number;  default = nan ) : number of the figure to plot MCMC traces (nan means do not plot) 
%       sampleHyperparameters (logical; default = true) : true/false to sample hyperparams. If false, leaves them fixed given params_init
%       trial_weights         (array size gmlm.dim_M; default = []) : weighting of the trials (empty for all trial weights 1)

function [samples, summary, HMC_settings, paramStruct, M] = runHMC_simple(obj, params_init, HMC_settings, varargin)
p = inputParser;
p.CaseSensitive = false;

addRequired(p, 'params_init',  @(aa)(isstruct(aa) | isempty(aa)));
addRequired(p, 'settings', @isstruct);
addParameter(p, 'figure' ,    nan, @isnumeric);
addParameter(p, 'optStruct' ,   [], @(aa) isempty(aa) | obj.verifyComputeOptionsStruct(aa));
addParameter(p, 'sampleHyperparameters', true, @islogical);
addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

parse(p, params_init, HMC_settings, varargin{:});
% then set/get all the inputs out of this structure
optStruct          = p.Results.optStruct;
paramStruct        = p.Results.params_init;
HMC_settings       = p.Results.settings;
trial_weights      = p.Results.trial_weights;
sampleHyperparameters = p.Results.sampleHyperparameters;
figNum = p.Results.figure;
    
if(~obj.isOnGPU())
    error('Must load gmlm onto GPU before running HMC!');
end

%% sets up the hmc momentum cov matrices
if(isempty(optStruct))
    optStruct = obj.getComputeOptionsStruct(true, 'trial_weights', ~isempty(trial_weights), 'includeHyperparameters', sampleHyperparameters);
end
optStruct_empty = obj.getComputeOptionsStruct(false, 'trial_weights', ~isempty(trial_weights), 'includeHyperparameters', sampleHyperparameters);
if(~isempty(trial_weights))
    optStruct_empty.trial_weights(:) = trial_weights;
    optStruct.trial_weights(:) = trial_weights;
end

nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct);

% if(sampleHyperparameters)
%     TotalParameters = obj.getNumberOfParameters(-1) + obj.dim_H(-1);
% else
%     TotalParameters = obj.getNumberOfParameters(-1);
% end

H_var = (1/10)^2; %initial momentum term for hyperparams

paramStruct_2 = paramStruct;
paramStruct_2.W(:) = 1;
if(isfield(paramStruct_2, 'B'))
    paramStruct_2.B(:) = 1;
end
if(isfield(paramStruct_2, 'H'))
    paramStruct_2.H(:) = H_var;
end
for jj = 1:obj.dim_J
    if(isfield(paramStruct_2.Groups(jj), 'H'))
        paramStruct_2.Groups(jj).H(:) = H_var;
    end
    paramStruct_2.Groups(jj).V(:) = 1;
    for ss = 1:obj.dim_S(jj)
         paramStruct_2.Groups(jj).T{ss}(:) = 1;
    end
end

TotalParameters = numel(obj.vectorizeParams(paramStruct, optStruct));
M = ones(TotalParameters,1);
M = obj.vectorizeParams(paramStruct_2, optStruct);

%% initialize space for samples
TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;

samples.log_p_accept  = nan(  TotalSamples,1);
samples.errors        = false(TotalSamples,1);
samples.accepted      = false(TotalSamples,1);
samples.e             = nan(2,TotalSamples);

if(~isinf(HMC_settings.MH_scale.sample_every) && ~isnan(HMC_settings.MH_scale.sample_every))
    warning(sprintf('Metropolis-Hasting rescaling step enabled: This is only valid for zero-mean Gaussian priors!\nMake sure that your priors are valid or disable this step by setting HMC_settings.MH_scale.sample_every = nan\n'));
    nMH = sum(obj.dim_R);
else
    nMH = 0;
end
samples.MH.accepted      = nan(nMH, TotalSamples);
samples.MH.log_p_accept  = nan(nMH, TotalSamples);

if(obj.gpuDoublePrecision)
    dataType = 'double';
else
    dataType = 'single';
end

samples.H       = nan(obj.dim_H,            TotalSamples, 'double');
samples.W       = nan(obj.dim_P,            TotalSamples, dataType);
samples.B       = nan(obj.dim_B, obj.dim_P, TotalSamples, dataType);
samples.log_post = nan(1, TotalSamples);
samples.log_like = nan(1, TotalSamples);

for jj = 1:obj.dim_J
    samples.Groups(jj).H = nan(obj.dim_H(jj),                TotalSamples, 'double');
    samples.Groups(jj).V = nan(obj.dim_P    , obj.dim_R(jj), TotalSamples, dataType);
    samples.Groups(jj).T = cell(obj.dim_S(jj), 1);
    for ss = 1:obj.dim_S(jj)
         samples.Groups(jj).T{ss} = nan(obj.dim_T(jj, ss), obj.dim_R(jj), TotalSamples, dataType);
    end
end

%save trial log likelihoods to harddrive in a piece-wise manner (otherwise, I'd fill up RAM)
samplesBlockSize      = min(HMC_settings.samplesBlockSize, TotalSamples);
samples_block.idx     = nan(samplesBlockSize, 1);
samples_block.trialLL = nan([obj.dim_trialLL(1), obj.dim_trialLL(2), samplesBlockSize], dataType);

if(exist(HMC_settings.samplesFile, 'file'))
    continue_opt = input(sprintf('Temporary storage file already found (%s)! Overwrite and continue? (y/n)\n ', HMC_settings.samplesFile), 's');
    if(startsWith(continue_opt, 'y', 'IgnoreCase', true))
        fprintf('Continuing... will overwrite file.\n');
    else
        error('Temporary file for storing trial log likelihood samples already exists!\nSpecify another filename or delete if not in use.\n\tfile: %s', HMC_settings.samplesFile);
    end
end
a = 1; % dummy variable
save(HMC_settings.samplesFile, 'a', '-nocompression', '-v7.3');
samples_file = matfile(HMC_settings.samplesFile, 'Writable',true);
samples_file.trialLL = zeros([obj.dim_trialLL(1), obj.dim_trialLL(2), samplesBlockSize], dataType);
obj.temp_storage_file = HMC_settings.samplesFile;

NB = ceil(TotalSamples / samplesBlockSize); % I'm trying here to pre-allocate space in the samples file without filling up RAM (there's probably a better way to do this, but I don't care)
for ii = 1:NB
    if(ii == NB)
        idx = ((NB-1)*samplesBlockSize + 1):TotalSamples;
    else
        idx = (1:samplesBlockSize) + (NB-1)*samplesBlockSize;
    end
    samples_file.trialLL(:, :, idx) = nan([obj.dim_trialLL(1), obj.dim_trialLL(2), numel(idx)], dataType);
end

%% initialize HMC state
HMC_state.stepSize.e       = HMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar   = HMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t     = 0;
HMC_state.stepSize.H_sum   = 0;
HMC_state.steps            = min(HMC_settings.stepSize.maxSteps, ceil(HMC_settings.stepSize.stepL / HMC_state.stepSize.e));

%% adds the initial point to the samples
resultStruct = obj.computeLogPosterior(paramStruct, optStruct);
samples.W(:,1)   = paramStruct.W(:);
samples.B(:,:,1) = paramStruct.B;
samples.H(:,1)   = paramStruct.H(:);

for jj = 1:obj.dim_J
    samples.Groups(jj).H(:,1) = paramStruct.Groups(jj).H;
    samples.Groups(jj).V(:,:,1) = paramStruct.Groups(jj).V;
    for ss = 1:obj.dim_S(jj)
        samples.Groups(jj).T{ss}(:,:,1) = paramStruct.Groups(jj).T{ss};
    end
end

samples_block.idx(1) = 1;
samples_block.trialLL(:, :, 1) = resultStruct.trialLL;

samples.log_post(1) = resultStruct.log_post;
samples.log_like(1) = resultStruct.log_likelihood;
samples.e(:,1)      = HMC_state.stepSize.e;

samples.log_p_accept(1) = log(1);

fprintf('Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e, max HMC steps = %d\n', TotalSamples, HMC_settings.nWarmup, samples.log_post(1), HMC_state.stepSize.e, HMC_settings.stepSize.maxSteps);

if(~isnan(figNum) && ~isinf(figNum))
    figure(figNum);
    clf;
    drawnow;
end

%%
vectorizedSamples             = zeros(TotalParameters, HMC_settings.M_est.samples(end));
start_idx = 2;

%% run sampler
for sample_idx = start_idx:TotalSamples
    %% set paramStruct to MAP estimate (should only be done early in warmup if at all)
    
    if(ismember(sample_idx, HMC_settings.fitMAP))
        fprintf("Attempting to accelerate mixing by finding MAP estimate given current hyperparameters...\n");
        paramStruct = obj.computeMAP(paramStruct, "optStruct", optStruct, "alternating_opt", false, "max_iters", 5, "max_quasinewton_steps", 250);
        %fprintf("done.\n");
    end
    
    %% run MH rescaling step
    % the decompositions have some directions of unidentifiability in the likelihood (V*T' = (V*R)*(T*R^-1)')
    % These optional steps do some fast MH proposals to quickly move around in that space, and does not require any likelihood computations.
    if(mod(sample_idx,HMC_settings.MH_scale.sample_every) == 0 && HMC_settings.MH_scale.N > 0)
        MH_accepted_c = nan(size(samples.MH.accepted,1),HMC_settings.MH_scale.N);
        MH_log_p      = nan(size(samples.MH.accepted,1),HMC_settings.MH_scale.N);
        for ii = 1:HMC_settings.MH_scale.N
            [paramStruct, MH_accepted_c(:,ii), MH_log_p(:,ii)] = obj.scalingMHStep(paramStruct, HMC_settings.MH_scale, optStruct);
        end
        samples.MH.accepted(:,sample_idx) = nanmean(MH_accepted_c,2);
        samples.MH.log_p_accept(:,sample_idx) = nanmean(MH_log_p,2);
    end
    
    %% get HMC sample
    % run HMC step
    w_init = obj.vectorizeParams(paramStruct, optStruct);
    [samples.accepted(sample_idx), samples.errors(sample_idx), w_new, samples.log_p_accept(sample_idx), resultStruct] = kgmlm.fittingTools.HMCstep_diag(w_init, M, nlpostFunction, HMC_state, sample_idx < HMC_settings.N_acceptAllImprovements);
    if(samples.accepted(sample_idx))
        paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct);
    end
    
    
    %% store samples
    samples.W(:,  sample_idx) = paramStruct.W(:);
    samples.B(:,:,sample_idx) = paramStruct.B(:,:);
    samples.H(:,  sample_idx) = paramStruct.H(:);
    
    for jj = 1:obj.dim_J
        samples.Groups(jj).H(:,sample_idx) = paramStruct.Groups(jj).H;
        samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
        
        for ss = 1:obj.dim_S(jj)
            samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
        end
    end
    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_likelihood;
    
    if(sample_idx <= HMC_settings.M_est.samples(end))
        vectorizedSamples(:, sample_idx) = w_new; 
    end
    
    %temp storage of trialLL
    idx_c = mod(sample_idx-1, samplesBlockSize) + 1;
    samples_block.idx(       idx_c) = sample_idx;
    samples_block.trialLL(:, :, idx_c) = resultStruct.trialLL;
    if(mod(sample_idx, samplesBlockSize) == 0 || sample_idx == TotalSamples)
        %save to file
        xx = ~isnan(samples_block.idx);
        samples_file.trialLL(:,:,samples_block.idx(xx))  = samples_block.trialLL;
    end
    
    %% print any updates
    if(sample_idx <= 50 || (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == TotalSamples || (HMC_settings.verbose && mod(sample_idx,20) == 0))
        if(sample_idx == TotalSamples)
            ww = (HMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        mean_MH_accepted = nanmean(samples.MH.accepted(:,ww),'all');
        
        fprintf('HMC step %d / %d (accept per. = %.1f in last %d steps, curr log post = %e, (log like = %e)\n', sample_idx, TotalSamples, mean(samples.accepted(ww))*100, numel(ww), samples.log_post(sample_idx), samples.log_like(sample_idx));
        fprintf('\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d, mean MH accepted = %.3f\n', HMC_state.stepSize.e, HMC_state.steps, nansum(samples.errors),  mean_MH_accepted);
        clear ww;
        
        if(~isnan(figNum) && ~isinf(figNum))
            kgmlm.utils.sfigure(figNum);
            obj.plotSamples(samples, paramStruct, sample_idx);
            drawnow;
        end
    end
    
    %% adjust step size
    HMC_state = kgmlm.fittingTools.adjustHMCstepSize(sample_idx, HMC_state, HMC_settings.stepSize, samples.log_p_accept(sample_idx));
    samples.e(:,sample_idx) = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];
    
    %% updates the covariance matrix of the hyperparameters
    if(ismember(sample_idx, HMC_settings.M_est.samples ) ) 
        start_idx = HMC_settings.M_est.first_sample(HMC_settings.M_est.samples == sample_idx);
        ww = start_idx:sample_idx;
        %diagonal only
        M = (1./var(vectorizedSamples(:,ww),[],2));
    end
end

%% finish sampler
ss_idx = (HMC_settings.nWarmup+1):sample_idx;

fprintf('computing WAIC and PSIS-LOO... ');
T_n = zeros(obj.dim_trialLL(1), obj.dim_trialLL(2));
V_n = zeros(obj.dim_trialLL(1), obj.dim_trialLL(2));

summary.PSISLOOS   = zeros(obj.dim_trialLL(1), obj.dim_trialLL(2));
summary.PSISLOO_PK = zeros(obj.dim_trialLL(1), obj.dim_trialLL(2));

% ll = samples.trialLL(:,ss_idx);
for ii = 1:obj.dim_trialLL(1)
    for jj = 1:obj.dim_trialLL(2)
        ll_c = double(samples_file.trialLL(ii,jj,ss_idx));
        T_n(ii,jj) = -kgmlm.utils.logMeanExp(ll_c,2);
        V_n(ii,jj) = mean(ll_c.^2,2) - mean(ll_c,2).^2;

        [~,summary.PSISLOOS(ii,jj),summary.PSISLOO_PK(ii,jj)] = kgmlm.PSISLOO.psisloo(ll_c(:));
    end
end
summary.WAICS = T_n + V_n;
summary.WAIC  = mean(summary.WAICS,'all');
summary.PSISLOO = sum(summary.PSISLOOS,'all');

badSamples   = sum(summary.PSISLOO_PK >= 0.7,'all');
if(badSamples > 0)
    fprintf('Warning: PSISLOO PK large (>0.7) for %d / %d observations! \n', badSamples, numel(summary.PSISLOO_PK ));
else
    fprintf('PSISLOO diagnostics passed (all PK < 0.7). \n');
end

ss_all = (HMC_settings.nWarmup+1):TotalSamples;
summary.earlyRejects = sum(samples.errors(ss_all));
summary.earlyReject_prc = mean(samples.errors(ss_all));
summary.HMC_state            = HMC_state;
summary.acceptRate   = mean(samples.accepted(ss_all));

fprintf('done.\n');   

delete(samples_file.Properties.Source); % delete the temporary storage file for trial log likelihoods
obj.temp_storage_file = [];
end





