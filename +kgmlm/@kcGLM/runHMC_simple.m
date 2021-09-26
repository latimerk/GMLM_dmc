% Runs the HMC algorithm for the GLM
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
%       trial_weights         (array size glm.dim_M; default = []) : weighting of the trials (empty for all trial weights 1)

function [samples, summary, HMC_settings, paramStruct, M] = runHMC_simple(obj, params_init, HMC_settings, varargin)
p = inputParser;
p.CaseSensitive = false;

addRequired(p, 'params_init',  @(aa)(isstruct(aa) | isempty(aa)));
addRequired(p, 'settings', @isstruct);
addParameter(p, 'figure' ,    nan, @isnumeric);
addParameter(p, 'sampleHyperparameters', true, @islogical);
addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

parse(p, params_init, HMC_settings, varargin{:});
% then set/get all the inputs out of this structure
paramStruct        = p.Results.params_init;
HMC_settings       = p.Results.settings;
trial_weights      = p.Results.trial_weights;
sampleHyperparameters = p.Results.sampleHyperparameters;
figNum = p.Results.figure;
    
if(~obj.isOnGPU())
    error('Must load glm onto GPU before running HMC!');
end

%% sets up the hmc momentum cov matrices
optStruct = obj.getComputeOptionsStruct(true, 'trial_weights', ~isempty(trial_weights), 'includeHyperparameters', sampleHyperparameters);
optStruct.d2K = false;
optStruct_empty = obj.getComputeOptionsStruct(false, 'trial_weights', ~isempty(trial_weights), 'includeHyperparameters', sampleHyperparameters);
if(~isempty(trial_weights))
    optStruct_empty.trial_weights(:) = trial_weights;
    optStruct.trial_weights(:) = trial_weights;
end

nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct);

if(sampleHyperparameters)
    TotalParameters = obj.dim_K(-1) + obj.dim_H();
else
    TotalParameters = obj.dim_K(-1);
end
M = ones(TotalParameters,1);

%% initialize space for samples
TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;

samples.log_p_accept  = nan(  TotalSamples,1);
samples.errors        = false(TotalSamples,1);
samples.accepted      = false(TotalSamples,1);
samples.e             = nan(2,TotalSamples);

if(obj.gpuDoublePrecision)
    dataType = 'double';
else
    dataType = 'single';
end

samples.H       = nan(obj.dim_H,            TotalSamples, 'double');
samples.log_post = nan(1, TotalSamples);
samples.log_like = nan(1, TotalSamples);

samples.Ks = cell(obj.dim_J, 1);
for jj = 1:obj.dim_J
    samples.Ks{jj} = nan(obj.dim_K(jj),                TotalSamples, 'double');
end

trialLL = nan(obj.dim_M, TotalSamples, dataType);


%% initialize HMC state
HMC_state.stepSize.e       = HMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar   = HMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t     = 0;
HMC_state.stepSize.H_sum   = 0;
HMC_state.steps            = min(HMC_settings.stepSize.maxSteps, ceil(HMC_settings.stepSize.stepL / HMC_state.stepSize.e));

%% adds the initial point to the samples
resultStruct = obj.computeLogPosterior(paramStruct, optStruct);

for jj = 1:obj.dim_J
    samples.Ks{jj}(:,1) = paramStruct.Ks{jj};
end

trialLL(:, 1) = resultStruct.trialLL;

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
    %% get HMC sample
    % run HMC step
    w_init = obj.vectorizeParams(paramStruct, optStruct);
    [samples.accepted(sample_idx), samples.errors(sample_idx), w_new, samples.log_p_accept(sample_idx), resultStruct] = kgmlm.fittingTools.HMCstep_diag(w_init, M, nlpostFunction, HMC_state);
    if(samples.accepted(sample_idx))
        paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct);
    end
    
    %% store samples
    samples.H(:,  sample_idx) = paramStruct.H(:);
    
    for jj = 1:obj.dim_J
        samples.Ks{jj}(:,sample_idx) = paramStruct.Ks{jj};
    end
    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_likelihood;
    
    if(sample_idx <= HMC_settings.M_est.samples(end))
        vectorizedSamples(:, sample_idx) = w_new; 
    end
    
    %temp storage of trialLL
    trialLL(:, sample_idx) = resultStruct.trialLL;
    
    %% print any updates
    if(sample_idx <= 50 || (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == TotalSamples || (HMC_settings.verbose && mod(sample_idx,20) == 0))
        if(sample_idx == TotalSamples)
            ww = (HMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        
        fprintf('HMC step %d / %d (accept per. = %.1f in last %d steps, curr log post = %e, (log like = %e)\n', sample_idx, TotalSamples, mean(samples.accepted(ww))*100, numel(ww), samples.log_post(sample_idx), samples.log_like(sample_idx));
        fprintf('\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d\n', HMC_state.stepSize.e, HMC_state.steps, nansum(samples.errors));
        clear ww;
        
        if(~isnan(figNum) && ~isinf(figNum))
            kgmlm.utils.sfigure(figNum);
            obj.plotSamples(samples, sample_idx);
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
dim_M = double(obj.dim_M);
ss_idx = (HMC_settings.nWarmup+1):sample_idx;

fprintf('computing WAIC and PSIS-LOO... ');
T_n = zeros(dim_M,1);
V_n = zeros(dim_M,1);

summary.PSISLOOS   = zeros(dim_M,1);
summary.PSISLOO_PK = zeros(dim_M,1);

% ll = samples.trialLL(:,ss_idx);
for ii = 1:dim_M
    ll_c = double(trialLL(ii,ss_idx));
    T_n(ii) = -kgmlm.utils.logMeanExp(ll_c,2);
    V_n(ii) = mean(ll_c.^2,2) - mean(ll_c,2).^2;
    
    [~,summary.PSISLOOS(ii),summary.PSISLOO_PK(ii)] = kgmlm.PSISLOO.psisloo(ll_c(:));
end
summary.WAICS = T_n + V_n;
summary.WAIC  = mean(summary.WAICS,1);
summary.PSISLOO = sum(summary.PSISLOOS);

badSamples   = sum(summary.PSISLOO_PK >= 0.7);
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

end





