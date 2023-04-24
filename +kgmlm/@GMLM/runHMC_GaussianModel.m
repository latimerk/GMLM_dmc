% Runs the HMC algorithm for the GMLM
% Computes PSISLOO and WAIC on the samples.
%
% NOTE: uses HMC_settings.trialLLfile to store some partial results. This file can get big, but is deleted after running is complete.
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

addRequired(p, "params_init",  @(aa)(isstruct(aa) | isempty(aa)));
addRequired(p, "settings", @isstruct);
addParameter(p, "figure" ,    nan, @isnumeric);
addParameter(p, "optStruct" ,   [], @(aa) isempty(aa) | obj.verifyComputeOptionsStruct(aa));
addParameter(p, "sampleHyperparameters", true, @islogical);
addParameter(p, "trial_weights"   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));
addParameter(p, "saveUnscaled" ,    false, @islogical);
addParameter(p, "saveSinglePrecision" ,    false, @islogical);

parse(p, params_init, HMC_settings, varargin{:});
% then set/get all the inputs out of this structure
optStruct          = p.Results.optStruct;
paramStruct        = p.Results.params_init;
HMC_settings       = p.Results.settings;
trial_weights      = p.Results.trial_weights;
sampleHyperparameters = p.Results.sampleHyperparameters;
figNum = p.Results.figure;
saveUnscaled      = p.Results.saveUnscaled;
saveSinglePrecision      = p.Results.saveSinglePrecision;
    
if(~obj.isOnGPU())
    error("Must load gmlm onto GPU before running HMC!");
end

J = obj.dim_J;
S = obj.dim_S;

%%
dim_N_trial = obj.dim_N();
P = obj.dim_P;
numTrials = obj.dim_M;
if(~obj.isSimultaneousPopulation)
    NeuronNums = [obj.trials(:).neuron];
    dim_Ns = zeros(P,1);
    for pp = 1:P
        dim_Ns(pp) = sum(dim_N_trial(NeuronNums == pp));
    end
else
    dim_Ns = sum(dim_N_trial);
end
sigmaPrior.alpha = 1;
sigmaPrior.beta  = 1;

%% sets up the hmc momentum cov matrices
optStruct = obj.getComputeOptionsStruct(true, "trial_weights", true, "includeHyperparameters", sampleHyperparameters);
paramStruct.Sigma2 = ones(obj.dim_P,1);

optStruct_empty = obj.getComputeOptionsStruct(false, "trial_weights", ~isempty(trial_weights), "includeHyperparameters", sampleHyperparameters);
if(~isempty(trial_weights))
    optStruct_empty.trial_weights(:) = trial_weights;
    optStruct.trial_weights(:) = trial_weights;
end


H_var = (1)^2; %initial momentum term for hyperparams
% H_var = 0.1;

M_scale = 1;
W_scale = M_scale;
B_scale = M_scale;
T_scale = M_scale;
V_scale = M_scale;


paramStruct_2 = paramStruct;
paramStruct_2.W(:) = W_scale;
if(isfield(paramStruct_2, "B"))
    paramStruct_2.B(:) = B_scale;
end
if(isfield(paramStruct_2, "H"))
    paramStruct_2.H(:) = H_var;
end
for jj = 1:J
    if(isfield(paramStruct_2.Groups(jj), "H"))
        paramStruct_2.Groups(jj).H(:) = H_var;
    end
    paramStruct_2.Groups(jj).V(:) = V_scale;
    for ss = 1:S(jj)
         paramStruct_2.Groups(jj).T{ss}(:) = T_scale;
    end
end


TotalParameters = numel(obj.vectorizeParams(paramStruct, optStruct));
M = obj.vectorizeParams(paramStruct_2, optStruct);
if(isfield(HMC_settings, "M_init"))
    M = obj.vectorizeParams(HMC_settings.M_init, optStruct);
end

%% initialize space for samples
TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;

samples.log_p_accept  = nan(  TotalSamples,1);
samples.errors        = false(TotalSamples,1);
samples.accepted      = false(TotalSamples,1);
samples.e             = nan(2,TotalSamples);

samples.e_scale    = ones(TotalSamples, 1);
for ii = 1:size(HMC_settings.stepSize.scaleRanges,1)
    rr = HMC_settings.stepSize.scaleRanges(ii,1):HMC_settings.stepSize.scaleRanges(ii,2);
    if(~isempty(rr))
        samples.e_scale(rr) = mnrnd(1, HMC_settings.stepSize.P_scales   , numel(rr)) * HMC_settings.stepSize.scales(:);
    end
end

scaled_WB = isfield(obj.GMLMstructure, "scaleParams") && ~isempty(obj.scaleParams);

scaled_VT = false(J,1);
for jj = 1:J
    if(isfield(obj.GMLMstructure.Groups(jj), "scaleParams") && ~isempty(obj.GMLMstructure.Groups(jj).scaleParams))
        scaled_VT(jj) = true;
    end
end


if(obj.gpuDoublePrecision)
    dataType = "double";
else
    dataType = "single";
end

if(~saveSinglePrecision && obj.gpuDoublePrecision)
    dataType_samples = "double";
else
    dataType_samples = "single";
end
if(saveSinglePrecision)
    dataType_samples_hyper = "single";
else
    dataType_samples_hyper = "double";
end


samples.H       = nan(obj.dim_H,            TotalSamples, dataType_samples_hyper);
samples.H_gibbs = nan(obj.dim_H_gibbs,      TotalSamples, dataType_samples_hyper);
samples.W       = nan(obj.dim_P,            TotalSamples, dataType_samples);
samples.Sigma2  = nan(obj.dim_P,            TotalSamples, dataType_samples);
samples.B       = nan(obj.dim_B, obj.dim_P, TotalSamples, dataType_samples);
samples.log_post = nan(1, TotalSamples);
samples.log_like = nan(1, TotalSamples);

if(scaled_WB && saveUnscaled)
    samples.W_scaled = samples.W;
    samples.B_scaled = samples.B;
else
    samples.W_scaled = [];
    samples.B_scaled = [];
end

for jj = 1:J
    metrics.Groups(jj).N = nan(obj.dim_R(jj), TotalSamples, dataType_samples);
    samples.Groups(jj).H = nan(obj.dim_H(jj),                TotalSamples, dataType_samples_hyper);
    samples.Groups(jj).H_gibbs = nan(obj.dim_H_gibbs(jj),    TotalSamples, dataType_samples_hyper);
    samples.Groups(jj).V = nan(obj.dim_P    , obj.dim_R(jj), TotalSamples, dataType_samples);

    if(scaled_VT(jj) && saveUnscaled)
        samples.Groups(jj).V_scaled = samples.Groups(jj).V;
    else
        samples.Groups(jj).V_scaled = [];
    end
    samples.Groups(jj).T = cell(S(jj), 1);
    samples.Groups(jj).T_scaled = cell(S(jj), 1);
    for ss = 1:S(jj)
         samples.Groups(jj).T{ss} = nan(obj.dim_T(jj, ss), obj.dim_R(jj), TotalSamples, dataType_samples);
        if(scaled_VT(jj) && saveUnscaled)
            samples.Groups(jj).T_scaled{ss} = samples.Groups(jj).T{ss};
        else
            samples.Groups(jj).T_scaled{ss} = [];
        end
    end
end

%save trial log likelihoods to harddrive in a piece-wise manner (otherwise, I'd fill up RAM)
DT = [obj.dim_trialLL(1) obj.dim_trialLL(2)];
samplesBlockSize      = min(HMC_settings.samplesBlockSize, TotalSamples);
samples_block.idx     = nan(samplesBlockSize, 1);
samples_block.trialLL = nan([samplesBlockSize DT(1) DT(2)], dataType);

if(exist(HMC_settings.trialLLfile, "file"))
    if(isfield(HMC_settings, "delete_temp_file"))
        continue_opt = HMC_settings.delete_temp_file;
    else
        continue_opt = input(sprintf("Temporary storage file already found (%s)! Overwrite and continue? (y/n)\n ", HMC_settings.trialLLfile), "s");
        continue_opt = startsWith(continue_opt, "y", "IgnoreCase", true);
    end
    if(continue_opt)
        fprintf("Deleting temporary storage file and continuing...\n");
    else
        error("Temporary file for storing trial log likelihood samples already exists!\nSpecify another filename or delete if not in use.\n\tfile: %s", HMC_settings.trialLLfile);
    end
end

%makes space for trialLL without ever making the full matrix in RAM: this is a cludge around the compression that mafile puts in automatically
Z = zeros(TotalSamples,1,dataType);
fprintf("Preallocating HD space to store LLs for each trial (~%.3f gb)...\n", whos("Z").bytes * DT(1) * DT(2) / 1e9);
obj.temp_storage_file = HMC_settings.trialLLfile;
fileID = fopen(HMC_settings.trialLLfile, "w");
for ii = 1:DT(1)
    for jj = 1:DT(2)
        fwrite(fileID, Z, dataType);
    end
end
clear Z;
fclose(fileID);

trialLL_file = memmapfile(HMC_settings.trialLLfile,...
               "Format",{dataType,[TotalSamples DT(1) DT(2)],"trialLL"}, ...
               "Writable", true);
fprintf("Done.\n")

%% initialize HMC state
HMC_state.stepSize.e       = HMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar   = HMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t     = 0;
HMC_state.stepSize.H_sum   = 0;
HMC_state.steps            = min(HMC_settings.stepSize.maxSteps, ceil(HMC_settings.stepSize.stepL / HMC_state.stepSize.e));

%% adds the initial point to the samples
resultStruct_empty = obj.getEmptyResultsStruct(optStruct_empty);
resultStruct = obj.computeLogPosterior(paramStruct, optStruct);

resultStruct.log_post = resultStruct.log_post + sum(sigmaPrior.alpha*log(sigmaPrior.beta) - gammaln(sigmaPrior.alpha) - (sigmaPrior.alpha - 1) * log(paramStruct.Sigma2) - sigmaPrior.beta * paramStruct.Sigma2);
nc = sum(dim_Ns.*(log(paramStruct.Sigma2) - log(2*pi))./2);
resultStruct.log_post = resultStruct.log_post + nc;
resultStruct.log_likelihood = resultStruct.log_likelihood + nc;

sample_idx = 1;
if(scaled_WB)

    params_0 = obj.GMLMstructure.scaleParams(paramStruct);

    samples.W(:,  sample_idx) = params_0.W(:);
    samples.B(:,:,sample_idx) = params_0.B(:,:);

    if(saveUnscaled)
        samples.W_scaled(:,  sample_idx) = paramStruct.W(:);
        samples.B_scaled(:,:,sample_idx) = paramStruct.B(:,:);
    end
else
    samples.W(:,  sample_idx) = paramStruct.W(:);
    samples.B(:,:,sample_idx) = paramStruct.B(:,:);
end
samples.Sigma2(:,  sample_idx) = paramStruct.Sigma2(:);
samples.H(:,1)   = paramStruct.H(:);
samples.H_gibbs(:,1)   = paramStruct.H_gibbs(:);

for jj = 1:J
    
    samples.Groups(jj).H(:,1) = paramStruct.Groups(jj).H;
    samples.Groups(jj).H_gibbs(:,1) = paramStruct.Groups(jj).H_gibbs;

    if(scaled_VT(jj))

        params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));

        samples.Groups(jj).V(:,:,sample_idx) = params_0.V;
        metrics.Groups(jj).N(:, sample_idx) = sum(params_0.V.^2,1);
        if(saveUnscaled)
            samples.Groups(jj).V_scaled(:,:,sample_idx) = paramStruct.Groups(jj).V;
        end
        for ss = 1:S(jj)
            samples.Groups(jj).T{ss}(:,:,sample_idx) = params_0.T{ss};
            metrics.Groups(jj).N(:, sample_idx) = metrics.Groups(jj).N(:, sample_idx).*sum(params_0.T{ss}.^2,1)';
            if(saveUnscaled)
                samples.Groups(jj).T_scaled{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            end
        end
    else
        samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
        metrics.Groups(jj).N(:, sample_idx) = sum(paramStruct.Groups(jj).V.^2,1);
        for ss = 1:S(jj)
            samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            metrics.Groups(jj).N(:, sample_idx) = metrics.Groups(jj).N(:, sample_idx).*sum(paramStruct.Groups(jj).T{ss}.^2,1)';
        end
    end
    metrics.Groups(jj).N(:, 1) = sqrt(metrics.Groups(jj).N(:, 1));
end

samples_block.idx(1) = 1;
samples_block.trialLL(1, :, :) = resultStruct.trialLL;

samples.log_post(1) = resultStruct.log_post;
samples.log_like(1) = resultStruct.log_likelihood;
samples.e(:,1)      = HMC_state.stepSize.e;
samples.log_p_accept(1) = log(1);

fprintf("Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e,\n\tmax HMC steps = %d\n", TotalSamples, HMC_settings.nWarmup, samples.log_post(1), HMC_state.stepSize.e, HMC_settings.stepSize.maxSteps);
for ii = 1:size(HMC_settings.stepSize.schedule ,1)
    fprintf("\tStep size estimation samples %d - %d, target accept rate = %.2f\n", HMC_settings.stepSize.schedule(ii,1), HMC_settings.stepSize.schedule(ii,2), HMC_settings.stepSize.delta(min(ii, numel(HMC_settings.stepSize.delta))));
end

if(~isnan(figNum) && ~isinf(figNum))
    figure(figNum);
    clf;
    drawnow;
end

%%
vectorizedSamples = nan(TotalParameters, HMC_settings.M_est.samples(end), dataType);
if(sample_idx <= HMC_settings.M_est.samples(end))
    vectorizedSamples(:, sample_idx) = obj.vectorizeParams(paramStruct, optStruct);
end
start_idx = 2;



%% run sampler
for sample_idx = start_idx:TotalSamples
    %% set paramStruct to MAP estimate (should only be done early in warmup if at all)
    
    if(isfield(HMC_settings, "fitMAP") && ismember(sample_idx, HMC_settings.fitMAP))
        fprintf("Attempting to accelerate mixing by finding MAP estimate given current hyperparameters...\n");
        fprintf("   Alternating steps...\n");
        paramStruct = obj.computeMAP(paramStruct, "optStruct", optStruct, "alternating_opt", true, "max_iters", 5, "max_quasinewton_steps", 100);
        fprintf("   All params at once...\n");
        paramStruct = obj.computeMAP(paramStruct, "optStruct", optStruct, "alternating_opt", false, "max_iters", 5, "max_quasinewton_steps", 1000);
        %fprintf("done.\n");
    end
    
    
    %% run any Gibbs steps - can be defined for the whole GMLM or tensor groups
    if(~isempty(obj.GMLMstructure.gibbs_step) && optStruct.H_gibbs)
        paramStruct = obj.GMLMstructure.gibbs_step.sample_func(obj, paramStruct, optStruct, sample_idx, optStruct_empty, resultStruct_empty);
    end
    for jj = 1:J
        if(~isempty(obj.GMLMstructure.Groups(jj).gibbs_step) && optStruct.Groups(jj).H_gibbs && isfield(obj.GMLMstructure.Groups(jj).gibbs_step, "sample_func") && ~isempty(obj.GMLMstructure.Groups(jj).gibbs_step.sample_func))
            paramStruct = obj.GMLMstructure.Groups(jj).gibbs_step.sample_func(obj, paramStruct, optStruct, sample_idx, jj, optStruct_empty, resultStruct_empty);
        end
    end


    %% sample sigmas
    params_0 = obj.computeLogPosterior(paramStruct, optStruct_empty);
    if(~obj.isSimultaneousPopulation)
        for pp = 1:P
            sigmaPost.beta = sigmaPrior.beta - sum(params_0.trialLL(NeuronNums == pp));
            sigmaPost.alpha = sigmaPrior.alpha +   dim_Ns(pp)/2;
            paramStruct.Sigma2(pp) = gamrnd(sigmaPost.alpha, 1./sigmaPost.beta); % PRECISION
        end
    else
        sigmaPost.beta = sigmaPrior.beta -  sum(params_0.trialLL,1);
        sigmaPost.alpha = sigmaPrior.alpha +   dim_Ns/2;
        paramStruct.Sigma2(:) = gamrnd(sigmaPost.alpha, 1./sigmaPost.beta); % PRECISION
    end
    
    %% get HMC sample
    % run HMC step
    if(~obj.isSimultaneousPopulation)
        for pp = 1:P
            optStruct.trial_weights(NeuronNums == pp) = paramStruct.Sigma2(pp);
        end
    else
        optStruct.trial_weights(:,:) = repmat(paramStruct.Sigma2', [numTrials 1]);
    end
    w_init = obj.vectorizeParams(paramStruct, optStruct);
    nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct, resultStruct);
%     try
        HMC_state.e_scale = samples.e_scale(sample_idx);
        [samples.accepted(sample_idx), samples.errors(sample_idx), w_new, samples.log_p_accept(sample_idx), resultStruct] = kgmlm.fittingTools.HMCstep_diag(w_init, HMC_settings.M_const * M, nlpostFunction, HMC_state);
        if(samples.accepted(sample_idx))
            paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct);
        end
%     catch
%         error("HMC step failed");
%     end
    % adjust step size: during warmup
    HMC_state = kgmlm.fittingTools.adjustHMCstepSize(sample_idx, HMC_state, HMC_settings.stepSize, samples.log_p_accept(sample_idx));
    samples.e(:,sample_idx) = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];

    %% add Sigma2 prior and Gaussian normalizing constant to output log post
    resultStruct.log_post = resultStruct.log_post + sum(sigmaPrior.alpha*log(sigmaPrior.beta) - gammaln(sigmaPrior.alpha) - (sigmaPrior.alpha - 1) * log(paramStruct.Sigma2) - sigmaPrior.beta * paramStruct.Sigma2);
    nc = sum(dim_Ns.*(log(paramStruct.Sigma2) - log(2*pi))./2);
    resultStruct.log_post = resultStruct.log_post + nc;
    resultStruct.log_likelihood = resultStruct.log_likelihood + nc;



    
    
    %% store samples
    if(scaled_WB)
        params_0 = obj.GMLMstructure.scaleParams(paramStruct);

        samples.W(:,  sample_idx) = params_0.W(:);
        samples.B(:,:,sample_idx) = params_0.B(:,:);

        if(saveUnscaled)
            samples.W_scaled(:,  sample_idx) = paramStruct.W(:);
            samples.B_scaled(:,:,sample_idx) = paramStruct.B(:,:);
        end
    else
        samples.W(:,  sample_idx) = paramStruct.W(:);
        samples.B(:,:,sample_idx) = paramStruct.B(:,:);
    end
    samples.Sigma2(:,  sample_idx) = paramStruct.Sigma2(:);

    samples.H(:,  sample_idx) = paramStruct.H(:);
    samples.H_gibbs(:,  sample_idx) = paramStruct.H_gibbs(:);
    
    for jj = 1:J
        samples.Groups(jj).H(:,sample_idx) = paramStruct.Groups(jj).H;
        samples.Groups(jj).H_gibbs(:,sample_idx) = paramStruct.Groups(jj).H_gibbs;

        if(scaled_VT(jj))
            params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));

            samples.Groups(jj).V(:,:,sample_idx) = params_0.V;
            metrics.Groups(jj).N(:, sample_idx) = sum(params_0.V.^2,1);

            if(saveUnscaled)
                samples.Groups(jj).V_scaled(:,:,sample_idx) = paramStruct.Groups(jj).V;
            end
            for ss = 1:S(jj)
                samples.Groups(jj).T{ss}(:,:,sample_idx) = params_0.T{ss};
                if(saveUnscaled)
                    samples.Groups(jj).T_scaled{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
                end
                metrics.Groups(jj).N(:, sample_idx) = metrics.Groups(jj).N(:, sample_idx).*sum(params_0.T{ss}.^2,1)';
            end
        else
            samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
            metrics.Groups(jj).N(:, sample_idx) = sum(paramStruct.Groups(jj).V.^2,1);
            for ss = 1:S(jj)
                samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
                metrics.Groups(jj).N(:, sample_idx) = metrics.Groups(jj).N(:, sample_idx).*sum(paramStruct.Groups(jj).T{ss}.^2,1)';
            end
        end

        metrics.Groups(jj).N(:, sample_idx) = sqrt(metrics.Groups(jj).N(:, sample_idx));
    end
    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_likelihood;
    
    if(sample_idx <= HMC_settings.M_est.samples(end))
        vectorizedSamples(:, sample_idx) = w_new; 
    end
     
    %temp storage of trialLL
    idx_c = mod(sample_idx-1, samplesBlockSize) + 1;
    samples_block.idx(       idx_c) = sample_idx;
    samples_block.trialLL(idx_c, :, :) = resultStruct.trialLL;
    if(mod(sample_idx, samplesBlockSize) == 0 || sample_idx == TotalSamples)
        %save to file
        xx = ~isnan(samples_block.idx);
        trialLL_file.Data.trialLL(samples_block.idx(xx),:,:)  = samples_block.trialLL;
    end
    
    
    %% print any updates
    if(sample_idx <= 50 || (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == TotalSamples || (HMC_settings.verbose && mod(sample_idx,20) == 0))
        if(sample_idx == TotalSamples)
            ww = (HMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        accept_rate = mean(samples.accepted(ww))*100;
        fprintf("HMC step %d / %d (accept per. = %.1f in last %d steps, curr log post = %e, (log like = %e)\n", sample_idx, TotalSamples, accept_rate, numel(ww), samples.log_post(sample_idx), samples.log_like(sample_idx));
        fprintf("\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d\n", HMC_state.stepSize.e, HMC_state.steps, sum(samples.errors, "omitnan"));
        
        if(isfield(HMC_settings, "sample_M") && HMC_settings.sample_M)
            accept_rate_M = mean(samples.accepted_sM(ww))*100;
            fprintf("\t\tsample M: current step size = %e, HMC steps = %d, num HMC early rejects = %d, recent accept per. = %.1f\n", HMC_state_sM.stepSize.e , HMC_state_sM.steps,  sum(samples.errors_sM, "omitnan"), accept_rate_M);
        end
        if(isfield(HMC_settings, "sample_H") && HMC_settings.sample_H)
            accept_rate_H = mean(samples.accepted_sH(ww))*100;
            fprintf("\t\tsample H: current step size = %e, HMC steps = %d, num HMC early rejects = %d, recent accept per. = %.1f\n", HMC_state_sH.stepSize.e, HMC_state_sH.steps, sum(samples.errors_sH, "omitnan"), accept_rate_H);
        end
        clear ww;
        
        if(~isnan(figNum) && ~isinf(figNum))
            kgmlm.utils.sfigure(figNum);
            obj.plotSamples(samples, metrics, paramStruct, sample_idx);
            drawnow;
        end
    end
    
    
    %% updates the covariance matrix of the hyperparameters
    if(ismember(sample_idx, HMC_settings.M_est.samples ) ) 
        start_idx = HMC_settings.M_est.first_sample(HMC_settings.M_est.samples == sample_idx);
        ww = start_idx:sample_idx;
        %diagonal only
        M = (1./var(vectorizedSamples(:,ww),[],2));

        if(all(sample_idx >= HMC_settings.M_est.samples))
            clear vectorizedSamples;
        end
    end
end

%% finish sampler
ss_idx = (HMC_settings.nWarmup+1):sample_idx;

fprintf("computing WAIC and PSIS-LOO... \n");
V_n = zeros(obj.dim_trialLL(1), obj.dim_trialLL(2));
T_n = zeros(size(V_n));

summary.PSISLOOS   = zeros(size(V_n));
summary.PSISLOO_PK = zeros(size(V_n));

blk_size = min(4, size(V_n,1));
NB = ceil(size(V_n,2)/blk_size);

% ll = samples.trialLL(:,ss_idx);
for ii = 1:size(V_n,2)
    if(size(V_n,2) > 1)
        fprintf("\tneuron block  %d / %d\n", ii, size(V_n,2));
    end

    for kk = 1:NB
        if(kk == 1 || mod(kk,20) == 0)
            fprintf("\t\ttrial block  %d / %d\n", kk, NB);
        end
        if(kk < NB)
            jj_idx = (kk-1)*blk_size + (1:blk_size);
        else
            jj_idx = ((kk-1)*blk_size + 1):size(V_n,1);
        end

        JJ = numel(jj_idx);
        PL_c = zeros(JJ,1);
        PLK_c = zeros(JJ,1);
        ll_c = squeeze(double(trialLL_file.Data.trialLL(ss_idx,jj_idx,ii))); 

        if(blk_size > 1)
            parfor (jj = 1:JJ, blk_size)
                [~,PL_c(jj),PLK_c(jj)] = kgmlm.PSISLOO.psisloo(ll_c(:,jj));
            end
        else
            jj = 1;
            [~,PL_c(jj),PLK_c(jj)] = kgmlm.PSISLOO.psisloo(ll_c(:,jj));
        end
        T_n(jj_idx,ii) = (-kgmlm.utils.logMeanExp(ll_c,1))';
        V_n(jj_idx,ii) = (mean(ll_c.^2,1) - mean(ll_c,1).^2)';
        summary.PSISLOOS(jj_idx,ii) = PL_c;
        summary.PSISLOO_PK(jj_idx,ii) = PLK_c;
    end
end
summary.WAICS = T_n + V_n;
summary.WAIC  = mean(summary.WAICS,"all");
summary.PSISLOO = sum(summary.PSISLOOS,"all");

badSamples   = sum(summary.PSISLOO_PK >= 0.7,"all");
if(badSamples > 0)
    fprintf("\tWarning: PSISLOO PK large (>0.7) for %d / %d observations! \n", badSamples, numel(summary.PSISLOO_PK ));
else
    fprintf("\tPSISLOO diagnostics passed (all PK < 0.7). \n");
end

ss_all = (HMC_settings.nWarmup+1):TotalSamples;
summary.earlyRejects = sum(samples.errors(ss_all));
summary.earlyReject_prc = mean(samples.errors(ss_all));
summary.HMC_state            = HMC_state;
summary.acceptRate   = mean(samples.accepted(ss_all));

fprintf("done.\n");   

delete(trialLL_file.Filename); % delete the temporary storage file for trial log likelihoods
obj.temp_storage_file = [];
end





