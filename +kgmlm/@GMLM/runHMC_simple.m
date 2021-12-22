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

addRequired(p, "params_init",  @(aa)(isstruct(aa) | isempty(aa)));
addRequired(p, "settings", @isstruct);
addParameter(p, "figure" ,    nan, @isnumeric);
addParameter(p, "optStruct" ,   [], @(aa) isempty(aa) | obj.verifyComputeOptionsStruct(aa));
addParameter(p, "sampleHyperparameters", true, @islogical);
addParameter(p, "trial_weights"   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

parse(p, params_init, HMC_settings, varargin{:});
% then set/get all the inputs out of this structure
optStruct          = p.Results.optStruct;
paramStruct        = p.Results.params_init;
HMC_settings       = p.Results.settings;
trial_weights      = p.Results.trial_weights;
sampleHyperparameters = p.Results.sampleHyperparameters;
figNum = p.Results.figure;
    
if(~obj.isOnGPU())
    error("Must load gmlm onto GPU before running HMC!");
end

%% sets up the hmc momentum cov matrices
if(isempty(optStruct))
    optStruct = obj.getComputeOptionsStruct(true, "trial_weights", ~isempty(trial_weights), "includeHyperparameters", sampleHyperparameters);
end
optStruct_empty = obj.getComputeOptionsStruct(false, "trial_weights", ~isempty(trial_weights), "includeHyperparameters", sampleHyperparameters);
if(~isempty(trial_weights))
    optStruct_empty.trial_weights(:) = trial_weights;
    optStruct.trial_weights(:) = trial_weights;
end

optStruct_no_dH = optStruct;
optStruct_no_dH.dH = false;
for jj = 1:numel(optStruct_no_dH.Groups)
    optStruct_no_dH.Groups(jj).dH = false;
end


optStruct_dH = optStruct_empty;
optStruct_dH.dH = optStruct.dH;
for jj = 1:numel(optStruct_dH.Groups)
    optStruct_dH.Groups(jj).dH = optStruct.Groups(jj).dH;
end

H_var = (1/5)^2; %initial momentum term for hyperparams

W_scale = 1^2;%1./obj.dim_P;
B_scale = 1^2;%1./obj.dim_P;


paramStruct_2 = paramStruct;
paramStruct_2.W(:) = W_scale;
if(isfield(paramStruct_2, "B"))
    paramStruct_2.B(:) = B_scale;
end
if(isfield(paramStruct_2, "H"))
    paramStruct_2.H(:) = H_var;
end
for jj = 1:obj.dim_J
    if(isfield(paramStruct_2.Groups(jj), "H"))
        paramStruct_2.Groups(jj).H(:) = H_var;
    end
    paramStruct_2.Groups(jj).V(:) = 1;
    for ss = 1:obj.dim_S(jj)
         paramStruct_2.Groups(jj).T{ss}(:) = 1;
    end
end

paramStruct_H = paramStruct;
paramStruct_H.W(:) = 0;
if(isfield(paramStruct_H, "B"))
    paramStruct_H.B(:) = 0;
end
if(isfield(paramStruct_H, "H"))
    paramStruct_H.H(:) = 1;
end
for jj = 1:obj.dim_J
    if(isfield(paramStruct_H.Groups(jj), "H"))
        paramStruct_H.Groups(jj).H(:) = 1;
    end
    paramStruct_H.Groups(jj).V(:) = 0;
    for ss = 1:obj.dim_S(jj)
         paramStruct_H.Groups(jj).T{ss}(:) = 0;
    end
end

paramStruct_P = paramStruct;
paramStruct_P.W(:) = 1;
if(isfield(paramStruct_P, "B"))
    paramStruct_P.B(:) = 1;
end
if(isfield(paramStruct_P, "H"))
    paramStruct_P.H(:) = 0;
end
for jj = 1:obj.dim_J
    if(isfield(paramStruct_P.Groups(jj), "H"))
        paramStruct_P.Groups(jj).H(:) = 0;
    end
    paramStruct_P.Groups(jj).V(:) = 1;
    for ss = 1:obj.dim_S(jj)
         paramStruct_P.Groups(jj).T{ss}(:) = 1;
    end
end

TotalParameters = numel(obj.vectorizeParams(paramStruct, optStruct));
%M = ones(TotalParameters,1);
M = obj.vectorizeParams(paramStruct_2, optStruct);
if(isfield(HMC_settings, "M_init"))
    M = obj.vectorizeParams(HMC_settings.M_init, optStruct);
end

M_H   = obj.vectorizeParams(paramStruct_H, optStruct);
M_P   = obj.vectorizeParams(paramStruct_P, optStruct);

%% initialize space for samples
TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;

samples.log_p_accept  = nan(  TotalSamples,1);
samples.errors        = false(TotalSamples,1);
samples.accepted      = false(TotalSamples,1);
samples.e             = nan(2,TotalSamples);

samples.log_p_accept_sM  = nan(  TotalSamples,1);
samples.log_p_accept_sH = nan(  TotalSamples,1);
samples.errors_sM        = false(TotalSamples,1);
samples.errors_sH       = false(TotalSamples,1);
samples.accepted_sM      = false(TotalSamples,1);
samples.accepted_sH     = false(TotalSamples,1);
samples.e_sM             = nan(2,TotalSamples);
samples.e_sH            = nan(2,TotalSamples);


if(obj.gpuDoublePrecision)
    dataType = "double";
else
    dataType = "single";
end

samples.H       = nan(obj.dim_H,            TotalSamples, "double");
samples.H_gibbs = nan(obj.dim_H_gibbs,      TotalSamples, "double");
samples.W       = nan(obj.dim_P,            TotalSamples, dataType);
samples.B       = nan(obj.dim_B, obj.dim_P, TotalSamples, dataType);
samples.log_post = nan(1, TotalSamples);
samples.log_like = nan(1, TotalSamples);

for jj = 1:obj.dim_J
    samples.Groups(jj).N = nan(obj.dim_R(jj), TotalSamples, dataType);
    samples.Groups(jj).H = nan(obj.dim_H(jj),                TotalSamples, "double");
    samples.Groups(jj).H_gibbs = nan(obj.dim_H_gibbs(jj),    TotalSamples, "double");
    samples.Groups(jj).V = nan(obj.dim_P    , obj.dim_R(jj), TotalSamples, dataType);
    samples.Groups(jj).T = cell(obj.dim_S(jj), 1);
    for ss = 1:obj.dim_S(jj)
         samples.Groups(jj).T{ss} = nan(obj.dim_T(jj, ss), obj.dim_R(jj), TotalSamples, dataType);
    end
end

%save trial log likelihoods to harddrive in a piece-wise manner (otherwise, I'd fill up RAM)
DT = [obj.dim_trialLL(1) obj.dim_trialLL(2)];
samplesBlockSize      = min(HMC_settings.samplesBlockSize, TotalSamples);
samples_block.idx     = nan(samplesBlockSize, 1);
samples_block.trialLL = nan([samplesBlockSize DT(1) DT(2)], dataType);

if(exist(HMC_settings.samplesFile, "file"))
    if(isfield(HMC_settings, "delete_temp_file"))
        continue_opt = HMC_settings.delete_temp_file;
    else
        continue_opt = input(sprintf("Temporary storage file already found (%s)! Overwrite and continue? (y/n)\n ", HMC_settings.samplesFile), "s");
        continue_opt = startsWith(continue_opt, "y", "IgnoreCase", true);
    end
    if(continue_opt)
        fprintf("Deleting temporary storage file and continuing...\n");
    else
        error("Temporary file for storing trial log likelihood samples already exists!\nSpecify another filename or delete if not in use.\n\tfile: %s", HMC_settings.samplesFile);
    end
end

%makes space for trialLL without ever making the full matrix in RAM: this is a cludge around the compression that mafile puts in automatically
fprintf("Preallocating HD space to store LLs for each trial...\n")

obj.temp_storage_file = HMC_settings.samplesFile;
fileID = fopen(HMC_settings.samplesFile, "w");
Z = zeros(TotalSamples,1,dataType);
for ii = 1:DT(1)
    for jj = 1:DT(2)
        fwrite(fileID, Z, dataType);
    end
end
clear Z;
fclose(fileID);

samples_file = memmapfile(HMC_settings.samplesFile,...
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

HMC_state_sM = HMC_state;
HMC_state_sM.steps            = min(HMC_settings.stepSize_sM.maxSteps, ceil(HMC_settings.stepSize_sM.stepL / HMC_state_sM.stepSize.e));
HMC_state_sH = HMC_state;
HMC_state_sH.steps            = min(HMC_settings.stepSize_sH.maxSteps, ceil(HMC_settings.stepSize_sH.stepL / HMC_state_sH.stepSize.e));

%% adds the initial point to the samples
resultStruct_empty = obj.getEmptyResultsStruct(optStruct_empty);
resultStruct_dH = obj.getEmptyResultsStruct(optStruct_dH);
resultStruct_no_dH = obj.getEmptyResultsStruct(optStruct_no_dH);
resultStruct = obj.computeLogPosterior(paramStruct, optStruct);
samples.W(:,1)   = paramStruct.W(:);
samples.B(:,:,1) = paramStruct.B;
samples.H(:,1)   = paramStruct.H(:);
samples.H_gibbs(:,1)   = paramStruct.H_gibbs(:);

for jj = 1:obj.dim_J
    samples.Groups(jj).N(:, 1) = sum(paramStruct.Groups(jj).V.^2,1);
    
    samples.Groups(jj).H(:,1) = paramStruct.Groups(jj).H;
    samples.Groups(jj).H_gibbs(:,1) = paramStruct.Groups(jj).H_gibbs;
    samples.Groups(jj).V(:,:,1) = paramStruct.Groups(jj).V;
    for ss = 1:obj.dim_S(jj)
        samples.Groups(jj).T{ss}(:,:,1) = paramStruct.Groups(jj).T{ss};
        
        samples.Groups(jj).N(:, 1) = samples.Groups(jj).N(:, 1).*sum(paramStruct.Groups(jj).T{ss}.^2,1)';
    end
    samples.Groups(jj).N(:, 1) = sqrt(samples.Groups(jj).N(:, 1));
end

samples_block.idx(1) = 1;
samples_block.trialLL(1, :, :) = resultStruct.trialLL;

samples.log_post(1) = resultStruct.log_post;
samples.log_like(1) = resultStruct.log_likelihood;
samples.e(:,1)      = HMC_state.stepSize.e;
samples.e_sM(:,1)      = HMC_state_sM.stepSize.e;
samples.e_sH(:,1)      = HMC_state_sH.stepSize.e;

samples.log_p_accept(1) = log(1);

fprintf("Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e, max HMC steps = %d\n", TotalSamples, HMC_settings.nWarmup, samples.log_post(1), HMC_state.stepSize.e, HMC_settings.stepSize.maxSteps);

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
    for jj = 1:obj.dim_J
        if(~isempty(obj.GMLMstructure.Groups(jj).gibbs_step) && optStruct.Groups(jj).H_gibbs && isfield(obj.GMLMstructure.Groups(jj).gibbs_step, "sample_func") && ~isempty(obj.GMLMstructure.Groups(jj).gibbs_step.sample_func))
            paramStruct = obj.GMLMstructure.Groups(jj).gibbs_step.sample_func(obj, paramStruct, optStruct, sample_idx, jj, optStruct_empty, resultStruct_empty);
        end
    end
    %%
    if(isfield(HMC_settings, "sample_M") && HMC_settings.sample_M)
        var_struct  = obj.devectorizeParams(inf(size(M)), paramStruct, optStruct);
        if(isfield(obj.GMLMstructure.HMC_setup, "getPriorVar") && ~isempty(obj.GMLMstructure.HMC_setup.getPriorVar))
            var_struct = obj.GMLMstructure.HMC_setup.getPriorVar(paramStruct);
        end
        for jj = 1:obj.dim_J
            if(isfield(obj.GMLMstructure.HMC_setup.Groups(jj), "getPriorVar") && ~isempty(obj.GMLMstructure.HMC_setup.Groups(jj).getPriorVar))
                var_struct.Groups(jj) = obj.GMLMstructure.HMC_setup.Groups(jj).getPriorVar(paramStruct.Groups(jj));
            end
        end
        M_c_0 = M(M_P > 0);
        M_c_1 = obj.vectorizeParams(var_struct, optStruct_no_dH);
        M_c = max(M_c_0, 1./M_c_1);
        
        w_init = obj.vectorizeParams(paramStruct, optStruct_no_dH);
        nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct_no_dH, resultStruct_no_dH);
        try
            [samples.accepted_sM(sample_idx), samples.errors_sM(sample_idx), w_new, samples.log_p_accept_sM(sample_idx), results_sM] = kgmlm.fittingTools.HMCstep_diag(w_init, M_c, nlpostFunction, HMC_state_sM);
            if(samples.accepted_sM(sample_idx))
                paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct_no_dH);
            end
        
            HMC_state_sM = kgmlm.fittingTools.adjustHMCstepSize(sample_idx, HMC_state_sM, HMC_settings.stepSize_sM, samples.log_p_accept_sM(sample_idx));
            samples.e_sM(:,sample_idx) = [HMC_state_sM.stepSize.e; HMC_state_sM.stepSize.e_bar];
        catch
            error("HMC-M step failed");
        end
    end
    
    if(isfield(HMC_settings, "sample_H") && HMC_settings.sample_H)
        M_c = M(M_H > 0);
        if(~isempty(M_c))
            w_init = obj.vectorizeParams(paramStruct, optStruct_dH);
            nlpriorFunction = @(ww) obj.vectorizedNLPrior_func(ww, paramStruct, optStruct_dH, resultStruct_dH);
            try
                [samples.accepted_sH(sample_idx), samples.errors_sH(sample_idx), w_new, samples.log_p_accept_sH(sample_idx), results_sH] = kgmlm.fittingTools.HMCstep_diag(w_init, M_c, nlpriorFunction, HMC_state_sH);
                if(samples.accepted_sH(sample_idx))
                    paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct_dH);
                end

                HMC_state_sH = kgmlm.fittingTools.adjustHMCstepSize(sample_idx, HMC_state_sH, HMC_settings.stepSize_sH, samples.log_p_accept_sH(sample_idx));
                samples.e_sH(:,sample_idx) = [HMC_state_sH.stepSize.e; HMC_state_sH.stepSize.e_bar];
            catch
                error("HMC-H step failed");
            end
        end
    end
    
    %% get HMC sample
    % run HMC step
    w_init = obj.vectorizeParams(paramStruct, optStruct);
    nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct, resultStruct);
    try
        [samples.accepted(sample_idx), samples.errors(sample_idx), w_new, samples.log_p_accept(sample_idx), resultStruct] = kgmlm.fittingTools.HMCstep_diag(w_init, M, nlpostFunction, HMC_state);
        if(samples.accepted(sample_idx))
            paramStruct = obj.devectorizeParams(w_new, paramStruct, optStruct);
        end
    catch
        error("HMC step failed");
    end
    % adjust step size: during warmup
    HMC_state = kgmlm.fittingTools.adjustHMCstepSize(sample_idx, HMC_state, HMC_settings.stepSize, samples.log_p_accept(sample_idx));
    samples.e(:,sample_idx) = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];
    
    
    %% store samples
    samples.W(:,  sample_idx) = paramStruct.W(:);
    samples.B(:,:,sample_idx) = paramStruct.B(:,:);
    samples.H(:,  sample_idx) = paramStruct.H(:);
    samples.H_gibbs(:,  sample_idx) = paramStruct.H_gibbs(:);
    
    for jj = 1:obj.dim_J
        samples.Groups(jj).H(:,sample_idx) = paramStruct.Groups(jj).H;
        samples.Groups(jj).H_gibbs(:,sample_idx) = paramStruct.Groups(jj).H_gibbs;
        samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
        samples.Groups(jj).N(:, sample_idx) = sum(paramStruct.Groups(jj).V.^2,1);
        
        for ss = 1:obj.dim_S(jj)
            samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            samples.Groups(jj).N(:, sample_idx) = samples.Groups(jj).N(:, sample_idx).*sum(paramStruct.Groups(jj).T{ss}.^2,1)';
        end
        samples.Groups(jj).N(:, sample_idx) = sqrt(samples.Groups(jj).N(:, sample_idx));
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
        samples_file.Data.trialLL(samples_block.idx(xx),:,:)  = samples_block.trialLL;
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
            obj.plotSamples(samples, paramStruct, sample_idx);
            drawnow;
        end
    end
    
    
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
        ll_c = squeeze(double(samples_file.Data.trialLL(ss_idx,jj_idx,ii))); 

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

delete(samples_file.Filename); % delete the temporary storage file for trial log likelihoods
obj.temp_storage_file = [];
end





