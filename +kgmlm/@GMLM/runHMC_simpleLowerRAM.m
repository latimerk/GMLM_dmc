% Runs the HMC algorithm for the GMLM
% Computes PSISLOO and WAIC on the samples.
%
% NOTE: uses HMC_settings.samplesFile to store results on the fly.
%       uses HMC_settings.trialLLfile to This file can get big, but is deleted after running is complete.
%
%  Inputs:
%      params_init  : the initial parameters for sample = 1
%      HMC_settings : HMC settings struct (see setupHMCparams)
% 
%    Optional (key/value pairs)
%       figure                (number;  default = nan ) : number of the figure to plot MCMC traces (nan means do not plot) 
%       sampleHyperparameters (logical; default = true) : true/false to sample hyperparams. If false, leaves them fixed given params_init
%       trial_weights         (array size gmlm.dim_M; default = []) : weighting of the trials (empty for all trial weights 1)

function [samples, samples_file_format, summary, HMC_settings, paramStruct, M] = runHMC_simpleLowerRAM(obj, params_init, HMC_settings, varargin)
p = inputParser;
p.CaseSensitive = false;

addRequired(p, "params_init",  @(aa)(isstruct(aa) | isempty(aa)));
addRequired(p, "settings", @isstruct);
addParameter(p, "figure" ,    nan, @isnumeric);
addParameter(p, "printFunc" ,    []);
addParameter(p, "optStruct" ,   [], @(aa) isempty(aa) | obj.verifyComputeOptionsStruct(aa));
addParameter(p, "sampleHyperparameters", true, @islogical);
addParameter(p, "trial_weights"   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));
addParameter(p, "saveUnscaled" ,    false, @islogical);
addParameter(p, "saveSinglePrecision" ,    false, @islogical);
addParameter(p, "modelInfo" ,    []);

parse(p, params_init, HMC_settings, varargin{:});
% then set/get all the inputs out of this structure
optStruct          = p.Results.optStruct;
paramStruct        = p.Results.params_init;
HMC_settings       = p.Results.settings;
trial_weights      = p.Results.trial_weights;
modelInfo      = p.Results.modelInfo;
sampleHyperparameters = p.Results.sampleHyperparameters;
figNum = p.Results.figure;
printFunc      = p.Results.printFunc;
saveUnscaled      = p.Results.saveUnscaled;
saveSinglePrecision      = p.Results.saveSinglePrecision;
    
if(~obj.isOnGPU())
    error("Must load gmlm onto GPU before running HMC!");
end

J = obj.dim_J;
S = obj.dim_S;
%% sets up the hmc momentum cov matrices
if(isempty(optStruct))
    optStruct = obj.getComputeOptionsStruct(true, "trial_weights", ~isempty(trial_weights), "includeHyperparameters", sampleHyperparameters);
end
optStruct_empty = obj.getComputeOptionsStruct(false, "trial_weights", ~isempty(trial_weights), "includeHyperparameters", sampleHyperparameters);
if(~isempty(trial_weights))
    optStruct_empty.trial_weights(:) = trial_weights;
    optStruct.trial_weights(:) = trial_weights;
end
resultStruct_empty = obj.getEmptyResultsStruct(optStruct_empty);


H_var = (1/1)^2; %initial momentum term for hyperparams

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

save_H.B = ~isempty(paramStruct.B);
save_H.H = ~isempty(paramStruct.H);
save_H.H_gibbs = ~isempty(paramStruct.H_gibbs);
for jj = 1:J
    save_H.Groups(jj).H = ~isempty(paramStruct.Groups(jj).H);
    save_H.Groups(jj).H_gibbs = ~isempty(paramStruct.Groups(jj).H_gibbs);
end


TotalParameters = numel(obj.vectorizeParams(paramStruct, optStruct));
%M = ones(TotalParameters,1);
M = obj.vectorizeParams(paramStruct_2, optStruct);
if(isfield(HMC_settings, "M_init"))
    M = obj.vectorizeParams(HMC_settings.M_init, optStruct);
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

%% check for partial output files
if(isfield(HMC_settings, "savePartialProgressFile") && exist(HMC_settings.savePartialProgressFile, "file") ...
        && isfield(HMC_settings, "savePartialProgressValid") && HMC_settings.savePartialProgressValid ...
        && exist(HMC_settings.trialLLfile, "file") && exist(HMC_settings.samplesFile, "file"))
    partialFileFound = true;
else
    partialFileFound = false;
end

%% initialize space for samples
if(~partialFileFound)
    TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;
    
    samples.log_p_accept  = nan(  TotalSamples,1);
    samples.errors        = false(TotalSamples,1);
    samples.accepted      = false(TotalSamples,1);
    samples.e             = nan(2,TotalSamples);
    samples.log_post = nan(1, TotalSamples);
    samples.log_like = nan(1, TotalSamples);
    
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
    
    %save trial log likelihoods to harddrive in a piece-wise manner (otherwise, I'd fill up RAM)
    DT = [obj.dim_trialLL(1) obj.dim_trialLL(2)];
    samples_block.samplesBlockSize = min(HMC_settings.samplesBlockSize, TotalSamples);
    samples_block.idx     = nan(samples_block.samplesBlockSize, 1);
    samples_block.trialLL = nan([samples_block.samplesBlockSize DT(1) DT(2)], dataType);
    
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
    
    if(exist(HMC_settings.samplesFile, "file"))
        if(isfield(HMC_settings, "delete_samples_file"))
            continue_opt = HMC_settings.delete_samples_file;
        else
            continue_opt = input(sprintf("Samples storage file already found (%s)! Overwrite and continue? (y/n)\n ", HMC_settings.samplesFile), "s");
            continue_opt = startsWith(continue_opt, "y", "IgnoreCase", true);
        end
        if(continue_opt)
            fprintf("Deleting samples storage file and continuing...\n");
        else
            error("Temporary file for storing samples already exists!\nSpecify another filename or delete if not in use.\n\tfile: %s", HMC_settings.samplesFile);
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
    
    samples_file_format = cell(0, 3);
    ctr = 1;
    totalParams = 0;
    
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.W) TotalSamples];
    samples_file_format{ctr, 3} = "W"; totalParams = totalParams + numel(paramStruct.W);
    ctr = ctr + 1;
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [size(paramStruct.B) TotalSamples];
    samples_file_format{ctr, 3} = "B"; totalParams = totalParams + numel(paramStruct.B);
    ctr = ctr + 1;
    
    if(saveUnscaled && scaled_WB)
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [numel(paramStruct.W) TotalSamples];
        samples_file_format{ctr, 3} = "W_scaled"; totalParams = totalParams + numel(paramStruct.W);
        ctr = ctr + 1;
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [size(paramStruct.B) TotalSamples];
        samples_file_format{ctr, 3} = "B_scaled"; totalParams = totalParams + numel(paramStruct.B);
        ctr = ctr + 1;
    end
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.H) TotalSamples];
    samples_file_format{ctr, 3} = "H"; totalParams = totalParams + numel(paramStruct.H);
    ctr = ctr + 1;
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.H_gibbs) TotalSamples];
    samples_file_format{ctr, 3} = "H_gibbs"; totalParams = totalParams + numel(paramStruct.H_gibbs);
    ctr = ctr + 1;
    
    for jj = 1:J
        
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).V) TotalSamples];
        samples_file_format{ctr, 3} = sprintf("G%d_V", jj);
        ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).V);
    
        for ss = 1:S(jj) 
            samples_file_format{ctr, 1} = dataType_samples;
            samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).T{ss}) TotalSamples];
            samples_file_format{ctr, 3} = sprintf("G%d_T_%d", jj, ss);
            ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).T{ss});
        end
    
        if(saveUnscaled && scaled_VT(jj))
            samples_file_format{ctr, 1} = dataType_samples;
            samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).V) TotalSamples];
            samples_file_format{ctr, 3} = sprintf("G%d_V_scaled", jj);
            ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).V);
        
            for ss = 1:S(jj) 
                samples_file_format{ctr, 1} = dataType_samples;
                samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).T{ss}) TotalSamples];
                samples_file_format{ctr, 3} = sprintf("G%d_T_%d_scaled", jj, ss);
                ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).T{ss});
            end
        end
    
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [numel(paramStruct.Groups(jj).H) TotalSamples];
        samples_file_format{ctr, 3} = sprintf("G%d_H", jj);
        ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).H);
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [numel(paramStruct.Groups(jj).H_gibbs) TotalSamples];
        samples_file_format{ctr, 3} = sprintf("G%d_H_gibbs", jj);
        ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).H_gibbs);
    end
    
    Z = zeros(TotalSamples, 1,dataType_samples);
    fprintf("Preallocating HD space to store samples (~%.3f gb)...\n", whos("Z").bytes * totalParams / 1e9);
    fileID = fopen(HMC_settings.samplesFile, "w");
    for ii = 1:totalParams
        fwrite(fileID, Z, dataType_samples);
    end
    clear Z;
    fclose(fileID);
    
    samples_file = memmapfile(HMC_settings.samplesFile,...
                   "Format",samples_file_format, ...
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
    resultStruct = obj.computeLogPosterior(paramStruct, optStruct);
    sample_idx = 1;
    paramStruct2 = paramStruct;
    if(scaled_WB)
        params_0 = obj.GMLMstructure.scaleParams(paramStruct);
    
        samples_file.Data.W(:,  sample_idx) = params_0.W(:);
        if(save_H.B)
            samples_file.Data.B(:,:,sample_idx) = params_0.B(:,:);
        end
        paramStruct2.W(:) = params_0.W(:);
        paramStruct2.B(:) = params_0.B(:);
    
        if(saveUnscaled)
            samples_file.Data.W_scaled(:,  sample_idx) = paramStruct.W(:);
            if(save_H.B)
                samples_file.Data.B_scaled(:,:,sample_idx) = paramStruct.B(:,:);
            end
        end
    else
        samples_file.Data.W(:,  sample_idx) = paramStruct.W(:);
        if(save_H.B)
            samples_file.Data.B(:,:,sample_idx) = paramStruct.B(:,:);
        end
    end
    if(save_H.H)
        samples_file.Data.H(:,1)   = paramStruct.H(:);
    end
    if(save_H.H_gibbs)
        samples_file.Data.H_gibbs(:,1)   = paramStruct.H_gibbs(:);
    end
    
    for jj = 1:J
        if(save_H.Groups(jj).H)
            samples_file.Data.(sprintf("G%d_H", jj))(:,1) = paramStruct.Groups(jj).H;
        end
        if(save_H.Groups(jj).H_gibbs)
            samples_file.Data.(sprintf("G%d_H_gibbs", jj))(:,1) = paramStruct.Groups(jj).H_gibbs;
        end
        if(scaled_VT(jj))
            params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));
    
            samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx) = params_0.V;
            paramStruct2.Groups(jj).V(:) = params_0.V(:);

            if(saveUnscaled)
                samples_file.Data.(sprintf("G%d_V_scaled", jj))(:,:,sample_idx) = paramStruct.Groups(jj).V;
            end
            for ss = 1:S(jj)
                samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx) = params_0.T{ss};
                paramStruct2.Groups(jj).T{ss}(:) = params_0.T{ss}(:);
                if(saveUnscaled)
                    samples_file.Data.(sprintf("G%d_T_%d_scaled", jj, ss))(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
                end
            end
        else
            samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx) = paramStruct.Groups(jj).V;
            for ss = 1:S(jj)
                samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            end
        end
    end
    
    samples_block.idx(1) = 1;
    samples_block.trialLL(1, :, :) = resultStruct.trialLL;
    
    samples.log_post(1) = resultStruct.log_post;
    samples.log_like(1) = resultStruct.log_likelihood;
    samples.e(:,1)      = HMC_state.stepSize.e;
    samples.log_p_accept(1) = log(1);
    start_idx = 2;
else
    load(HMC_settings.savePartialProgressFile, "HMC_settings", "HMC_state", "paramStruct", "resultStruct", "paramStruct2", "samples_file_format", "samples_block", "samples", "scaled_WB", "scaled_VT", "sample_idx", "randomNumberState");
    TotalSamples = HMC_settings.nWarmup + HMC_settings.nSamples;
    DT = [obj.dim_trialLL(1) obj.dim_trialLL(2)];
    trialLL_file = memmapfile(HMC_settings.trialLLfile,...
                   "Format",{dataType,[TotalSamples DT(1) DT(2)],"trialLL"}, ...
                   "Writable", true);
    samples_file = memmapfile(HMC_settings.samplesFile,...
                   "Format",samples_file_format, ...
                   "Writable", true);
    start_idx = sample_idx;
    rng(randomNumberState);
    fprintf("Loading partial progress file at sample %d...\n", start_idx);
end


fprintf("Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e,\n\tmax HMC steps = %d\n", TotalSamples, HMC_settings.nWarmup, samples.log_post(1), HMC_state.stepSize.e, HMC_settings.stepSize.maxSteps);
for ii = 1:size(HMC_settings.stepSize.schedule ,1)
    fprintf("\tStep size estimation samples %d - %d, target accept rate = %.2f\n", HMC_settings.stepSize.schedule(ii,1), HMC_settings.stepSize.schedule(ii,2), HMC_settings.stepSize.delta(min(ii, numel(HMC_settings.stepSize.delta))));
end

% if(~isnan(figNum) && ~isinf(figNum))
%     figure(figNum);
%     clf;
%     drawnow;
% end

%%
vectorizedSamples             = nan(TotalParameters, HMC_settings.M_est.samples(end), dataType);
if(sample_idx <= HMC_settings.M_est.samples(end))
    vectorizedSamples(:, sample_idx) = obj.vectorizeParams(paramStruct, optStruct);
end

%% run sampler
for sample_idx = start_idx:TotalSamples
    %% save partial progress if requested
    if(isfield(HMC_settings, "savePartialProgressN") && HMC_settings.savePartialProgressN > 0 && mod(sample_idx, HMC_settings.savePartialProgressN) == 0 && sample_idx > start_idx)
        obj.destroy_temp_storage_file = false;

        randomNumberState = rng();
        save(HMC_settings.savePartialProgressFile, "-v7.3", "randomNumberState", "HMC_settings", "paramStruct2", "HMC_state", "paramStruct", "resultStruct", "samples_file_format", "samples_block", "samples", "scaled_WB", "scaled_VT", "sample_idx", "modelInfo");
    end

    %% set paramStruct to MAP estimate (should only be done early in warmup if at all)
    
    if(isfield(HMC_settings, "fitMAP") && ismember(sample_idx, HMC_settings.fitMAP))
        fprintf("Attempting to accelerate mixing by finding MAP estimate given current hyperparameters...\n");
%         fprintf("   Alternating steps...\n");
%         paramStruct = obj.computeMAP(paramStruct, "optStruct", optStruct, "alternating_opt", true, "max_iters", 5, "max_quasinewton_steps", 500);
        fprintf("   All params at once...\n");
        paramStruct = obj.computeMAP(paramStruct, "optStruct", optStruct, "alternating_opt", false, "max_iters", 1, "max_quasinewton_steps", 500);
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
    
    %% get HMC sample
    % run HMC step
    w_init = obj.vectorizeParams(paramStruct, optStruct);
    nlpostFunction = @(ww) obj.vectorizedNLPost_func(ww, paramStruct, optStruct, resultStruct);
    try
        HMC_state.e_scale = samples.e_scale(sample_idx);
        [samples.accepted(sample_idx), samples.errors(sample_idx), w_new, samples.log_p_accept(sample_idx), resultStruct] = kgmlm.fittingTools.HMCstep_diag(w_init, HMC_settings.M_const * M, nlpostFunction, HMC_state);
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
    paramStruct2 = paramStruct;
    if(scaled_WB)
        params_0 = obj.GMLMstructure.scaleParams(paramStruct);

        samples_file.Data.W(:,  sample_idx) = params_0.W(:);
        if(save_H.B)
            samples_file.Data.B(:,:,sample_idx) = params_0.B(:,:);
        end
        paramStruct2.W(:) = params_0.W(:);
        paramStruct2.B(:) = params_0.B(:);

        if(saveUnscaled)
            samples_file.Data.W_scaled(:,  sample_idx) = paramStruct.W(:);
            if(save_H.B)
                samples_file.Data.B_scaled(:,:,sample_idx) = paramStruct.B(:,:);
            end
        end
    else
        samples_file.Data.W(:,  sample_idx) = paramStruct.W(:);
        if(save_H.B)
            samples_file.Data.B(:,:,sample_idx) = paramStruct.B(:,:);
        end
    end

    if(save_H.H)
        samples_file.Data.H(:,  sample_idx) = paramStruct.H(:);
    end
    if(save_H.H_gibbs)
        samples_file.Data.H_gibbs(:,  sample_idx) = paramStruct.H_gibbs(:);
    end

    for jj = 1:J
        if(save_H.Groups(jj).H)
            samples_file.Data.(sprintf("G%d_H", jj))(:,sample_idx) = paramStruct.Groups(jj).H;
        end
        if(save_H.Groups(jj).H_gibbs)
            samples_file.Data.(sprintf("G%d_H_gibbs", jj))(:,sample_idx) = paramStruct.Groups(jj).H_gibbs;
        end

        if(scaled_VT(jj))
            params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));

            samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx) = params_0.V;
            paramStruct2.Groups(jj).V(:) = params_0.V(:);

            if(saveUnscaled)
                samples_file.Data.(sprintf("G%d_V_scaled", jj))(:,:,sample_idx) = paramStruct.Groups(jj).V;
            end
            for ss = 1:S(jj)
                samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx) = params_0.T{ss};
                paramStruct2.Groups(jj).T{ss}(:) = params_0.T{ss}(:);
                if(saveUnscaled)
                    samples_file.Data.(sprintf("G%d_T_%d_scaled", jj, ss))(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
                end
            end
        else
            samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
            for ss = 1:S(jj)
                samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            end
        end
    end
    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_likelihood;
    if(sample_idx <= HMC_settings.M_est.samples(end))
        vectorizedSamples(:, sample_idx) = w_new; 
    end
    
    %temp storage of trialLL
    idx_c = mod(sample_idx-1, samples_block.samplesBlockSize) + 1;
    samples_block.idx(       idx_c) = sample_idx;
    samples_block.trialLL(idx_c, :, :) = resultStruct.trialLL;
    if(mod(sample_idx, samples_block.samplesBlockSize) == 0 || sample_idx == TotalSamples)
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
        if(sample_idx > HMC_settings.nWarmup)
            total_errors_post = sum(samples.errors((HMC_settings.nWarmup+1):sample_idx), "omitnan");
            total_errors_during = sum(samples.errors(1:HMC_settings.nWarmup), "omitnan");
            fprintf("\tcurrent step size = %e, HMC steps = %d, num HMC early rejects post warmup = %d (during warmup %d)\n", HMC_state.stepSize.e, HMC_state.steps, total_errors_post, total_errors_during);
        else
            total_errors = sum(samples.errors(1:sample_idx), "omitnan");
            fprintf("\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d\n", HMC_state.stepSize.e, HMC_state.steps, total_errors);
        end
        if(~isempty(printFunc))
            printFunc(paramStruct2);
        end
        
        clear ww;
        
%         if(~isnan(figNum) && ~isinf(figNum))
%             kgmlm.utils.sfigure(figNum);
%             obj.plotSamples(samples, metrics, paramStruct, sample_idx);
%             drawnow;
%         end
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
obj.destroy_temp_storage_file = true;

end





