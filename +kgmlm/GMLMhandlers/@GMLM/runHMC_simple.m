function [samples, summary, HMC_settings, paramStruct, M_chol] = runHMC_simple(obj, HMC_settings, paramStruct, initializerScale, textOutputFile)

if(~isfield(HMC_settings,'figNum') || isempty(HMC_settings.figNum))
    figNum = 5;
else
    figNum = HMC_settings.figNum;
end
if(~obj.isOnGPU())
    error('Must load gmlm onto GPU before running HMC!');
end

%% if no output file givem prints status to screen
if(nargin < 5 || isempty(textOutputFile))
    textOutputFile = 1;
end

%% setup HMC settings
%if not given settings
if(nargin < 2 || isempty(HMC_settings))
    %% covariance estimation
    warning('No HMC settings given. Starting a default run!');
    HMC_settings = obj.setupHMCparams();
end

%% sets up the hmc momentum cov matrices

optStruct = obj.getEmptyOptsStruct(true,false);
optStruct = obj.getParamCount(optStruct);
optStruct.compute_trialLL = true;

M_chol.H = ones(optStruct.totalHyperparams,1);
M_chol.W = ones(optStruct.totalParams,1);

%% initialize space for samples
samples.log_p_accept  = nan(HMC_settings.nTotal,1);
samples.errors        = false(HMC_settings.nTotal,1);
samples.accepted      = false(HMC_settings.nTotal,1);
samples.e             = nan(2,HMC_settings.nTotal);

nMH = 0;
for jj = 1:obj.dim_J
    nMH = nMH + obj.Groups(jj).dim_R;
end

samples.MH.accepted      = nan(nMH, HMC_settings.nTotal);
samples.MH.log_p_accept  = nan(nMH, HMC_settings.nTotal);

dataType = 'single';
if(~obj.gpuSinglePrecision)
    dataType = 'double';
end

samples.H       = nan(obj.dim_H,HMC_settings.nTotal,'double');
samples.W       = nan(obj.dim_P,HMC_settings.nTotal,dataType);
samples.B       = nan(obj.dim_K,obj.dim_P,HMC_settings.nTotal,dataType);
samples.log_post = nan(1,HMC_settings.nTotal);
samples.log_like = nan(1,HMC_settings.nTotal,dataType);

for jj = 1:obj.dim_J
    samples.Groups(jj).H = nan(obj.Groups(jj).dim_H,HMC_settings.nTotal,'double');
    samples.Groups(jj).V = nan(obj.dim_P,obj.Groups(jj).dim_R,HMC_settings.nTotal,dataType);
    samples.Groups(jj).T = cell(obj.Groups(jj).dim_S,1);
    for ss = 1:obj.Groups(jj).dim_S
         samples.Groups(jj).T{ss} = nan(obj.Groups(jj).dim_T(ss),obj.Groups(jj).dim_R,HMC_settings.nTotal,dataType);
    end
end

% samples.T_1_samples = nan(obj.Groups(1).dim_T(1),HMC_settings.nTotal);
samplesBlockSize = min(HMC_settings.samplesBlockSize, HMC_settings.nTotal);
samples_block.idx     = nan(samplesBlockSize, 1);
samples_block.trialLL = nan(obj.dim_M, samplesBlockSize, dataType);

a = 1;
if(exist(HMC_settings.samplesFile, 'file'))
    %status = fopen(HMC_settings.samplesFile)
    
    continue_opt = input(sprintf('Temporary storage file already found (%s)! Overwrite and continue? (y/n)\n ', HMC_settings.samplesFile), 's');
    if(startsWith(continue_opt, 'y', 'IgnoreCase', true))
        fprintf('Continuing... will overwrite file.\n');
    else
        error('Temporary file for storing trial log likelihood samples already exists!\nSpecify another filename or delete if not in use.\n\tfile: %s', HMC_settings.samplesFile);
    end
end
save(HMC_settings.samplesFile, 'a', '-nocompression', '-v7.3');
samples_file = matfile(HMC_settings.samplesFile, 'Writable',true);
samples_file.trialLL = zeros(obj.dim_M, samplesBlockSize, dataType);

NB = ceil(HMC_settings.nTotal / samplesBlockSize);
for ii = 1:NB
    if(ii == NB)
        idx = ((NB-1)*samplesBlockSize + 1):HMC_settings.nTotal;
    else
        idx = (1:samplesBlockSize) + (NB-1)*samplesBlockSize;
    end
    samples_file.trialLL(:, idx) = nan(obj.dim_M, numel(idx), dataType);
end


%% set initial point

if(nargin <= 2 || isempty(paramStruct))
    if(nargin < 4 || isempty(initializerScale))
        initializerScale = 1;
    end
    warning('No initialization point provided: generating random initialization!');
    paramStruct = obj.getRandomParamStruct(initializerScale);
    if(isfield(HMC_settings, 'map_init') && HMC_settings.map_init)
        paramStruct = obj.getMLE_MAP(paramStruct, 10);
    end
end

%% initialize HMC state
HMC_state.stepSize.e = HMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar = HMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t = 0;
HMC_state.stepSize.H_sum = 0;
HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps, ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));

%% adds the initial point to the samples
resultStruct = obj.computeLPost(paramStruct,optStruct,true);
samples.W(:,1)   = paramStruct.W(:);
samples.B(:,:,1) = paramStruct.B;
samples.H(:,1)   = paramStruct.H(:);

for jj = 1:obj.dim_J
    samples.Groups(jj).H(:,1) = paramStruct.Groups(jj).H;
    samples.Groups(jj).V(:,:,1) = paramStruct.Groups(jj).V;
    for ss = 1:obj.Groups(jj).dim_S
        samples.Groups(jj).T{ss}(:,:,1) = paramStruct.Groups(jj).T{ss};
    end
end

samples_block.idx(1) = 1;
samples_block.trialLL(:, 1) = resultStruct.trialLL;

samples.log_post(1) = resultStruct.log_post;
samples.log_like(1) = resultStruct.log_like;
samples.e(:,1)      = HMC_state.stepSize.e;

samples.log_p_accept(1) = log(1);

printMsg(textOutputFile,'Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e, max HMC steps = %d\n', HMC_settings.nTotal, HMC_settings.nWarmup, samples.log_post(1), HMC_state.stepSize.e, HMC_settings.HMC_step.maxSteps);

if(HMC_settings.showPlots)
    figure(figNum);
    clf;
    drawnow;
end

%%
vectorizedSamples = zeros(optStruct.totalParams, HMC_settings.M_est.samples(end));
vectorizedSamples_hyperparams = zeros(optStruct.totalHyperparams, HMC_settings.M_est.samples(end));
start_idx = 2;

%% run sampler
for sample_idx = start_idx:HMC_settings.nTotal

    
    %% run MH rescaling step
    % the decompositions have some directions of unidentifiability in the likelihood (V*T' = (V*R)*(T*R^-1)')
    % These optional steps do some fast MH proposals to quickly move around in that space, and does not require any likelihood computations.
    if(mod(sample_idx,HMC_settings.MH_scale.sample_every) == 0 && HMC_settings.MH_scale.N > 0)
        MH_accepted_c = nan(size(samples.MH.accepted,1),HMC_settings.MH_scale.N);
        MH_log_p      = nan(size(samples.MH.accepted,1),HMC_settings.MH_scale.N);
        for ii = 1:HMC_settings.MH_scale.N
            [paramStruct, MH_accepted_c(:,ii), MH_log_p(:,ii)] = obj.scalingMHStep(paramStruct, HMC_settings.MH_scale);
        end
        samples.MH.accepted(:,sample_idx) = nanmean(MH_accepted_c,2);
        samples.MH.log_p_accept(:,sample_idx) = nanmean(MH_log_p,2);
        if(~all(~MH_accepted_c,'all'))
            resultStruct = obj.computeLPost(paramStruct, optStruct, true);
        end
    end
    
    %% get HMC sample
    if(sample_idx < HMC_settings.HMC_step.nSamples_init)
        HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps_init, ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));
    else
        HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps,      ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));
    end
    
    % run HMC step
    [samples.accepted(sample_idx), samples.errors(sample_idx), paramStruct, samples.log_p_accept(sample_idx), resultStruct] = obj.HMCstep_simple(paramStruct, M_chol, HMC_state, resultStruct);
    
    %% store samples
    samples.W(:,  sample_idx) = paramStruct.W(:);
    samples.B(:,:,sample_idx) = paramStruct.B(:,:);
    samples.H(:,  sample_idx) = paramStruct.H(:);
    
    for jj = 1:obj.dim_J
        samples.Groups(jj).H(:,sample_idx) = paramStruct.Groups(jj).H;
        samples.Groups(jj).V(:,:,sample_idx) = paramStruct.Groups(jj).V;
        
        for ss = 1:obj.Groups(jj).dim_S
            samples.Groups(jj).T{ss}(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
        end
    end
    
%     V_c = samples.Groups(1).V(1,:,sample_idx).* samples.Groups(1).T{1}(:,:,sample_idx);
%     for ss = 2:numel(samples.Groups(1).T)
%         V_c = V_c .* samples.Groups(1).T{ss}(1,:,sample_idx);
%     end
%     samples.T_1_samples(:,sample_idx) = sum(V_c,2);

    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_like;
    
    if(sample_idx <= HMC_settings.M_est.samples(end))
        vectorizedSamples(:, sample_idx) = paramStruct.W_all; %obj.vectorizeParamStruct(paramStruct, optStruct);
        vectorizedSamples_hyperparams(:, sample_idx) = paramStruct.H_all;
    end
    
    %temp storage of trialLL
    idx_c = mod(sample_idx-1, samplesBlockSize) + 1;
    samples_block.idx(idx_c) = sample_idx;
    samples_block.trialLL(:, idx_c) = resultStruct.trialLL;
    if(mod(sample_idx, samplesBlockSize) == 0 || sample_idx == HMC_settings.nTotal)
        %save to file
        xx = ~isnan(samples_block.idx);
        samples_file.trialLL(:,samples_block.idx(xx))  = samples_block.trialLL;
    end
    
    %% print any updates
    if(sample_idx <= 50 || (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == HMC_settings.nTotal || (HMC_settings.verbose && mod(sample_idx,20) == 0))
        if(sample_idx == HMC_settings.nTotal)
            ww = (HMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        mean_MH_accepted = nanmean(samples.MH.accepted(:,ww),'all');
        
        printMsg(textOutputFile,'HMC step %d / %d (accept per. = %.1f in last %d steps, curr log post = %e, (log like = %e)\n',sample_idx,HMC_settings.nTotal,mean(samples.accepted(ww))*100,numel(ww),samples.log_post(sample_idx),samples.log_like(sample_idx));
        printMsg(textOutputFile,'\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d, mean MH accepted = %.3f\n', HMC_state.stepSize.e, HMC_state.steps,nansum(samples.errors),  mean_MH_accepted);
        clear ww;
        
        if(HMC_settings.showPlots)
            sfigure(figNum);
            plotSamples_gmlm(samples, HMC_settings, paramStruct, sample_idx);%, samples.T_1_samples
            drawnow;
        end
    end
    
    %% adjust step size
    HMC_state.stepSize  = adjustHMCstepSize(sample_idx, HMC_state.stepSize, HMC_settings.stepSize, samples.log_p_accept(sample_idx));
    samples.e(:,sample_idx)   = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];
    
    %% updates the covariance matrix of the hyperparameters
    if(ismember(sample_idx, HMC_settings.M_est.samples ) ) 
        start_idx = HMC_settings.M_est.first_sample(HMC_settings.M_est.samples == sample_idx);
        ww = start_idx:sample_idx;
        
        if(~HMC_settings.M_est.diagOnly(HMC_settings.M_est.samples == sample_idx))
            
            if(HMC_settings.M_est.useMask)
                MASK = zeros(optStruct.totalParams, optStruct.totalParams);
                vw = double(optStruct.vw_ranks);
                p = double(obj.dim_P);
                MASK(1:(p*vw), 1:(p*vw)) = kron(ones(vw,vw), eye(p));

                ctr = p*vw;
                for jj = 1:obj.dim_J
                    rr = double(obj.Groups(jj).dim_R);

                    for ss = 1:obj.Groups(jj).dim_S
                        tt = double(obj.Groups(jj).dim_T(ss));

                        MASK(ctr + (1:(tt*rr)), ctr + (1:(tt*rr))) = kron(eye(rr), ones(tt,tt));
                        ctr = ctr + tt*rr;
                    end
                end
            else
                MASK = ones(optStruct.totalParams, optStruct.totalParams);
            end
            
            %full cov mats (block diag structure for hyperparams and params)
            M_chol.W = chol(inv(cov(vectorizedSamples(:,ww)').*MASK));
        else
            %diagonal only
            M_chol.W = (1./var(vectorizedSamples(:,ww),[],2));
        end
        
        if(~HMC_settings.M_est.diagOnly_hyper(HMC_settings.M_est.samples == sample_idx))
            M_chol.H = chol(inv(cov(vectorizedSamples_hyperparams(:,ww)')));
        else
            M_chol.H = (1./var(vectorizedSamples_hyperparams(:,ww),[],2));
        end
    end
end

%% finish sampler
dim_M = double(obj.dim_M);
ss_idx = (HMC_settings.nWarmup+1):sample_idx;

printMsg(textOutputFile,'computing WAIC and PSIS-LOO... ');
T_n = zeros(dim_M,1);
V_n = zeros(dim_M,1);

summary.PSISLOOS   = zeros(dim_M,1);
summary.PSISLOO_PK = zeros(dim_M,1);

% ll = samples.trialLL(:,ss_idx);
for ii = 1:dim_M
    ll_c = double(samples_file.trialLL(ii,ss_idx));
    T_n(ii) = -logMeanExp(ll_c,2);
    V_n(ii) = mean(ll_c.^2,2) - mean(ll_c,2).^2;
    
    [~,summary.PSISLOOS(ii),summary.PSISLOO_PK(ii)] = psisloo(ll_c(:));
end
summary.WAICS = T_n + V_n;
summary.WAIC  = mean(summary.WAICS,1);

summary.PSISLOO = sum(summary.PSISLOOS);

badSamples   = sum(summary.PSISLOO_PK >= 0.7);
if(badSamples > 0)
    printMsg(textOutputFile,'Warning: PSISLOO PK large (>0.7) for %d / %d observations! ',badSamples, numel(summary.PSISLOO_PK ));
else
    printMsg(textOutputFile,'PSISLOO diagnostics passed (all PK < 0.7). ');
end

ss_all = (HMC_settings.nWarmup+1):HMC_settings.nTotal;
summary.earlyRejects = sum(samples.errors(ss_all));
summary.earlyReject_prc = mean(samples.errors(ss_all));
summary.HMC_state            = HMC_state;
summary.acceptRate   = mean(samples.accepted(ss_all));

printMsg(textOutputFile,'done.\n');   

delete(samples_file.Properties.Source);

end


%% for printing to file and command window
function [] = printMsg(file,str,varargin)

fprintf(file,str,varargin{:});
if(file ~= 1)
    fprintf(str,varargin{:});
end
end



