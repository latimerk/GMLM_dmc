function [samples, summary, HMC_settings, outputMsg, paramStruct] = runHMC(obj, HMC_settings, textOutputFile, msgStr_0)
% HMC_settings.verbose = true;
if(nargin < 3 || isempty(textOutputFile))
    textOutputFile = 1;
end
if(nargin < 4)
    msgStr_0 = '';
end

outputMsg = '';

if(nargin < 2 || isempty(HMC_settings))
    fprintf('Using default HMC settings.\n');
    HMC_settings = obj.setupHMCparams();
end


%% initialize HMC state
HMC_state.stepSize.e = HMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar = HMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t = 0;
HMC_state.stepSize.H_sum = 0;
HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps, ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));

%% setup samples
samples = struct();
sample_idx = 1;

if(obj.gpuSinglePrecision)
    dataType = 'single';
else
    dataType = 'double';
end

samples.W = nan(obj.dim_P,HMC_settings.nTotal,dataType);
samples.H = nan(obj.dim_H,HMC_settings.nTotal);

samples.accepted      = false(HMC_settings.nTotal,1);
samples.errors        = false(HMC_settings.nTotal,1);
samples.log_post      = nan(HMC_settings.nTotal,1);
samples.log_like      = nan(HMC_settings.nTotal,1,dataType);
samples.log_p_accept  = nan(HMC_settings.nTotal,1);

samples.e         = nan(2,HMC_settings.nTotal);

ll_trs = nan(obj.dim_M,HMC_settings.nTotal,dataType);


%%
figNum = 10;

%%
M_chol.W = ones(obj.dim_P,1);
M_chol.H = ones(obj.dim_H,1);


%%
paramStruct = obj.getRandomParamStruct();


optStruct = obj.getEmptyOptsStruct(true,false);
optStruct.compute_trialLL = true;
resultStruct = obj.computeLPost(paramStruct,optStruct,true);
samples.W(:,1) = paramStruct.W(:);
samples.H(:,1) = paramStruct.H(:);

samples.trialLL(:,1)  = resultStruct.trialLL;
samples.log_post(1)   = resultStruct.log_post;
samples.log_like(1)   = resultStruct.log_like;
samples.e(:,1)        = HMC_state.stepSize.e;
%%

printMsg(textOutputFile,'Starting HMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e\n',HMC_settings.nTotal,HMC_settings.nWarmup,samples.log_post(1),HMC_state.stepSize.e);
        
for sample_idx = sample_idx:HMC_settings.nTotal
    if(sample_idx > 1)
        %% get HMC sample - params & hyperparamers;
        if(sample_idx < HMC_settings.HMC_step.nSamples_init)
            HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps_init, ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));
        else
            HMC_state.steps   = min(HMC_settings.HMC_step.maxSteps,      ceil(HMC_settings.HMC_step.stepL/HMC_state.stepSize.e));
        end

        [samples.accepted(sample_idx),samples.errors(sample_idx),paramStruct,samples.log_p_accept(sample_idx),resultStruct] = obj.hmcStep(paramStruct, M_chol, HMC_state, resultStruct);
    end
    
    %% store current sample
    samples.W(:,sample_idx)   = paramStruct.W;
    samples.H(:,sample_idx)   = paramStruct.H;
    samples.log_post(sample_idx) = resultStruct.log_post;
    samples.log_like(sample_idx) = resultStruct.log_like;
    ll_trs(:,sample_idx)      = resultStruct.trialLL;
    
    %% print out progress
    if(sample_idx <= 50 || (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == HMC_settings.nTotal || (HMC_settings.verbose && mod(sample_idx,20) == 0))
        if(sample_idx == HMC_settings.nTotal)
            ww = (HMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        msgStr = sprintf('%sHMC step %d / %d (accept per. = %.1f in last %d steps, curr log post = %e, (log like = %e)\n',msgStr_0, sample_idx,HMC_settings.nTotal,mean(samples.accepted(ww))*100,numel(ww),samples.log_post(sample_idx),samples.log_like(sample_idx));
        msgStr = sprintf('%s\tcurrent step size = %e, HMC steps = %d, num HMC early rejects = %d\n', msgStr, HMC_state.stepSize.e, HMC_state.steps, nansum(samples.errors));
        printMsg(textOutputFile,msgStr);
        clear ww;
    
        if(HMC_settings.showPlots)
            %%
            sfigure(figNum);
            clf;
            
            plotSamples_glm(samples, HMC_settings, paramStruct, sample_idx);

            drawnow;
        end
    end
    
    %% adjust sampling rate
    HMC_state.stepSize  = adjustHMCstepSize(sample_idx, HMC_state.stepSize, HMC_settings.stepSize, samples.log_p_accept(sample_idx));
    samples.e(:,sample_idx)   = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];
    
    %% updates the covariance matrix of the parameters & hyperparameters
    if(ismember(sample_idx, HMC_settings.M_est.samples ) ) 
        start_idx = HMC_settings.M_est.first_sample(HMC_settings.M_est.samples == sample_idx);
        ww = start_idx:sample_idx;
        
        if(~HMC_settings.M_est.diagOnly(HMC_settings.M_est.samples == sample_idx))
            %full cov mats (block diag structure for hyperparams and params)
            M_chol.W = chol(inv(cov(samples.W(:,ww)')));
        else
            %diagonal only
            M_chol.W = (1./var(samples.W(:,ww),[],2));
        end
        
        if(~HMC_settings.M_est.diagOnly_hyper(HMC_settings.M_est.samples == sample_idx))
            M_chol.H = chol(inv(cov(samples.H(:,ww)')));
        else
            M_chol.H = (1./var(samples.H(:,ww),[],2));
        end
    end
end


%% finish sampler
summary = struct;
printMsg(textOutputFile,'computing WAIC... ');
ll = ll_trs(:,(HMC_settings.nWarmup+1):sample_idx);
T_n = -logMeanExp(ll,2);
V_n = mean(ll.^2,2) - mean(ll,2).^2;
summary.WAICS = T_n + V_n;
summary.WAIC  = mean(summary.WAICS,1);
printMsg(textOutputFile,'done.\n');    

printMsg(textOutputFile,'computing PSIS-LOO... ');
[summary.PSISLOO,summary.PSISLOOS,summary.PSISLOO_PK] = psisloo(ll');

summary.PSISLOOS = summary.PSISLOOS(:);
summary.PSISLOO_PK = summary.PSISLOO_PK(:);

badSamples = sum(summary.PSISLOO_PK > 0.7);
if(badSamples > 0)
    outputMsg = sprintf('%s\n\tWarning: PSISLOO failed for %d samples! ',outputMsg,badSamples);
    printMsg(textOutputFile,'Warning: PSISLOO failed for %d samples! ',badSamples);
else
    outputMsg = sprintf('%s\n\tPSISLOO diagnostics did not show any failures.',outputMsg);
    printMsg(textOutputFile,'PSISLOO diagnostics did not show any failures. ');
end

ss_all = (HMC_settings.nWarmup+1):HMC_settings.nTotal;
summary.w_mu     = mean(samples.W(:,ss_all),2);
summary.H_mu     = mean(samples.H(:,ss_all),2);
summary.H_median     = median(samples.H(:,ss_all),2);
summary.w_sig    = cov(samples.W(:,ss_all)');
summary.H_sig    = cov(samples.H(:,ss_all)');  

summary.earlyRejects = sum(samples.errors(ss_all));
summary.earlyReject_prc = mean(samples.errors(ss_all));
summary.HMC_state            = HMC_state;
summary.acceptRate   = mean(samples.accepted(ss_all));

printMsg(textOutputFile,'done.\n');  
end


function [] = printMsg(file,str,varargin)

fprintf(file,str,varargin{:});
if(file ~= 1)
    fprintf(str,varargin{:});
end
end