function [samples, summary, MCMC_settings, paramStruct, M_est] = runBarkerMC(obj, MCMC_settings, paramStruct, initializerScale)

figNum = 5;
if(~obj.isOnGPU())
    error('Must load gmlm onto GPU before running HMC!');
end

%% sets up the hmc momentum cov matrices
optStruct = obj.getEmptyOptsStruct(true,false);
optStruct = obj.getParamCount(optStruct);
optStruct.compute_trialLL = true;

%% initialize space for samples
samples.log_p_accept  = nan(2,MCMC_settings.nTotal);
samples.errors        = nan(2,MCMC_settings.nTotal);
samples.accepted      = nan(2,MCMC_settings.nTotal);
samples.e             = nan(2,MCMC_settings.nTotal);

nMH = 0;
for jj = 1:obj.dim_J
    nMH = nMH + obj.Groups(jj).dim_R;
end

samples.MH.accepted      = nan(nMH, MCMC_settings.nTotal);
samples.MH.log_p_accept  = nan(nMH, MCMC_settings.nTotal);

dataType = 'single';
if(~obj.gpuSinglePrecision)
    dataType = 'double';
end

samples.H       = nan(obj.dim_H,MCMC_settings.nTotal,'double');
samples.W       = nan(obj.dim_P,MCMC_settings.nTotal,dataType);
samples.B       = nan(obj.dim_K,obj.dim_P,MCMC_settings.nTotal,dataType);
% samples.trialLL = nan(obj.dim_M,MCMC_settings.nTotal,dataType);
samples.log_post = nan(1,MCMC_settings.nTotal);
samples.log_like = nan(1,MCMC_settings.nTotal,dataType);

for jj = 1:obj.dim_J
    samples.Groups(jj).H = nan(obj.Groups(jj).dim_H,MCMC_settings.nTotal,'double');
    samples.Groups(jj).V = nan(obj.dim_P,obj.Groups(jj).dim_R,MCMC_settings.nTotal,dataType);
    samples.Groups(jj).T = cell(obj.Groups(jj).dim_S,1);
    for ss = 1:obj.Groups(jj).dim_S
         samples.Groups(jj).T{ss} = nan(obj.Groups(jj).dim_T(ss),obj.Groups(jj).dim_R,MCMC_settings.nTotal,dataType);
    end
end

samples.T_1_samples = nan(obj.Groups(1).dim_T(1),MCMC_settings.nTotal);
%% set initial point
if(nargin <= 2 || isempty(paramStruct))
    if(nargin < 4 || isempty(initializerScale))
        initializerScale = 1;
    end
    warning('No initialization point provided: generating random initialization!');
    paramStruct = obj.getRandomParamStruct(initializerScale);
    if(isfield(MCMC_settings, 'map_init') && MCMC_settings.map_init)
        paramStruct = obj.getMLE_MAP(paramStruct, 10);
    end
end


%% initialize HMC state
HMC_state.stepSize.e = MCMC_settings.stepSize.e_0;
HMC_state.stepSize.e_bar = MCMC_settings.stepSize.e_0;
HMC_state.stepSize.x_bar_t = 0;
HMC_state.stepSize.x_t = 0;
HMC_state.stepSize.H_sum = 0;

%% adds the initial point to the samples
resultStruct_0 = obj.computeLL(paramStruct, optStruct, true);
resultStruct   = obj.addLPrior(paramStruct, resultStruct_0);
resultStruct.log_post = resultStruct.log_like_0 + resultStruct.log_prior;
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

samples.trialLL(:,1)  = resultStruct.trialLL;
samples.log_post(1)   = resultStruct.log_post;
samples.log_like(1)   = resultStruct.log_like;
samples.e(:,1)          = MCMC_settings.e_init;
[resultStruct.dW_all, resultStruct.dH_all] = obj.vectorizeResultsStruct(resultStruct, optStruct);
[paramStruct.W_all,   paramStruct.H_all] = obj.vectorizeParamStruct(paramStruct, optStruct);

samples.log_p_accept(1) = log(1);


if(MCMC_settings.showPlots)
    figure(figNum);
    clf;
    drawnow;
end

%%

M_est.mu.W    = paramStruct.W_all;
M_est.mu.H    = paramStruct.H_all;
M_est.sigma.W = eye(optStruct.totalParams);
M_est.sigma.W_chol = ones(optStruct.totalParams,1);%eye(optStruct.totalParams);
M_est.sigma.H = eye(optStruct.totalHyperparams);
M_est.sigma.H_chol = eye(optStruct.totalHyperparams);
M_est.log_e   = log(MCMC_settings.e_init);
M_est.e       = MCMC_settings.e_init;
M_est.log_e2   = log(MCMC_settings.e_init);
M_est.e2       = MCMC_settings.e_init;

%%
start_idx = 2;

fprintf('Starting MCMC for %d samples (%d warmup) with initial log posterior = %e, initial step size = %e\n', MCMC_settings.nTotal, MCMC_settings.nWarmup, samples.log_post(1), M_est.e);
load('C:\Users\latim\gitCode\dmc_orientation\Results\samples_a.mat','vectorizedSamples');

%% run sampler
for sample_idx = start_idx:MCMC_settings.nTotal
    
    err_c = nan(MCMC_settings.N,1);
    err_c2 = nan(MCMC_settings.N,1);
    
    accepted = nan(MCMC_settings.N,1);
    accepted_2 = nan(MCMC_settings.N,1);
    log_alpha = nan(MCMC_settings.N,1);
    log_alpha_2 = nan(MCMC_settings.N,1);
    MH_accepted_c = nan(size(samples.MH.accepted,1),MCMC_settings.N);
    MH_log_p      = nan(size(samples.MH.accepted,1),MCMC_settings.N);
        
    for nn = 1:MCMC_settings.N
        %% run MH rescaling step
        % the decompositions have some directions of unidentifiability in the likelihood (V*T' = (V*R)*(T*R^-1)')
        % These optional steps do some fast MH proposals to quickly move around in that space, and does not require any likelihood computations.
        [paramStruct, MH_accepted_c(:,nn), MH_log_p(:,nn)] = obj.scalingMHStep(paramStruct, MCMC_settings.MH_scale);
        if(~all(~MH_accepted_c(:,nn),'all'))
            resultStruct_0 = obj.computeLL(paramStruct, optStruct, true);
            resultStruct   = obj.addLPrior(paramStruct, resultStruct_0);
            resultStruct.log_post = resultStruct.log_like_0 + resultStruct.log_prior;
            [resultStruct.dW_all, resultStruct.dH_all] = obj.vectorizeResultsStruct(resultStruct, optStruct);
            [paramStruct.W_all,   paramStruct.H_all] = obj.vectorizeParamStruct(paramStruct, optStruct);
        end
    

        %% do Barker proposal with i.i.d. Gaussian
        
        bs = 1:optStruct.totalParams ;%randperm(optStruct.totalParams);
        blockSize = optStruct.totalParams ;
        NB = ceil(optStruct.totalParams / blockSize);
        
        log_alpha_cc = nan(NB,1);
        for ii = 1:NB
            bs_ii = bs(((ii-1)*blockSize + 1):min(optStruct.totalParams, ii*blockSize));
            
%             z_w = (randn(1, numel(bs_ii)) * M_est.sigma.W_chol)' * (M_est.e);
            z_w = (randn(numel(bs_ii),1) .* M_est.sigma.W_chol(bs_ii)) * (M_est.e);

            z_d_w = -z_w .* resultStruct.dW_all(bs_ii);
            log_p_z_w = -z_d_w;
            log_p_z_w(z_d_w < 30) = -log1p(exp(z_d_w(z_d_w < 30)));

            b_w = (log(rand(numel(bs_ii)    ,1)) < log_p_z_w)*2 - 1;

            W_star = paramStruct.W_all;
            W_star(bs_ii) = paramStruct.W_all(bs_ii) + b_w .* z_w;
            paramStruct_star = obj.devectorizeParamStruct(W_star, paramStruct, optStruct, paramStruct.H_all);

            resultStruct_star_0 = obj.computeLL(paramStruct_star, optStruct, true);
            resultStruct_star   = obj.addLPrior(paramStruct_star, resultStruct_star_0);
            resultStruct_star.log_post = resultStruct_star.log_like_0 + resultStruct_star.log_prior;
            [resultStruct_star.dW_all, resultStruct_star.dH_all] = obj.vectorizeResultsStruct(resultStruct_star, optStruct);

            log_p_x_w = (paramStruct.W_all(bs_ii) - W_star(bs_ii)) .* resultStruct.dW_all(bs_ii);
            log_p_x_w(log_p_x_w < 30) = log1p(exp(log_p_x_w(log_p_x_w < 30)));

            log_p_y_w = (W_star(bs_ii) - paramStruct.W_all(bs_ii)) .* resultStruct_star.dW_all(bs_ii);
            log_p_y_w(log_p_y_w < 30) = log1p(exp(log_p_y_w(log_p_y_w < 30)));

            log_alpha_cc(ii) = min(0, resultStruct_star.log_post - resultStruct.log_post + sum(log_p_x_w)- sum(log_p_y_w));

            if(~isinf(log_alpha_cc(ii)) && ~isnan(log_alpha_cc(ii)) && log(rand) < log_alpha_cc(ii))
                paramStruct  = paramStruct_star;
                resultStruct = resultStruct_star;
                resultStruct_0 = resultStruct_star_0;
                accepted(nn) = true;
                err_c(nn) = false;
            else
                if(isinf(log_alpha(nn)) || isnan(log_alpha(nn)))
                    err_c(nn) = true;
                    log_alpha_cc(ii) = log(1e-5);
                else
                    err_c(nn) = false;
                end
                accepted(nn) = false;
            end
        end
        log_alpha(nn) = mean(log_alpha_cc);

        %% do Barker proposal with i.i.d. Gaussian
        z_h = (randn(1, optStruct.totalHyperparams) * M_est.sigma.H_chol)' * (M_est.e2);

        z_d_h = -z_h .* resultStruct.dH_all;
        log_p_z_h = -z_d_h;
        log_p_z_h(z_d_h < 30) = -log1p(exp(z_d_h(z_d_h < 30)));

        b_h = (log(rand(optStruct.totalHyperparams,1)) < log_p_z_h)*2 - 1;

        H_star = paramStruct.H_all + b_h .* z_h;
        paramStruct_star = obj.devectorizeParamStruct(paramStruct.W_all, paramStruct, optStruct, H_star);

        resultStruct_star = obj.addLPrior(paramStruct_star, resultStruct_0);
        resultStruct_star.log_post = resultStruct_star.log_like_0 + resultStruct_star.log_prior;
        [resultStruct_star.dW_all, resultStruct_star.dH_all] = obj.vectorizeResultsStruct(resultStruct_star, optStruct);

        log_p_x_h = (paramStruct.H_all - H_star) .* resultStruct.dH_all;
        log_p_x_h(log_p_x_h < 30) = log1p(exp(log_p_x_h(log_p_x_h < 30)));

        log_p_y_h = (H_star - paramStruct.H_all) .* resultStruct_star.dH_all;
        log_p_y_h(log_p_y_h < 30) = log1p(exp(log_p_y_h(log_p_y_h < 30)));

        log_alpha_2(nn) = min(0, resultStruct_star.log_prior - resultStruct.log_prior + sum(log_p_x_h) - sum(log_p_y_h));

        if(~isinf(log_alpha_2(nn)) && ~isnan(log_alpha_2(nn)) && log(rand) < log_alpha_2(nn))
            paramStruct  = paramStruct_star;
            resultStruct = resultStruct_star;
            accepted_2(nn) = true;
            err_c2(nn) = false;
        else
            if(isinf(log_alpha_2(nn)) || isnan(log_alpha_2(nn)))
                err_c2(nn) = true;
                log_alpha_2(nn) = log(1e-5);
            else
                err_c2(nn) = false;
            end
            accepted_2(nn) = false;
        end
    end
    samples.MH.accepted(:,sample_idx) = nanmean(MH_accepted_c,2);
    samples.MH.log_p_accept(:,sample_idx) = nanmean(MH_log_p,2);
    accepted = mean(accepted);
    accepted_2 = mean(accepted_2);
    
    log_alpha = mean(log_alpha);
    log_alpha_2 = mean(log_alpha_2);
    
    %% save sample
    samples.accepted(1,sample_idx)     = accepted;
    samples.accepted(2,sample_idx)     = accepted_2;
    samples.log_p_accept(1, sample_idx) = log_alpha;
    samples.log_p_accept(2, sample_idx) = log_alpha_2;
    
    samples.errors(1,sample_idx) = nanmean(err_c);
    samples.errors(2,sample_idx) = nanmean(err_c2);
    
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
    
    V_c = samples.Groups(1).V(1,:,sample_idx).* samples.Groups(1).T{1}(:,:,sample_idx);
    for ss = 2:numel(samples.Groups(1).T)
        V_c = V_c .* samples.Groups(1).T{ss}(1,:,sample_idx);
    end
    samples.T_1_samples(:,sample_idx) = sum(V_c,2);

%     samples.trialLL(:,sample_idx)  = resultStruct.trialLL;
    samples.log_post(sample_idx)   = resultStruct.log_post;
    samples.log_like(sample_idx)   = resultStruct.log_like;
    
    samples.e(1,sample_idx)   = M_est.e;
    samples.e(2,sample_idx)   = M_est.e2;
    
    
    %% adjust step size
    HMC_state.stepSize  = adjustHMCstepSize(sample_idx, HMC_state.stepSize, MCMC_settings.stepSize, log_alpha);
    %samples.e(:,sample_idx)   = [HMC_state.stepSize.e; HMC_state.stepSize.e_bar];
    
    %% update estimates
    if(sample_idx <= MCMC_settings.M_est.end)
        gamma_t = ((sample_idx/MCMC_settings.M_est.gamma)^(-MCMC_settings.M_est.kappa)) / MCMC_settings.M_est.gamma;
        
        %M_est.log_e  = M_est.log_e  + gamma_t * expm1(log_alpha   - log(MCMC_settings.M_est.alpha))*MCMC_settings.M_est.alpha;
        M_est.log_e2 = M_est.log_e2 + gamma_t * expm1(log_alpha_2 - log(MCMC_settings.M_est.alpha))*MCMC_settings.M_est.alpha;
        
        M_est.mu.H = M_est.mu.H + gamma_t * (paramStruct.H_all - M_est.mu.H);
        M_est.mu.W = M_est.mu.W + gamma_t * (paramStruct.W_all - M_est.mu.W);
        
        M_est.sigma.H = M_est.sigma.H + gamma_t * ((paramStruct.H_all - M_est.mu.H)*(paramStruct.H_all - M_est.mu.H)' - M_est.sigma.H);
        M_est.sigma.W = M_est.sigma.W + gamma_t * ((paramStruct.W_all - M_est.mu.W)*(paramStruct.W_all - M_est.mu.W)' - M_est.sigma.W);
        M_est.sigma.H_chol = chol(M_est.sigma.H);
        
        if(sample_idx >= 200)
            M_est.sigma.W_chol = std(vectorizedSamples(:,101:end),[],2);%max(5e-2, sqrt(diag(M_est.sigma.W)));%chol(M_est.sigma.W);
        else
            M_est.sigma.W_chol = ones(optStruct.totalParams,1);
        end
        
        if(isinf(exp(M_est.log_e)) || isnan(exp(M_est.log_e)))
            fprintf('fuck\n');
        end
        %M_est.e  = exp(M_est.log_e);
        M_est.e2 = exp(M_est.log_e2);
        
        M_est.e = HMC_state.stepSize.e;
    else
        M_est.e = HMC_state.stepSize.e_bar;
    end
    
    %% print any updates
    if( (sample_idx <= 500 && mod(sample_idx,20) == 0) ||  mod(sample_idx,50) == 0 || sample_idx == MCMC_settings.nTotal || (MCMC_settings.verbose && mod(sample_idx,50) == 0))
        if(sample_idx == MCMC_settings.nTotal)
            ww = (MCMC_settings.nWarmup+1):sample_idx;
        else
            ww = max(2,sample_idx-99):sample_idx;
        end
        
        mean_MH_accepted = nanmean(samples.MH.accepted(:,ww),'all');
        mean_accepted = nanmean(samples.accepted(:,ww),2)*100;
        
        fprintf('HMC step %d / %d (accept per. = %.1f, %.1f in last %d steps, curr log post = %e, (log like = %e)\n', sample_idx, MCMC_settings.nTotal, mean_accepted(1), mean_accepted(2), numel(ww), samples.log_post(sample_idx),samples.log_like(sample_idx));
        fprintf('\tcurrent step size = %e, mean MH accepted = %.3f\n', M_est.e, mean_MH_accepted);
        clear ww;
        
        if(MCMC_settings.showPlots)
            sfigure(figNum);
            plotSamples_gmlm(samples, MCMC_settings, paramStruct, sample_idx, samples.T_1_samples);
            drawnow;
        end
    end
end


%% finish sampler
% fprintf('computing WAIC... ');
% ll = samples.trialLL(:,(MCMC_settings.nWarmup+1):sample_idx);
% 
% T_n = zeros(size(ll,1),1);
% V_n = zeros(size(ll,1),1);
% for ii = 1:size(ll,1)
%     ll_c = double(ll(ii,:));
%     T_n(ii) = -logMeanExp(ll_c,2);
%     V_n(ii) = mean(ll_c.^2,2) - mean(ll_c,2).^2;
% end
% summary.WAICS = T_n + V_n;
% summary.WAIC  = mean(summary.WAICS,1);
% fprintf('done.\n');  
% 
% fprintf('computing PSIS-LOO... ');
% [summary.PSISLOO,summary.PSISLOOS,summary.PSISLOO_PK] = psisloo(ll');
% 
% summary.PSISLOOS   = summary.PSISLOOS(  :);
% summary.PSISLOO_PK = summary.PSISLOO_PK(:);
% 
% badSamples   = sum(summary.PSISLOO_PK >= 0.7);
% if(badSamples > 0)
%     fprintf('Warning: PSISLOO PK large (>0.7) for %d / %d observations!',badSamples, numel(summary.PSISLOO_PK ));
% else
%     fprintf('PSISLOO diagnostics passed (all PK < 0.7). ');
% end

ss_all = (MCMC_settings.nWarmup+1):MCMC_settings.nTotal;
summary.earlyRejects = sum(samples.errors(ss_all));
summary.earlyReject_prc = mean(samples.errors(ss_all));
summary.HMC_state            = HMC_state;
summary.acceptRate   = mean(samples.accepted(ss_all));

fprintf('done.\n');   

end
