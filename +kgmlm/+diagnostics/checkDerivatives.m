%% finite difference tool for all GMLM parameters
% can check with stochastic gradients by giving a "numberOfTrialsForSGrun" (weights per trial will be random)
function [results_est, results_all, ll_host, params, opts_all] = checkDerivatives(gmlm, params, numberOfTrialsForSGrun, perNeuronWeights, use_posterior)

if(nargin < 2 || isempty(params))
    params     = gmlm.getRandomParamStruct();
end
opts_empty = gmlm.getComputeOptionsStruct(false);
opts_all   = gmlm.getComputeOptionsStruct(true);

if(nargin > 2 && ~isempty(numberOfTrialsForSGrun) && numberOfTrialsForSGrun > 0)
    nz = randperm(gmlm.dim_M, numberOfTrialsForSGrun);
    if(nargin < 4 || isempty(perNeuronWeights) || ~perNeuronWeights || ~gmlm.isSimultaneousPopulation)
        opts_empty.trial_weights = zeros(gmlm.dim_M,1);
        ws = rand(numberOfTrialsForSGrun,1)*2;
        opts_empty.trial_weights(nz) = ws;
    else
        opts_empty.trial_weights = zeros(gmlm.dim_M, gmlm.dim_P);
        ws = rand(numberOfTrialsForSGrun,gmlm.dim_P)*2;
        opts_empty.trial_weights(nz, :) = ws;
    end
    opts_all.trial_weights = opts_empty.trial_weights;
end

if(~gmlm.gpuDoublePrecision && ~isempty(opts_all.trial_weights))
    opts_all.trial_weights = single(opts_all.trial_weights);
    opts_empty.trial_weights = single(opts_empty.trial_weights);
end

if(nargin < 5 || isempty(use_posterior) || (~isnan(use_posterior) && use_posterior))
    lfun = @(p,o) gmlm.computeLogPosterior(p, o);
    rf = "log_post";
elseif(nargin >= 5 && ~isempty(use_posterior) && isnan(use_posterior))
    lfun = @(p,o) gmlm.computeLogPrior(p, o);
    rf = "log_prior";
else
    lfun = @(p,o) gmlm.computeLogLikelihood(p, o);
    rf = "log_likelihood";
end

results_all =  lfun(params, opts_all);

% results_host = gmlm.computeLogLikelihood(params, opts_empty, [], true);
% ll_host = results_host.trialLL(:);
% ll_host = ll_host(:);
[~,lls_host] = gmlm.computeLogLikelihoodHost(params, opts_empty);
ll_host = cell2mat({lls_host(:).log_like}');
ll_host = ll_host(:);

if(gmlm.gpuDoublePrecision)
    dx = 1e-5;
else
    dx = 1e-3;
end
cs  = [-1/60 3/20 -3/4 3/4 -3/20 1/60]; %centered FD coefficients
xxs = [-3:-1 1:3]*dx;

results_est = results_all;

for ww = 1:numel(params.W)
    dd = zeros(numel(xxs),1);
    for ii = 1:numel(xxs)
        params_c = params;
        params_c.W(ww) = params_c.W(ww) + xxs(ii);
%         try
            results_c = lfun(params_c, opts_empty);
%         catch
%             fprintf("error!\n");
%         end
        dd(ii) = results_c.(rf);
    end
    results_est.dW(ww) = (cs*dd)./dx;
end
for ww = 1:numel(params.B)
    dd = zeros(numel(xxs),1);
    for ii = 1:numel(xxs)
        params_c = params;
        params_c.B(ww) = params_c.B(ww) + xxs(ii);
        results_c = lfun(params_c, opts_empty);
        dd(ii) = results_c.(rf);
    end
    results_est.dB(ww) = (cs*dd)./dx;
end
if(isfield(params, 'H') && ~isempty(params.H))
    for ww = 1:numel(params.H)
        dd = zeros(numel(xxs),1);
        for ii = 1:numel(xxs)
            params_c = params;
            params_c.H(ww) = params_c.H(ww) + xxs(ii);
            results_c = lfun(params_c, opts_empty);
            dd(ii) = results_c.(rf);
        end
        results_est.dH(ww) = (cs*dd)./dx;
    end
end
for jj = 1:numel(params.Groups)
    for ww = 1:numel(params.Groups(jj).V)
        dd = zeros(numel(xxs),1);
        for ii = 1:numel(xxs)
            params_c = params;
            params_c.Groups(jj).V(ww) = params_c.Groups(jj).V(ww) + xxs(ii);
            results_c = lfun(params_c, opts_empty);
            dd(ii) = results_c.(rf);
        end
        results_est.Groups(jj).dV(ww) = (cs*dd)./dx;
    end
    
    for ss = 1:numel(params.Groups(jj).T)
        for ww = 1:numel(params.Groups(jj).T{ss})
            dd = zeros(numel(xxs),1);
            for ii = 1:numel(xxs)
                params_c = params;
                params_c.Groups(jj).T{ss}(ww) = params_c.Groups(jj).T{ss}(ww) + xxs(ii);
                results_c = lfun(params_c, opts_empty);
                dd(ii) = results_c.(rf);
            end
            results_est.Groups(jj).dT{ss}(ww) = (cs*dd)./dx;
        end
    end
    
    if(isfield(params.Groups(jj), 'H') && ~isempty(params.Groups(jj).H))
        for ww = 1:numel(params.Groups(jj).H)
            dd = zeros(numel(xxs),1);
            for ii = 1:numel(xxs)
                params_c = params;
                params_c.Groups(jj).H(ww) = params_c.Groups(jj).H(ww) + xxs(ii);
                results_c = lfun(params_c, opts_empty);
                dd(ii) = results_c.(rf);
            end
            results_est.Groups(jj).dH(ww) = (cs*dd)./dx;
        end
    end
end