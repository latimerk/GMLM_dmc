%% GMLM.computeMAPevidenceOptimization
%   optimizes the log-evidence (plus log hyperprior) of the GMLM using quasi-newton optimization & Laplace approximation for the evidence
%
%   inputs:
%       params_init (OPTIONAL) :a param struct of the initial value of the optimization. If not given or empty, a random initialization point will be generated by GMLM.getRandomParamStruct
%
%       optional key/value pairs
%
%           max_steps        (default = 10000)             : iters for the trust-region optimization
%           display          (default = 'off')             : display settings for fminunc. If 'off', it will only display progress between each attempt at running fminunc
%           opt_step_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           opt_func_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           trial_weights      (default = empty)           : weighting of each trial for the optimization (useful for cross validation)
%           fitAll             (default = true)            : to fit a final model to all training data
%
%   outputs:
%       params_mle  : the parameter struct at the optimal value
%       results_mle : the final results struct at params_mle
%       
function [params_map, results_test_map, results_train_map] = computeMAPevidenceOptimization_crossValidated(obj, foldIDs,  varargin)

    p = inputParser;
    p.CaseSensitive = false;

    addRequired( p, 'foldIDs'         ,   @(aa)((isa(aa, 'cvpartition') && aa.NumObservations == 100) || (numel(aa) == obj.dim_M)));
    addOptional( p, 'params_init'     ,    [], @(aa)(isempty(aa) | isstruct(aa)));
    addParameter(p, 'max_steps'       ,  10e3, @isnumeric);
    addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'display'         , 'none', @(aa)(isstring(aa) | ischar(aa)));
    addParameter(p, 'fitAll'     ,  true, @islogical);

    parse(p, foldIDs, varargin{:});
    % then set/get all the inputs out of this structure
    params_init      = p.Results.params_init;
    max_steps        = p.Results.max_steps;
    opt_step_tolerance = p.Results.opt_step_tolerance;
    opt_func_tolerance = p.Results.opt_func_tolerance;
    display = p.Results.display;
    fitAll = p.Results.fitAll;
    
    %% CV info
    if(isa(foldIDs, 'cvpartition'))
        K = foldIDs.NumTestSets;
    else
        uds = unique(foldIDs);
        uds = uds(~isnan(uds));
        K = numel(uds);
    end

    %% sets up initial params
    if(isempty(params_init))
        params_init = repmat(obj.getRandomParamStruct('includeHyperparameters', true), [K+fitAll 1]);
        for kk = 2:(K + fitAll)
            params_init(kk) = obj.getRandomParamStruct('includeHyperparameters', true);
        end
    else
        if(numel(params_init) == 1)
            params_init = repmat(params_init, [K+fitAll 1]);
        elseif(numel(params_init) ~= K + fitAll)
            error('initial parameters for CV must contain 1 parameter (for all folds), or a specific initial point for each fold.');
        end
    end

    %% for each fold
    for kk = 1:(K + fitAll)
        if(kk <= K)
            fprintf('Fitting fold %d / %d...\n', kk, K);
        else
            fprintf('Fitting final model with all training data...\n');
        end

        %sets weights for training trials to 1, all other trials to 0
        if(isa(foldIDs, 'cvpartition'))
            if(kk > foldIDs.NumTestSets)
                trial_weights_train = [];
            else
                trial_weights_train = zeros(obj.dim_M, 1);
                trial_weights_test  = zeros(obj.dim_M, 1);
                trial_weights_train(foldsIDs.training(kk)) = 1;
                trial_weights_test(foldsIDs.test(kk)) = 1;
            end
        else
            trial_weights_train = zeros(obj.dim_M, 1);
            trial_weights_test  = zeros(obj.dim_M, 1);
            if(kk > numel(uds))
                trial_weights_train(~isnan(foldIDs)) = 1;
            else
                trial_weights_train(foldIDs ~= uds(kk) & ~isnan(foldIDs)) = 1;
                trial_weights_test( foldIDs == uds(kk) & ~isnan(foldIDs)) = 1;
            end
        end

        %runs optimization
        [params_map(kk), results_train_map(kk)] = obj.computeMAPevidenceOptimization('params_init', params_init(kk),  ...
                                        'max_steps', max_steps,  ...
                                        'opt_step_tolerance',  opt_step_tolerance, 'opt_func_tolerance', opt_func_tolerance, ...
                                        'display'         , display,  ...
                                        'trial_weights',   trial_weights_train); %#ok<*AGROW>
        results_train_map(kk).trialLL(trial_weights_train ~= 1) = nan;

        %gets test log likelihood
        if(kk <= K)
            opts_empty  = obj.getComputeOptionsStruct(false, 'trial_weights', true, 'includeHyperparameters', false);
            opts_empty.trial_weights(:) = trial_weights_test;
            results_test_map(kk) = obj.computeLogLikelihood(params_map(kk), opts_empty);
            results_test_map(kk).trialLL(trial_weights_test ~= 1) = nan;
        end
    end
    fprintf('Done fitting cross-validation sets.\n');
end