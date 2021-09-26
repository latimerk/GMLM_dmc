%% GMLM.computeMLE_crossValidated
%   Fits the MLE using cross validation. Cross-validation is done over trials, not individual bins.
%
%   inputs:
%       foldIDs (cvparition or array of length obj.dim_M) : the folds for cross validation (K folds)
%                   if foldIDs is a vector, specifies the test set for each trial. You can set values to nan to withhold trials completely from the CV (not included in any train or test set)
%                       The number of unique (non-nan) elements of foldIDs is K
%
%       params_init (OPTIONAL) : a param struct of the initial value of the optimization. If not given or empty, a random initialization point will be generated by GMLM.getRandomParamStruct
%                                If params_init is an array (of length K + fitAll), then it specifies an init point for each fit.
%
%
%       optional key/value pairs
%           fitAll           (true/false; default = true)  : if true, also fits a final model using all training data (last element of params_mle and results_train_mle)
%
%           max_steps        (default = 10000)             : iters for the trust-region optimization
%           display          (default = 'off')             : display settings for fminunc. If 'off', it will only display progress between each attempt at running fminunc
%           opt_step_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           opt_func_tolerance (default = 1e-6)            : tolerance settings for fminunc
%
%   outputs:
%       params_mle : fit for each fold (if fitAll is true, last element is fit to all train data)
%       results_test_mle  : results for each fold on the test set.  results_test_mle.trialLL has the log like for each trial in the test set (nan if not in test set)
%       results_train_mle : results for each fold on the train set. results_train_mle.trialLL has the log like for each trial in the train set (nan if not in training set)

function [params_mle, results_test_mle, results_train_mle, params_init] = computeMLE_crossValidated(obj, foldIDs, varargin)


p = inputParser;
p.CaseSensitive = false;

addRequired( p, 'foldIDs'         ,   @(aa)((isa(aa, 'cvpartition') && aa.NumObservations == 100) || (numel(aa) == obj.dim_M)));
addOptional( p, 'params_init'     ,    [], @(aa)(isempty(aa) | isstruct(aa)));
addParameter(p, 'max_steps'       ,  10e3, @isnumeric);
addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
addParameter(p, 'display'         , 'off', @(aa)(isstring(aa) | ischar(aa)));
addParameter(p, 'fitAll'     ,  true, @islogical);

parse(p, foldIDs, varargin{:});

% then set/get all the inputs out of this structure
params_init      = p.Results.params_init;
max_steps        = p.Results.max_steps;
fitAll = p.Results.fitAll;

opt_step_tolerance = p.Results.opt_step_tolerance;
opt_func_tolerance = p.Results.opt_func_tolerance;
display = p.Results.display;

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
    params_init = repmat(obj.getRandomParamStruct('includeHyperparameters', false), [K+fitAll 1]);
    for kk = 2:(K + fitAll)
        params_init(kk) = obj.getRandomParamStruct('includeHyperparameters', false);
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
    [params_mle(kk), results_train_mle(kk)] = obj.computeMLE('params_init', params_init(kk),  ...
                                    'max_steps', max_steps,  ...
                                    'opt_step_tolerance',  opt_step_tolerance, 'opt_func_tolerance', opt_func_tolerance, ...
                                    'display'         , display,  ...
                                    'trial_weights',   trial_weights_train); %#ok<*AGROW>
    results_train_mle(kk).trialLL(trial_weights_train ~= 1) = nan;
    
    %gets test log likelihood
    if(kk <= K)
        opts_empty  = obj.getComputeOptionsStruct(false, 'trial_weights', true, 'includeHyperparameters', false);
        opts_empty.trial_weights(:) = trial_weights_test;
        results_test_mle(kk) = obj.computeLogLikelihood(params_mle(kk), opts_empty);
        results_test_mle(kk).trialLL(trial_weights_test ~= 1) = nan;
    end
end
fprintf('Done fitting cross-validation sets.\n');
