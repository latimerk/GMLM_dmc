%% GMLM.computeMAP
%   optimizes the log-posterior of the GMLM given fixed hyperparameters using quasi-newton optimization
%
%   inputs:
%       params_init (REQUIRED) :a param struct of the initial value of the optimization.
%
%       optional key/value pairs
%
%           alternating_opt  (true/false; default = false) : if true, maximizes posterior by alternating between parameters (each conditional will effictively be a GLM)
%           max_quasinewton_steps        (default = 10000)             : iters for the quasi-newton optimization
%           max_iters (default = 4)                 : times to run fminunc (note: sometimes restarting this algorithm will help it go further - annoying by it happens)
%           display          (default = 'off')             : display settings for fminunc. If 'off', it will only display progress between each attempt at running fminunc
%           opt_step_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           opt_func_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           trial_weights      (default = empty)           : weighting of each trial for the optimization (useful for cross validation)
%
%   outputs:
%       params_map  : the parameter struct at the optimal value
%       results_map : the final results struct at params_mle
%       
function [params_map, results_map, hess_est] = computeMAP(obj, params_init, varargin)

    p = inputParser;
    p.CaseSensitive = false;

    addRequired( p, 'params_init'     ,    @(aa)obj.verifyParamStruct(aa, 'includeHyperparameters', true));
    addParameter(p, 'alternating_opt' , false, @islogical);
    addParameter(p, 'max_quasinewton_steps'       ,  10e3, @isnumeric);
    addParameter(p, 'max_iters',     4, @isnumeric);
    addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'display'         , 'off', @(aa)(isstring(aa) | ischar(aa)));
    addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));
    addParameter(p, 'optStruct' ,   [], @(aa) isempty(aa) | obj.verifyComputeOptionsStruct(aa));
    addParameter(p, 'optHyperparams'         , true, @islogical);

    parse(p, params_init, varargin{:});
    % then set/get all the inputs out of this structure
    params_init      = p.Results.params_init;
    alternating_opt  = p.Results.alternating_opt;
    optStruct  = p.Results.optStruct;
    max_quasinewton_steps        = p.Results.max_quasinewton_steps;
    max_iters = p.Results.max_iters;
    optHyperparams = p.Results.optHyperparams;
    
    %% fminunc settings
    fminunc_opts = optimoptions(@fminunc, 'algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, 'Hessian', 'off', 'HessUpdate', 'bfgs');
    fminunc_opts.MaxIterations     = max_quasinewton_steps;
    fminunc_opts.StepTolerance     = p.Results.opt_step_tolerance;
    fminunc_opts.FunctionTolerance = p.Results.opt_func_tolerance;
    fminunc_opts.Display           = p.Results.display;
    
    %% setup optimization groups
    [optSetup, opts_empty] = obj.getOptimizationSettings( alternating_opt, p.Results.trial_weights, optStruct, optHyperparams);
    
    %% print inital log post value
    start_time = tic;
    params_map  = params_init;
    results_map = obj.computeLogPosterior(params_map, opts_empty);
    fprintf('Starting MAP optimization. log post = %.5e\n', results_map.log_post);
    results_0 = cell(numel(optSetup),1);
    
    %% do optimization
    for aa = 1:max_iters
        %% each opt setup
        for ss = 1:numel(optSetup)
            start_time_part = tic;
            w_init = obj.vectorizeParams(params_map, optSetup(ss));

            if(aa == 1)
                results_0{ss} = obj.getEmptyResultsStruct(optSetup(ss));
            end
            nllFunc = @(ww)obj.vectorizedNLPost_func(ww, params_map, optSetup(ss), results_0{ss});
            try
                if(aa == max_iters && nargout > 2)
                    [w_fit, fval, ~, ~, ~, hess_est]  = fminunc(nllFunc, w_init, fminunc_opts);
                else
                    [w_fit, fval]  = fminunc(nllFunc, w_init, fminunc_opts);
                end
            catch
                fprintf('optimization failure! Trying a different fminunc setting.\n');
                fminunc_opts2 = fminunc_opts;
                fminunc_opts2.MaxIterations = ceil(fminunc_opts2.MaxIterations/5);
                if(aa == max_iters && nargout > 2)
                    [w_fit, fval, ~, ~, ~, hess_est]  = fminunc(nllFunc, w_init, fminunc_opts2);
                else
                    [w_fit, fval, ~, ~, ~, hess_est]  = fminunc(nllFunc, w_init, fminunc_opts2);
                end
            end
            params_map = obj.devectorizeParams(w_fit, params_map, optSetup(ss));
            
            if(numel(optSetup) > 1)
                fprintf('\t\t part %d / %d,\tlog post = %.5e,\tpart fit time %.1f seconds\n', ss, numel(optSetup), -fval, toc(start_time_part));
            end
        end
        
        %% print progress
        results_map = obj.computeLogPosterior(params_map, opts_empty);
        fprintf('\t iter %d / %d,\tlog post = %.5e,\ttime elapsed %.1f seconds\n', aa, max_iters, results_map.log_post, toc(start_time));
        
    end
    fprintf('done.\n');
end