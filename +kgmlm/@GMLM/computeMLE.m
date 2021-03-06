%% GMLM.computeMLE
%   optimizes the log-likelihood of the GMLM using quasi-newton optimization
%
%   inputs:
%       params_init (OPTIONAL) :a param struct of the initial value of the optimization. If not given or empty, a random initialization point will be generated by GMLM.getRandomParamStruct
%
%       optional key/value pairs
%
%           alternating_opt  (true/false; default = false) : if true, maximizes likelihood by alternating between parameters (each conditional will effictively be a GLM)
%           renormalize      (true/false; default = true ) : if true, does a renormalization opt on the tensor parameters (makes columns of each Groups.T{ss} normal)
%           max_quasinewton_steps        (default = 10000)             : iters for the quasi-newton optimization
%           max_iters (default = 4)                 : times to run fminunc (note: sometimes restarting this algorithm will help it go further - annoying by it happens)
%           display          (default = 'off')             : display settings for fminunc. If 'off', it will only display progress between each attempt at running fminunc
%           opt_step_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           opt_func_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           trial_weights      (default = empty)           : weighting of each trial for the optimization (useful for cross validation)
%
%   outputs:
%       params_mle  : the parameter struct at the optimal value
%       results_mle : the final results struct at params_mle
%       
function [params_mle, results_mle, params_init] = computeMLE(obj,  varargin)

    p = inputParser;
    p.CaseSensitive = false;

    addOptional( p, 'params_init'     ,    [], @(aa)(isempty(aa) | isstruct(aa)));
    addParameter(p, 'alternating_opt' , false, @islogical);
    addParameter(p, 'max_quasinewton_steps'       ,  10e3, @isnumeric);
    addParameter(p, 'max_iters',     4, @isnumeric);
    addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'display'         , 'off', @(aa)(isstring(aa) | ischar(aa)));
    addParameter(p, 'renormalize'     ,  true, @islogical);
    addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

    
    parse(p, varargin{:});
    % then set/get all the inputs out of this structure
    params_init      = p.Results.params_init;
    alternating_opt  = p.Results.alternating_opt;
    max_quasinewton_steps        = p.Results.max_quasinewton_steps;
    max_iters = p.Results.max_iters;
    renormalize = p.Results.renormalize;
    
    
    if(isempty(params_init))
        % generate random init point if requested
        params_init = obj.getRandomParamStruct('includeHyperparameters', false);
    else
        assert(obj.verifyParamStruct(params_init, 'includeHyperparameters', false), "invalid parameter structure for initial value");
    end
    
    %% fminunc settings
    fminunc_opts = optimoptions(@fminunc, 'algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, 'Hessian', 'off','HessUpdate', 'bfgs');
    fminunc_opts.MaxIterations = max_quasinewton_steps;
    fminunc_opts.StepTolerance = p.Results.opt_step_tolerance;
    fminunc_opts.FunctionTolerance = p.Results.opt_func_tolerance;
    fminunc_opts.Display = p.Results.display;
    
    %% setup optimization groups
    [optSetup, opts_empty] = obj.getOptimizationSettings( alternating_opt, p.Results.trial_weights);
    
    %% print inital log likelihood
    start_time = tic;
    params_mle  = params_init;
    if(renormalize)
        params_mle = obj.normalizeTensorParams(params_mle);
    end
    results_mle = obj.computeLogLikelihood(params_mle, opts_empty);
    fprintf('Starting MLE optimization. log like = %.5e\n', results_mle.log_likelihood);
    results_0 = cell(numel(optSetup),1);
    
    %% do optimization
    for aa = 1:max_iters
        %% each opt setup
        for ss = 1:numel(optSetup)
            start_time_part = tic;
            w_init = obj.vectorizeParams(params_mle, optSetup(ss));

            if(aa == 1)
                results_0{ss} = obj.getEmptyResultsStruct(optSetup(ss));
            end
            nllFunc = @(ww)obj.vectorizedNLL_func(ww, params_mle, optSetup(ss), results_0{ss});
            try
                [w_fit, fval]  = fminunc(nllFunc, w_init, fminunc_opts);
            catch
                % occasionally with a lot of BFGS steps, the line search will fail and fminunc throws an error.
                % I'm not sure why this happens, so I catch it and try again with fewer steps.
                fprintf('optimization failure! Trying a different fminunc setting.\n');
                fminunc_opts2 = fminunc_opts;
                fminunc_opts2.MaxIterations = ceil(fminunc_opts2.MaxIterations/5);
                [w_fit, fval]  = fminunc(nllFunc, w_init, fminunc_opts2);
            end
            params_mle = obj.devectorizeParams(w_fit, params_mle, optSetup(ss));
            if(renormalize)
                params_mle = obj.normalizeTensorParams(params_mle);
            end
            
            if(numel(optSetup) > 1)
                fprintf('\t\t part %d / %d,\tlog like = %.5e,\tpart fit time %.1f seconds\n', ss, numel(optSetup), -fval, toc(start_time_part));
            end
        end
        
        %% print progress
        results_mle = obj.computeLogLikelihood(params_mle, opts_empty);
        fprintf('\t iter %d / %d,\tlog like = %.5e,\ttime elapsed %.1f seconds\n', aa, max_iters, results_mle.log_likelihood, toc(start_time));
        
    end
    fprintf('done.\n');
end