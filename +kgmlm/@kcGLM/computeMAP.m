%% GMLM.computeMAP
%   optimizes the log-posterior of the GMLM given fixed hyperparameters using quasi-newton optimization
%
%   inputs:
%       params_init (REQUIRED) :a param struct of the initial value of the optimization.
%
%       optional key/value pairs
%
%           max_steps        (default = 10000)             : iters for the trust-region optimization
%           display          (default = 'off')             : display settings for fminunc. If 'off', it will only display progress between each attempt at running fminunc
%           opt_step_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           opt_func_tolerance (default = 1e-6)            : tolerance settings for fminunc
%           trial_weights      (default = empty)           : weighting of each trial for the optimization (useful for cross validation)
%
%   outputs:
%       params_map  : the parameter struct at the optimal value
%       results_map : the final results struct at params_mle
%       
function [params_map, results_map] = computeMAP(obj, params_init, varargin)

    p = inputParser;
    p.CaseSensitive = false;

    addRequired( p, 'params_init'     ,    @(aa)obj.verifyParamStruct(aa, 'includeHyperparameters', true));
    addParameter(p, 'max_steps'       ,  10e3, @isnumeric);
    addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'display'         , 'off', @(aa)(isstring(aa) | ischar(aa)));
    addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

    parse(p, params_init, varargin{:});
    % then set/get all the inputs out of this structure
    params_init      = p.Results.params_init;
    max_steps        = p.Results.max_steps;
    trial_weights        = p.Results.trial_weights;
    display = p.Results.display;
    
    %% fminunc settings
    fminunc_opts = optimoptions(@fminunc, 'algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, 'Hessian', 'on');
    fminunc_opts.MaxIterations     = max_steps;
    fminunc_opts.StepTolerance     = p.Results.opt_step_tolerance;
    fminunc_opts.FunctionTolerance = p.Results.opt_func_tolerance;
    if(~strcmpi(display, 'none'))
        fminunc_opts.Display           = display;
    else
        fminunc_opts.Display           = 'off';
    end
    
    optSetup   = obj.getComputeOptionsStruct(true , 'includeHyperparameters', false, 'trial_weights', trial_weights);
    opts_empty = obj.getComputeOptionsStruct(false, 'includeHyperparameters', false, 'trial_weights', trial_weights);
    
    %% print inital log post value
    start_time = tic;
    params_map  = params_init;
    results_map = obj.computeLogPosterior(params_map, opts_empty);
    if(~strcmpi(display, 'none'))
        fprintf('Starting MAP optimization. log post = %.5e\n', results_map.log_post);
    end
    
    %% do optimization
    w_init = obj.vectorizeParams(params_map, optSetup);
    nlpostFunc = @(ww)obj.vectorizedNLPost_func(ww, params_map, optSetup);
    w_fit  = fminunc(nlpostFunc, w_init, fminunc_opts);
    params_map = obj.devectorizeParams(w_fit, params_map, optSetup);

    %% print progress
    results_map = obj.computeLogPosterior(params_map, opts_empty);
        
    if(~strcmpi(display, 'none'))
        fprintf('done,\tlog post = %.5e,\t total time %.1f seconds.\n', results_map.log_post, toc(start_time));
    end
end