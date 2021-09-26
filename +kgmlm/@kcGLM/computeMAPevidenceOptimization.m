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
%
%   outputs:
%       params_mle  : the parameter struct at the optimal value
%       results_mle : the final results struct at params_mle
%       
function [params_map, results_map, params_init] = computeMAPevidenceOptimization(obj,  varargin)

    p = inputParser;
    p.CaseSensitive = false;

    addOptional( p, 'params_init'     ,    [], @(aa)(isempty(aa) | isstruct(aa)));
    addParameter(p, 'max_steps'       ,  10e3, @isnumeric);
    addParameter(p, 'opt_step_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'opt_func_tolerance',  1e-6, @isnumeric);
    addParameter(p, 'display'         , 'iter', @(aa)(isstring(aa) | ischar(aa)));
    addParameter(p, 'trial_weights'   ,  [], @(aa) isempty(aa) | (numel(aa) == obj.dim_M & isnumeric(aa)));

    parse(p, varargin{:});
    % then set/get all the inputs out of this structure
    params_init      = p.Results.params_init;
    max_steps        = p.Results.max_steps;
    trial_weights        = p.Results.trial_weights;
    display = p.Results.display;
    
    if(isempty(params_init))
        % generate random init point if requested
        params_init = obj.getRandomParamStruct('includeHyperparameters', true);
    else
        assert(obj.verifyParamStruct(params_init, 'includeHyperparameters', true), "invalid parameter structure for initial value");
    end
    
    %% fminunc settings
    fminunc_opts = optimoptions(@fminunc, 'algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, 'Hessian', 'off');
    fminunc_opts.MaxIterations = max_steps;
    fminunc_opts.StepTolerance = p.Results.opt_step_tolerance;
    fminunc_opts.FunctionTolerance = p.Results.opt_func_tolerance;
    if(~strcmpi(display, 'none'))
        fminunc_opts.Display           = display;
    else
        fminunc_opts.Display           = 'off';
    end
    
    %% get initial MAP estimate
    start_time = tic;
    [params_map, results_map] = obj.computeMAP(params_init, 'display', 'none', 'trial_weights', trial_weights);
    if(~strcmpi(display, 'none'))
        fprintf('Starting evidence optimization estimate. log post at init point = %.5e\n', results_map.log_post);
    end
    
    %% do optimization
    h_init = params_map.H(:);
    h_fit  = fminunc(@(hh)obj.computeNLEvidence(hh, params_map, trial_weights), h_init, fminunc_opts);
    params_map.H(:) = h_fit;
      
    %% get MAP at final point
    [params_map, results_map] = obj.computeMAP(params_map, 'display', 'none', 'trial_weights', trial_weights);
    
    if(~strcmpi(display, 'none'))    
        fprintf('done,\tlog post = %.5e,\t total time %.1f seconds.\n', results_map.log_post, toc(start_time));
    end
end