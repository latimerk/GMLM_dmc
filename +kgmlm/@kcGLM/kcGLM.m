%% CLASS kcGLM
%
%  My own GPU-GLM interface. 
%  I don't recommend using a non-standard library for GLMs (like this one).
%  I made this purely to line up nicely with my GMLM code, and to be very fast for evidence optimization.
%
%
%  Package GMLM_dmc for dimensionality reduction of neural data.
%   
%  References
%   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
%   decisions in parietal cortex reflects long-term training history.
%   bioRxiv
%
%  Copyright (c) 2021 Kenneth Latimer
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
classdef kcGLM < handle
    properties (GetAccess = public, SetAccess = private)
        GLMstructure struct
        trials        struct
        bin_size 
        
        logLikeType
        validLogLikeTypes
        
        gpuObj_ptr uint64 %uint64 pointer to the c++ object for the GLM loaded to the GPU
        gpus       uint32 % array for which GPUs are in use
        gpuDoublePrecision logical %if current GPU loading is double precision (if false, then single)

        LL_info % for performing computations on CPU: add a bunch of extra variables to avoid accessing object too much (really slow in MATLAB)
    end
    
    %% Constructor
    methods 
        function obj = kcGLM(GLMstructure, trials, bin_size, logLikeType)
            if(nargin < 2 || nargin > 4)
                error("kcGLM constructor: two struct inputs required (GLMstructure and trials)");
            end
            
            %% check for valid log like type
            obj.validLogLikeTypes = ["poissExp", "sqErr", "truncatedPoissExp", "poissSoftRec"];
            if(nargin < 4)
                logLikeType = obj.validLogLikeTypes(1);
            end
            if(all(~strcmpi(logLikeType, obj.validLogLikeTypes)))
                fprintf("Valid log likelihood types are:\n");
                for ii = 1:numel(obj.validLogLikeTypes)
                    fprintf("\t%s\n", obj.validLogLikeTypes(ii));
                end
                error("Invalid log likelihood type.");
            end
            obj.logLikeType = logLikeType;
            
            %% check the GLM structure
            if(~isstruct(GLMstructure))
                error("GLM constructor: input GLMstructure must be a struct");
            end
            % check for linear dimension (non-negative integer)
            if(~isfield(GLMstructure, "dim_Ks") || ~isnumeric(GLMstructure.dim_Ks) || ~all(fix(GLMstructure.dim_Ks) == GLMstructure.dim_Ks) || ~all(GLMstructure.dim_Ks > 0) || isempty(GLMstructure.dim_Ks))
                error("GLM constructor: GLMstructure must have property dim_Ks (dimensionalty of the linear terms - which can be broken up into pieces: positive integer)");
            end
            dim_J = numel(GLMstructure.dim_Ks);
            
            if(~isfield(GLMstructure, "group_names") || ~isstring(GLMstructure.group_names) || numel(GLMstructure.group_names) ~= dim_J)
                error("GLM constructor: GLMstructure must have string array propery group_names (name of each group of parameters) of the same length as dim_Ks");
            end
                
            %makes sure all group names are unique
            if(numel(unique(GLMstructure.group_names)) ~= numel(GLMstructure.group_names))
                error("GLM constructor: parameter groups must have unique names");
            end
            
            %if contains a prior, should be empty or a function
            if(isfield(GLMstructure, "prior"))
                if(~isempty(GLMstructure.prior))
                    if(~isstruct(GLMstructure.prior) || ~isfield(GLMstructure.prior, 'log_prior_func') || ~isfield(GLMstructure.prior, 'dim_H'))
                        error("GLM constructor: GLMstructure.prior must be empty or structure with fields 'log_prior_func' and 'dim_H'");
                    end

                    if(~isa(GLMstructure.prior.log_prior_func,'function_handle'))
                        error("GLM constructor: GLMstructure.prior.log_prior_func must be a function handle");
                    end

                    if(~isscalar(GLMstructure.prior.dim_H) || GLMstructure.prior.dim_H < 0 || fix(GLMstructure.prior.dim_H) ~= GLMstructure.prior.dim_H)
                        error("GLM constructor: GLMstructure.prior.dim_H must be a non-negative integer");
                    end
                end
            end
            
            
            %% check each trial to see if it matches the structure
            if(~isstruct(trials) || isempty(trials))
                error("GLM constructor: input trials must be non-empty struct array");
            end
            % check for spike observations
               %exists                      for each trial     vector           numbers        contains obs.     is integers (spike counts -> this could be relaxed for more general models)
            if(~isfield(trials, "Y") || ~all(arrayfun(@(aa) isvector(aa.Y) & isnumeric(aa.Y) & ~all(aa.Y < 0, 'all') & all(fix(aa.Y) == aa.Y, 'all'), trials), 'all'))
                error("GLM constructor: trials must have spike count vector Y, which contains only integers (may contain negatives to indicate censored bins, but each trial needs at least one observation!)");
            end
            
            %check for linear terms
            if(~isfield(trials, "X"))
                error("GLM constructor: trials requires field X!");
            end
            
            for mm = 1:numel(trials)
                X = trials(mm).X;
                if(dim_J == 1 && ismatrix(X))
                    X = {X};
                end
                if(iscell(X) && numel(X) == dim_J)
                    for jj = 1:dim_J
                        if(~isnumeric(X{jj}) || ~ismatrix(X{jj})|| size(X{jj},1) ~= numel(trials(mm).Y) || size(X{jj},2) ~= GLMstructure.dim_Ks(jj))
                            error("GLM constructor: each trials(jj).X should be a cell array containing matrices JJ of size numel(trials(jj).Y) x GLMstructure.dim_Ks(jj)");
                        end
                    end
                else
                    error("GLM constructor: each trials(jj).X should be a cell array containing matrices JJ of size numel(trials(jj).Y) x GLMstructure.dim_Ks(jj)");
                end
                
                trials(mm).X = X;
            end
            
            %% check bin_size
            if(nargin < 3 || isempty(bin_size))
                bin_size = 1;
            end
            if(~isscalar(bin_size) || bin_size <= 0)
                error("Invalid bin size (should be a scalar in units of seconds)");
            end
            obj.bin_size = bin_size;
            
            %% save structs
            obj.GLMstructure = GLMstructure;
            obj.trials = trials;
            
            % check for prior, otherwise is null by default
            if(isfield(GLMstructure, "prior"))
                obj.GLMstructure.prior = GLMstructure.prior;
            else
                obj.GLMstructure.prior = [];
            end
            
            %% set that no GPU is currently in use
            obj.gpuObj_ptr = 0;
            obj.gpus = [];
            obj.gpuDoublePrecision = true;
        end
    end
    
    %% destructor
    % takes care of clearing any GPU space 
    methods
        function delete(obj)
            if(obj.isOnGPU())
                fprintf('kcGLM destructor: clearing GLM from GPU.\n');
                obj.freeGPU();
            end
        end
    end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for dimension information (Public)
    % mostly to help me with my notation
    methods (Access = public)
        
        function [dd] = dim_J(obj) % number of tensor coefficient groups
            dd = numel(obj.GLMstructure.dim_Ks);
        end
        
        
        function [mm] = dim_M(obj) % dim_M is the number of trials
            mm = numel(obj.trials);
        end
        
        function [nn] = dim_N(obj, tr) % length of each trial
            nn = arrayfun(@(a)numel(a.Y), obj.trials);
            if(nargin > 1)
                nn = nn(tr);
            end
        end
        
        function [kk] = dim_K(obj, idx) % size of each parameter group
            if(nargin < 2 || isempty(idx))
                kk = zeros(obj.dim_J,1);
                for jj = 1:obj.dim_J
                    kk(jj) = obj.dim_K(jj);
                end
            elseif(idx == -1)
                kk = sum(obj.GLMstructure.dim_Ks);
            else
                idx = obj.getGroupIdx(idx);
                kk = obj.GLMstructure.dim_Ks(idx);
            end
        end
        
        function [hh] = dim_H(obj) %number of hyperparams
            %gets number of hyperparams
            if(~isempty(obj.GLMstructure.prior))
                hh = obj.GLMstructure.prior.dim_H;
            else
                hh = 0;
            end
        end
        
        %gets the index of a group by its name
        function [idx] = getGroupIdx(obj, name) % checks out a group index given by "name" to see if it's correct. If it is a string, then it looks for the group index number
            if(isnumeric(name))
                if(name ~= fix(name) || name <= 0 || name > obj.dim_J)
                    error("Invalid group index!")
                end
                idx = name;
                return;
            elseif(~isstring(name) && ~ischar(name))
                error("Invalid group name!")
            end
            
            name = string(name);
            
            idx = nan;
            for jj = 1:obj.dim_J
                if(strcmp(obj.GLMstructure.group_names(jj), name))
                    idx = jj;
                    break;
                end
            end
            
            if(isnan(idx))
                error("Invalid group name!")
            end
        end
        
        function [valid,idx] = isGroupId(obj, name)
            try 
                idx = obj.getGroupIdx(name);
                valid = true;
            catch
                valid = false;
                idx = [];
            end
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating parameter structures (Public)
    methods (Access = public)    
                
        %% METHOD [paramStruct] = getEmptyParamStruct()
        %	Builds a blank structure with all the GLM parameters given all currently included groups.
        %
        %   By default, all values are set to 0, but everything is the correct size.
        %
        %   outpional inputs:
        %       includeHyperparameters (logical; default=TRUE) if including the hyperparams or not (it's less confusing for MLE to not include these at all)
        %       useDoublePrecision (logical; default=obj.gpuDoublePrecision) if using double precision arithmetic on GPU (if false, uses single)
        %                                                           obj.gpuDoublePrecision is by default set to TRUE, but if GPU is loaded with single, this
        %                                                           changes
        %                                                  Hyperparameters are always double as they are used only on the Host and not GPU code
        %
        %	returns:
        %       params [struct]        
        %           A blank (zero-filled) structure of the model parameters.
        %
        %           Fields for model parameters:
        %               Ks [dim_S[jj] x 1]  (cell array) : each entry is a vector for one group of parameters (it kept separated for easier organization)
        %               H [dim_H x 1]       (real) : the hyperparameters for W (if requested)
        %               group_names         (string array) : names of each param group
        %
        function [params, dataType] = getEmptyParamStruct(obj, varargin)
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addParameter(p, 'useDoublePrecision', obj.gpuDoublePrecision, @islogical);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            parse(p,varargin{:});
            
            useDoublePrecision = p.Results.useDoublePrecision;
            includeHyperparameters = p.Results.includeHyperparameters;
            
            %%
            if(useDoublePrecision)
                dataType = 'double';
            else
                dataType = 'single';
            end
            if(includeHyperparameters)
                params.H = zeros(obj.dim_H, 1, 'double');
            end
            
            %sets up top-level params
            params.Ks = cell(obj.dim_J,1);
            params.group_names = obj.GLMstructure.group_names;
            
            %sets up each group
            J = obj.dim_J();
            for jj = 1:J
                params.Ks{jj} = zeros(obj.dim_K(jj), 1, dataType);
            end
        end
        %randomly fills a paramStruct (see getEmptyParamStruct comments for details)
        function [params, dataType] = getRandomParamStruct(obj, varargin)
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addParameter(p, 'useDoublePrecision', obj.gpuDoublePrecision, @islogical);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            parse(p,varargin{:});
            
            useDoublePrecision = p.Results.useDoublePrecision;
            includeHyperparameters = p.Results.includeHyperparameters;
            
            %%
            [params, dataType] = obj.getEmptyParamStruct('includeHyperparameters', includeHyperparameters, 'useDoublePrecision', useDoublePrecision);
            
            if(isfield(params, 'H'))
                params.H(:) = randn(size(params.H));
            end
            
            %for each group
            J = obj.dim_J();
            for jj = 1:J
                params.Ks{jj}(:) = randn(size(params.Ks{jj}));
            end
        end
        
        %checks if a structure is a valid param struct
        %   Does not check datatypes (single or double)
        function [isValid] = verifyParamStruct(obj, params, varargin)
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addRequired(p, 'params', @isstruct);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            parse(p,params,varargin{:});
            
            includeHyperparameters = p.Results.includeHyperparameters;
            %%
            
            isValid = true(size(params));
            for ii = 1:numel(params)

                if(includeHyperparameters)
                    if(~isfield(params(ii), 'H') || ~isvector(params(ii).H) || ~isnumeric(params(ii).H) || numel(params(ii).H) ~= obj.dim_H)
                        isValid(ii) = false;
                        continue;
                    end
                end

                if(~isfield(params(ii), 'Ks') || numel(params(ii).Ks) ~= obj.dim_J || ~iscell(params(ii).Ks))
                    isValid(ii) = false;
                    continue;
                end

                %for each group
                for jj = 1:obj.dim_J
                    if(~isnumeric(params(ii).Ks{jj}) || numel(params(ii).Ks{jj}) ~= obj.dim_K(jj))
                        isValid(ii) = false;
                        continue;
                    end
                end
            end
        end
    end    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for computing rate, log likelihood, and prior on Host (Public)
    % These functions weren't designed to be computationally efficient, but as a readable check for the optimized code.
    % Fitting would take forever using this function (but it's useful for looking at the rates of a fitted model)
    methods (Access = public)
        function [log_rate_per_trial] = computeLogRate(obj, params)
            
            %for each trial
            log_rate_per_trial = struct('log_rate', cell(size(obj.trials)));
            for mm = 1:obj.dim_M
                dim_N = numel(obj.trials(mm).Y);
                G = zeros(dim_N, obj.dim_J);
                
                %add each group
                for jj = 1:obj.dim_J
                    G(:, jj) = obj.trials(mm).X{jj}*params.Ks{jj};
                end
                log_rate_per_trial(mm).log_rate = sum(G,2);
            end
        end
        
        function [log_like, log_like_per_trial] = computeLogLikelihoodHost(obj, params, opts)
            log_like_per_trial = obj.computeLogRate(params);
            
            if(nargin > 2 && isfield(opts, "trial_weights") && numel(opts.trial_weights) == obj.dim_M)
                tw = opts.trial_weights;
            else
                tw = ones(obj.dim_M,1);
            end
            
            ll_idx = 0;
            for ii = 1:numel(obj.validLogLikeTypes)
                if(strcmpi(obj.logLikeType, obj.validLogLikeTypes(ii)))
                    ll_idx = ii;
                    break;
                end
            end
                
            %for each trial
            for mm = 1:obj.dim_M
                if(ll_idx == 1) %poissExp
                    vv = obj.trials(mm).Y >= 0; %check for any censored values
                    rr = log_like_per_trial(mm).log_rate(vv) + log(obj.bin_size);
                    log_like_per_trial(mm).log_like = tw(mm)*(-sum(exp(rr)) + rr'*obj.trials(mm).Y(vv) - sum(gammaln(obj.trials(mm).Y(vv)+1)));
                elseif(ll_idx == 2) %sqErr
                    log_like_per_trial(mm).log_like = -tw(mm)*sum((log_like_per_trial(mm).log_rate(:) - obj.trials(mm).Y(:)).^2);
                elseif(ll_idx == 3) %truncatedPoissExp
                    rr = log_like_per_trial(mm).log_rate + log(obj.bin_size);
                    log_like_per_trial(mm).log_like = kgmlm.utils.truncatedPoiss(rr(:),  obj.trials(mm).Y(:));
                elseif(ll_idx == 4) %poisson sofrec
                    vv = obj.trials(mm).Y >= 0; %check for any censored values
                    rr = kgmlm.utils.softrec(log_like_per_trial(mm).log_rate(vv))*(obj.bin_size);
                    log_like_per_trial(mm).log_like = tw(mm)*(-sum(rr) + log(rr)'*obj.trials(mm).Y(vv) - sum(gammaln(obj.trials(mm).Y(vv)+1)));
                else
                    error("invalid likelihood setting");
                end
            end
            
            log_like = sum([log_like_per_trial(:).log_like], 'all');
        end

        function [results] = computeLogLikelihoodhod_v2(obj, params, opts, results)
            obj.setupComputeStructuresHost();
            if(nargin < 4)
                results = obj.getEmptyResultsStruct(opts);
            end
            
            LL_info_c = obj.LL_info;

            K = cell2mat(params.Ks);
            if(opts.d2K)
                [ll,dll,n_d2ll_2] = LL_info_c.logLikeFun(LL_info_c.X*K, LL_info_c.Y, LL_info_c.bin_size);
            elseif(opts.dK)
                [ll,dll] = LL_info_c.logLikeFun(LL_info_c.X*K, LL_info_c.Y, LL_info_c.bin_size);
                results.dK(:) = LL_info_c.X'*dll;
            else
                ll = LL_info_c.logLikeFun(LL_info_c.X*K, LL_info_c.Y, LL_info_c.bin_size);
            end


            if(opts.trialLL)
                M = size(LL_info_c.dim_N_ranges,1) - 1;
                for mm = 1:M
                    results.trialLL(mm) = sum(ll(LL_info_c.dim_N_ranges(mm):(LL_info_c.dim_N_ranges(mm+1)-1),:),1) + LL_info_c.Y_const(mm);
            
                    if(~isempty(opts.trial_weights))
                        results.trialLL(mm)  = results.trialLL(mm) .* opts.trial_weights(mm);
                    end
                end
            end


            if(opts.dK)
                if(~isempty(opts.trial_weights))
                    dll(LL_info_c.dim_N_ranges(mm):(LL_info_c.dim_N_ranges(mm+1)-1)) = dll(LL_info_c.dim_N_ranges(mm):(LL_info_c.dim_N_ranges(mm+1)-1)) * opts.trial_weights(mm);
                end
                results.dK(:) = LL_info_c.X'*dll;
            end
            if(opts.d2K)
                if(~isempty(opts.trial_weights))
                    n_d2ll_2(LL_info_c.dim_N_ranges(mm):(LL_info_c.dim_N_ranges(mm+1)-1)) = n_d2ll_2(LL_info_c.dim_N_ranges(mm):(LL_info_c.dim_N_ranges(mm+1)-1)) * sqrt(opts.trial_weights(mm));
                end
                X2 = LL_info_c.X.*n_d2ll_2;
                results.d2K(:,:) = -(X2'*X2);
            end
        end

        
        % computes the log prior (as far as this function is aware, this is always on host)
        %   can add the result to a results struct or create a new struct
        function [results] = computeLogPrior(obj, params, opts, results)
            if(nargin < 4)
                results = obj.getEmptyResultsStruct(opts);
            end
            
            results.log_prior = 0;
            
            %% if a prior for W,B exists, adds it
            if(isfield(obj.GLMstructure, "prior") && ~isempty(obj.GLMstructure.prior))
                results = obj.GLMstructure.prior.log_prior_func(params, results);
            end
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating GPU computation option structures (Public)
    methods (Access = public)

        %% METHOD [opts] = getComputeOptionsStruct(enableAll)
        %	Builds a structure for requesting specific derivates of the model parameters
        %
        %   optional inputs:
        %       enableAll (logical; default=TRUE) selects all derivatives (if false, all deselected)
        %       includeHyperparameters (logical; default=TRUE) if including the hyperparams or not (it's less confusing for MLE to not include these at all)
        %       trial_weights (logical; default=false) include full list of weights for each trial (used for SGD). By default, is empty (meaning each weight = 1).
        %
        %	returns:
        %       opts [struct]        
        %           A blank (zero-filled) structure of the model parameters. Everything is of type logical
        %
        %           Fields for derivatives of each of the model parameters:
        %               dK  : derivative
        %               d2K : Hessian 
        %               dH : (if requested)
        %           Note: Group structure is ignored here: computes all or none of the derivatives
        %
        function [opts] = getComputeOptionsStruct(obj, varargin)
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addOptional(p, 'enableAll', true, @islogical);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            addParameter(p, 'trial_weights', false, (@(aa)islogical(aa) || isempty(aa) || numel(aa) == obj.dim_M));
            parse(p,varargin{:});
            
            enableAll = p.Results.enableAll;
            includeHyperparameters = p.Results.includeHyperparameters;
            trial_weights = p.Results.trial_weights;
            
            %%
            %sets up top-level params
            opts.trialLL = true;
            opts.dK = enableAll;
            opts.d2K = enableAll;
            
            if(includeHyperparameters)
                opts.dH = true;
            else
                opts.dH = false;
            end
            
            opts.trial_weights = [];
            if(~isempty(trial_weights))
                if(numel(trial_weights) == obj.dim_M)
                    opts.trial_weights = trial_weights(:);
                elseif(islogical(trial_weights) && trial_weights)
                    opts.trial_weights = ones(obj.dim_M, 1);
                end
            end
            
            
            if(~obj.gpuDoublePrecision)
                opts.trial_weights = single(opts.trial_weights);
            end
        end
        %checks if a structure is a valid compute options struct
        %   Does not check datatypes (single or double)
        %   optional inputs:
        %       includeHyperparameters (logical; default=TRUE) if including the hyperparams or not 
        function [isValid] = verifyComputeOptionsStruct(obj, opts, varargin)
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addRequired(p, 'opts', @isstruct);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            parse(p,opts,varargin{:});
            
            includeHyperparameters = p.Results.includeHyperparameters;
            %%
            
            isValid = true;
            if(~isfield(opts, 'dK') || ~isfield(opts, 'd2K'))
                isValid = false;
                return;
            end
            
            if(~isscalar(opts.dK) || ~islogical(opts.dK))
                isValid = false;
                return;
            end
            
            if(~isscalar(opts.d2K) || ~islogical(opts.d2K))
                isValid = false;
                return;
            end
            
            if(~isfield(opts, 'trial_weights'))
                isValid = false;
                return;
            end
            if(~isempty(opts.trial_weights) && (~isnumeric(opts.trial_weights) || ~isvector(opts.trial_weights) || numel(opts.trial_weights) ~= obj.dim_M))
                isValid = false;
                return;
            end
            
            if(includeHyperparameters)
                if(~isfield(opts, 'dH') || ~isscalar(opts.dH) || ~islogical(opts.dH))
                    isValid = false;
                    return;
                end
            end
        end
    end    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for vectorizing the set of parameters
    methods (Access = public)
        %  opts is a computeOptions struct
        %       Here it is used, not to select derivatives, but to select which parameters to vectorize
        function [ww] = vectorizeParams(obj, params, opts)
            dks = obj.dim_K(-1);
            ww = zeros(dks + obj.dim_H(), 1); %pre-allocates a vector large enough to store everything possible
            ctr = 0; % counter to keep place in ww
            
            if(opts.dK || opts.d2K)
                for jj = 1:numel(params.Ks)
                    params.Ks{jj} = params.Ks{jj}(:);
                end
                ww((1:dks)) = cell2mat(params.Ks(:));
                ctr = ctr + dks;
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(params, 'H') && opts.dH)
                ss_c = numel(params.H);
                ww(ctr+(1:ss_c)) = params.H(:);
                ctr = ctr + ss_c;
            end
            
            ww = ww(1:ctr); %cuts down vector to the pieces that were filled
        end
        
        % given a vectorized set of params for opts in ww, and a full param struct, devectorizes
        function [params] = devectorizeParams(obj, ww, params, opts)
            ctr = 0; % counter to keep place in ww
            
            if(opts.dK || opts.d2K)
                for jj = 1:obj.dim_J
                    params.Ks{jj}(:) = ww(ctr+(1:obj.dim_K(jj)));
                    ctr = ctr + obj.dim_K(jj);
                end
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(params, 'H') && opts.dH)
                ss_c = numel(params.H);
                params.H(:) = ww(ctr+(1:ss_c));
                ctr = ctr + ss_c;
            end
        end
        
        % given a vectorized set of params for opts in ww, and a full param struct, devectorizes
        function [ww] = vectorizeResults(obj, results, opts)
            dks = obj.dim_K(-1);
            ww = zeros(dks + obj.dim_H(), 1); %pre-allocates a vector large enough to store everything possible
            ctr = 0; % counter to keep place in ww
            
            if(opts.dK || opts.d2K)
                ss_c = numel(results.dK);
                ww(ctr+(1:ss_c)) = results.dK(:);
                ctr = ctr + ss_c;
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(results, 'dH') && opts.dH)
                ss_c = numel(results.dH);
                ww(ctr+(1:ss_c)) = results.dH(:);
                ctr = ctr + ss_c;
            end
            
            ww = ww(1:ctr); %cuts down vector to the pieces that were filled
        end
    end    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for accessing GPU object (Public)
    methods (Access = public)    
        %% toGPU(deviceNumbers, varargin)
        %   Sends all the data to the GPU and sets up all the computational space
        %
        %   inputs:
        %       deviceNumbers (int) : array (or scalar) of GPUs to send the GLM to
        %       
        %   optional input key/value pairs:
        %       useDoublePrecision (bool; default=TRUE) : set to false if you want to use single precision
        %       max_trials_for_sparse_run      (int; default=128)   : number of trials that can be run sparsely on each gpu. If the given trial_weights is sparse, this can 
        %                                                             speed up computation (i.e., for SGD). Any run where sum(trial_weights[on this block] != 0) > max_trials_for_sparse_run
        %                                                             will take the same time as running the whole set. This can be a scalar or vector to match deviceNumbers 
        %                                                             if you want to have different numbers of blocks on different GPUs.
        %
        %   pointer to the c++ GLM object is stored in obj.gpuObj_ptr.
        %
        function [obj] = toGPU(obj, deviceNumbers, varargin)
            if(obj.isOnGPU())
                error("GLM is already loaded to GPU. Clear before re-loading. If problems occured, clear manually by using 'kcResetDevice(gpuNums)' and 'clear mex'.");
            end
            
            %% check device numbers
            if((nargin < 2 || isempty(deviceNumbers)) && gpuDeviceCount() == 1)
                %if computer has only one GPU, can be set by default 
                deviceNumbers = 0;
            end
            if((nargin < 2 || isempty(deviceNumbers)) || ~isnumeric(deviceNumbers) || ~all(fix(deviceNumbers) == deviceNumbers, 'all') || ~all(deviceNumbers >= 0 & deviceNumbers < gpuDeviceCount()))
                error("Invalid GPU device numbers given");
            end
            
            %% parse optional inputs
            p = inputParser;
            p.CaseSensitive = false;
            p.Parameters;
            p.Results;
            p.KeepUnmatched = true;
            
            % set the desired and optional input arguments
            addParameter(p, 'useDoublePrecision', obj.gpuDoublePrecision, @islogical);
            addParameter(p, 'max_trials_for_sparse_run', min(obj.dim_M, 128), @(x) isnumeric(x) && (isscalar(x) || numel(x) == numel(deviceNumbers)) && all(fix(x) == x, 'all'));
            parse(p,varargin{:});
            
            useDoublePrecision = p.Results.useDoublePrecision;
            max_trials_for_sparse_run = p.Results.max_trials_for_sparse_run;
            
            %% sets up GLMstructure to send to GPU in correct datatypes and 0 indexing
            GLMstructure_GPU = obj.GLMstructure;
            
            GLMstructure_GPU.max_trial_length = uint64(max(arrayfun(@(aa) numel(aa.Y), obj.trials)));
            GLMstructure_GPU.dim_K       = uint64(obj.dim_K(-1));
            if(useDoublePrecision)
                GLMstructure_GPU.binSize = double(obj.bin_size);
            else
                GLMstructure_GPU.binSize = single(obj.bin_size);
            end
            
            GLMstructure_GPU.logLikeSettings = 0;
            for ii = 1:numel(obj.validLogLikeTypes)
                if(strcmpi(obj.logLikeType, obj.validLogLikeTypes(ii)))
                    GLMstructure_GPU.logLikeSettings = ii - 1;
                    break;
                end
            end
            GLMstructure_GPU.logLikeSettings = int32(GLMstructure_GPU.logLikeSettings);
            
            GLMstructure_GPU.logLikeParams = [];
            if(useDoublePrecision)
                GLMstructure_GPU.logLikeParams = double(GLMstructure_GPU.logLikeParams);
            else
                GLMstructure_GPU.logLikeParams = single(GLMstructure_GPU.logLikeParams);
            end
            
            %% sets up trials to send to GPU in correct datatypes and 0 indexing
            trialBlocks = struct('GPU', cell(numel(deviceNumbers), 1), 'max_trials_for_sparse_run', [], 'trials', []);
            
            trialsPerBlock = ceil(obj.dim_M / numel(trialBlocks));
            for bb = 1:numel(trialBlocks)
                trialBlocks(bb).GPU = int32(deviceNumbers(bb)); 
                trialBlocks(bb).max_trials_for_sparse_run = int32(max_trials_for_sparse_run); 
                %selects which trials to load to block
                if(bb < numel(trialBlocks))
                    trial_indices = (1:trialsPerBlock) + (bb-1)*trialsPerBlock;
                else
                    trial_indices = ((bb-1)*trialsPerBlock+1):obj.dim_M;
                end
                
                %copies trials
                trialBlocks(bb).trials = obj.trials(trial_indices);
                
                %for each trial
                for mm = 1:numel(trialBlocks(bb).trials)
                    % makes sure spike count is floating point type (NO LONGER INT)
                    if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Y, 'double'))
                        trialBlocks(bb).trials(mm).Y = double(trialBlocks(bb).trials(mm).Y);
                    elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Y, 'single'))
                        trialBlocks(bb).trials(mm).Y = single(trialBlocks(bb).trials(mm).Y);
                    end
                    
                    % makes sure X is correct floating point type
                    trialBlocks(bb).trials(mm).X = cell2mat(trialBlocks(bb).trials(mm).X(:)');
                    if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).X, 'double'))
                        trialBlocks(bb).trials(mm).X = double(trialBlocks(bb).trials(mm).X);
                    elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).X, 'single'))
                        trialBlocks(bb).trials(mm).X = single(trialBlocks(bb).trials(mm).X);
                    end
                    
                    %sets trial_idx (int32 and 0 indexed)
                    trialBlocks(bb).trials(mm).trial_idx = uint32(trial_indices(mm) - 1);
                end
            end
            
            %% call mex with (GLMstructure_GPU, trialBlocks, useDoublePrecision), get pointer in return
            obj.gpuObj_ptr = kgmlm.CUDAlib.kcGLM_mex_create(GLMstructure_GPU, trialBlocks, useDoublePrecision);
            obj.gpus = deviceNumbers;
            obj.gpuDoublePrecision = useDoublePrecision;
        end
        
        %% check if is already loaded to GPU
        function [onGPU] = isOnGPU(obj)
            onGPU = ~isempty(obj.gpus); % should be loaded if this item is set. Not checking thoroughly here
        end
        
        %% unload from GPU
        function [obj] = freeGPU(obj)
            if(~obj.isOnGPU())
                warning("GLM is not loaded to GPU. Nothing to free.");
            else
                % call mex file to delete GPU object pointer
                kgmlm.CUDAlib.kcGLM_mex_clear(obj.gpuObj_ptr, obj.gpuDoublePrecision);
                % erase pointer value
                obj.gpuObj_ptr = 0;
                obj.gpus = [];
            end
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %methods for computing log likelihood (and derivatives) on GPU (Public)
    methods (Access = public)    
        
        function [results] = computeLogLikelihood(obj, params, opts, results, runHost)
            if(nargin < 5 || isempty(runHost))
                runHost = ~obj.isOnGPU();
            end
            if(~obj.isOnGPU() && ~runHost)
                error("GLM is not on GPU: must select runHost option to run on CPU.");
            end
            
            %allocate space for results
            if(nargin < 4 || isempty(results))
                results = obj.getEmptyResultsStruct(opts);
            end

            %send to GPU
            for jj = 1:numel(params.Ks)
                params.Ks{jj} = params.Ks{jj};
            end
            params.K = cell2mat(params.Ks);
            if(~runHost)
                kgmlm.CUDAlib.kcGLM_mex_computeLL(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, results, opts.trial_weights);
            else
                results = obj.computeLogLikelihoodhod_v2(params, opts, results);
            end
            results.log_likelihood = sum(results.trialLL);
        end
        
        function [results] = computeLogPosterior(obj, params, opts, computeDerivativesForEvidenceOptimization, results, runHost)
            if(nargin < 6)
                runHost = [];
            end
            if(nargin < 5)
                results = [];
            end
            
            %gets log likelihood
            results = obj.computeLogLikelihood(params, opts, results, runHost);
            
            %adds the prior
            if(nargin > 3 && computeDerivativesForEvidenceOptimization)
                results.dprior_sigma_inv = [];
                %adding this field should tell the prior function to compute derivatives of the prior cov matrix (or hessian) w.r.t each hyperparam
                %this is used for computing the analytic derivative in the evidence optimization funciton
                %  dprior_sigma_inv is dim_K x dim_K x dim_H where the third indexes the hyperparam
                %  It is the derivative of the inverse of the Hessian of the log prior for each hyperparam
            end
            results = obj.computeLogPrior(params, opts, results);
            
            %sums up results
            results.log_post = results.log_likelihood + results.log_prior;
        end
        
        % hessian can only contain params - no hessian of hyperparams taken
        function [nlog_like, ndl_like, n2dl_like, results, params] = vectorizedNLL_func(obj, w_c, params, opts)
            if(nargout > 1)
                opts_0 = opts;
            else
                opts_0 = obj.getComputeOptionsStruct("enableAll", false, "includeHyperparameters", false);
            end
            opts_0.compute_trialLL = true;

            params = obj.devectorizeParams(w_c, params, opts);

            results    = obj.computeLogLikelihood(params, opts_0);
            nlog_like  = -results.log_likelihood;

            if(nargout > 1)
                ndl_like =  obj.vectorizeResults(results, opts_0);
                ndl_like = -ndl_like;
            end
            if(nargout > 2)
                n2dl_like = -results.d2K;
            end
        end
        
        function [nlog_post, ndl_post, n2dl_post, results, params] = vectorizedNLPost_func(obj, w_c, params, opts)
            if(nargout > 1)
                opts_0 = opts;
            else
                opts_0 = obj.getComputeOptionsStruct("enableAll", false, "includeHyperparameters", false);
            end
            opts_0.compute_trialLL = true;

            params = obj.devectorizeParams(w_c, params, opts);

            results    = obj.computeLogPosterior(params, opts_0);
            nlog_post  = -results.log_post;

            if(nargout > 1)
                ndl_post =  obj.vectorizeResults(results, opts_0);
                ndl_post = -ndl_post;
            else
                ndl_post = [];
            end
            if(nargout > 2)
                n2dl_post = -results.d2K;
            else
                n2dl_post = [];
            end
        end
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %externally defined methods for inference
    methods (Access = public)
        [params_mle, results_mle, params_init] = computeMLE(obj, varargin);
        [params_map, results_map] = computeMAP(obj, params_init, varargin);
        [params_mle, results_test_mle, results_train_mle, params_init] = computeMLE_crossValidated(obj, foldIDs, varargin);
        [params_map, results_test_map, results_train_map] = computeMAP_crossValidated(obj, foldIDs, params_init, varargin);
        
        [samples, summary, HMC_settings, paramStruct, M] = runHMC_simple(obj, params_init, settings, varargin);
        [params, acceptedProps, log_p_accept] = scalingMHStep(obj, params, MH_scaleSettings);
        [HMC_settings] = setupHMCparams(obj, nWarmup, nSamples, debugSettings);
        
        [nle, ndle, params, results, ld, d_ld, sigma] = computeNLEvidence(obj, H, params, trial_weights);
        [params_map, results_map, params_init] = computeMAPevidenceOptimization(obj,  varargin);
        [params_map, results_test_map, results_train_map] = computeMAPevidenceOptimization_crossValidated(obj, foldIDs,  varargin);

        [] = setupComputeStructuresHost(obj, reset);
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating computation result structures (Private)
    methods (Access = public)
        %% METHOD [paramStruct] = getEmptyResultsStruct(opts)
        %	Builds a blank structure with all the space for the derivatives of the GLM parameters and the log likelihood
        %
        %   By default, all values are set to 0, but everything is the correct size.
        %
        %   inputs:
        %       opts (struct) : an opts struct with which fields need to exist
        %
        %	returns:
        %       params [struct]        
        %           A blank (zero-filled) structure of the model parameters.
        %
        %           Fields for model parameters:
        %               log_likelihood (real)      : scalar for the sum of trialLL
        %               log_post       (real)      : total log posterior
        %               log_prior      (real)      : total log prior
        %
        %               trialLL [dim_M x 1] (real) : log likelihoods computed per trial
        %
        %               all derivates will be over both the likelihood and prior (if computing posterior, otherwise obviously only over likelihood)
        %
        %               dK  [dim_K x 1]      (real) : 
        %               d2K [dim_K x dim_K]  (real) : 
        %
        %
        function [results, dataType] = getEmptyResultsStruct(obj, opts)
            useDoublePrecision = isa(opts.trial_weights, 'double');
            if(useDoublePrecision)
                dataType = 'double';
            else
                dataType = 'single';
            end
            
            if(~isfield(opts, 'trialLL') || opts.trialLL)
                results.trialLL = zeros(obj.dim_M, 1, dataType);
            else
                results.trialLL = [];
            end
            
            %sets up top-level params
            if(opts.dK)
                results.dK = zeros(obj.dim_K(-1), 1, dataType);
            else
                results.dK = zeros(0, 0, dataType);
            end
            if(opts.d2K)
                results.d2K = zeros(obj.dim_K(-1), obj.dim_K(-1), dataType);
            else
                results.d2K = zeros(0, 0, dataType);
            end
            if(isfield(opts, 'dH'))
                if(opts.dH)
                    results.dH = zeros(obj.dim_H, 1, 'double');
                else
                    results.dH = zeros(0, 0, 'double');
                end
            end
        end
    end
end