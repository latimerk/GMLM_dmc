%% CLASS GMLM
%  Generalized multilinear modeling for spike trains of a population of neurons.
%  The log rate depends on a sum tensor groups of regressors (plus a pure linear term and a constrant for each neuron)
%  Each tensor group is a multilinear function. The coefficients and regressors both have a CP decomposition.
%  For a given tensor group:
%       There are dim_A "events" (i.e., number of stimulus presentations)
%       The parameter CP decomposition of rank dim_R
%
%       Group j's contribution to the (log) rate for an observation at t and neuron p is (the index over j is dropped from all terms)
%
%           \sum_{a=1}^{dim_A}   \sum_{r = 1}^{dim_R}   \prod_{f}^{dim_D} X_{n,a,f}*F^{f}_r * V_{p,r}
%
%           The coefficient is X_{t,f,a}
%
%           F^{f}_r is the r'th column of F^{f} is a matrix, which also can have a decomposition
%
%               F^{f} = Khatri-Rao Product (T_{ii} : ii \in factor_idx == ff) where the matrices T are the base parameters
%                                                                             factor_idx just says how to combine T's (if no factorization is needed here, factor_idx not needed)
%               dim_F(f) = size(F^{f}, 1)

%           The matrix V gives the loading weights for this low-dimensional term
%           
%           The regressor term X_{f} is a multidimensional array of size (N x size(F^f,1) x dim_A) where N is the number of observations
%               (if X_{f} is the same for all events, the third dimension can be set to 1 regardless of dim_A)
%           
%               Then, X_{n,a,f} is a row vector of size (1 x dim_F(f))
%
%               The X's can be defined in two ways depending on sparsity structure
%
%                   If X is dense, each trial defines X in trials.Groups(jj).X_local{f}
%                           (set each trials.Groups(jj).iX_shared{f} = empty)
%                   If X has repeating structure, it can be defined using two variables:
%                           trials.Groups(jj).X_local{f} = empty
%                           GMLMstructure.Groups(jj).X_shared{f} = dim_X x dim_F(f))
%                           trials.Groups(jj).iX_shared{f} = N x dim_A. Each entry of iX_shared is an integer index into a row of GMLMstructure.Groups(jj).X_shared{f}
%                                                                       (if index out-of-bounds, value is assumed to be 0)
%
%   See example/DMC.modelBuilder.constructGMLMRegressors for a complete example on setting up the arguments for a model
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
classdef GMLM < handle
    properties (GetAccess = public, SetAccess = private)
        GMLMstructure struct
        trials        struct
        neuronIdx
        bin_size 
        
        logLikeType
        validLogLikeTypes
        
        max_params %maximum number of parameters in the model (assuming max rank of all tensors) : does not include hyperparams
        
        gpuObj_ptr uint64 %uint64 pointer to the c++ object for the GMLM loaded to the GPU
        gpus       uint32 % array for which GPUs are in use
        gpuDoublePrecision logical %if current GPU loading is double precision (if false, then single)
        
        isSimultaneousPopulation logical % if data is structured as a simultaneously recorded population
        
        temp_storage_file
        destroy_temp_storage_file logical

        LL_info
        X_groups

    end
    
    %% Constructor
    methods 
        function obj = GMLM(GMLMstructure, trials, bin_size, logLikeType)
            if(nargin < 2 || nargin > 4)
                error("GMLM constructor: two struct inputs required (GMLMstructure and trials)");
            end
            obj.X_groups= [];
            obj.LL_info = [];
            
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
            
            %% check the GMLM structure
            if(~isstruct(GMLMstructure))
                error("GMLM constructor: input GMLMstructure must be a struct");
            end
            % check for linear dimension (non-negative integer)
            if(~isfield(GMLMstructure, "dim_B") || ~isnumeric(GMLMstructure.dim_B) || fix(GMLMstructure.dim_B) ~= GMLMstructure.dim_B || GMLMstructure.dim_B < 0)
                error("GMLM constructor: GMLMstructure must have property dim_B (dimensionalty of the linear term: non-negative integer)");
            end
            
            % check each group
            if(~isfield(GMLMstructure, "Groups") || ~isstruct(GMLMstructure.Groups) || isempty(GMLMstructure.Groups) || ...
                    ~isfield(GMLMstructure.Groups, "dim_names") || ~isfield(GMLMstructure.Groups, "dim_T") || ~isfield(GMLMstructure.Groups, "dim_R_max") || ~isfield(GMLMstructure.Groups, "dim_A") || ...
                    ~isfield(GMLMstructure.Groups, "name"))
                error("GMLM constructor: GMLMstructure must contain non-empty struct array 'Groups' with field 'dim_T', 'dim_R_max', 'dim_A', 'dim_names', and 'name'");
            end
            
            
            %if contains a prior, should be empty or a function
            if(isfield(GMLMstructure, "prior"))
                if(~isempty(GMLMstructure.prior))
                    if(~isstruct(GMLMstructure.prior) || ~isfield(GMLMstructure.prior, 'log_prior_func') || ~isfield(GMLMstructure.prior, 'dim_H'))
                        error("GMLM constructor: GMLMstructure.prior must be empty or structure with fields 'log_prior_func' and 'dim_H'");
                    end

                    if(~isa(GMLMstructure.prior.log_prior_func,'function_handle'))
                        error("GMLM constructor: GMLMstructure.prior.log_prior_func must be a function handle");
                    end

                    if(~isscalar(GMLMstructure.prior.dim_H) || GMLMstructure.prior.dim_H < 0 || fix(GMLMstructure.prior.dim_H) ~= GMLMstructure.prior.dim_H)
                        error("GMLM constructor: GMLMstructure.prior.dim_H must be a non-negative integer");
                    end
                end
            end
            if(isfield(GMLMstructure, "gibbs_step"))
                if(~isempty(GMLMstructure.gibbs_step))
                    if(~isstruct(GMLMstructure.gibbs_step) || ~isfield(GMLMstructure.gibbs_step, 'sample_func') || ~isfield(GMLMstructure.gibbs_step, 'dim_H'))
                        error("GMLM constructor: GMLMstructure.gibbs_step must be empty or structure with fields 'sample_func' and 'dim_H'");
                    end

                    if(~isa(GMLMstructure.gibbs_step.sample_func,'function_handle'))
                        error("GMLM constructor: GMLMstructure.gibbs_step.sample_func must be a function handle");
                    end

                    if(~isscalar(GMLMstructure.gibbs_step.dim_H) || GMLMstructure.gibbs_step.dim_H < 0 || fix(GMLMstructure.gibbs_step.dim_H) ~= GMLMstructure.gibbs_step.dim_H)
                        error("GMLM constructor: GMLMstructure.gibbs_step.dim_H must be a non-negative integer");
                    end
                end
            end
                
            for jj = 1:numel(GMLMstructure.Groups)
                %check max rank setting (positive integer)
                dim_R_max = GMLMstructure.Groups(jj).dim_R_max;
                if(~isnumeric(dim_R_max) || fix(dim_R_max) ~= dim_R_max || dim_R_max <= 0)
                    error("GMLM constructor: GMLMstructure.Groups().dim_R_max (maximum tensor rank) must be positive integer");
                end
                
                % check dimensions
                dim_T = GMLMstructure.Groups(jj).dim_T;
                dim_S = numel(dim_T);
                if(isempty(dim_T) || numel(dim_T) > 8 || ~isnumeric(dim_T) || ~all(fix(dim_T) == dim_T) || ~all(dim_T > 0))
                    error("GMLM constructor: GMLMstructure.Groups().dim_T (dimensions of each tensor components) must be an array of positive integers (and cannot have more than 8 elements).");
                end
                
                % checks factorization
                if(~isfield(GMLMstructure.Groups(jj), 'factor_idx') || isempty(GMLMstructure.Groups(jj).factor_idx))
                    fidxs = (1:dim_S)';
                else
                    fidxs = GMLMstructure.Groups(jj).factor_idx(:);
                end
                dim_D = max(fidxs);
                if(~all(ismember(fidxs, 1:dim_D),'all')  || ~all(ismember(1:dim_D, fidxs), 'all') || numel(fidxs) ~= dim_S)
                    error("GMLM constructor: GMLMstructure.Groups().factor_idx is invalid - GMLMstructure.Groups().factor_idx can be empty or numel(GMLMstructure.Groups().factor_idx) = numel(GMLMstructure.Groups().dim_T) and unique(GMLMstructure.Groups().factor_idx) = 1:max(GMLMstructure.Groups().factor_idx).");
                end
                dim_F = ones(dim_D,1);
                for ss = 1:dim_S
                    dim_F(fidxs(ss)) = dim_F(fidxs(ss)) * dim_T(ss);
                end
                GMLMstructure.Groups(jj).factor_idx = fidxs;
                
                %check number of events (positive integer)
                dim_A = GMLMstructure.Groups(jj).dim_A;
                if(~isscalar(dim_A) || ~isnumeric(dim_A) || fix(dim_A) ~= dim_A || dim_A <= 0)
                    error("GMLM constructor: GMLMstructure.Groups().dim_A (number of events for the coefficients) must be positive integer");
                end
                
                %check name (string or char)
                name = GMLMstructure.Groups(jj).name;
                if((~ischar(name) && ~isstring(name)) || isempty(name))
                    error("GMLM constructor: GMLMstructure.Groups().name must be a non-empty string");
                end
                

                %check X_shared
                if(dim_D == 1 && ismatrix(GMLMstructure.Groups(jj).X_shared) && ~iscell(GMLMstructure.Groups(jj).X_shared))
                    GMLMstructure.Groups(jj).X_shared = {GMLMstructure.Groups(jj).X_shared};
                end
                X_shared = GMLMstructure.Groups(jj).X_shared;
                if(~iscell(X_shared) || numel(X_shared) ~= dim_D)
                    error("GMLM constructor: GMLMstructure.Groups().X_shared must be a non-empty cell array of same length as the number of factors values.");
                end
                for ff = 1:dim_D
                    if(~isempty(X_shared{ff}))
                        if(ndims(X_shared{ff}) > 2) %#ok<ISMAT>
                            tts = size(X_shared{ff}, 1 + (1:sum(fidxs == ff)));
                            dts = dim_T(fidxs == ff);
                            if(~all(tts(:) == dts(:),'all'))
                                error("GMLM constructor: GMLMstructure.Groups().X_shared{ff} dims does not match dim_Ts");
                            end
                            X_shared{ff} = reshape(X_shared{ff}, size(X_shared{ff},1), []);
                        end
                        if(size(X_shared{ff},2) ~= dim_F(ff) || ~isnumeric(X_shared{ff}) || ~ismatrix(X_shared{ff}))
                            error("GMLM constructor: GMLMstructure.Groups().X_shared{ff} must have dim_F(ff) columns where dim_F(ff) = prod(dim_T(factor_idx == ff)).");
                        end
                    end
                end
                GMLMstructure.Groups(jj).X_shared = X_shared;


                
                %if contains a prior, should be empty or a function
                if(isfield(GMLMstructure.Groups, "prior"))
                    if(~isempty(GMLMstructure.Groups(jj).prior))
                        if(~isstruct(GMLMstructure.Groups(jj).prior) || ~isfield(GMLMstructure.Groups(jj).prior, 'log_prior_func') || ~isfield(GMLMstructure.Groups(jj).prior, 'dim_H'))
                            error("GMLM constructor: GMLMstructure.Groups().prior must be empty or structure with fields 'log_prior_func' and 'dim_H'");
                        end
                        
                        if(~isa(GMLMstructure.Groups(jj).prior.log_prior_func,'function_handle'))
                            error("GMLM constructor: GMLMstructure.Groups().prior.log_prior_func must be a function handle");
                        end
                        
                        validScalar = isnumeric(GMLMstructure.Groups(jj).prior.dim_H) && isscalar(GMLMstructure.Groups(jj).prior.dim_H) && GMLMstructure.Groups(jj).prior.dim_H >= 0 && fix(GMLMstructure.Groups(jj).prior.dim_H) == GMLMstructure.Groups(jj).prior.dim_H;
                        
                        if(~validScalar && ~isa(GMLMstructure.Groups(jj).prior.dim_H, 'function_handle'))
                            error("GMLM constructor: GMLMstructure.Groups().prior.dim_H must be a non-negative integer or function");
                        end
                    end
                end
                %if contains a gibbs step, should be empty or a function
                if(isfield(GMLMstructure.Groups, "gibbs_step"))
                    if(~isempty(GMLMstructure.Groups(jj).gibbs_step))
                        if(~isstruct(GMLMstructure.Groups(jj).gibbs_step) || ~isfield(GMLMstructure.Groups(jj).gibbs_step, 'sample_func') || ~isfield(GMLMstructure.Groups(jj).gibbs_step, 'dim_H'))
                            error("GMLM constructor: GMLMstructure.Groups().gibbs_step must be empty or structure with fields 'sample_func' and 'dim_H'");
                        end
                        
                        if(~isa(GMLMstructure.Groups(jj).gibbs_step.sample_func,'function_handle') && ~isempty(GMLMstructure.Groups(jj).gibbs_step.sample_func))
                            error("GMLM constructor: GMLMstructure.Groups().gibbs_step.sample_func must be a function handle");
                        end
                        
                        validScalar = isnumeric(GMLMstructure.Groups(jj).gibbs_step.dim_H) && isscalar(GMLMstructure.Groups(jj).gibbs_step.dim_H) && GMLMstructure.Groups(jj).gibbs_step.dim_H >= 0 && fix(GMLMstructure.Groups(jj).gibbs_step.dim_H) == GMLMstructure.Groups(jj).gibbs_step.dim_H;
                        
                        if(~validScalar && ~isa(GMLMstructure.Groups(jj).gibbs_step.dim_H,'function_handle'))
                            error("GMLM constructor: GMLMstructure.Groups().gibbs_step.dim_H must be a non-negative integer or function");
                        end
                    end
                end
                
                %check dim_names (string)
                dim_names = GMLMstructure.Groups(jj).dim_names;
                if( ~isstring(dim_names) || numel(dim_names) ~= dim_S)
                    error("GMLM constructor: GMLMstructure.Groups().dim_names must be a non-empty string array with entries for each tensor dimension");
                end
                %makes sure all dim names are unique
                if(numel(unique(dim_names)) ~= numel(dim_names))
                    error("GMLM constructor: tensor coefficient dimensions must have unique names (within each group: same names can exist in different groups)");
                end
                
                % factor names
                if(~isfield(GMLMstructure.Groups(jj), "factor_names") || isempty(GMLMstructure.Groups(jj).factor_names))
                    % field is not required - setup default values
                    GMLMstructure.Groups(jj).factor_names = repmat("factor", dim_D, 1);
                    for ff = 1:dim_D
                        ss_s = find(fidxs == ff);
                        for ss = 1:numel(ss)
                            GMLMstructure.Groups(jj).factor_names(ff) = sprintf("%s_%s", GMLMstructure.Groups(jj).factor_names(ff), dim_names(ss_s(ss)));
                        end
                    end
                end
                %check factor_names (string)
                factor_names = GMLMstructure.Groups(jj).factor_names;
                if( ~isstring(factor_names) || numel(factor_names) ~= dim_D)
                    error("GMLM constructor: GMLMstructure.Groups().factor_names must be a non-empty string array with entries for each tensor factor");
                end
                %makes sure all factor names are unique
                if(numel(unique(factor_names)) ~= numel(factor_names))
                    error("GMLM constructor: tensor coefficient factors must have unique names (within each group: same names can exist in different groups)");
                end
            end
            
            %makes sure all group names are unique
            if(numel(unique([GMLMstructure.Groups(:).name])) ~= numel(GMLMstructure.Groups))
                error("GMLM constructor: tensor coefficient groups must have unique names");
            end
            
            %% check each trial to see if it matches the structure
            if(~isstruct(trials) || isempty(trials) || ~isfield(trials, "Y"))
                error("GMLM constructor: input trials must be non-empty struct array with a Y field (spike counts).");
            end
            
            N_neurons_per_trial = 1;
            if(~isfield(trials, "neuron") || all(arrayfun(@(aa) isempty(aa.neuron), trials), 'all'))
                N_neurons_per_trial = size(trials(1).Y, 2);
                obj.isSimultaneousPopulation = true;
                if(N_neurons_per_trial == 1)
                    warning('Only 1 neuron found!');
                end
            else
                obj.isSimultaneousPopulation = false;
            end
            
            % check for spike observations
               %exists                      for each trial     vector           numbers        contains obs.     is integers (spike counts -> this could be relaxed for more general models)
            if(~all(arrayfun(@(aa) size(aa.Y, 2) == N_neurons_per_trial & isnumeric(aa.Y) & ~isempty(aa.Y), trials), 'all'))
                error("GMLM constructor: trials must have spike count vector/matrix Y (for Poisson likelihood, may contain negatives to indicate censored bins, but each trial needs at least one observation!)");
            end
            
            %check for linear terms
            if(GMLMstructure.dim_B == 0 && isfield(trials, "X_lin"))
                %if X_lin is given, but expected to be empty, makes sure it's empty!
                if(~all(arrayfun(@(aa) isempty(aa.X_lin), trials), 'all'))
                    error("GMLM constructor: trials.X_lin expected to be empty (GMLMstructure.dim_B > 0), but found non-empty entries!");
                end
            elseif(GMLMstructure.dim_B > 0)
                %X_lin required
                if(~isfield(trials, "X_lin"))
                    error("GMLM constructor: trials requires field X_lin!");
                end
                
                X_lin_size = 1;
                if(obj.isSimultaneousPopulation)
                    X_lin_size = size(trials(1).X_lin, 3);
                    if(X_lin_size ~= N_neurons_per_trial && X_lin_size ~= 1)
                        error("GMLM constructor: X_lin is not a valid size!");
                    end
                end
                
                %X_lin size must be correct
                if(~all(arrayfun(@(aa) isnumeric(aa.X_lin) &  size(aa.X_lin,1) == size(aa.Y,1) & size(aa.X_lin, 2) == GMLMstructure.dim_B & size(aa.X_lin, 3) == X_lin_size, trials), 'all'))
                    error("GMLM constructor: each trials(jj).X_lin should be a matrix of size numel(trials(jj).Y) x GMLMstructure.dim_B");
                end
            end
            
            %checks neuron number
            if(~obj.isSimultaneousPopulation)
                if(~all(arrayfun(@(aa) isnumeric(aa.neuron), trials), 'all') && ~all(arrayfun(@(aa) ischar(aa.neuron) | isstring(aa.neuron), trials), 'all'))
                    error("GMLM constructor: invalid neuron indentifiers for trials (must be all numeric or all strings)");
                end
            end
            
            %checks each group
                %consistent group numbers
            if(~all(arrayfun(@(aa) numel(aa.Groups) == numel(GMLMstructure.Groups), trials), 'all'))
                error("GMLM constructor: inconsistent number of groups in trials");
            end
            for jj = 1:numel(GMLMstructure.Groups)
                
                %checks for consistent number of dimensions
                dim_T = GMLMstructure.Groups(jj).dim_T;
                dim_S = numel(dim_T);
                fidxs = GMLMstructure.Groups(jj).factor_idx(:);
                dim_D = max(fidxs);
                dim_F = ones(dim_D,1);
                for ss = 1:dim_S
                    dim_F(fidxs(ss)) = dim_F(fidxs(ss)) * dim_T(ss);
                end

                if( ~all(arrayfun(@(aa) (iscell(aa.Groups(jj).iX_shared) & numel(aa.Groups(jj).iX_shared) == dim_D), trials)))
                    error("GMLM constructor: trials.Groups requires fields 'iX_shared' to be cell array of dim_D (one for each tensor factor).");
                end
                if( ~all(arrayfun(@(aa) (iscell(aa.Groups(jj).X_local) & numel(aa.Groups(jj).X_local) == dim_D), trials)))
                    error("GMLM constructor: trials.Groups requires fields 'X_local' to be cell array of dim_D (one for each tensor factor).");
                end

                %checks each dimension
                for ff = 1:dim_D
                    %when X_shared is populated
                    if(~isempty(GMLMstructure.Groups(jj).X_shared{ff}))
                        %check to see if iX is correct size
                        if(~all(arrayfun(@(aa) isnumeric(aa.Groups(jj).iX_shared{ff}) && ismatrix(aa.Groups(jj).iX_shared{ff}) && ...
                                               size(aa.Groups(jj).iX_shared{ff},1) == size(aa.Y,1) && ... 
                                               size(aa.Groups(jj).iX_shared{ff},2) == GMLMstructure.Groups(jj).dim_A && ...
                                               all(aa.Groups(jj).iX_shared{ff} == fix(aa.Groups(jj).iX_shared{ff}), 'all') ...
                                               , trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).iX_shared{ff} must be matrix of integers of size (numel(trials.Y) x GMLMstructure.Groups(jj).dim_A) when GMLMstructure.Groups(jj).X_shared{ss} is populated.");
                        end
                    else
                        
                        %when X_shared is empty
                        %check to see if X_local is correct size
                        for tt = 1:numel(trials)
                            if(ndims(trials(tt).Groups(jj).X_local{ff}) > 2) %#ok<ISMAT>
                                tts = size(trials(tt).Groups(jj).X_local{ff}, 1 + (1:(sum(fidxs == ff) + 1)));
                                dts = dim_T(fidxs == ff);
                                if((tts(1) ~= dim_F(ff) && (tts(2) ~= dim_A || tts(2) ~= 1)) && ~all(tts(:) == [dts(:);dim_A],'all') && ~all(tts(:) == [dts(:);1],'all'))
                                    error("GMLM constructor: trials(tt).Groups(jj).X_local{ff} dims does not match dim_Ts");
                                end
                                
                                if(tts(1) ~= dim_F(ff))
                                    new_size = [size(trials(tt).Groups(jj).X_local{ff},1), dim_F(ff), tts(end)];
                                    if(numel(new_size)-1 ~= numel(tts) || ~all(tts == new_size(2:end))) 
                                        trials(tt).Groups(jj).X_local{ff} = reshape(trials(tt).Groups(jj).X_local{ff}, new_size);
                                    end
                                end
                            end
                        end
                        if(~all(arrayfun(@(aa) isnumeric(aa.Groups(jj).X_local{ff}) && ndims(aa.Groups(jj).X_local{ff}) <= 3 && ...
                                               size(aa.Groups(jj).X_local{ff},1) == size(aa.Y,1) && ... 
                                               size(aa.Groups(jj).X_local{ff},2) == dim_F(ff) && ... 
                                               (size(aa.Groups(jj).X_local{ff},3) == GMLMstructure.Groups(jj).dim_A || size(aa.Groups(jj).X_local{ff},3) == 1) ...
                                               , trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).X_local{ff} must be matrix of integers of size (numel(trials.Y) x dim_F(ff)) when ff is not a shared dim.");
                        end
                    end
                end
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
            obj.GMLMstructure = GMLMstructure;
            %goes through each trial
            %  if there are shared regressors, eliminates any unused refs
            for jj = 1:numel(GMLMstructure.Groups)
                %if multiple coefs have shared regressors
                ff = find(obj.isSharedRegressor(jj));

                if(numel(ff) > 1)
                    TT = cellfun(@(a) size(a,1), GMLMstructure.Groups(jj).X_shared);
                    TT = TT(ff);

                    for mm = 2:obj.dim_M
                        vv = true(size(trials.Y,1), GMLMstructure.Groups(jj).dim_A, numel(ff));
                        for ss = 1:numel(ff)
                            vv_c = trials(mm).Groups(jj).iX_shared{ff(ss)} > 0 & trials(mm).Groups(jj).iX_shared{ff(1)} <= TT(ss);
                            vv(:,:,ss) = vv_c;
                        end
                        vv_a = all(vv,3);
                        for ss = 1:numel(ff)
                            trials(mm).Groups(jj).iX_shared{ff(ss)}(~vv(:,:,ss) & vv_a) = 0; % element is 0'd in at least one dimension
                        end
                    end
                end
            end

            obj.trials = trials;
            
            
            % check for prior, otherwise is null by default
            if(isfield(GMLMstructure, "prior"))
                obj.GMLMstructure.prior = GMLMstructure.prior;
            else
                obj.GMLMstructure.prior = [];
            end
            if(isfield(GMLMstructure, "gibbs_step"))
                obj.GMLMstructure.gibbs_step = GMLMstructure.gibbs_step;
            else
                obj.GMLMstructure.gibbs_step = [];
            end
                
            %set default rank, prior, and makes sure name is a string (not char)
            for jj = 1:numel(obj.GMLMstructure.Groups)
                obj.GMLMstructure.Groups(jj).dim_R = obj.GMLMstructure.Groups(jj).dim_R_max;
                obj.GMLMstructure.Groups(jj).name  = string(obj.GMLMstructure.Groups(jj).name);
                
                % check for prior, otherwise is null by default
                if(isfield(GMLMstructure.Groups, "prior"))
                    obj.GMLMstructure.Groups(jj).prior = GMLMstructure.Groups(jj).prior;
                else
                    obj.GMLMstructure.Groups(jj).prior = [];
                end
                
                % check for gibbs_step, otherwise is null by default
                if(isfield(GMLMstructure.Groups, "gibbs_step"))
                    obj.GMLMstructure.Groups(jj).gibbs_step = GMLMstructure.Groups(jj).gibbs_step;
                else
                    obj.GMLMstructure.Groups(jj).gibbs_step = [];
                end
            end
            
            %sets local neuron index
            if(~obj.isSimultaneousPopulation)
                [obj.neuronIdx, ~, idxs] = unique([trials(:).neuron]);
                idxs = num2cell(idxs);
                [obj.trials(:).neuron_idx] = idxs{:};
            end
            
            %computes maximum number of parameters in the model
            obj.max_params = obj.dim_P + obj.dim_P*obj.dim_B;
            for jj = 1:obj.dim_J
                obj.max_params = obj.max_params + obj.dim_P*obj.dim_R_max(jj);
                for ss = 1:obj.dim_S(jj)
                    obj.max_params = obj.max_params + obj.dim_T(jj,ss)*obj.dim_R_max(jj);
                end
            end

           
            
            %% set that no GPU is currently in use
            obj.gpuObj_ptr = 0;
            obj.gpus = [];
            obj.gpuDoublePrecision = true;
            
            obj.temp_storage_file = [];
            obj.destroy_temp_storage_file = true;
        end
    end
    
    %% destructor
    % takes care of clearing any GPU space 
    methods
        function delete(obj)
            if(obj.isOnGPU())
                fprintf('GMLM destructor: clearing GMLM from GPU.\n');
                obj.freeGPU();
            end
            
            % delete any storage files used (for example, by MCMC)
            if( ~isempty(obj.temp_storage_file) && obj.destroy_temp_storage_file &&(isstring(obj.temp_storage_file) || ischar(obj.temp_storage_file)) && exist(obj.temp_storage_file, 'file'))
                fprintf('GMLM destructor: deleting temporary storage file (%s).\n', obj.temp_storage_file);
                delete(obj.temp_storage_file);
                obj.temp_storage_file = [];
            end
        end
    end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for dimension information (Public)
    % mostly to help me with my notation
    methods (Access = public)

        function [pp] = dim_P(obj) %number of neurons
            if(obj.isSimultaneousPopulation)
                pp = size(obj.trials(1).Y,2);
            else
                pp = numel(obj.neuronIdx);
            end
        end
        function [ss] = dim_trialLL(obj, dim) %size of the trial log likelihood
            if(obj.isSimultaneousPopulation)
                ss = [obj.dim_M() obj.dim_P()];
            else
                ss = [obj.dim_M() 1];
            end
            if(nargin > 1)
                ss = ss(dim);
            end
        end
        
        function [dd] = dim_J(obj) % number of tensor coefficient groups
            dd = numel(obj.GMLMstructure.Groups);
        end
        
        function [kk] = dim_B(obj) % size of linear term
            kk = obj.GMLMstructure.dim_B;
        end
        
        function [mm] = dim_M(obj) % dim_M is the number of trials
            mm = numel(obj.trials);
        end
        
        function [nn] = dim_N(obj, tr) % length of each trial
            nn = arrayfun(@(a)size(a.Y,1), obj.trials);
            if(nargin > 1)
                nn = nn(tr);
            end
        end
        
        function [aa] = dim_A(obj, idx) % number of 'events' for a particular tensor coefficient group (given by idx)
            idx = obj.getGroupIdx(idx);
            if(nargin < 2 || isempty(idx))
                aa = zeros(obj.dim_J,1);
                for jj = 1:obj.dim_J
                    aa(jj) = obj.dim_A(jj);
                end
            else
                idx = obj.getGroupIdx(idx);
                aa = obj.GMLMstructure.Groups(idx).dim_A;
            end
        end
        
        function [ss] = dim_S(obj, idx) % tensor order (minus the dimension for the neuron loading weights) for a particular tensor coefficient group (given by idx)
            if(nargin < 2 || isempty(idx))
                ss = zeros(obj.dim_J,1);
                for jj = 1:obj.dim_J
                    ss(jj) = obj.dim_S(jj);
                end
            else
                idx = obj.getGroupIdx(idx);
                ss = numel(obj.GMLMstructure.Groups(idx).dim_T);
            end
        end
        function [ss] = dim_D(obj, idx) % tensor order (minus the dimension for the neuron loading weights) of the factorization of the tensor coefficient group (given by idx)
            if(nargin < 2 || isempty(idx))
                ss = zeros(obj.dim_J,1);
                for jj = 1:obj.dim_J
                    ss(jj) = obj.dim_D(jj);
                end
            else
                idx = obj.getGroupIdx(idx);
                ss = max(obj.GMLMstructure.Groups(idx).factor_idx, [], 'all');
            end
        end
        
        
        function [rr] = dim_R(obj, idx) % current tensor rank (between 0 and dim_R_max) for a particular tensor coefficient group (given by idx)
            if(nargin < 2 || isempty(idx))
                rr = zeros(obj.dim_J,1);
                for jj = 1:obj.dim_J
                    rr(jj) = obj.dim_R(jj);
                end
            else
                idx = obj.getGroupIdx(idx);
                rr = obj.GMLMstructure.Groups(idx).dim_R;
            end
        end
        
        function [tt] = dim_T(obj, idx, dim) % the size of a tensor dimension (given by dim) for a particular tensor coefficient group (given by idx). If dim is empty, returns for all dims
            idx = obj.getGroupIdx(idx);
            if(nargin < 3 || isempty(dim))
                tt = zeros(obj.dim_S(idx),1);
                for ss = 1:obj.dim_S(idx)
                    tt(ss) = obj.dim_T(idx, ss);
                end
            else
                dim = obj.getGroupDimIdx(idx, dim);
                tt = obj.GMLMstructure.Groups(idx).dim_T(dim);
            end
        end
        function [factors] = getFactorIdxs(obj, idx)
            idx = obj.getGroupIdx(idx);
            factors = obj.GMLMstructure.Groups(idx).factor_idx;
        end
        
        function [ff] = dim_F(obj, idx, factor) % the size of a tensor dimension (given by dim) for a particular tensor coefficient group factor. If factor is empty, returns for all dims
            idx = obj.getGroupIdx(idx);
            if(nargin < 3 || isempty(factor))
                ff = zeros(obj.dim_D(idx),1);
                for ss = 1:obj.dim_D(idx)
                    ff(ss) = obj.dim_F(idx, ss);
                end
            else
                factor = obj.getGroupFactorIdx(idx, factor);
                ff = prod(obj.GMLMstructure.Groups(idx).dim_T(obj.GMLMstructure.Groups(idx).factor_idx == factor));
            end
        end
        
        function [rr_m] = dim_R_max(obj, idx) % max allowed tensor rank for a particular tensor coefficient group (given by idx). A max is set for pre-allocating GPU space.
            idx = obj.getGroupIdx(idx);
            rr_m = obj.GMLMstructure.Groups(idx).dim_R_max;
        end
        
        function [isShared] = isSharedRegressor(obj, idx, factor) % if the tensor dimension (given by dim) for a particular tensor coefficient group (given by idx) uses a shared regressor set, rather than locally defined, dense regressors for each trial
            idx = obj.getGroupIdx(idx);
            if(nargin < 3 || isempty(factor))
                isShared = zeros(obj.dim_D(idx),1);
                for ss = 1:obj.dim_D(idx)
                    isShared(ss) =  ~isempty(obj.GMLMstructure.Groups(idx).X_shared{ss});
                end
            else
                factor = obj.getGroupFactorIdx(idx, factor);
                isShared = ~isempty(obj.GMLMstructure.Groups(idx).X_shared{factor});
            end
        end
        
        function [hh] = dim_H(obj, idx) %number of hyperparams (that can be sampled by HMC): hyperparams on W and B if idx is not given, hyperparams on T and V if idx gives a tensor group, and all hyperparams in model if idx = -1 
            if(nargin < 2)
                %gets number of hyperparams for top-level prior
                if(~isempty(obj.GMLMstructure.prior))
                    hh = obj.GMLMstructure.prior.dim_H;
                else
                    hh = 0;
                end
            elseif(idx == -1)
                %gets total hyperparams in entire model
                hh = obj.dim_H();
                for jj = 1:obj.dim_J
                    hh = hh + obj.dim_H(jj);
                end
            else
                %gets number of hyperparams for group-level prior
                idx = obj.getGroupIdx(idx);
                if(~isempty(obj.GMLMstructure.Groups(idx).prior))
                    hh = obj.GMLMstructure.Groups(idx).prior.dim_H;
                    if(isa(hh, 'function_handle'))
                        hh = hh(obj.dim_R(idx));
                    end
                else
                    hh = 0;
                end
            end
        end
        
        function [hh] = dim_H_gibbs(obj, idx) %number of hyperparams (that are sampled by a gibbs step): hyperparams on W and B if idx is not given, hyperparams on T and V if idx gives a tensor group, and all hyperparams in model if idx = -1 
            if(nargin < 2)
                %gets number of hyperparams for top-level prior
                if(~isempty(obj.GMLMstructure.gibbs_step))
                    hh = obj.GMLMstructure.gibbs_step.dim_H;
                else
                    hh = 0;
                end
            elseif(idx == -1)
                %gets total hyperparams in entire model
                hh = obj.dim_H();
                for jj = 1:obj.dim_J
                    hh = hh + obj.dim_H(jj);
                end
            else
                %gets number of hyperparams for group-level prior
                idx = obj.getGroupIdx(idx);
                if(~isempty(obj.GMLMstructure.Groups(idx).gibbs_step))
                    hh = obj.GMLMstructure.Groups(idx).gibbs_step.dim_H;
                    if(isa(hh, 'function_handle'))
                        hh = hh(obj.dim_R(idx));
                    end
                else
                    hh = 0;
                end
            end
        end
        
        function [nn] = getNumberOfParameters(obj, idx)
            if(nargin < 2)
                nn = (obj.dim_B + 1)*obj.dim_P;
            elseif(idx == -1)
                nn = obj.getNumberOfParameters();
                for jj = 1:obj.dim_J
                    nn = nn + obj.getNumberOfParameters(jj);
                end
            else
                idx = obj.getGroupIdx(idx);
                nn = (sum(obj.dim_T(idx)) + obj.dim_P)*obj.dim_R(idx);
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
                if(strcmp(obj.GMLMstructure.Groups(jj).name, name))
                    idx = jj;
                    break;
                end
            end
            
            if(isnan(idx))
                error("Invalid group name!")
            end
        end
        
        %gets the index of a group dimension by its name
        function [idx] = getGroupDimIdx(obj, groupIdx, name) % checks out a dimension index given by "name" to see if it's correct for a group. If it is a string, then it looks for the dimension index number
            groupIdx = obj.getGroupIdx(groupIdx);
            if(isnumeric(name))
                if(name ~= fix(name) || name <= 0 || name > obj.dim_S(groupIdx))
                    error("Invalid group dim index!")
                end
                idx = name;
                return;
            elseif(~isstring(name) && ~ischar(name))
                error("Invalid group dim name!")
            end
            
            name = string(name);
            
            idx = nan;
            for ss = 1:obj.dim_S(groupIdx)
                if(strcmp(obj.Groups(groupIdx).dim_name(ss), name))
                    idx = ss;
                    break;
                end
            end
            
            if(isnan(idx))
                error("Invalid group dim name!")
            end
        end
        %gets the index of a group factor by its name
        function [idx] = getGroupFactorIdx(obj, groupIdx, name) % checks out a dimension index given by "name" to see if it's correct for a group. If it is a string, then it looks for the dimension index number
            groupIdx = obj.getGroupIdx(groupIdx);
            if(isnumeric(name))
                if(name ~= fix(name) || name <= 0 || name > obj.dim_D(groupIdx))
                    error("Invalid group factor index!")
                end
                idx = name;
                return;
            elseif(~isstring(name) && ~ischar(name))
                error("Invalid group factor name!")
            end
            
            name = string(name);
            
            idx = nan;
            for ss = 1:obj.dim_D(groupIdx)
                if(strcmp(obj.Groups(groupIdx).factor_name(ss), name))
                    idx = ss;
                    break;
                end
            end
            
            if(isnan(idx))
                error("Invalid group dim name!")
            end
        end
        
        %change the rank of a group
        function [obj] = setDimR(obj, groupIdx, dim_R_new)
            if(~isscalar(dim_R_new) || fix(dim_R_new) ~= dim_R_new || dim_R_new < 0)
                error("dim_R must be a non-negative integer");
            end
            groupIdx = obj.getGroupIdx(groupIdx);
            
            rr_m = obj.dim_R_max(groupIdx);
            if(dim_R_new > rr_m)
                error("given dim_R is greater than maximum allocated rank (dim_R_new = %d, dim_R_max = %d)", dim_R_new, rr_m);
            end
            
            obj.GMLMstructure.Groups(groupIdx).dim_R = dim_R_new;
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating parameter structures (Public)
    methods (Access = public)    
                
        %% METHOD [paramStruct] = getEmptyParamStruct()
        %	Builds a blank structure with all the GMLM parameters given all currently included groups.
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
        %
        %	returns:
        %       params [struct]        
        %           A blank (zero-filled) structure of the model parameters.
        %
        %           Fields for model parameters:
        %               W [dim_P x 1]      (real) : vector of constant rate parameters per neuron 
        %               B [dim_B x dim_P]  (real) : matrix of the linear coefficients
        %               H [dim_H x 1]      (real) : the hyperparameters for W (if requested)
        %
        %               Groups [J x 1] (struct): parameters for each tensor group
        %                   Fields for model parameters:
        %                       V [dim_P x dim_R[jj]] (real)       : The low-rank factors for the neuron weighting.
        %                       T [dim_S[jj] x 1]     (cell array) : the matrices of PARAFAC decomposed factors for the remaining
        %                                                            tensor dimenions.  Each is [dim_T[jj,ii] x dim_R[jj]]
        %                       H [dim_H[jj] x 1]      (real)      : the hyperparameters for this group (if requested)
        %
        %                       dim_names        (string array) : names of each dimension in T
        %                       name             (string)       : names of group
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
            
            %sets up top-level params
            params.W = zeros(obj.dim_P, 1, dataType);
            params.B = zeros(obj.dim_B, obj.dim_P, dataType);
            if(includeHyperparameters)
                params.H = zeros(obj.dim_H, 1, 'double');
                params.H_gibbs = zeros(obj.dim_H_gibbs, 1, 'double');
                params.Groups = struct('T', [], 'V', [], 'H', [], 'H_gibbs', [], 'dim_names', [], 'name', []);
            else
                params.Groups = struct('T', [], 'V', [], 'dim_names', [], 'name',[]);
            end
            
            %sets up each group
            J = obj.dim_J();
            for jj = 1:J
                rr = obj.dim_R(jj);
                params.Groups(jj).V = zeros(obj.dim_P, rr, dataType);
                
                params.Groups(jj).name = obj.GMLMstructure.Groups(jj).name;
                params.Groups(jj).dim_names = obj.GMLMstructure.Groups(jj).dim_names;
                
                %for each dimension
                ds = obj.dim_S(jj);
                params.Groups(jj).T = cell(ds, 1);
                for ss = 1:ds
                    params.Groups(jj).T{ss} = zeros(obj.dim_T(jj,ss), rr, dataType);
                end
                
                if(includeHyperparameters)
                    params.Groups(jj).H = zeros(obj.dim_H(jj), 1, 'double');
                    params.Groups(jj).H_gibbs = zeros(obj.dim_H_gibbs(jj), 1, 'double');
                end
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
            
            %% gets the log mean firing rate for each neuron 
            if(obj.isSimultaneousPopulation)
                total_bins = 0;
                total_spks = zeros(obj.dim_P,1);
                for tt = 1:obj.dim_M
                    total_bins = total_bins + size(obj.trials(tt).Y,1);
                    total_spks = total_spks + sum(obj.trials(tt).Y)';
                end
                log_rate_mu = log(total_spks) - log(total_bins) - log(obj.bin_size);
            else
                log_rate_mu = zeros(obj.dim_P,1);
                for pp = 1:obj.dim_P
                    total_bins = 0;
                    total_spks = 0;

                    for tt = 1:obj.dim_M
                        if(obj.trials(tt).neuron_idx == pp)
                            total_bins = total_bins + numel(obj.trials(tt).Y);
                            total_spks = total_spks + sum(obj.trials(tt).Y);
                        end
                    end
    %                 log_rate_mu(pp) = log((total_spks / total_bins)./obj.bin_size);
                    log_rate_mu(pp) = log(total_spks) - log(total_bins) - log(obj.bin_size);
                end
            end
            %%
            params.W(:) = randn(size(params.W))*std(log_rate_mu) + mean(log_rate_mu);
            params.B(:) = randn(size(params.B)) * 0.1;
            if(isfield(params, 'H'))
                if(isfield(obj.GMLMstructure.prior, 'generator'))
                    params.H(:) = obj.GMLMstructure.prior.generator();
                else
                    params.H(:) = randn(size(params.H));
                end
            end
            if(isfield(params, 'H_gibbs'))
                if(isfield(obj.GMLMstructure.gibbs_step, 'generator'))
                    params.H_gibbs(:) = obj.GMLMstructure.gibbs_step.generator();
                else
                    params.H_gibbs(:) = randn(size(params.H_gibbs));
                end
            end
            
            %for each group
            J = obj.dim_J();
            for jj = 1:J
                params.Groups(jj).V(:) = randn(size(params.Groups(jj).V));
                
                %for each dimension
                ds = obj.dim_S(jj);
                for ss = 1:ds
                    rt = randn(size(params.Groups(jj).T{ss}));
                    if(size(rt,2) <= size(rt,1))
                        params.Groups(jj).T{ss}(:) = orth(rt);
                    else
                        params.Groups(jj).T{ss}(:) = rt./sqrt(sum(rt.^2,1)); %in this case, the orth operation would cut out columns
                    end
                end
                
                %hyperparams
                if(isfield(params.Groups(jj), 'H'))
                    if(isfield(obj.GMLMstructure.Groups(jj).prior, 'generator'))
                        params.Groups(jj).H(:) = obj.GMLMstructure.Groups(jj).prior.generator(obj.dim_R(jj));
                    else
                        params.Groups(jj).H(:) = randn(size(params.Groups(jj).H));
                    end
                end
                if(isfield(params.Groups(jj), 'H_gibbs'))
                    if(isfield(obj.GMLMstructure.Groups(jj).gibbs_step, 'generator'))
                        params.Groups(jj).H_gibbs(:) = obj.GMLMstructure.Groups(jj).gibbs_step.generator(obj.dim_R(jj));
                    else
                        params.Groups(jj).H_gibbs(:) = randn(size(params.Groups(jj).H_gibbs));
                    end
                end
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
                if( ~isfield(params(ii), 'W') || ~isfield(params(ii), 'B'))
                    isValid(ii) = false;
                    continue;
                end

                if(~isvector(params(ii).W) || ~isnumeric(params(ii).W) || numel(params(ii).W) ~= obj.dim_P)
                    isValid(ii) = false;
                    continue;
                end

                if(~ismatrix(params(ii).B) || ~isnumeric(params(ii).B) || ~all(size(params(ii).B) == [obj.dim_B, obj.dim_P]))
                    isValid(ii) = false;
                    continue;
                end

                if(includeHyperparameters)
                    if(~isfield(params(ii), 'H') || ~isvector(params(ii).H) || ~isnumeric(params(ii).H) || numel(params(ii).H) ~= obj.dim_H)
                        isValid(ii) = false;
                        continue;
                    end
                    if(~isfield(params(ii), 'H_gibbs') || ~isvector(params(ii).H_gibbs) || ~isnumeric(params(ii).H_gibbs) || numel(params(ii).H_gibbs) ~= obj.dim_H_gibbs)
                        isValid(ii) = false;
                        continue;
                    end
                end

                if(~isfield(params(ii), 'Groups') || ~isstruct(params(ii).Groups) || ~isfield(params(ii).Groups, 'V') || ~isfield(params(ii).Groups, 'T') || numel(params(ii).Groups) ~= obj.dim_J)
                    isValid(ii) = false;
                    continue;
                end

                %for each group
                for jj = 1:obj.dim_J
                    if(~ismatrix(params(ii).Groups(jj).V) || ~isnumeric(params(ii).Groups(jj).V) || ~all(size(params(ii).Groups(jj).V) == [obj.dim_P, obj.dim_R(jj)]))
                        isValid(ii) = false;
                        continue;
                    end

                    %for each dimension
                    if(~iscell(params(ii).Groups(jj).T) || numel(params(ii).Groups(jj).T) ~= obj.dim_S(jj))
                        isValid = false;
                        return;
                    end
                    for ss = 1:obj.dim_S(jj)
                        if(~ismatrix(params(ii).Groups(jj).T{ss}) || ~isnumeric(params(ii).Groups(jj).T{ss}) || ~all(size(params(ii).Groups(jj).T{ss}) == [obj.dim_T(jj, ss), obj.dim_R(jj)]))
                            isValid(ii) = false;
                            continue;
                        end
                    end

                    %hyperparams if requested
                    if(includeHyperparameters)
                        if(~isfield(params(ii).Groups, 'H') || ~isvector(params(ii).Groups(jj).H) || ~isnumeric(params(ii).Groups(jj).H) || numel(params(ii).Groups(jj).H) ~= obj.dim_H(jj))
                            isValid(ii) = false;
                            continue;
                        end
                        if(~isfield(params(ii).Groups, 'H_gibbs') || ~isvector(params(ii).Groups(jj).H_gibbs) || ~isnumeric(params(ii).Groups(jj).H_gibbs) || numel(params(ii).Groups(jj).H_gibbs) ~= obj.dim_H_gibbs(jj))
                            isValid(ii) = false;
                            continue;
                        end
                    end
                end
            end
        end
        
        %unfolds the tensor structure of the Ts
        % A regressor F for a factor is the Khatri-Rao product of the T's for that factor
        function [F] = getF(obj, params, idx, factor)
            idx = obj.getGroupIdx(idx);
            if(nargin < 4)
                F = cell(obj.dim_D(idx), 1);
                for ff = 1:obj.dim_D(idx)
                    F{ff} = obj.getF(params, idx, ff);
                end
            else
                fis = sort(find(obj.getFactorIdxs(idx) == factor));
                F   = nan(obj.dim_F(idx, factor), obj.dim_R(idx));
                dim_R_c = size(F,2);
                
                for rr = 1:dim_R_c
                    F_r = params.Groups(idx).T{fis(1)}(:,rr);
                    for ss = 2:numel(fis)
                        F_r = kron(params.Groups(idx).T{fis(ss)}(:,rr), F_r);
                    end
                    F(:, rr) = F_r;
                end
            end
        end
        
        % spreads the component magnitudes among all modes
        %OLD VERSION: normalizes a param struct so that each Groups(:).T{:} holds only normal vectors (for identifiability in the MLE)
        function [params] = normalizeTensorParams(obj, params, V_only)
            if(nargin < 3)
                V_only = false;
            end
            for jj = 1:obj.dim_J
                R = obj.dim_R(jj);
                if(R > 0)
                    S = obj.dim_S(jj);
                    if(S > 1)
                        mm = nan(obj.dim_S(jj)+1,R);
                        for ss = 1:obj.dim_S(jj)
                            mm(ss,:) = sqrt(sum(params.Groups(jj).T{ss}.^2, 1));
                        end
                        mm(end,:) = sqrt(sum(params.Groups(jj).V.^2, 1));
                        if(~V_only)
                            mm_0 = prod(mm,1).^(1/(S+1));
                            mm = mm_0./mm;
    
                            for ss = 1:S
                                params.Groups(jj).T{ss} = params.Groups(jj).T{ss} .*mm(ss,:);
                            end
                            params.Groups(jj).V = params.Groups(jj).V.*mm(end,:);
                        else
                            for ss = 1:S
                                params.Groups(jj).T{ss} = params.Groups(jj).T{ss} ./mm(ss,:);
                            end
                            mm_0 = prod(mm(1:S,:),1);
                            params.Groups(jj).V = params.Groups(jj).V.*mm_0;

                        end
                    else
                        [u,s,v] = svd(params.Groups(jj).T{1} * params.Groups(jj).V');
                        if(~V_only)
                            sq = sqrt(s(1:R, 1:R));
                            params.Groups(jj).T{1} = u(:, 1:R) * sq;
                            params.Groups(jj).V = v(:, 1:R) * sq;
                        else
                            params.Groups(jj).T{1} = u(:, 1:R);
                            params.Groups(jj).V = v(:, 1:R) * s(1:R, 1:R);
                        end
                    end
                end
            end
        end
    end    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for computing rate, log likelihood, and prior on Host (Public)
    % These functions weren't designed to be computationally efficient (could make this MUCH faster), but as a readable check for the optimized code.
    % Fitting would take forever using this function (but it's useful for looking at the rates of a fitted model)
    methods (Access = public)
        function [log_rate_per_trial] = computeLogRate(obj, params)
            %setup any shared regressors
            shared_regressors = struct('XF', cell(obj.dim_J, 1), 'F', []);

            
            J = obj.dim_J;
            params_0 = params;
            for jj = 1:J
                if(isfield(obj.GMLMstructure.Groups(jj), "scaleParams") && ~isempty(obj.GMLMstructure.Groups(jj).scaleParams))
                    params.Groups(jj) = obj.GMLMstructure.Groups(jj).scaleParams(params_0.Groups(jj));
                end
            end
            if(isfield(obj.GMLMstructure, "scaleParams") && ~isempty(obj.GMLMstructure.scaleParams))
                params = obj.GMLMstructure.scaleParams(params);
            end
           
            for jj = 1:obj.dim_J       
                shared_regressors(jj).F  = obj.getF(params, jj);
                shared_regressors(jj).XF = cell(obj.dim_S(jj), 1);
                for ff = 1:obj.dim_D(jj)
                    if(obj.isSharedRegressor(jj, ff))
                        shared_regressors(jj).XF{ff} = obj.GMLMstructure.Groups(jj).X_shared{ff} * shared_regressors(jj).F{ff};
                    end
                end   
            end
            
            %for each trial
            log_rate_per_trial = struct('log_rate', cell(size(obj.trials)));
            for mm = 1:obj.dim_M
                dim_N = size(obj.trials(mm).Y, 1);
                if(obj.isSimultaneousPopulation)
                    %add linear and constant term
                    if(obj.GMLMstructure.dim_B > 0)
                        if(size(obj.trials(mm).X_lin, 3) == 1)
                            log_rate_per_trial(mm).log_rate = obj.trials(mm).X_lin * params.B + params.W(:)';
                        else
                            log_rate_per_trial(mm).log_rate = nan(dim_N, obj.dim_P());
                            for pp = 1:obj.dim_P()
                                log_rate_per_trial(mm).log_rate(:,pp) = obj.trials(mm).X_lin(:, :, pp) * params.B(:, pp) + params.W(pp);
                            end
                        end
                    else
                        log_rate_per_trial(mm).log_rate = params.W(:)';
                    end
                else
                    neuron_idx = obj.trials(mm).neuron_idx;

                    %add linear and constant term
                    if(obj.GMLMstructure.dim_B > 0)
                        log_rate_per_trial(mm).log_rate = obj.trials(mm).X_lin * params.B(:, neuron_idx) + params.W(neuron_idx);
                    else
                        log_rate_per_trial(mm).log_rate = params.W(neuron_idx);
                    end
                end
                
                %add each group
                for jj = 1:obj.dim_J

                   
                    
                    G = ones(dim_N, obj.dim_R(jj), obj.dim_A(jj));
                    %for each event
                    for aa = 1:obj.dim_A(jj)
                        %get each dimension for the tensor components
                        for ff = 1:obj.dim_D(jj)
                            if(obj.isSharedRegressor(jj, ff))
                                iX_c = obj.trials(mm).Groups(jj).iX_shared{ff}(:, aa);
                                vv = iX_c > 0 & iX_c <= size(shared_regressors(jj).XF{ff}, 1);

                                G(~vv, :, aa) = 0;
                                G( vv, :, aa) = G(vv, :, aa) .* shared_regressors(jj).XF{ff}(iX_c(vv), :);
                            else
                                aa_c = min(aa, size(obj.trials(mm).Groups(jj).X_local{ff},3));
                                
                                G(:, :, aa) = G(:, :, aa) .* (obj.trials(mm).Groups(jj).X_local{ff}(:, :, aa_c) * shared_regressors(jj).F{ff});
                            end
                        end
                    end

                    % sum over events
                    G = sum(G, 3);
                    
                    % linearly weight the components and add to rate
                    if(obj.isSimultaneousPopulation)
                        log_rate_per_trial(mm).log_rate = G * params.Groups(jj).V' + log_rate_per_trial(mm).log_rate;
                    else
                        log_rate_per_trial(mm).log_rate = G * params.Groups(jj).V(neuron_idx,:)' + log_rate_per_trial(mm).log_rate;
                    end
                end
            end
        end
        
        function [log_like, log_like_per_trial] = computeLogLikelihoodHost(obj, params, opts)
            log_like_per_trial = obj.computeLogRate(params);
            
            if(nargin > 2 && isfield(opts, "trial_weights") && size(opts.trial_weights,1) == obj.dim_M)
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
                    
                    log_like_per_trial(mm).log_like_0  = zeros(size(obj.trials(mm).Y));
                    
                    for pp = 1:size(obj.trials(mm).Y, 2)
                        vv = obj.trials(mm).Y(:, pp) >= 0; %check for any censored values
                        rr = log_like_per_trial(mm).log_rate(vv, pp) + log(obj.bin_size);
                        log_like_per_trial(mm).log_like_0(vv, pp) = -exp(rr) + rr.*obj.trials(mm).Y(vv, pp) - (gammaln(obj.trials(mm).Y(vv, pp)+1));
                    end
                    log_like_per_trial(mm).log_like   = tw(mm,:).*sum(log_like_per_trial(mm).log_like_0, 1);
                elseif(ll_idx == 2) %sqErr
                    log_like_per_trial(mm).log_like = -0.5*tw(mm,:).*sum((log_like_per_trial(mm).log_rate(:) - obj.trials(mm).Y(:)).^2, 1);

                elseif(ll_idx == 3) %truncated Poisson
                    log_like_per_trial(mm).log_like_0  = zeros(size(obj.trials(mm).Y));
                    
                    for pp = 1:size(obj.trials(mm).Y, 2)
                        rr = log_like_per_trial(mm).log_rate(:, pp) + log(obj.bin_size);
                        

                        log_like_per_trial(mm).log_like_0(:, pp) = kgmlm.utils.truncatedPoiss(rr, obj.trials(mm).Y(:, pp) );

                    end
                    log_like_per_trial(mm).log_like   = tw(mm,:).*sum(log_like_per_trial(mm).log_like_0, 1);
                elseif(ll_idx == 4) %poisson sofrec
                    log_like_per_trial(mm).log_like_0  = zeros(size(obj.trials(mm).Y));
                    
                    for pp = 1:size(obj.trials(mm).Y, 2)
                        vv = obj.trials(mm).Y(:, pp) >= 0; %check for any censored values
                        rr = kgmlm.utils.softrec(log_like_per_trial(mm).log_rate(vv, pp)); 
                        log_like_per_trial(mm).log_like_0(vv, pp) = -rr* obj.bin_size + (log(rr) + log(obj.bin_size)).*obj.trials(mm).Y(vv, pp) - (gammaln(obj.trials(mm).Y(vv, pp)+1));
                    end
                    log_like_per_trial(mm).log_like   = tw(mm,:).*sum(log_like_per_trial(mm).log_like_0, 1);
                else
                    error("invalid likelihood setting");
                end
            end
            
            log_like = sum([log_like_per_trial(:).log_like], 'all');
        end
        
        % computes the log prior (as far as this function is aware, this is always on host)
        %   can add the result to a results struct or create a new struct
        function [results] = computeLogPrior(obj, params, opts, results, priorOnly)
            if(nargin < 4 || isempty(results))
                results = obj.getEmptyResultsStruct(opts);
            end
            if(nargin < 5)
                priorOnly = true;
            end
            
            results.log_prior = 0;
            
            %% if a prior for W,B exists, adds it
            if(isfield(obj.GMLMstructure, "prior") && ~isempty(obj.GMLMstructure.prior))
                results = obj.GMLMstructure.prior.log_prior_func(params, results, priorOnly);
                results.log_prior = results.log_prior + results.log_prior_WB;
            end
            
            %% for each group
            for jj = 1:obj.dim_J
                %% if prior for V,T exists, adds it
                if(isfield(obj.GMLMstructure.Groups(jj), "prior") && ~isempty(obj.GMLMstructure.Groups(jj).prior))
                    if(obj.GMLMstructure.Groups(jj).prior.log_prior_func([], [], [], priorOnly) > 1)
                        results = obj.GMLMstructure.Groups(jj).prior.log_prior_func(params, results, jj, priorOnly);
                    else
                        results.Groups(jj) = obj.GMLMstructure.Groups(jj).prior.log_prior_func(params, results, jj, priorOnly);
                    end
                    results.log_prior = results.log_prior + results.Groups(jj).log_prior_VT;
                end
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
        %               dW :
        %               dB : 
        %               dH : (if requested)
        %
        %               Groups [J x 1] (struct): parameters for each tensor group
        %                   Fields for model parameters:
        %                       dV 
        %                       dT [dim_S[jj] x 1]  : can select each dimension independently
        %                       dH : (if requested)
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
            addParameter(p, 'trial_weights', [], @(aa)(islogical(aa) || isempty(aa) || numel(aa) == obj.dim_M));
            parse(p,varargin{:});
            
            enableAll = p.Results.enableAll;
            includeHyperparameters = p.Results.includeHyperparameters;
            trial_weights = p.Results.trial_weights;
            
            %%
            %sets up top-level params
            opts.trialLL = true;
            opts.dW = enableAll;
            opts.dB = enableAll;
            if(includeHyperparameters)
                opts.dH = enableAll;
                opts.H_gibbs = enableAll;
                opts.Groups = struct('dT', [], 'dV', [], 'dH', [], 'H_gibbs', []);
            else
                opts.Groups = struct('dT', [], 'dV', []);
            end
            
            opts.trial_weights = [];
            if(~isempty(trial_weights) && ~(isscalar(trial_weights) && islogical(trial_weights) && ~trial_weights))
                if(numel(trial_weights) == obj.dim_M)
                    opts.trial_weights = trial_weights(:);
                elseif(obj.isSimultaneousPopulation && size(trial_weights,1) == obj.dim_M && size(trial_weights,2) == obj.dim_P())
                    opts.trial_weights = trial_weights;
                elseif(islogical(trial_weights) && trial_weights)
                    opts.trial_weights = ones(obj.dim_M, 1);
                end
            end
            
            if(~obj.gpuDoublePrecision)
                opts.trial_weights = single(opts.trial_weights);
            end
            
            %sets up each group
            J = obj.dim_J();
            for jj = 1:J
                opts.Groups(jj).dV = enableAll;
                opts.Groups(jj).dT = true(obj.dim_S(jj), 1) & enableAll;
                
                if(includeHyperparameters)
                    opts.Groups(jj).dH = enableAll;
                    opts.Groups(jj).H_gibbs = enableAll;
                end
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
            if(~isfield(opts, 'dW') || ~isfield(opts, 'dB'))
                isValid = false;
                return;
            end
            
            if(~isscalar(opts.dW) || ~islogical(opts.dW))
                isValid = false;
                return;
            end
            
            if(~isscalar(opts.dB) || ~islogical(opts.dB))
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
                if(~isfield(opts, 'H_gibbs') || ~isscalar(opts.H_gibbs) || ~islogical(opts.H_gibbs))
                    isValid = false;
                    return;
                end
            end
            
            if(~isfield(opts, 'Groups') || ~isstruct(opts.Groups) || ~isfield(opts.Groups, 'dV') || ~isfield(opts.Groups, 'dT') || numel(opts.Groups) ~= obj.dim_J)
                isValid = false;
                return;
            end
            
            %for each group
            for jj = 1:obj.dim_J
                if(~isscalar(opts.Groups(jj).dV) || ~islogical(opts.Groups(jj).dV))
                    isValid = false;
                    return;
                end
                
                %for each dimension
                if(~isvector(opts.Groups(jj).dT) || ~islogical(opts.Groups(jj).dT) || numel(opts.Groups(jj).dT) ~= obj.dim_S(jj))
                    isValid = false;
                    return;
                end
                
                %hyperparams if requested
                if(includeHyperparameters)
                    if(~isfield(opts.Groups, 'dH') || ~isscalar(opts.Groups(jj).dH) || ~islogical(opts.Groups(jj).dH) )
                        isValid = false;
                        return;
                    end
                    if(~isfield(opts.Groups, 'H_gibbs') || ~isscalar(opts.Groups(jj).H_gibbs) || ~islogical(opts.Groups(jj).H_gibbs) )
                        isValid = false;
                        return;
                    end
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
            ww = zeros(obj.max_params + obj.dim_H(-1), 1); %pre-allocates a vector large enough to store everything possible
            ctr = 0; % counter to keep place in ww
            
            if(opts.dW)
                ss_c = numel(params.W);
                ww(ctr+(1:ss_c)) = params.W(:);
                ctr = ctr + ss_c;
            end
            if(opts.dB)
                ss_c = numel(params.B);
                ww(ctr+(1:ss_c)) = params.B(:);
                ctr = ctr + ss_c;
            end
            %each group
            for jj = 1:obj.dim_J
                if(opts.Groups(jj).dV)
                    ss_c = numel(params.Groups(jj).V);
                    ww(ctr+(1:ss_c)) = params.Groups(jj).V(:);
                    ctr = ctr + ss_c;
                end
                for ss = 1:obj.dim_S(jj)
                    if(opts.Groups(jj).dT(ss))
                        ss_c = numel(params.Groups(jj).T{ss});
                        ww(ctr+(1:ss_c)) = params.Groups(jj).T{ss}(:);
                        ctr = ctr + ss_c;
                    end
                end
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(params, 'H') && opts.dH)
                ss_c = numel(params.H);
                ww(ctr+(1:ss_c)) = params.H(:);
                ctr = ctr + ss_c;
            end
            for jj = 1:obj.dim_J
                if(isfield(opts.Groups, 'dH') && isfield(params.Groups, 'H') && opts.Groups(jj).dH)
                    ss_c = numel(params.Groups(jj).H);
                    ww(ctr+(1:ss_c)) = params.Groups(jj).H(:);
                    ctr = ctr + ss_c;
                end
            end
            
            ww = ww(1:ctr); %cuts down vector to the pieces that were filled
        end
        
        % given a vectorized set of params for opts in ww, and a full param struct, devectorizes
        function [params] = devectorizeParams(obj, ww, params, opts)
            ctr = 0; % counter to keep place in ww
            
            if(opts.dW)
                ss_c = numel(params.W);
                params.W(:) = ww(ctr+(1:ss_c));
                ctr = ctr + ss_c;
            end
            if(opts.dB)
                ss_c = numel(params.B);
                params.B(:) = ww(ctr+(1:ss_c));
                ctr = ctr + ss_c;
            end
            %each group
            for jj = 1:obj.dim_J
                if(opts.Groups(jj).dV)
                    ss_c = numel(params.Groups(jj).V);
                    params.Groups(jj).V(:) = ww(ctr+(1:ss_c));
                    ctr = ctr + ss_c;
                end
                for ss = 1:obj.dim_S(jj)
                    if(opts.Groups(jj).dT(ss))
                        ss_c = numel(params.Groups(jj).T{ss});
                        params.Groups(jj).T{ss}(:) = ww(ctr+(1:ss_c));
                        ctr = ctr + ss_c;
                    end
                end
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(params, 'H') && opts.dH)
                ss_c = numel(params.H);
                params.H(:) = ww(ctr+(1:ss_c));
                ctr = ctr + ss_c;
            end
            for jj = 1:obj.dim_J
                if(isfield(opts.Groups, 'dH') && isfield(params.Groups, 'H') && opts.Groups(jj).dH)
                    ss_c = numel(params.Groups(jj).H);
                    params.Groups(jj).H(:) = ww(ctr+(1:ss_c));
                    ctr = ctr + ss_c;
                end
            end
        end
        
        % given a vectorized set of params for opts in ww, and a full param struct, devectorizes
        function [ww] = vectorizeResults(obj, results, opts)
            ww = zeros(obj.max_params + obj.dim_H(-1), 1); %pre-allocates a vector large enough to store everything possible
            ctr = 0; % counter to keep place in ww
            
            if(opts.dW)
                ss_c = numel(results.dW);
                ww(ctr+(1:ss_c)) = results.dW(:);
                ctr = ctr + ss_c;
            end
            if(opts.dB)
                ss_c = numel(results.dB);
                ww(ctr+(1:ss_c)) = results.dB(:);
                ctr = ctr + ss_c;
            end
            %each group
            for jj = 1:obj.dim_J
                if(opts.Groups(jj).dV)
                    ss_c = numel(results.Groups(jj).dV);
                    ww(ctr+(1:ss_c)) = results.Groups(jj).dV(:);
                    ctr = ctr + ss_c;
                end
                for ss = 1:obj.dim_S(jj)
                    if(opts.Groups(jj).dT(ss))
                        ss_c = numel(results.Groups(jj).dT{ss});
                        ww(ctr+(1:ss_c)) = results.Groups(jj).dT{ss}(:);
                        ctr = ctr + ss_c;
                    end
                end
            end
            
            %do hyperparams last
            if(isfield(opts, 'dH') && isfield(results, 'dH') && opts.dH)
                ss_c = numel(results.dH);
                ww(ctr+(1:ss_c)) = results.dH(:);
                ctr = ctr + ss_c;
            end
            for jj = 1:obj.dim_J
                if(isfield(opts.Groups, 'dH') && isfield(results.Groups, 'dH') && opts.Groups(jj).dH)
                    ss_c = numel(results.Groups(jj).dH);
                    ww(ctr+(1:ss_c)) = results.Groups(jj).dH(:);
                    ctr = ctr + ss_c;
                end
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
        %       deviceNumbers (int) : array (or scalar) of GPUs to send the GMLM to
        %       
        %   optional input key/value pairs:
        %       useDoublePrecision (bool; default=TRUE) : set to false if you want to use single precision
        %       max_trials_for_sparse_run      (int; default=128)   : number of trials that can be run sparsely on each gpu. If the given trial_weights is sparse, this can 
        %                                                             speed up computation (i.e., for SGD). Any run where sum(trial_weights[on this block] != 0) > max_trials_for_sparse_run
        %                                                             will take the same time as running the whole set. This can be a scalar or vector to match deviceNumbers 
        %                                                             if you want to have different numbers of blocks on different GPUs.
        %
        %   pointer to the c++ GMLM object is stored in obj.gpuObj_ptr.
        %
        function [obj] = toGPU(obj, deviceNumbers, varargin)
            if(obj.isOnGPU())
                error("GMLM is already loaded to GPU. Clear before re-loading. If problems occured, clear manually by using 'kcResetDevice(gpuNums)' and 'clear mex'.");
            end
            
            %% check device numbers
            if((nargin < 2 || isempty(deviceNumbers)) && gpuDeviceCount() == 1)
                %if computer has only one GPU, can be set by default 
                deviceNumbers = 0;
            end
            if(~isnumeric(deviceNumbers) || ~all(fix(deviceNumbers) == deviceNumbers, 'all') || ~all(deviceNumbers >= 0 & deviceNumbers < gpuDeviceCount()))
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
            
            %% sets up GMLMstructure to send to GPU in correct datatypes and 0 indexing
            GMLMstructure_GPU = obj.GMLMstructure;
            
            GMLMstructure_GPU.isSimultaneousPopulation = obj.isSimultaneousPopulation;
            GMLMstructure_GPU.dim_P = uint64(obj.dim_P); 
            GMLMstructure_GPU.max_trial_length = uint64(max(arrayfun(@(aa) size(aa.Y,1), obj.trials)));
            GMLMstructure_GPU.dim_B       = uint64(obj.dim_B);
            if(useDoublePrecision)
                GMLMstructure_GPU.binSize = double(obj.bin_size);
            else
                GMLMstructure_GPU.binSize = single(obj.bin_size);
            end
            
            GMLMstructure_GPU.logLikeSettings = 0;
            for ii = 1:numel(obj.validLogLikeTypes)
                if(strcmpi(obj.logLikeType, obj.validLogLikeTypes(ii)))
                    GMLMstructure_GPU.logLikeSettings = ii - 1;
                    break;
                end
            end
            GMLMstructure_GPU.logLikeSettings = int32(GMLMstructure_GPU.logLikeSettings);
            
            GMLMstructure_GPU.logLikeParams = [];
            if(useDoublePrecision)
                GMLMstructure_GPU.logLikeParams = double(GMLMstructure_GPU.logLikeParams);
            else
                GMLMstructure_GPU.logLikeParams = single(GMLMstructure_GPU.logLikeParams);
            end

            for jj = 1:obj.dim_J
                GMLMstructure_GPU.Groups(jj).dim_A      = uint64(obj.dim_A(jj));
                GMLMstructure_GPU.Groups(jj).dim_R_max  = uint64(obj.dim_R_max(jj));
                GMLMstructure_GPU.Groups(jj).dim_T      = uint64(obj.dim_T(jj));
                GMLMstructure_GPU.Groups(jj).factor_idx = uint32(obj.getFactorIdxs(jj) - 1);
                
                %for each dimension
                for ff = 1:obj.dim_D(jj)
                    if(~(obj.isSharedRegressor(jj, ff)))
                        GMLMstructure_GPU.Groups(jj).X_shared{ff} = [];
                    end
                    %makes sure local regressors are correct floating point type
                    if(useDoublePrecision && ~isa(GMLMstructure_GPU.Groups(jj).X_shared{ff}, 'double'))
                        GMLMstructure_GPU.Groups(jj).X_shared{ff} = double(GMLMstructure_GPU.Groups(jj).X_shared{ff});
                    elseif(~useDoublePrecision && ~isa(GMLMstructure_GPU.Groups(jj).X_shared{ff}, 'single'))
                        GMLMstructure_GPU.Groups(jj).X_shared{ff} = single(GMLMstructure_GPU.Groups(jj).X_shared{ff});
                    end
                end
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
                
                %sets trial_idx (int32 and 0 indexed)
                idxs = num2cell(uint32(trial_indices-1));
                [trialBlocks(bb).trials(:).trial_idx] = idxs{:};
                
                %for each trial
                for mm = 1:numel(trialBlocks(bb).trials)
                    % makes sure spike count is floating point type (NO LONGER INT)
                    if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Y, 'double'))
                        trialBlocks(bb).trials(mm).Y = double(trialBlocks(bb).trials(mm).Y);
                    elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Y, 'single'))
                        trialBlocks(bb).trials(mm).Y = single(trialBlocks(bb).trials(mm).Y);
                    end
                    
                    % makes sure X_lin is correct floating point type
                    if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).X_lin, 'double'))
                        trialBlocks(bb).trials(mm).X_lin = double(trialBlocks(bb).trials(mm).X_lin);
                    elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).X_lin, 'single'))
                        trialBlocks(bb).trials(mm).X_lin = single(trialBlocks(bb).trials(mm).X_lin);
                    end
                    
                    %sets neuron index to int32 and 0 indexed 
                    if(~obj.isSimultaneousPopulation)
                        trialBlocks(bb).trials(mm).neuron_idx = uint32(trialBlocks(bb).trials(mm).neuron_idx - 1);
                    end
                    
                    %for each Group
                    for jj = 1:obj.dim_J
                        %for each dimension
                        for ff = 1:obj.dim_D(jj) 
                            if(obj.isSharedRegressor(jj, ff))
                                %sets any shared regressor indices to int32 and 0 indexed
                                trialBlocks(bb).trials(mm).Groups(jj).iX_shared{ff} = int32(trialBlocks(bb).trials(mm).Groups(jj).iX_shared{ff} - 1);
                                trialBlocks(bb).trials(mm).Groups(jj).X_local{ff}   = [];
                            else
                                trialBlocks(bb).trials(mm).Groups(jj).iX_shared{ff} = int32([]);
                            end
                            %makes sure local regressors are correct floating point type
                            if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Groups(jj).X_local{ff}, 'double'))
                                trialBlocks(bb).trials(mm).Groups(jj).X_local{ff} = double(trialBlocks(bb).trials(mm).Groups(jj).X_local{ff});
                            elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Groups(jj).X_local{ff}, 'single'))
                                trialBlocks(bb).trials(mm).Groups(jj).X_local{ff} = single(trialBlocks(bb).trials(mm).Groups(jj).X_local{ff});
                            end
                        end
                    end
                end
            end
            
            %% call mex with (GMLMstructure_GPU, trialBlocks, useDoublePrecision), get pointer in return
            obj.gpuObj_ptr = kgmlm.CUDAlib.kcGMLM_mex_create(GMLMstructure_GPU, trialBlocks, useDoublePrecision);
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
                warning("GMLM is not loaded to GPU. Nothing to free.");
            else
                % call mex file to delete GPU object pointer
                kgmlm.CUDAlib.kcGMLM_mex_clear(obj.gpuObj_ptr, obj.gpuDoublePrecision);
                % erase pointer value
                obj.gpuObj_ptr = 0;
                obj.gpus = [];
            end
        end
        function [obj] = clearGPU_ptr(obj)
            warning("clearGPU_ptr is only for debugging.");
            obj.gpuObj_ptr = 0;
            obj.gpus = [];
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
                error("GMLM is not on GPU: must select runHost option to run on CPU.");
            end
            
            useAsync = true;
            params_0 = params;

            J = obj.dim_J;
            scaled_VT = false(J,1);
            scaleP = cell(J,1);
            for jj = 1:J
                if(isfield(obj.GMLMstructure.Groups(jj), "scaleParams") && ~isempty(obj.GMLMstructure.Groups(jj).scaleParams))

                    scaled_VT(jj) = true;
                    [params.Groups(jj), scaleP{jj}] = obj.GMLMstructure.Groups(jj).scaleParams(params_0.Groups(jj));

                    if(isfield(opts.Groups(jj), "dH"))
                        opts.Groups(jj).dV = opts.Groups(jj).dV | opts.Groups(jj).dH;
                        opts.Groups(jj).dT(:) = opts.Groups(jj).dT(:) | opts.Groups(jj).dH;
        
                        if(nargin > 3 && ~isempty(results))
                            if(opts.Groups(jj).dV && isempty(results.Groups(jj).dV))
                                error("Invalid results struct.");
                            end
                            for ss = 1:numel(opts.Groups(jj).dT)
                                if(opts.Groups(jj).dT(ss) && isempty(results.Groups(jj).dT{ss}))
                                    error("Invalid results struct.");
                                end
                            end
                        end
                    end
                end
            end
            scaled_WB = isfield(obj.GMLMstructure, "scaleParams") && ~isempty(obj.GMLMstructure.scaleParams);
            if(scaled_WB)
                [params, scaleP_WB] = obj.GMLMstructure.scaleParams(params);

                if(isfield(opts, "dH"))
                    opts.dB = opts.dB | opts.dH;
                    opts.dW = opts.dW | opts.dH;
    
                    if(nargin > 3 && ~isempty(results))
                        if(opts.dW &&  isempty(results.dW))
                            error("Invalid results struct.");
                        end
                        if(opts.dB && isempty(results.dB) && obj.dim_B > 0)
                            error("Invalid results struct.");
                        end
                    end
                end
            end

            if(runHost)
                if(nargin < 4 || isempty(results))
                    results = obj.getEmptyResultsStruct(opts);
                end
                results = obj.computeLogLikelihood_host_v2(params, opts, results);
            elseif(useAsync)
                % sends LL computation to GPU
                kgmlm.CUDAlib.kcGMLM_mex_computeLL_async(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, opts, opts.trial_weights);

                %sets up results
                if(nargin >= 4 && ~isempty(results))
                    results = obj.clearResultsStruct(results);
                else
                    results = obj.getEmptyResultsStruct(opts);
                end
    
                % gets GPU results
                kgmlm.CUDAlib.kcGMLM_mex_computeLL_gather(obj.gpuObj_ptr, obj.gpuDoublePrecision, results);
            else
                %allocate space for results
                if(nargin < 4 || isempty(results)) %#ok<UNRCH> 
                    results = obj.getEmptyResultsStruct(opts);
                end
                
                %send to GPU
                kgmlm.CUDAlib.kcGMLM_mex_computeLL(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, results, opts.trial_weights);
            end


            if(scaled_WB)
                results = obj.GMLMstructure.scaleDerivatives(results, params_0, true, scaleP_WB);
            end
            for jj = 1:J
                if(scaled_VT(jj))
                    results.Groups(jj) = obj.GMLMstructure.Groups(jj).scaleDerivatives(results.Groups(jj), params_0.Groups(jj), false, scaleP{jj});
                end
            end
            results.log_likelihood = sum(results.trialLL, 'all');
        end
        
        function [results] = computeLogPosterior(obj, params, opts, results, runHost)
            if(nargin < 5 || isempty(runHost))
                runHost = ~obj.isOnGPU();
            end
            if(~obj.isOnGPU() && ~runHost)
                error("GMLM is not on GPU: must select runHost option to run on CPU.");
            end

            useAsync = true;
            params_0 = params;


            J = obj.dim_J;
            scaled_VT = false(J,1);
            scaleP = cell(J,1);
            for jj = 1:J
                if(isfield(obj.GMLMstructure.Groups(jj), "scaleParams") && ~isempty(obj.GMLMstructure.Groups(jj).scaleParams))

                    scaled_VT(jj) = true;
                    [params.Groups(jj), scaleP{jj}] = obj.GMLMstructure.Groups(jj).scaleParams(params_0.Groups(jj));

                    if(isfield(opts.Groups(jj), "dH"))
                        opts.Groups(jj).dV = opts.Groups(jj).dV | opts.Groups(jj).dH;
                        opts.Groups(jj).dT(:) = opts.Groups(jj).dT(:) | opts.Groups(jj).dH;
        
                        if(nargin > 3 && ~isempty(results))
                            if(opts.Groups(jj).dV && isempty(results.Groups(jj).dV))
                                error("Invalid results struct.");
                            end
                            for ss = 1:numel(opts.Groups(jj).dT)
                                if(opts.Groups(jj).dT(ss) && isempty(results.Groups(jj).dT{ss}))
                                    error("Invalid results struct.");
                                end
                            end
                        end
                    end
                end
            end
            scaled_WB = isfield(obj.GMLMstructure, "scaleParams") && ~isempty(obj.GMLMstructure.scaleParams);
            if(scaled_WB)
                [params, scaleP_WB] = obj.GMLMstructure.scaleParams(params);

                if(isfield(opts, "dH"))
                    opts.dB = opts.dB | opts.dH;
                    opts.dW = opts.dW | opts.dH;
    
                    if(nargin > 3 && ~isempty(results))
                        if(opts.dW && isempty(results.dW))
                            error("Invalid results struct.");
                        end
                        if(opts.dB && isempty(results.dB) && obj.dim_B > 0)
                            error("Invalid results struct.");
                        end
                    end
                end
            end

            if(runHost)
                if(nargin < 4 || isempty(results))
                    results = obj.getEmptyResultsStruct(opts);
                end
                results = obj.computeLogLikelihood_host_v2(params, opts, results);
                results.log_likelihood = sum(results.trialLL, 'all');
                results = obj.computeLogPrior(params_0, opts, results, false);
            elseif(useAsync)
                % sends LL computation to GPU
                kgmlm.CUDAlib.kcGMLM_mex_computeLL_async(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, opts, opts.trial_weights);
                
                %adds the prior
                if(nargin >= 4 && ~isempty(results))
                    results = obj.clearResultsStruct(results);
                else
                    results = obj.getEmptyResultsStruct(opts);
                end
                results = obj.computeLogPrior(params_0, opts, results, false);
    
                % gets GPU results
                kgmlm.CUDAlib.kcGMLM_mex_computeLL_gather(obj.gpuObj_ptr, obj.gpuDoublePrecision, results);
                results.log_likelihood = sum(results.trialLL, 'all');
            else
                if(nargin >= 4 && ~isempty(results)) %#ok<UNRCH> 
                    results = obj.clearResultsStruct(results);
                else
                    results = obj.getEmptyResultsStruct(opts);
                end
                kgmlm.CUDAlib.kcGMLM_mex_computeLL(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, results, opts.trial_weights);
                results.log_likelihood = sum(results.trialLL, 'all');
                results = obj.computeLogPrior(params_0, opts, results, false);
            end

            if(scaled_WB)
                results = obj.GMLMstructure.scaleDerivatives(results, params_0, true, scaleP_WB);
            end
            for jj = 1:J
                if(scaled_VT(jj))
                    results.Groups(jj) = obj.GMLMstructure.Groups(jj).scaleDerivatives(results.Groups(jj), params_0.Groups(jj), true, scaleP{jj});
                end
            end
            
            %sums up results
            results.log_post = results.log_likelihood + results.log_prior;
        end
        
        function [nlog_like, ndl_like, params, results] = vectorizedNLL_func(obj, w_c, params, opts, results, makeDouble)
            if(nargout > 1)
                opts_0 = opts;
            else
                opts_0 = obj.getComputeOptionsStruct("enableAll", false, "includeHyperparameters", false);
            end
            if(nargin < 6 || isempty(makeDouble))
                makeDouble = true; % to make fminunc happy
            end
            opts_0.compute_trialLL = true;

            params = obj.devectorizeParams(w_c, params, opts);

            if(nargin < 5)
                results    = obj.computeLogLikelihood(params, opts_0);
            else
                results    = obj.computeLogLikelihood(params, opts_0, results);
            end
            nlog_like  = -results.log_likelihood;
            if(makeDouble && ~isa(nlog_like, "double"))
                nlog_like = double(nlog_like);
            end

            if(nargout > 1)
                ndl_like =  obj.vectorizeResults(results, opts_0);
                ndl_like = -ndl_like;
                if(makeDouble && ~isa(ndl_like, "double"))
                    ndl_like = double(ndl_like);
                end
            end
        end
        
        function [nlog_post, ndl_post, params, results] = vectorizedNLPost_func(obj, w_c, params, opts, results, makeDouble)
            if(nargout > 1)
                opts_0 = opts;
            else
                opts_0 = obj.getComputeOptionsStruct("enableAll", false, "includeHyperparameters", false);
            end
            if(nargin < 6 || isempty(makeDouble))
                makeDouble = true; % to make fminunc happy
            end
            opts_0.compute_trialLL = true;

            params = obj.devectorizeParams(w_c, params, opts);

            if(nargin < 5)
                results    = obj.computeLogPosterior(params, opts_0);
            else
                results    = obj.computeLogPosterior(params, opts_0, results);
            end
            nlog_post  = -results.log_post;
            if(makeDouble && ~isa(nlog_post, "double"))
                nlog_post = double(nlog_post);
            end

            if(nargout > 1)
                ndl_post =  obj.vectorizeResults(results, opts_0);
                ndl_post = -ndl_post;
                if(makeDouble && ~isa(ndl_post, "double"))
                    ndl_post = double(ndl_post);
                end
            end

        end
        
        function [nlog_prior, ndl_prior, params, results] = vectorizedNLPrior_func(obj, w_c, params, opts, results)
            if(nargout > 1)
                opts_0 = opts;
            else
                opts_0 = obj.getComputeOptionsStruct("enableAll", false, "includeHyperparameters", true);
            end

            params = obj.devectorizeParams(w_c, params, opts);

            if(nargin < 5)
                results    = obj.computeLogPrior(params, opts_0, [],      true);
            else
                results    = obj.computeLogPrior(params, opts_0, results, true);
            end
            nlog_prior  = -results.log_prior;

            if(nargout > 1)
                ndl_prior =  obj.vectorizeResults(results, opts_0);
                ndl_prior = -ndl_prior;
            end
        end
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %externally defined methods for inference
    methods (Access = public)
        [params_mle, results_mle, params_init]                         = computeMLE(obj, varargin);
        [params_map, results_map, hess_est]                            = computeMAP(obj, params_init, varargin);
        [params_mle, results_test_mle, results_train_mle, params_init] = computeMLE_crossValidated(obj, foldIDs, varargin);
        [params_map, results_test_map, results_train_map]              = computeMAP_crossValidated(obj, foldIDs, params_init, varargin);
        
        [samples, summary, HMC_settings, paramStruct, M] = runHMC_simple(obj, params_init, settings, varargin);
        [samples, samples_file_format, summary, HMC_settings, paramStruct, M] = runHMC_simpleLowerRAM(obj, params_init, settings, varargin);
        [] = saveSampleToFile(obj, samples_file, paramStruct, paramStruct_rescales, sample_idx, scaled_WB, scaled_VT, save_H, saveUnscaled);
        [paramStruct_rescaled] = rescaleParamStruct(obj, paramStruct, scaled_WB, scaled_VT);
        [samples_file_format, totalParams] = getSampleFileFormat(obj, TotalSamples, dataType_samples, paramStruct, scaled_WB, scaled_VT, saveUnscaled)
        [HMC_settings]                                   = setupHMCparams(obj, nWarmup, nSamples, debugSettings);


        [results] = computeLogLikelihood_host_v2(obj, params, opts, results);
        [log_rate, xx, R] = computeLogRate_host_v2(obj, params);
        [] = setupComputeStructuresHost(obj, reset, order);
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating computation result structures (Private)
    methods (Access = public)
        %% METHOD [paramStruct] = getEmptyResultsStruct(opts)
        %	Builds a blank structure with all the space for the derivatives of the GMLM parameters and the log likelihood
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
        %               log_prior_WB   (real)      : log prior for the W and B parameters (and H for those params)
        %
        %               trialLL [dim_M x 1] (real) : log likelihoods computed per trial
        %
        %               all derivates will be over both the likelihood and prior (if computing posterior, otherwise obviously only over likelihood)
        %
        %               dW [dim_P x 1]      (real) : 
        %               dB [dim_B x dim_P]  (real) : 
        %               dH [dim_H x 1]      (real) : 
        %
        %               Groups [J x 1] (struct): parameters for each tensor group
        %                   Fields for model parameters:
        %                       dV [dim_P x dim_R[jj]] (real)       
        %                       dT [dim_S[jj] x 1]     (cell array)  of dim_T[jj,ss] x dim_R[jj] matrices
        %                       dH [dim_H[jj] x 1]     (real)      
        %
        %
        function [results, dataType] = getEmptyResultsStruct(obj, opts)
            useDoublePrecision = obj.gpuDoublePrecision;%isa(opts.trial_weights, 'double');
            if(useDoublePrecision)
                dataType = 'double';
            else
                dataType = 'single';
            end
            
            if(~isfield(opts, 'trialLL') || opts.trialLL)
                results.trialLL = zeros(obj.dim_trialLL(), dataType);
            else
                results.trialLL = [];
            end
            
            %sets up top-level params
            if(opts.dW)
                results.dW = zeros(obj.dim_P, 1, dataType);
            else
                results.dW = zeros(0, 0, dataType);
            end
            if(opts.dB)
                results.dB = zeros(obj.dim_B, obj.dim_P, dataType);
            else
                results.dB = zeros(0, 0, dataType);
            end
            if(isfield(opts, 'dH'))
                if(opts.dH)
                    results.dH = zeros(obj.dim_H, 1, 'double');
                else
                    results.dH = zeros(0, 0, 'double');
                end
            end
            if(isfield(opts.Groups, 'dH'))
                results.Groups = struct('dT', [], 'dV', [], 'dH', []);
            else
                results.Groups = struct('dT', [], 'dV', []);
            end
            
            %sets up each group
            J = obj.dim_J();
            for jj = 1:J
                rr = obj.dim_R(jj);
                if(opts.Groups(jj).dV)
                    results.Groups(jj).dV = zeros(obj.dim_P, rr, dataType);
                else
                    results.Groups(jj).dV  = zeros(0, 0, dataType);
                end
                
                %for each dimension
                ds = obj.dim_S(jj);
                results.Groups(jj).dT = cell(ds, 1);
                for ss = 1:ds
                    if(opts.Groups(jj).dT(ss))
                        results.Groups(jj).dT{ss} = zeros(obj.dim_T(jj,ss), rr, dataType);
                    else
                        results.Groups(jj).dT{ss} = zeros(0, 0, dataType);
                    end
                end
                
                if(isfield(opts.Groups(jj), 'dH'))
                    if(opts.Groups(jj).dH)
                        results.Groups(jj).dH = zeros(obj.dim_H(jj), 1, 'double');
                    else
                        results.Groups(jj).dH = zeros(0, 0, 'double');
                    end
                end
                
                results.Groups(jj).log_prior_VT = nan;
            end
        end
        function [results] = clearResultsStruct(obj, results) %#ok<INUSL> 
            fs = ["trialLL", "dW", "dB", "dH", "log_likelihood", "log_prior", "log_prior_WB", "log_post"];
            for ii = 1:numel(fs)
                if(isfield(results, fs(ii)))
                    results.(fs(ii))(:) = 0;
                end
            end
            fs2 = ["dV", "dH", "log_prior_VT"];
            if(isfield(results.Groups, "dT"))
                for jj = 1:numel(results.Groups)
                    for ss = 1:numel(results.Groups(jj).dT)
                        results.Groups(jj).dT{ss}(:) = 0;

                    end
                end
            end
            for ii = 1:numel(fs2)
                if(isfield(results.Groups, fs2(ii)))
                    for jj = 1:numel(results.Groups)
                        results.Groups(jj).(fs2(ii))(:) = 0;
                    end
                end
            end
        end
        
        [optSetup, opts_empty] = getOptimizationSettings(obj, alternating_opt, trial_weights, optStruct, includeHyperparams);
    end
end