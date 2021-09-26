%% CLASS GMLMv2
%
%
%
%
classdef GMLMv2 < handle
    properties (GetAccess = public, SetAccess = private)
        GMLMstructure struct
        trials        struct
        neuronIdx
        bin_size 
        
        max_params %maximum number of parameters in the model (assuming max rank of all tensors) : does not include hyperparams
        
        gpuObj_ptr uint64 %uint64 pointer to the c++ object for the GMLM loaded to the GPU
        gpus       uint32 % array for which GPUs are in use
        gpuDoublePrecision logical %if current GPU loading is double precision (if false, then single)
    end
    
    %% Constructor
    methods 
        function obj = GMLMv2(GMLMstructure, trials, bin_size)
            if(nargin < 2 || nargin > 3)
                error("GMLM constructor: two struct inputs required (GMLMstructure and trials)");
            end
            %% check the GMLM structure
            if(~isstruct(GMLMstructure))
                error("GMLM constructor: input GMLMstructure must be a struct");
            end
            % check for linear dimension (non-negative integer)
            if(~isfield(GMLMstructure, "dim_K") || ~isnumeric(GMLMstructure.dim_K) || fix(GMLMstructure.dim_K) ~= GMLMstructure.dim_K || GMLMstructure.dim_K < 0)
                error("GMLM constructor: GMLMstructure must have property dim_K (dimensionalty of the linear term: non-negative integer)");
            end
            
            % check each group
            if(~isfield(GMLMstructure, "Groups") || ~isstruct(GMLMstructure.Groups) || isempty(GMLMstructure.Groups) || ~isfield(GMLMstructure.Groups, "dim_names") || ~isfield(GMLMstructure.Groups, "F_shared") || ~isfield(GMLMstructure.Groups, "dim_R_max") || ~isfield(GMLMstructure.Groups, "name"))
                error("GMLM constructor: GMLMstructure must contain non-empty struct array 'Groups' with field 'F_shared', 'dim_R_max', 'dim_names', and 'name'");
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
                
            for jj = 1:numel(GMLMstructure.Groups)
                %check max rank setting (positive integer)
                dim_R_max = GMLMstructure.Groups(jj).dim_R_max;
                if(~isnumeric(dim_R_max) || fix(dim_R_max) ~= dim_R_max || dim_R_max <= 0)
                    error("GMLM constructor: GMLMstructure.Groups().dim_R_max (maximum tensor rank) must be positive integer");
                end
                
                %check number of events (positive integer)
                dim_A = GMLMstructure.Groups(jj).dim_A;
                if(~isnumeric(dim_A) || fix(dim_A) ~= dim_A || dim_A <= 0)
                    error("GMLM constructor: GMLMstructure.Groups().dim_A (number of events for the coefficients) must be positive integer");
                end
                
                %check name (string or char)
                name = GMLMstructure.Groups(jj).name;
                if((~ischar(name) && ~isstring(name)) || isempty(name))
                    error("GMLM constructor: GMLMstructure.Groups().name must be a non-empty string");
                end
                
                
                %check F_shared
                F_shared = GMLMstructure.Groups(jj).F_shared;
                if(~iscell(F_shared))
                    error("GMLM constructor: GMLMstructure.Groups().F_shared must be a non-empty cell array (even if entries are empty, this still defines the tensor dimensions)");
                end
                if(~all(cellfun(@(aa) isnumeric(aa) & ismatrix(aa), F_shared), 'all'))
                    error("GMLM constructor: GMLMstructure.Groups().F_shared be a cell array containing numeric matrices (these may be empty, but require a second dimension to give the tensor size)");
                end
                
                %if contains a prior, should be empty or a function
                if(isfield(GMLMstructure.Groups, "prior"))
                    if(~isempty(GMLMstructure.Groups(jj).prior))
                        if(~isstruct(GMLMstructure.Groups(jj).prior) || ~isfield(GMLMstructure.Groups(jj).prior, 'log_prior_func') || ~isfield(GMLMstructure.Groups(jj).prior, 'dim_H'))
                            error("GMLM constructor: GMLMstructure.Groups().prior must be empty or structure with fields 'log_prior_func' and 'dim_H'");
                        end
                        
                        if(~isa(GMLMstructure.Groups(jj).prior.log_prior_func,'function_handle'))
                            error("GMLM constructor: GMLMstructure.Groups().prior.log_prior_func must be a function handle");
                        end
                        
                        if(~isscalar(GMLMstructure.Groups(jj).prior.dim_H) || GMLMstructure.Groups(jj).prior.dim_H < 0 || fix(GMLMstructure.Groups(jj).prior.dim_H) ~= GMLMstructure.Groups(jj).prior.dim_H)
                            error("GMLM constructor: GMLMstructure.Groups().prior.dim_H must be a non-negative integer");
                        end
                    end
                end
                
                %check dim_names (string)
                dim_names = GMLMstructure.Groups(jj).dim_names;
                if( ~isstring(dim_names) || numel(dim_names) ~= numel(F_shared))
                    error("GMLM constructor: GMLMstructure.Groups().dim_names must be a non-empty string array of the same size as GMLMstructure.Groups().F_shared");
                end
                
                %makes sure all dim names are unique
                if(numel(unique(dim_names)) ~= numel(dim_names))
                    error("GMLM constructor: tensor coefficient dimensions must have unique names (within each group: same names can exist in different groups)");
                end
            end
            
            %makes sure all group names are unique
            if(numel(unique([GMLMstructure.Groups(:).name])) ~= numel(GMLMstructure.Groups))
                error("GMLM constructor: tensor coefficient groups must have unique names");
            end
            
            %% check each trial to see if it matches the structure
            if(~isstruct(trials) || isempty(trials))
                error("GMLM constructor: input trials must be non-empty struct array");
            end
            % check for spike observations
               %exists                      for each trial     vector           numbers        contains obs.     is integers (spike counts -> this could be relaxed for more general models)
            if(~isfield(trials, "Y") || ~all(arrayfun(@(aa) isvector(aa.Y) & isnumeric(aa.Y) & ~all(aa.Y < 0, 'all') & all(fix(aa.Y) == aa.Y, 'all'), trials), 'all'))
                error("GMLM constructor: trials must have spike count vector Y, which contains only integers (may contain negatives to indicate censored bins, but each trial needs at least one observation!)");
            end
            
            %check for linear terms
            if(GMLMstructure.dim_K == 0 && isfield(trials, "F_lin"))
                %if F_lin is given, but expected to be empty, makes sure it's empty!
                if(~all(arrayfun(@(aa) isempty(aa.F_lin), trials), 'all'))
                    error("GMLM constructor: trials.F_lin expected to be empty (GMLMstructure.dim_K > 0), but found non-empty entries!");
                end
            elseif(GMLMstructure.dim_K > 0)
                %F_lin required
                if(~isfield(trials, "F_lin"))
                    error("GMLM constructor: trials requires field F_lin!");
                end
                %F_lin size must be correct
                if(~all(arrayfun(@(aa) isnumeric(aa.F_lin) & ismatrix(aa.F_lin) & size(aa.F_lin,1) == numel(aa.Y) & size(aa.F_lin, 2) == GMLMstructure.dim_K, trials), 'all'))
                    error("GMLM constructor: each trials(jj).F_lin should be a matrix of size numel(trials(jj).Y) x GMLMstructure.dim_K");
                end
            end
            
            %checks neuron number
            if(~all(arrayfun(@(aa) isnumeric(aa.neuron), trials), 'all') && ~all(arrayfun(@(aa) ischar(aa.neuron) | isstring(aa.neuron), trials), 'all'))
                error("GMLM constructor: invalid neuron indentifiers for trials (must be all numeric or all strings)");
            end
            
            %checks each group
                %consistent group numbers
            if(~all(arrayfun(@(aa) numel(aa.Groups) == numel(GMLMstructure.Groups), trials), 'all'))
                error("GMLM constructor: inconsistent number of groups in trials");
            end
            for jj = 1:numel(GMLMstructure.Groups)
                %checks for fields in each group
                if(~all(arrayfun(@(aa) isfield(aa.Groups(jj), "F_local") && isfield(aa.Groups(jj), "ind_A_shared") && iscell(aa.Groups(jj).F_local) && iscell(aa.Groups(jj).ind_A_shared) , trials), 'all'))
                    error("GMLM constructor: trials.Groups requires cell arrays in fields 'F_local' and 'ind_A_shared'.");
                end
                
                %checks for consistent number of dimensions
                dim_S = numel(GMLMstructure.Groups(jj).F_shared);
                if(~all(arrayfun(@(aa) numel(aa.Groups(jj).F_local) == dim_S && numel(aa.Groups(jj).ind_A_shared) == dim_S, trials), 'all'))
                    error("GMLM constructor: trials.Groups(jj).(F_local,ind_A_shared) must be cell array of the same size as GMLMstructure.Groups(jj).F_shared.");
                end
                
                %checks each dimension
                for ss = 1:dim_S
                    %when F_shared is populated
                    if(~isempty(GMLMstructure.Groups(jj).F_shared{ss}))
                        %check to see if F_local is empty
                        if(~all(arrayfun(@(aa) isempty(aa.Groups(jj).F_local{ss}), trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).F_local{ss} must be empty if GMLMstructure.Groups(jj).F_shared{ss} is populated.");
                        end
                
                        %check to see if ind_A is correct size
                        if(~all(arrayfun(@(aa) isnumeric(aa.Groups(jj).ind_A_shared{ss}) && ismatrix(aa.Groups(jj).ind_A_shared{ss}) && ...
                                               size(aa.Groups(jj).ind_A_shared{ss},1) == numel(aa.Y) && ... 
                                               size(aa.Groups(jj).ind_A_shared{ss},2) == GMLMstructure.Groups(jj).dim_A && ...
                                               all(aa.Groups(jj).ind_A_shared{ss} == fix(aa.Groups(jj).ind_A_shared{ss}), 'all') ...
                                               , trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).ind_A_shared{ss} must be matrix of integers of size (numel(trials.Y) x GMLMstructure.Groups(jj).dim_A) when GMLMstructure.Groups(jj).F_shared{ss} is populated.");
                        end
                    else
                    %when F_shared is empty
                        %if F_shared does not contain the dim_T[jj,ss] information, sets by using first trial
                        if(size(GMLMstructure.Groups(jj).F_shared{ss}, 2) == 0)
                            dim_T_c = size(trials(1).Groups(jj).F_local{ss}, 2);
                            if(dim_T_c ~= 0)
                                GMLMstructure.Groups(jj).F_shared{ss} = zeros([0 dim_T_c]);
                            else
                                error("GMLM constructor: tensor dimension is 0 - must be positive!");
                            end
                        end
                    
                        %check to see if F_local is correct size
                        if(~all(arrayfun(@(aa) isnumeric(aa.Groups(jj).F_local{ss}) && ndims(aa.Groups(jj).F_local{ss}) <= 3 && ...
                                               size(aa.Groups(jj).F_local{ss},1) == numel(aa.Y) && ... 
                                               size(aa.Groups(jj).F_local{ss},2) == size(GMLMstructure.Groups(jj).F_shared{ss}, 2) && ... 
                                               size(aa.Groups(jj).F_local{ss},3) == GMLMstructure.Groups(jj).dim_A ...
                                               , trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).F_local{ss} must be matrix of integers of size (numel(trials.Y) x size(GMLMstructure.Groups(jj).F_shared{ss},2)) when GMLMstructure.Groups(jj).F_shared{ss} is empty (F_shared still holds dimension info, but number of rows is 0).");
                        end

                        %check to see if ind_A is empty
                        if(~all(arrayfun(@(aa) isempty(aa.Groups(jj).ind_A_shared{ss}), trials), 'all'))
                            error("GMLM constructor: trials.Groups(jj).ind_A_shared{ss} must be empty if GMLMstructure.Groups(jj).F_shared{ss} is empty (F_local is populated).");
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
            obj.trials = trials;
            
            % check for prior, otherwise is null by default
            if(isfield(GMLMstructure, "prior"))
                obj.GMLMstructure.prior = GMLMstructure.prior;
            else
                obj.GMLMstructure.prior = [];
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
            end
            
            %sets local neuron index
            [obj.neuronIdx, ~, idxs] = unique([trials(:).neuron]);
            for tt = 1:numel(obj.trials)
                obj.trials(tt).neuron_idx = idxs(tt);
            end
            
            %computes maximum number of parameters in the model
            obj.max_params = obj.dim_P + obj.dim_P*obj.dim_K;
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
        end
    end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for dimension information (Public)
    % mostly to help me with my notation
    methods (Access = public)
        function [pp] = dim_P(obj) %number of neurons
            pp = numel(obj.neuronIdx);
        end
        
        function [dd] = dim_J(obj) % number of tensor coefficient groups
            dd = numel(obj.GMLMstructure.Groups);
        end
        
        function [kk] = dim_K(obj) % size of linear term
            kk = obj.GMLMstructure.dim_K;
        end
        
        function [mm] = dim_M(obj) % dim_M is the number of trials
            mm = numel(obj.trials);
        end
        
        function [aa] = dim_A(obj, idx) % number of 'events' for a particular tensor coefficient group (given by idx)
            idx = obj.getGroupIdx(idx);
            aa = obj.GMLMstructure.Groups(idx).dim_A;
        end
        
        function [ss] = dim_S(obj, idx) % tensor order (minus the dimension for the neuron loading weights) for a particular tensor coefficient group (given by idx)
            idx = obj.getGroupIdx(idx);
            ss = numel(obj.GMLMstructure.Groups(idx).F_shared);
        end
        
        function [rr] = dim_R(obj, idx) % current tensor rank (between 0 and dim_R_max) for a particular tensor coefficient group (given by idx)
            idx = obj.getGroupIdx(idx);
            rr = obj.GMLMstructure.Groups(idx).dim_R;
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
                tt = size(obj.GMLMstructure.Groups(idx).F_shared{dim}, 2);
            end
        end
        
        function [rr_m] = dim_R_max(obj, idx) % max allowed tensor rank for a particular tensor coefficient group (given by idx). A max is set for pre-allocating GPU space.
            idx = obj.getGroupIdx(idx);
            rr_m = obj.GMLMstructure.Groups(idx).dim_R_max;
        end
        
        function [isShared] = isSharedRegressor(obj, idx, dim) % if thetensor dimension (given by dim) for a particular tensor coefficient group (given by idx) uses a shared regressor set, rather than locally defined, dense regressors for each trial
            idx = obj.getGroupIdx(idx);
            dim = obj.getGroupDimIdx(idx, dim);
            isShared = size(obj.GMLMstructure.Groups(idx).F_shared{dim}, 1) > 0;
        end
        
        function [hh] = dim_H(obj, idx) %number of hyperparams: hyperparams on W and B if idx is not given, hyperparams on T and V if idx gives a tensor group, and all hyperparams in model if idx = -1 
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
                else
                    hh = 0;
                end
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
                if(strcmp(obj.Groups(jj).name, name))
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
        
        %change the rank of a group
        function [obj] = setDimR(obj, groupIdx, dim_R_new)
            if(~isscalar(dim_R_new) || fix(dim_R_new) ~= dim_R_new || dim_R_new < 0)
                error("dim_R must be a non-negative integer");
            end
            groupIdx = obj.getGroupIdx(groupIdx);
            
            rr_m = obj.dim_R_max(groupIdx);
            if(dim_R_new > rr_m)
                error("given dim_R is greater than maximum allocated rank (dim_R_new = %d, dim_R_max = %d)", dim_R_new, rrm);
            end
            
            obj.GMLMstructure.Groups(groupIdx).dim_R = dim_R_new;
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating parameter structures (Public)
    methods (Access = public)    
                
        %% METHOD [paramStruct] = getEmptyParamStruct()
        %	Builds a blank structure with all the GMLM model parameters given all currently included groups.
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
        %               B [dim_K x dim_P]  (real) : matrix of the linear coefficients
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
            params.B = zeros(obj.dim_K, obj.dim_P, dataType);
            if(includeHyperparameters)
                params.H = zeros(obj.dim_H, 1, 'double');
                params.Groups = struct('T', [], 'V', [], 'H', [], 'dim_names', [], 'name', []);
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
            
            params.W(:) = randn(size(params.W));
            params.B(:) = randn(size(params.B)) * 0.1;
            if(isfield(params, 'H'))
                params.H(:) = randn(size(params.H));
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
                    params.Groups(jj).H(:) = randn(size(params.Groups(jj).H));
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
            
            isValid = true;
            
            if( ~isfield(params, 'W') || ~isfield(params, 'B'))
                isValid = false;
                return;
            end
            
            if(~isvector(params.W) || ~isnumeric(params.W) || numel(params.W) ~= obj.dim_P)
                isValid = false;
                return;
            end
            
            if(~ismatrix(params.B) || ~isnumeric(params.B) || ~all(size(params.B) == [obj.dim_K, obj.dim_P]))
                isValid = false;
                return;
            end
            
            if(includeHyperparameters)
                if(~isfield(params, 'H') || ~isvector(params.H) || ~isnumeric(params.H) || numel(params.H) ~= obj.dim_H)
                    isValid = false;
                    return;
                end
            end
            
            if(~isfield(params, 'Groups') || ~isstruct(params.Groups) || ~isfield(params.Groups, 'V') || ~isfield(params.Groups, 'T') || numel(params.Groups) ~= obj.dim_J)
                isValid = false;
                return;
            end
            
            %for each group
            for jj = 1:obj.dim_J
                if(~ismatrix(params.Groups(jj).V) || ~isnumeric(params.Groups(jj).V) || ~all(size(params.Groups(jj).V) == [obj.dim_P, obj.dim_R(jj)]))
                    isValid = false;
                    return;
                end
                
                %for each dimension
                if(~iscell(params.Groups(jj).T) || numel(params.Groups(jj).T) ~= obj.dim_S(jj))
                    isValid = false;
                    return;
                end
                for ss = 1:obj.dim_S(jj)
                    if(~ismatrix(params.Groups(jj).T{ss}) || ~isnumeric(params.Groups(jj).T{ss}) || ~all(size(params.Groups(jj).T{ss}) == [obj.dim_T(jj, ss), obj.dim_R(jj)]))
                        isValid = false;
                        return;
                    end
                end
                
                %hyperparams if requested
                if(includeHyperparameters)
                    if(~isfield(params.Groups, 'H') || ~isvector(params.Groups(jj).H) || ~isnumeric(params.Groups(jj).H) || numel(params.Groups(jj).H) ~= obj.dim_H(jj))
                        isValid = false;
                        return;
                    end
                end
            end
        end
    end    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for computing log likelihood and rate on Host (Public)
    methods (Access = public)
        function [log_rate_per_trial] = computeLogRate(obj, params)
            %setup any shared regressors
            shared_regressors = struct('FT', cell(obj.dim_J, 1));
            for jj = 1:obj.dim_J
                shared_regressors(jj).FT = cell(obj.dim_S(jj), 1);
                for ss = 1:obj.dim_S(jj)
                    if(obj.isSharedRegressor(jj, ss))
                        shared_regressors(jj).FT{ss} = obj.GMLMstructure.Groups(jj).F_shared{ss} * params.Groups(jj).T{ss};
                    end
                end
            end
            
            %for each trial
            log_rate_per_trial = struct('log_rate', cell(size(obj.trials)));
            for mm = 1:obj.dim_M
                dim_N = numel(obj.trials(mm).Y);
                neuron_idx = obj.trials(mm).neuron_idx;
                
                %add linear and constant term
                if(obj.GMLMstructure.dim_K > 0)
                    log_rate_per_trial(mm).log_rate = obj.trials(mm).F_lin*params.B(:, neuron_idx) + params.W(neuron_idx) + log(obj.bin_size);
                end
                
                %add each group
                for jj = 1:obj.dim_J
                    G = ones(dim_N, obj.dim_R(jj), obj.dim_A(jj));
                    %for each event
                    for aa = 1:obj.dim_A(jj)
                        %get each dimension for the tensor components
                        for ss = 1:obj.dim_S(jj)
                            if(obj.isSharedRegressor(jj, ss))
                                ind_A_c = obj.trials(mm).Groups(jj).ind_A_shared{ss}(:, aa);
                                vv = ind_A_c > 0 & ind_A_c <= size(shared_regressors(jj).FT{ss}, 1);

                                G(~vv, :, aa) = 0;
                                G( vv, :, aa) = G(vv, :, aa) .* shared_regressors(jj).FT{ss}(ind_A_c(vv), :);
                            else
                                G(:, :, aa) = G(:, :, aa) .* (obj.trials(mm).Groups(jj).F_local{ss}(:, :, aa) * params.Groups(jj).T{ss});
                            end
                        end
                    end
                    
                    % sum over events
                    G = sum(G, 3);
                    
                    % linearly weight the components and add to rate
                    log_rate_per_trial(mm).log_rate = G * params.Groups(jj).V(neuron_idx,:)' + log_rate_per_trial(mm).log_rate;
                end
            end
        end
        
        function [log_like, log_like_per_trial] = computeLogLikelihoodHost(obj, params)
            log_like_per_trial = obj.computeLogRate(params);
            
            %for each trial
            for mm = 1:obj.dim_M
                vv = obj.trials(mm).Y >= 0; %check for any censored values
                log_like_per_trial(mm).log_like = -sum(exp(log_like_per_trial(mm).log_rate(vv))) + log_like_per_trial(mm).log_rate(vv)'*obj.trials(mm).Y(vv) - sum(gammaln(obj.trials(mm).Y(vv)+1));
            end
            
            log_like = sum([log_like_per_trial(:).log_like], 'all');
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
            addParameter(p, 'enableAll', true, @islogical);
            addParameter(p, 'includeHyperparameters', true, @islogical);
            addParameter(p, 'trial_weights', false, @islogical);
            parse(p,varargin{:});
            
            enableAll = p.Results.enableAll;
            includeHyperparameters = p.Results.includeHyperparameters;
            trial_weights = p.Results.trial_weights;
            
            %%
            %sets up top-level params
            opts.dW = enableAll;
            opts.dB = enableAll;
            if(includeHyperparameters)
                opts.dH = enableAll;
                opts.Groups = struct('dT', [], 'dV', [], 'dH', []);
            else
                opts.Groups = struct('dT', [], 'dV', []);
            end
            
            opts.trial_weights = [];
            if(trial_weights)
                opts.trial_weights = ones(obj.dim_M, 1);
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
                ss_c = numel(params.W);
                ww(ctr+(1:ss_c)) = params.W(:);
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
                ss_c = numel(params.W);
                params.W(:) = ww(ctr+(1:ss_c));
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
        %       N_comp_blocks      (int; default=128)   : number of compute blocked to setup on each gpu. A compute block is enough to handle a single trial at
        %                                                 once. This can be a scalar or vector to match deviceNumbers if you want to have different numbers of 
        %                                                 blocks on different GPUs.
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
            addParameter(p, 'N_comp_blocks', min(obj.dim_M, 128), @(x) isnumeric(x) && (isscalar(x) || numel(x) == numel(deviceNumbers)) && all(x > 0, 'all') && all(fix(x) == x, 'all'));
            parse(p,varargin{:});
            
            useDoublePrecision = p.Results.useDoublePrecision;
            N_comp_blocks = p.Results.N_comp_blocks;
            
            %% sets up GMLMstructure to send to GPU in correct datatypes and 0 indexing
            GMLMstructure_GPU = obj.GMLMstructure;
            
            GMLMstructure_GPU.dim_P = uint64(obj.dim_P);
            GMLMstructure_GPU.max_trial_length = uint64(max(arrayfun(@(aa) numel(aa.Y), obj.trials)));
            GMLMstructure_GPU.max_trials       = uint64(obj.dim_M);
            GMLMstructure_GPU.dim_K       = uint64(obj.dim_K);
            if(useDoublePrecision)
                GMLMstructure_GPU.log_dt = double(log(obj.bin_size));
            else
                GMLMstructure_GPU.log_dt = single(log(obj.bin_size));
            end

            for jj = 1:obj.dim_J
                GMLMstructure_GPU.Groups(jj).dim_A = uint64(obj.dim_A(jj));
                GMLMstructure_GPU.Groups(jj).dim_R_max = uint64(obj.dim_R_max(jj));
                GMLMstructure_GPU.Groups(jj).dim_T = uint64(obj.dim_T(jj));
                
                %for each dimension
                for ss = 1:obj.dim_S(jj)
                    %makes sure local regressors are correct floating point type
                    if(useDoublePrecision && ~isa(GMLMstructure_GPU.Groups(jj).F_shared{ss}, 'double'))
                        GMLMstructure_GPU.Groups(jj).F_local{ss} = double(GMLMstructure_GPU.Groups(jj).F_shared{ss});
                    elseif(~useDoublePrecision && ~isa(GMLMstructure_GPU.Groups(jj).F_shared{ss}, 'single'))
                        GMLMstructure_GPU.Groups(jj).F_local{ss} = single(GMLMstructure_GPU.Groups(jj).F_shared{ss});
                    end
                end
            end
            
            %% sets up trials to send to GPU in correct datatypes and 0 indexing
            trialBlocks = struct('GPU', mat2cell(int32(deviceNumbers(:)), numel(deviceNumbers), 1), 'N_comp_blocks', int32(N_comp_blocks), 'trials', []);
            
            trialsPerBlock = ceil(obj.dim_M / numel(trialBlocks));
            for bb = 1:numel(trialBlocks)
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
                    % makes sure spike count is int32
                    trialBlocks(bb).trials(mm).Y = int32(trialBlocks(bb).trials(mm).Y);
                    
                    % makes sure F_lin is correct floating point type
                    if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).F_lin, 'double'))
                        trialBlocks(bb).trials(mm).F_lin = double(trialBlocks(bb).trials(mm).F_lin);
                    elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).F_lin, 'single'))
                        trialBlocks(bb).trials(mm).F_lin = single(trialBlocks(bb).trials(mm).F_lin);
                    end
                    
                    %sets neuron index to int32 and 0 indexed 
                    trialBlocks(bb).trials(mm).neuron_idx = uint32(trialBlocks(bb).trials(mm).neuron_idx - 1);
                    
                    %sets trial_idx (int32 and 0 indexed)
                    trialBlocks(bb).trials(mm).trial_idx = uint32(trial_indices(mm) - 1);
                    
                    %for each Group
                    for jj = 1:obj.dim_J
                        %for each dimension
                        for ss = 1:obj.dim_S(jj)
                            %makes sure local regressors are correct floating point type
                            if(useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Groups(jj).F_local{ss}, 'double'))
                                trialBlocks(bb).trials(mm).Groups(jj).F_local{ss} = double(trialBlocks(bb).trials(mm).Groups(jj).F_local{ss});
                            elseif(~useDoublePrecision && ~isa(trialBlocks(bb).trials(mm).Groups(jj).F_local{ss}, 'single'))
                                trialBlocks(bb).trials(mm).Groups(jj).F_local{ss} = single(trialBlocks(bb).trials(mm).Groups(jj).F_local{ss});
                            end
                            
                            %sets any shared regressor indices to int32 and 0 indexed
                            trialBlocks(bb).trials(mm).Groups(jj).ind_A_shared{ss} = int32(trialBlocks(bb).trials(mm).Groups(jj).ind_A_shared{ss} - 1);
                        end
                    end
                end
            end
            
            %% call mex with (GMLMstructure_GPU, trialBlocks, useDoublePrecision), get pointer in return
            obj.gpuObj_ptr = kcGMLMv2_mex_create(GMLMstructure_GPU, trialBlocks, useDoublePrecision);
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
                kcGMLMv2_mex_clear(obj);
                % erase pointer value
                obj.gpuObj_ptr = 0;
                obj.gpus = [];
            end
        end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %methods for computing log likelihood (and derivatives) on GPU (Public)
    methods (Access = public)    
        
        function [results] = computeLogLikelihood(obj, params, opts)
            if(~obj.isOnGPU())
                error("GMLM is not on GPU.");
            end
            
            %allocate space for results
            results = obj.getEmptyResultsStruct(opts);
            
            %send to GPU
            kcGMLMv2_mex_computeLL(obj.gpuObj_ptr, obj.gpuDoublePrecision, params, results, opts.trial_weights);
        end
    end
    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for generating computation result structures (Private)
    methods (Access = public)
        %% METHOD [paramStruct] = getEmptyResultsStruct(includeHyperparameters, useDoublePrecision)
        %	Builds a blank structure with all the space for the derivatives of the GMLM model parameters and the log likelihood
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
        %               dB [dim_K x dim_P]  (real) : 
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
            useDoublePrecision = isa(opts.trial_weights, 'double');
            if(useDoublePrecision)
                dataType = 'double';
            else
                dataType = 'single';
            end
            
            results.trialLL = zeros(obj.dim_M, 1, dataType);
            
            %sets up top-level params
            if(opts.dW)
                results.dW = zeros(obj.dim_P, 1, dataType);
            else
                results.dW = zeros(0, 0, dataType);
            end
            if(opts.dB)
                results.dB = zeros(obj.dim_K, obj.dim_P, dataType);
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
            end
        end
    end
end