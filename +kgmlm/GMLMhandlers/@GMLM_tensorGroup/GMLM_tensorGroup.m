%% CLASS GMLM_tensorGroup
%
%   This class holds the regressor and dimension information for one group of coefficients in a GMLM that have
%   low-dimensional tensor structure.
%
%   This class is indended to be accessed only by a GMLM object, which holds one or more GMLM_tensorGroups.
%
classdef GMLM_tensorGroup
    properties (Dependent)
        dim_R %rank of group
    end
    properties (SetAccess = private, GetAccess = private)
        dim_R_private %rank of group
        dim_Ts
    end
    properties (SetAccess = private, GetAccess = public)
        
        dim_H %number of hyperparameters
        function_prior %a function that computes the log prior (and its derivatives)
        dimNames
        dim_S %order of tensor
        dim_P %number of neurons
        
        
        constraintTypes
    end
    
    properties (SetAccess = public, GetAccess = public)
        multistreamed = true;
    end
    
    properties (SetAccess = immutable, GetAccess = public)
        name %a name for the group
        max_dim_R %max rank of group: set for gpu storage. Rank can be changed, but only up to this value
        
        dim_N %number of observations (GMLM class also has this info - also needed here) 
        obsRange_Y % same as GMLM's obsRange_Y (is duplicated here for CPU log-likelihood computation purposes)
        
        
        %dim_T % covariates F{ii} are dim_F(ii) x dim_T(ii) in size
        %dim_F % covariates F{ii} are dim_F(ii) x dim_T(ii) in size
        %dim_S %order of tensor (minus the neuron dimension)
        dim_A %number of indices
        
        gpuSinglePrecision;
        RegressorIndex struct %table of indices to covariates (each field should be matrices N x dim_A OR empty)
        Regressors struct %struct of covariates
        
        %FOR FUTURE WORK: method to add possible events or dimensions
        %    would change some of the above variables (no longer immutable)
    end
    
    methods
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = GMLM_tensorGroup(Regressors, RegressorIndex,gpuSinglePrecision,rank,dim_N,obsRange_Y,name,function_prior, dim_H, max_rank, constraintTypes, multistreamed)
        %% CONSTRUCTOR GMLM_tensorGroup(Regressors, RegressorIndex,rank,dim_N,obsRange_Y,name,function_prior,dim_H)
        %	Default constructor for creating a GMLM tensor group. This should only be called from GMLM.
        %   This constructor takes the regressor information, rank, and information about the GMLM's dimensions.
        %   The sizes of the regressors are checked for consistency.
        %
        %   inputs:
        %       Regressors, RegressorIndex
        %           pairs of covariates and indices for each dimension of
        %           the tensor (besides the implicit final "neuron" loading
        %           dimension").
        %           We use this indexing system because some events in a task (like a stimulus presentation)
        %           may be the same (up to an onset time) across many trials. Therefore, the regressors for that event
        %           are shared, and do not need to be evented in a design matrix. We just need an auxialary variable, (ind_A) 
        %           to keep track of the timing of that event across all the trials.
        %
        %           RegressorIndex is a table.
        %           Regressors is a struct.
        %               Each have corresponding field names, which correspond the the tensor dimensions.
        %               Note: Regressors is not given as a table because the rows of each piece may vary
        %
        %           For field ss of RegressorIndex:
        %               Regressors.ss     [dim_F[ss] x dim_T[ss]]  (real) the regressor matrix
        %               RegressorIndex.ss [dim_N x dim_A] OR EMPTY (int) 
        %
        %                   When Regressors.ss is a full design matrix, RegressorIndex.ss can be empty. This implies 
        %                   RegressorIndex.ss[ii,aa] = (dim_N * (aa-1)) + ii -> this requires that dim_F[ss] = dim_A*dim_N 
        %
        %               dim_A is the number of "events" for this tensor group: it must be consistent across all dimensions of the group.
        %
        %
        %                   NOTE: if ind_A indexes out-of-bound (RegressorIndex.ss[ii,aa] <= 0 or RegressorIndex.ss[ii,aa] > dim_F[ss]),
        %                         it's assumed that Regressors.ss[RegressorIndex.ss[ii,aa],:] = 0 for convenience
        %
        %       gpuSinglePrecision [scalar] (logical)
        %           Whether or not Regressors should be single precision
        %
        %       rank [scalar] (int)
        %           The rank for this group. Must be at least 1. (is dim_R)
        %
        %       dim_N [scalar] (int)
        %           Number of observations in GMLM for dimension checking (passed in by GMLM)
        %
        %       obsRange_Y [(dim_P+1) x 1] (ints)
        %           Index vector (NOTE: in MATLAB's native 1 indexing) into the spike count vector Y where 
        %           obsRange_Y(ii):(obsRange_Y(ii+1)-1) is the range of observations for neuron ii. This is pass in by
        %           GMLM and used here only for local CPU computations of the log likelihood.
        %
        %       name (string)
        %           A name for this group
        %
        %       function_prior (function)
        %           OPTIONAL (default = empty)
        %           Function for log prior distribution over group parameters.
        %           Set to empty if no prior.
        %           It also can return derivates + portions of hessian too.
        %           If a field in the result group struct is empty, it does not need to be computed!
        %           Function must take the form:  [GMLM_resultGroup] =  prior(GMLM_paramGroup,GMLM_resultGroup) 
        %               This function must compute 
        %                   GMLM_resultGroup.dH
        %                   GMLM_resultGroup.log_prior
        %               This function should only ADD to 
        %                   GMLM_resultGroup.dV
        %                   GMLM_resultGroup.d2V
        %                   GMLM_resultGroup.dT  (for each field)
        %                   GMLM_resultGroup.d2T (for each field) 
        %
        %       dim_H [scalar] (int)
        %           OPTIONAL (default = 0)
        %           Number of hyperparameters for the group. (How many hyperparameters to send into function_prior) 
        %
        %       max_rank [scalar] (int)
        %           OPTIONAL (default = rank)
        %           The maximum allowed rank for this group. Must be at least 1.
        %
            if(nargin < 7)
                error('GMLM_tensorGroup constructor requires at least 7 inputs.');
            end
            
            
            
            %default values
            if(nargin < 9 || isempty(dim_H) )
                dim_H = 0;
            end
            if(nargin < 8 || isempty(function_prior) )
                function_prior = [];
                dim_H = 0;
            end
            if(~isscalar(dim_H) || ceil(dim_H) ~= dim_H || dim_H < 0)
                error('number of hyperparameters must be a non-zero integer.');
            end
            if(~isempty(function_prior) && ~isa(function_prior,'function_handle'))
                error('prior function must be a function handle or empty');
            end

            if(nargin < 10 || isempty(max_rank))
                max_rank = rank;
            end
            
            
            %must have at least one factor in tensor decomposition
            if(~isscalar(rank) ||  ~isscalar(max_rank) || max_rank < 1 || ceil(max_rank) ~= max_rank || ceil(rank) ~= rank || rank > max_rank)
                error('rank of group must be positive integer!');
            end
            
            %GPU parallelization option must be a logical
            
            if(nargin >= 12 && ~isempty(multistreamed) && islogical(multistreamed))
                obj.multistreamed = multistreamed;
            end
            
            %check types for regressor information
            if((~istable(Regressors) && ~isstruct(Regressors)) || (~istable(RegressorIndex) && ~isstruct(RegressorIndex)))
                error('"Regressors" and "RegressorIndex" must be a data table or struct.');
                %note: Regressors is allowed to be a table even though this isn't mentioned in the header: each entry 
                %      could be complete (one row for each observation) and the regressors indices could be all empty
            end
            
            %check for consistent dim_S across ind_A and F arguments
            if(istable(RegressorIndex))
                dimNames = sort(string(RegressorIndex.Properties.VariableNames));
            else
                dimNames = sort(string(fieldnames(RegressorIndex)));
            end
            dim_S = length(dimNames);
            
            
            %sets up dimension variables

            obj.dim_N = dim_N;

            obj.Regressors     = struct();
            obj.RegressorIndex = struct();
            

            %checks each dimension before storing
            for ss = 1:dim_S
                dimName = dimNames{ss};
                if(~isfield(Regressors,dimName))
                    error('Field name "%s" is not contained in Regressors.',dimName);
                end
                
                if(size(RegressorIndex.(dimName),1) ~= obj.dim_N)% && ~isempty(RegressorIndex.(dimName)))
                    error('RegressorIndex.(%s) must have dim_N = %d rows (has %d).', dimName, obj.dim_N, size(RegressorIndex.(dimName),1));
                end
                
                %% checks data type and assigns variable
                if(gpuSinglePrecision)
                	obj.Regressors.(dimName) = single(Regressors.(dimName));
                else
                	obj.Regressors.(dimName) = double(Regressors.(dimName));
                end
                obj.RegressorIndex.(dimName) = uint32(RegressorIndex.(dimName));
                
                
                %% checks index dimension
                dim_A_c = size(RegressorIndex.(dimName),2);
                %if current index is empty: no indexing necessary.
                % Regressors.ss must be a full design matrix: rows of Regressors.ss correspond to each observation
                if(dim_A_c == 0)
                    %size of F_c must be a multiple of dimN
                    if(obj.dim_F(dimName) == dim_N )
                        %the number of events is the number of rows in F_c divided by dim_N
                        aa_c = size(obj.Regressors.(dimName), 3);

                        %if this is the first dimension, we set the number of events for this group
                        %  otherwise, we check that the number of events matches (dim_A must be the same for all
                        %  ind_A_c, regardless of how they are indexed)
                        if(ss == 1)
                            obj.dim_A = uint32(aa_c);
                        elseif(aa_c ~= obj.dim_A)
                            error('Invalid size of input F to group (part %d): number of event indices must be equal across parts.',ss);
                        end
                    else
                        error('Invalid size of input F to group (part %d): if A is empty, size(F,1) must be a multiple of N (total number of observations).',ss);
                    end
                else
                    %if this is the first dimension, we set the number of events for this group
                    %  otherwise, we check that the number of events matches (dim_A must be the same for all
                    %  ind_A_c, regardless of how they are indexed)
                    if(ss == 1)
                        obj.dim_A = uint32(dim_A_c);
                    elseif(dim_A_c ~= obj.dim_A)
                        error('Invalid size of input F to group (part %d): number of event indices must be equal across parts.',ss);
                    end
                end
                
            end
            
            %sets up dim_T information
            obj.dim_Ts = zeros(1,dim_S,'uint32');
            for ss = 1:dim_S
                dimName = dimNames(ss);
                obj.dim_Ts(ss) = uint32(size(obj.Regressors.(dimName),2));
            end
            
            %converts type of obj.RegressorIndex
            for ss = 1:dim_S
                dimName = dimNames(ss);
                
                if(~isempty(obj.RegressorIndex.(dimName)))
                    %% add row of 0s to F if needed to accomodate any out of range indicies.
                    outOfRange = obj.RegressorIndex.(dimName) < 1 | obj.RegressorIndex.(dimName) > obj.dim_F(dimName);
                    if(sum(sum(outOfRange)) > 0)
                        zeroIdx = find(all(obj.Regressors.(dimName) == 0,2),1,'first'); %if the matrix contains a row of all zeros, finds it
                        
                        if(isempty(zeroIdx))
                            %otherwise, adds the row to Regressors
                            if(~gpuSinglePrecision)
                                obj.Regressors.(dimName) = [obj.Regressors.(dimName); zeros(1,obj.dim_T(dimName))];
                            else
                                obj.Regressors.(dimName) = [obj.Regressors.(dimName); zeros(1,obj.dim_T(dimName),'single')];
                            end
                            obj.RegressorIndex.(dimName)(outOfRange) = obj.dim_F(dimName);
                        else
                            obj.RegressorIndex.(dimName)(outOfRange) = zeroIdx;
                        end
                    end
                    
                    obj.RegressorIndex.(dimName) = uint32(obj.RegressorIndex.(dimName)-1);
                end
            end
            
            %sets rank
            obj.max_dim_R = uint32(max_rank);
            obj.dim_R = uint32(rank);
            
            %stup information about the observations for CPU computations
            obj.obsRange_Y = obsRange_Y;
            
            %sets the hyperparameter vars
            obj.function_prior = function_prior;
            obj.dim_H = uint32(dim_H);
            
            obj.name = name;
            obj.gpuSinglePrecision = gpuSinglePrecision;
            
            obj.dimNames = sort(string(fieldnames(obj.Regressors)));
            
            obj.dim_P = uint32(numel(obj.obsRange_Y)-1);
            obj.dim_S = uint32(numel(obj.dim_Ts));
            
            if(nargin < 11 || isempty(constraintTypes))
                constraintTypes = zeros(obj.dim_S,1);
            end
            if(numel(constraintTypes) ~= obj.dim_S || ~all(ismember(constraintTypes,0:2)))
                error('invalid compute types: should be 0 (no constraints), 1 (normalized), 2 (orthonormal)');
            end
            obj.constraintTypes = constraintTypes;
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% METHODS to get consistent dimension information
        function [dim] = dim_T(obj,dimName_0)
            if(nargin < 2)
                dim = obj.dim_Ts;
            else
                if(isscalar(dimName_0) && ~isstring(dimName_0))
                    dim = obj.dim_Ts(dimName_0);
                else
                    dim = uint32(size(obj.Regressors.(dimName_0),2));
                end

            end
        end
        
        function [dim] = dim_F(obj, dimName_0)
            if(nargin < 2)
                dim = zeros(1,obj.dim_S,'uint32');
                for ss = 1:obj.dim_S
                    dimName = obj.dimNames(ss);
                    dim(ss) = uint32(size(obj.Regressors.(dimName),1));
                end
            else
                if(isscalar(dimName_0) && ~isstring(dimName_0))
                    dimName = obj.dimNames(dimName_0);
                else
                    dimName = dimName_0;
                end

                dim = uint32(size(obj.Regressors.(dimName),1));
            end
        end
        
        function [obj] = replaceRegressors(obj, dimName, regressors)
            if(all(size(regressors) == size(obj.Regressors.(dimName))))
                obj.Regressors.(dimName)(:) = regressors;
            else
                error("Cannot replace regressors: size must stay the same!");
            end
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = setPrior(obj,function_prior,dim_H)
        %% METHOD [] = setPrior(function_prior,dim_H);
        %   Changes the log prior for the group groups parameters.
        %
        %  	inputs:
        %       function_prior
        %       dim_H 
        %           SEE GMLM_tensorGroup constructor
        
            %default values
            if(isempty(dim_H) || nargin < 2)
                dim_H = 0;
            end
            if(isempty(function_prior) || nargin < 1)
                function_prior = [];
                dim_H = 0;
            end
            %checks values
            if(~isscalar(dim_H) || ceil(dim_H) ~= dim_H || dim_H < 0)
                error('number of hyperparameters must be a non-zero integer.');
            end
            if(~isempty(function_prior) && ~isa(function_prior,'function_handle'))
                error('prior function must be a function handle or empty');
            end
            
            %sets new priors
            obj.dim_H = uint32(dim_H);
            obj.function_prior = function_prior;
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [rr] = get.dim_R(obj)
        %% METHOD [name] = getRank(obj)
        %	 Getter method for the protected group rank.
            rr = obj.dim_R_private;
        end
        
        %	 Setter method for the group rank
        function obj = set.dim_R(obj,rank)
            if(isscalar(rank) && ceil(rank) == rank && rank>=0 && rank <= obj.max_dim_R)
                obj.dim_R_private = uint32(rank);
            else
                error('invalid rank: must be non-negative integer!');
            end
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [lambda] = computeRate_cpu(obj,paramObj_group)
        %% METHOD [lambda] = computeRate_cpu(paramObj_group)
        %	Computes the portion of trhe (log) spike rate for each observation from this group of coefficients for one
        %	setting of the model parameters.
        %   The computation is performed on the CPU (no GPUs needed), and thus can be very slow.
        %
        %   This function is primarily intended for debugging, and thus does not return any derivatives.
        %
        %   inputs:
        %       paramObj_group (struct)
        %           The current model parameters for the tensor group (not complete GMLM). See getEmptyParamObj for the details on the structure.
        %
        %   returns:
        %       lambda [dim_N x 1] (real)
        %           The portion of the log rate for each observation from this GMLM group.
        
            %compute rate contribution for the group on CPU
                
            % for each dimension, computes regressors * parameters (factors)
            FT = struct();
            for ss = 1:obj.dim_S
                dimName = obj.dimNames(ss);

                if(size(obj.Regressors.(dimName),3) == 1)
                    FT.(dimName) = obj.Regressors.(dimName)*paramObj_group.T{ss};
                else
                    FT.(dimName) = zeros(obj.dim_N,obj.dim_R,obj.dim_A);
                    for ii = 1:obj.dim_A
                        FT.(dimName)(:,:,ii) = obj.Regressors.(dimName)(:,:,ii)*paramObj_group.T{ss};
                    end
                end
            end

            % for each event of the group
            lambda = zeros(obj.dim_N,obj.dim_R);
            for aa = 1:obj.dim_A
                %for each dimension
                lambda_0 = zeros(obj.dim_N,obj.dim_R,obj.dim_S);
                for ss = 1:obj.dim_S
                    dimName = obj.dimNames(ss);
                    % gets the indices into the rows of FT for the observations
                    if(isempty(obj.RegressorIndex.(dimName)))
                        lambda_0(: ,:,ss) = FT.(dimName)(:, :, aa);
                    else
                        A_c = obj.RegressorIndex.(dimName)(:,aa);
                        vv = A_c >= 0 & A_c < obj.dim_F(ss); 

                        lambda_0(vv,:,ss) = FT.(dimName)(A_c(vv)+1,:);
                    end
                end
                % takes the product across dimensions, and adds the current event to the total
                lambda = lambda + prod(lambda_0,3);
            end

            %for each neuron, multiplies in the neuron weights to each dimension
            for pp = 1:obj.dim_P()
                Y_idx = (obj.obsRange_Y(pp)+1):(obj.obsRange_Y(pp+1));

                if(~isempty(Y_idx))
                    lambda(Y_idx,:) = lambda(Y_idx,:).*paramObj_group.V(pp,:);
                end
            end

            %sums up all ranks/factors
            lambda = sum(lambda,2);
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [results_group]  = addLPrior(obj,params_group,results_group)
        %% METHOD [results_group] = addLPrior(paramObj_group)
        %   
        %  Compute the log prior of the model paramters for this group.
        %
        %   inputs:
        %       
        %       params
        %           The current model parameters for the group. 
        %
        %   inputs/outputs:
        %       results_group
        %           The object containing all requested results. Prior terms are added.
        %
            
            if(~isempty(obj.function_prior))
                results_group = obj.function_prior(obj,params_group,results_group);
            else
                results_group.log_prior = 0;
                if(~isempty(results_group.dH))
                    results_group.dH(:) = 0;
                end
            end
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct_group] = getEmptyParamStruct(obj,includeH)
        %% METHOD [paramStruct_group] = getEmptyParamStruct()
        %	Builds a empty structure (all zeros) for the model parameters for this single group.
        %
        %	returns:
        %       paramStruct_group [struct]        
        %           Fields for model parameters:
        %               V [dim_P x dim_R] (real)      : The low-rank factors for the neuron weighting.
        %               T                 (cell array): the matrices of PARAFAC decomposed factors for the remaining
        %                                               tensor dimenions.  Each T.ss [dim_T[ii] x rank_jj]
        %               H [dim_H x 1]     (real)      : the hyperparameters for this group
        %
        %               dimNames          (string array): names for each dimension in T
        %
            
            if(nargin < 2 || isempty(includeH))
                includeH = true;
            end
            
            dataType = 'double';
            if(obj.gpuSinglePrecision)
                dataType = 'single';
            end
        
            %sets up empty (zeros) matrices for the coefficients
            paramStruct_group.V = zeros(obj.dim_P,obj.dim_R,dataType);
            paramStruct_group.T     = cell(obj.dim_S,1);
            paramStruct_group.T_type= zeros(obj.dim_S,1);
            for ss = 1:obj.dim_S
                paramStruct_group.T{ss} = zeros(obj.dim_T(ss),obj.dim_R,dataType);
                paramStruct_group.T_type(ss) = obj.constraintTypes(ss);
            end
            
            %empty vector for the hyperparameters
            if(includeH)
                paramStruct_group.H = zeros(obj.dim_H,1);
            end
            
            %get the dimension names
            paramStruct_group.dimNames = obj.dimNames;
            paramStruct_group.name = obj.name;
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [optsStruct_group] = getEmptyOptsStruct(obj, defaultValue, defaultValueHessians)
        %% METHOD [optsStruct_group] = getEmptyOptsStruct()
        %	Builds a compute options for the tensor group's parameters with all fields set to defaultValue.
        %
        %   inputs:
        %       defaultValue
        %           OPTIONAL (default = false)
        %           Whether all options are on or off
        %       defaultValueHessians
        %           OPTIONAL (default = defaultValue)
        %           Whether options to compute hessians are on or off
        %
        %	returns:
        %       optsStruct_group [struct]        
        %           A structure of the compute options for the likelihood/posterior for this tensor group. All values are logical
        %
        %           Fields for model parameters:
        %
        %               compute_dV    : compute derivative of of posteior w.r.t V
        %               compute_d2V   : compute second derivative of of posteior w.r.t V
        %
        %               compute_dT  [length dim_S]  : compute derivative of of posteior w.r.t each T
        %               compute_d2T [length dim_S]  : compute second derivative of of posteior w.r.t each T
        %
        %               compute_dH    : compute derivative of prior over V,T w.r.t the hyperparams
        %              
        %
            if(nargin < 2)
                defaultValue = false;
            end
            if(nargin < 3)
                defaultValueHessians = defaultValue;
            end
            
            if(~islogical(defaultValue))
                error('defaultValue must be a logical');
            end
            if(~islogical(defaultValueHessians))
                error('defaultValueHessians must be a logical');
            end
            
            optsStruct_group.compute_dV  = defaultValue;
            optsStruct_group.compute_d2V = defaultValueHessians;
            optsStruct_group.compute_dT  = repmat(defaultValue,obj.dim_S,1);
            optsStruct_group.compute_d2T = repmat(defaultValueHessians,obj.dim_S,1);
            optsStruct_group.compute_dH  = defaultValue;
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct_group] = getEmptyResultsStruct(obj, optsStruct_group)
        %% METHOD [resultsStruct_group] = getEmptyResultsStruct()
        %	Builds a empty results (all zeros) for the likelihood outputs for this single group.
        %
        %   inputs
        %       OPTIONAL (default is all true)
        %       Options for what fields to include in computation (see getEmptyOptsStruct)
        %
        %	returns:
        %       paramStruct_group [struct]        
        %           Fields for model parameters:
        %               dV [dim_P x dim_R]  (real)       : The low-rank factors for the neuron weighting.
        %               d2V [dim_P x dim_R] (real)       : The low-rank factors for the neuron weighting: stored in compact form
        %               dT                  (cellArray)  : the matrices of PARAFAC decomposed factors for the remaining
        %                                                  tensor dimenions.  Each T.ss [dim_T[ii] x rank_jj]
        %               d2T                 (cellArray)  : the matrices of PARAFAC decomposed factors for the remaining
        %                                                  tensor dimenions.  
        %
            dataType = 'double';
            if(obj.gpuSinglePrecision)
                dataType = 'single';
            end
            if(nargin < 2)
                optsStruct_group = obj.getEmptyOptsStruct(true);
            end
            
            dim_R_c = obj.dim_R;
            dim_P_c = obj.dim_P;
            
            
            %sets up empty (zeros) matrices for the coefficients
            if(optsStruct_group.compute_dV)
                dV     = zeros(dim_P_c,dim_R_c,dataType);
                if(optsStruct_group.compute_d2V)
                    d2V = zeros(dim_P_c*dim_R_c,dim_R_c,dataType);
                else
                    d2V = [];
                end
            else
                dV     = [];
                d2V    = [];
            end
            
            if(optsStruct_group.compute_dH)
                dH     = zeros(obj.dim_H,1);
            else
                dH     = [];
            end
            
            dT  = cell(obj.dim_S,1);
            d2T = cell(obj.dim_S,1);
            for ss = 1:obj.dim_S
                if(optsStruct_group.compute_dT(ss))
                    dT{ss}  = zeros(obj.dim_T(ss),dim_R_c,dataType);
                    if(optsStruct_group.compute_d2T(ss))
                        d2T{ss} = zeros(obj.dim_T(ss)*dim_R_c,obj.dim_T(ss)*dim_R_c,'double');%dataType);
                    else
                        d2T{ss} = [];
                    end
                else
                    dT{ss} = [];
                    d2T{ss} = [];
                end
            end
            
            log_prior = zeros(1,1);
            
            
            resultsStruct_group = struct('dT',{dT},'dV',dV,'d2V',d2V,'d2T',{d2T},'dH',dH,'log_prior',log_prior);
        end
    end
end