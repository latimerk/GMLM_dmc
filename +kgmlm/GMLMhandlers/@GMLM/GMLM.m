%% CLASS GMLM
%
% This class holds all the regressors and data for a GMLM. It handles all calls to the log likelihood functions, and can
% load the data onto the GPU.
%
% The Generalized linear tensor model (GMLM) is a tool for fitting the responses of multiple neurons (which can be
% inidividually or simulataneously recorded) during a task, where it's assumed that the responses across different task
% variables, time, and neurons has low-dimensional structure which can be decomposed as a tensor.
% A full-rank GMLM (not low-D) is the equivalent of fitting GLMs independently to each neuron.
%
% To build a GMLM, start with a vector of observations (Y), which are ordered by neuron index.
%   Constructor: GMLM(Y,neuron_startTimes,trial_startTimes,dt)
%
% To add a group of coefficients with low-rank tensor structure: addGroup method
%
% Once all coefficients have been added, the data can be loaded to the GPU(s) with a call to: toGPU
%
% An empty initial parameter structure is given by: getEmptyParamObj()
%
% The log posterior is computed by: computeLPost
% 
% The GPU data is cleared by: freeGPU()
%
%
% LaTeX definition of the model:
% 
% For $i = 1,2,\dots,N$: 
% 
% \lambda_i & = 
%
% To construct a GMLM, start with the vector $y$ where the observations are ordered by neuron number (in ascending order).
%   Then add groups of coefficients and the corresponding covariates (\mathbf{F}^{(j,\cdot}} and \mathbf{A}^{(j,\cdot}}
%   respectively) that are assumed to have low-rank tensor structure. We divide this into groups so that the low-rank
%   structure of different task pieces (e.g., stimulus and spike history) have independent tensor decompositions.
%
% This class will setup the appropriate paramter structures given the regressor matrices.
%
classdef GMLM
    properties (Dependent, GetAccess = public, SetAccess = immutable)
        dim_N
        dim_P
        dim_J
        dim_M
    end
    properties (GetAccess = public, SetAccess = private)
        Groups GMLM_tensorGroup %array of GMLM_tensorGroups
        
        gpuObj_ptr uint64 %uint64 pointer to the c++ object for the GMLM loaded to the GPU
        gpus uint32
        
        dim_H uint32 %number of hyperparameters
        function_prior %a function that computes the log prior (and its derivatives)
        
        dim_K uint32
        
        Y_mask_trial logical
        
        obsRange_blocks
    end
    properties (GetAccess = public, SetAccess = public)
        hessiansEnabled logical
    end
    properties (GetAccess = public, SetAccess = immutable)
        %dim_N %number of observations
        %dim_P %number of neurons
        %dim_M %number of trials
        
        Y int32 %the observations: spike counts
        
        GLMGroup %GLM regressors
        
        const_dt %size of time bins
        
        obsRange_Y     uint32 %index variable for which observations go with which neuron: obsRange_Y(ii):(obsRange_Y(ii)-1) is the range of Y for neuron ii
        obsRange_trial uint32 %indicies for which observations go with which trial
        
        const_poissLL %the constant term of the Poisson log likelihood
        const_trial_poissLL %the constant term of the Poisson log likelihood for each trial
        
        gpuSinglePrecision logical % [scalar] (logical) if using single precision on group. If false, using double.
       
        trsPerNeuron
    end
    
    
    methods 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = GMLM(Y, neuron_startTimes, trial_startTimes, dt, gpuSinglePrecision, function_prior, dim_H, GLMGroup)
        %% CONSTRUCTOR GMLM(Y,Y_startTimes,trial_startTimes,dt)
        %	Default constructor for creating a GMLM object.
        %   This constructor takes all the observations (spike count) information.
        %   The covariates can be added on later.
        %
        %   inputs:
        %       Y [N x 1] (real)    
        %           Vector of all the spike count observations (N is total number of bins).
        %       	Must be structured in terms of neuron first (and optionally
        %       	trial within neuron).  The trial structure beahves as if neurons
        %           are independently recorded.
        %           	[Y_{neuron 1, trial 1},
        %               Y_{neuron 1,trial_2},
        %               ...,
        %               Y_{neuron 1, trial M_1},
        %               Y_{neuron 2, trial M_1+1},
        %               ...,
        %               Y_{neuron 2, trial M_1+M_2},
        %               ...,
        %               Y_{neuron P, trial M_1+M_2+...+M_P];
        %
        %       neuron_startTimes [dim_P x 1] (ints)
        %           Index vector into Y where neuron_startTimes(ii) is the first element
        %           of Y for neuron ii. Total number of neurons is P.
        %
        %       trial_startTimes [dim_M x 1] (ints)
        %           OPTIONAL (default = 0: makes all obs one big trial, doesn't affect computation but indivLL=total LL)
        %           Index vector into Y where trial_startTimes(ii) is the first
        %           element of Y for trial ii
        %
        %       dt [scalar] (real)
        %           OPTIONAL (default = 1)
        %           width of the timebines
        %
        %       gpuSinglePrecision [scalar] (logical)
        %           OPTIONAL (default = true)
        %           Whether or not to use single precision.
        %       
        %       function_prior (function)
        %           OPTIONAL (default = empty)
        %           Function for log prior distribution over neuron weighting.
        %           Set to empty if no prior.
        %           It also can return derivates + portions of hessian too.
        %           If a field in the results struct is empty, the value does not need to be computed.
        %           Function must take the form:  [GMLM_results] =  prior(GMLM_params,GMLM_results) 
        %               This function must only try to set
        %                   GMLM_results.log_prior_w
        %                   GMLM_results.dH
        %               This function should ADD to
        %                   GMLM_results.dW
        %                   GMLM_results.d2W
        %
        %       dim_H [scalar] (int)
        %           OPTIONAL (default = 0)
        %           Number of hyperparameters for the prior over W. (How many hyperparameters to send into function_prior) 
        %
            if(nargin < 2)
                error("Two arguments required: Must input observation information!");
            end
            
            if(nargin < 3 || isempty(trial_startTimes))
                %if no trial start times are given, default value is no
                %independt trials (just one block)
                trial_startTimes = 1;
            end
            if(nargin < 4 || isempty(dt))
                %default value for bin width
                dt = 1;
            end
            
            %default values
            if(nargin < 5 || isempty(gpuSinglePrecision))
                gpuSinglePrecision = true;
            end
            if(~isscalar(gpuSinglePrecision) || ~islogical(gpuSinglePrecision))
                error('gpuSinglePrecision must be a logical value!');
            end
            
            if(nargin < 7 || isempty(dim_H))
                dim_H = 0;
            end
            if(nargin < 6 || isempty(function_prior) )
                function_prior = [];
                dim_H = 0;
            end
            if(~isscalar(dim_H) || ceil(dim_H) ~= dim_H || dim_H < 0)
                error('number of hyperparameters must be a non-zero integer.');
            end
            if(~isempty(function_prior) && ~isa(function_prior,'function_handle'))
                error('prior function must be a function handle or empty');
            end
            
            if(nargin < 8)
                GLMGroup = [];
            end
            
            %initial values for GMLM object
            obj.const_dt = dt;
            obj.Y = int32(Y(:)); %converts spike counts to ints
            
            
            
            %checks the start time indices: must be strictly increasing, positive, and
            %not greater than N.
            d = diff(neuron_startTimes);
            if(isempty(neuron_startTimes) || sum(d <= 0) > 0 || sum(neuron_startTimes > obj.dim_N()) > 0 || sum(neuron_startTimes < 1) > 0)
                error('Invalid Y_startTimes: ith element of vector must be index of Y for first obs for neuron i. Must be increasing.');
            end
            d = diff(trial_startTimes);
            if(isempty(trial_startTimes) || sum(d <= 0) > 0 || sum(trial_startTimes > obj.dim_N()) > 0 || sum(trial_startTimes < 1) > 0)
                error('Invalid trial_startTimes: ith element of vector must be index of Y for first obs for trial i (assumed to be trial per neuron). Must be increasing.');
            end
            
            %sets up start time indicies 
            %  appends an "N" onto the ends so that, for all neurons, the
            %  index can be obj.obsRange_Y[ii]:(obj.obsRange_Y[ii]-1)
            obj.obsRange_Y     = uint32([neuron_startTimes(:);obj.dim_N()+1])-1;
            obj.obsRange_trial = uint32([trial_startTimes(:);obj.dim_N()+1])-1;
            
            %precomputes the constant term of the Poisson log likelihood
            %for the bin width and the Y's
            % NOTE: not sure why these are here anymore
            obj.const_poissLL =  0;%- sum(gammaln(double(Y)+1));
            obj.const_trial_poissLL =0;% nan(obj.dim_M,1);
            
            %sets the hyperparameter vars
            obj.function_prior = function_prior;
            obj.dim_H = uint32(dim_H);
            
            %null values
            obj.gpuSinglePrecision = gpuSinglePrecision;
            obj.gpuObj_ptr = uint64(0);
            obj.gpus = [];
        
            obj.hessiansEnabled = true;
            %obj.Groups = [];
            
            if(all(ismember(obj.obsRange_Y,obj.obsRange_trial)))
                obj.trsPerNeuron = cell(obj.dim_P,1);
                for pp = 1:obj.dim_P
                    trs = obj.obsRange_trial >= obj.obsRange_Y(pp) & obj.obsRange_trial < obj.obsRange_Y(pp+1);
                    obj.trsPerNeuron{pp} = find(trs);
                end
            else
                obj.trsPerNeuron = [];
                warning('trial structure needs to be one neuron per trial: SG computations not enabled!');
            end
            
            if(~isempty(GLMGroup))
                if(size(GLMGroup,1) ~= obj.dim_N)
                    error('GLMGroup contains an invalid number of rows!');
                end
                
                obj.dim_K = size(GLMGroup,2);
            else
                obj.dim_K = 0;
            end
            if(gpuSinglePrecision)
                obj.GLMGroup = single(GLMGroup);
            else
                obj.GLMGroup = double(GLMGroup);
            end
            
            obj.Y_mask_trial = true(obj.dim_M,1);
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% METHODS to get consistent dimension information
        function [dim] = get.dim_P(obj)
            dim = uint32(length(obj.obsRange_Y)-1);
        end
        function [dim] = get.dim_M(obj)
            dim = uint32(length(obj.obsRange_trial)-1);
        end
        function [dim] = get.dim_N(obj)
            dim = uint32(length(obj.Y));
        end
        function [dim] = get.dim_J(obj)
            dim = uint32(length(obj.Groups));
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = addGroup(obj,rank_c,max_rank,prior,nhyperparams,name,Regressors, RegressorIndex, constraintTypes)
        %% METHOD [obj] = addGroup(rank_c,prior,nhyperparams,name,Regressors, RegressorIndex)
        %	Adds one new group of covariates and coefficients to the GMLM.
        %   The coefficient group is structured as a low-rank tensor.
        %
        %   inputs:
        %       rank [scalar] (positive int)
        %           rank of the group
        %
        %       max_rank [scalar] (positive int)
        %           maximum rank of the group
        %
        %       prior [1 x 1] [function(params_group)]
        %           Function for log prior distribution over group parameters.
        %           Set to empty if no prior.
        %           It also can return derivates + portions of hessian too.
        %           Function must take the form:  [log_prior] =  prior(params_group) 
        %
        %               H [nhyperparameters x 1] (real)
        %               vector of hyperparameters 
        %               V [P x rank_c] (real)
        %               	matrix of neuron weighting (coefficients for this tensor
        %                   group)
        %
        %               T [R x 1] (cell)
        %                   T{ii} = [dim_T[ii] x rank_c] (real)
        %                   Coefficients for the ith dimension of this tensor group
        %
        %               log_prior is struct with fields
        %                   log_prior is scalar
        %                   dT [dim_S x 1] (cell array), derivative for each T. dT{ii}
        %                           cell is the same size as T{ii}
        %                   dV [] matrix, derivative for V
        %                   dH = matrix, derivative for hyperparams
        %                   d2T [dim_S x 1] (cell array), hessian for each T (individually)
        %                   d2V = matrix, hessian for V
        %
        %       nhyperparams [scalar] (non-negative int)
        %           number of hyperparameters that "prior" takes
        %
        %       name (string)
        %           OPTIONAL (set empty: default = "Group %d")
        %           Name for the group
        %           
        %       
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
        %                   When Regressors.ss  is a full design matrix,  RegressorIndex.ss can be empty. This implies 
        %                   RegressorIndex.ss[ii,aa] = (dim_N * (aa-1)) + ii -> this requires that dim_F[ss] = dim_A*dim_N 
        %
        %                   When RegressorIndex.ss is not empty, RegressorIndex.ss[ii,aa], the regressors for observation ii, event aa,
        %                   are Ressors.ss[ind_A_ss[ii,aa],:].
        %
        %                   NOTE: if ind_A indexes out-of-bound (RegressorIndex.ss[ii,aa] <= 0 or RegressorIndex.ss[ii,aa] > dim_F[ss]),
        %                         it's assumed that Regressors.ss[RegressorIndex.ss[ii,aa],:] = 0 for convenience
        %
        %               Total contribution of this group to the rate for observation ii is
        %                   \sum_{aa=1}^{dim_A} \left( \sum_{rr = 1}^{dim_R} V[ind_C[ii],rr] \left(\prod_{ss = 1}^S (F_{ss}[ind_A[ii,aa],:]*T_{ss}[:,rr]  \right)\right)
        %
        %           For a 3 dim tensor of time x stim x neuron, the arguments will hole:
        %           	Regressor.time,RegressorIndex.time,Regressor.stim,RegressorIndex.stim (the third dim - the neuron dim, is implicit)
        %               where Regressor.ss is a covariate matrix, and RegressorIndex.ss is of size dim_Nxdim_A  
        %               Each RegressorIndex.ss indexes a row in Regressor.ss for observation i.
        %           	If dim_A>1, the tensor group is evented K times for the different rows of the RegressorIndex.ss: in DMC task, dim_A=2 for the "sample" and "test" stimulus presentations
        %
        %
            if(obj.isOnGPU())
                warning('Freeing GPU vars before adding new grpi[ to GMLM. The GPU vars must be reinitialized.');
                obj.freeGPU();
            end
            
            if(isempty(name))
                name = sprintf("Group %d",numel(obj.Groups)+1);
            end
            if(~ischar(name) && ~isstring(name))
                error('call to GMLM/addGroup: input ''name'' must be a string');
            end
            
            %check if group name is unique
            if(obj.getGroupNum(name) > 0)
                error('Group with name %s already exists!',name);
            end
            
            if((~istable(RegressorIndex) && ~isstruct(RegressorIndex)) || (~istable(RegressorIndex) && ~isstruct(RegressorIndex)))
                error('"Regressors" and "ResgressorIndex" must be a struct or data table.');
                %note: Regressors is allowed to be a table even though this isn't mentioned in the header: each entry 
                %      could be complete (one row for each observation) and the regressors indices could be all empty
            end
            
            if(nargin < 9)
                constraintTypes = [];
            end
            
            newGroup = GMLM_tensorGroup(Regressors, RegressorIndex,obj.gpuSinglePrecision,rank_c,obj.dim_N(),obj.obsRange_Y,name,prior,nhyperparams,max_rank,constraintTypes);
            if(~isempty(newGroup))
                if(isempty(obj.Groups))
                    obj.Groups = newGroup;
                else
                    obj.Groups(numel(obj.Groups)+1) = newGroup;
                end
            end
        end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [jj] = getGroupNum(obj,name)
        %% METHOD [J] = getGroupNum(name)
        %	Returns the group number of a tensor group with the given name
        %
        %   inputs;
        %       name (string)
        %           The name of the desired group (NOT CASE SENSITIVE)
        %
        %	returns:
        %       jj [scalar] (int)       
        %           The group number if found. Otherwise, returns 0.
            J = numel(obj.Groups);
            jj = 0;
            if(~ischar(name) && ~isstring(name))
                error('call to GMLM/getNumGroups(name): name must be a string');
            end
            
            for jj_idx = 1:J
                if(strcmpi(obj.Groups(jj_idx).name,name))
                    if(jj ~= 0)
                        error('Multiple groups with name %s found!',name);
                    else
                        jj = jj_idx;
                    end
                end
            end
        end
        
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [J] = getNumGroups(obj)
        %% METHOD [J] = getNumGroups()
        %	Returns the number of tensor groups currently placed in the GMLM.
        %
        %	returns:
        %       J [scalar] (int)       
        %           Number of groups
            J = numel(obj.Groups);
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [R] = getGroupRank(obj,grp)
        %% METHOD [R] = getGroupRank(grp)
        %	Returns the tensor rank of specified coefficient group.
        %
        %   inputs:
        %       grp (string) OR [scalar] (positive int)
        %           The requested group name OR number.
        %
        %	returns:
        %       R [scalar] (int)       
        %           Rank of group jj. 
        %           Defaults to 0 if the group number is invalid.
        %
        %
            if(isstring(grp) || ischar(grp))
                jj = obj.getGroupNum(grp);
            elseif(~isscalar(grp) || ceil(grp) ~= grp)
                error('Invalid argument type.');
            end
                
            if(jj < 1 || jj > numel(obj.Groups))
                warning('Invalid group number!');
                R = 0;
            else
                R = obj.Groups(jj).dim_R;
            end
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = setGroupRank(obj,grp,rank)
        %% METHOD [obj] = setGroupRank(grp,rank,paramObj)
        %	Sets the tensor rank of specified coefficient group.
        %
        %   inputs:
        %       grp (string) OR [scalar] (positive int)
        %           The requested group name OR number.
        %
        %       rank [scalar] (positive int)       
        %           new rank of group jj. 
        %
            if(isstring(grp) || ischar(grp))
                jj = obj.getGroupNum(grp);
            elseif(~isscalar(grp))
                error('Invalid argument type for ''grp''.');
            else
                jj = grp;
            end
                
            if(jj < 1 || jj > numel(obj.Groups)  || ceil(jj) ~= jj)
                error('Invalid group!');
            else
                 obj.Groups(jj).dim_R = rank;
            end
        end

        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = setPrior(obj,function_prior,dim_H)
        %% METHOD [] = setPrior(function_prior,dim_H);
        %   Changes the log prior for W.
        %
        %  	inputs:
        %       function_prior (function)
        %           OPTIONAL (default = empty
        %           Function for log prior distribution over baseline rates.
        %           Set to empty if no prior.
        %           See GMLM constructor for a complete description.
        %
        %       dim_H [scalar] (int)
        %           OPTIONAL (default = 0)
        %           Number of hyperparameters for the prior over neuron base rates. (How many hyperparameters to send into function_prior) 
        
            %default values
            if(nargin < 2 || isempty(dim_H))
                dim_H = 0;
            end
            if(nargin < 1 || isempty(function_prior))
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
        function [obj] = setGroupPrior(obj,groupId,function_prior,dim_H)
        %% METHOD [] = setGroupPrior(function_prior,dim_H);
        %   Changes the log prior for a group's parameters.
        %
        %  	inputs:
        %       function_prior (function)
        %           OPTIONAL (default = empty
        %           Function for log prior distribution over baseline rates.
        %           Set to empty if no prior.
        %           See GMLM constructor for a complete description.
        %
        %       dim_H [scalar] (int)
        %           OPTIONAL (default = 0)
        %           Number of hyperparameters for the prior over neuron base rates. (How many hyperparameters to send into function_prior) 
            obj.Groups(groupId) = obj.Groups(groupId).setPrior(function_prior,dim_H);
        end
        

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [onGPU] = isOnGPU(obj)
        %% METHOD [valid] = isOnGPU()
        %	Checks all the GMLM variables to see if they are currently loaded to one or more GPUs.
        %
        %	returns:
        %       onGPU [scalar] (logical)       
        %           True if the GMLM data appears to be completely loaded onto GPU(s).
        %
            onGPU = obj.gpuObj_ptr ~= 0;
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = freeGPU(obj)
        %% METHOD [obj] = freeGPU()
        %	If the GMLM is loaded to the GPU, frees all GPU variables.
        %
            if(obj.isOnGPU())
                %call free mex file
                kcGMLM_mex_clear(obj);
                
                %make sure pointer set to 0
                obj.gpuObj_ptr = uint64(0);
                obj.gpus = [];
            end
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = toGPU(obj,gpusToUse, batchSize)
        %% METHOD [obj] = toGPU(gpusToUse,isSinglePrecision)
        %	Loads the GMLM data onto one or more GPUs. The data is split into one block per GPU, for a total of D
        %	blocks. The data is split approximately evenly across devices (assumes each GPU is about equal).
        %
        %   inputs:
        %       gpusToUse [D x 1] (non-negative ints)
        %           The GPUs to load the data onto. D is the number of blocks the data will be split into.
        %
        %       isSinglePrecision [scalar] (logical)
        %           OPTIONAL (default = true)
        %           If true, uses single precision arithmetic on GPU and double precision otherwise. 
            
            if(obj.isOnGPU())
                error('GMLM is already loaded to GPU!');
            end
            
            if(nargin < 2 && gpuDeviceCount() == 1)
                gpusToUse = 0;
            end
            if(nargin < 3 || isempty(batchSize))
                batchSize = 2^20;
            end
            
            if(nargin < 2 || isempty(gpusToUse))
                error('no GPUs specified!');
            elseif(sum(gpusToUse(:)<0) > 0 || sum(gpusToUse(:) >= gpuDeviceCount()) > 0 || sum(gpusToUse ~= round(gpusToUse)) > 0)
                error('invalid GPUs given!');
            elseif(obj.getNumGroups() < 1)
                error('Cannot put object on GPU: needs at least 1 tensor group!');
            else
                
                gpusToUse = uint32(gpusToUse(1:min(obj.dim_N,numel(gpusToUse)))); %this shouldn't ever matter, but can't use more GPUs than observations
                obj.gpus = gpusToUse(:);
                
                %set which observations go to which gpu
                ngpus = numel(gpusToUse);
                tper = ceil(double(obj.dim_N)/ngpus);
                
                obsRange = uint32((0:(ngpus-1))*tper);
                obj.obsRange_blocks = obsRange;
                
                %call new mex function to construct GPUGMLM C++ object with data loaded to GPUs
                maxBatchSize = uint64(batchSize);
                ptr = kcGMLM_mex_create(obj, obsRange, gpusToUse, maxBatchSize);
                obj.gpuObj_ptr = ptr;
                
                % set mask if needed
                if(~all(obj.Y_mask_trial))
                    kcGMLM_mex_maskTrials(obj);
                end
            end
        end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = maskTrials(obj, trialMask)
            if(nargin < 2 || isempty(trialMask))
                trialMask = true(obj.dim_M, 1);
            end
            
            if(~islogical(trialMask))
                trialMask = logical(trialMask);
            end
            
            if(numel(trialMask) ~= obj.dim_M && ~isempty(trialMask))
                error('Input to mask trials must be of size dim_M!');
            end
            
            obj.Y_mask_trial(:) = trialMask;
            
            if(obj.isOnGPU)
                kcGMLM_mex_maskTrials(obj);
            end
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [lambda] = computeRate_cpu(obj,paramObj)
        %% METHOD [lambda] = computeRate_cpu(paramObj)
        %	Computes the (log) spike rate for each observation for a setting of the model parameters.
        %   The computation is performed on the CPU (no GPUs needed), and thus can be very slow.
        %
        %   This function is primarily intended for debugging, and thus does not return any derivatives.
        %
        %   inputs:
        %       paramObj (GMLM_params)
        %           The current model parameters. 
        %
        %   returns:
        %       lambda [dim_N x 1] (real)
        %           The log rate for each observation under the GMLM.
            
            %if(isa(paramObj,'GMLM_params') && paramObj.isValidParamObj(obj))
                lambda = zeros(obj.dim_N(),1);
                
                %% get constribution of each group
                J = obj.getNumGroups();
                for jj = 1:J
                    lambda = lambda + obj.Groups(jj).computeRate_cpu(paramObj.Groups(jj));
                end

                %% add constant
                for pp = 1:obj.dim_P()
                    Y_idx = (obj.obsRange_Y(pp)+1):(obj.obsRange_Y(pp+1));

                    if(~isempty(Y_idx))
                        lambda(Y_idx) = lambda(Y_idx) + paramObj.W(pp);
                    end
                    
                    %% add GLM terms
                    if(obj.dim_K > 0)
                        lambda(Y_idx) = lambda(Y_idx) + obj.GLMGroup(Y_idx,:)*paramObj.B(:,pp);
                    end
                end
                
                lambda = lambda + log(obj.const_dt);
%                 lambda(:) = 0.01;
%             else
%                 error('Invalid parameters!');
%             end
        end
        
        
       %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ll, lambda, ll_0, ll_t] = computeLL_cpu(obj,paramObj)
        %% METHOD [lambda] = computeLL_cpu(paramObj)
        %	Computes the log likelihood for a setting of the model parameters.
        %   The computation is performed on the CPU (no GPUs needed), and thus can be very slow.
        %
        %   This function is primarily intended for debugging, and thus does not return any derivatives.
        %
        %   inputs:
        %       paramObj (GMLM_params)
        %           The current model parameters. 
        %
        %   returns:
        %       ll [scalar] (real)
        %           The Poisson log likelihood for the GMLM.
        %
        %       lambda [dim_N x 1] (real)
        %           The log rate for each observation under the GMLM.
            lambda = obj.computeRate_cpu(paramObj);
            
            Y_c = double(obj.Y);
%             Y_c(:) = 1;
            ll_0 = -exp(min(90,lambda)) + Y_c.*lambda - gammaln(Y_c + 1);
            
            ll_t = nan(obj.dim_M,1);
            for ii = 1:obj.dim_M
                si = obj.obsRange_trial(ii)+1;
                ei = obj.obsRange_trial(ii+1);

                ll_t(ii) = sum(ll_0(si:ei));
            end
            ll = sum(ll_t .* obj.Y_mask_trial);
        end
    
            
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct] = computeLPost(obj, paramStruct, optsStruct, highPrecisionLikelihood, singleNeuronOption)
        %% METHOD [resultsStruct = computeLPost(paramStruct, resultsStruct)
        %
        %   Computes the log posterior of the GMLM for a given set of parameters.
        %   This function can also compute derivatives of the log posterior.
        %
        %   inputs:
        %       paramStruct
        %           The struct of the current model parameters. 
        %
        %       optsStruct
        %           The struct of opts for what to compute
        %
        %       highPrecisionLikelihood
        %           OPTIONAL (default = false)
        %           If true and optsStruct.compute_trialLL, this will compute the likelihood as a double given the sum of the likelihoods for each trial.
        %           When the precision is single, the accuracy of the total likelihood may suffer a bit. But if we're already bringing the individual trial
        %           likelihoods to CPU memory, we might as well sum up the total likelihood in higher precision based on those individual trial scores.
        %
        %   outputs:
        %       resultsStruct 
        %           The struct containing all requested results. 
        %
            if(nargin < 4)
                highPrecisionLikelihood = false;
            end
            if(nargin < 5)
                singleNeuronOption = -1;
            end
            resultsStruct = obj.computeLL(paramStruct, optsStruct, highPrecisionLikelihood, singleNeuronOption);

            resultsStruct = obj.addLPrior(paramStruct, resultsStruct);
            
            resultsStruct.log_post = resultsStruct.log_like_0 + resultsStruct.log_prior;
        end
        
        function [resultsStruct] = computeLL(obj, paramStruct, optsStruct, highPrecisionLikelihood, singleNeuronOption)
            if(~obj.isOnGPU())
                error('GMLM has not been loaded to GPU! (call gmlm.toGPU(deviceNumbers)')
            end
            if(nargin < 5)
                singleNeuronOption = -1;
            end
            
            resultsStruct = obj.getEmptyResultsStruct(optsStruct);
            obj.computeLL_internal(paramStruct, resultsStruct, singleNeuronOption);
            if(nargin > 3 && highPrecisionLikelihood  && optsStruct.compute_trialLL)
                resultsStruct.log_like_0 = sum(double(resultsStruct.trialLL));
            else
                resultsStruct.log_like_0 = resultsStruct.log_like;
            end
            %unfolds hessian of W
            if(~isempty(resultsStruct.d2W_like))
                resultsStruct.d2W = diag(resultsStruct.d2W_like);
            else
                resultsStruct.d2W = [];
            end
            
        end
        
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [totalHess, totalChol, cholFailures] = getNegativeNeuronLoadingHessians(obj, resultsStruct, computeChol, fullBlkDiag, diagOnly, nns)
            if(nargin < 6 || isempty(nns))
                nns = 1:obj.dim_P;
            end
            P = numel(nns);
            
            sizePerNeuron = 1 + obj.dim_K;
            for jj = 1:obj.dim_J
                sizePerNeuron = sizePerNeuron + obj.Groups(jj).dim_R;
            end
            
            if(nargin < 5 || isempty(diagOnly))
                diagOnly = false;
            end
            if(nargin < 4 || isempty(fullBlkDiag))
                fullBlkDiag = false;
            end
            if(nargin < 3 || isempty(computeChol))
                computeChol = true;
            end
            
            if(fullBlkDiag)
                totalChol = zeros(sizePerNeuron*P,sizePerNeuron*P);
            else
                totalChol = zeros(sizePerNeuron,sizePerNeuron,P);
            end
            
            totalHess = totalChol;
            cholFailures = 0;
            for pp_idx = 1:P
                pp = nns(pp_idx);
                %% assemble the Hessian for each neuron
                neuronHess = zeros(sizePerNeuron,sizePerNeuron);
                neuronHess(1,1) = resultsStruct.d2W(pp,pp); %d2W
                neuronHess(1 + (1:obj.dim_K), 1 + (1:obj.dim_K)) = resultsStruct.d2B((1:obj.dim_K) + (pp-1)*obj.dim_K,:);
                neuronHess(1                , 1 + (1:obj.dim_K)) = resultsStruct.d2WB(:,pp)';
                
                r_ctr = 1 + obj.dim_K;
                for jj = 1:obj.dim_J
                    rr = obj.Groups(jj).dim_R;
                    neuronHess(1,(1:rr) + r_ctr) = resultsStruct.d2WV{jj}(pp,:); %d2Vw
                    neuronHess(1 + (1:obj.dim_K),(1:rr) + r_ctr) = resultsStruct.d2BV{jj}((1:obj.dim_K) + (pp-1)*obj.dim_K,:);
                    
                    d2V_c = resultsStruct.Groups(jj).d2V((1:rr) + (pp-1)*rr,:);
                    neuronHess((1:rr) + r_ctr   ,(1:rr) + r_ctr) = d2V_c; %d2V
                    
                    
                    r_ctr2  = r_ctr+rr;
                    for jj2 = (jj+1):obj.dim_J
                        rr2 = obj.Groups(jj2).dim_R;
                        neuronHess((1:rr) + r_ctr,(1:rr2) + r_ctr2) = resultsStruct.d2VV{jj,jj2}((1:rr) + (pp-1)*rr,:); %d2V
                        r_ctr2 = r_ctr2 + rr2;
                    end
                    
                    r_ctr = r_ctr + rr;
                end
                
                if(diagOnly)
                    neuronHess = diag(diag(neuronHess));
                else
                    neuronHess = triu(neuronHess) + triu(neuronHess,1)'; %force symmetry
                end
                c = -neuronHess;
                
                %%
                if(computeChol)
                    try
                        ch = chol(c);
                    catch
                        cholFailures = cholFailures + 1;
                        c = -neuronHess;

                        ch = diag(sqrt(diag(c)));
                    end
                    c = ch;
                end
                if(fullBlkDiag)
                    idx = (pp_idx-1)*sizePerNeuron + (1:sizePerNeuron);
                    totalHess(idx,idx) = c;
                else
                    totalHess(:,:,pp_idx) = c;
                end
            end
            
            if(cholFailures > 0)
                fprintf('>>Warning: Cholesky decomposition of Hessian failed in GMLM/getNegativeNeuronLoadingHessians: appoximated for %d cells\n', cholFailures);
            end
        end
        
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [] = computeLL_internal(obj, paramStruct, resultsStruct, singleNeuronOption)
        %% METHOD computeLL
        %
        %   Computes the Poisson log likelihood of the GMLM for a given set of parameters.
        %   This function can also compute derivatives of the log likelihood.
        %
        %   inputs:
        %       paramStruct
        %           The struct of the current model parameters. 
        %
        %   inputs/outputs:
        %       resultsStruct
        %           writes requested results into this object if provides. If any fields are empty, results are not computed.
        %           NOTE: This calls a mex function and it is pass by reference, not value as is typical in MATLAB!
        %
            if(nargin < 4)
                singleNeuronOption = -1;
            end
            kcGMLM_mex_computeLL(obj.gpuObj_ptr, obj.gpuSinglePrecision, paramStruct, resultsStruct, int32(singleNeuronOption-1)); %the minus 1 is to translate into 0 indexing
        end
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct] = addLPrior(obj,paramStruct,resultsStruct)
        %% METHOD [resultsStruct] = addLPrior(paramObj)
        %  Computes the log prior (and any requested derivatives) for the given parameters.
        %  Adds the results to the given results struct.
        %
        %   inputs:
        %       
        %       paramStruct
        %           The struct of the current model parameters. 
        %       resultsStruct
        %           writes requested results into this object if provides. If any fields are empty, results are not computed.
        %
        %   outputs:
        %       resultsStruct 
        %           The results struct containing all requested results with prior terms added.
        %

            if(nargout == 0)
                return;
            end
            
            if(~isempty(obj.function_prior))
                resultsStruct = obj.function_prior(obj,paramStruct,resultsStruct);
            else
                resultsStruct.log_prior_w = 0;
                if(~isempty(resultsStruct.dH))
                    resultsStruct.dH(:) = 0;
                end
            end
            for jj = 1:obj.dim_J
                resultsStruct.Groups(jj) = obj.Groups(jj).addLPrior(paramStruct.Groups(jj),resultsStruct.Groups(jj));
            end
            resultsStruct.log_prior = double(resultsStruct.log_prior_w);
            for jj = 1:obj.dim_J
                resultsStruct.log_prior = resultsStruct.log_prior + double(resultsStruct.Groups(jj).log_prior);
            end
        end
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct] = getEmptyParamStruct(obj,includeH)
        %% METHOD [paramStruct] = getEmptyParamStruct()
        %	Builds a blank structure with all the GMLM model parameters given all currently included groups.
        %
        %   By default, all values are set to 0, but everything is the correct size.
        %
        %	returns:
        %       paramStruct [struct]        
        %           A blank (zero-filled) structure of the model parameters.
        %
        %           Fields for model parameters:
        %               W [P x 1] (real): vector of constant rate parameters per neuron 
        %               H [dim_H x 1]  (real)   : the hyperparameters for W
        %
        %               Groups [J x 1] (struct): parameters for each tensor group
        %                   Fields for model parameters:
        %                       V [P x rank_jj] (real)      : The low-rank factirs for the neuron weighting.
        %                       T [S_jj x 1]    (cell array): the matrices of PARAFAC decomposed factors for the remaining
        %                                                     tensor dimenions.  Each is [tt[ii] x rank_jj]
        %                       H [nhyper_jj x 1]  (real)   : the hyperparameters for this group
        %
        %                       dimNames        (string array): names of each dimension in T
        %
        %
            dataType = 'double';
            if(obj.gpuSinglePrecision)
                dataType = 'single';
            end
            if(nargin < 2 || isempty(includeH))
                includeH = true;
            end
            
            paramStruct.W = zeros(obj.dim_P,1,dataType);
            paramStruct.B = zeros(obj.dim_K,obj.dim_P,dataType);
            if(includeH)
                paramStruct.H = zeros(obj.dim_H,1,'double');
                paramStruct.Groups = struct('T',[],'T_type',[],'V',[],'H',[],'dimNames',[],'name',[]);
            else
                paramStruct.Groups = struct('T',[],'T_type',[],'V',[],'dimNames',[],'name',[]);
            end
            
            
            J = obj.getNumGroups();
            for jj = 1:J
                paramStruct.Groups(jj) = obj.Groups(jj).getEmptyParamStruct(includeH);
            end
        end
        
        function [paramStruct] = getRandomParamStruct(obj, init_gain, includeH, init_gain_glm)
        %% METHOD [paramStruct] = getRandomParamStruct(init_gain)
        %	Builds a random structure with all the GMLM model parameters given all currently included groups.
        %
        %   By default, all Groups(jj).T values are orthonormal. Groups(jj).V are zero-mean normal with std init_gain.
        %   W are selected from a normal with mean and standard deviation given by the mean rates in Y across neurons.
        %   All hyperparams H are i.i.d standard normal.
        %
            if(nargin < 2 || isempty(init_gain))
                init_gain = 1;
            end
            if(nargin < 4 || isempty(init_gain_glm))
                init_gain_glm = 0.1;
            end
            if(nargin < 3 || isempty(includeH))
                includeH = true;
            end
            
            paramStruct = obj.getEmptyParamStruct(includeH);
            
    
            %W are centered at the log mean firing rates
            mr = nan(obj.dim_P,1);
            for pp = 1:obj.dim_P
                mr(pp) = mean(obj.Y((obj.obsRange_Y(pp)+1):obj.obsRange_Y(pp+1)));
                if(mr(pp) == 0)
                    error('Zero spikes found for neuron %d: recommend not using this cell!',pp);
                end
            end
            mr  = log(mr) - log(obj.const_dt);
            paramStruct.W(:) = randn(obj.dim_P,1)*std(mr) + mean(mr);
            
            
            paramStruct.B(:) = randn(obj.dim_K,obj.dim_P)*init_gain_glm;
    
            %H are i.i.d. standard normal
            if(isfield(paramStruct, 'H') && ~isempty(paramStruct.H))
                paramStruct.H(:) = randn(obj.dim_H,1);
            end

            %for each group
            for jj = 1:obj.dim_J
                %H are i.i.d. standard normal
                if(isfield(paramStruct.Groups(jj), 'H') && ~isempty(paramStruct.Groups(jj).H(:)))
                    paramStruct.Groups(jj).H(:) = randn(size(paramStruct.Groups(jj).H));
                end

                %V are i.i.d. standard normal
                paramStruct.Groups(jj).V(:) = randn(size(paramStruct.Groups(jj).V))*init_gain;

                %for each dimension
                for ss = 1:obj.Groups(jj).dim_S
                    %orthonormal vectors
                    rr = randn(size(paramStruct.Groups(jj).T{ss}));
                    if(size(rr,2) <= size(rr,1))
                        paramStruct.Groups(jj).T{ss}(:) = orth(rr);
                    else
                        paramStruct.Groups(jj).T{ss}(:) = rr./sqrt(sum(rr.^2,1));
                    end
                end
            end
        end
  
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [optsStruct] = getEmptyOptsStruct(obj,defaultValue,defaultValueHessians)
        %% METHOD [optsStruct] = getEmptyOptsStruct()
        %	Builds a compute options with all fields set to defaultValue.
        %
        %   inputs:
        %       defaultValue
        %           OPTIONAL (default = false)
        %           Whether all options are on or off
        %       defaultValueHessians
        %           OPTIONAL (default = defaultValue)
        %           Whether all options for hessian computation are on or off
        %
        %
        %	returns:
        %       optsStruct [struct]        
        %           A structure of the compute options for the likelihood/posterior. All values are logical
        %
        %           Fields for model parameters:
        %               trialLL       : compute the log likelihood of each trial
        %
        %               compute_dW    : compute derivative of of posteior w.r.t W
        %               compute_d2W   : compute second derivative of of posteior w.r.t W
        %
        %               compute_dH    : compute derivative of prior over W w.r.t the hyperparams
        %              
        %               Groups [J x 1] (struct): options for each tensor group. See GMLM_tensorGroup.getEmptyOptsStruct();
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
            
            optsStruct.compute_trialLL = defaultValue;
            optsStruct.compute_dW = defaultValue;
            optsStruct.compute_d2W = defaultValueHessians;
            optsStruct.compute_d2WV = logical(ones(obj.dim_J,1)*defaultValueHessians);
            optsStruct.compute_d2BV = logical(ones(obj.dim_J,1)*(defaultValueHessians  & obj.dim_K > 0));
            optsStruct.compute_d2VV = logical(ones(obj.dim_J,obj.dim_J)*defaultValueHessians);
            
            optsStruct.compute_dB   = defaultValue & obj.dim_K > 0;
            optsStruct.compute_d2B  = defaultValueHessians & obj.dim_K > 0;
            optsStruct.compute_d2WB = defaultValueHessians & obj.dim_K > 0;
            
            optsStruct.compute_dH = defaultValue;
            optsStruct.Groups      = struct('compute_dT',[],'compute_dV',[],'compute_d2T',[],'compute_d2V',[],'compute_dH',[]);
            
            J = obj.getNumGroups();
            for jj = 1:J
                optsStruct.Groups(jj) = obj.Groups(jj).getEmptyOptsStruct(defaultValue,defaultValueHessians);
            end
        end
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct] = getEmptyResultsStruct(obj, optsStruct)
        %% METHOD [resultsStruct] = getEmptyResultsStruct()
        %	Builds a blank structure with all the GMLM model likelihood outputs given all currently included groups.
        %
        %   By default, all values are set to 0, but everything is the correct size.
        %
        %   inputs:
        %       optsStruct
        %           OPTIONAL (by default, everything will be on)
        %           A compute options struct for what should be computed by the likelihood/prior functions. See getEmptyOptsStruct()
        %
        %
        %	returns:
        %       resultsStruct [struct]        
        %           A blank (zero-filled) structure of the possible model outputs.
        %
        %           Fields for model parameters:
        %               log_like  [scalar] : the log likelihood
        %               trialLL  [M x 1] (real): vector of constant rate parameters per neuron 
        %
        %               dW       [P x 1] (real): vector of constant rate parameters per neuron 
        %               dH       [H x 1] (real): vector of constant rate parameters per neuron 
        %               d2W_like [P x 1] (real): vector of constant rate parameters per neuron. Note: only stores the diagonal
        %
        %               Groups [J x 1] (struct): results for each tensor group. See GMLM_tensorGroup.getEmptyResultsStruct();
        %
            if(nargin < 2)
                optsStruct = obj.getEmptyOptsStruct(true);
            end
            
            if(obj.gpuSinglePrecision)
                dataType = 'single';
            else
                dataType = 'double';
            end
            
            if(optsStruct.compute_dW)
                resultsStruct.dW  = zeros(obj.dim_P,1,dataType);
                
                if(optsStruct.compute_d2W)
                    resultsStruct.d2W_like = zeros(obj.dim_P,1,dataType);
                else
                    resultsStruct.d2W_like = [];
                end
            else
                resultsStruct.dW  = [];
                resultsStruct.d2W_like = [];
            end
            
            if(optsStruct.compute_dB)
                resultsStruct.dB  = zeros(obj.dim_K,obj.dim_P,dataType);
                
                if(optsStruct.compute_d2B)
                    resultsStruct.d2B = zeros(obj.dim_K*obj.dim_P,obj.dim_K,dataType);
                else
                    resultsStruct.d2B = [];
                end
            else
                resultsStruct.dB  = [];
                resultsStruct.d2B = [];
            end
            if(optsStruct.compute_d2WB)
                resultsStruct.d2WB = zeros(obj.dim_K,obj.dim_P,dataType);
            else
                resultsStruct.d2WB = [];
            end
            
            if(optsStruct.compute_trialLL)
                resultsStruct.trialLL     = zeros(obj.dim_M,1,dataType);
            else
                resultsStruct.trialLL     = [];
            end
            
            if(optsStruct.compute_dH)
                resultsStruct.dH     = zeros(obj.dim_H,1);
            else
                resultsStruct.dH     = [];
            end
            
            resultsStruct.Groups      = struct('dT',[],'dV',[],'d2V',[],'d2T',[],'dH',[],'log_prior',[]);
            
            resultsStruct.log_like    = zeros(1,1,dataType);
            resultsStruct.log_prior_w = zeros(1,1);
            resultsStruct.log_post    = zeros(1,1);
            
            J = obj.getNumGroups();
            resultsStruct.d2VV = cell(obj.dim_J,obj.dim_J);
            resultsStruct.d2WV = cell(obj.dim_J,1);
            resultsStruct.d2BV = cell(obj.dim_J,1);
            
            for jj = 1:J
                resultsStruct.Groups(jj) = obj.Groups(jj).getEmptyResultsStruct(optsStruct.Groups(jj));
                
                if(optsStruct.compute_d2W && optsStruct.Groups(jj).compute_d2V && optsStruct.compute_d2WV(jj))
                     resultsStruct.d2WV{jj} = zeros(obj.dim_P,obj.Groups(jj).dim_R,dataType);
                end
                if(optsStruct.compute_d2B && optsStruct.Groups(jj).compute_d2V && optsStruct.compute_d2BV(jj))
                     resultsStruct.d2BV{jj} = zeros(obj.dim_P*obj.dim_K,obj.Groups(jj).dim_R,dataType);
                end
            end
            for jj1 = 1:(J-1)
                for jj2 = (jj1+1):J
                    
                    if(optsStruct.Groups(jj1).compute_d2V && optsStruct.Groups(jj2).compute_d2V && optsStruct.compute_d2VV(jj1,jj2))
                         resultsStruct.d2VV{jj1,jj2} = zeros(obj.dim_P*obj.Groups(jj1).dim_R,obj.Groups(jj2).dim_R,dataType);
                    end
                end
            end
        end
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct_xv, results_xv] = getMLE_MAP_xval(obj, trialFoldNumbers, paramStruct_init, max_iters, convergence_delta, jitterAttempts, msgStr_0)
            if(nargin < 2 || numel(trialFoldNumbers) ~= obj.dim_M)
                error('Cross validation assumes trials are withheld');
            end
            if(nargin < 3)
                paramStruct_init = [];
            end
            if(nargin < 4)
                max_iters = [];
            end
            if(nargin < 5)
                convergence_delta = [];
            end
            if(nargin < 6)
                jitterAttempts = [];
            end
            if(nargin < 7)
                msgStr_0 = [];
            end
            
            if(~obj.isOnGPU())
                error('GMLM must be placed on GPU to perform XVal fits');
            end
            
            %% save original mask information
            Y_mask_0 = obj.Y_mask_trial;
            
            %% fit GMLM for each fold
            folds = unique(trialFoldNumbers);
            for ff = 1:numel(folds)
                tts = trialFoldNumbers == folds(ff);
                trialMask = true(obj.dim_M, 1);
                trialMask(tts) = false;
                obj = obj.maskTrials(trialMask);
                
                msgStr = sprintf('Fold %d / %d - %s', ff, numel(folds), msgStr_0);
                if(numel(paramStruct_init) == numel(folds))
                    paramStruct_init_c = paramStruct_init(ff);
                else
                    paramStruct_init_c = paramStruct_init;
                end
                [paramStruct_xv(ff),~,~,~,compute_MAP] = obj.getMLE_MAP(paramStruct_init_c, max_iters, convergence_delta, jitterAttempts, msgStr); %#ok<AGROW>
                %[paramStruct_xv(ff), ~, compute_MAP] = getMLE_MAP(obj, paramStruct_init, msgStr, max_iters); %#ok<AGROW>
            end
            
            %% get test and train likelihoods for each trial
            results_xv.trialLL_train = nan(obj.dim_M, numel(folds));
            results_xv.trialLL_test  = nan(obj.dim_M, numel(folds));
            optsStruct = obj.getEmptyOptsStruct(false, false);
            optsStruct.compute_trialLL = true;
            
            obj = obj.maskTrials(true(obj.dim_M,1));
            for ff = 1:numel(folds)
                if(compute_MAP)
                    resultsStruct = obj.computeLPost(paramStruct_xv(ff), optsStruct);
                else
                    resultsStruct = obj.computeLL(paramStruct_xv(ff), optsStruct);
                end
                
                tts = trialFoldNumbers == folds(ff);
                
                results_xv.trialLL_train(~tts,ff) = resultsStruct.trialLL(~tts);
                results_xv.trialLL_test(  tts,ff) = resultsStruct.trialLL( tts);
            end
            
            %% reset mask
            if(~all(Y_mask_0))
                obj.maskTrials(Y_mask_0);
            end
        end
        
        
    
        function [nlog_post,ndl_post,paramStruct,resultsStruct] = vectorizedNLL_func(obj, w_c, paramStruct, optsStruct, compute_MAP)
            if(nargout > 1)
                optsStruct_0 = optsStruct;
            else
                optsStruct_0 = obj.getEmptyOptsStruct(false, false);
            end
            optsStruct_0.compute_trialLL = true;
            
            paramStruct = obj.devectorizeParamStruct(w_c, paramStruct, optsStruct);
            
            if(compute_MAP)
                resultsStruct = obj.computeLPost(paramStruct, optsStruct_0, true);
                nlog_post = -resultsStruct.log_post;
            else
                resultsStruct = obj.computeLL(paramStruct, optsStruct_0, true);
                resultsStruct.log_post = sum(double(resultsStruct.trialLL));
                
                nlog_post = -resultsStruct.log_post;
            end
            
            if(nargout > 1)
                ndl_post =  obj.vectorizeResultsStruct(resultsStruct, optsStruct, paramStruct);
                ndl_post = -ndl_post;
            end
        end
        
        function [nlog_post,ndl_post,d2nl_post,paramStruct,resultsStruct] = vectorizedNLL_neuronLoadingWeights_func(obj, w_c, paramStruct, optStruct, compute_MAP, nns)
            if(nargin < 6 || isempty(nns))
                nns = 1:obj.dim_P;
            end
            
            optsStruct_0 = optStruct;
            if(nargout ~= 3)
                optsStruct_0.compute_d2W = false;
                optsStruct_0.compute_d2B = false;
                for jj = 1:numel(optsStruct_0.Groups)
                    optsStruct_0.Groups(jj).compute_d2V = false;
                end
            end
            if(nargout ~= 2 && nargout ~= 3)
                optsStruct_0.compute_dW = false;
                optsStruct_0.compute_dB = false;
                for jj = 1:numel(optsStruct_0.Groups)
                    optsStruct_0.Groups(jj).compute_dV = false;
                end
            end
            optsStruct_0.compute_trialLL = true;
            
            paramStruct = obj.devectorizeNeuronLoadingWeights(w_c, paramStruct, nns);
            
            nns_c = -1;
            if(numel(nns) == 1)
                nns_c = nns;
            end
            
            if(compute_MAP)
                resultsStruct = obj.computeLPost(paramStruct, optsStruct_0, true, nns_c);
                nlog_post = -resultsStruct.log_post;
            else
                resultsStruct = obj.computeLL(paramStruct, optsStruct_0, true, nns_c);
                resultsStruct.log_post = sum(double(resultsStruct.trialLL));
                
                nlog_post = -resultsStruct.log_post;
            end
            
            if(nargout == 2 || nargout == 3)
                ndl_post =  obj.vectorizeNeuronLoadingWeightResultsStruct(resultsStruct, optStruct, nns);
                ndl_post = -ndl_post;
            else
                ndl_post = [];
            end
            if(nargout == 3)
                d2nl_post = obj.getNegativeNeuronLoadingHessians(resultsStruct, false, true, false, nns);
            else
                d2nl_post = [];
            end
        end
        
        function [paramStruct] = normalizeParams(obj,paramStruct, useCPFactorization)
            if(nargin < 3 || isempty(useCPFactorization))
                useCPFactorization = false;
            end
            for jj = 1:obj.dim_J
                if(obj.Groups(jj).dim_S == 1)
                    MM = paramStruct.Groups(jj).V * paramStruct.Groups(jj).T{1}';
                    [u,s,v] = svd(double(MM));

                    rr = obj.Groups(jj).dim_R;
                    paramStruct.Groups(jj).V(:)    = u(:,1:rr) * s(1:rr,1:rr);
                    paramStruct.Groups(jj).T{1}(:) = v(:,1:rr);
                else
                    if(useCPFactorization && obj.Groups(jj).dim_R > 1)
                        Us = cell(1, obj.Groups(jj).dim_S + 1);
                        Us{end} = double(paramStruct.Groups(jj).V);
                        for ss = 1:obj.Groups(jj).dim_S
                            Us{ss} = double(paramStruct.Groups(jj).T{ss});
                        end
                        a = ktensor(ones(obj.Groups(jj).dim_R,1),Us);
                        Gt = cp_als(a,obj.Groups(jj).dim_R,'printitn',0,'tol',1e-10,'maxiters',100e3);
                        for ss = 1:obj.Groups(jj).dim_S
                            paramStruct.Groups(jj).T{ss}(:) = Gt.U{ss};
                        end
                        paramStruct.Groups(jj).V(:) = Gt.U{end}.*Gt.lambda';
                    else
                        nn_tot = ones(1,obj.Groups(jj).dim_R);
                        for ss = 1:obj.Groups(jj).dim_S
                            nn = sqrt(sum(paramStruct.Groups(jj).T{ss}.^2,1));
                            paramStruct.Groups(jj).T{ss} = paramStruct.Groups(jj).T{ss}./nn;
                            nn_tot = nn_tot .* nn;
                        end
                        paramStruct.Groups(jj).V     = paramStruct.Groups(jj).V    .*nn_tot;
                    end
                end
            end
        end
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [optsStruct] = getParamCount(obj,optsStruct)
            if(~isfield(optsStruct,'totalParams') || ~isfield(optsStruct,'vw_ranks'))
                %gets total param count
                optsStruct.totalParams = 0;
                optsStruct.vw_ranks    = 0;
                optsStruct.totalHyperparams = 0;
                if(nargin < 2 || optsStruct.compute_dH)
                    optsStruct.totalHyperparams = optsStruct.totalHyperparams + obj.dim_H;
                end
                if(nargin < 2 || optsStruct.compute_dW)
                    optsStruct.totalParams = optsStruct.totalParams + obj.dim_P;
                    optsStruct.vw_ranks    = optsStruct.vw_ranks + 1;
                end
                if(nargin < 2 || optsStruct.compute_dB)
                    optsStruct.totalParams = optsStruct.totalParams + obj.dim_P*obj.dim_K;
                    optsStruct.vw_ranks    = optsStruct.vw_ranks + obj.dim_K;
                end
                for jj = 1:obj.dim_J
                    if(nargin < 2 || optsStruct.Groups(jj).compute_dH)
                        optsStruct.totalHyperparams = optsStruct.totalHyperparams + obj.Groups(jj).dim_H;
                    end
                    if(nargin < 2 || optsStruct.Groups(jj).compute_dV)
                        optsStruct.totalParams = optsStruct.totalParams + (obj.Groups(jj).dim_R*obj.dim_P);
                        optsStruct.vw_ranks    = optsStruct.vw_ranks + obj.Groups(jj).dim_R;
                    end
                end
                for jj = 1:obj.dim_J
                    for ss = 1:obj.Groups(jj).dim_S
                        if(nargin < 2 || optsStruct.Groups(jj).compute_dT(ss))
                            optsStruct.totalParams = optsStruct.totalParams + (obj.Groups(jj).dim_T(ss)*obj.Groups(jj).dim_R);
                        end
                    end
                end
            end
        end
        
        function [HStruct] = getHessianStructure(obj,optsStruct)
            [optsStruct] = getParamCount(obj,optsStruct);
            
            HStruct = ones(optsStruct.totalParams,optsStruct.totalParams);
            
            r = [1;zeros(obj.dim_P-1,1)];
            
            HStruct(1:(optsStruct.vw_ranks*obj.dim_P),1:(optsStruct.vw_ranks*obj.dim_P)) = toeplitz(repmat(r,[optsStruct.vw_ranks,1]));
        end
        
        function [ww] = vectorizeNeuronLoadingWeights(obj,paramStruct,optsStruct,nns)
            if(nargin < 4 || isempty(nns))
                nns = 1:obj.dim_P;
            end
            optsStruct = obj.getParamCount(optsStruct);
            
            %adds each selected param
            ww = zeros(optsStruct.vw_ranks,obj.dim_P);
            
            %adds the W param
            ww(1,:) = paramStruct.W(:);
            ctr = 1;

            %adds the B param
            if(obj.dim_K > 0)
                B = paramStruct.B;
                ww((1:size(B,1))+ctr,:) = B;
                ctr = ctr+size(B,1);
            end

            %adds V param for each group
            for jj = 1:obj.dim_J
                V_c = paramStruct.Groups(jj).V';
                r_c = size(V_c,1);
                ww((1:r_c)+ctr,:) = V_c;
                ctr = ctr + r_c;
            end
            ww = ww(:,nns);
            ww = ww(:);
        end
        
        function [paramStruct] = devectorizeNeuronLoadingWeights(obj,ww, paramStruct, nns)
            if(nargin < 4 || isempty(nns))
                nns = 1:obj.dim_P;
            end
            
            ww = reshape(ww,[],numel(nns));
            
            %sets the W param
            paramStruct.W(nns) = ww(1,:);
            ctr = 1;
            
            %sets the B param
            if(obj.dim_K > 0)
                B = ww((1:obj.dim_K)+ctr,:);
                paramStruct.B(:,nns) = B;
                ctr = ctr+double(size(B,1));
            end

            %adds V param for each group
            for jj = 1:obj.dim_J
                r_c = size(paramStruct.Groups(jj).V, 2);
                V = ww((1:r_c)+ctr,:);
                paramStruct.Groups(jj).V(nns,:) = V';
                ctr = ctr + r_c;
            end
        end
        
        function [dww] = vectorizeNeuronLoadingWeightResultsStruct(obj,resultsStruct,optsStruct, nns)
            if(nargin < 4 || isempty(nns))
                nns = 1:obj.dim_P;
            end
            optsStruct = obj.getParamCount(optsStruct);
            
            %adds each selected param
            dww = zeros(optsStruct.vw_ranks,obj.dim_P);
            
            %adds the W param
            dww(1,:) = resultsStruct.dW;
            ctr = 1;

            %adds the B param
            if(obj.dim_K > 0)
                dB = resultsStruct.dB;
                dww((1:size(dB,1))+ctr,:) = dB;
                ctr = ctr+size(dB,1);
            end

            %adds V param for each group
            for jj = 1:obj.dim_J
                dV_c = resultsStruct.Groups(jj).dV';
                r_c = size(dV_c,1);
                dww((1:r_c)+ctr, :) = dV_c;
                ctr = ctr + r_c;
            end
            dww = dww(:,nns);
            dww = dww(:);
        end
        
        function [ww, hh] = vectorizeParamStruct(obj,paramStruct,optsStruct)
            optsStruct = obj.getParamCount(optsStruct);
            
            %adds each selected param
            ctr = 0; %counter for number of params already added
            ww = zeros(optsStruct.totalParams,1);
            
            %adds the W param
            if(optsStruct.compute_dW)
                ww((1:obj.dim_P)+ctr) = paramStruct.W(:);
                ctr = ctr+double(obj.dim_P);
            end
            
            %adds the B param
            if(optsStruct.compute_dB && obj.dim_K > 0)
                B = paramStruct.B';
                ww((1:numel(B))+ctr) = B(:);
                ctr = ctr+double(numel(B));
            end

            %adds V param for each group
            for jj = 1:obj.dim_J
                if(optsStruct.Groups(jj).compute_dV)
                    n_c = numel(paramStruct.Groups(jj).V);
                    ww((1:n_c)+ctr) = paramStruct.Groups(jj).V(:);
                    ctr = ctr + n_c;
                end
            end

            %adds T params for each group
            for jj = 1:obj.dim_J
                for ss = 1:obj.Groups(jj).dim_S
                    if(optsStruct.Groups(jj).compute_dT(ss))
                        n_c = numel(paramStruct.Groups(jj).T{ss});
                        
                        K = paramStruct.Groups(jj).T{ss};
                        
                        if(optsStruct.Groups(jj).compute_dV)
                            %K = K./sqrt(sum(K.^2,1));
                        end
                        
                        ww((1:n_c)+ctr) = K(:);
                        ctr = ctr + n_c;
                    end
                end
            end
            
            %hyperparams
            if(nargout > 1)
                hh = zeros(optsStruct.totalHyperparams,1);
                ctr = 0;
                if(optsStruct.compute_dH)
                    hh((1:obj.dim_H)+ctr) = paramStruct.H(:);
                    ctr = ctr+double(obj.dim_H);
                end
                for jj = 1:obj.dim_J
                    if(optsStruct.Groups(jj).compute_dH)
                        hh((1:obj.Groups(jj).dim_H)+ctr) = paramStruct.Groups(jj).H(:);
                        ctr = ctr+double(obj.Groups(jj).dim_H);
                    end
                end
            end
        end
        
        function [paramStruct] = devectorizeParamStruct(obj, ww, paramStruct, optsStruct, hh)
            ctr = 0; %counter for number of params already added
            
            %sets the W param
            if(optsStruct.compute_dW)
                paramStruct.W(:) = ww((1:obj.dim_P)+ctr);
                ctr = ctr+double(obj.dim_P);
            end
            
            %sets the B param
            if(optsStruct.compute_dB && obj.dim_K > 0)
                B = ww((1:(obj.dim_P*obj.dim_K))+ctr);
                B = reshape(B,obj.dim_P,obj.dim_K);
                paramStruct.B(:,:) = B';
                ctr = ctr+double(numel(B));
            end

            %adds V param for each group
            for jj = 1:obj.dim_J
                if(optsStruct.Groups(jj).compute_dV)
                    n_c = numel(paramStruct.Groups(jj).V);
                    paramStruct.Groups(jj).V(:) = ww((1:n_c)+ctr);
                    ctr = ctr + n_c;
                end
            end

            %adds T params for each group
            for jj = 1:obj.dim_J
                for ss = 1:obj.Groups(jj).dim_S
                    if(optsStruct.Groups(jj).compute_dT(ss))
                        n_c = numel(paramStruct.Groups(jj).T{ss});
                        
                        
                        K = reshape(ww((1:n_c)+ctr), size(paramStruct.Groups(jj).T{ss}));
                        if(optsStruct.Groups(jj).compute_dV)
                            %K = K./sqrt(sum(K.^2,1));
                        end
                        
                        paramStruct.Groups(jj).T{ss}(:) = K;
                        ctr = ctr + n_c;
                    end
                end
            end
            paramStruct.W_all = ww;
            
            %hyperparams
            if(nargin > 4)
                ctr = 0;
                if(optsStruct.compute_dH)
                    paramStruct.H((1:obj.dim_H)+ctr) = hh((1:obj.dim_H)+ctr);
                    ctr = ctr+double(obj.dim_H);
                end
                for jj = 1:obj.dim_J
                    if(optsStruct.Groups(jj).compute_dH)
                        paramStruct.Groups(jj).H(:) = hh((1:obj.Groups(jj).dim_H)+ctr);
                        ctr = ctr+double(obj.Groups(jj).dim_H);
                    end
                end
                
                paramStruct.H_all = hh;
            end
        end
        
        function [dww, dhh] = vectorizeResultsStruct(obj, resultsStruct, optsStruct, paramStruct)
            optsStruct = obj.getParamCount(optsStruct);
            
            %adds each selected param
            ctr = 0; %counter for number of params already added
            dww = zeros(optsStruct.totalParams,1);
            
            %adds deriv of the W param
            if(optsStruct.compute_dW)
                dww((1:obj.dim_P)+ctr) = resultsStruct.dW(:);
                ctr = ctr+double(obj.dim_P);
            end
            %adds deriv of the B param
            if(optsStruct.compute_dB && obj.dim_K > 0)
                dB =  resultsStruct.dB';
                dww((1:(obj.dim_P*obj.dim_K))+ctr) = dB(:);
                ctr = ctr+double(numel(resultsStruct.dB));
            end

            %adds deriv of V param for each group
            for jj = 1:obj.dim_J
                if(optsStruct.Groups(jj).compute_dV)
                    n_c = numel(resultsStruct.Groups(jj).dV);
                    dww((1:n_c)+ctr) = resultsStruct.Groups(jj).dV(:);
                    ctr = ctr + n_c;
                end
            end

            %adds deriv of T params for each group
            for jj = 1:obj.dim_J
                for ss = 1:obj.Groups(jj).dim_S
                    if(optsStruct.Groups(jj).compute_dT(ss))
                        n_c = numel(resultsStruct.Groups(jj).dT{ss});
                        
                        dZ = resultsStruct.Groups(jj).dT{ss};
                        dK = dZ;
                        
                        if(optsStruct.Groups(jj).compute_dV)
                            %K = paramStruct.Groups(jj).T{ss};
                            %nK = sqrt(sum(K.^2));
                            for ii = 1:size(dZ, 2)
                                %dZdK = (nK(ii)^2*eye(size(dZ,1)) - K(:, ii) * K(:, ii)') ./ nK(ii)^3;
                                %dK(:, ii) = dZdK * dZ(:, ii);
                                
                                %kn = K(:, ii)./nK(ii);
                                %pkn = dZ(:, ii)'*kn;
                                %dK(:, ii) = dZ(:, ii) - kn * pkn;
                            end
                        end
                        
                        dww((1:n_c)+ctr) = dK(:);
                        ctr = ctr + n_c;
                    end
                end
            end
            
            %hyperparams
            if(nargout > 1)
                dhh = zeros(optsStruct.totalHyperparams,1);
                ctr = 0;
                if(optsStruct.compute_dH)
                    dhh((1:obj.dim_H)+ctr) = resultsStruct.dH(:);
                    ctr = ctr+double(obj.dim_H);
                end
                for jj = 1:obj.dim_J
                    if(optsStruct.Groups(jj).compute_dH)
                        dhh((1:obj.Groups(jj).dim_H)+ctr) = resultsStruct.Groups(jj).dH(:);
                        ctr = ctr+double(obj.Groups(jj).dim_H);
                    end
                end
            end
        end
        
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [paramStruct_opt, resultsStruct_opt, log_post, STATUS, compute_MAP] = getMLE_MAP(obj, paramStruct_init, max_iters, convergence_delta, jitterAttempts, msgStr, optAllParamsTogether, maxIters_BFGS)
        %[paramStruct_opt, resultsStruct_opt, compute_MAP] = getMLE_MAP(obj, paramStruct_init, msgStr, max_iters);
        
        %functions for MCMC inference
        [HMC_settings] = setupHMCparams(obj, nWarmup, nTotal, debugSettings)
        [paramStruct, acceptedProps, log_p_accept] = scalingMHStep(obj, paramStruct, HM_scaleSettings)
        [samples, summary, HMC_settings, paramStruct, M_chol] = runHMC_simple(obj, HMC_settings, paramStruct, initializerScale, textOutputFile)
        [accepted,err,paramStruct_new,log_p_accept,resultStruct] = HMCstep_simple(obj, paramStruct_0, M_chol, HMC_state, resultStruct_0)
        
        [HMC_settings] = setupHMCparams_adaptive(obj, nWarmup, nTotal, debugSettings)
        [samples, summary, HMC_settings, paramStruct, M_chol] = runHMC_adaptive(obj, HMC_settings, paramStruct, initializerScale, textOutputFile)
        [MCMC_settings] = setupBarkerMCparams(obj, nWarmup, nTotal, debugSettings)
        [samples, summary, MCMC_settings, paramStruct, M_est] = runBarkerMC(obj, MCMC_settings, paramStruct, initializerScale)
    end
end