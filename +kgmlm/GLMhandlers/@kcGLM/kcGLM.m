%% CLASS kcGLM
%
% This class holds all the regressors and data for a GLM. It handles all calls to the log likelihood functions, and can
% load the data onto the GPU.
% This class only exists to handle the evidence optimization and sampling I needed for the DMC task cells (most applications would be better handled by a stardard GLM library)
% 
%
classdef kcGLM
    properties (Dependent, GetAccess = public, SetAccess = immutable)
        dim_N
        dim_P
        dim_M
        XY
    end
    properties (GetAccess = public, SetAccess = private)
        
        gpuObj_ptr uint64 %uint64 pointer to the c++ object for the GLM loaded to the GPU
        gpus uint32
        
        dim_H uint32 %number of hyperparameters
        function_prior %a function that computes the log prior (and its derivatives)
        
        Y_mask_trial logical
        XY_value
    end
    properties (GetAccess = public, SetAccess = immutable)
        %dim_N %number of observations
        %dim_P %number of coefficients
        %dim_M %number of trials
        
        Y int32 %the observations: spike counts
        X  %the regressors
        
        const_dt %size of time bins
        
        obsRange_trial uint32 %indicies for which observations go with which trial
        
        const_poissLL %the constant term of the Poisson log likelihood
        const_trial_poissLL %the constant term of the Poisson log likelihood for each trial
        
        gpuSinglePrecision logical % [scalar] (logical) if using single precision on group. If false, using double.
    end
    
    
    methods 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = kcGLM(Y, X, trial_startTimes, dt, gpuSinglePrecision, function_prior, dim_H)
        %% CONSTRUCTOR kcGLM(Y,Y_startTimes,trial_startTimes,dt)
        %	Default constructor for creating a GLM object.
        %   This constructor takes all the observations (spike count) information.
        %   The covariates can be added on later.
        %
        %   inputs:
        %       Y [N x 1] (real)    
        %           Vector of all the spike count observations (N is total number of bins).
        %       	Must be structured in terms of trials.
        %       X [N x P] (real)    
        %           Vector of all the regressors (N is total number of bins, P is number of coefficients).
        %       	Must be structured in terms of trials.
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
        %           Function must take the form:  [GLM_results] =  prior(GLM_params,GLM_results) 
        %               This function must only try to set
        %                   GLM_results.log_prior 
        %                   GLM_results.dH
        %               This function should ADD to
        %                   GLM_results.dW
        %                   GLM_results.d2W
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
            
            %initial values for GLM object
            obj.const_dt = dt;
            obj.Y = int32(Y(:)); %converts spike counts to unsigned ints
            if(numel(obj.Y) ~= size(X,1))
                error('design matrix and observation vector are inconsistent');
            end
            
            if(gpuSinglePrecision)
                obj.X = single(X);
            else
                obj.X = double(X);
            end
            obj.XY_value = [];
            
            
            %checks the start time indices: must be strictly increasing, positive, and
            %not greater than N.
            d = diff(trial_startTimes);
            if(isempty(trial_startTimes) || sum(d <= 0) > 0 || sum(trial_startTimes > obj.dim_N()) > 0 || sum(trial_startTimes < 1) > 0)
                error('Invalid trial_startTimes: ith element of vector must be index of Y for first obs for trial i (assumed to be trial per neuron). Must be increasing.');
            end
            
            %sets up start time indicies 
            %  appends an "N" onto the ends so that, for all neurons, the
            %  index can be obj.obsRange_Y[ii]:(obj.obsRange_Y[ii]-1)
            obj.obsRange_trial = uint32([trial_startTimes(:);obj.dim_N()+1])-1;
            
            
            %precomputes the constant term of the Poisson log likelihood
            %for the bin width and the Y's
            
            obj.const_trial_poissLL = nan(obj.dim_M,1);
            for mm = 1:obj.dim_M
                tt = (obj.obsRange_trial(mm)+1):obj.obsRange_trial(mm+1);
                obj.const_trial_poissLL(mm) = - sum(gammaln(double(Y(tt))+1));
            end
            obj.const_poissLL =  sum(obj.const_trial_poissLL);
            
            %sets the hyperparameter vars
            obj.function_prior = function_prior;
            obj.dim_H = uint32(dim_H);
            
            %null values
            obj.gpuSinglePrecision = gpuSinglePrecision;
            obj.gpuObj_ptr = uint64(0);
            obj.gpus = [];
        
            obj.Y_mask_trial = true(obj.dim_M,1);
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% METHODS to get consistent dimension information
        function [dim] = get.dim_P(obj)
            dim = uint32(size(obj.X,2));
        end
        function [dim] = get.dim_M(obj)
            dim = uint32(length(obj.obsRange_trial)-1);
        end
        function [dim] = get.dim_N(obj)
            dim = uint32(length(obj.Y));
        end
        function [XY] = get.XY(obj)
            if(isempty(obj.XY_value))
                obj.XY_value = double(double(obj.X')*double(obj.Y));
            end
            XY = obj.XY_value;
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
                kcGLM_mex_maskTrials(obj);
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
        %           See GLM constructor for a complete description.
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
        function [onGPU] = isOnGPU(obj)
        %% METHOD [valid] = isOnGPU()
        %	Checks all the GLM variables to see if they are currently loaded to one or more GPUs.
        %
        %	returns:
        %       onGPU [scalar] (logical)       
        %           True if the GLM data appears to be completely loaded onto GPU(s).
        %
            onGPU = obj.gpuObj_ptr ~= 0;
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = freeGPU(obj)
        %% METHOD [obj] = freeGPU()
        %	If the GLM is loaded to the GPU, frees all GPU variables.
        %
            if(obj.isOnGPU())
                %call free mex file
                kcGLM_mex_clear(obj);
                
                %make sure pointer set to 0
                obj.gpuObj_ptr = uint64(0);
                obj.gpus = [];
            end
        end
        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [obj] = toGPU(obj,gpusToUse)
        %% METHOD [obj] = toGPU(gpusToUse,isSinglePrecision)
        %	Loads the GLM data onto one or more GPUs. The data is split into one block per GPU, for a total of D
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
                error('GLM is already loaded to GPU!');
            end
            
            if(nargin < 2 && gpuDeviceCount() == 1)
                gpusToUse = 0;
            end
            
            if(nargin < 2 || isempty(gpusToUse))
                error('no GPUs specified!');
            elseif(sum(gpusToUse(:)<0) > 0 || sum(gpusToUse(:) >= gpuDeviceCount()) > 0 || sum(gpusToUse ~= round(gpusToUse)) > 0)
                error('invalid GPUs given!');
            else
                
                gpusToUse = uint32(gpusToUse(1:min(obj.dim_N,numel(gpusToUse)))); %this shouldn't ever matter, but can't use more GPUs than observations
                obj.gpus = gpusToUse(:);
                
                %set which observations go to which gpu
                ngpus = numel(gpusToUse);
                tper = ceil(double(obj.dim_N)/ngpus);
                
                obsRange_blocks = uint32((0:(ngpus-1))*tper);
                
                %call new mex function to construct GPUGLM C++ object with data loaded to GPUs
                ptr = kcGLM_mex_create(obj,obsRange_blocks,gpusToUse);
                obj.gpuObj_ptr = ptr;
            end
        end

        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [lambda] = computeRate_cpu(obj,paramObj)
        %% METHOD [lambda] = computeRate_cpu(paramObj)
        %	Computes the (log) spike rate for each observation for a setting of the model parameters on the CPU.
        %
        %   This function is primarily intended for debugging, and thus does not return any derivatives.
        %
        %   inputs:
        %       paramObj (GLM_params)
        %           The current model parameters. 
        %
        %   returns:
        %       lambda [dim_N x 1] (real)
        %           The log rate for each observation under the GLM.
            
            %if(isa(paramObj,'GLM_params') && paramObj.isValidParamObj(obj))
                
                lambda = obj.X*paramObj.W + log(obj.const_dt);
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
        %       paramObj (GLM_params)
        %           The current model parameters. 
        %
        %   returns:
        %       ll [scalar] (real)
        %           The Poisson log likelihood for the GLM.
        %
        %       lambda [dim_N x 1] (real)
        %           The log rate for each observation under the GLM.
            
            lambda = obj.computeRate_cpu(paramObj);
            
            Y_c = double(obj.Y);%in MATLAB, ints and floating points don't play nicely
            ll_0 = -exp(lambda) + Y_c.*lambda - gammaln(Y_c + 1);

            ll_t = nan(obj.dim_M,1);
            for ii = 1:obj.dim_M
                si = obj.obsRange_trial(ii)+1;
                ei = obj.obsRange_trial(ii+1);

                ll_t(ii) = sum(ll_0(si:ei));
            end
            ll = sum(ll_t .* obj.Y_mask_trial);
        end
    
            
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct] = computeLPost(obj, paramStruct, optsStruct, highPrecisionLikelihood, computeDerivativesForEvidenceOptimization)
        %% METHOD [resultsStruct = computeLPost(paramStruct, resultsStruct)
        %
        %   Computes the log posterior of the GLM for a given set of parameters.
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
            resultsStruct = obj.getEmptyResultsStruct(optsStruct);
        
            obj.computeLL(paramStruct, resultsStruct);
            
            if(nargin > 4 && computeDerivativesForEvidenceOptimization)
                resultsStruct.dprior_sigma_inv = [];
            end
            
            resultsStruct = obj.addLPrior(paramStruct,resultsStruct);
            
            
            if(nargin > 3 && highPrecisionLikelihood)
                resultsStruct.log_like_0 = sum(double(resultsStruct.trialLL));
            else
                resultsStruct.log_like_0 = resultsStruct.log_like;
            end
            if(resultsStruct.log_post > 0)
                warning('likelihood value is greater than 1: numerical errors must have occured!');
            end
            
            resultsStruct.log_post = resultsStruct.log_like_0 + resultsStruct.log_prior;
        end
       
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [] = computeLL(obj,paramStruct,resultsStruct)
        %% METHOD computeLL
        %
        %   Computes the Poisson log likelihood of the GLM for a given set of parameters.
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
            
            kcGLM_mex_computeLL(obj.gpuObj_ptr,obj.gpuSinglePrecision,paramStruct,resultsStruct);
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
                if(~isempty(resultsStruct.dH))
                    resultsStruct.dH(:) = 0;
                end
            end
        end
    
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct] = getEmptyParamStruct(obj)
        %% METHOD [paramStruct] = getEmptyParamStruct()
        %	Builds a blank structure with all the GLM model parameters/hyperparams.
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
            dataType = 'double';
            if(obj.gpuSinglePrecision)
                dataType = 'single';
            end
            paramStruct.W = zeros(obj.dim_P,1,dataType);
            paramStruct.H = zeros(obj.dim_H,1);
        end
        
        %% METHOD [paramStruct] = getRandomParamStruct(init_gain)
        %	Builds a random structure with all the GLM model parameters/hyperparams.
        %
        function [paramStruct] = getRandomParamStruct(obj,init_gain)
            if(nargin < 2)
                init_gain = 1;
            end
            
            paramStruct = obj.getEmptyParamStruct();
    
            %W normal with std init_gain/sqrt(dim_P);
            
            paramStruct.W(:) = randn(obj.dim_P,1)./sqrt(double(obj.dim_P))*init_gain;
    
            %H are i.i.d. standard normal
            if(~isempty(paramStruct.H))
                paramStruct.H(:) = randn(obj.dim_H,1);
            end
        end
  
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [optsStruct] = getEmptyOptsStruct(~,defaultValue,defaultValueHessians)
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
            
            optsStruct.compute_dH = defaultValue;
        end
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [resultsStruct] = getEmptyResultsStruct(obj, optsStruct)
        %% METHOD [resultsStruct] = getEmptyResultsStruct()
        %	Builds a blank structure with all the GLM model likelihood outputs.
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
        %               dW       [P x 1] (real): 1st deriv of params
        %               dH       [H x 1] (real): 1st deriv of hyperparams 
        %               d2W      [P x P] (real): 2nd deriv of params 
        %
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
                    resultsStruct.d2W = zeros(obj.dim_P,obj.dim_P,dataType);
                else
                    resultsStruct.d2W = [];
                end
            else
                resultsStruct.dW  = [];
                resultsStruct.d2W = [];
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
            
            
            resultsStruct.log_like    = zeros(1,1,dataType);
            resultsStruct.log_prior   = zeros(1,1);
            resultsStruct.log_post    = zeros(1,1);

        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct_xv, results_xv] = optimizeEvidence_MAP_xval(obj, trialFoldNumbers, displayProgress)
            if(nargin < 2 || numel(trialFoldNumbers) ~= obj.dim_M)
                error('Cross validation assumes trials are withheld');
            end
            if(nargin < 3 || isempty(displayProgress))
                displayProgress = false;
            end
            
            if(~obj.isOnGPU())
                error('GMLM must be placed on GPU to perform XVal fits');
            end
            
            %% save original mask information
            Y_mask_0 = obj.Y_mask_trial;
            
            %% fit GMLM for each fold
            folds = unique(trialFoldNumbers);
            for ff = 1:numel(folds)
                if(displayProgress)
                    fprintf('Fitting fold %d / %d ...', ff, numel(folds));
                end
                tts = trialFoldNumbers == folds(ff);
                trialMask = true(obj.dim_M, 1);
                trialMask(tts) = false;
                obj = obj.maskTrials(trialMask);
                
                paramStruct_xv(ff) = obj.optimizeEvidence();
                if(displayProgress)
                    fprintf(' done\n');
                end
            end
            
            %% get test and train likelihoods for each trial
            results_xv.trialLL_train = nan(obj.dim_M, numel(folds));
            results_xv.trialLL_test  = nan(obj.dim_M, numel(folds));
            optsStruct = obj.getEmptyOptsStruct(false, false);
            optsStruct.compute_trialLL = true;
            
            obj = obj.maskTrials(true(obj.dim_M,1));
            for ff = 1:numel(folds)
                resultsStruct = obj.computeLPost(paramStruct_xv(ff), optsStruct);
                tts = trialFoldNumbers == folds(ff);
                
                results_xv.trialLL_train(~tts,ff) = resultsStruct.trialLL(~tts);
                results_xv.trialLL_test(  tts,ff) = resultsStruct.trialLL( tts);
            end
            
            %% reset mask
            if(~all(Y_mask_0))
                obj.maskTrials(Y_mask_0);
            end
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [paramStruct_xv, results_xv] = get_MLE_xval(obj, trialFoldNumbers, displayProgress)
            if(nargin < 2 || numel(trialFoldNumbers) ~= obj.dim_M)
                error('Cross validation assumes trials are withheld');
            end
            if(nargin < 3)
                displayProgress = false;
            end
            
            if(~obj.isOnGPU())
                error('GMLM must be placed on GPU to perform XVal fits');
            end
            
            %% save original mask information
            Y_mask_0 = obj.Y_mask_trial;
            
            %% fit GMLM for each fold
            folds = unique(trialFoldNumbers);
            for ff = 1:numel(folds)
                if(displayProgress)
                    fprintf('Fitting fold %d / %d ...', ff, numel(folds));
                end
                tts = trialFoldNumbers == folds(ff);
                trialMask = true(obj.dim_M, 1);
                trialMask(tts) = false;
                obj = obj.maskTrials(trialMask);
                
                paramStruct_xv(ff) = obj.getMLE();
                if(displayProgress)
                    fprintf(' done\n');
                end
            end
            
            %% get test and train likelihoods for each trial
            results_xv.trialLL_train = nan(obj.dim_M, numel(folds));
            results_xv.trialLL_test  = nan(obj.dim_M, numel(folds));
            optsStruct = obj.getEmptyOptsStruct(false, false);
            optsStruct.compute_trialLL = true;
            
            obj = obj.maskTrials(true(obj.dim_M,1));
            for ff = 1:numel(folds)
                resultsStruct = obj.computeLPost(paramStruct_xv(ff), optsStruct);
                tts = trialFoldNumbers == folds(ff);
                
                results_xv.trialLL_train(~tts,ff) = resultsStruct.trialLL(~tts);
                results_xv.trialLL_test(  tts,ff) = resultsStruct.trialLL( tts);
            end
            
            %% reset mask
            if(~all(Y_mask_0))
                obj.maskTrials(Y_mask_0);
            end
        end

        function [paramStruct,resultsStruct] = optimizeEvidence(obj)
            opts = optimoptions('fminunc','gradobj','on','hessian','off','display','off');
            
            H_init = randn(obj.dim_H,1)*0.1;

            levidFunc = @(hh)obj.negativeLogEvidence(hh);
            H_mle = fminunc(levidFunc,H_init,opts);

            [~,~,paramStruct,resultsStruct] = levidFunc(H_mle);
                
        end
        
        function [w_map] = getMAP(obj, paramStruct)
            opts = optimoptions('fminunc', 'gradobj', 'on', 'hessian', 'on', 'display', 'off', 'algorithm', 'trust-region');%'quasi-newton');
            w_init = double(paramStruct.W(:));
            
            lpostFunc = @(ww)obj.computeNLPost_params(ww, paramStruct, false);
            w_map = fminunc(lpostFunc, w_init, opts);
        end
        
        function [nle, ndle, paramStruct, resultsStruct, ld, d_ld, sigma] = negativeLogEvidence(obj, H, W)
            paramStruct = obj.getEmptyParamStruct();
            paramStruct.H(:) = H;
            if(nargin > 2)
                paramStruct.W(:) = W;
            end
            paramStruct.W(:) = obj.getMAP(paramStruct);
            w_map = paramStruct.W(:);
            
            if(nargout > 1)
                [nlp, ~, sigma, dH_npost, resultsStruct] = obj.computeNLPost_params(w_map, paramStruct, true);
                d_ld = zeros(obj.dim_H,1);
                if(all(~isinf(sigma), 'all') && all(~isnan(sigma),'all'))
                    for ii = 1:obj.dim_H
                        rc = rcond(sigma);
                        if(~isnan(rc) && ~isinf(rc) && rc > 1e-16)
                            d_ld(ii) = 1/2*trace(sigma\resultsStruct.dprior_sigma_inv(:,:,ii));
                        else
                            d_ld(ii) = 1/2*trace(pinv(sigma)*resultsStruct.dprior_sigma_inv(:,:,ii));
                        end
                        if(isinf(d_ld(ii)) || isnan(d_ld(ii)))
                            if(isinf(d_ld(ii)))
                                fprintf('d_ld(%d) is inf\n', ii);
                            else
                                fprintf('d_ld(%d) is nan\n', ii);
                            end
                            fprintf(' sigma inf %d, sigma nan %d, dprior nan %d, dprior inf %d, rc = %f\n', sum(isinf(sigma(:))), sum(isnan(sigma(:))), sum(sum(isnan(resultsStruct.dprior_sigma_inv(:,:,ii)))),sum(sum(isinf(resultsStruct.dprior_sigma_inv(:,:,ii)))), rcond(sigma));
                        end
                    end
                else
                    fprintf('sigma for evidence optimization invalid (contains inf/nan)\n');
                    d_ld(:) = nan;
                end
                ndle = dH_npost + d_ld;
                
                bc = ~isnan(ndle) | ~isinf(ndle);
                if(~all(bc))
                    fprintf('dH_post contains inf/nan\n');
                    bc = find(~bc);
                    for ii = 1:numel(bc)
                        fprintf('bad entries: %d, H = %.4f\n', ii, paramStruct.H(ii));
                    end
                    %values to steer the optimizer away without crashing
                    ndle(:) = 0;
                    nlp = inf;
                end
            else
                [nlp,~,sigma] = obj.computeNLPost_params(w_map, paramStruct, true);
            end
            
            if(all(~isinf(sigma), 'all') && all(~isnan(sigma),'all'))
                ld = 1/2*logdet(sigma);
                nle = nlp + ld;
            else
                fprintf('sigma for evidence optimization invalid (contains inf/nan)\n');
                nle = inf;
            end
            
            if(~all(~isnan(nle)) || ~all(~isinf(nle)))
                fprintf('nle showing inf/nan\n');
            end
        end
        
        function [nlp, dnlp, d2nlp, dnlpH, resultsStruct] = computeNLPost_params(obj, W, paramStruct, computeDerivativeMatrices)
            paramStruct.W(:) = W;
            
            optsStruct_map                  = obj.getEmptyOptsStruct(false, false);
            optsStruct_map.compute_dW       = nargout > 1;
            optsStruct_map.compute_d2W      = nargout > 2;
            optsStruct_map.compute_dH       = nargout > 3;
            optsStruct_map.compute_trialLL  = true;
            resultsStruct = obj.computeLPost(paramStruct, optsStruct_map, true, computeDerivativeMatrices & nargout > 4);
            
            nlp = -double(resultsStruct.log_post);
            if(~all(~isnan(nlp)) || ~all(~isinf(nlp)))
                fprintf('nlp is inf/nan\n');
            end
            if(nargout > 1)
                dnlp = -double(resultsStruct.dW);
                
                if(~all(~isnan(dnlp)) || ~all(~isinf(dnlp)))
                    fprintf('dnlp is inf/nan!\n');
                    for ii = 1:numel(paramStruct.H)
                        fprintf('\tH[%d] = %.4f\n', ii, paramStruct.H(ii));
                    end
                    %values to make optimizer not crash -> but still check out any errors
                    dnlp(:) = 0;
                    nlp = inf;
                end
            end
            if(nargout > 2)
                d2nlp = -double(resultsStruct.d2W);
            end
            if(nargout > 3)
                dnlpH = -double(resultsStruct.dH);
            end
            
        end
        
        function [paramStruct,resultsStruct] = getMLE(obj)
%             opts = optimoptions('fminunc','gradobj','on','hessian','on','display','off','algorithm','trust-region');
            opts = optimoptions('fminunc','gradobj','on','hessian','off','display','off','algorithm','quasi-newton');
            
            paramStruct = obj.getEmptyParamStruct();
            
            nllFunc = @(ww)obj.negativeLogLikelihood(ww,paramStruct);
            W_mle = fminunc(nllFunc,double(paramStruct.W(:)),opts);
            
            if(nargout > 1)
                [~,~,~,paramStruct,resultsStruct] = nllFunc(W_mle);
            else
                paramStruct.W(:) = W_mle;
            end
        end
        
        function [nll,ndll,n2dll,paramStruct,resultsStruct] = negativeLogLikelihood(obj,W,paramStruct)
            optsStruct = obj.getEmptyOptsStruct(false,false);
            optsStruct.compute_trialLL = true;
            optsStruct.compute_dW  = nargout > 1;
            optsStruct.compute_d2W = nargout > 2;
            
            paramStruct.W(:) = W(:);
            
            resultsStruct = obj.getEmptyResultsStruct(optsStruct);
            kcGLM_mex_computeLL(obj.gpuObj_ptr,obj.gpuSinglePrecision,paramStruct,resultsStruct);
            
            nll = -sum(double(resultsStruct.trialLL));
            if(nargout > 1)
                ndll = -double(resultsStruct.dW);
            end
            if(nargout > 2)
                n2dll = -double(resultsStruct.d2W);
            end
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %methods for MCMC fits
        [M_chol] = posteriorLaplaceFits(obj, paramStruct, M_chol, s_idx, computeChol)
        [HMC_settings] = setupHMCparams(obj, nWarmup, nTotal, debugSettings)
        [accepted,err,paramStruct_new,log_p_accept,resultStruct] = hmcStep(obj,paramStruct_0,M_chol,s_idx,HMC_settings,resultStruct_0)
        [samples, summary, HMC_settings, outputMsg, paramStruct] = runHMC(obj, HMC_settings, textOutputFile, msgStr_0)
    end
end