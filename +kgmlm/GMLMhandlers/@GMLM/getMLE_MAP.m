%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [paramStruct_opt,resultsStruct_opt,log_post,STATUS,compute_MAP] = getMLE_MAP(obj, paramStruct_init, max_iters, convergence_delta, jitterAttempts, msgStr, optAllParamsTogether, maxIters_BFGS)
    STATUS = 0;

    if(nargin < 2 || isempty(paramStruct_init) || ~isstruct(paramStruct_init))
        compute_MAP = false;
    elseif(isstruct(paramStruct_init) && isfield(paramStruct_init,'Groups') && numel(paramStruct_init.Groups) == obj.dim_J && isfield(paramStruct_init,'H'))
        totalH = numel(paramStruct_init.H);
        totalH_gmlm = obj.dim_H;
        for jj = 1:obj.dim_J
            if(isfield(paramStruct_init.Groups(jj), 'H'))
                totalH     = totalH + numel(paramStruct_init.Groups(jj).H);
            else
                totalH = nan;
                break;
            end
            totalH_gmlm = totalH_gmlm + obj.Groups(jj).dim_H;
        end

        if(totalH ~= totalH_gmlm)
            compute_MAP = false;
        else
            compute_MAP = true;
        end

    else
        compute_MAP = false;
    end

    if(nargin < 3 || isempty(max_iters))
        max_iters = 500;
    end
    if(nargin < 4 || isempty(convergence_delta))
        convergence_delta = 1e-6;
    end
    if(nargin < 5 || isempty(jitterAttempts))
        jitterAttempts = 0;
    end
    if(nargin < 6 || isempty(msgStr))
        msgStr = '';
    end
    if(nargin < 7 || isempty(optAllParamsTogether))
        optAllParamsTogether = true;
    end
    if(nargin < 7 || isempty(maxIters_BFGS))
        maxIters_BFGS = 10e3;
    end

    start_time = tic;
    log_post = nan(max_iters+1,1);

    if(nargin > 1 && ~isempty(paramStruct_init) && isstruct(paramStruct_init) && isfield(paramStruct_init,'W') && ~isempty(paramStruct_init.W))
        paramStruct = paramStruct_init;
    else
        paramStruct = obj.getRandomParamStruct([],compute_MAP);
    end
    paramStruct_opt = paramStruct;
    resultsStruct_opt = [];

    jitterCtr = 0;

    max_dim_S = 1;
    for jj = 1:obj.dim_J
        max_dim_S = max(max_dim_S,double(obj.Groups(jj).dim_S));
    end

    optsStruct_empty = obj.getEmptyOptsStruct(false,false);
    optsStruct_empty.compute_trialLL = true;


    ctr = 1;
    optsStruct_paramSets  = obj.getEmptyOptsStruct(false,false);
    optsStruct_paramSets = obj.getParamCount(optsStruct_paramSets);


    %optimize all the parameters together
    optsStruct_all  = obj.getEmptyOptsStruct(true, false);
    if(optAllParamsTogether)
        optsStruct_paramSets(ctr) = obj.getParamCount(optsStruct_all); 
        ctr = ctr+1; 
    end

    %optimize loading weights (W+B+V): this is separate because we have convient Hessians for this
    optsStruct_individualNeuronLoadings_trustRegion = obj.getEmptyOptsStruct(false,false);
    optsStruct_individualNeuronLoadings_trustRegion.compute_dB = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_dW = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2W  = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2B  = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2WB = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2WV(:) = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2BV(:) = true;
    optsStruct_individualNeuronLoadings_trustRegion.compute_d2VV(:) = true;
    for jj = 1:obj.dim_J
        optsStruct_individualNeuronLoadings_trustRegion.Groups(jj).compute_dV = true;
        optsStruct_individualNeuronLoadings_trustRegion.Groups(jj).compute_d2V = true;
    end
    optsStruct_individualNeuronLoadings_trustRegion = obj.getParamCount(optsStruct_individualNeuronLoadings_trustRegion); 

%             optsStruct_individualNeuronLoadings_trustRegion = [];

    %optimize each order + W
    for ss = 1:max_dim_S
        optsStruct_tw_c  = obj.getEmptyOptsStruct(false,false);
        optsStruct_tw_c.compute_dW = true;
        optsStruct_tw_c.compute_dB = false;
        for jj = 1:obj.dim_J
            if(obj.Groups(jj).dim_S >= ss)
                optsStruct_tw_c.Groups(jj).compute_dT(ss) = true;
            else
                optsStruct_tw_c.Groups(jj).compute_dT(1) = false;
                optsStruct_tw_c.Groups(jj).compute_dV    = false;
            end
        end
        optsStruct_paramSets(ctr) = obj.getParamCount(optsStruct_tw_c); 
        ctr = ctr+1;
    end

    all_prev = obj.vectorizeParamStruct(paramStruct,optsStruct_all);

    %setup opimization parameters
    maxIter_trustRegion = 100;
    MaxFunctionEvaluations = maxIters_BFGS*numel(all_prev);
    OptimalityTolerance = min(convergence_delta, 1e-6);
    StepTolerance       = min(convergence_delta, 1e-6);
    for ss = 1:numel(optsStruct_paramSets)
        fminuncOpts_paramSets(ss)  = optimoptions('fminunc', 'gradobj', 'on' , 'hessian', 'off','display','off','algorithm','quasi-newton','MaxIterations',maxIters_BFGS,...
            'MaxFunctionEvaluations', MaxFunctionEvaluations,'OptimalityTolerance',OptimalityTolerance,'StepTolerance',StepTolerance,'HessUpdate', 'bfgs'); %#ok<AGROW> %quasi-newton
    end
    %fminuncOpts_paramSets(1).MaxIterations = maxIters_BFGS;

    fminuncOpts_individualNeuronLoadings_trustRegion = optimoptions('fminunc', 'gradobj', 'on', 'hessian','on','display','off','algorithm','trust-region','MaxIterations',maxIter_trustRegion,...
        'MaxFunctionEvaluations', MaxFunctionEvaluations, 'OptimalityTolerance',OptimalityTolerance,'StepTolerance',StepTolerance); %,'MaxPCGIter',100
    init = false;


    for ii = 1:max_iters


        timings = nan(numel(optsStruct_paramSets) + 1, 1);

        %% maximize each set of Ts, W
        optOrder = 1:numel(optsStruct_paramSets); %randperm(numel(optsStruct_conditionalGLM)); 
        for ss = optOrder


            %%
            ss_t = tic;
            tw_0 = obj.vectorizeParamStruct(paramStruct,optsStruct_paramSets(ss));
            if(isempty(tw_0))
                fprintf('\t Warning: GMLM optimization errors during conditional optimization! No parameters in slice %d\n', ss);
                continue;
            end
            nlpost_tw = @(vv)obj.vectorizedNLL_func(vv, paramStruct, optsStruct_paramSets(ss), compute_MAP);

            if(~init)
                init = true;
                [~,~,paramStruct,resultsStruct] = nlpost_tw(tw_0);
                fprintf('Starting GMLM optimization: fval init = %e\t\t%s\n',resultsStruct.log_post,msgStr);
                log_post(1) = resultsStruct.log_post;
            end

            try

                TW_map = fminunc(nlpost_tw,tw_0,fminuncOpts_paramSets(ss));
                [~,~,paramStruct_0,resultsStruct_0] = nlpost_tw(TW_map);

                if(resultsStruct_0.log_post < resultsStruct.log_post - 1)
                    fprintf('\t Warning: GMLM optimization errors during conditional optimization [%d]! New result worse than last iteration!\n\t\tnew log post/like: %e\n\t\told log post/like: %e\n', ss, resultsStruct_0.log_post, resultsStruct.log_post);
                elseif(resultsStruct_0.log_post < resultsStruct.log_post)
                    %fprintf('\t\t Unable to increase likelihood/posterior in GMLM optimization during conditional optimization [%d]!\n', ss);
                else
                    paramStruct   = paramStruct_0;
                    resultsStruct = resultsStruct_0;
                end
                %% if doing MLE, renormalize all Ts that were just fit
                if(~compute_MAP)
                    paramStruct_normalized = obj.normalizeParams(paramStruct);

                    rs = obj.computeLL(paramStruct_normalized,optsStruct_empty,true);
                    if((rs.log_like_0 - resultsStruct.log_like_0) < -1e-1)
                        warning('log likelihood changed after renormalization! (optimization %d)',ss);
                    else
                        paramStruct = paramStruct_normalized;
                    end
                end

            catch %ee
                warning('Minimization failed for optimization %d. Continuing with other search directions...', ss);
                TW_map = tw_0;
            end
            timings(ss) = toc(ss_t);
        end

        %% maximize individual neuron loadings (W, B, V)
        ss_t = tic;
        if(~isempty(optsStruct_individualNeuronLoadings_trustRegion))
            WBV_0 = obj.vectorizeNeuronLoadingWeights(paramStruct, optsStruct_individualNeuronLoadings_trustRegion);
            WBV_map = WBV_0;
            %optimizes each neuron's weight independently: optimizer often likes this much more despite the known Hessian
            for pp = 1:double(obj.dim_P)
                wbv_0 = obj.vectorizeNeuronLoadingWeights(paramStruct, optsStruct_individualNeuronLoadings_trustRegion,pp);
                nlpost_wbv_c = @(wbv)obj.vectorizedNLL_neuronLoadingWeights_func(wbv, paramStruct, optsStruct_individualNeuronLoadings_trustRegion, compute_MAP,pp);
                WBV_map_c = fminunc(nlpost_wbv_c, wbv_0, fminuncOpts_individualNeuronLoadings_trustRegion);

                K = numel(WBV_map_c);
                WBV_map((1:K) + (pp-1)*K) = WBV_map_c;
            end

            nlpost_wbv = @(wbv)obj.vectorizedNLL_neuronLoadingWeights_func(wbv, paramStruct, optsStruct_individualNeuronLoadings_trustRegion, compute_MAP);

            if(~init)
                init = true;
                [~,~,~,paramStruct,resultsStruct] = nlpost_wbv(WBV_0);
                fprintf('Starting GMLM optimization: fval init = %e\t\t%s\n',resultsStruct.log_post,msgStr);
                log_post(1) = resultsStruct.log_post;
            end

            [~,~,~,paramStruct_0,resultsStruct_0] = nlpost_wbv(WBV_map);
            if(resultsStruct_0.log_post < resultsStruct.log_post - 1)
                fprintf('\t Warning: GMLM optimization errors during conditional GLM conditional loading weights optimization! New result worse than last iteration!\n\t\tnew log post/like: %e\n\t\told log post/like: %e\n', resultsStruct_0.log_post, resultsStruct.log_post);
            elseif(resultsStruct_0.log_post < resultsStruct.log_post)
                %fprintf('\t\t Unable to increase likelihood/posterior in GMLM optimization during conditional neuron loading weights optimization!\n');
            else
                paramStruct   = paramStruct_0;
                resultsStruct = resultsStruct_0;
            end
        end
        timings(end) = toc(ss_t);

        %% check for termination condition
        log_post(ii+1) = resultsStruct.log_post;
        [opt_lpost,opt_idx] = nanmax(log_post);
        if(opt_idx == ii+1)
            paramStruct_opt = paramStruct;
            resultsStruct_opt = resultsStruct;
        end

        all_0 = obj.vectorizeParamStruct(paramStruct,optsStruct_all);
        stepNorm = norm(all_0 - all_prev);
        all_prev = all_0;

        run_time = toc(start_time);
        delta_ll = log_post(ii+1) - log_post(ii) ;

        if(jitterAttempts > 0)
            fprintf('iter %d / %d: fval = %e (change = %e, step size = %e)\tbest fit: %e\t%s\t\telapsed time: %.1f\n', ii, max_iters, resultsStruct.log_post, delta_ll, stepNorm, opt_lpost, msgStr, run_time);
        else
            fprintf('iter %d / %d: fval = %e (change = %e, step size = %e)\t%s\t\telapsed time: %.1f\n', ii, max_iters, resultsStruct.log_post, delta_ll, stepNorm, msgStr, run_time);
        end

        if(ii >= 2 && delta_ll < convergence_delta && delta_ll >= 0 && jitterCtr < jitterAttempts)
            all_0 = obj.vectorizeParamStruct(paramStruct,optsStruct_all);
            ktheta  = 1;
            ktheta2 =  0.01^2;

            theta = ktheta2/ktheta;
            k     = ktheta/theta;
            all_0 = all_0.*gamrnd(k,theta,size(all_0)) +  randn(size(all_0))*1e-2;
            nlpost_all = @(vv)obj.vectorizedNLL_func(vv, paramStruct, optsStruct_all, compute_MAP);

            [~,~,paramStruct,resultsStruct] = nlpost_all(all_0);
            jitterCtr = jitterCtr + 1;
            fprintf('\tJittering current esimate to see if optimizer is just a little stuck (jitter attempt %d / %d)\n', jitterCtr, jitterAttempts);
        elseif(ii >= 2 && delta_ll < convergence_delta && delta_ll >= 0)
            end_time = toc(start_time);
            fprintf('  difference in log post/likelihood is below tolerance: possible maximum found! Total optimization time = %.1f\n',end_time);
            break;
        %elseif(ii >= 3 && log_post(ii) - log_post(ii-1) > 1)
        %    randomizedCtr = 0;
        elseif(ii == max_iters)
            end_time = toc(start_time);
            fprintf('  maximum iterations reached: no maximum found! Total optimization time = %.1f\n',end_time);
        end
    end

    if(isfield(paramStruct_opt, 'W_all'))
        paramStruct_opt = rmfield(paramStruct_opt, 'W_all');
    end
    if(isfield(paramStruct_opt, 'H_all'))
        paramStruct_opt = rmfield(paramStruct_opt, 'H_all');
    end
end