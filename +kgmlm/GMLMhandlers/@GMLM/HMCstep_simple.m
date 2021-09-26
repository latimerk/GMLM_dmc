function [accepted,err,paramStruct_new,log_p_accept,resultStruct] = HMCstep_simple(obj, paramStruct_0, M_chol, HMC_state, resultStruct_0)
    optsStruct   = obj.getEmptyOptsStruct(true,false);
    optsStruct = obj.getParamCount(optsStruct);
    
    [paramStruct_0.W_all, paramStruct_0.H_all] = obj.vectorizeParamStruct(paramStruct_0, optsStruct);
    
    %% generate initial momentum
    momentumStruct_0   = struct();
    momentumStruct_0.W_all = generateMomentum(numel(paramStruct_0.W_all), M_chol.W);
    
    if(~isempty(paramStruct_0.H_all))
        momentumStruct_0.H_all = generateMomentum(numel(paramStruct_0.H_all), M_chol.H);
    end

    momentumStruct = momentumStruct_0;
    
    %% get initial probability of momentum
    lp_momentum_0 = logProbMomentum(momentumStruct_0, paramStruct_0, M_chol);
    
    %% get initial probability of params and derivatives
    if(nargin < 6)
        optsStruct.compute_trialLL = true;
        resultStruct_0 = obj.computeLPost(paramStruct_0, optsStruct, true);
    end
    optsStruct.compute_trialLL = false;
    resultStruct   = resultStruct_0;
    [resultStruct.dW_all, resultStruct.dH_all] = obj.vectorizeResultsStruct(resultStruct, optsStruct);
    H_0 = double(resultStruct_0.log_post) + lp_momentum_0; 
    
    %% run the HMC
    err = false;
    paramStruct = paramStruct_0;
    try
        for tt = 1:HMC_state.steps
            %% move momentums
            [momentumStruct,errs] = momentumStep(paramStruct, momentumStruct, resultStruct, HMC_state);
            if(errs)
                resultStruct.log_post = -inf;
                break;
            end

            %% move positions
            [paramStruct, momentumStruct, errs] = paramStep(paramStruct, momentumStruct, M_chol, HMC_state, obj.gpuSinglePrecision);
            paramStruct = obj.devectorizeParamStruct(paramStruct.W_all, paramStruct, optsStruct, paramStruct.H_all);
            if(errs)
                resultStruct.log_post = -inf;
                break;
            end
            
            optsStruct.compute_trialLL = tt == HMC_state.steps;
            resultStruct = obj.computeLPost(paramStruct,optsStruct,optsStruct.compute_trialLL);
            [resultStruct.dW_all, resultStruct.dH_all] = obj.vectorizeResultsStruct(resultStruct, optsStruct);
            if(isinf(resultStruct.log_post) || isnan(resultStruct.log_post))
                resultStruct.log_post = -inf;
                break;
            end

            %% move momentums
            [momentumStruct, errs] = momentumStep(paramStruct, momentumStruct, resultStruct, HMC_state);
            if(errs)
                resultStruct.log_post = -inf;
                break;
            end
        end
        
        %% get final state log prob
        lp_momentum = logProbMomentum(momentumStruct,paramStruct,M_chol);
        H_s = double(resultStruct.log_post) + lp_momentum; 
        
        log_p_accept = H_s - H_0;
        if(isnan(log_p_accept) || isinf(log_p_accept))
            error('HMC accept probability is nan!');
        end
    catch ee %#ok<NASGU>
        p_accept = 1e-4;
        err = true;
        log_p_accept = log(p_accept);
        paramStruct_new = paramStruct_0;
        accepted        = false;
        resultStruct    = resultStruct_0;
        
%         msgText = getReport(ee,'extended');
%         fprintf('HMC reaching inf/nan values with step size %.4f: %s\n\tAuto-rejecting sample and setting p_accept = %e.\n\tError Message: %s\n',ees,errorMessageStr,p_accept,msgText);
%         fprintf('>>end error message<<\n');
        fprintf('\t\t>>>HMC sampler reaching numerically unstable values (infinite/nan): rejecting sample early<<<\n');
        
        return;
    end
    
    
    %% check for acceptance
    u = log(rand);
    
    if(u < log_p_accept)
        paramStruct_new = paramStruct;
        
        accepted = true;
    else
        paramStruct_new = paramStruct_0;
        resultStruct    = resultStruct_0;
        
        accepted = false;
    end
    
end
 

function [vv] = generateMomentum(n,M_chol)
    if(iscolumn(M_chol))
        vv = (randn(n,1).*sqrt(M_chol));
    else
        vv = (randn(1,n)*M_chol)';
    end
end
%% gets the probability of a momentum term
function [lp] = logProbMomentum(momentumStruct, paramStruct, M_chol)
    lp = 0;
    
    lp = lp + logProbMomentum_standard(momentumStruct.W_all, M_chol.W);
    if(~isempty(paramStruct.H_all))
        lp = lp + logProbMomentum_standard(momentumStruct.H_all, M_chol.H);
    end
    
end

function [lp] = logProbMomentum_standard(mm,M_chol) 
    if(iscolumn(M_chol))
        lp = -1/2*sum(M_chol.\mm.^2);
    else
        opts.UT = true;
        opts.TRANSA = true;
        mc = linsolve(M_chol, mm(:), opts);
        lp = -1/2*(mc'*mc);
    end
end


%% complete parameter step
function [paramStruct,momentumStruct,errs] = paramStep(paramStruct, momentumStruct, M_chol, HMC_state, useSinglePrecsion)
    errs = false;

    % standard step for W: using M_chol.W
    
    p_new = standardStep(paramStruct.W_all,momentumStruct.W_all, HMC_state.stepSize.e, M_chol.W);
    if(~all(~isnan(p_new) & ~isinf(p_new)))
        errs = true;
        return;
    end
    if(useSinglePrecsion)
        p_new = single(p_new);
    else
        p_new = double(p_new);
    end
    paramStruct.W_all(:) = p_new;
    
    %standard step for H: using M_chol.H 
    if(~isempty(paramStruct.H_all))
        p_new = standardStep(paramStruct.H_all,momentumStruct.H_all,HMC_state.stepSize.e,M_chol.H);
        if(~all(~isnan(p_new) & ~isinf(p_new)))
            errs = true;
            return;
        end
        paramStruct.H_all = p_new;
    end
end


%% momentum step for all parameters
function [momentumStruct,errs] = momentumStep(paramStruct, momentumStruct, resultStruct, HMC_state)
    errs = false;

    
    
    % standard step for W
    m_new = standardMomentumStep(momentumStruct.W_all, resultStruct.dW_all, HMC_state.stepSize.e);
    if(~all(~isnan(m_new) & ~isinf(m_new)))
        errs = true;
        return;
    end
    momentumStruct.W_all  = m_new;

    % standard step for H
    if(~isempty(paramStruct.H_all))
        m_new = standardMomentumStep(momentumStruct.H_all, resultStruct.dH_all, HMC_state.stepSize.e);
        if(~all(~isnan(m_new) & ~isinf(m_new)))
            errs = true;
            return;
        end
        momentumStruct.H_all  = m_new;
    end
end

%% Standard HMC leapfrog functions===============================================================================================
function [vv] = standardMomentumStep(vv,dl,e)
    vv = vv + e/2*reshape(double(dl),[],1);
end
%%
function [xx] = standardStep(xx,vv,e,M_chol)
    if(iscolumn(M_chol))
        xx = xx + e*(M_chol.\vv(:));
    else
        opts.UT     = true;
        opts.TRANSA = true;
        opts2.UT    = true;
        xx = xx + e*reshape(linsolve(M_chol,linsolve(M_chol,vv(:),opts),opts2),size(xx));
    end
end