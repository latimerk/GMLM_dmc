%% runs an HMC step with a diagonal M matrix (M should be a column)
%
% I had previously programmed up HMC for the GMLM to try geodesic Monte Carlo, but I found no benefit.
% I wound it back to the simple HMC step.
%
% Takes in a negative log posterior function (return [nlpost, dnlpost] given vector of parameters)
%   The negative is so that this function uses the same function as optimizers
function [accepted, divergent, w_new, log_p_accept, results] = HMCstep_diag(w_init, M, nlpostFunction, HMC_state, p_init)
    if(isfield(HMC_state, "e_scale") && ~isempty(HMC_state.e_scale))
        HMC_state.stepSize.e = HMC_state.stepSize.e * HMC_state.e_scale;
    end
    
    %% generate initial momentum
    if(nargin < 5 || isempty(p_init))
        p_init = generateMomentum(M);
    end
    
    %% get initial probability of momentum
    lp_momentum_0 = logProbMomentum(p_init, M);
    
    %% get initial probability of params and derivatives
    [nlpost_0, ndW, ~, results_init] = nlpostFunction(w_init);
    H_0 = -nlpost_0 + lp_momentum_0; 
    
    %% run the HMC
    divergent = false;
    w = w_init;
    p = p_init;
    
    if(isnan(nlpost_0) || isinf(nlpost_0))
        error('HMC initial state shows nan/inf!');
    end
    
    try
        for tt = 1:HMC_state.steps
            %% move momentums
            [p, errs] = momentumStep(p, -ndW, HMC_state);
            if(errs)% divergent trajectory
                nlpost = inf; % divergent trajectory
                break;
            end

            %% move positions
            [w, errs] = paramStep(w, p, M, HMC_state);
            if(errs)
                nlpost = inf; % divergent trajectory
                break;
            end
            
            [nlpost, ndW, ~, results] = nlpostFunction(w);
            
            %% move momentums
            [p, errs] = momentumStep(p, -ndW, HMC_state);
            if(errs)% divergent trajectory
                nlpost = inf; % divergent trajectory
                break;
            end
            
            %% check current divergence
            lp_momentum = logProbMomentum(p, M);
            H_s = -nlpost + lp_momentum; 

            divergence = H_s - H_0;
            if(isnan(divergence) || isinf(divergence))
                nlpost = inf; % divergent trajectory
                break;
            end
        end
        
        %% get final state log prob
        log_p_accept = min(0, divergence);
        
        if(isnan(log_p_accept) || isinf(log_p_accept) || isnan(nlpost) || isinf(nlpost) || isnan(lp_momentum) || isinf(lp_momentum))
            error('HMC accept probability is nan!');
        end
    catch ee %#ok<NASGU>
        log_p_accept = nan;
        w_new        = w_init;
        results      = results_init;
        accepted     = false;
        divergent    = true;
        
        %msgText = getReport(ee,'extended');
        %fprintf('HMC reaching inf/nan values with step size %.4f: %s\n\tAuto-rejecting sample and setting p_accept = %e.\n\tError Message: %s\n',ees,errorMessageStr,p_accept,msgText);
        %fprintf('>>end error message<<\n');
        %fprintf('\t\t>>>HMC sampler reaching numerically unstable values (infinite/nan): rejecting sample early<<<\n');
        return;
    end
    
    %% check for acceptance
    u = log(rand);
    
    if(u < log_p_accept)
        w_new = w;
        accepted = true;
    else
        w_new = w_init;
        accepted = false;
        results = results_init;
    end

    if(abs(divergence) > 1e3)
        divergent = true;
    end
end
 
%% gets initial momentum
function [vv] = generateMomentum(M)
    vv = (randn(numel(M),1).*sqrt(M));
end

%% gets the probability of a momentum term
function [lp] = logProbMomentum(mm,M) 
    lp = -1/2*sum(M.\mm.^2);
end

%% complete parameter step
function [w,errs] = paramStep(w, p, M, HMC_state)
    w(:) = w + HMC_state.stepSize.e*(M.\p(:));
    if(~all(~isnan(w) & ~isinf(w)))
        errs = true;
    else
        errs = false;
    end
end

%% momentum step for all parameters
function [p,errs] = momentumStep(p, dW, HMC_state)
    p(:) = p + HMC_state.stepSize.e/2*double(dW(:));
    if(~all(~isnan(p) & ~isinf(p)))
        errs = true;
    else
        errs = false;
    end
end