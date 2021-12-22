%% adjustHMCstepSize
%    Updates the HMC step size parameter using dual averaging given a sample
%    inputs:
%       ss  - current sample number
%       HMC_state - current HMC state (constructed by runHMC_simple)
%       setupSizeSettings - settings for algorithm, including update schedule (see setupHMCparams)
%       log_p_accept_new  - the log probability of the MH accept step for sample ss
function [HMC_state] = adjustHMCstepSize(ss, HMC_state, stepSizeSettings, log_p_accept_new)
ww = find(ss >= stepSizeSettings.schedule(:,1) & ss <= stepSizeSettings.schedule(:, 2),1,'first'); %get current estimation block (if exists)

stepSizeState = HMC_state.stepSize;

if(~isempty(ww))
    sample_1 = stepSizeSettings.schedule(ww,1);
    ps = stepSizeSettings;
    ps.e_0 = max(stepSizeSettings.e_0,stepSizeState.e);
    ps.mu = log(10*stepSizeSettings.e_0);
    
    log_h = min(0,log_p_accept_new);
    tt = ss - sample_1 + 1;
    [stepSizeState.x_t,stepSizeState.x_bar_t,stepSizeState.H_sum] = nesterovStepSizeUpdate_internal(ps, log_h, stepSizeState.H_sum, stepSizeState.x_bar_t, tt);
    stepSizeState.e_bar  = exp(stepSizeState.x_bar_t);

    if(ss == stepSizeSettings.schedule(ww,2))
        stepSizeState.e  = exp(stepSizeState.x_bar_t);
    else
        stepSizeState.e  = exp(stepSizeState.x_t);
    end
elseif(ss >= max(stepSizeSettings.schedule,[],'all'))
    stepSizeState.e          = stepSizeState.e_bar;
end

HMC_state.stepSize = stepSizeState;
HMC_state.steps   = min(stepSizeSettings.maxSteps,      ceil(stepSizeSettings.stepL/HMC_state.stepSize.e));

end

%%
function [x_t, x_bar_t, H_sum] = nesterovStepSizeUpdate_internal(ps, log_h, H_sum, x_bar_t, tt)
    if(tt == 1 || isnan(x_bar_t) || isinf(x_bar_t))
        %reset estimation
        x_t     = x_bar_t;
        if(isnan(x_bar_t) || isinf(x_bar_t))
            x_bar_t = log(max(1e-8,double(ps.e_0)));
        end
        H_sum   = 0;
    end
    %update with last step
    H_sum   = H_sum + ps.delta-min(1,exp(double(log_h)));
%         H_sum   = H_sum + ps.delta-exp(double(log_h));
    nu_t    = (tt-1)^(-ps.kappa);
    x_t     = ps.mu - sqrt(tt-1)/ps.gamma *(1/(tt-1+ps.t_0)) * H_sum;
    x_bar_t = nu_t*x_t + (1-nu_t)*x_bar_t;
    
end