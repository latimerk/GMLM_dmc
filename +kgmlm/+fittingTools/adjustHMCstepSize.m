%% adjustHMCstepSize
%    Updates the HMC step size parameter using dual averaging given a sample
%    inputs:
%       ss  - current sample number
%       HMC_state - current HMC state (constructed by runHMC_simple)
%       setupSizeSettings - settings for algorithm, including update schedule (see setupHMCparams)
%       log_p_accept_new  - the log probability of the MH accept step for sample ss
function [HMC_state] = adjustHMCstepSize(ss, HMC_state, stepSizeSettings, log_p_accept_new)
if(numel(stepSizeSettings.schedule) >= 2)
    ww = find(ss >= stepSizeSettings.schedule(:,1) & ss <= stepSizeSettings.schedule(:, 2),1,'first'); %get current estimation block (if exists)
else
    ww = [];
end
stepSizeState = HMC_state.stepSize;

if(~isempty(ww))
    sample_1 = stepSizeSettings.schedule(ww,1);
    ps = stepSizeSettings;
    ps.e_0 = max(stepSizeSettings.e_0,stepSizeState.e);
    ps.mu = log(10*stepSizeSettings.e_0);

    ps.delta = ps.delta(min(numel(ps.delta), ww));
    
    tt = ss - sample_1 + 1;
    [stepSizeState.x_t, stepSizeState.x_bar_t, stepSizeState.H_sum] = dualAverageStepSizeUpdate_internal(ps, log_p_accept_new, stepSizeState.H_sum, stepSizeState.x_bar_t, tt, log(stepSizeSettings.max_step_size(min(ww, numel(stepSizeSettings.max_step_size)))));
    stepSizeState.e_bar  = exp(stepSizeState.x_bar_t);

    if(ss == stepSizeSettings.schedule(ww,2))
        stepSizeState.e  = exp(stepSizeState.x_bar_t);
    else
        stepSizeState.e  = exp(stepSizeState.x_t);
    end
elseif(ss >= max(stepSizeSettings.schedule,[],'all'))
    stepSizeState.e          = stepSizeState.e_bar;
end


% stepSizeState.e     = min(stepSizeSettings.max_step_size, stepSizeState.e );
% stepSizeState.e_bar = min(stepSizeSettings.max_step_size, stepSizeState.e_bar );

HMC_state.stepSize = stepSizeState;

ll = [];
if(isfield(stepSizeSettings, "maxSteps_trial"))
    ll = find(ss <= stepSizeSettings.maxSteps_trial, 1, "last");
end
if(isempty(ll))
    ll = numel(stepSizeSettings.maxSteps);
end

HMC_state.steps   = min(stepSizeSettings.maxSteps(ll),      ceil(stepSizeSettings.stepL/HMC_state.stepSize.e));

end

%%
function [x_t, x_bar_t, H_sum] = dualAverageStepSizeUpdate_internal(ps, log_h, H_sum, x_bar_t, tt, max_x)
    if(tt == 1 || isnan(x_bar_t) || isinf(x_bar_t))
        %reset estimation
        x_t     = x_bar_t;
        if(isnan(x_bar_t) || isinf(x_bar_t))
            x_bar_t = log(max(1e-8,double(ps.e_0)));
        end
        H_sum   = 0;
    end

    if(~isnan(log_h))
        a_tt = min(1,exp(double(log_h)));
    else
        % penalty for divergent transitions?
        a_tt = 0;
    end

    %update with last step (algorithm 5 in NUTS paper)
    aa_H = 1/(tt + ps.t_0);
    H_tt = ps.delta - a_tt;
    H_sum = aa_H * H_tt + (1 - aa_H) * H_sum;
    x_t   = ps.mu  - sqrt(tt)/ps.gamma * H_sum;
    if(nargin >= 6)
        x_t   = min(max_x, x_t);
    end
        
    aa_x =tt^(-ps.kappa);
    x_bar_t = aa_x * x_t + (1 - aa_x)*x_bar_t;
end