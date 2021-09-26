% simple i.i.d. standard normal prior (NO hyperparams) over all the elements of V and T in one ternsor group

function [results_group] = gmlmPrior_normalVT(results_group, params_group)

results_group.log_prior_VT = -1/2*sum(params_group.V.^2, 'all');
if(~isempty(results_group.dV)) %if is empty, not required to compute derivative, otherwise ADD derivative of log prior to struct
    results_group.dV = results_group.dV - params_group.V;
end


for ss = 1:numel(params_group.T)
    results_group.log_prior_VT = results_group.log_prior_VT - 1/2*sum(params_group.T{ss}.^2, 'all');

    if(~isempty(results_group.dT{ss}))
        results_group.dT{ss} = results_group.dT{ss} - params_group.T{ss};
    end
end