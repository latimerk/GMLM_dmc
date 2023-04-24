function [results] = scaleDWB3(results, params, couplingGroup, precomputedPortion)

% i.i.d non-mean Gaussian over spike history coefficients & constant


K = size(params.B,1);


if(~isempty(results.dW) && isfield(results, "dH") && ~isempty(results.dH))
    w_mu_idx  = 1;
    dW = double(results.dW);
    results.dH(w_mu_idx)  = results.dH(w_mu_idx)  + sum(dW);
end

%% linear terms
if(K > 0 && ~isempty(results.dB))

    dB = double(results.dB);
    if(isfield(results, "dH") && ~isempty(results.dH))
        b_mu_idx = 2 + (1:K);
        results.dH(b_mu_idx)  = results.dH(b_mu_idx)  + sum(dB,2);
    end
    if(~isempty(couplingGroup))
        if(~isempty(results.Groups(couplingGroup).dT{1}))
            results.Groups(couplingGroup).dT{1} = results.Groups(couplingGroup).dT{1} - dB*(precomputedPortion.U.*precomputedPortion.V);
        end
        if(~isempty(results.Groups(couplingGroup).dT{2}))
            results.Groups(couplingGroup).dT{2} = results.Groups(couplingGroup).dT{2} - (dB'*precomputedPortion.T).*precomputedPortion.V;
        end
        if(~isempty(results.Groups(couplingGroup).dV))
            results.Groups(couplingGroup).dV(:,:) = results.Groups(couplingGroup).dV - (dB'*precomputedPortion.T).*precomputedPortion.U;
        end
    end
end
