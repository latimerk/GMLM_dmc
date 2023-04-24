function [results] = scaleDWB3(results, params, spkHistPrior_setup, addPriorForWB, couplingGroup, precomputedPortion)

% i.i.d non-mean Gaussian over spike history coefficients & constant


K = size(params.B,1);

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

%w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);


if(~isempty(results.dW))
    w = double(params.W);
    if(nargin < 4 || isempty(precomputedPortion))
        precomputedPortion.w_sig_0 = exp(log_w_sig);
        precomputedPortion.w_sig =  precomputedPortion.w_sig_0 + spkHistPrior_setup.hyperprior.min_w_sig;
    end
    dW = double(results.dW);
    if(addPriorForWB)
        results.dW(:) = -w + dW*precomputedPortion.w_sig;
    else
        results.dW(:) = dW*precomputedPortion.w_sig;
    end

    if(isfield(results, "dH") && ~isempty(results.dH))
        results.dH(w_mu_idx)  = results.dH(w_mu_idx)  + sum(dW);
        results.dH(w_sig_idx) = results.dH(w_sig_idx) + sum(dW.*w.*precomputedPortion.w_sig_0);
    end
end

%% linear terms
if(K > 0 && ~isempty(results.dB))
    b_mu_idx = 2 + (1:K);
    b_sig_idx = 2 + K + 1;
    %b_mu = H(b_mu_idx);
    log_b_sig = H(b_sig_idx);

    b = double(params.B);
    if(nargin < 4 || isempty(precomputedPortion))
        precomputedPortion.b_sig_0 = exp(log_b_sig);
        precomputedPortion.b_sig =  precomputedPortion.b_sig_0 + spkHistPrior_setup.hyperprior.min_b_sig;
    end
    dB = double(results.dB);
    %% if transformed
    if(isfield(spkHistPrior_setup, "U_transform") && ~isempty(spkHistPrior_setup.U_transform) && ~isempty(dB))
        dB2 = spkHistPrior_setup.U_transform*dB;
        b2 = spkHistPrior_setup.U_transform'*b;
    else
        dB2 = dB;
        b2 = B;
    end

    if(addPriorForWB)
        dlp_b = -b;
        results.dB(:,:) = dlp_b + dB2.*precomputedPortion.b_sig;
    else
        results.dB(:,:) = dB2.*precomputedPortion.b_sig;
    end



    if(isfield(results, "dH") && ~isempty(results.dH))
        results.dH(b_mu_idx)  = results.dH(b_mu_idx)  + sum(dB,2);
        results.dH(b_sig_idx) = results.dH(b_sig_idx) + sum(dB.*b2.*precomputedPortion.b_sig_0, "all");
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
