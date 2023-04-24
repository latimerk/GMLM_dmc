function [results] = scaleDWB5(results, params, spkHistPrior_setup, addPriorForWB, couplingGroup, precomputedPortion)

% i.i.d non-mean Gaussian over spike history coefficients & constant


K = size(params.B,1);
K2 = K/2;

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

%w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);


if(~isempty(results.dW))
    w = double(params.W);
    if(nargin < 4 || isempty(precomputedPortion))
        precomputedPortion.w_sig = exp(log_w_sig);
    end
    dW = double(results.dW);
    if(addPriorForWB)
        results.dW(:) = -w + dW*precomputedPortion.w_sig;
    else
        results.dW(:) = dW*precomputedPortion.w_sig;
    end

    if(~isempty(results.dH))
        results.dH(w_mu_idx)  = results.dH(w_mu_idx)  + sum(dW);
        results.dH(w_sig_idx) = results.dH(w_sig_idx) + sum(dW.*w.*precomputedPortion.w_sig );
    end
end

%% linear terms
if(K > 0 && ~isempty(results.dB))
    b_mu_idx = 2 + (1:K);
    b_sig_idx1 = 2 + K + 1;
    b_sig_idx2 = 2 + K + 2;
    %b_mu = H(b_mu_idx);
    log_b_sig1 = H(b_sig_idx1);
    log_b_sig2 = H(b_sig_idx2);

    b = double(params.B);
    if(nargin < 4 || isempty(precomputedPortion))
        precomputedPortion.b_sig1 = exp(log_b_sig1);
        precomputedPortion.b_sig2 = exp(log_b_sig2);
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
        use_U_sigma = isfield(spkHistPrior_setup,"U_chol_sigma_inv") && ~isempty(spkHistPrior_setup.U_chol_sigma_inv);
        if(use_U_sigma)
            dlp_b = -spkHistPrior_setup.U_sigma_inv*b;
        else
            dlp_b = -b;
        end
        results.dB((1:K2) +  0,:) = dlp_b((1:K2) +  0,:) + dB2((1:K2) +  0,:).*precomputedPortion.b_sig1;
        results.dB((1:K2) + K2,:) = dlp_b((1:K2) + K2,:) + dB2((1:K2) + K2,:).*precomputedPortion.b_sig2;
    else
        results.dB((1:K2) +  0,:) = dB2((1:K2) +  0,:).*precomputedPortion.b_sig1;
        results.dB((1:K2) + K2,:) = dB2((1:K2) + K2,:).*precomputedPortion.b_sig2;
    end



    if(~isempty(results.dH))
        results.dH(b_mu_idx)   = results.dH(b_mu_idx)  + sum(dB,2);
        results.dH(b_sig_idx1) = results.dH(b_sig_idx1) + sum(dB((1:K2) +  0,:).*b2((1:K2) +  0,:).*precomputedPortion.b_sig1, "all");
        results.dH(b_sig_idx2) = results.dH(b_sig_idx2) + sum(dB((1:K2) + K2,:).*b2((1:K2) + K2,:).*precomputedPortion.b_sig2, "all");
    end
    if(~isempty(couplingGroup))


        if(~isempty(results.Groups(couplingGroup).dT{1}))
            results.Groups(couplingGroup).dT{1} = results.Groups(couplingGroup).dT{1} - dB((1:K2) + 0,:)*(precomputedPortion.UVW1);
            results.Groups(couplingGroup).dT{1} = results.Groups(couplingGroup).dT{1} - dB((1:K2) + K2,:)*(precomputedPortion.UVW2);
        end
        if(~isempty(results.Groups(couplingGroup).dT{2}))
            results.Groups(couplingGroup).dT{2} = results.Groups(couplingGroup).dT{2} - (dB((1:K2) + 0,:)'*precomputedPortion.T).*precomputedPortion.VW1;
            results.Groups(couplingGroup).dT{2} = results.Groups(couplingGroup).dT{2} - (dB((1:K2) + K2,:)'*precomputedPortion.T).*precomputedPortion.VW2;
        end
        if(~isempty(results.Groups(couplingGroup).dT{3}))
            results.Groups(couplingGroup).dT{3}(1,:) = results.Groups(couplingGroup).dT{3}(1,:) - sum((dB((1:K2) + 0,:)'*precomputedPortion.T).*precomputedPortion.UV);
            results.Groups(couplingGroup).dT{3}(2,:) = results.Groups(couplingGroup).dT{3}(2,:) - sum((dB((1:K2) + K2,:)'*precomputedPortion.T).*precomputedPortion.UV);
        end
        if(~isempty(results.Groups(couplingGroup).dV))
            results.Groups(couplingGroup).dV(:,:) = results.Groups(couplingGroup).dV - (dB((1:K2) + 0,:)'*precomputedPortion.T).*precomputedPortion.UW1;
            results.Groups(couplingGroup).dV(:,:) = results.Groups(couplingGroup).dV - (dB((1:K2) + K2,:)'*precomputedPortion.T).*precomputedPortion.UW2;
        end
    end
end
