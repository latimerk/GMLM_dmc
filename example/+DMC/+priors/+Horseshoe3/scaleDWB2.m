function [results] = scaleDWB2(results, params, addPriorForWB, precomputedPortion)

% i.i.d non-mean Gaussian over spike history coefficients & constant


K = size(params.B,1);

w_sig_idx = 1;

b_sig_idx = 2;

H = double(params.H);

%w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);
%b_mu = H(b_mu_idx);
log_b_sig = H(b_sig_idx);


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
        results.dH(w_sig_idx) = results.dH(w_sig_idx) + sum(dW.*w.*precomputedPortion.w_sig );
    end
end

%% linear terms
if(K > 0 && ~isempty(results.dB))
    b = double(params.B);
    if(nargin < 4 || isempty(precomputedPortion))
        precomputedPortion.b_sig = exp(log_b_sig);
    end
    dB = double(results.dB);
    if(addPriorForWB)
        results.dB(:,:) = -b + dB.*precomputedPortion.b_sig;
    else
        results.dB(:,:) = dB.*precomputedPortion.b_sig;
    end

    if(~isempty(results.dH))
        results.dH(b_sig_idx) = results.dH(b_sig_idx) + sum(dB.*b.*precomputedPortion.b_sig,"all");
    end
end
