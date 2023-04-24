function [params, precomputedPortion] = scaleWB3(params, spkHistPrior_setup, couplingGroup)

% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);


w = double(params.W);
precomputedPortion.w_sig_0 = exp(log_w_sig);
precomputedPortion.w_sig =  precomputedPortion.w_sig_0 + spkHistPrior_setup.hyperprior.min_w_sig;
params.W(:) = w*precomputedPortion.w_sig + w_mu;
%% linear terms
if(K > 0)
    b_mu_idx = 2 + (1:K);
    b_sig_idx = 2 + K + 1;
    b_mu = H(b_mu_idx);
    log_b_sig = H(b_sig_idx);

    b = double(params.B);

    precomputedPortion.b_sig_0 = exp(log_b_sig);
    precomputedPortion.b_sig =  precomputedPortion.b_sig_0 + spkHistPrior_setup.hyperprior.min_b_sig;

    %% if transformed
    if(isfield(spkHistPrior_setup, "U_transform") && ~isempty(spkHistPrior_setup.U_transform))
        b = spkHistPrior_setup.U_transform'*b;
    end

    b = b.*precomputedPortion.b_sig ;
     
    params.B(:,:) = b + b_mu; 

    if(~isempty(couplingGroup))
        precomputedPortion.T = params.Groups(couplingGroup).T{1};
        precomputedPortion.U = params.Groups(couplingGroup).T{2};
        precomputedPortion.V = params.Groups(couplingGroup).V;
        precomputedPortion.UV = (params.Groups(couplingGroup).T{2}.*params.Groups(couplingGroup).V);
        precomputedPortion.TUV = params.Groups(couplingGroup).T{1}*precomputedPortion.UV';
        params.B(:,:) = params.B(:,:) - precomputedPortion.TUV; 
    end
end
