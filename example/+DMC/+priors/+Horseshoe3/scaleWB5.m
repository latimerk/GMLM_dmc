function [params, precomputedPortion] = scaleWB5(params, spkHistPrior_setup, couplingGroup)

% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);
K2 = K/2;

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);


w = double(params.W);
precomputedPortion.w_sig = exp(log_w_sig);
params.W(:) = w*precomputedPortion.w_sig + w_mu;
%% linear terms
if(K > 0)
    b_mu_idx = 2 + (1:K);
    b_sig_idx1 = 2 + K + 1;
    b_sig_idx2 = 2 + K + 2;
    b_mu = H(b_mu_idx);
    log_b_sig1 = H(b_sig_idx1);
    log_b_sig2 = H(b_sig_idx2);

    b = double(params.B);

    precomputedPortion.b_sig1 = exp(log_b_sig1);
    precomputedPortion.b_sig2 = exp(log_b_sig2);

    %% if transformed
    if(isfield(spkHistPrior_setup, "U_transform") && ~isempty(spkHistPrior_setup.U_transform))
        b = spkHistPrior_setup.U_transform'*b;
    end

    b(1:K2,:)      = b(1:K2,:).*precomputedPortion.b_sig1 ;
    b((1:K2)+K2,:) = b((1:K2)+K2,:).*precomputedPortion.b_sig2 ;
     
    params.B(:,:) = b + b_mu; 

    if(~isempty(couplingGroup))
        precomputedPortion.T = params.Groups(couplingGroup).T{1};
        precomputedPortion.U = params.Groups(couplingGroup).T{2};
        precomputedPortion.W = params.Groups(couplingGroup).T{3};
        precomputedPortion.V = params.Groups(couplingGroup).V;
        precomputedPortion.UV = precomputedPortion.U  .* precomputedPortion.V;
        precomputedPortion.UW1 = precomputedPortion.U .* precomputedPortion.W(1,:);
        precomputedPortion.UW2 = precomputedPortion.U .* precomputedPortion.W(2,:);
        precomputedPortion.VW1 = params.Groups(couplingGroup).V.*precomputedPortion.W(1,:);
        precomputedPortion.VW2 = params.Groups(couplingGroup).V.*precomputedPortion.W(2,:);
        precomputedPortion.UVW1 = precomputedPortion.U.*precomputedPortion.VW1;
        precomputedPortion.UVW2 = precomputedPortion.U.*precomputedPortion.VW2;
        precomputedPortion.TUVW1 = precomputedPortion.T*precomputedPortion.UVW1';
        precomputedPortion.TUVW2 = precomputedPortion.T*precomputedPortion.UVW2';

        params.B((1:K2) +  0,:) = params.B((1:K2) +  0,:) - precomputedPortion.TUVW1; 
        params.B((1:K2) + K2,:) = params.B((1:K2) + K2,:) - precomputedPortion.TUVW2; 
    end
end
