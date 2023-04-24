function [params, precomputedPortion] = scaleWB3(params, couplingGroup)

% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);
H = double(params.H);

w_mu_idx  = 1;
w_mu = H(w_mu_idx);

w = double(params.W);
params.W(:) = w + w_mu;

precomputedPortion.T = [];
%% linear terms

if(K > 0)
    b_mu_idx = 2 + (1:K);
    b_mu = H(b_mu_idx);

    b = double(params.B);
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
