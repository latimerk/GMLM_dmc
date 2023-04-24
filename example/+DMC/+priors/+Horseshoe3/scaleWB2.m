function [params, precomputedPortion] = scaleWB2(params)

% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);

w_sig_idx = 1;
b_sig_idx = 2;

H = double(params.H);

log_w_sig = H(w_sig_idx);
log_b_sig = H(b_sig_idx);


w = double(params.W);
precomputedPortion.w_sig = exp(log_w_sig);
params.W(:) = w*precomputedPortion.w_sig;
%% linear terms
if(K > 0)
    b = double(params.B);
    precomputedPortion.b_sig = exp(log_b_sig);
    params.B(:,:) = b.*precomputedPortion.b_sig; 
end
