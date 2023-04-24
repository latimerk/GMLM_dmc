function [results] = spkHistPrior2(params, results, spkHistPrior_setup, addDPriorForWB)
% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);

w_sig_idx = 1;
b_sig_idx = 2;

H = double(params.H);

log_w_sig = H(w_sig_idx);
log_b_sig = H(b_sig_idx);


%% log hyperprior
if(isfield(results, 'dH') && ~isempty(results.dH))
    [lp_H1, results.dH(w_sig_idx)] = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    [lp_H2, results.dH(b_sig_idx)] = DMC.priors.halfTPrior(log_b_sig, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
    

    lp_H = sum(lp_H1) + sum(lp_H2);
else
    lp_H1 = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    lp_H2 = DMC.priors.halfTPrior(log_b_sig, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
    

    lp_H = sum(lp_H1) + sum(lp_H2);
end

    
%% log prior of constant
w = double(params.W);
lpw = -0.5*sum(w.^2,"all") - numel(w)/2*log(2*pi);
if(~isempty(results.dW) && addDPriorForWB)
    results.dW(:) = results.dW(:) - w;
end

%% derivatives and prior for any GLM-like terms
if(K > 0)
    b = double(params.B);
    
    %% log prior
    lpb = -0.5*sum(b.^2,"all") - numel(b)/2*log(2*pi);
    if(~isempty(results.dB) && addDPriorForWB)
        results.dB(:,:) = results.dB(:,:) - B;
    end
else
    lpb = 0;
end

results.log_prior_WB = lpb + lpw + sum(lp_H);
