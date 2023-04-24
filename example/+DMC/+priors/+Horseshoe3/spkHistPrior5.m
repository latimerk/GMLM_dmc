function [results] = spkHistPrior5(params, results, spkHistPrior_setup, addDPriorForWB)
% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);
K2 = K/2;

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);

if(K > 0)
    b_mu_idx = 2 + (1:K);
    b_sig_idx1 = 2 + K + 1;
    b_sig_idx2 = 2 + K + 2;
    b_mu = H(b_mu_idx);
    log_b_sig1 = H(b_sig_idx1);
    log_b_sig2 = H(b_sig_idx2);
end

%% log hyperprior
if(isfield(results, 'dH') && ~isempty(results.dH))
    [lp_H1, results.dH(w_sig_idx)] = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    
    [lp_H3, results.dH(w_mu_idx)] = DMC.priors.simpleGaussianPrior(w_mu, spkHistPrior_setup.hyperprior.log_w_mu_sig);

    if(K > 0)
        [lp_H2, results.dH(b_sig_idx1)] = DMC.priors.halfTPrior(log_b_sig1, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        [lp_H5, results.dH(b_sig_idx2)] = DMC.priors.halfTPrior(log_b_sig2, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        [lp_H4, results.dH(b_mu_idx)] = DMC.priors.simpleGaussianPrior(b_mu, spkHistPrior_setup.hyperprior.log_b_mu_sig);
    else
        lp_H2 = 0;
        lp_H5 = 0;
        lp_H4 = 0;
    end

else
    lp_H1 = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    
    lp_H3 = DMC.priors.simpleGaussianPrior(w_mu, spkHistPrior_setup.hyperprior.log_w_mu_sig);
    if(K > 0)
        lp_H2 = DMC.priors.halfTPrior(log_b_sig1, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        lp_H5 = DMC.priors.halfTPrior(log_b_sig2, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        lp_H4 = DMC.priors.simpleGaussianPrior(b_mu, spkHistPrior_setup.hyperprior.log_b_mu_sig);
    else
        lp_H2 = 0;
        lp_H5 = 0;
        lp_H4 = 0;
    end

end
lp_H = sum(lp_H1) + sum(lp_H2) + sum(lp_H3) + sum(lp_H4) + sum(lp_H5);

    
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

    use_U_sigma = isfield(spkHistPrior_setup,"U_chol_sigma_inv") && ~isempty(spkHistPrior_setup.U_chol_sigma_inv);
    if(use_U_sigma)
        Ur = b'*spkHistPrior_setup.U_chol_sigma_inv; % looks for lower triangular?
        normConst = - numel(b)/2*log(2*pi) - sum(log(diag(spkHistPrior_setup.U_chol_sigma_inv)));
        lpb = -0.5*sum(Ur.^2, "all") - normConst;
    else
        normConst = - numel(b)/2*log(2*pi) - numel(b);
        lpb = -1/2*sum(b.^2,"all") + normConst;
    end

    if(~isempty(results.dB) && addDPriorForWB)
        if(use_U_sigma)
            dlp_b = -spkHistPrior_setup.U_sigma_inv*b;
        else
            dlp_b = -b;
        end
        results.dB(:,:) = results.dB(:,:) + dlp_b;
    end
else
    lpb = 0;
end

results.log_prior_WB = lpb + lpw + sum(lp_H);
