function [results] = spkHistPrior3(params, results, spkHistPrior_setup)
% i.i.d non-mean Gaussian over spike history coefficients & constant

K = size(params.B,1);

w_mu_idx  = 1;
w_sig_idx = 2;


H = double(params.H);

w_mu = H(w_mu_idx);
log_w_sig = H(w_sig_idx);

if(K > 0)
    b_mu_idx = 2 + (1:K);
    b_sig_idx = 2 + K + 1;
    b_mu = H(b_mu_idx);
    log_b_sig = H(b_sig_idx);
end

%% log hyperprior
if(isfield(results, 'dH') && ~isempty(results.dH))
    [lp_H1, results.dH(w_sig_idx)] = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    [lp_H3, results.dH(w_mu_idx)] = DMC.priors.simpleGaussianPrior(w_mu, spkHistPrior_setup.hyperprior.log_w_mu_sig);

    if(K > 0)
        [lp_H2, results.dH(b_sig_idx)] = DMC.priors.halfTPrior(log_b_sig, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        [lp_H4, results.dH(b_mu_idx)] = DMC.priors.simpleGaussianPrior(b_mu, spkHistPrior_setup.hyperprior.log_b_mu_sig);
    else
        lp_H2 = 0;
        lp_H4 = 0;
    end

    lp_H = sum(lp_H1) + sum(lp_H2) + sum(lp_H3) + sum(lp_H4);
else
    lp_H1 = DMC.priors.halfTPrior(log_w_sig, spkHistPrior_setup.hyperprior.w_sig_nu, spkHistPrior_setup.hyperprior.log_w_sig_scale, true);
    lp_H3 = DMC.priors.simpleGaussianPrior(w_mu, spkHistPrior_setup.hyperprior.log_w_mu_sig);

    if(K > 0)
        lp_H2 = DMC.priors.halfTPrior(log_b_sig, spkHistPrior_setup.hyperprior.b_sig_nu, spkHistPrior_setup.hyperprior.log_b_sig_scale, true);
        lp_H4 = DMC.priors.simpleGaussianPrior(b_mu, spkHistPrior_setup.hyperprior.log_b_mu_sig);
    else
        lp_H2 = 0;
        lp_H4 = 0;
    end

    lp_H = sum(lp_H1) + sum(lp_H2) + sum(lp_H3) + sum(lp_H4);
end

    
%% log prior of constant
w = double(params.W);
if(isfield(results, 'dH') && ~isempty(results.dH))
    [lp_w, dlp_w, dlp_log_w_sig] = DMC.priors.simpleGaussianPrior(w, log_w_sig);
    results.dH(w_sig_idx) = results.dH(w_sig_idx) + dlp_log_w_sig;
    results.dW(:) = results.dW(:) + dlp_w;
elseif(~isempty(results.dB))
    [lp_w, dlp_w] = DMC.priors.simpleGaussianPrior(w, log_w_sig);
    results.dW(:) = results.dW(:) + dlp_w;
else
    lp_w = DMC.priors.simpleGaussianPrior(w, log_w_sig);
end 


%% derivatives and prior for any GLM-like terms
if(K > 0)
    b = double(params.B);
    
    %% log prior
    if(isfield(results, 'dH') && ~isempty(results.dH))
        [lp_b, dlp_b, dlp_log_b_sig] = DMC.priors.simpleGaussianPrior(b, log_b_sig, spkHistPrior_setup.U_sigma_chol_inv, spkHistPrior_setup.U_sigma_inv);
        results.dH(b_sig_idx) = results.dH(b_sig_idx) + dlp_log_b_sig;
        results.dB(:,:) = results.dB(:,:) + dlp_b;
    elseif(~isempty(results.dB))
        [lp_b, dlp_b] = DMC.priors.simpleGaussianPrior(b, log_b_sig, spkHistPrior_setup.U_sigma_chol_inv, spkHistPrior_setup.U_sigma_inv);
        results.dB(:,:) = results.dB(:,:) + dlp_b;
    else
        lp_b = DMC.priors.simpleGaussianPrior(b, log_b_sig, spkHistPrior_setup.U_sigma_chol_inv, spkHistPrior_setup.U_sigma_inv);
    end
else
    lp_b = 0;
end

results.log_prior_WB = lp_b + lp_w + sum(lp_H);
