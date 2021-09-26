function [results] = GMLMprior_spkHist(params, results, spkHistPrior_setup)
% i.i.d zero mean Gaussian over spike history coefficients, improper uniform over baseline rate terms

H = double(params.H);
%% log hyperprior
if(isfield(results, 'dH') && ~isempty(results.dH))
    [lp_H, results.dH(:)] = DMC.priors.halfTPrior(H, spkHistPrior_setup.hyperprior.nu);
else
    lp_H = DMC.priors.halfTPrior(H, spkHistPrior_setup.hyperprior.nu);
end
    

%% derivatives and prior for any GLM-like terms
if(numel(params.B) > 0)
    b = double(params.B);
    
    %% log prior
    if(isfield(results, 'dH') && ~isempty(results.dH))
        [lp, dlp_w, dlp_h] = DMC.priors.simpleGaussianPrior(b, H(1));
    elseif(~isempty(results.dB))
        [lp, dlp_w] = DMC.priors.simpleGaussianPrior(b, H(1));
    else
        lp = DMC.priors.simpleGaussianPrior(b, H(1));
    end
    
    %% sum the two together
    if(~isempty(results.dB))
        results.dB = results.dB + dlp_w;
    end
    if(isfield(results, 'dH') && ~isempty(results.dH))
        results.dH(1) = dlp_h + results.dH(1);
    end
    results.log_prior_WB = lp + sum(lp_H);
    
else
    %% if no parameters, still looks at hyperparameters (makes sampling happy if you're calling it without any linear terms for some dumb reason)
    results.log_prior_WB = sum(lp_H);
end
