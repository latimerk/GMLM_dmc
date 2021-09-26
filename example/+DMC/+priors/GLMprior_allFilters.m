function [resultsStruct] = GLMprior_allFilters(paramStruct, resultsStruct, stimPrior_setup, levPrior_setup, spkHistPrior_setup)
%this prior is setup assuming I pieces together the GLM design matrix in a particular way - the only real flexibility is that I can include spk hist or not

%% pulls out hyperparameters
H = double(paramStruct.H);

H_stim = H(1:stimPrior_setup.NH);
dks = sum(cellfun(@numel,paramStruct.Ks));

useSpk = ismember("hspk", paramStruct.group_names);

lev_idx  = stimPrior_setup.NH + 1; %assumes 1 lever hyperparam
hspk_idx = stimPrior_setup.NH + 2; %assumes 1 spk hist hyperparam

H_lev = H(lev_idx);
if(useSpk)
    H_spk = H(hspk_idx);
end

%% gets filters
nf = stimPrior_setup.numFilters; 
ws = double(reshape(paramStruct.Ks{1},[],nf))';
nb = size(ws,2);
wl = double(paramStruct.Ks{2});
if(useSpk)
    wh = double(paramStruct.Ks{3});
end
%% get all log prior info
lp_spk = 0;
dlp_h_spk = [];
dlp_k_spk = [];
d2lp_k_spk = [];
if(~isempty(resultsStruct.d2K))
    [lp_stim, dlp_k_stim, dlp_h_stim, stim_sig_0, stim_dsig_0, d2lp_k_stim] = DMC.priors.stimulusGaussianPrior(ws, H_stim, stimPrior_setup);
    [lp_lev , dlp_k_lev,  dlp_h_lev,   dprior_sigma_inv_lev, d2lp_k_lev]    = DMC.priors.simpleGaussianPrior(wl, H_lev);
    if(useSpk)
        [lp_spk , dlp_k_spk, dlp_h_spk, dprior_sigma_inv_spk, d2lp_k_spk] = DMC.priors.simpleGaussianPrior(wh, H_spk);
    end
elseif((isfield(resultsStruct, 'dH') && ~isempty(resultsStruct.dH)) || isfield(resultsStruct,'dprior_sigma_inv'))
    [lp_stim, dlp_k_stim, dlp_h_stim, stim_sig_0, stim_dsig_0] = DMC.priors.stimulusGaussianPrior(ws, H_stim, stimPrior_setup);
    [lp_lev , dlp_k_lev,  dlp_h_lev, dprior_sigma_inv_lev]   = DMC.priors.simpleGaussianPrior(wl, H_lev);
    if(useSpk)
        [lp_spk , dlp_k_spk,   dlp_h_spk, dprior_sigma_inv_spk] = DMC.priors.simpleGaussianPrior(wh, H_spk);
    end
elseif(~isempty(resultsStruct.dK))
    [lp_stim, dlp_k_stim] = DMC.priors.stimulusGaussianPrior(ws, H_stim, stimPrior_setup);
    [lp_lev , dlp_k_lev] = DMC.priors.simpleGaussianPrior(wl, H_lev);
    if(useSpk)
        [lp_spk , dlp_k_spk] = DMC.priors.simpleGaussianPrior(wh, H_spk);
    end
else
    lp_stim = DMC.priors.stimulusGaussianPrior(ws, H_stim, stimPrior_setup);
    lp_lev = DMC.priors.simpleGaussianPrior(wl, H_lev);
    if(useSpk)
        lp_spk = DMC.priors.simpleGaussianPrior(wh, H_spk);
    end
end

% hessian info for evidence optimization
%  dprior_sigma_inv is dim_K x dim_K x dim_H where the third indexes the hyperparam
%  It is the derivative of the inverse of the Hessian of the log prior for each hyperparam
if(isfield(resultsStruct, 'dprior_sigma_inv'))
    resultsStruct.dprior_sigma_inv = zeros(dks, dks, numel(H));
    rc = rcond(stim_sig_0);
    for ii = 1:size(stim_dsig_0,3)
        if(~isnan(rc) && ~isinf(rc) && rc > 1e-16)
            xx_0 = -(stim_sig_0\(stim_dsig_0(:,:,ii)/stim_sig_0));
        else
            p_inv_stim_sig = pinv(stim_sig_0);
            xx_0 = -(p_inv_stim_sig*(stim_dsig_0(:,:,ii)*p_inv_stim_sig));
        end
        xx = kron(xx_0,eye(nb));
        resultsStruct.dprior_sigma_inv(1:size(xx,1),1:size(xx,1),ii) = xx;
    end
    
    idx = size(stim_sig_0,1)*nb + (1:numel(wl));
    resultsStruct.dprior_sigma_inv(idx,idx,lev_idx) = dprior_sigma_inv_lev;
    if(useSpk)
        idx = size(stim_sig_0,1)*nb + numel(wl) + (1:numel(wh));
        resultsStruct.dprior_sigma_inv(idx,idx,hspk_idx) = dprior_sigma_inv_spk;
    end
end

%% log hyperprior
nus_stim_all = stimPrior_setup.hyperprior.nu;
        
nus_spk = [];
if(useSpk)
    nus_spk = spkHistPrior_setup.hyperprior.nu;
end
nu = [nus_stim_all(1:stimPrior_setup.NH);
      levPrior_setup.hyperprior.nu;
      nus_spk];

if(isfield(resultsStruct, 'dH') && ~isempty(resultsStruct.dH))
        [lp_H, dlp_H_hyper] = DMC.priors.halfTPrior(H, nu);
else
    lp_H = DMC.priors.halfTPrior(H, nu);
end

%% sum it all together
if(~isempty(resultsStruct.dK))
    resultsStruct.dK = resultsStruct.dK + [dlp_k_stim(:); dlp_k_lev(:); dlp_k_spk(:);0];
    if(~isempty(resultsStruct.d2K))
        prior_d2K = blkdiag(d2lp_k_stim, d2lp_k_lev, d2lp_k_spk, 0);
        resultsStruct.d2K = resultsStruct.d2K + prior_d2K;
    end
end
if(isfield(resultsStruct, 'dH') && ~isempty(resultsStruct.dH))
    resultsStruct.dH = [dlp_h_stim; dlp_h_lev; dlp_h_spk] + dlp_H_hyper;
end 
resultsStruct.log_prior = lp_stim + lp_lev + lp_spk + sum(lp_H);


