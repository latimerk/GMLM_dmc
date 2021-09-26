function [lp, dlp_w, dlp_h, stim_sig_0, stim_dsig_0, d2lp_w] = stimulusGaussianPrior(ws, H_stim, stimPrior_setup, trp)
% i.i.d. multivariate normal prior distribution over columns of ws with covariance matrix
% defined in  GMLMprior_dmcStimVar
%
% calculates prior and all derivatives

if(isempty(H_stim) && isempty(ws))
    lp    = 0;
    dlp_w = [];
    dlp_h = [];
    stim_sig_0 = [];
    stim_dsig_0 = [];
    d2lp_w = [];
    
    return;
end


if(nargin < 4)
    trp = true;
end

nb = size(ws, 2); 

if(~isfield(stimPrior_setup, 'mu'))
    mu = 0;
else
    mu = stimPrior_setup.mu;
end
%% gets the correlation matrix (for the rows of ws)
if(nargout > 2)
    [stim_sig_0, stim_dsig_0] = DMC.priors.GMLMprior_dmcStimVar(H_stim, stimPrior_setup);
else
    [stim_sig_0] = DMC.priors.GMLMprior_dmcStimVar(H_stim, stimPrior_setup);
end

%% gets the log prior
rc = rcond(stim_sig_0);
if(rc < 1e-14 || isnan(rc) || isinf(rc))
    %% hacky bit for poorly conditioned matrices: hopefully this section never is run
    %fprintf('\t\twarning: poorly conditioned prior: rcond = %e!\n',rcond(sig));
    [u, s, ~] = svd(stim_sig_0);
    vv = diag(s) > 1e-10;
    s2 = diag(1./sqrt(diag(s)));
    inv_sqrt_stim_sig = s2(:,vv) * u(:,vv)';
    UU_C = inv_sqrt_stim_sig * (ws - mu);
    logdet_UU = max(-0.5e8, kgmlm.utils.logdet(stim_sig_0));

    if(nargout > 1)
        inv_stim_sig_0 = pinv(stim_sig_0);
        UU_C2 = (inv_stim_sig_0 * (ws - mu));
    end

    badCondition = true;
else
    [logdet_UU, sqrt_stim_sig] = kgmlm.utils.logdet(stim_sig_0);
    opts.UT    = true;
    opts.TRANSA = true;
    UU_C = linsolve(sqrt_stim_sig, (ws - mu), opts);
    if(nargout > 1)
        opts.UT     = true;
        opts.TRANSA = false;
        UU_C2 = linsolve(sqrt_stim_sig, UU_C, opts);
    end

    badCondition = false;
end

MAX_PRIOR = 1e20; %keeps prior from exploding to infinity and crashing (that situation shouldn't happen!)
lp = - 0.5 * min(MAX_PRIOR, trace(UU_C' * UU_C)) - nb/2 * logdet_UU - numel(ws)/2*log(2*pi);

%% gets derivative of log prior w.r.t. ws
if(nargout > 1)
    if(trp)
        dlp_w = -UU_C2';
    else
        dlp_w = -UU_C2;
    end
end

%% gets Hessian of log prior w.r.t. ws
if(nargout > 5)
    if(~badCondition)
        inv_stim_sig_0 = inv(stim_sig_0);
    end
    if(trp) 
        d2lp_w = -kron(inv_stim_sig_0, eye(nb));
    else
        d2lp_w = -kron(eye(nb), inv_stim_sig_0);
    end
end

%% gets derivative of log prior w.r.t. H
if(nargout > 2) 
    dlp_h = zeros(numel(H_stim), 1);
    for ii = 1:numel(H_stim)
        if(badCondition)
            dlp_h(ii) = 1/2*min(1e8,trace((UU_C2)' *stim_dsig_0(:,:,ii)*(UU_C2))) - nb/2*trace(inv_stim_sig_0*stim_dsig_0(:,:,ii));
        else
            dlp_h(ii) = 1/2*trace((UU_C2)' *stim_dsig_0(:,:,ii)*(UU_C2 )) - nb/2*trace(stim_sig_0\stim_dsig_0(:,:,ii));
        end
    end
end


        