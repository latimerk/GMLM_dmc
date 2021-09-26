function [lp, dlp_w, dlp_h, dprior_sigma_inv, d2lp_w] = simpleGaussianPrior(ws, H)
%i.i.d. Gaussian with zero mean and std exp(H(1))

nb = numel(ws);

%% transform hyperparam into std/var units
sig2 =  exp(max(-20, min(20, 2*H))); %keeps the variances within some numerically okay bounds

%% get log prior
sse  = sum(ws.^2,'all');
lp = - 1/2*sse./sig2 - nb*H - nb/2*log(2*pi);

%% get derivative of log prior w.r.t. params
if(nargout > 1)
    dlp_w =  -ws./sig2;
end

%% get Hessian of log prior w.r.t. params
if(nargout >= 5)
    %% full Hessian for GLM
    d2lp_w  = -eye(numel(ws))./sig2;
end

%% get derivative of log prior w.r.t. hyperparam
if(nargout > 2)
    dlp_h  =  sse./sig2 - nb; 
    
    d_inv_sig2 = -2/sig2;
    dprior_sigma_inv = eye(numel(ws))*d_inv_sig2;
end

end