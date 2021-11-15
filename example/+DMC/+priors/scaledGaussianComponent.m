function [lp, dlp_w, dlp_log_scales, dlp_h, D, Q] = scaledGaussianComponent(ws, mu, log_scales_0, H_c, comp_prior_setup)
% i.i.d. multivariate zero-mean normal prior distribution over columns of ws with covariance matrix
% defined in  GMLMprior_dmcStimVar
%
% calculates prior and all derivatives

if(isempty(H_c) && isempty(ws))
    lp         = 0;
    dlp_w      = [];
    dlp_log_scales = [];
    dlp_h      = [];
    return;
end


dlp_log_scales = zeros(size(log_scales_0));
dlp_h      = zeros(numel(H_c), 1);
dlp_w      = zeros(size(ws));
    
P = size(ws, 1);
R = size(ws, 2);
NC = numel(comp_prior_setup.parts);


H_scales = log_scales_0;

ws_mu = (ws - mu);
ctr = 1;
lp = -R*P/2*log(2*pi);

if(nargout > 4)
    compute_D = true;
else
    compute_D = false;
end

if(compute_D)
    D = zeros(size(ws_mu));
end
Q = zeros(R,1);

for cc = 1:NC
    type = comp_prior_setup.parts(cc).type;
    
    if(strcmpi(type, "all_group"))
        S_c = H_scales(ctr, :);
        
        enS_c2 = DMC.utils.boundedExp(-0.5*S_c);
        if(compute_D)
            D = repmat(DMC.utils.boundedExp(S_c), size(ws_mu,1), 1);
        end

        w_c2 = ws_mu .* enS_c2;
        eS_c = sum(w_c2.^2, 1);

        lp = lp - 0.5 * sum(eS_c) - 0.5 * P * sum(S_c);
        
        Q = eS_c';

        if(nargout > 2) 
            dlp_log_scales(ctr, :) = 0.5 * eS_c - 0.5 * P;
        end
        if(nargout > 1) 
            enS_c = DMC.utils.boundedExp(-S_c);
            dlp_w = -ws_mu .* enS_c; 
        end
        ctr = ctr + 1;

    elseif(strcmpi(type, "all_independent"))
        idx_sc = ctr + (0:(P-1));
        S_c = H_scales(idx_sc, :);

        enS_c2 = DMC.utils.boundedExp(-0.5*S_c);
        if(compute_D)
            D = DMC.utils.boundedExp(S_c);
        end

        w_c2 = ws_mu .* enS_c2;
        eS_c = w_c2 .^ 2;
        
        Q = sum(eS_c,1)';

        lp = lp - 0.5 * sum(eS_c,'all') - 0.5 * sum(S_c,'all');

        if(nargout > 2) 
            dlp_log_scales(idx_sc, :) =  0.5 * eS_c - 0.5;
        end
        if(nargout > 1) 
            enS_c = DMC.utils.boundedExp(-S_c);
            dlp_w  = dlp_w - ws_mu .* enS_c; 
        end
        
        ctr = ctr + P;
    elseif(strcmpi(type, "group"))
        S_c = H_scales(ctr, :);
        idx_ps =  comp_prior_setup.parts(cc).idx_params(:);
        
        w = ws_mu(idx_ps,:);
        enS_c2 = DMC.utils.boundedExp(-0.5*S_c);
        if(compute_D)
            D(idx_ps,:) = repmat(DMC.utils.boundedExp(S_c), size(w,1), 1);
        end

        w_c2 = w .* enS_c2;
        eS_c = sum(w_c2 .^ 2,1);
        Q = Q + eS_c';

        lp = lp - 0.5 * sum(eS_c,'all') - 0.5 * sum(S_c) * size(w,1);

        if(nargout > 2) 
            dlp_log_scales(ctr, :) = 0.5 * eS_c - 0.5 * size(w,1);
        end
        if(nargout > 1) 
            enS_c = DMC.utils.boundedExp(-S_c);
            dlp_w(idx_ps, :)   = -w .* enS_c; 
        end
        ctr = ctr + 1;
    elseif(strcmpi(type, "std"))
        if(isfield(comp_prior_setup.parts(cc), 'var') && ~isempty(comp_prior_setup.parts(cc).var))
            var_c = comp_prior_setup.parts(cc).var;
        else
            var_c = 1;
        end

        w_c2 = ws_mu ./ sqrt(var_c);
        eS_c = w_c2 .^ 2;
        if(compute_D)
            D(:,:) = var_c;
        end

        Q = sum(eS_c,1)';
        lp = lp - 0.5 * sum(eS_c, 'all') - 0.5 * R*P * log(var_c);

        if(nargout > 1) 
            dlp_w  = -ws_mu ./ var_c; 
        end
        ctr = ctr + 0;
    else
        error("Invalid covariance type");
    end
    
end



end


%% construct specific types of covariances
function [K, dK_alpha, dK_tau] = polarCovKernel_dmc(K, alpha, tau)

% Kernel in
% Padonou, Espéran, and Olivier Roustant. 
%   "Polar Gaussian processes and experimental designs in circular domains."
%   SIAM/ASA Journal on Uncertainty Quantification 4.1 (2016): 1014-1033.
%
% equation 3.7 + 3.9 
%NOTE: this is only used for a GP over stim directions. It is not used at all in the sin/cos tuning model

if(numel(ps) ~= 2)
    error('polarCovKernel: wrong number of parameters!');
end

%alpha  = alpha; % alpha is var
dalpha = 1;

dtau  = exp(tau);
tau   = dtau + 4;

K_0 = K./pi;
mx = max(0,1-K_0);
mt = mx.^tau;
kt = (1+tau*K_0);
K = alpha*kt.*mt;
K(isnan(K_0)) = 0;

if(nargout > 1)
    dmt = mt;
    
    mx0 = mx>0;
    dmt(mx0) = mt(mx0).*log(mx(mx0))*dtau;
    
    dkt = dtau*K_0;
    
    dK_tau = alpha*(dkt.*mt + kt.*dmt);
    dK_tau(isnan(K_0)) = 0;
    
    dK_alpha = dalpha*kt.*mt;
    dK_alpha(isnan(K_0)) = 0;
end

end
        

