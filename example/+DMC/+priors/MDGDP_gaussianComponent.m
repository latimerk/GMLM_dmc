function [lp, dlp_w, dlp_scales, dlp_h] = MDGDP_gaussianComponent(ws, mu, H_scales, H_c, comp_prior_setup, computeC_for_Gibbs)
% i.i.d. multivariate zero-mean normal prior distribution over columns of ws with covariance matrix
% defined in  GMLMprior_dmcStimVar
%
% calculates prior and all derivatives

if(isempty(H_c) && isempty(ws))
    lp         = 0;
    dlp_w      = [];
    dlp_scales = [];
    dlp_h      = [];
    return;
end

if(nargin < 6 || isempty(computeC_for_Gibbs))
    computeC_for_Gibbs = false;
end

lp = 0;
dlp_scales = zeros(size(H_scales));
dlp_h      = zeros(numel(H_c), 1);
dlp_w      = zeros(size(ws));
    
P = size(ws, 1);
R = size(ws, 2);
NC = numel(comp_prior_setup.parts);

c_rs = nan(R, 1);

for rr = 1:R
    %% gets the correlation matrix (for the rows of ws)
    stim_sig_0 = zeros(P, 1);
    diag_sig   = true; % keeps track of whether or not it is a diagonal matrix
    dK = cell(NC,1);
    ctr = 1;
    
    for cc = 1:NC
        type = comp_prior_setup.parts(cc).type;
        
        if(strcmpi(type, "group"))
            H_scales_c = H_scales(ctr, rr);
            ctr = ctr + 1;
            
            idx_ps = comp_prior_setup.parts(cc).idx_ps;
            
            S_c = comp_prior_setup.parts(cc).S * H_scales_c;
        elseif(strcmpi(type, "angular_gp"))
            H_scales_c = H_scales(ctr, rr);
            ctr = ctr + 1;
            
            idx_hps  = comp_prior_setup.parts(cc).idx_hyperparams;
            idx_ps = comp_prior_setup.parts(cc).idx_ps;
            
            if(nargout > 2) 
                [S_c, dK_alpha, dK_tau] = polarCovKernel_dmc(comp_prior_setup.parts(cc).S, H_scales_c, H_c(idx_hps));
                dK{cc} = cat(3, dK_alpha, dK_tau);
            else
                S_c = polarCovKernel_dmc(comp_prior_setup.parts(cc).S, H_scales_c, H_c(idx_hps));
            end
            
        elseif(strcmpi(type, "all_group"))
            S_c = H_scales(ctr, rr) * ones(P, 1);
            ctr = ctr + 1;
            idx_ps = 1:P;
            
        elseif(strcmpi(type, "all_independent"))
            S_c = H_scales(ctr + (0:(P-1)), rr);
            ctr = ctr + P;
            
            idx_ps = 1:P;
            
        else
            error("Invalid covariance type");
        end
        
        % adds to current prior
        if(~isdiag(S_c) && ~iscolumn(S_c) && diag_sig)
            % if isn't actually diagonal, makes it diagonal now
            diag_sig = false;
            stim_sig_0 = diag(stim_sig_0);
        end
        if(diag_sig)
            if(~iscolumn(S_c))
                S_c = diag(S_c);
            end
            stim_sig_0(idx_ps, 1) = stim_sig_0(idx_ps, 1) + S_c;
        else
            if(iscolumn(S_c))
                S_c = diag(S_c);
            end
            stim_sig_0(idx_ps, idx_ps) = stim_sig_0(idx_ps, idx_ps) + S_c;
        end
    end
    
    %% gets the log prior
    if(~diag_sig)
        rc = rcond(stim_sig_0);
    end
    if(~diag_sig && (rc < 1e-14 || isnan(rc) || isinf(rc)))
        %% hacky bit for poorly conditioned matrices: hopefully this section never is run
        %fprintf('\t\twarning: poorly conditioned prior: rcond = %e!\n',rcond(sig));
        [u, s, ~] = svd(stim_sig_0);
        vv = diag(s) > 1e-10;
        s2 = diag(1./sqrt(diag(s)));
        inv_sqrt_stim_sig = s2(:,vv) * u(:,vv)';
        UU_C = inv_sqrt_stim_sig * (ws(:, rr) - mu);
        logdet_UU = max(-0.5e8, kgmlm.utils.logdet(stim_sig_0));

        if(nargout > 1)
            inv_stim_sig_0 = pinv(stim_sig_0);
            UU_C2 = (inv_stim_sig_0 * (ws(:, rr) - mu));
        end
    else
        if(diag_sig)
            logdet_UU = sum(log(stim_sig_0));
            UU_C = (ws(:, rr) - mu)./sqrt(stim_sig_0);
            if(nargout > 1)
                inv_stim_sig_0 = 1./stim_sig_0;
                UU_C2 =  (ws(:, rr) - mu)./stim_sig_0;
            end
        else
            [logdet_UU, sqrt_stim_sig] = kgmlm.utils.logdet(stim_sig_0);
            opts.UT    = true;
            opts.TRANSA = true;
            UU_C = linsolve(sqrt_stim_sig, (ws(:, rr) - mu), opts);
            if(nargout > 1)
                inv_stim_sig_0 = inv(stim_sig_0);
                opts.UT     = true;
                opts.TRANSA = false;
                UU_C2 = linsolve(sqrt_stim_sig, UU_C, opts);
            end
        end
    end
    
    c_rs(rr) = UU_C' * UU_C;
    if(computeC_for_Gibbs)
        %% if only getting C_r for Gibbs step
        continue;
    end
    
    MAX_PRIOR = 1e20; %keeps prior from exploding to infinity and crashing (that situation shouldn't happen!)
    lp = lp - 0.5 * min(MAX_PRIOR, c_rs(rr)) - 1/2 * logdet_UU - P/2*log(2*pi);

    %% gets derivative of log prior w.r.t. ws
    if(nargout > 1)
        dlp_w(:, rr) = -UU_C2';
    end

    %% gets derivative of log prior w.r.t. H
    if(nargout > 2) 
        ctr = 1;
        for cc = 1:NC
            type = comp_prior_setup.parts(cc).type;

            if(strcmpi(type, "group"))

                idx_ps = comp_prior_setup.parts(cc).idx_ps;

                if(diag_sig)
                    tt = inv_stim_sig_0(idx_ps)' * diag(comp_prior_setup.parts(cc).S);
                else
                    tt = sum(inv_stim_sig_0(idx_ps, idx_ps)' .* comp_prior_setup.parts(cc).S, 'all');
                end
                dlp_scales(ctr,rr) = dlp_scales(ctr,rr) + 1/2*min(1e8,UU_C2_c' * comp_prior_setup.parts(cc).S * UU_C2_c) - 1/2*tt;

                ctr = ctr + 1;
            elseif(strcmpi(type, "angular_gp"))

                idx_hps  = comp_prior_setup.parts(cc).idx_hyperparams;
                idx_ps = comp_prior_setup.parts(cc).idx_ps;
                
                UU_C2_c = UU_C2(idx_ps,:);
                if(diag_sig)
                    tt1 = inv_stim_sig_0(idx_ps)' * diag(dK{cc}(:, :, 2));
                    tt2 = inv_stim_sig_0(idx_ps)' * diag(dK{cc}(:, :, 1));
                else
                    tt1 = sum(inv_stim_sig_0(idx_ps, idx_ps)' * dK{cc}(:, :, 2), 'all');
                    tt2 = sum(inv_stim_sig_0(idx_ps, idx_ps)' * dK{cc}(:, :, 1), 'all');
                end
                dlp_h(    idx_hps) = dlp_h(    idx_hps) + 1/2*min(1e8,UU_C2_c' * dK{cc}(:, :, 2) * UU_C2_c) - 1/2*tt1;
                dlp_scales(ctr,rr) = dlp_scales(ctr,rr) + 1/2*min(1e8,UU_C2_c' * dK{cc}(:, :, 1) * UU_C2_c) - 1/2*tt2;

                ctr = ctr + 1;
            elseif(strcmpi(type, "all_group"))
                if(diag_sig)
                    tt = sum(inv_stim_sig_0);
                else
                    tt = trace(inv_stim_sig_0);
                end
                dlp_scales(ctr,rr) = dlp_scales(ctr,rr) + 1/2*min(1e8,UU_C2' * UU_C2) - 1/2*tt;
                
                ctr = ctr + 1;
            elseif(strcmpi(type, "all_independent"))
                H_idx = ctr + (0:(P-1));

                if(diag_sig)
                    tt = inv_stim_sig_0;
                else
                    tt = diag(inv_stim_sig_0);
                end
                dlp_scales(H_idx,rr) = dlp_scales(H_idx,rr) + 1/2*min(1e8,UU_C2 .* UU_C2) - 1/2*tt;
                ctr = ctr + P;
            else
                error("Invalid covariance type");
            end
        end
    end
end

if(computeC_for_Gibbs)
    lp = c_rs;
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
        