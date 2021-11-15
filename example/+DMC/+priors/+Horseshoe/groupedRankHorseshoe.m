%% regularized Horseshoe prior for rank reduction and shrinkage
% Combining ideas from 
%
%   Chakraborty, Antik, Anirban Bhattacharya, and Bani K. Mallick. "Bayesian sparse multiple regression for simultaneous rank reduction and variable selection." Biometrika 107.1 (2020): 205-221.
%
%   Piironen, Juho, and Aki Vehtari. "Sparsity information and regularization in the horseshoe and other shrinkage priors." Electronic Journal of Statistics 11.2 (2017): 5018-5051.
%
% Implementation by Kenneth Latimer (2021)
%
function [results_group] = groupedRankHorseshoe(params, results, groupNum, prior_setup)

if(isempty(params))
    results_group = 1;
    return;
end

params_group  = params.Groups( groupNum);
results_group = results.Groups(groupNum);

H = double(params_group.H(:));
D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

compute_dH = isfield(results_group, 'dH') && ~isempty(results_group.dH);
if(compute_dH)
    results_group.dH(:) = 0;
end
results_group.log_prior_VT(:) = 0;

NS_total = 0;
for ss = 1:D
    if(ss == 1) %
        NS_total = NS_total + prior_setup.V.N_scales;   
    else %T{ss-1}
        NS_total = NS_total + prior_setup.T(ss-1).N_scales;      % number of scales
    end
end
includeTau = NS_total > 0;

% parameters
% c2    - controls slab size (regularized horseshoe)  inv-gamma(a_c, b_c)
% tau_r - global scales per rank - (t+_{df_global}(0, tau_0^2))
% lambda_kr,j - local scales (t+_{df_local}(0, 1))
% b_kr,j ~ N(0, tau_r^2*lambda_tilde^2_kj,r)
% lambda_tilde^2_kj,r = c2*(lambda_kr,j)^2./(c2 + (tau_r*lambda_kr,j)^2)

%% fixed hyperpriors
a_c = prior_setup.hyperparams.c.a;   % default: df_local/2      (2?)
b_c = prior_setup.hyperparams.c.b;   % default: df_local/2*s^2  (8?)

t_0 = prior_setup.hyperparams.tau_0; % suggested = f_0 / (1 - f_0) * 1/(sqrt(mu_tilde) * sqrt(N))    mu_tilde = mean observation, N = number of observations, f_0 = fraction of effective variables

df_global = prior_setup.hyperparams.df_global; % default: 3
df_local  = prior_setup.hyperparams.df_local;  % default: 3


if(includeTau)
    %% get global params
    if(prior_setup.regularized)
        idx_c2 = 1;
    else
        idx_c2 = 0;
    end
    
    if(prior_setup.rankwise_tau)
        idx_tau = idx_c2 + (1:R);
    else
        idx_tau = idx_c2 + 1;
    end
    ctr = max(idx_tau);

    log_tau = H(idx_tau);
    %tau   = DMC.utils.boundedExp(log_tau);

    %% log P(tau)
    if(compute_dH)
        [lp_log_tau, dlp_log_tau] = DMC.priors.halfTPrior(log_tau, df_global, t_0); % scale = t_0,  NOT t_0^2 (squaring is done in the halfTPrior function)
        results_group.dH(idx_tau) = results_group.dH(idx_tau) + dlp_log_tau;
    else
        lp_log_tau = DMC.priors.halfTPrior(log_tau, df_global, t_0);
    end
    results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_tau);

    %% log P(c2)
    if(prior_setup.regularized)
        log_c2 = H(idx_c2);
        [lp_c2, dlp_c2_0] = DMC.priors.inverseGaussianPrior(log_c2, a_c, b_c);
        % d log P(c2) / d_c2_0
        if(compute_dH)
            results_group.dH(idx_c2) = results_group.dH(idx_c2) + dlp_c2_0;
        end
        results_group.log_prior_VT = results_group.log_prior_VT + lp_c2;
    end

    %% log P(psi)

    if(prior_setup.crossrankwise_psi && R > 1)
        idx_psi = ctr + (1:NS_total);
        log_psi = H(idx_psi);
        [lp_log_psi, dlp_log_psi] = DMC.priors.halfTPrior(log_psi, df_global); 
        results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_psi);
        if(compute_dH)
            results_group.dH(idx_psi) = results_group.dH(idx_psi) + dlp_log_psi;
        end

        ctr = ctr + NS_total;
    else
        log_psi = 0;
    end
else
    log_c2 = 0;
    log_tau = 0;
    log_psi = 0;
    ctr = 0;
end

ctr_NS = 0;

%% goes through each piece
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        mu = prior_setup.V.mu;
        
        NS = prior_setup.V.N_scales;      % number of scales
        NH = prior_setup.V.N_hyperparams; % number of extra hyperparams in this set
        prior_setup_ss = prior_setup.V.setup;
        
        compute_dU = ~isempty(results_group.dV);
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        mu = prior_setup.T(ss-1).mu;
        NS = prior_setup.T(ss-1).N_scales;      % number of scales
        NH = prior_setup.T(ss-1).N_hyperparams; % number of extra hyperparams in this set
        prior_setup_ss = prior_setup.T(ss-1).setup;
        
        compute_dU = ~isempty(results_group.dT{ss-1});
    end
    if(isa(NH, 'function_handle'))
        NH = NH(R);
    end


    if(prior_setup.rankwise_phi && NS > 0 && R > 1)
        idx_phi = ctr + (1:(R));
        log_phi = reshape(H(idx_phi), [], R);
        ctr = ctr + R;
    else
        log_phi = 0;
    end

    
    idx_lambda = ctr + (1:(NS*R));
    log_lambda_ss = reshape(H(idx_lambda), [], R);
    %lambda_ss = DMC.utils.boundedExp(log_lambda_ss); % current lambda
    %tau2lambda2 = DMC.utils.boundedExp(2*(log_lambda_ss + log_tau'));
    
    ctr = ctr + NS*R;
    
    idx_hc = ctr + (1:NH);
    H_ss = H(idx_hc); % current other hyperparams
    ctr = ctr + NH;
    

    if(NS > 0)
        if(prior_setup.crossrankwise_psi && R > 1)
            idx_psi_c = (1:NS) + ctr_NS;
            ctr_NS = ctr_NS + NS;
        else
            idx_psi_c = 1;
        end

        if(prior_setup.regularized)
        %     scales_ss_0 = 2*log_tau' + log_c2 + 2*log_lambda_ss - log(c2 + tau2lambda2);
            scales_ss_0 = 2*log_tau' + 2*log_lambda_ss + 2*log_phi + 2*log_psi(idx_psi_c) - log1p(DMC.utils.boundedExp(2*(log_lambda_ss + log_tau' + log_phi + log_psi(idx_psi_c) - 0.5*log_c2)));
        else
            scales_ss_0 = 2*log_tau' + 2*log_lambda_ss + 2*log_phi + 2*log_psi(idx_psi_c) ;
        end
    else
        scales_ss_0 = [];
    end
    
    %% log p(U | tau, lambda, c, H_c)
    if(compute_dH)
        [lp, dlp_u, dlp_scales, dlp_h] = DMC.priors.scaledGaussianComponent(U, mu, scales_ss_0, H_ss, prior_setup_ss);
    elseif(compute_dU)
        [lp, dlp_u] = DMC.priors.scaledGaussianComponent(U, mu, scales_ss_0, H_ss, prior_setup_ss);
    else
        lp = DMC.priors.scaledGaussianComponent(U, mu, scales_ss_0, H_ss, prior_setup_ss);
    end
    
    results_group.log_prior_VT = results_group.log_prior_VT + lp;
    
    if(compute_dU)
        % d log p(U | tau, phi, w, H_c) / dU
        if(ss == 1)
            results_group.dV = results_group.dV + dlp_u;
        else
            results_group.dT{ss-1} = results_group.dT{ss-1} + dlp_u;
        end
    end
    
    if(compute_dH && NS > 0)

        if(prior_setup.regularized)
            % d log p(U | tau, lambda, c2, H_c) / dlambda_0
    %         dscales_ss_0_dlambda = 2 - 2.*tau2lambda2./(c2 + tau2lambda2);
    %         dscales_ss_0_dlambda = 2 - 2./(c2./tau2lambda2 + 1);
            dscales_ss_0_dlambda = 2 - 2./(1 + DMC.utils.boundedExp(-2*(log_lambda_ss + log_tau' + log_phi - 0.5*log_c2)));
            results_group.dH(idx_lambda) = reshape(results_group.dH(idx_lambda), [], R) + dlp_scales.*dscales_ss_0_dlambda;
            
            % d log p(U | tau, lambda, c2, H_c) / dtau_0
            dscales_ss_0_dtau = dscales_ss_0_dlambda; % 2 - 2.*tau2lambda2./(c2 + tau2lambda2);
    
            if(prior_setup.rankwise_tau)
                results_group.dH(idx_tau) = reshape(results_group.dH(idx_tau), [], R) + sum(dlp_scales.*dscales_ss_0_dtau, 1);
            else
                results_group.dH(idx_tau) = results_group.dH(idx_tau) + sum(dlp_scales.*dscales_ss_0_dtau, 'all');
            end

            if(prior_setup.rankwise_phi && R > 1)
                results_group.dH(idx_phi) = reshape(results_group.dH(idx_phi), [], R) + sum(dlp_scales.*dscales_ss_0_dtau, 1);
            end

            if(prior_setup.crossrankwise_psi && R > 1)
                results_group.dH(idx_psi(idx_psi_c)) = results_group.dH(idx_psi(idx_psi_c)) + sum(dlp_scales.*dscales_ss_0_dtau, 2);
            end
    
            % d log p(U | tau, lambda, c2, H_c) / dc2_0
    %         dscales_ss_0_dc2 = 1 - c2./(c2 + tau2lambda2);
            dscales_ss_0_dc2 = 1 - 1./(1 + DMC.utils.boundedExp(2*(log_lambda_ss + log_tau' + log_phi - 0.5*log_c2)));
            results_group.dH(idx_c2) = results_group.dH(idx_c2) + sum(dlp_scales.*dscales_ss_0_dc2, 'all');
        else
            % d log p(U | tau, lambda, c2, H_c) / dlambda_0
            dscales_ss_0_dlambda = 2 ;
            results_group.dH(idx_lambda) = reshape(results_group.dH(idx_lambda), [], R) + dlp_scales.*dscales_ss_0_dlambda;
            
            % d log p(U | tau, lambda, c2, H_c) / dtau_0
            dscales_ss_0_dtau = dscales_ss_0_dlambda; 
            if(prior_setup.rankwise_tau)
                results_group.dH(idx_tau) = reshape(results_group.dH(idx_tau), [], R) + sum(dlp_scales.*dscales_ss_0_dtau, 1);
            else
                results_group.dH(idx_tau) = results_group.dH(idx_tau) + sum(dlp_scales.*dscales_ss_0_dtau, 'all');
            end

            if(prior_setup.crossrankwise_psi && R > 1)
                results_group.dH(idx_psi(idx_psi_c) ) = results_group.dH(idx_psi(idx_psi_c) ) + sum(dlp_scales.*dscales_ss_0_dtau, 2);
            end

            if(prior_setup.rankwise_phi && R > 1)
                results_group.dH(idx_phi) = reshape(results_group.dH(idx_phi), [], R) + sum(dlp_scales.*dscales_ss_0_dtau, 1);
            end
        end
    end
    
    if(NS > 0)
        %% log p(lambda) 
        if(compute_dH)
            [lp_log_lambda, dlp_log_lambda] = DMC.priors.halfTPrior(log_lambda_ss(:), df_local, 1);
            results_group.dH(idx_lambda) = results_group.dH(idx_lambda)+ dlp_log_lambda;
        else
            lp_log_lambda = DMC.priors.halfTPrior(log_lambda_ss(:), df_local, 1);
        end
        results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_lambda);

        %% log p(phi)
        if(prior_setup.rankwise_phi)
            if(compute_dH)
                [lp_log_phi, dlp_log_phi] = DMC.priors.halfTPrior(log_phi(:), df_global, 1);
                results_group.dH(idx_phi) = results_group.dH(idx_phi) + dlp_log_phi;
            else
                lp_log_phi = DMC.priors.halfTPrior(log_lambda_ss(:), df_global, 1);
            end
            results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_phi);
        end
    end
     
    %% log p(H_ss) : assumes all are i.i.d. half-t distributions with unit scale and H.nu df
    if(NH > 0)
        if(compute_dH)
            % dlog p(H_ss) / dH_ss
            [lp_H, dlp_H_hyper] = DMC.priors.halfTPrior(H_ss, prior_setup.hyperparams.H.nu);
            results_group.dH(idx_hc) = dlp_h + dlp_H_hyper;
        else
            lp_H = DMC.priors.halfTPrior(H_ss, prior_setup.hyperparams.H.nu);
        end
        results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_H);
    end
end

end
