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

ls_m3 = 1;

params_group  = params.Groups( groupNum);
results_group = results.Groups(groupNum);

H = double(params_group.H(:));
D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

if(prior_setup.include_rank_constant)
    nd = sum([prior_setup.T(:).on]) + prior_setup.V.on;
    rank_scale = -log(R)/nd;
else
    rank_scale = 0;
end

compute_dH = isfield(results_group, 'dH') && ~isempty(results_group.dH);
if(compute_dH)
    results_group.dH(:) = 0;
end
results_group.log_prior_VT(:) = 0;


%% gets regularizer
idx_log_c2 = 1;
log_c2 = H(idx_log_c2);

% d log P(c2) / d_c2_0
if(compute_dH)
    [lp_c, dlp_c_0] = DMC.priors.inverseGammaPrior(log_c2, prior_setup.c.a, prior_setup.c.b);
    results_group.dH(idx_log_c2) = results_group.dH(idx_log_c2) + dlp_c_0;
else
    lp_c = DMC.priors.inverseGammaPrior(log_c2, prior_setup.c.a, prior_setup.c.b);
end
results_group.log_prior_VT = results_group.log_prior_VT + lp_c;

%% gets global tau
idx_tau = 2;
log_tau = H(idx_tau);

% log P(tau)
if(compute_dH)
    [lp_log_tau, dlp_log_tau] = DMC.priors.halfTPrior(log_tau, prior_setup.tau.dfs, prior_setup.log_constant_scale, true); % scale = t_0,  NOT t_0^2 (squaring is done in the halfTPrior function)
    results_group.dH(idx_tau) = results_group.dH(idx_tau) + dlp_log_tau;
else
    lp_log_tau = DMC.priors.halfTPrior(log_tau, prior_setup.tau.dfs, prior_setup.log_constant_scale, true);
end
results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_tau);

%% gets rank phi
idx_phi = 2 + (1:(R));
log_phi = reshape(H(idx_phi), [], R);

if(compute_dH)
    [lp_log_phi, dlp_log_phi, ~, dlp_log_tau] = DMC.priors.halfTPrior(log_phi(:), prior_setup.phi.dfs, log_tau + rank_scale, true);
    results_group.dH(idx_phi) = results_group.dH(idx_phi) + dlp_log_phi;
    results_group.dH(idx_tau) = results_group.dH(idx_tau) + sum(dlp_log_tau,"all");
else
    lp_log_phi = DMC.priors.halfTPrior(log_phi(:), prior_setup.phi.dfs, log_tau + rank_scale, true);
end
results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_phi);

%% goes through each piece
ctr = 2 + R;
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        prior_setup_ss = prior_setup.V;
        
        compute_dU = ~isempty(results_group.dV);
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        prior_setup_ss = prior_setup.T(ss-1);
        
        compute_dU = ~isempty(results_group.dT{ss-1});
    end
    mu = prior_setup_ss.mu;

    %% get global scale stuff
    if(prior_setup_ss.on)
        if(isempty(prior_setup_ss.grps))
            NS = size(U,1);
        else
            NS = numel(prior_setup_ss.grps);
        end

        idx_lambda = ctr + (1:(NS*R));
        log_lambda_ss = reshape(H(idx_lambda), [], R);
        
        ctr = ctr + NS*R;

        %% log p(lambda) 
        log_phi_0 = repmat(log_phi, size(log_lambda_ss,1),1) - log(ls_m3);
        if(compute_dH)
            [lp_log_lambda, dlp_log_lambda, ~, dlp_log_phi] = DMC.priors.halfTPrior(log_lambda_ss, prior_setup.lambda.dfs, log_phi_0, true);
            results_group.dH(idx_lambda) = results_group.dH(idx_lambda) + dlp_log_lambda(:);
            results_group.dH(idx_phi) = results_group.dH(idx_phi) + sum(dlp_log_phi,1)';
        else
            lp_log_lambda = DMC.priors.halfTPrior(log_lambda_ss, prior_setup.lambda.dfs, log_phi_0, true);
        end
        results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_lambda,"all");


        compute_derivative = compute_dH && (NS > 0);
    
        log_scale2_part         = 2*(  ls_m3*log_lambda_ss + prior_setup_ss.lambda_log_scale(:) );
%         log_scales2_regularized = log_scale2_part;
        log_scales2_regularized = log_scale2_part + log_c2 - kgmlm.utils.logSumExp_pair( log_c2, log_scale2_part);
    
        se = (U).^2;
        log_Z = DMC.priors.HorseshoeHierarchical2.scaledGaussianComponent(U, log_scales2_regularized, prior_setup_ss.grps);
        iZ2 = kgmlm.utils.boundedExp(-2*log_Z);
        if(compute_derivative)
            dlp_U_dZ = se.*iZ2 - 1;
            [~, dlp_scales] = DMC.priors.HorseshoeHierarchical2.scaledGaussianComponent(U, log_scales2_regularized, prior_setup_ss.grps, dlp_U_dZ, log_Z);
        end

        lp_U = -0.5*se.*iZ2 - log_Z - 0.5*log(2*pi);
        lp_U = sum(lp_U, "all");

        results_group.log_prior_VT = results_group.log_prior_VT + lp_U;

        if(compute_dU)
            dlp_u = -U.*iZ2;
            if(ss == 1)
                results_group.dV = results_group.dV + dlp_u;
            else
                results_group.dT{ss-1} = results_group.dT{ss-1} + dlp_u;
            end
        end


        if(compute_derivative)

            dscales_ss_0_dc2  = 1 - 1./(1 + DMC.utils.boundedExp(log_scale2_part - log_c2));
            results_group.dH(idx_log_c2) = results_group.dH(idx_log_c2) + sum(dlp_scales.*dscales_ss_0_dc2, 'all');

            dscales_ss_0_dnum = 2 - 2./(1 + DMC.utils.boundedExp(log_c2 - log_scale2_part));
%             dscales_ss_0_dnum = 2;
            dn = dlp_scales.*dscales_ss_0_dnum;
            
            results_group.dH(idx_lambda) = reshape(results_group.dH(idx_lambda), [], R) + ls_m3*dn;
        end
    else
        %% log p(U | tau, lambda, c, H_c)
    
    
        if(compute_dU )
            [lp_u, dlp_u] = DMC.priors.simpleGaussianPrior(U - mu, 0, prior_setup_ss.U_sigma_chol_inv, prior_setup_ss.U_sigma_inv);
            if(ss == 1)
                results_group.dV = results_group.dV + dlp_u;
            else
                results_group.dT{ss-1} = results_group.dT{ss-1} + dlp_u;
            end
        else
            lp_u = DMC.priors.simpleGaussianPrior(U - mu, 0, prior_setup_ss.U_sigma_chol_inv, prior_setup_ss.U_sigma_inv);
        end
        results_group.log_prior_VT = results_group.log_prior_VT + lp_u;
    end
end

