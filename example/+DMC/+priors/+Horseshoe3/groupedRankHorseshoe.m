%% regularized Horseshoe prior for rank reduction and shrinkage
% Combining ideas from 
%
%   Chakraborty, Antik, Anirban Bhattacharya, and Bani K. Mallick. "Bayesian sparse multiple regression for simultaneous rank reduction and variable selection." Biometrika 107.1 (2020): 205-221.
%
%   Piironen, Juho, and Aki Vehtari. "Sparsity information and regularization in the horseshoe and other shrinkage priors." Electronic Journal of Statistics 11.2 (2017): 5018-5051.
%
% Implementation by Kenneth Latimer (2021)
%
function [results_group] = groupedRankHorseshoe(params, results, groupNum, prior_setup, addDPriorForVT)

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


%% gets regularizer
idx_log_c2 = 1;
log_c2 = H(idx_log_c2);

% d log P(c2) / d_c2_0
if(compute_dH)
    [lp_c2, dlp_c2_0] = DMC.priors.inverseGammaPrior(log_c2, prior_setup.c.a, prior_setup.c.b);
    results_group.dH(idx_log_c2) = results_group.dH(idx_log_c2) + dlp_c2_0;
else
    lp_c2 = DMC.priors.inverseGammaPrior(log_c2, prior_setup.c.a, prior_setup.c.b);
end
results_group.log_prior_VT = results_group.log_prior_VT + lp_c2;

%% gets global tau
idx_tau = 2;
log_tau = H(idx_tau);

% log P(tau)
if(compute_dH)
    [lp_log_tau, dlp_log_tau] = DMC.priors.halfTPrior(log_tau, prior_setup.tau.dfs, prior_setup.tau.log_scale, true); % scale = t_0,  NOT t_0^2 (squaring is done in the halfTPrior function)
    results_group.dH(idx_tau) = results_group.dH(idx_tau) + dlp_log_tau;
else
    lp_log_tau = DMC.priors.halfTPrior(log_tau, prior_setup.tau.dfs, prior_setup.tau.log_scale, true);
end
results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_tau);

%% gets rank phi
idx_phi = 2 + (1:(R));
log_phi = reshape(H(idx_phi), [], R);

if(compute_dH)
    [lp_log_phi, dlp_log_phi] = DMC.priors.halfTPrior(log_phi(:), prior_setup.phi.dfs, prior_setup.phi.log_scale, true);
    results_group.dH(idx_phi) = results_group.dH(idx_phi) + dlp_log_phi;
else
    lp_log_phi = DMC.priors.halfTPrior(log_phi(:), prior_setup.phi.dfs, prior_setup.phi.log_scale, true);
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
        if(compute_dH)
            [lp_log_lambda, dlp_log_lambda] = DMC.priors.halfTPrior(log_lambda_ss(:), prior_setup.lambda.dfs, 0, true);
            results_group.dH(idx_lambda) = results_group.dH(idx_lambda) + dlp_log_lambda;
        else
            lp_log_lambda = DMC.priors.halfTPrior(log_lambda_ss(:), prior_setup.lambda.dfs, 0, true);
        end
        results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_log_lambda);
    end

    %% log p(U | tau, lambda, c, H_c)
    normConst = - numel(U)/2*log(2*pi) - numel(U);
    lp = -1/2*sum(U.^2,"all") + normConst;

    results_group.log_prior_VT = results_group.log_prior_VT + lp;

    if(compute_dU && addDPriorForVT)
        dlp_u = -U;
        if(ss == 1)
            results_group.dV = results_group.dV + dlp_u;
        else
            results_group.dT{ss-1} = results_group.dT{ss-1} + dlp_u;
        end
    end
end

