%% multiway Dirichlet generalized double Pareto (M-DGDP) prior
%
% Prior from: Guhaniyogi, Rajarshi, Shaan Qamar, and David B. Dunson. "Bayesian tensor regression." The Journal of Machine Learning Research 18.1 (2017): 2733-2763.
%
% Implementation by Kenneth Latimer (2021)
%
function [results_group] = GMLMprior_MDGDP(params, results, groupNum, prior_setup)
if(isempty(params))
    results_group = 1;
    return;
end

params_group  = params.Groups( groupNum);
results_group = results.Groups(groupNum);

H   = double(params_group.H(:));
phi = double(params_group.H_gibbs);

D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

compute_dH = isfield(results_group, 'dH') && ~isempty(results_group.dH);
if(compute_dH)
    results_group.dH(:) = 0;
end
results_group.log_prior_VT(:) = 0;
% compute_dH_gibbs = ~isempty(results_group.dH_gibbs);

% parameters
% alpha - Dirichlet parameter for phi
% phi   - scale for each Rank, lives on simplex

% tau   - global scale

% w_ji,r - local scales (Exp(lmabda_j,r^2/2) where exp is parameterized as scale)
% lambda_j,r - for each rank/dim

%% fixed hyperpriors
a_alpha = prior_setup.hyperparams.alpha.a;   % default: 1,  beta over alpha
b_alpha = prior_setup.hyperparams.alpha.b;   % default: 1.5,  beta over alpha
p_alpha_beta = prior_setup.hyperparams.alpha.is_beta; % if is a beta distribution or gamma

a_lambda = prior_setup.hyperparams.lambda.a;   % default: 3,  beta over lambda
b_lambda = prior_setup.hyperparams.lambda.b;   % default: a_lambda^(1/(2*D)),  gamma over lambda

nu = 1;

%% hyperpriors - all exp transform
idx_tau   = 1;
idx_alpha = 2;
tau_0 = H(idx_tau);
tau   = exp(tau_0);

if(p_alpha_beta)
    alpha_0 = H(idx_alpha);
    alpha = 1./(1+exp(-alpha_0)); 
    dalpha_alpha_0 = alpha * (1 - alpha);
else
    alpha_0 = H(idx_alpha);
    alpha = exp(alpha_0); 
end

a_tau = R*alpha;
b_tau = alpha*(R/nu)^(1/D);


%% log P(tau)

lp_tau = a_tau * log(b_tau) - gammaln(a_tau) + (a_tau-1) * tau_0 - b_tau * tau;
lp_tau = lp_tau + tau_0; % for exp transform

results_group.log_prior_VT = lp_tau;
if(compute_dH)
    % d log p(tau) / dtau
    results_group.dH(idx_tau) = (a_tau-1) - b_tau * tau + 1;

    % d log p(tau) / dalpha
    da_tau = R;
    db_tau = (R/nu)^(1/D);
    %results_group.dH(idx_alpha) = da_tau * log(b_tau) + a_tau./b_tau *db_tau - psi(a_tau) * da_tau + da_tau * tau_0 - db_tau * tau;
    results_group.dH(idx_alpha) = (da_tau * log(b_tau) + R - psi(a_tau) * da_tau + da_tau * tau_0 - db_tau * tau) * dalpha_alpha_0;
end

%% log p(phi | alpha)
lp_phi = gammaln(alpha * R) - R * gammaln(alpha) + (alpha - 1) * sum(log(phi));

results_group.log_prior_VT = results_group.log_prior_VT  + lp_phi;

% if(compute_dH_gibbs)
%     % dlog p(phi | alpha) / dphi
%     results_group.dH_gibbs = (alpha - 1) ./ phi;
% end

if(compute_dH)
    % dlog p(phi | alpha) / dalpha
    results_group.dH(idx_alpha) = results_group.dH(idx_alpha) + (R  * psi(alpha * R) - R  * psi(alpha) +  sum(log(phi))) * dalpha_alpha_0;
end

%% log p(alpha)
if(p_alpha_beta)
    lp_alpha = gammaln(a_alpha + b_alpha) - gammaln(a_alpha) - gammaln(b_alpha) - (a_alpha - 1) * log(alpha) + (b_alpha - 1) * log(1 - alpha);
    lp_alpha = lp_alpha + log(dalpha_alpha_0); % for logistic transform

    results_group.log_prior_VT = results_group.log_prior_VT  + lp_alpha;

    if(compute_dH)
        % dlog p(alpha) / dalpha
        results_group.dH(idx_alpha) = results_group.dH(idx_alpha) + ((a_alpha - 1) ./ alpha - (b_alpha - 1) ./ (1 - alpha)) * dalpha_alpha_0 + 1 - 2 *alpha; % for the logistic transform part: 1 - 2* alpha
    end
else

    lp_alpha = a_alpha * log(b_alpha) - gammaln(a_alpha) + (a_alpha-1) * alpha_0 - b_alpha * alpha;
    lp_alpha = lp_alpha + alpha_0; % for exp transform
    results_group.log_prior_VT = results_group.log_prior_VT  + lp_alpha;
    if(compute_dH)
        results_group.dH(idx_alpha) = results_group.dH(idx_alpha) + (a_alpha-1) - b_alpha * alpha + 1;
    end
end

%% goes through each piece
ctr = 2;
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        mu = prior_setup.V.mu;
        
        NS = prior_setup.V.N_scales;      % number of scales
        NH = prior_setup.V.N_hyperparams; % number of extra hyperparams in this set
        stimPrior_setup_ss = prior_setup.V.setup;
        
        compute_dU = ~isempty(results_group.dV);
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        mu = prior_setup.T(ss-1).mu;
        NS = prior_setup.T(ss-1).N_scales;      % number of scales
        NH = prior_setup.T(ss-1).N_hyperparams; % number of extra hyperparams in this set
        stimPrior_setup_ss = prior_setup.T(ss-1).setup;
        
        compute_dU = ~isempty(results_group.dT{ss-1});
    end
    
    idx_lambda = ctr + (1:R);
    lambda_ss_0 = H(idx_lambda);
    lambda_ss = exp(lambda_ss_0); % current lambda
    ctr = ctr + R;
    
    idx_w = ctr + (1:(NS*R));
    w_ss_0 = reshape(H(idx_w), [], R); % current w
    w_ss = exp(w_ss_0);
    ctr = ctr + NS*R;
    
    idx_hc = ctr + (1:NH);
    H_ss = H(idx_hc); % current other hyperparams
    ctr = ctr + NH;
    
    scales_ss = tau .* phi(:)' .* w_ss;
    
    %% log p(U | tau, phi, w, H_c)
    if(compute_dH)
        [lp, dlp_u, dlp_scales, dlp_h] = DMC.priors.MDGDP_gaussianComponent(U, mu, scales_ss, H_ss, stimPrior_setup_ss);
    elseif(compute_dU)
        [lp, dlp_u] = DMC.priors.MDGDP_gaussianComponent(U, mu, scales_ss, H_ss, stimPrior_setup_ss);
    else
        lp = DMC.priors.MDGDP_gaussianComponent(U, mu, scales_ss, H_ss, stimPrior_setup_ss);
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
    
    if(compute_dH)
        dlp_scales_expTransform = dlp_scales .* scales_ss; 
        
        % d log p(U | tau, phi, w, H_c) / dw
        results_group.dH(idx_w) = results_group.dH(idx_w) + dlp_scales_expTransform(:);

        % d log p(U | tau, phi, w, H_c) / dtau
        results_group.dH(idx_tau) = results_group.dH(idx_tau) + sum(dlp_scales_expTransform, 'all'); % ???
    end
    
%     if(compute_dH_gibbs)
%         % d log p(U | tau, phi, w, H_c) / dphi
%         results_group.dH_gibbs = results_group.dH_gibbs + sum(dlp_scales .* w_ss .* tau, 1)';
%     end
    
    %% log p(w | lambda)
    %b_w  = lambda_ss(:)'.^2./2;
    %lp_w = -log(b_w) - w_ss ./ b_w + w_ss_0;
    lambda_inv_2 = exp(-2*lambda_ss_0(:)');
    lp_w = -(2 * lambda_ss_0(:)' - log(2)) - 2 * w_ss .* lambda_inv_2 + w_ss_0;
    results_group.log_prior_VT = results_group.log_prior_VT + sum(lp_w, 'all');
    
    if(compute_dH)
    
        % d log p(w | lambda) / dw
        dw = - 2*w_ss.*lambda_inv_2 + 1;
        results_group.dH(idx_w) = results_group.dH(idx_w) + dw(:);

        % d log p(w | lambda) / dlambda
        results_group.dH(idx_lambda) = results_group.dH(idx_lambda) -2*size(w_ss,1) + 4 * sum(w_ss ,1)'.* lambda_inv_2(:);
    end
    
    %% log p(lambda) 
    lp_lambda = R * (a_lambda * log(b_lambda) - gammaln(a_lambda)) - b_lambda * sum(lambda_ss) + sum(lambda_ss_0);
    results_group.log_prior_VT = results_group.log_prior_VT + lp_lambda;
    
    if(compute_dH)
        % d log p(lambda) dlambda -> dlambda
        results_group.dH(idx_lambda) = results_group.dH(idx_lambda) - b_lambda * lambda_ss + 1;
    end
     
    %% log p(H_ss) : assumes all are i.i.d. half-t distributions
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