function [params_group] = getPriorVariance(params_group, prior_setup)


H   = double(params_group.H(:));

D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

NS_total = 0;
for ss = 1:D
    if(ss == 1) %
        NS_total = NS_total + prior_setup.V.N_scales;   
    else %T{ss-1}
        NS_total = NS_total + prior_setup.T(ss-1).N_scales;      % number of scales
    end
end
includeTau = NS_total > 0;

if(includeTau)
    if(prior_setup.regularized)
        idx_c2 = 1;
        log_c2 = H(idx_c2);
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
    
    
    if(prior_setup.crossrankwise_psi && R > 1)
        idx_psi = ctr + (1:NS_total);
        log_psi = H(idx_psi);
        ctr = ctr + NS_total;
    else
        log_psi = 0;
    end
else
    log_tau = 0;
    log_psi = 0;
    log_c2 = 0;
end

ctr_NS = 0;

for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        mu = prior_setup.V.mu;

        NS = prior_setup.V.N_scales;      % number of scales
        NH = prior_setup.V.N_hyperparams; % number of extra hyperparams in this set
        prior_setup_ss = prior_setup.V.setup;
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        mu = prior_setup.T(ss-1).mu;
        NS = prior_setup.T(ss-1).N_scales;      % number of scales
        NH = prior_setup.T(ss-1).N_hyperparams; % number of extra hyperparams in this set
        prior_setup_ss = prior_setup.T(ss-1).setup;
    end
    if(isa(NH, 'function_handle'))
        NH = NH(R);
    end

    if(NS > 0 && prior_setup.rankwise_phi && R > 1)
        idx_phi = ctr + (1:(R));
        log_phi = reshape(H(idx_phi), [], R);
        ctr = ctr + R;
    else
        log_phi = 0;
    end
    idx_lambda = ctr + (1:(NS*R));
    log_lambda_ss = reshape(H(idx_lambda), [], R);
    %lambda_ss = DMC.utils.boundedExp(log_lambda_ss); % current lambda
    %tau2lambda2 = DMC.utils.boundedExp(2*(log_lambda_ss + log_tau' + log_phi));
    
    ctr = ctr + NS*R;

    idx_hc = ctr + (1:NH);
    H_ss = H(idx_hc); % current other hyperparams
    ctr = ctr + NH;


    if(prior_setup.crossrankwise_psi && R > 1)
        idx_psi_c = (1:NS) + ctr_NS;
        ctr_NS = ctr_NS + NS;
    else
        idx_psi_c = 1;
    end

    if(NS > 0)
        if(prior_setup.regularized)
        %     scales_ss_0 = 2*log_tau' + log_c2 + 2*log_lambda_ss - log(c2 + tau2lambda2);
            scales_ss_0 = 2*log_tau' + 2*log_lambda_ss + 2*log_phi + 2*log_psi(idx_psi_c) - log1p(DMC.utils.boundedExp(2*(log_lambda_ss + log_tau' + log_phi + log_psi(idx_psi_c) - 0.5*log_c2)));
        else
            scales_ss_0 = 2*log_tau' + 2*log_lambda_ss + 2*log_phi + 2*log_psi(idx_psi_c);
        end
    else
        scales_ss_0 = [];
    end


    %% log p(U | tau, phi, w, H_c)
    [~, ~, ~, ~, U] = DMC.priors.scaledGaussianComponent(U, mu, scales_ss_0, H_ss, prior_setup_ss);

    if(ss == 1) % V
        params_group.V(:) = U;
    else %T{ss-1}
        params_group.T{ss-1}(:) = U;
    end
end