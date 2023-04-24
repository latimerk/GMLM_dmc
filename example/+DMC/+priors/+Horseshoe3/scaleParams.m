function [params_group, precomputedPortion] = scaleParams(params_group, prior_setup)


H   = double(params_group.H(:));
D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

if(prior_setup.include_rank_constant)
    nd = sum([prior_setup.T(:).on]) + prior_setup.V.on;
    rank_scale = -log(R)/nd;
%     rank_scale = -log(1:R)/nd;
else
    rank_scale = 0;
end

%% gets regularization var
idx_log_c2 = 1;
log_c2 = H(idx_log_c2);

%% gets global tau
idx_tau = 2;
log_tau = H(idx_tau);

%% gets rank phi
idx_phi = 2 + (1:(R));
log_phi = reshape(H(idx_phi), 1, R);

%% goes through each piece
ctr = 2 + R;
precomputedPortion = struct("Z", cell(D,1), "log_unregularized_scale2", [], "log_scales_ss_0", []);
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        prior_setup_ss = prior_setup.V;

    else %T{ss-1}
        U = double(params_group.T{ss-1});
        prior_setup_ss = prior_setup.T(ss-1);
    end
    mu = prior_setup_ss.mu;

    %% if transformed
    if(isfield(prior_setup_ss, "U_transform") && ~isempty(prior_setup_ss.U_transform))
        U = prior_setup_ss.U_transform'*U;
    end

    %% get local scales
    if(prior_setup_ss.on)
        if(isempty(prior_setup_ss.grps))
            NS = prior_setup_ss.dim_T;
        else
            NS = numel(prior_setup_ss.grps);
        end

        idx_lambda = ctr + (1:(NS*R));
        if(NS == 0)
            log_lambda_ss = zeros(1, R);
        else
            log_lambda_ss = reshape(H(idx_lambda), [], R);
        end
        ctr = ctr + NS*R;

        % variance is: lambda_{i,r}^2 * phi_{r}^2 * tau^2 * c^2 / (c^2 + lambda_{i,r}^2 * phi_{r}^2 * tau^2)
   
        log_scale2_part          = 2*(prior_setup.log_constant_scale + log_phi + log_tau + log_lambda_ss + prior_setup_ss.lambda_log_scale(:) + rank_scale);
        log_scales2_regularized  = log_scale2_part + log_c2 - kgmlm.utils.logSumExp_pair(log_c2, log_scale2_part);
    
        Z = DMC.priors.Horseshoe3.scaledGaussianComponent(U, log_scales2_regularized, prior_setup_ss.grps);


        precomputedPortion(ss).log_unregularized_scale2 = log_scale2_part;
        precomputedPortion(ss).log_scales_ss_0 = log_scales2_regularized;
    else
        Z = 1;
    end
    precomputedPortion(ss).Z = Z;


    UZ = (U).*Z ;

    %% log p(U | tau, phi, w, H_c)

    if(ss == 1) % V
        params_group.V(:,:) = UZ + mu;
    else %T{ss-1}
        params_group.T{ss-1}(:,:) = UZ + mu;
    end
end