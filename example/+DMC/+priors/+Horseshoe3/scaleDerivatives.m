function [results_group] = scaleDerivatives(results_group, params_group, prior_setup, addPriorForVT, precomputedPortion)


H = double(params_group.H(:));
D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

compute_dH = isfield(results_group, 'dH') && ~isempty(results_group.dH);


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
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        dU_0 = double(results_group.dV);
        prior_setup_ss = prior_setup.V;
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        dU_0 = double(results_group.dT{ss-1});
        prior_setup_ss = prior_setup.T(ss-1);
    end


    %% if transformed
    if(isfield(prior_setup_ss, "U_transform") && ~isempty(prior_setup_ss.U_transform) && ~isempty(dU_0))
        dU_2 = prior_setup_ss.U_transform*dU_0;
        U2 = prior_setup_ss.U_transform'*U;
    else
        dU_2 = dU_0;
        U2 = U;
    end


    %% get local scales
    if(prior_setup_ss.on)
    
        if(isempty(prior_setup_ss.grps))
            NS = prior_setup_ss.dim_T;
        else
            NS = numel(prior_setup_ss.grps);
        end

        idx_lambda = ctr + (1:(NS*R));
        ctr = ctr + NS*R;


        compute_derivative = compute_dH && (NS > 0);
   
        if(nargin < 5 || isempty(precomputedPortion) || isempty(precomputedPortion(ss).log_scales_ss_0))
            if(NS == 0)
                log_lambda_ss = zeros(1, R);
            else
                log_lambda_ss = reshape(H(idx_lambda), [], R);
            end
            
            % variance is: lambda_{i,r}^2 * phi_{r}^2 * tau^2 * c^2 / (c^2 + lambda_{i,r}^2 * phi_{r}^2 * tau^2)
            log_scale2_part         = 2*(prior_setup.log_constant_scale + log_phi + log_tau + log_lambda_ss + prior_setup_ss.lambda_log_scale(:) + rank_scale);
            log_scales2_regularized = log_scale2_part + log_c2 - kgmlm.utils.logSumExp_pair(log_c2, log_scale2_part);

            if(compute_derivative)
                [Z, dlp_scales] = DMC.priors.Horseshoe3.scaledGaussianComponent(U2, log_scales2_regularized, prior_setup_ss.grps, dU_0);
            else
                Z = DMC.priors.Horseshoe3.scaledGaussianComponent(U2, log_scales2_regularized, prior_setup_ss.grps);
            end

        else
            Z = precomputedPortion(ss).Z;
            if(compute_derivative)
                log_scale2_part = precomputedPortion(ss).log_unregularized_scale2;
                log_scales2_regularized = precomputedPortion(ss).log_scales_ss_0;
                [~, dlp_scales] = DMC.priors.Horseshoe3.scaledGaussianComponent(U2, log_scales2_regularized, prior_setup_ss.grps, dU_0, Z);
            end
        end
        

        %% log p(U | tau, phi, w, H_c)
        if(compute_derivative)

            dscales_ss_0_dc2  = 1 - 1./(1 + DMC.utils.boundedExp(log_scale2_part - log_c2));
            results_group.dH(idx_log_c2) = results_group.dH(idx_log_c2) + sum(dlp_scales.*dscales_ss_0_dc2, 'all');

            dscales_ss_0_dnum = 2 - 2./(1 + DMC.utils.boundedExp(log_c2 - log_scale2_part));
            dn = dlp_scales.*dscales_ss_0_dnum;
            
            results_group.dH(idx_tau) = results_group.dH(idx_tau) + sum(dn, "all");
            results_group.dH(idx_phi) = reshape(results_group.dH(idx_phi), [], R) + sum(dn, 1);
            results_group.dH(idx_lambda) = reshape(results_group.dH(idx_lambda), [], R) + dn;
        
        end
    else
        Z = 1;

    end

    if(addPriorForVT)
        dlp_u = -U;
    end




    if(ss == 1) % V
        if(~isempty(results_group.dV))
            dUZ = dU_2.*Z;
            if(addPriorForVT) % prior for unscaled U is i.i.d. standard normal
                results_group.dV(:,:) = dlp_u + dUZ; %DERIVATIVE ADDED HERE BECAUSE OF SCALING
            else
                results_group.dV(:,:) = dUZ; 
            end
        end
    else %T{ss-1}
        if(~isempty(results_group.dT{ss-1}))
            dUZ = dU_2.*Z;
            if(addPriorForVT)
                results_group.dT{ss-1}(:,:) = dlp_u + dUZ;
            else
                results_group.dT{ss-1}(:,:) =  dUZ;
            end
        end
    end
end