function [H] = generateH(R, prior_setup, generate_with_t, use_log_scale_for_mean)

if(nargin < 3)
    generate_with_t = false;
end
if(nargin < 4)
    use_log_scale_for_mean = false;
end

if(prior_setup.include_rank_constant)
    nd = sum([prior_setup.T(:).on]) + prior_setup.V.on;
    rank_scale = -log(R)/nd;
else
    rank_scale = 0;
end


dim_H = DMC.priors.Horseshoe3.getDimH(R, prior_setup);

H = zeros(dim_H,1);
D = numel(prior_setup.T) + 1; % order


% parameters
% c2    - controls slab size (regularized horseshoe)  inv-gamma(a_c, b_c)
idx_c2 = 1;
H(idx_c2) = -log(gamrnd(prior_setup.c.a, 1./prior_setup.c.b));

% tau - global scale- (t+_{df_tau}(0, tau_0^2))
idx_tau2 = 2;
if(generate_with_t)
    H(idx_tau2) = log(abs(trnd(prior_setup.tau.dfs) * exp(prior_setup.tau.log_scale)));
else
    t_std = 0.5;
    if(use_log_scale_for_mean)
        t_mu  = prior_setup.tau.log_scale;
    else
        t_mu  = -prior_setup.log_constant_scale;
    end
    H(idx_tau2) = randn* t_std + t_mu;
end

% phi_r - scales per rank - (t+_{df_phi}(0, tau_0^2))
idx_phi2 = 2 + (1:R);
p_std = 0.5;
if(use_log_scale_for_mean)
    p_mu  = prior_setup.phi.log_scale ;
else
    p_mu  = -rank_scale;
end
H(idx_phi2) = randn(R,1)* p_std + p_mu;

%% goes through each piece
ctr = 2 + R;
for ss = 1:D
    if(ss == 1) % V
        prior_setup_ss = prior_setup.V;
    else %T{ss-1}
        prior_setup_ss = prior_setup.T(ss-1);
    end

    if(prior_setup_ss.on)
        if(isempty(prior_setup_ss.grps))
            NS = prior_setup_ss.dim_T;
        else
            NS = numel(prior_setup_ss.grps);
        end

        idx_lambda = ctr + (1:(NS*R));

        % lambda_r,j - local scales - (t+_{df_lambda}(0, 1))
        if(generate_with_t)
            H(idx_lambda) = log(abs(trnd(prior_setup.lambda.dfs, [NS*R 1])));
        else
            l_std = 0.5;
            if(use_log_scale_for_mean)
                l_mu  = -prior_setup_ss.lambda_log_scale;
            else
                l_mu  =  0;
            end
            ll = randn(NS,R) * l_std + l_mu;
            H(idx_lambda) = ll(:);
        end
    end
end

end
