function [H] = generateH(R, prior_setup)

dim_H = DMC.priors.Horseshoe.getDimH(R, prior_setup);

H = zeros(dim_H,1);
D = numel(prior_setup.T) + 1; % order

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

min_df = 4;
df_global = max(min_df, prior_setup.hyperparams.df_global); % default: 3
df_local  = max(min_df, prior_setup.hyperparams.df_local);  % default: 3


if(includeTau)
    %% get global params
    if(prior_setup.regularized)
        idx_c2 = 1;
        H(idx_c2) = -log(gamrnd(a_c, 1./b_c));
    else
        idx_c2 = 0;
    end
    
    if(prior_setup.rankwise_tau)
        idx_tau = idx_c2 + (1:R);
    else
        idx_tau = idx_c2 + 1;
    end
    ctr = max(idx_tau);

    H(idx_tau) = log(abs(trnd(df_global, [numel(idx_tau) 1]) * t_0));

    if(prior_setup.crossrankwise_psi && R > 1)
        idx_psi = ctr + (1:NS_total);
        H(idx_psi) = log(abs(trnd(df_global, [NS_total 1])));
        ctr = ctr + NS_total;
    end

end


%% goes through each piece
for ss = 1:D
    if(ss == 1) % V
        NS = prior_setup.V.N_scales;      % number of scales
        NH = prior_setup.V.N_hyperparams; % number of extra hyperparams in this set
    else %T{ss-1}
        NS = prior_setup.T(ss-1).N_scales;      % number of scales
        NH = prior_setup.T(ss-1).N_hyperparams; % number of extra hyperparams in this set
    end
    if(isa(NH, 'function_handle'))
        NH = NH(R);
    end
    
    if(NS > 0)
        if(prior_setup.rankwise_phi && R > 1)
            idx_phi = ctr + (1:(R));
            H(idx_phi) = abs(trnd(df_global, [R 1]));
            ctr = ctr + R;
        end
        idx_lambda = ctr + (1:(NS*R));
        H(idx_lambda) = abs(trnd(df_local, [NS*R 1]));
        ctr = ctr + NS*R;
    end
    
    
    idx_hc = ctr + (1:NH);
    if(NH > 0)
        H(idx_hc) = abs(trnd(max(min_df, prior_setup.hyperparams.H.nu), [NH 1]));
    end
    ctr = ctr + NH;
end

end
