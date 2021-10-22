% Samples the phis (dirichlet mixing weights) in the M-DGDP prior (plus tau)
% Also calls the rescaling MH step, because why not?
%
% I chose to use the Gibbs step for this parameter because of HMC/GeodesicMC having problems sampling from Dirichlet distributions with tiny alpha (which is the
% case we want here). I don't know if this same problem would occur sampling from a posterior.
% This is Step 1(b) in section 5.1 of Guhaniyogi et al (2017)
function [params] = MDGDP_samplePhi(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, stimPrior_setup)

params_group = params.Groups(groupNum);
H   = double(params_group.H(:));

D = numel(params_group.T) + 1; % order
R = size(params_group.V,2); % rank

p_alpha_beta = stimPrior_setup.hyperparams.alpha.is_beta; % if is a beta distribution or gamma

idx_tau   = 1;
idx_alpha = 2;

if(p_alpha_beta)
    alpha = 1./(1+exp(-H(idx_alpha))); 
else
    alpha_0 = H(idx_alpha);
    alpha = exp(alpha_0); 
end

nu = 1;
a_tau = R*alpha;
b_tau = alpha*(R/nu)^(1/D);

%% compute C_r for each rank
ctr = 2;
C_rj = nan(R, D);
for ss = 1:D
    if(ss == 1) % V
        U = double(params_group.V);
        mu = stimPrior_setup.V.mu;
        
        NS = stimPrior_setup.V.N_scales;      % number of scales
        NH = stimPrior_setup.V.N_hyperparams; % number of extra hyperparams in this set
        stimPrior_setup_ss = stimPrior_setup.V.setup;
        
    else %T{ss-1}
        U = double(params_group.T{ss-1});
        mu = stimPrior_setup.T(ss-1).mu;
        NS = stimPrior_setup.T(ss-1).N_scales;      % number of scales
        NH = stimPrior_setup.T(ss-1).N_hyperparams; % number of extra hyperparams in this set
        stimPrior_setup_ss = stimPrior_setup.T(ss-1).setup;
    end
    
    idx_w = ctr + (1:(NS*R));
    w_ss_0 = reshape(H(idx_w), [], R); % current w
    w_ss = exp(w_ss_0);
    ctr = ctr + NS*R;
    
    idx_hc = ctr + (1:NH);
    H_ss = H(idx_hc); % current other hyperparams
    ctr = ctr + NH;
    
    C_rj(:, ss) = DMC.priors.MDGDP_gaussianComponent(U, mu, w_ss, H_ss, stimPrior_setup_ss, true);
end

p_0 = gmlm.dim_P + sum(gmlm.dim_T(groupNum));
C_r = sum(C_rj, 2);

%% sample phi
psi_r = nan(R, 1);
for rr = 1:R
    psi_r(rr) = DMC.utils.gigrnd(alpha - p_0/2, 2 * b_tau, 2 * C_r(rr));
end
phi = psi_r ./ sum(psi_r);
%phi(:) = 1 ./ R;
params.Groups(groupNum).H_gibbs(:) = phi;

%% sample tau
D_r = C_r ./ phi;
tau = DMC.utils.gigrnd( a_tau - R*p_0/2, 2 * b_tau, 2 * sum(D_r));
params.Groups(groupNum).H(idx_tau) = log(tau); % can't forget the log tranform because this part can also be sampled by HMC


%% rescaling step
% params = DMC.GibbsSteps.scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, stimPrior_setup);

