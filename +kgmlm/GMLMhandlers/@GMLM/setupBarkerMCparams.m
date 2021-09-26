function [MCMC_settings] = setupBarkerMCparams(obj, nWarmup, nTotal, debugSettings)
if(nargin < 4)
    debugSettings = false;
end
MCMC_settings.DEBUG = debugSettings;

if(nargin < 2 || isempty(nWarmup))
    nWarmup = 25e3;
    if(debugSettings)
        nWarmup = 2500;
    end
end
if(nargin < 3 || isempty(nTotal))
    nTotal  = 50e3 + nWarmup;
    if(debugSettings)
        nTotal = 7500 + nWarmup;
    end
end

%% covariance estimation
if(~debugSettings)
    if(nWarmup < 25e3)
        warning('HMC warmup schedule espected at least 25000 samples. Schedule will need to be modified.');
    end
    MCMC_settings.M_est.end = 20e3;
    MCMC_settings.verbose = false;
    MCMC_settings.showPlots = ispc;
else
    if(nWarmup < 2500)
        warning('HMC warmup schedule espected at least 2500 samples. Schedule will need to be modified.');
    end
    MCMC_settings.M_est.end = nWarmup - 500;
    MCMC_settings.verbose = true;
    MCMC_settings.showPlots = true;
end

%%
MCMC_settings.M_est.kappa = 0.8;
MCMC_settings.M_est.alpha = 0.4;

MCMC_settings.M_est.alpha = 0.4;
MCMC_settings.e_init = 1e-2;

MCMC_settings.N = 1;
MCMC_settings.M_est.gamma = 1;

MCMC_settings.nWarmup = nWarmup;
MCMC_settings.nTotal  = nTotal;

%% parameters for an MH step to quickly traverse the scaler part of each component of tensor
MCMC_settings.MH_scale.rho = -0.9;
MCMC_settings.MH_scale.sig =  0.2;

%%
MCMC_settings.stepSize.e_0 = 1e-1;
MCMC_settings.stepSize.delta  = 0.8;
MCMC_settings.stepSize.gamma = 0.05;
MCMC_settings.stepSize.kappa = 0.75;
MCMC_settings.stepSize.t_0   = 10;
MCMC_settings.stepSize.mu    = log(10*MCMC_settings.stepSize.e_0);
MCMC_settings.stepSize.schedule   = [2 MCMC_settings.M_est.end];

