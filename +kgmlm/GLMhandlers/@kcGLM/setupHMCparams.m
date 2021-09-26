function [HMC_settings] = setupHMCparams(obj, nWarmup, nTotal, debugSettings)
if(nargin < 4)
    debugSettings = false;
end
HMC_settings.DEBUG = debugSettings;

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
    
    HMC_settings.M_est.first_sample  = [2001 4001 ];
    HMC_settings.M_est.samples       = [4000 19001];
    HMC_settings.M_est.diagOnly      = [true true];
    HMC_settings.M_est.diagOnly_hyper = [true true];
    
    HMC_settings.stepSize.schedule   = [2 4001 19001 24000];
    
    HMC_settings.verbose = false;
    HMC_settings.showPlots = false;
    
    %max number of steps per sample
    HMC_settings.HMC_step.maxSteps      = 100;
    %early phase sampling: may allow more steps
    HMC_settings.HMC_step.nSamples_init = 2000;
    HMC_settings.HMC_step.maxSteps_init = 100;
else
    if(nWarmup < 2500)
        warning('HMC warmup schedule espected at least 2500 samples. Schedule will need to be modified.');
    end
    
    HMC_settings.M_est.first_sample = [501  501];
    HMC_settings.M_est.samples      = [1000 2000];
    HMC_settings.M_est.diagOnly       = [true true];
    HMC_settings.M_est.diagOnly_hyper = [true true];
    
    HMC_settings.stepSize.schedule   = [2 1001 2001 2400];
    
    HMC_settings.verbose = true;
    HMC_settings.showPlots = true;
    
    HMC_settings.HMC_step.nSamples_init = 100;
    HMC_settings.HMC_step.maxSteps_init = 100;
    HMC_settings.HMC_step.maxSteps  = 100;
end
    
%%

HMC_settings.HMC_step.stepL     = 1.0;

HMC_settings.stepSize.e_0 = 1e-1;
HMC_settings.stepSize.delta  = 0.8;
HMC_settings.stepSize.gamma = 0.05;
HMC_settings.stepSize.kappa = 0.75;
HMC_settings.stepSize.t_0   = 10;
HMC_settings.stepSize.mu    = log(10*HMC_settings.stepSize.e_0);

HMC_settings.nWarmup = nWarmup;
HMC_settings.nTotal  = nTotal;