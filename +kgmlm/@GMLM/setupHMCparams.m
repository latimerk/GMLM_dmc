%% setupHMCparams returns default setup for HMC
%    inputs (optional):
%       nWarmup  (default = 25000) : number of warmup samples (to be discarded later)
%       nSamples (default = 50000) : number of samples to take post warmup
%       debugSettings (default = false) : if true, changes settings for a debug run (fewer samples)
%
function [HMC_settings] = setupHMCparams(obj, nWarmup, nSamples, debugSettings)
if(nargin < 4)
    debugSettings = false;
end
HMC_settings.DEBUG = debugSettings;

if(nargin < 2 || isempty(nWarmup))
    nWarmup = 25e3;
    if(debugSettings)
        nWarmup = 1500;
    end
end
if(nargin < 3 || isempty(nSamples))
    nSamples  = 50e3;
    if(debugSettings)
        nSamples = 500;
    end
end

TempFolder = 'TempData/';
if(~isfolder(TempFolder))
    mkdir(TempFolder);
end
if(~debugSettings)
    if(nWarmup < 25e3)
        warning('Default HMC warmup schedule espected at least 25000 samples. Schedule will need to be modified.');
    end
    
    HMC_settings.M_est.first_sample  = [2001 4001 ]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
    HMC_settings.M_est.samples       = [4000 19001];
    
    HMC_settings.stepSize.schedule   = [2     4000;
                                        4001 24000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
    HMC_settings.samplesFile = [TempFolder 'tmp_GMLM_samples.mat'];
    HMC_settings.samplesBlockSize = 1e3;
    
    HMC_settings.verbose   = false;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 1.0; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = 100; %max number of steps per sample
    
    HMC_settings.fitMAP = [100 500 900]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
    
else
    if(nWarmup < 1500)
        warning('Default HMC warmup schedule (DEBUG) espected at least 1500 samples. Schedule will need to be modified.');
    end
    
    HMC_settings.M_est.first_sample = [101  501];
    HMC_settings.M_est.samples      = [500 1000];
    HMC_settings.M_est.diagOnly       = [true true];
    HMC_settings.M_est.diagOnly_hyper = [true true];
    
    HMC_settings.stepSize.schedule   = [2 500; 
                                        501 1400];
    
    HMC_settings.samplesFile = [TempFolder 'tmp_GMLM_samples_DEBUG.mat'];
    HMC_settings.samplesBlockSize = 200;
    
    HMC_settings.verbose   = true;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 0.5; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = 50; %max number of steps per sample
    HMC_settings.fitMAP = [50 250 450]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
end

%%
%step size paramters
    %for the dual-averging updates
HMC_settings.stepSize.e_0 = 1e-1;
HMC_settings.stepSize.delta  = 0.8;
HMC_settings.stepSize.gamma = 0.05;
HMC_settings.stepSize.kappa = 0.75;
HMC_settings.stepSize.t_0   = 10;
HMC_settings.stepSize.mu    = log(10*HMC_settings.stepSize.e_0);


HMC_settings.nWarmup  = nWarmup;
HMC_settings.nSamples = nSamples;


