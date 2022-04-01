%% setupHMCparams returns default setup for HMC
%    inputs (optional):
%       nWarmup  (default = 25000) : number of warmup samples (to be discarded later)
%       nSamples (default = 50000) : number of samples to take post warmup
%       debugSettings (default = false) : if true, changes settings for a debug run (fewer samples)
%
function [HMC_settings] = setupHMCparams(~, nWarmup, nSamples, debugSettings)
if(nargin < 4)
    debugSettings = false;
end
HMC_settings.DEBUG = debugSettings;

%%
nWarmup_default = 25e3;

if(nargin < 2 || isempty(nWarmup))
    nWarmup = nWarmup_default;
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


%step size paramters
    %for the dual-averging updates
HMC_settings.stepSize.e_0 = 1e-4;
HMC_settings.stepSize.delta  = [0.6; 0.9];
HMC_settings.stepSize.gamma = 0.05;
HMC_settings.stepSize.kappa = 0.75;
HMC_settings.stepSize.t_0   = 10;
HMC_settings.stepSize.mu    = log(10*HMC_settings.stepSize.e_0);
HMC_settings.stepSize.max_step_size = 0.25;
    % randomly scales step size (make smaller somtimes)

HMC_settings.stepSize.scales   = [1    0.5  0.1];  %scales
HMC_settings.stepSize.P_scales = [0.85 0.1  0.05];  %P(scale): must sum to 1

HMC_settings.stepSize.scaleRanges = nWarmup + [1 nSamples];

HMC_settings.M_const = 1;


if(~debugSettings)
    if(nWarmup < nWarmup_default)
        warning('Default HMC warmup schedule espected at least %d samples. Schedule will need to be modified.', nWarmup_default);
    end
    
    if(nWarmup >= 50e3)
        HMC_settings.M_est.first_sample  = [13001 33001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [23000 48000];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  12000;
                                            23001  30000;
                                            30001  32000;
                                            48001  nWarmup]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [12001 23000;
                                             32001 48000;
                                             nWarmup + [1 nSamples]];
    else
        HMC_settings.M_est.first_sample  = [8001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [23000];
        
        HMC_settings.stepSize.schedule   = [2      5000;
                                            5001   6000;
                                            23001  25000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [6001  23000;
                                             nWarmup + [1 nSamples]];
    end
    

    HMC_settings.trialLLfile = [TempFolder 'tmp_GMLM_samples.dat'];
    HMC_settings.samplesBlockSize = 1e3;
    
    HMC_settings.verbose   = false;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 1.0; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = 100; %max number of steps per sample
    HMC_settings.stepSize.maxSteps_trial = [];
     
%     sM_stepL = 0.5;
%     sM_maxSteps = 50;
    
    
    
    HMC_settings.fitMAP = [];%[100 1100]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
    
else
    if(nWarmup < 1500)
        warning('Default HMC warmup schedule (DEBUG) espected at least 1500 samples. Schedule will need to be modified.');
    end
    
    HMC_settings.M_est.first_sample = [101  501];
    HMC_settings.M_est.samples      = [500 1000];
    
    
    HMC_settings.stepSize.schedule   = [2 500; 
                                        501 1400];
                                    
    HMC_settings.trialLLfile = [TempFolder 'tmp_GMLM_samples_DEBUG.mat'];
    HMC_settings.samplesBlockSize = 200;
    
    HMC_settings.verbose   = true;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 0.5; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = [100;50]; %max number of steps per sample
    HMC_settings.stepSize.maxSteps_trial = [100;inf];
    HMC_settings.fitMAP = [100 1000]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
end

%%


HMC_settings.nWarmup  = nWarmup;
HMC_settings.nSamples = nSamples;



