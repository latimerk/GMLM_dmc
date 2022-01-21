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


%step size paramters
    %for the dual-averging updates
HMC_settings.stepSize.e_0 = 1e-4;
HMC_settings.stepSize.delta  = 0.8; %0.8
HMC_settings.stepSize.gamma = 0.05;
HMC_settings.stepSize.kappa = 0.75;
HMC_settings.stepSize.t_0   = 10;
HMC_settings.stepSize.mu    = log(10*HMC_settings.stepSize.e_0);
HMC_settings.stepSize.max_step_size = 0.25;
    % randomly scales step size (make smaller somtimes)

HMC_settings.stepSize.scales   = [1    0.5  0.1];  %scales
HMC_settings.stepSize.P_scales = [0.85 0.1  0.05];  %P(scale): must sum to 1
HMC_settings.stepSize.scaleDuringWarmup = false;


HMC_settings.stepSize_sM = HMC_settings.stepSize;
HMC_settings.stepSize_sH = HMC_settings.stepSize;
HMC_settings.sample_M_setScale = true;

if(~debugSettings)
    if(nWarmup < 25e3)
        warning('Default HMC warmup schedule espected at least 25000 samples. Schedule will need to be modified.');
    end
    
    HMC_settings.M_est.first_sample  = [4001  10001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
    HMC_settings.M_est.samples       = [9000 20000];
    
    HMC_settings.stepSize.schedule   = [2      9000;
                                        9001  24900]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    HMC_settings.stepSize_sM.schedule   = HMC_settings.stepSize.schedule;
    HMC_settings.stepSize_sH.schedule   = HMC_settings.stepSize.schedule;
    
    HMC_settings.samplesFile = [TempFolder 'tmp_GMLM_samples.dat'];
    HMC_settings.samplesBlockSize = 1e3;
    
    HMC_settings.verbose   = false;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 1.0; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = 10; %max number of steps per sample
     
    sM_stepL = HMC_settings.stepSize.stepL;
    sM_maxSteps = HMC_settings.stepSize.maxSteps;
%     sM_stepL = 0.5;
%     sM_maxSteps = 50;
    
    HMC_settings.stepSize_sM.stepL     = sM_stepL; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize_sM.maxSteps  = sM_maxSteps; %max number of steps per sample
    HMC_settings.stepSize_sH.stepL     = sM_stepL; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize_sH.maxSteps  = sM_maxSteps; %max number of steps per sample

    
    
    HMC_settings.fitMAP = [];%[100 1100]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
    
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
    HMC_settings.stepSize_sM.schedule   = HMC_settings.stepSize.schedule;
    HMC_settings.stepSize_sH.schedule   = HMC_settings.stepSize.schedule;
                                    
    HMC_settings.samplesFile = [TempFolder 'tmp_GMLM_samples_DEBUG.mat'];
    HMC_settings.samplesBlockSize = 200;
    
    HMC_settings.verbose   = true;
    
    
    %step size paramters
    HMC_settings.stepSize.stepL     = 0.5; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize.maxSteps  = 50; %max number of steps per sample
    HMC_settings.stepSize_sM.stepL     = 0.2; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize_sM.maxSteps  = 20; %max number of steps per sample
    HMC_settings.stepSize_sH.stepL     = 0.2; %total steps to take is min(maxSteps , ceil(stepL/e))
    HMC_settings.stepSize_sH.maxSteps  = 20; %max number of steps per sample
    HMC_settings.fitMAP = [100 1000]; % samples to fit MAP estimate (current parameter sample as init point, fixing current hyperparam sample). May help speed up mixing(?)
end

%%


HMC_settings.nWarmup  = nWarmup;
HMC_settings.nSamples = nSamples;


