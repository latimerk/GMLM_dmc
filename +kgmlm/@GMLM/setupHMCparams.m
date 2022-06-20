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
HMC_settings.stepSize.e_0 = 1e-3;
HMC_settings.stepSize.delta  = [0.8; 0.9];
HMC_settings.stepSize.gamma = 0.05;
HMC_settings.stepSize.kappa = 0.75;
HMC_settings.stepSize.t_0   = 10;
HMC_settings.stepSize.mu    = log(10*HMC_settings.stepSize.e_0);
HMC_settings.stepSize.max_step_size = 0.25;
    % randomly scales step size (make smaller somtimes)

HMC_settings.stepSize.scales   = [1    0.5   0.1];  %scales
HMC_settings.stepSize.P_scales = [0.9  0.06  0.04];  %P(scale): must sum to 1

HMC_settings.stepSize.scaleRanges = nWarmup + [1 nSamples];

HMC_settings.M_const = 1;


if(~debugSettings)
    nWarmup_min = 15e3;
    if(nWarmup < nWarmup_min)
        warning('Default HMC warmup schedule espected at least %d samples. Schedule will need to be modified.', nWarmup_min);
    end
    if(nWarmup >= 60e3)
        HMC_settings.M_est.first_sample  = [25001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [55000];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  25000;
                                            55001  60000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [25001  55000;
                                             nWarmup + [1 nSamples]];
    elseif(nWarmup >= 50e3)
        HMC_settings.M_est.first_sample  = [25001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [45000];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  25000;
                                            45001  50000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [25001  45000;
                                             nWarmup + [1 nSamples]];

    elseif(nWarmup >= 40e3)
        HMC_settings.M_est.first_sample  = [15001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [35000];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  15000;
                                            35001  40000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [15001  35000;
                                             nWarmup + [1 nSamples]];
    elseif(nWarmup >= 35e3)
        HMC_settings.M_est.first_sample  = [20001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [32000];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  20000;
                                            32001  3500]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [20001  32000;
                                             nWarmup + [1 nSamples]];
    elseif(nWarmup >= 30e3)
%         HMC_settings.M_est.first_sample  = [12001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
%         HMC_settings.M_est.samples       = [27000];
%         
%         HMC_settings.stepSize.schedule   = [2       6000;
%                                             12001  12000;
%                                             27001  30000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
%     
%         HMC_settings.stepSize.scaleRanges = [12001  27000;
%                                              nWarmup + [1 nSamples]];

        
        HMC_settings.M_est.first_sample  = []; 
        HMC_settings.M_est.samples       = [];
        
        HMC_settings.stepSize.schedule   = [2      15000;
                                            15001  25000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [25001 nSamples];
    elseif(nWarmup >= 25e3)
        HMC_settings.M_est.first_sample  = [10001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [22000];
        
        HMC_settings.stepSize.schedule   = [2      4000;
                                            4001   7000;
                                            22001  25000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [7001  22000;
                                             nWarmup + [1 nSamples]];
    elseif(nWarmup >= 20e3)
        
        HMC_settings.M_est.first_sample  = []; 
        HMC_settings.M_est.samples       = [];
        
        HMC_settings.stepSize.schedule   = [2      10000;
                                            10001  18000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [18001 nSamples];
    else
        HMC_settings.M_est.first_sample  = [ 8001]; %when to estimate cov matrix. At sample=samples(ii), will use first_sample(samples(ii)):sample
        HMC_settings.M_est.samples       = [13000];
        
        HMC_settings.stepSize.schedule   = [2       5000;
                                             5001  8000;
                                            13001  15000]; %each row gives a range of trials to estimate step size (restarts estimation at each sample = schedule(ii,1))
    
        HMC_settings.stepSize.scaleRanges = [8001  13000;
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


if(isempty(HMC_settings.M_est.samples))
    warning("HMC settings will only use the initial mass matrix - no estimation of the posterior covariance.\n");
end
