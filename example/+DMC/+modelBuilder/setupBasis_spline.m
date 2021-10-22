%% creates basis functions for modeling the DMC task
%  
%
%  Uses modified cardinal splines from
%   Sarmashghi, M., Jadhav, S. P., & Eden, U. (2021). Efficient Spline Regression for Neural Spiking Data. bioRxiv.
%
%
function [bases] = setupBasis_spline(binSize_ms, stimLength_ms, delayPeriod_ms, highResPostStim_on_ms, highResPostStim_off_ms)
if(nargin < 1 || isempty(binSize_ms))
    binSize_ms = 5;
end
if(nargin < 2 || isempty(stimLength_ms))
    stimLength_ms = 550;
end
if(nargin < 3 || isempty(delayPeriod_ms))
    delayPeriod_ms = 1200;
end
if(nargin < 4 || isempty(delayPeriod_ms))
    delayPeriod_ms = 1200;
end
if(nargin < 5 || isempty(highResPostStim_on_ms))
    highResPostStim_on_ms = 200;
end
if(nargin < 6 || isempty(highResPostStim_off_ms))
    highResPostStim_off_ms = 200;
end
   
bases.binSize_ms = binSize_ms;
dd = 1;

%% spike history basis
bases.spkHist.s    = []; % empty = default
bases.spkHist.c_pt = [0 2 5 10 20 40 60 80 120 160 240 320 400+dd];
bases.spkHist.lag  = 400;
bases.spkHist.zeroEndPoints = [false false];

[bases.spkHist.B, bases.spkHist.B_0, bases.spkHist.tts_0, bases.spkHist.tts] = DMC.modelBuilder.ModifiedCardinalSpline(bases.spkHist.lag, bases.spkHist.c_pt, bases.spkHist.s, binSize_ms, bases.spkHist.zeroEndPoints);


%%  stimulus filter basis
bases.stim.s    = []; % empty = default

high_res_delta = 25;
low_res_delta = 50;
stimFilter_end = stimLength_ms*2 + delayPeriod_ms + 50;

bases.stim.c_pt = [0:high_res_delta:highResPostStim_on_ms ...
                 (highResPostStim_on_ms):low_res_delta:(stimLength_ms) ...
                 stimLength_ms:high_res_delta:(stimLength_ms + highResPostStim_off_ms) ...
                 (stimLength_ms + highResPostStim_off_ms):low_res_delta:stimFilter_end];
bases.stim.c_pt = unique(bases.stim.c_pt);
             
bases.stim.c_pt(end) = bases.stim.c_pt(end) + dd;
bases.stim.c_pt(1)   = bases.stim.c_pt(1)   - dd;
bases.stim.lag  = [0 stimFilter_end];
bases.stim.zeroEndPoints = [false false];

[bases.stim.B, bases.stim.B_0, bases.stim.tts_0, bases.stim.tts] = DMC.modelBuilder.ModifiedCardinalSpline(bases.stim.lag, bases.stim.c_pt, bases.stim.s, binSize_ms, bases.stim.zeroEndPoints);


%% responses/touch bar
bases.response.s       = []; % empty = default
bases.response.c_pt    = [-300-dd -250 -200 -150 -100 -60 -40 -20 0 20 40 50+dd];
bases.response.window  = [-300 50];
bases.response.zeroEndPoints = [false false];

[bases.response.B, bases.response.B_0, bases.response.tts_0, bases.response.tts] = DMC.modelBuilder.ModifiedCardinalSpline(bases.response.window, bases.response.c_pt, bases.response.s, binSize_ms, bases.response.zeroEndPoints);


