%  Package GMLM_dmc for dimensionality reduction of neural data.
%   
%  References
%   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
%   decisions in parietal cortex reflects long-term training history.
%   bioRxiv
%
%  Copyright (c) 2021 Kenneth Latimer
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
function [basisStruct] = setupBasis(varargin)
%% sets up the basis sets for the DMC task
%     input key, val pairs
%         bins_post_lever = number of time bins for the lever filter basis to cover post release (default in DMC.modelBuilder.getDefaultTrialSettings)
%         delta_t         = bin size in SECONDS (default in DMC.modelBuilder.getDefaultTrialSettings)
%         stimBasis_len   = stim basis length in MILLISECONDS (why the different units? I don't know) default = 1500 ms
%         stimBasis_N     = number of stim basis vectors (default = 24)
%         plotBases       = true/false to plot the bases for review (default = false)


p = inputParser;
p.CaseSensitive = false;

% set the desired and optional input arguments
[~,bins_post_lever_default, delta_t_default] = DMC.modelBuilder.getDefaultTrialSettings([], false, false, false);
addParameter(p, 'bins_post_lever', bins_post_lever_default, @isnumeric);
addParameter(p, 'delta_t', delta_t_default, @isnumeric);
addParameter(p, 'plotBases', false, @islogical)
addParameter(p, 'stimBasis_len', 1500, @(aa)(isnumeric(aa) & aa > 0))
addParameter(p, 'stimBasis_N', 24, @(aa)(isnumeric(aa) & aa > 0))


% parse the input
parse(p,varargin{:});
% then set/get all the inputs out of this structure
bins_post_lever     = p.Results.bins_post_lever;
delta_t             = p.Results.delta_t;
plotBases           = p.Results.plotBases;
TT                  = p.Results.stimBasis_len;
N                   = p.Results.stimBasis_N;

%delta_t is in s
downsampleRate = delta_t/1e-3;

%% gets stimulus bases

if(nargin < 4 || isempty(TT))
    TT = 1500; %peaks of cosine basis will be between 0 and TT (in milliseconds)
end
if(nargin < 5 || isempty(N))
    N  = 24; %number of stim basis functions (this'll need go from sample stim onset through the entire delay)
end

[~,~,basis_1] = DMC.modelBuilder.makeRaisedCosBasis(N, delta_t, 1e-3*[0 TT], 0.2, [], true); %get the raised cosine basis

%cut out valid times and shift basis a little
    %note that this shift makes it so that the 1st basis peak will now no longer be at 0 - (it takes time for stim to reach LIP)
    %this basis function will be close to 0 at stimulus onset and allow an okay impulse response function
basis_0 = basis_1((find(basis_1(:,1)>0,1,'first')-1):end,:); 


%get the peak times of each basis function
peaks = zeros(N,1);
for ii = 1:N
    [~,peaks(ii)] = max(basis_0(:,ii));
end

tt_0    = size(basis_0,1);
tts_0   =  -tt_0:(tt_0-1);
basis_0 = [zeros(tt_0,N); basis_0];

stim_0 = basis_0(:,1:N);
stim = orth(stim_0); %orthogonalizes for numerical reasons

basisStruct.stimBasis_0 = stim_0((tt_0+1):end,:);

basisStruct.stimBasis    = stim((tt_0+1):end,:);
basisStruct.stimBasis_tts = tts_0((tt_0+1):end);

%% lever basis
N_lev = find(peaks > 300/downsampleRate,1,'first'); %select number of basis, lever length should be only 
lever_0 = circshift(basis_0(end:-1:1,1:N_lev),[bins_post_lever+20/downsampleRate 0]);
lever = orth(lever_0); %orthogonalizes for numerical reasons
tt_v = sum(lever.^2,2)>1e-8;
basisStruct.leverBasis_0 = lever_0(tt_v, :);
basisStruct.leverBasis     = lever(tt_v,:);
basisStruct.leverBasis_tts = tts_0(tt_v);

%% get spike history filter
%filter can use delta functions for a period of the refractory period pluse 
N_spkHist_long   = 6;%number of raised cosine bases
N_spkHist_refrac = ceil(10e-3/delta_t);%set first filters during refractory period to be delta functions (set to be 10ms)
T_spkHist_refrac = 1; %one delta function for bin
T_total_refrac = N_spkHist_refrac*T_spkHist_refrac;

spkHistB = 0.05; %controls how the raised cosines will be spaced

bc    = [ones(T_spkHist_refrac,1);zeros((N_spkHist_refrac-1)*T_spkHist_refrac,1)];
for ii = 2:N_spkHist_refrac
    bc = [bc circshift(bc(:,ii-1),[T_spkHist_refrac 0])]; %#ok<AGROW>
end
bc = orth(bc);%*10;
if(N_spkHist_long > 0)
    [~, ~, spkHistBasis2] = DMC.modelBuilder.makeRaisedCosBasis(N_spkHist_long, delta_t, [T_total_refrac*1e-3 200e-3],spkHistB);
    spkHistBasis2(1:T_total_refrac,:) = 0;

    bc = [bc;zeros(size(spkHistBasis2,1)-size(bc,1),size(bc,2))];
    bb = [bc spkHistBasis2];
    
    basisStruct.spkHistBasis_0 = bb;

    
    sb = [orth(bc) orth(spkHistBasis2)];

    basisStruct.spkHistBasis = sb; 
else
    basisStruct.spkHistBasis = bc;
end
basisStruct.spkHistBasis_tts = 1:size(basisStruct.spkHistBasis,1);

%% plot the bases
if(plotBases)
    if(islogical(plotBases))
        figure();
    else
        figure(plotBases);
    end
    clf
    subplot(2,3,1);
    plot(basisStruct.stimBasis_tts * delta_t * 1e3, basisStruct.stimBasis_0);
    xlabel('time from stimulus onset (ms)');
    title('stimulus filter basis');
    
    subplot(2,3,4);
    plot(basisStruct.stimBasis_tts * delta_t * 1e3, basisStruct.stimBasis);
    xlabel('time from stimulus onset (ms)');
    title('orthormalized stimulus filter basis');
    
    subplot(2,3,2);
    plot(basisStruct.leverBasis_tts * delta_t * 1e3, basisStruct.leverBasis_0);
    xlabel('time from lever release onset (ms)');
    title('lever filter basis');
    
    subplot(2,3,5);
    plot(basisStruct.leverBasis_tts * delta_t * 1e3, basisStruct.leverBasis);
    xlabel('time from lever release (ms)');
    title('orthormalized lever filter basis');
    
    subplot(2,3,3);
    plot(basisStruct.spkHistBasis_tts * delta_t * 1e3, basisStruct.spkHistBasis_0);
    xlabel('time from spike (ms)');
    title('spk hist filter basis');
    
    subplot(2,3,6);
    plot(basisStruct.spkHistBasis_tts * delta_t * 1e3, basisStruct.spkHistBasis);
    xlabel('time from spike (ms)');
    title('orthormalized spk hist filter basis');
end
