function [] = plotGLMFit(paramStruct, R_sample_stim, R_sample_stim_dirOnly, bases, TaskInfo, figNum, titleStr)
% all filter coefficients are in paramStruct.W - this takes them out in the correct order

if(nargin < 7 || isempty(titleStr))
    titleStr = '';
else
    titleStr = sprintf('%s\n', titleStr);
end

%% get stimulus filters
NF = size(R_sample_stim,  2);  %number of stim filters
NB = size(bases.stim.B,2); %number of stim bases
stimFilters = bases.stim.B*reshape(paramStruct.Ks{1},[],NF);

stimFilter_tts = bases.stim.tts_0;
test_ts = stimFilter_tts <= 650; %only plot first 650 ms of test stim filters

NR = 2;
NC = 4;
ND = numel(TaskInfo.Directions);

directionColors = DMC.plottingTools.getDirColors_oneBoundaryTask(TaskInfo.Directions);
%%
if(nargin < 6 || isempty(figNum))
    figure();
else
    figure(figNum);
end
clf
%cat filters
subplot(NR, NC, 1);
hold on
plot(stimFilter_tts, stimFilters(:, 1), 'color', [0 0 1]);
plot(stimFilter_tts, stimFilters(:, 2), 'color', [1 0 0]);

plot(stimFilter_tts(test_ts), stimFilters(test_ts, 3), 'color', [0.1 0.1 0.8]);
plot(stimFilter_tts(test_ts), stimFilters(test_ts, 4), 'color', [0.7 0 0.2]);

legend({'sample: cat 1', 'sample: cat 2', 'test: cat 1', 'test: cat 2'});
title(sprintf('%scategory filters', titleStr));
ylabel('log gain');
xlabel('time from stim onset (ms)');

% base direction filters
subplot(NR, NC, 2)
hold on
plot(stimFilter_tts, stimFilters(:,5:6));
legend({'sine','cosine'});
title('base direction tuning filters');
xlabel('time from stim onset (ms)');

% direction tuning
subplot(NR, NC, 3)
hold on
for cc = 1:ND
    plot(stimFilter_tts, stimFilters*R_sample_stim_dirOnly(cc,:)', 'color', directionColors(cc,:));
end
title('task direction tuning filters');
xlabel('time from stim onset (ms)');

% total sample response
subplot(NR, NC, 4)
hold on
for cc = 1:ND
    plot(stimFilter_tts, stimFilters*R_sample_stim(cc,:)', 'color', directionColors(cc,:));
end
title('sample stim: total tuning');
xlabel('time from stim onset (ms)');

%% plot lever filter
leverFilter = bases.response.B*paramStruct.Ks{2};

subplot(NR, NC, NC + 1);
hold on
plot(bases.response.tts_0 * TaskInfo.binSize_ms, leverFilter);
title('lever filter');
ylabel('log gain');
xlabel('time from lever release (ms)');


%% plot spkHist filter
spkHistFilter = bases.spkHist.B*paramStruct.Ks{3};

subplot(NR, NC, NC + 2);
hold on
plot(bases.spkHist.tts_0 * TaskInfo.binSize_ms, spkHistFilter);
title('spk hist filter');
xlabel('time from last spk');

