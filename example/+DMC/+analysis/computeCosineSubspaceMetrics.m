% Some summary analyses for the MCMC output for the cosine-tuning model
%   inputs:
%       samples - the structure of sampels returned by the GMLM
%       HMC_settings - the settings used for the samples (only needs to know how many burn-in samples)
%       bases - the filter basis structure
%       delta_t - the bin size in seconds (not milliseconds!)
%
%   for filtStats, the last dim of each summary object indexes the percentile range for that quantity (which prctiles are computed are given in
%   filtStats.prcTiles). Note: the last entry isn't a percentile, it's the posterior mean.
%
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
function [filtStats, timePeriods, timeInfo] = computeCosineSubspaceMetrics(samples, HMC_settings, bases, delta_t)
 


%%
timePeriods = struct('name', [], 'range_ms', [], 'range_bins', []);

timePeriods(1).name = 'stim_early';
timePeriods(1).range_ms = 1:300;
timePeriods(2).name = 'stim_late';
timePeriods(2).range_ms = 350:650;
timePeriods(3).name = 'stim';
timePeriods(3).range_ms = 1:650;
timePeriods(4).name = 'delay_early';
timePeriods(4).range_ms = 700:1000;
timePeriods(5).name = 'delay_middle';
timePeriods(5).range_ms = 1000:1300;
timePeriods(6).name = 'delay_late';
timePeriods(6).range_ms = 1300:1600;
timePeriods(7).name = 'delay';
timePeriods(7).range_ms = 700:1600;
timePeriods(8).name = 'stim_delay';
timePeriods(8).range_ms = 1:1600;

timeInfo = struct();

timeInfo.stimBasis_ms  =  bases.stim.tts_0;
timeInfo.leverBasis_ms =  bases.response.tts_0;
for ii = 1:numel(timePeriods)
    timePeriods(ii).range_bins = timeInfo.stimBasis_ms  >= timePeriods(ii).range_ms(1) & timeInfo.stimBasis_ms  <= timePeriods(ii).range_ms(end);
end

timeInfo.test_tts = timeInfo.stimBasis_ms <= 700;

%%
TT = size(bases.stim.B,1);
NP = numel(timePeriods);
N_samples = HMC_settings.nSamples;

filtStats_sample.dir.major_norm = nan(TT, N_samples);
filtStats_sample.dir.minor_norm = nan(TT, N_samples);
filtStats_sample.dir.major_angle_minus135       = nan(TT, N_samples);
filtStats_sample.dir.major_angle_minus135_means = nan(NP, N_samples);

filtStats_sample.cat.sample_norm = nan(TT, N_samples);
filtStats_sample.cat.test_norm   = nan(TT, N_samples);

%% setup coefficients for different vectors

R_cat_sample = [1 -1 0  0 0 0];
R_cat_test   = [0  0 1 -1 0 0];

R_cosine  = [0 0 0 0 0 1];
R_sine    = [0 0 0 0 1 0];

dim_R_stim = size(samples.Groups(1).V,2) ;
dim_R_tot = dim_R_stim + size(samples.Groups(2).V,2);
dim_P = size(samples.Groups(1).V,1);
%% space
stimFilt_cat_sample_means = zeros(NP , dim_R_stim);
stimFilt_cat_test_means   = zeros(NP , dim_R_stim);

%% for each sample
for ss = 1:N_samples
    if(mod(ss, 1000) == 0 || ss == 1 || ss == N_samples)
        fprintf('Sample %d / %d...\n', ss, N_samples);
    end
    sample_idx = HMC_settings.nWarmup + ss;
    

    %% get projection space

    V_stim_0 = samples.Groups(1).V(:,:,sample_idx);
    V_all = orth(V_stim_0);

    V_stim = zeros(dim_R_stim, dim_R_stim);
    V_stim(:, 1:size(V_all,2)) = (V_stim_0'*V_all) ./ sqrt(dim_P);

    stim_T = bases.stim.B * samples.Groups(1).T{1}(:,:,sample_idx);
    stim_X = samples.Groups(1).T{2}(:,:,sample_idx);

    %% direction ellipse info

    dir_f1 = (stim_T .* (R_cosine * stim_X)) * V_stim; %cosine vector
    dir_f2 = (stim_T .* (R_sine   * stim_X)) * V_stim; %sine vector

    t_0 = 1/2*acot((sum(dir_f1.*dir_f1,2) - sum(dir_f2 .* dir_f2, 2)) ./ (2 * sum(dir_f1 .* dir_f2, 2)));  
    dir_v1 = (dir_f1.*cos(t_0       ) + dir_f2.*sin(t_0       )) - (dir_f1.*cos(t_0 + pi    ) + dir_f2.*sin(t_0 + pi    )); %major and minor axes
    dir_v2 = (dir_f1.*cos(t_0 + pi/2) + dir_f2.*sin(t_0 + pi/2)) - (dir_f1.*cos(t_0 + pi*3/2) + dir_f2.*sin(t_0 + pi*3/2));

    %compute major/minor axis magnitudes (over time, with CI)
    mg = [sqrt(sum(dir_v1.^2,2)) sqrt(sum(dir_v2.^2,2))];

    [filtStats_sample.dir.major_norm(:, ss), majorAngleIdx] = max(mg, [], 2);
     filtStats_sample.dir.minor_norm(:, ss)    = min(mg, [], 2);

    %compute elipse area (over time, with CI)
    filtStats_sample.dir.area(:, ss) = prod(mg, 2) * pi;

    %compute major axis angle relative to category direction (over time, with CI)
    optAngles = rad2deg(t_0);
    optAngles(majorAngleIdx == 2) = optAngles(majorAngleIdx == 2) + 90;
    optAngles(optAngles <  45) = optAngles(optAngles <  45) + 180;
    optAngles(optAngles <  45) = optAngles(optAngles <  45) + 180;
    optAngles(optAngles > 225) = optAngles(optAngles > 225) - 180;
    optAngles(optAngles > 225) = optAngles(optAngles > 225) - 180;

    for pp = 1:NP %mean axis angle during each period
        %takes circular mean
        x = sum(sind(optAngles(timePeriods(pp).range_bins)));
        y = sum(cosd(optAngles(timePeriods(pp).range_bins)));
        mm_0 = atan2d(x, y);
        while(mm_0 < 45)
            mm_0 = mm_0 + 180;
        end
        while(mm_0 > 225)
            mm_0 = mm_0 - 180;
        end
        mm = mm_0 - 135;
        if(mm < -90 || mm > 90)
            fprintf('invalid angles!');
        end
        filtStats_sample.dir.major_angle_minus135_means(pp, ss) = mm; % aligns to task
    end
    filtStats_sample.dir.major_angle_minus135(:, ss) = optAngles - 135;

    %% compute cat vector info
    stimFilt_cat_sample = (stim_T .* (R_cat_sample * stim_X)) * V_stim; 
    stimFilt_cat_test   = (stim_T .* (R_cat_test   * stim_X)) * V_stim;

    %get period averages
    for pp = 1:NP 
        stimFilt_cat_sample_means(pp, :) = mean(stimFilt_cat_sample(timePeriods(pp).range_bins, :), 1);
        stimFilt_cat_test_means(  pp, :) = mean(stimFilt_cat_test(  timePeriods(pp).range_bins, :), 1);
    end
    stimFilt_cat_sample_means = stimFilt_cat_sample_means ./ sqrt(sum(stimFilt_cat_sample_means.^2, 2));
    stimFilt_cat_test_means   = stimFilt_cat_test_means   ./ sqrt(sum(stimFilt_cat_test_means.^2  , 2));

    %vector magntitude (over time, with CI)
    filtStats_sample.cat.sample_norm(:, ss) = sqrt(sum(stimFilt_cat_sample .^ 2, 2));
    filtStats_sample.cat.test_norm(:, ss)   = sqrt(sum(stimFilt_cat_test .^ 2, 2)); 
end

%% get mean, median, and CI for each stat
fprintf('Computing summaries...\n');
prcTiles = [0.5 2.5 5 25 50 75 95 97.5 99.5];

NG = numel(prcTiles);


filtStats = struct();

fields_0 = string(fieldnames(filtStats_sample));
for ii = 1:numel(fields_0)
    main_field = fields_0(ii);
    
    fields_c = string(fieldnames(filtStats_sample.(main_field)));
    
    fprintf('\t part %d / %d... %d subparts... ', ii, numel(fields_0), numel(fields_c));
    for jj = 1:numel(fields_c)
        fprintf('%d  ', jj);
        sub_field = fields_c(jj);
        currentStats = filtStats_sample.(main_field).(sub_field);
        
        ss = size(currentStats);
        dd = numel(ss);
        
        filtStats.(main_field).(sub_field) = zeros([ss(1:dd-1) NG+1]);
        
        if(dd == 3)
            filtStats.(main_field).(sub_field)(:, :, end)  = mean(currentStats, dd);
            filtStats.(main_field).(sub_field)(:, :, 1:NG) = prctile(currentStats, prcTiles, dd);
        elseif(dd == 2)
            filtStats.(main_field).(sub_field)(:, end)  = mean(currentStats, dd);
            filtStats.(main_field).(sub_field)(:, 1:NG) = prctile(currentStats, prcTiles, dd);
        else
            error('invalid dim');
        end
    end
    fprintf('\n');
end

fprintf('  angular means...\n');
x = sum(sind(filtStats_sample.dir.major_angle_minus135),2);
y = sum(cosd(filtStats_sample.dir.major_angle_minus135),2);
filtStats.dir.major_angle_minus135(:, end) = atan2d(x, y);

filtStats.prcTiles = prcTiles;

filtStats.dir.major_angle_minus135_means = filtStats_sample.dir.major_angle_minus135_means;

fprintf('Done.\n');
