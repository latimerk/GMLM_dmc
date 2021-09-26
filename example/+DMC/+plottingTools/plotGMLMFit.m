function [] = plotGMLMFit(TaskInfo, paramStruct, bases, R_sample_stim, R_sample_stim_dirOnly, R_sample_stim_catOnly)
if(isempty(which('hosvd')))
    error('Requires tensor_toolbox to be in MATLAB path! (for the hosvd function)');
end

for jj = 1:numel(paramStruct.Groups)
    if(strcmpi(paramStruct.Groups(jj).name, 'stimulus'))
        stimGroup = jj;
    end
    if(strcmpi(paramStruct.Groups(jj).name, 'lever'))
        leverGroup = jj;
    end
end

%% plot the MLE fit from the example script
%get tensor of sample stimulus filters 
ND = numel(TaskInfo.Directions);
tts = (1:ceil(1500/TaskInfo.binSize_ms)); %here I'm cutting out the first 1500 ms of the filters for visualization
tts_ms = tts*5;

rank_c = size(paramStruct.Groups(stimGroup).T{1}, 2);

Components.Time   = bases.stimBasis(tts,:)*double(paramStruct.Groups(stimGroup).T{1});
Components.Stim   = R_sample_stim*double(paramStruct.Groups(stimGroup).T{2});
Components.Neuron = double(paramStruct.Groups(stimGroup).V);
sampleStimFilters = ktensor(ones(rank_c,1), {Components.Time, Components.Stim, Components.Neuron});
sampleStimFilters = double(sampleStimFilters);

%use high-order SVD to get top 3 dimensions (note that paramStruct_mle.Groups(1).V is not orthonormal. This decomposition will give you a orthonormal subspace)
M_all          = hosvd(tensor(sampleStimFilters), 1e-6, 'ranks', [rank_c ND rank_c]); 
M_mean_removed = hosvd(tensor(sampleStimFilters - mean(sampleStimFilters,2)),  1e-6, 'ranks', [rank_c ND rank_c]);

K_all = M_all;
K_all.U{3} = eye(rank_c);
K_all = double(K_all);

K_mean_removed = M_mean_removed;
K_mean_removed.U{3} = eye(rank_c);
K_mean_removed = double(K_mean_removed);

%scale by population size
dim_P = numel(paramStruct.W);
K_all = K_all ./ sqrt(dim_P);
K_mean_removed = K_mean_removed ./ sqrt(dim_P);

offsetTime = ceil(TaskInfo.StimLength_ms/TaskInfo.binSize_ms);

directionColors = DMC.plottingTools.getDirColors_oneBoundaryTask(TaskInfo.Directions);
catColors       = [0 0 1;
                   1 0 0];
               
if(rank_c >= 3)
    figure(1);
    clf
    for jj = 1:2
        if jj == 1
            K_c = K_all;
        else
            K_c = K_mean_removed;
        end
        subplot(1,2,jj);
        hold on
        for ii = 1:ND
            plot3(K_c(:, ii, 1), K_c(:, ii, 2), K_c(:, ii, 3), 'color', directionColors(ii,:));
            %plot sample stim onset & offset
            plot3(K_c(1, ii, 1), K_c(1, ii, 2), K_c(1, ii, 3), 'o', 'markersize', 6, 'color', directionColors(ii,:), 'markerfacecolor', directionColors(ii,:));
            plot3(K_c(offsetTime, ii, 1), K_c(offsetTime, ii, 2), K_c(offsetTime, ii, 3), 'v', 'markersize', 6, 'color', directionColors(ii,:), 'markerfacecolor', directionColors(ii,:));
        end
        
        max_ds = squeeze(max(K_c, [], [1 2]));
        min_ds = squeeze(min(K_c, [], [1 2]));
        cs = mean([min_ds max_ds], 2);
        rr = max(max_ds - min_ds);
        rr = max(1e-8, ceil(10*rr)/10);

        ls = cs + [-rr/2 rr/2];
        xlim(ls(1,:));
        ylim(ls(2,:));
        zlim(ls(3,:));
        
        xlabel('dim 1');
        ylabel('dim 2');
        zlabel('dim 3');
        if(jj == 1)
            title('Sample stim: low dim response');
        else
            title('mean removed');
        end
        axis square
        hold off
    end

elseif(rank_c == 2)
    figure(1);
    clf
    for jj = 1:2
        if jj == 1
            K_c = K_all;
        else
            K_c = K_mean_removed;
        end
        subplot(1,2,1);
        hold on
        for ii = 1:ND
            plot(K_c(:, ii, 1), K_c(:, ii, 2),  'color', directionColors(ii,:));
            %plot sample stim onset & offset
            plot(K_c(1, ii, 1), K_c(1, ii, 2),  'o', 'markersize', 6, 'color', directionColors(ii,:), 'markerfacecolor', directionColors(ii,:));
            plot(K_c(offsetTime, ii, 1), K_c(offsetTime, ii, 2),  'v', 'markersize', 6, 'color', directionColors(ii,:), 'markerfacecolor', directionColors(ii,:));
        end
        xlabel('dim 1');
        ylabel('dim 2');
        if jj == 1
            title('Sample stim: low dim response');
        else
            title('mean removed');
        end
        
        max_ds = squeeze(max(K_c, [], [1 2]));
        min_ds = squeeze(min(K_c, [], [1 2]));
        cs = mean([min_ds max_ds], 2);
        rr = max(max_ds - min_ds);
        rr = max(1e-8, ceil(10*rr)/10);

        ls = cs + [-rr/2 rr/2];
        xlim(ls(1,:));
        ylim(ls(2,:));
        
        axis square
        hold off
    end
end


%% plot fits of a few neurons
exampleNeurons = [1 5 27];
figure(3);
clf;
NR = rank_c + 1;
NC = numel(exampleNeurons);

for ii = 1:numel(exampleNeurons)
    loadings = zeros(size(Components.Time, 1), ND, rank_c);
    clear ax;
    for rr = 1:rank_c
        ax(rr) = subplot(NR, NC, ii + (rr-1)*NC);
        hold on
        loadings(:, :, rr) = Components.Time(:,rr)* Components.Stim(:, rr)' * Components.Neuron(exampleNeurons(ii), rr);
        
        for cc = 1:ND
            plot(tts_ms, loadings(:, cc, rr), 'color', directionColors(cc,:));
            %plot sample stim onset & offset
            plot(tts_ms(1), loadings(1, cc, rr),  'o', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
            plot(tts_ms(offsetTime), loadings(offsetTime, cc, rr), 'v', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
        end
        set(gca,'TickDir','out');
        if(ii == 1)
            ylabel(sprintf('component %d', rr))
        end
        if(rr == 1)
            title(sprintf('cell %d', exampleNeurons(ii)));
        end
    end
    total = sum(loadings, 3);
    
    ax(rank_c+1) = subplot(NR, NC, ii + (rank_c)*NC);
    hold on
    for cc = 1:ND
        plot(tts_ms, total(:, cc), 'color', directionColors(cc,:));
        %plot sample stim onset & offset
        plot(tts_ms(1), total(1, cc),  'o', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
        plot(tts_ms(offsetTime), total(offsetTime, cc), 'v', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
    end
    set(gca,'TickDir','out');
    if(ii == 1)
        xlabel('time from stim onset (ms)')
        ylabel('total log gain');
    end
    
    linkaxes(ax) %scale components the same to visualize relative contribution
end

%% plot lever and spk history filter for example neurons
lev_filters = (bases.leverBasis*paramStruct.Groups(leverGroup).T{1})*paramStruct.Groups(leverGroup).V';
spkHist_filters = bases.spkHistBasis * paramStruct.B;

figure(4);
clf;
NR = 2;
NC = numel(exampleNeurons);
for ii = 1:numel(exampleNeurons)
    %lever filter
    subplot(NR, NC, ii );
    plot(bases.leverBasis_tts * TaskInfo.binSize_ms, lev_filters(:, exampleNeurons(ii)));
    if(ii == 1)
        xlabel('time from lever release (ms)');
        ylabel('log gain');
    end
    
    title(sprintf('cell %d\nbaseline log rate = %.2f\nlever filter', exampleNeurons(ii), paramStruct.W(exampleNeurons(ii))));
    
    subplot(NR, NC, ii + NC);
    plot(bases.spkHistBasis_tts * TaskInfo.binSize_ms, spkHist_filters(:, exampleNeurons(ii)));
    if(ii == 1)
        xlabel('time from prev spike (ms)');
        ylabel('log gain');
    end
    title('spk hist filter');
end

%% plot components (note that this does not use the orthogonal space above, but the CP/PARAFAC decomposition components)
   %I reorder the components in terms of loading magnitude - note that this does not change the above hosvd and plot
if(nargin < 6)
    return;
end

[~, order] = sort(sum(Components.Neuron.^2), 'descend');
Components.Time = Components.Time(:, order);
Components.Stim = Components.Stim(:, order);
Components.Stim_directionOnly = R_sample_stim_dirOnly*double(paramStruct.Groups(stimGroup).T{2}(:, order));
Components.Stim_sampleCatOnly = R_sample_stim_catOnly*double(paramStruct.Groups(stimGroup).T{2}(:, order));

Components.Neuron = Components.Neuron(:, order);
NR = rank_c;
NC = 3;

figure(2);
clf
for rr = 1:rank_c
    T_c = Components.Time(:, rr); % current temporal component
    
    %plot temporal component
    subplot(NR, NC, (rr-1)*NC + 1);
    hold on
    plot(tts_ms, T_c, 'k');
    plot(tts_ms(1), T_c(1),  'o', 'markersize', 6, 'color',[0 0 0], 'markerfacecolor', [0 0 0]);
    plot(tts_ms(offsetTime), T_c(offsetTime), 'v', 'markersize', 6, 'color', [0 0 0], 'markerfacecolor', [0 0 0]);
    
    set(gca, 'TickDir', 'out');
    if(rr == rank_c)
        xlabel('time from stim onset (ms)')
    end
    if(rr == 1)
        title('temporal weights');
    end
    ylabel(sprintf('component %d', rr))
    
    %plot component with sample stim weights
    clear ax;
    ax(1) = subplot(NR, NC, (rr-1)*NC + 2);
    hold on
    loadings = T_c * Components.Stim_sampleCatOnly(:, rr)';
    for cc = 1:2
        plot(tts_ms, loadings(:, cc), 'color', catColors(cc,:));
        %plot sample stim onset & offset
        plot(tts_ms(1), loadings(1, cc),  'o', 'markersize', 6, 'color', catColors(cc,:), 'markerfacecolor', catColors(cc,:));
        plot(tts_ms(offsetTime), loadings(offsetTime, cc), 'v', 'markersize', 6, 'color', catColors(cc,:), 'markerfacecolor', catColors(cc,:));
    end
    
    if(rr == 1)
        title('sample cat weighted');
    end
    
    %plot component with direction weights
    ax(2) = subplot(NR, NC, (rr-1)*NC + 3);
    hold on
    loadings = T_c * Components.Stim_directionOnly(:, rr)';
    for cc = 1:ND
        plot(tts_ms, loadings(:, cc), 'color', directionColors(cc,:));
        %plot sample stim onset & offset
        plot(tts_ms(1), loadings(1, cc),  'o', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
        plot(tts_ms(offsetTime), loadings(offsetTime, cc), 'v', 'markersize', 6, 'color', directionColors(cc,:), 'markerfacecolor', directionColors(cc,:));
    end
    if(rr == 1)
        title('direction weighted');
    end
    
    linkaxes(ax)
end