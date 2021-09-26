function [stimPatch] = plotRegionForStimulus(ax, stim_on, stim_off, y_0, y_1)

stimPatch = patch(ax, [stim_on stim_on stim_off stim_off], [y_0 y_1 y_1 y_0], [0 0 0]); 
plot(ax, [stim_on  stim_on ], [y_0 y_1], ':k', 'linewidth', 0.5, 'HandleVisibility', 'off');
plot(ax, [stim_off stim_off], [y_0 y_1], ':k', 'linewidth', 0.5, 'HandleVisibility', 'off');
set(stimPatch, 'FaceColor', [0 0 0]);
set(stimPatch, 'EdgeColor', [0 0 0]);
set(stimPatch, 'LineStyle', 'none');
set(stimPatch, 'LineWidth', 1);
set(stimPatch, 'FaceAlpha', 20/255);
set(stimPatch, 'HandleVisibility', 'off')