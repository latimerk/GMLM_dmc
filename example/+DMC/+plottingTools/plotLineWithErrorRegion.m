function [pp, eb] = plotLineWithErrorRegion(ax, tts, y, eb_1, eb_2, color_c, lw, ls)

if(nargin < 6 || isempty(color_c))
    color_c = [0 0 0];
end
if(nargin < 7 || isempty(lw))
    lw = 1;
end
if(nargin < 8 || isempty(ls))
    ls = '-';
end

eb_y  = [eb_1(:); eb_2(end:-1:1)];
eb_y(isnan(eb_y)) = 0;
y(isnan(y)) = 0; %fix any nans

tts = tts(:);
tts_eb = [tts(:); tts(end:-1:1)];

eb = patch(ax, tts_eb, eb_y, color_c); 
set(eb, 'FaceColor', color_c);
set(eb, 'EdgeColor', color_c);
set(eb, 'LineStyle', 'none');
set(eb, 'LineWidth', 1);
set(eb, 'FaceAlpha', 60/255);
set(eb, 'HandleVisibility', 'off');
pp = plot(ax, tts, y, ls, 'linewidth', lw, 'color', color_c);