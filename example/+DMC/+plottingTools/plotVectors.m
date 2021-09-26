
%% plot vectors
function [] = plotVectors(ax, dirs, colors, includeBoundary, markedDirection, arc)
    hold(ax, 'on');
    
    
    %plot requested arcs
    if(nargin >= 6)
        for ii = 1:size(arc, 1)
            theta_1 = min(dirs(arc(ii,:)));
            theta_2 = max(dirs(arc(ii,:)));
            
            rr = 0.4;
            lw = 1;
            cs = [0 0 0];
            
            nn = 100;
            
            tts = linspace(theta_1, theta_2, nn);
            
            plot(cosd(tts)*rr, sind(tts)*rr, '-', 'linewidth', lw, 'color', cs);
        end
    end
    
    %plot vectors
    for ii = 1:numel(dirs)
        if(size(colors, 1) == 1)
            cs = colors;
        else
            cs = colors(ii, :);
        end
        
        plot(ax, [0 cosd(dirs(ii))], [0 sind(dirs(ii))], '-', 'Color', cs, 'LineWidth', 3);
    end
    
    %plot boundary
    if(nargin >= 4 && includeBoundary)
        plot([-1 1], [-1 1], '--', 'Color', [0 0 0], 'LineWidth', 1);
    end
    
    %plot markers
    if(nargin >= 5)
        rr = 1.08 * scale;
        delta = -12;
        ms = 6;
        lw = 1;
        cs = [0 136 170]./255;
        for ii = 1:numel(markedDirection)
            dir_c = markedDirection(ii) + delta;
            plot(ax, cosd(dir_c)*rr, sind(dir_c)*rr, '*', 'Color', cs, 'LineWidth', lw, 'MarkerSize', ms);
        end
    end
    
    %set lims
    xlim(ax, [-1.2 1.2]);
    ylim(ax, [-1.2 1.2]);
    axis(ax, 'square');
    
    axis(ax, 'off');
end
