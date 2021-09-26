
%% plot hemifield
function [] = plotHemifield(ax, dir, major_dir_eb, color_vec, color_region)
    hold(ax, 'on');
    
    dirs = dir + [0 180];
    
    %plot region
    if(nargin > 3 || ~isempty(color_region))
        rr = 1.2;
        
        %% region 1
        cs = color_region(1, :);
        theta = linspace(major_dir_eb(2) - 90, major_dir_eb(1) + 90, 180);

        xx = [cosd(theta(:)) * rr; 0];
        yy = [sind(theta(:)) * rr; 0];

        h = patch(xx, yy, cs);
        set(h, 'FaceColor', cs);
        set(h, 'EdgeColor', cs);
        set(h, 'LineWidth', 0.05);
        set(h, 'FaceAlpha', 0.2);
        
        
        %% region 2
        cs = color_region(2, :);
        theta = linspace(major_dir_eb(2) - 90 + 180, major_dir_eb(1) + 90 + 180, 180);

        xx = [cosd(theta(:)) * rr; 0];
        yy = [sind(theta(:)) * rr; 0];

        h = patch(xx, yy, cs);
        set(h, 'FaceColor', cs);
        set(h, 'EdgeColor', cs);
        set(h, 'LineWidth', 0.05);
        set(h, 'FaceAlpha', 0.2);
        
        
        %% err region 1
        cs = color_region(3, :);
        theta = linspace(major_dir_eb(1) - 90, major_dir_eb(2) + 90 - 180, 180);

        xx = [cosd(theta(:)) * rr; 0];
        yy = [sind(theta(:)) * rr; 0];

        h = patch(xx, yy, cs);
        set(h, 'FaceColor', cs);
        set(h, 'EdgeColor', cs);
        set(h, 'LineWidth', 0.05);
        set(h, 'FaceAlpha', 0.2);
        
        
        %% err region 2
        cs = color_region(3, :);
        theta = linspace(major_dir_eb(2) + 90, major_dir_eb(1) - 90 + 180, 180);

        xx = [cosd(theta(:)) * rr; 0];
        yy = [sind(theta(:)) * rr; 0];

        h = patch(xx, yy, cs);
        set(h, 'FaceColor', cs);
        set(h, 'EdgeColor', cs);
        set(h, 'LineWidth', 0.05);
        set(h, 'FaceAlpha', 0.2);
    end
    
    %plot vectors
    rr = 1.15;
    for ii = 1:numel(dirs)
        plot(ax, [0 cosd(dirs(ii))] * rr, [0 sind(dirs(ii))] * rr, ':', 'Color', color_vec, 'LineWidth', 1);
    end
    

    %set lims
    xlim(ax, [-1.2 1.2]);
    ylim(ax, [-1.2 1.2]);
    axis(ax, 'square');
    
    axis(ax, 'off');
end

