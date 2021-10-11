%  Modified Cardinal Spline basis functions proposed by
%
%  Sarmashghi, M., Jadhav, S. P., & Eden, U. (2021). Efficient Spline Regression for Neural Spiking Data. bioRxiv.
%
%  Modified by Kenneth Latimer from Repository:
%     https://github.com/MehradSm/Modified-Spline-Regression
%     by Mehrad Sarmashghi
%

% inputs: lag  = last time point (in ms)
%                If is pair: time points [t_0 t_end]. If only scalar, assumes t_0=binSize_ms
%         c_pt = knot locations (in ms)
%         s    = tension parameter (default = 0.5)
%         binSize_ms = size of bins (default = 1)
%         zeroEndPoints = logical pair (default = [false false]). Whether to set the end points at c_pt(1) and c_pt(end) to 0's (removes bases)

function [HistSpl_orth, HistSpl, tts, tts_bins] = ModifiedCardinalSpline(lag, c_pt, s, binSize_ms, zeroEndPoints)
if(nargin < 3 || isempty(s))
    s = 0.5;
end
if(nargin < 4 || isempty(binSize_ms))
    binSize_ms = 1;
end
if(nargin < 5 || isempty(zeroEndPoints))
    zeroEndPoints = [false false];
end

if(isscalar(lag))
    t_0   = binSize_ms;
    t_end = lag;
else
    t_0   = lag(1);
    t_end = lag(2);
end

tts = t_0:binSize_ms:t_end;
T = numel(tts);
tts_bins = floor(tts./binSize_ms);

HistSpl = zeros(T,length(c_pt));

%for each 1 ms timepoint, calculate the corresponding row of the glm input matrix
for tt_idx = 1:T
    tt = tts(tt_idx);
    
    nearest_c_pt_index = find(c_pt < tt, 1, 'last');
    nearest_c_pt_time  = c_pt(nearest_c_pt_index);
    next_c_pt_time     = c_pt(nearest_c_pt_index+1);
    
    % Compute the fractional distance between timepoint i and the nearest knot
    u  = (tt - nearest_c_pt_time)./(next_c_pt_time - nearest_c_pt_time);
    lb = (c_pt(3)   - c_pt(1    ))/(c_pt(2)   - c_pt(1));
    le = (c_pt(end) - c_pt(end-2))/(c_pt(end) - c_pt(end-1));
    
    % Beginning knot 
    
    if(nearest_c_pt_time == c_pt(1))
        S = [2-(s/lb)   -2  s/lb;
               (s/lb)-3  3 -s/lb; 
                0        0  0; 
                1        0  0];
        bbs = (nearest_c_pt_index):(nearest_c_pt_index+2);
           
    % End knot
    elseif(nearest_c_pt_time == c_pt(end-1))
        S = [-s/le  2   -2+(s/le);
            2*s/le -3  3-(2*s/le);
             -s/le  0       s/le;
              0     1      0];
        bbs = (nearest_c_pt_index-1):(nearest_c_pt_index+1);
           
    % Interior knots
    else
        privious_c_pt = c_pt(nearest_c_pt_index-1);
        next2 = c_pt(nearest_c_pt_index+2);
        l1 = next_c_pt_time - privious_c_pt;
        l2 = next2 - nearest_c_pt_time;
        S = [ -s/l1 2-(s/l2)       (s/l1)-2  s/l2;
             2*s/l1   (s/l2)-3 3-2*(s/l1)   -s/l2;
              -s/l1    0            s/l1     0;
               0       1            0        0];
        bbs = (nearest_c_pt_index-1):(nearest_c_pt_index+2);
    end
    p = [u^3 u^2 u 1]*S;
    HistSpl(tt_idx, bbs) = p; 
    if(~all(~isnan(p) & ~isinf(p), 'all'))
        fprintf("basis error!\n");
    end
end

if(zeroEndPoints(1))
    HistSpl = HistSpl(:, 2:end);
end
if(zeroEndPoints(2))
    HistSpl = HistSpl(:, 1:end-1);
end
    

HistSpl_orth = orth(HistSpl);
end


