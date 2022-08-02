function [log_m] = logMeanExp(log_x,dim)
if(nargin < 2)
    dim = 2;
end


NE = sum(~isnan(log_x), dim);
NE(NE == 0) = nan;

log_m = -log(NE) + kgmlm.utils.logSumExp(log_x,dim);
