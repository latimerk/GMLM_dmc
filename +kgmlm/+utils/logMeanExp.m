function [log_m] = logMeanExp(log_x,dim)
if(nargin < 2)
    dim = 2;
end


if(strcmpi(dim, 'all'))
    NE = numel(log_x);
else
    NE = size(log_x, dim);
end

log_m = -log(NE) + kgmlm.utils.logSumExp(log_x,dim);
