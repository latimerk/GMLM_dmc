function [log_m] = logSumExp(log_x, dim)
if(nargin < 2)
    dim = 2;
end


cs = max(log_x, [], dim);
log_x = log_x - cs;

if(isnumeric(dim) && dim == 1 && isa(log_x,'single'))
    %converts to double for numerical saftey without converting the entire matrix of log_x - just one column at a time
    log_m = cs;
    for ii = 1:numel(log_m)
        log_m(ii) = log_m(ii) + log(sum(exp(double(log_x(:,ii)))));
    end
elseif(isnumeric(dim) && dim == 2 && isa(log_x,'single'))
    log_m = cs;
    for ii = 1:numel(log_m)
        log_m(ii) = log_m(ii) + log(sum(exp(double(log_x(ii,:)))));
    end
else
    log_m = cs +log(sum(exp(log_x),dim));
end


%m = exp(cs)*(sum(exp(log_x),dim));