function [lp_log_x, dlp_log_x, d2lp_log_x] = inverseGammaPrior(log_x, a, b)
% x = exp(log_x) is an inverse Gamma
%

lp_log_x   = zeros(size(log_x));


log_x = log_x(:);
invx = DMC.utils.boundedExp(-log_x);
a = a(:);
b = b(:);


lp_log_x(:) = a .* log(b) - gammaln(a) + (-a-1).*log_x - b.*invx + log_x;
if(nargout > 1)
    dlp_log_x  =  (-a-1) + b.*invx + 1;
    dlp_log_x = reshape(dlp_log_x, size(log_x));
end

if(nargout > 2)
    d2lp_log_x = -b.*invx;
    d2lp_log_x = reshape(d2lp_log_x, size(log_x));
end