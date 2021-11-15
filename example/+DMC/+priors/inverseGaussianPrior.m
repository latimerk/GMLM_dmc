function [lp_log_x, dlp_log_x, d2lp_log_x] = inverseGaussianPrior(log_x, a, b)
% x = exp(log_x) is an inverse Gaussian
%
%when nu = 1, is half cauchy

lp_log_x   = zeros(size(log_x));


log_x = log_x(:);
nx = DMC.utils.boundedExp(-log_x);
a = a(:);
b = b(:);


lp_log_x(:) = a .* gammaln(b_c) - (a+1).*log_x - b.*nx + log_x;
if(nargout > 1)
    dlp_log_x  = zeros(size(log_x));
    dlp_log_x(:) = -(a+1) + b.*nx + 1;
end

if(nargout > 2)
    d2lp_log_x = zeros(size(log_x));
    d2lp_log_x(:) = -b.*nx;
end