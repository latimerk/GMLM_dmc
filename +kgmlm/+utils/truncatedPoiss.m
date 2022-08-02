function [ll, p0, log_p0] = truncatedPoiss(rr, Y)

ll = zeros(size(Y));

Y1 = Y > 0;
Y0 = Y == 0;

rl = rr < -30;
ll(Y1 & rl)  = rr(Y1 & rl);
ll(Y1 & ~rl) = log(1-exp(-exp(rr(Y1 & ~rl))));

ll(Y0) = -exp(rr(Y0));

if(nargout > 1)
    log_p0 = -exp(rr);
    p0 = exp(log_p0);
end
