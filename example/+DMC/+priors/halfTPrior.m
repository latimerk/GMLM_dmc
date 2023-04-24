function [lp_log_x, dlp_log_x, d2lp_log_x, dlp_log_scale] = halfTPrior(log_x, nu, scale, is_log)
% x = exp(log_x) is a standard deviation (or similary exp transformed positive parameters) with distribution half-t distribution of scale and nu degrees of
% freedom
%
%when nu = 1, is half cauchy

if(nargin < 3 || isempty(scale))
    log_scale = 0;
elseif(nargin < 4 || ~is_log)
    log_scale = log(scale);
else
    log_scale = scale;
end

gt = log_x*2 - log(nu) - 2*log_scale > 30;
any_gt = any(gt);
any_lt = any(~gt);
if(isscalar(nu))
    if(all(gt))
        gt_nu = 1;
        lt_nu = [];
    elseif(any_gt)
        gt_nu = 1;
        lt_nu = 1;
    else
        gt_nu = [];
        lt_nu = 1;
    end
else
    gt_nu = gt;
    lt_nu = ~gt;
end
if(isscalar(log_scale))
    if(all(gt))
        gt_log_scale = 1;
        lt_log_scale = [];
    elseif(any_gt)
        gt_log_scale = 1;
        lt_log_scale = 1;
    else
        gt_log_scale = [];
        lt_log_scale = 1;
    end
    
else
    gt_log_scale = gt;
    lt_log_scale = ~gt;
end


sig2 = exp(2 * (log_x(~gt) - log_scale(lt_log_scale)));
lp_log_x   = zeros(size(log_x));
if(any_gt)
    lp_log_x( gt) = -(nu(gt_nu) + 1)./2 .* (log_x(gt)*2 - log(nu(gt_nu)) - 2*log_scale(gt_log_scale)) + log_x(gt);
end
if(any_lt)
    lp_log_x(~gt) = -(nu(lt_nu) + 1)./2 .* log1p(sig2 ./ nu(lt_nu))  + log_x(~gt); %the plus log_x(ii) is for the exp transform
end
if(nargout > 1)
    %derivative of log hyperprior (w.r.t. H)
    dlp_log_x  = zeros(size(log_x));
    if(any_gt)
        dlp_log_x( gt) = -(nu(gt_nu) + 1) + 1; 
    end
    if(any_lt)
        dlp_log_x(~gt) = -(nu(lt_nu) + 1) .* (sig2 ./ (sig2 + nu(lt_nu))) + 1;
    end
end
if(nargout > 2)
    d2lp_log_x = zeros(size(log_x));
    if(any_gt)
        d2lp_log_x( gt) = 0; 
    end
    if(any_lt)
        d2lp_log_x(~gt) = -2*(nu(lt_nu) + 1) .* sig2 .* nu(lt_nu) ./ (nu(lt_nu) + sig2).^2; 
    end
end
if(nargout > 3)
    dlp_log_scale   = zeros(size(log_x));
    if(any_gt)
        dlp_log_scale( gt) = (nu(gt_nu) + 1); 
    end
    if(any_lt)
        dlp_log_scale(~gt) = (nu(lt_nu) + 1) .* (sig2 ./ (sig2 + nu(lt_nu)));
    end
    dlp_log_scale = dlp_log_scale - 1; % for the exp transform
end
lp_log_x = lp_log_x + log(2) + gammaln((nu + 1) ./ 2) - gammaln(nu./2) - 1/2*log(pi*nu) - log_scale; % log normalization constant
