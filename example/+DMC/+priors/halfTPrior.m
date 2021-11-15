function [lp_log_x, dlp_log_x, d2lp_log_x] = halfTPrior(log_x, nu, scale)
% x = exp(log_x) is a standard deviation (or similary exp transformed positive parameters) with distribution half-t distribution of scale and nu degrees of
% freedom
%
%when nu = 1, is half cauchy

lp_log_x   = zeros(size(log_x));
dlp_log_x  = zeros(size(log_x));
d2lp_log_x = zeros(size(log_x));

if(nargin < 3 || isempty(scale))
    log_scale = 0;
else
    log_scale = log(scale);
end

for ii = 1:numel(log_x)
    if(isscalar(nu))
        nu_c = nu;
    else
        nu_c = nu(ii);
    end
    if(isscalar(log_scale))
        log_scale_c = log_scale;
    else
        log_scale_c = log_scale(ii);
    end
    
    if(log_x(ii)*2 - log(nu_c) - 2*log_scale_c > 30)
        %% numerically safer(?)
        %log hyperprior
        lp_log_x(ii) = -(nu_c + 1)/2 .* (log_x(ii)*2 - log(nu_c) - 2*log_scale_c) + log_x(ii); %the plus log_x(ii) is for the exp transform
        if(nargout > 1)
            dlp_log_x(ii) = -(nu_c + 1) + 1; 
        end
        if(nargout > 2)
            d2lp_log_x(ii) = 0; 
        end
    else
        %log hyperprior
        sig2 = exp(2 * log_x(ii) - 2*log_scale_c);
        lp_log_x(ii) = -(nu_c + 1)/2. * log1p(sig2 / nu_c)  + log_x(ii); %the plus log_x(ii) is for the exp transform
        if(nargout > 1)
            %derivative of log hyperprior (w.r.t. H)
            dlp_log_x(ii) = -(nu_c + 1) * (sig2 / (sig2 + nu_c)) + 1;
        end
        if(nargout > 2)
            d2lp_log_x(ii) = -(nu_c + 1) * 2 * sig2 * nu_c ./ (nu_c + sig2).^2; 
        end
    end
    lp_log_x(ii) = lp_log_x(ii) + log(2) + gammaln((nu_c + 1) / 2) - gammaln(nu_c/2) - 1/2*log(pi*nu_c) - log_scale_c; % log normalization constant
end