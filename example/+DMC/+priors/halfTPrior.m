function [lp_H, dlp_H_hyper, d2lp_H_hyper] = halfTPrior(H, nu, A)
%H is assumed to be log of a standard deviation (or similary transformed positive parameters)
%
%from Gelman (2006)  "Prior distributions for variance parameters in hierarchical models (Comment on Article by Browne and Draper)" Bayesian Analysis, pg 520
%when nu = 1, is half cauchy

lp_H = zeros(size(H));
dlp_H_hyper = zeros(size(H));
d2lp_H_hyper = zeros(size(H));

if(nargin < 3 || isempty(A))
    A = ones(size(H));
end

for ii = 1:numel(H)
    if(H(ii)*2 - log(nu(ii)) - 2*log(A(ii)) > 30)
        %% numerically safer(?)
        %log hyperprior
        lp_H(ii) = -(nu(ii) + 1)/2 .* (H(ii)*2 - log(nu(ii)) - 2*log(A(ii))) + H(ii); %the plus H is for the exp transform
        if(nargout > 1)
            dlp_H_hyper(ii) = -(nu(ii) + 1) + 1; 
        end
        if(nargout > 2)
            d2lp_H_hyper(ii) = 0; 
        end
    else
        %log hyperprior
        sig2 = exp(2 * H(ii) - 2*log(A(ii)));
        lp_H(ii) = -(nu(ii) + 1)/2. * log1p(sig2 / nu(ii))  + H(ii); %the plus H is for the exp transform
        if(nargout > 1)
            %derivative of log hyperprior (w.r.t. H)
            dlp_H_hyper(ii) = -(nu(ii) + 1) * (sig2 / (sig2 + nu(ii))) + 1;
        end
        if(nargout > 2)
            d2lp_H_hyper(ii) = -(nu(ii) + 1) * 2 * sig2 * nu(ii) ./ (nu(ii) + sig2).^2; 
        end
    end
end