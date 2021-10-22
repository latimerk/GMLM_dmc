%generates dirichlet random variables
%  inputs
%      a = dirichlet parameters. If scalar (alpha = constant for all parts), p is the size of the vector
%      n = number of draws
function r = drchrnd(a,n,p)
if(numel(a) > 1 || nargin < 3)
    p = length(a);
else
    a = repmat(a(1), 1, p);
end

r = gamrnd(repmat(a,n,1),1,n,p);
r(a==0) = 0;
r = r ./ max(repmat(sum(r,2),1,p),1e-35);

for ii = 1:n
    if(abs(sum(r(ii,:))-1) >= 1e-20 || sum(isinf(r(ii,:))) > 0 || sum(isnan(r(ii,:))) > 0)

        %r(a>0) = 0.01;
        r(ii,a == 0)  = 0;
        r(ii,1) = 1- sum(r(ii,2:end),2);
    end
end