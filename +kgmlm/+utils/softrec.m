function [f,g,h] = softrec(lambda)

xx_g = lambda > 30;

f = lambda;
f(~xx_g) = log1p(exp(lambda(~xx_g)));

if(nargout > 1)
    g = ones(size(f));
    g(~xx_g) = 1./(1+exp(-lambda(~xx_g)));

    if(nargout > 2)
        h = g.*(1-g);

    end
end

% f = arrayfun(@sr_0, lambda);
% if(nargout > 1)
%     g = arrayfun(@drs_0, lambda);
%     if(nargout > 2)
%         h = g.*(1-g);
%     end
% end

end

% function [lambda] = sr_0(lambda)
%     if(lambda <= 30)
%         lambda = log1p(exp(lambda));
%     end
% end
% 
% function [dlambda] = dsr_0(lambda)
%     if(lambda <= 30)
%         dlambda = 1./(1+exp(-lambda));
%     else
%         dlambda = 1;
%     end
% end