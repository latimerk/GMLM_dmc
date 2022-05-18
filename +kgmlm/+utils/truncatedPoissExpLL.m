function [f,g,nh_2] = truncatedPoissExpLL(x, y, dt)

x = x + log(dt);
if(numel(y) == 1)
    y = repmat(y, size(x));
end
spks = y > 0;

rate  = kgmlm.utils.boundedExp(x);
expNrate = kgmlm.utils.boundedExp(-rate(spks));

f = -rate;
f(spks)  = log(1.0 - expNrate);

if(nargout > 1)
    exm1 = kgmlm.utils.boundedExp(rate(spks)) - 1;
    g = -rate;
    g(spks) = rate(spks)./exm1;

    if(nargout > 2)
        enxm1 = kgmlm.utils.boundedExp(-rate(spks) - 1);
        nh_2 = kgmlm.utils.boundedExp(0.5*x);
        nh_2(spks) = sqrt(-rate(spks).*(1.0-rate(spks) - expNrate)./(exm1 + enxm1));
    end
end
