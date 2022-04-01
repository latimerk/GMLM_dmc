function [f,g,nh_2] = poissExpLL(x, y, dt)

x = x + log(dt);
rate  = kgmlm.utils.boundedExp(x);

f = -rate + y.*x;

if(nargout > 1)
    g = -rate + y;
    if(nargout > 2)
        nh_2 = kgmlm.utils.boundedExp(0.5*x);
    end
end