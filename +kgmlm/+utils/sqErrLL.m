function [f,g,nh_2] = sqErrLL(x, y, dt)

d = x - y;
f = -d.^2;
if(nargout > 1)
    g = -2*d;
    if(nargout > 2)
        nh_2 = ones(size(x))*sqrt(2);
    end
end