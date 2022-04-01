function [f,g,nh_2] = poissSoftRecLL(x, y, dt)


if(nargout > 2)
    [rate,drate,d2rate] = kgmlm.utils.softrec(x);
    g = -drate*dt + y.*(drate./rate);
    nh_2 = sqrt(max(0, -1 * (-d2rate*dt + y.*((d2rate.*rate - drate.^2)./(rate.*rate)))));
elseif(nargout > 1)
    [rate,drate] = kgmlm.utils.softrec(x);
    g = -drate*dt + y.*(drate./rate);
else
    rate = kgmlm.utils.softrec(x);
end

log_rate = log(rate);
f = -rate*dt + y.*(log_rate + log(dt));


