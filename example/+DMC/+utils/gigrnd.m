function X = gigrnd(lambda, psi, chi)
%% Generalized inverse Gaussian random number generator
% generate random sample from the three parameter GIG distribution with
% density given by
% f_GIG(x) = 0.5*(psi/chi)^(lambda/2)/besselk(lambda,sqrt(psi*chi)) * x^(lambda-1)
%           * exp( -0.5*(chi/x + psi*x) )
% 
% this function only generates a single random variate!!
% based on the R function GIGrvg of 
% Hörmann, W. and Leydold, J., 2015. GIGrvg: Random variate generator for
% the GIG distribution. R package version 0.4.
% modified and translated to Matlab by Jan P. Hartkopf
% hartkopf (at) wiso.uni-koeln.de
%
%  Jan Patrick Hartkopf (2021). gigrnd (https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd), MATLAB Central File Exchange. Retrieved October 22, 2021. 
ZTOL = 10*eps;
if chi == 0, chi = 10*realmin; end
if psi == 0, psi = 10*realmin; end
if chi < ZTOL
    %% Special cases which are basically Gamma and Inverse Gamma distribution
    if lambda > 0
        % Gamma
        X = 2*randg(lambda)/psi;
    else
        % GIG or Inverse Gamma through rescaling
        X = gigrnd3(lambda,psi*chi,1)*chi;
    end
elseif psi < ZTOL
    %% Special cases which are basically Gamma and Inverse Gamma distribution
    if lambda > 0
        % GIG or Gamma through rescaling
        X = gigrnd3(lambda,1,psi*chi)/psi;
    else
        % Inverse Gamma
        X = 0.5*chi/randg(-lambda);
    end
    
else
    %% Sample from the GIG distribution
    lambda_old = lambda;
    if lambda < 0, lambda = -lambda; end
    alpha = sqrt(chi/psi);
    omega = sqrt(chi*psi);
    
    if (lambda > 2 || omega > 3)
        % Ratio of uniforms with shift by 'mode'
        X = ROU_shift(lambda, lambda_old, omega, alpha);
    elseif (lambda >= 1-2.25*omega^2 || omega > 0.2)
        % Ratio of uniforms without shift
        X = ROU_noshift(lambda, lambda_old, omega, alpha);
    elseif (lambda >= 0 && omega > 0)
        % alternative approach
        X = new_approach(lambda, lambda_old, omega, alpha);
    end
end
end % gigrnd3
function mode = gig_mode(lambda, omega)
%% mode of the GIG distribution
if lambda >= 1
    mode = (sqrt((lambda-1)^2 + omega^2) + lambda-1)/omega;
else
    mode = omega / (sqrt((1-lambda)^2 + omega^2) + 1-lambda);
end
end % gig_mode
function X = ROU_shift(lambda, lambda_old, omega, alpha)
%% Ratio-of-uniforms with shift by mode
% shortcuts
t = 0.5*(lambda-1);
s = 0.25*omega;
% mode
xm = gig_mode(lambda, omega);
% normalization constant
nc = t*log(xm) - s*(xm + 1/xm);
% location of minimum and maximum
% compute coefficients of cubic equation y^3+a*y^2+b*y+c=0
a = -(2*(lambda+1)/omega + xm); % < 0
b = (2*(lambda-1)*xm/omega - 1);
c = xm;
% we need the roots in (0,xm) and (xm,inf)
% substitute y = z-a/3 for depressed cubic equation z^3+p*z+q=0
p = b - a^2/3;
q = (2*a^3)/27 - (a*b)/3 + c;
% use Cardano's rule
fi = acos(-q/(2*sqrt(-p^3/27)));
fak = 2*sqrt(-p/3);
y1 = fak*cos(fi/3) - a/3;
y2 = fak*cos(fi/3 + 4/3*pi) - a/3;
% boundaries of minimal bounding rectangle:
% upper boundary: vmax = 1
% left hand boundary: uminus
% right hand boundary: uplus
uplus = (y1-xm) * exp(t*log(y1) - s*(y1 + 1/y1) - nc);
uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1/y2) - nc);
% generate random variate
done = false;
while(~done)
    U = uminus + (uplus-uminus)*rand; % U(u-,u+)
    V = rand; % U(0,1)
    X = U/V + xm;
    
    % acceptance / rejection
    done = ~(X <= 0 || log(V) > (t*log(X) - s*(X + 1/X) - nc));
end
% store random variate
if lambda_old < 0
    X = alpha/X;
else
    X = alpha*X;
end
end % ROU_shift
function X = ROU_noshift(lambda, lambda_old, omega, alpha)
%% Ratio-of-uniforms without shift
% shortcuts
t = 0.5*(lambda-1);
s = 0.25*omega;
% mode
xm = gig_mode(lambda, omega);
% normalization constant
nc = t*log(xm) - s*(xm + 1/xm);
% location of maximum
% we need the positive root of omega/2*y^2 - (lambda+1)*y - omega/2 = 0
ym = ((lambda+1) + sqrt((lambda+1)^2 + omega^2))/omega;
% boundaries of minimal bounding rectangle
% upper boundary: vmax = 1
% left hand boundary: umin = 0
% right hand boundary:
um = exp(0.5*(lambda+1)*log(ym) - s*(ym + 1/ym) - nc);
% generate random variate
done = false;
while(~done)
    U = um*rand; % U(0,um)
    V = rand; % U(0,vmax)
    X = U/V;
    
    % acceptance / rejection
    done = ( log(V) <= (t*log(X) - s*(X + 1/X) - nc) );
end
% store random variate
if lambda_old < 0
    X = alpha/X;
else
    X = alpha*X;
end
end % ROU_noshift
function X = new_approach(lambda, lambda_old, omega, alpha)
%%
% setup
% mode
xm = gig_mode(lambda, omega);
% splitting point
x0 = omega/(1-lambda);
% domain [0, x0]
k0 = exp((lambda-1)*log(xm) - 0.5*omega*(xm + 1/xm));
A0 = k0*x0;
if (x0 >= 2/omega) % [x0, infinity]
    k1 = 0;
    A1 = 1;
    k2 = x0^(lambda-1);
    A2 = k2*2*exp(-omega*x0/2)/omega;
else
    % domain [x0, 2/omega]
    k1 = exp(-omega);
    if (lambda == 0)
        A1 = k1*log(2/omega^2);
    else
        A1 = k1/lambda * ( (2/omega)^lambda - x0^lambda );
    end
    
    % domain [2/omega, infinity]
    k2 = (2/omega)^(lambda-1);
    A2 = k2*2*exp(-1)/omega;
end
% total area
Atot = A0 + A1 + A2;
% generate sample
while 1
    V = Atot*rand;
    
    while 1
        % domain [0, x0]
        if (V <= A0)
            X = x0*V/A0;
            hx = k0;
            break;
        end
        
        % domain [x0, 2/omega]
        V = V - A0;
        if (V <= A1)
            if (lambda == 0)
                X = omega*exp(exp(omega)*V);
                hx = k1/X;
            else
                X = (x0^lambda + lambda/k1*V)^(1/lambda);
                hx = k1*X^(lambda-1);
            end
            break;
        end
        
        % domain [max(x0,2/omega), infinity]
        V = V - A1;
        if (x0 > 2/omega)
            a = x0;
        else
            a = 2/omega;
        end
        X = -2/omega * log(exp(-omega/2*a) - omega/(2*k2)*V);
        hx = k2*exp(-omega/2 * X);
        break;
    end
    
    % acceptance / rejection
    U = rand*hx;
    if (log(U) <= (lambda-1)*log(X) - omega/2*(X + 1/X))
        break
    end
end
% store random variate
if lambda_old < 0
    X = alpha/X;
else
    X = alpha*X;
end
end % new_approach
%% end of file