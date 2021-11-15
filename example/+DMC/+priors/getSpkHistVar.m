function [params] = getSpkHistVar(params)
params.W(:) = inf;
params.B(:) = DMC.utils.boundedExp(2*params.H(1));
end

