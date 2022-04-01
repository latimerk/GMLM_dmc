%% gets combinations of parameters to optimize for MLE/MAP estimates
function [optSetup, opts_empty] = getOptimizationSettings(obj, alternating_opt, trial_weights, optStruct, includeHyperparams)
if(nargin < 5)
    includeHyperparams = false;
end

if(nargin < 4 || isempty(optStruct))
    optStruct = obj.getComputeOptionsStruct(true, 'trial_weights', trial_weights, 'includeHyperparameters', includeHyperparams);
end

opts_empty  = obj.getComputeOptionsStruct(false, 'trial_weights', trial_weights, 'includeHyperparameters', includeHyperparams);
    
if(alternating_opt && includeHyperparams)
    error("Invalid setup!");
end

if(alternating_opt)
    % setup subsets of variables to optimize alternatingly such that each optimization will be concave

    tensorOrders = obj.dim_S + 1;

    if(prod(tensorOrders) <= 16 && obj.dim_J <= 8)
        % if not too many combinations, does a lot of conditional GLMs (all possible such that adding another variable will keep it a GLM)
        glm_combs = kgmlm.utils.getCombs(tensorOrders);

        optSetup = repmat(opts_empty, [size(glm_combs, 2) 1]);
        for ss = 1:size(glm_combs, 2)
            optSetup(ss).dW = true & optStruct.dW;
            optSetup(ss).dB = obj.dim_B > 0 & optStruct.dB;

            for jj = 1:obj.dim_J
                if(glm_combs(jj,ss) > obj.dim_S(jj))
                    optSetup(ss).Groups(jj).dV = true & optStruct.Groups(jj).dV;
                else
                    optSetup(ss).Groups(jj).dT(glm_combs(jj,ss)) = true & optStruct.Groups(jj).dT(glm_combs(jj,ss)) ;
                end
            end
        end

    else
        %just does each order
        optSetup = repmat(opts_empty, [max(tensorOrders) 1]);
        for ss = 1:max(tensorOrders)
            optSetup(ss).dW = true & optStruct.dW;
            optSetup(ss).dB = obj.dim_B > 0 & optStruct.dB;

            for jj = 1:obj.dim_J
                if(ss > obj.dim_S(jj))
                    optSetup(ss).Groups(jj).dV = true & optStruct.Groups(jj).dV;
                else
                    optSetup(ss).Groups(jj).dT(ss) = true & optStruct.Groups(jj).dT(ss);
                end
            end
        end
    end
else
    optSetup  = obj.getComputeOptionsStruct(true, 'includeHyperparameters', includeHyperparams);
    optSetup.dW = true & optStruct.dW;
    optSetup.dB = obj.dim_B > 0 & optStruct.dB;

    for jj = 1:obj.dim_J
        optSetup.Groups(jj).dV = true & optStruct.Groups(jj).dV;
        for ss = 1:obj.dim_S(jj)
            optSetup.Groups(jj).dT(ss) = true & optStruct.Groups(jj).dT(ss);
        end
    end
end

end

