function [stds, d_log_scales] = scaledGaussianComponent(ws, log_scales_0, grps, dU_0, stds)
% i.i.d. multivariate zero-mean normal prior distribution over columns of ws with covariance matrix
% defined in  GMLMprior_dmcStimVar
%
% calculates prior and all derivatives

    
compute_D1 = nargout > 1;
precomputed_portion = nargin >= 5;
P = size(ws, 1);

H_scales = log_scales_0;

if(~precomputed_portion)
    stds = zeros(size(ws));
end
        
if(compute_D1)
    d_log_scales = zeros(size(log_scales_0)); 
end
    
if(isscalar(grps) && grps < 0)% all one
    S_c = H_scales(1, :);
    
    if(~precomputed_portion)
        stds = repmat(DMC.utils.boundedExp(0.5*S_c), P, 1);
    end
    if(compute_D1)
        d_log_scales(1, :) = sum(dU_0 .* (0.5*stds) .* ws, 1);
    end

elseif(isempty(grps)) % all independent
    S_c = H_scales;

    if(~precomputed_portion)
        stds = DMC.utils.boundedExp(0.5*S_c);
    end
    if(compute_D1)
        d_log_scales(:, :) = dU_0 .* (0.5*stds) .* ws;
    end
    
elseif(iscell(grps))
    ctr = 1;
    for cc = 1:numel(grps)
        S_c = H_scales(ctr, :);
        idx_ps =  grps{cc};
        if(~precomputed_portion)
            stds(idx_ps,:) = repmat(DMC.utils.boundedExp(0.5*S_c), numel(idx_ps), 1);
        end
        if(compute_D1)
            d_log_scales(ctr, :) = sum(dU_0(idx_ps, :) .* (0.5*stds(idx_ps, :)) .* ws(idx_ps,:), 1);
        end
        ctr = ctr + 1;
    end
else
    error("Invalid regularization setup");
end
    



end

