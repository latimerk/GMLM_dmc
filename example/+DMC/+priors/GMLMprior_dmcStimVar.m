function [sig, dsig, full_sig] = GMLMprior_dmcStimVar(H, prior)
K = zeros(size(prior.S));

if(nargout > 1)
    dK = zeros([size(prior.S,1),size(prior.S,2), numel(H)]);
    
    for ii = 1:size(prior.S, 3)
        hps_idx  = prior.hyperparams_idx{ii};
        type = prior.covTypes(ii);
        
        if(strcmpi(type, "independent"))
            [K(:,:,ii),dK(:,:,hps_idx)] = indGaussianKernel_dmc(prior.S(:,:,ii),H(hps_idx)); 
        elseif(strcmpi(type, "angular_gp"))
        	[K(:,:,ii),dK(:,:,hps_idx)] = polarCovKernel_dmc(prior.S(:,:,ii),H(hps_idx)); %if using GP tuning over stim directions
        else
            error("Unknown type of prior for stimulus kernel prior construction");
        end
    end

    %% projects the derivatives into the parameter space (instead of the higher-dimensional construction space)
    dsig = zeros(size(prior.M,1),size(prior.M,1),numel(H));
    for ii = 1:numel(H)
        dsig(:,:,ii) = prior.M*dK(:,:,ii)*prior.M';
    end
else
    for ii = 1:size(prior.S, 3)
        hps_idx  = prior.hyperparams_idx{ii};
        type = prior.covTypes(ii);
        
        if(strcmpi(type, "independent"))
            K(:,:,ii) = indGaussianKernel_dmc(prior.S(:,:,ii),H(hps_idx)); 
        elseif(strcmpi(type, "angular_gp"))
        	K(:,:,ii) = polarCovKernel_dmc(prior.S(:,:,ii),H(hps_idx)); %if using GP tuning over stim directions
        else
            error("Unknown type of prior for stimulus kernel prior construction");
        end
    end
end

full_sig = sum(K,3);
sig = prior.M*full_sig*prior.M';


end



%% independent Gaussian for the base elements of a filter: filters are linear combinations of these elements
function [K, dK] = indGaussianKernel_dmc(K_0, ps)

if(numel(ps) ~= 1)
    error('indCovKernel: wrong number of parameters!');
end


alpha = exp(min(20, ps(1)*2));
dalpha = 2*alpha;

K = alpha*K_0;
K(isnan(K_0)) = 0;

if(nargout > 1)
    dK = dalpha*K_0;
    dK(isnan(K_0)) = 0;
end
end


function [K, dK] = polarCovKernel_dmc(K,ps)

% Kernel in
% Padonou, Espéran, and Olivier Roustant. 
%   "Polar Gaussian processes and experimental designs in circular domains."
%   SIAM/ASA Journal on Uncertainty Quantification 4.1 (2016): 1014-1033.
%
% equation 3.7 + 3.9 
%NOTE: this is only used for a GP over stim directions. It is not used at all in the sin/cos tuning model

if(numel(ps) ~= 2)
    error('polarCovKernel: wrong number of parameters!');
end


alpha = exp(min(20, ps(1)*2));
dalpha = 2*alpha;

dtau  = exp(min(20, ps(2)));
tau   = dtau+4;


K_0 = K./pi;
mx = max(0,1-K_0);
mt = mx.^tau;
kt = (1+tau*K_0);
K = alpha*kt.*mt;
K(isnan(K_0)) = 0;

if(nargout > 1)
    dmt = mt;
    
    mx0 = mx>0;
    dmt(mx0) = mt(mx0).*log(mx(mx0))*dtau;
    
    dkt = dtau*K_0;
    
    dK_tau = alpha*(dkt.*mt + kt.*dmt);
    dK_tau(isnan(K_0)) = 0;
    
    dK_alpha = dalpha*kt.*mt;
    dK_alpha(isnan(K_0)) = 0;
    dK = cat(3,dK_alpha,dK_tau);
end
end