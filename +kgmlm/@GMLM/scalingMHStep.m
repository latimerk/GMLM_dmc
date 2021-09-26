% A tensor component-wise rescaling Metropolis-Hastings step.
% Useful for slighlty faster HMC mixing - plain HMC doesn't force any components to be normalized/orthogonal and this explores the scaling dimension faster.
% Important: This step assumes that the prior distributions are normal with mean 0!
function [params, acceptedProps, log_p_accept] = scalingMHStep(obj, params, MH_scaleSettings, optStruct)

MH_ctr = 1;

log_p_accept = nan(100,1);
acceptedProps = nan(100,1);

opts   = obj.getComputeOptionsStruct(false);


results = obj.getEmptyResultsStruct(opts);

if(MH_scaleSettings.sig > 0)
    for jj = 1:obj.dim_J
        results = obj.computeLogPrior(params, opts, results);
        
        %% MH for rescaling tensor components
        sd  = MH_scaleSettings.sig;
        
        for rr = 1:obj.dim_R(jj) %rescaling is component-wise
            
            log_scale = randn(obj.dim_S(jj), 1) * sd;
            if(nargin > 3 && ~isempty(optStruct))
                log_scale(~optStruct.Groups(jj).dT) = 0;
            end
            scale_T = exp(     log_scale); %effictively, this proposal is a random walk on the norms of the T vectors: this proposal is thus a log norm with mu = log(norm(T(:,rr)), sig = sd
            scale_V = exp(-sum(log_scale));
            
            dim_T = zeros(obj.dim_S(jj)+1, 1);
            
            params2 = params;
            params2.Groups(jj).V(:,rr) = params2.Groups(jj).V(:,rr) * scale_V;
            dim_T(end) = size(params2.Groups(jj).V(:,rr),1);
            for ss = 1:obj.dim_S(jj)
                dim_T(ss) = size(params2.Groups(jj).T{ss},1);
                params2.Groups(jj).T{ss}(:,rr) = params2.Groups(jj).T{ss}(:,rr) * scale_T(ss);
            end
            
            results2 = results;
            results2 = obj.computeLogPrior(params2, opts, results2);
            
            chi_correction = sum([log_scale;-sum(log_scale)].*(dim_T-1)) + sum(log_scale); 
                %this first sum changes the priors over the weights to chi distributions over the norm given weights (log difference between proposal and original spot)
                %the second sum is for the change of variables to by P(T,U, T*U*V) because we have the product of the scale fixed
                %  The probability density we sample over then become P(T,U| T*U*V) \propto P(T, U, T*U*V)
            
            log_p_accept(MH_ctr) = results2.Groups(jj).log_prior_VT - results.Groups(jj).log_prior_VT + chi_correction + sum(log_scale); % log_scale is log(q(pg|pg2)/q(pg2|pg)) - this isn't 0 because of the log normal transform!
            if(log(rand) < log_p_accept(MH_ctr))
                acceptedProps(MH_ctr) = true;
                params  = params2;
                results = results2;
            else
                acceptedProps(MH_ctr) = false;
            end
            MH_ctr = MH_ctr + 1;
        end
    end
end

acceptedProps     = acceptedProps(1:(MH_ctr-1));
log_p_accept = log_p_accept(1:(MH_ctr-1));