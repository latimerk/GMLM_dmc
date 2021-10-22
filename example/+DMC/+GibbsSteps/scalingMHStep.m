% A tensor component-wise rescaling Metropolis-Hastings step.
% Useful for slighlty faster HMC mixing - plain HMC doesn't force any components to be normalized/orthogonal and this explores the scaling dimension faster.
% Important: This step assumes that the prior distributions are normal with mean 0!
function [params, acceptedProps, log_p_accept] = scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, stimPrior_setup)


log_p_accept = nan(100,MH_scaleSettings.N);
acceptedProps = nan(100,MH_scaleSettings.N);

opts    = gmlm.getComputeOptionsStruct(false);
results = gmlm.getEmptyResultsStruct(opts);

if(optStruct.Groups(groupNum).dV && nargin > 6 && isfield(stimPrior_setup, 'V') && isfield(stimPrior_setup.V, 'mu'))
    scalable_V = all(stimPrior_setup.V.mu == 0, 'all');
else
    scalable_V = optStruct.Groups(groupNum).dV ;
end
scalable_T = optStruct.Groups(groupNum).dT;
for ss = 1:gmlm.dim_S(groupNum)
    if(scalable_T(ss) && nargin > 6 &&isfield(stimPrior_setup, 'T') && isfield(stimPrior_setup.T, 'mu'))
        scalable_T(ss) = all(stimPrior_setup.T(ss).mu == 0, 'all');
    end
end

Ts = find(scalable_T);
Ts = Ts(:)';

totalScale = scalable_V + sum(scalable_T);
if(mod(sampleNum, MH_scaleSettings.sample_every) ~= 0 || totalScale < 2 || MH_scaleSettings.sig <= 0)
    return;
end

% if(sampleNum < 100)
%     return;
% end

for nn = 1:MH_scaleSettings.N
    MH_ctr = 1;
    results = gmlm.computeLogPrior(params, opts, results);

    %% MH for rescaling tensor components
    sd  = MH_scaleSettings.sig;

    for rr = 1:gmlm.dim_R(groupNum) %rescaling is component-wise

        log_scale = randn(totalScale - 1, 1) * sd;
        scale_T = exp(     log_scale); %effictively, this proposal is a random walk on the norms of the T vectors: this proposal is thus a log norm with mu = log(norm(T(:,rr)), sig = sd
        scale_V = exp(-sum(log_scale));
        
        if(scalable_V)
            order = [Ts 0];
        else
            order = Ts;
            order(end) = -order(end);
        end

        dim_T = zeros(totalScale, 1);

        params2 = params;
        
        for ss_idx = 1:numel(order)
            ss = order(ss_idx);
            if(ss == 0)
                dim_T(ss_idx) = size(params2.Groups(groupNum).V,1);
                params2.Groups(groupNum).V(:,rr) = params2.Groups(groupNum).V(:,rr) * scale_V;
            elseif(ss < 0)
                dim_T(ss_idx) = size(params2.Groups(groupNum).T{-ss},1);
                params2.Groups(groupNum).T{-ss}(:,rr) = params2.Groups(groupNum).T{-ss}(:,rr) * scale_V;
            else
                dim_T(ss_idx) = size(params2.Groups(groupNum).T{ss},1);
                params2.Groups(groupNum).T{ss}(:,rr) = params2.Groups(groupNum).T{ss}(:,rr) * scale_T(ss_idx);
            end
        end

        results2 = results;
        results2 = gmlm.computeLogPrior(params2, opts, results2);

        chi_correction = sum([log_scale;-sum(log_scale)].*(dim_T-1)) + sum(log_scale); 
            %this first sum changes the priors over the weights to chi distributions over the norm given weights (log difference between proposal and original spot)
            %the second sum is for the change of variables to by P(T,U, T*U*V) because we have the product of the scale fixed
            %  The probability density we sample over then become P(T,U| T*U*V) \propto P(T, U, T*U*V)

        log_p_accept(MH_ctr, nn) = results2.Groups(groupNum).log_prior_VT - results.Groups(groupNum).log_prior_VT + chi_correction + sum(log_scale); % log_scale is log(q(pg|pg2)/q(pg2|pg)) - this isn't 0 because of the log normal transform!
        if(log(rand) < log_p_accept(MH_ctr, nn))
            acceptedProps(MH_ctr, nn) = true;
            params  = params2;
            results = results2;
        else
            acceptedProps(MH_ctr, nn) = false;
        end
        MH_ctr = MH_ctr + 1;
    end
end

acceptedProps = acceptedProps(1:(MH_ctr-1),:);
log_p_accept  = log_p_accept(1:(MH_ctr-1),:);