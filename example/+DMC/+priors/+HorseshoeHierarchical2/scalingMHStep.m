% A tensor component-wise rescaling Metropolis-Hastings step.
% Useful for slighlty faster HMC mixing - plain HMC doesn't force any components to be normalized/orthogonal and this explores the scaling dimension faster.
% Important: This step assumes that the prior distributions are normal with mean 0!
function [params, acceptedProps, log_p_accept] = scalingMHStep(gmlm, params, optStruct, sampleNum, groupNum, MH_scaleSettings, prior_setup)


log_p_accept = nan(100,MH_scaleSettings.N);
acceptedProps = nan(100,MH_scaleSettings.N);

% scalable_V = optStruct.Groups(groupNum).dV;
if(optStruct.Groups(groupNum).dV && nargin > 6 && isfield(prior_setup, 'V') && isfield(prior_setup.V, 'mu'))
    scalable_V = all(prior_setup.V.mu == 0, 'all');
else
    scalable_V = optStruct.Groups(groupNum).dV ;
end
scalable_T = optStruct.Groups(groupNum).dT;
for ss = 1:gmlm.dim_S(groupNum)
    if(scalable_T(ss) && nargin > 6 &&isfield(prior_setup, 'T') && isfield(prior_setup.T, 'mu'))
        scalable_T(ss) = all(prior_setup.T(ss).mu == 0, 'all');
    end
end

Ts = find(scalable_T);
Ts = Ts(:)';

totalScale = scalable_V + sum(scalable_T);
if(mod(sampleNum, MH_scaleSettings.sample_every) ~= 0 || totalScale < 2 || MH_scaleSettings.sig <= 0 || MH_scaleSettings.N <= 0)
    return;
end
if(scalable_V)
    order = [Ts 0];
else
    order = Ts;
    order(end) = -order(end);
end

R = gmlm.dim_R(groupNum);
lp = zeros(R,1);


%%

for nn = 1:MH_scaleSettings.N
    MH_ctr = 1;
    %results = gmlm.computeLogPrior(params, opts, results);

    %% MH for rescaling tensor components
    sd  = MH_scaleSettings.sig;

    for rr = 1:R %rescaling is component-wise
        if(nn == 1)
            lp(rr) = lprior(params.Groups(groupNum), order, rr);
        end

        log_scale = randn(totalScale - 1, 1) * sd;
        scale_T = exp(     log_scale); %effictively, this proposal is a random walk on the norms of the T vectors: this proposal is thus a log norm with mu = log(norm(T(:,rr)), sig = sd
        scale_V = exp(-sum(log_scale));
        
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

        %results2 = results;
        %results2 = gmlm.computeLogPrior(params2, opts, results2);
        lp2 = lprior(params2.Groups(groupNum), order, rr);

        %chi_correction = sum([log_scale;-sum(log_scale)].*(dim_T-2)) - sum(log_scale); 
            %this first sum changes the priors over the weights to chi distributions over the norm given weights (log difference between proposal and original spot)
            %the second sum is for the change of variables to P(T,U, T*U*V) because we have the product of the scale fixed
            %  The probability density we sample over then become P(T,U| T*U*V) \propto P(T, U, T*U*V)

        log_q_rat = sum(log_scale); %sum(log(lognpdf(c, log_c + log_scale , sd)) - log(lognpdf(c*exp(log_scale), log(c) , sd))); % c is original vector length
        %log_p_accept(MH_ctr, nn) = results2.Groups(groupNum).log_prior_VT - results.Groups(groupNum).log_prior_VT + chi_correction + log_q_rat; 
        log_p_accept(MH_ctr, nn) = lp2 - lp(rr) + log_q_rat;% + chi_correction;
        if(log(rand) < log_p_accept(MH_ctr, nn))
            acceptedProps(MH_ctr, nn) = true;
            params  = params2;
            %results = results2;
            lp(rr) = lp2;
        else
            acceptedProps(MH_ctr, nn) = false;
        end
        MH_ctr = MH_ctr + 1;
    end

end

acceptedProps = acceptedProps(1:(MH_ctr-1),:);
log_p_accept  = log_p_accept(1:(MH_ctr-1),:);


end

function [lp] = lprior(params, order, rr)
    lp = 0;
    for ss_idx = 1:numel(order)
        ss = order(ss_idx);
        if(ss == 0)
            x = sqrt(sum(params.V(:,rr).^2, "all"));
            k = size(params.V(:,rr),1);
        elseif(ss < 0)
            x = sqrt(sum(params.T{-ss}(:,rr).^2, "all"));
            k = size(params.T{-ss}(:,rr),1);
        else
            x = sqrt(sum(params.T{ss}(:,rr).^2, "all"));
            k = size(params.T{ss}(:,rr),1);
        end
        lp = lp + (k - 1)*log(x) - x^2/2; % chi distribution
%         lp = lp - x/2;
    end
end