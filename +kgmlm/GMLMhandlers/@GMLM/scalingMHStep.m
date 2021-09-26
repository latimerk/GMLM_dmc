function [paramStruct, acceptedProps, log_p_accept] = scalingMHStep(obj, paramStruct, HM_scaleSettings)
%This step assumes that the prior distributions are normal with mean 0!

MH_ctr = 1;

log_p_accept = nan(100,1);
acceptedProps = nan(100,1);

optsStruct   = obj.getEmptyOptsStruct(false, false);
optsStruct.compute_trialLL = true;
%resultStruct = gmlm.computeLPost(paramStruct, optsStruct, true);
resultStruct = obj.getEmptyResultsStruct(optsStruct);

if(HM_scaleSettings.sig > 0)
    for jj = 1:obj.dim_J
        resultStruct.Groups(jj) = obj.Groups(jj).addLPrior(paramStruct.Groups(jj), resultStruct.Groups(jj));
        
        %% MH for rescaling tensor components
        dim_S = double(obj.Groups(jj).dim_S);
        
        sd  = HM_scaleSettings.sig;
        
        for rr = 1:obj.Groups(jj).dim_R %rescaling is component-wise
            log_scale = randn(dim_S, 1) * sd;
            scale_T = exp(     log_scale); %effictively, this proposal is a random walk on the norms of the T vectors: this proposal is thus a log norm with mu = log(norm(T(:,rr)), sig = sd
            scale_V = exp(-sum(log_scale));
            
            dim_T = zeros(dim_S+1,1);
            
            paramStruct2 = paramStruct;
            paramStruct2.Groups(jj).V(:,rr) = paramStruct2.Groups(jj).V(:,rr) * scale_V;
            dim_T(end) = size(paramStruct2.Groups(jj).V(:,rr),1);
            for ss = 1:dim_S
                dim_T(ss) = size(paramStruct2.Groups(jj).T{ss},1);
                paramStruct2.Groups(jj).T{ss}(:,rr) = paramStruct2.Groups(jj).T{ss}(:,rr) * scale_T(ss);
            end
            
            resultStruct2 = resultStruct;
            resultStruct2.Groups(jj) = obj.Groups(jj).addLPrior(paramStruct2.Groups(jj), resultStruct2.Groups(jj));
            
            chi_correction = sum([log_scale;-sum(log_scale)].*(dim_T-1)) + sum(log_scale); 
                %this first sum changes the priors over the weights to chi distributions over the norm given weights (log difference between proposal and original spot)
                %the second sum is for the change of variables to by P(T,U, T*U*V) because we have the product of the scale fixed
                %  The probability density we sample over then become P(T,U| T*U*V) \propto P(T, U, T*U*V)
            
            log_p_accept(MH_ctr) = resultStruct2.Groups(jj).log_prior - resultStruct.Groups(jj).log_prior + chi_correction + sum(log_scale); % log_scale is log(q(pg|pg2)/q(pg2|pg)) - this isn't 0 because of the log normal transform!
            if(log(rand) < log_p_accept(MH_ctr))
                acceptedProps(MH_ctr) = true;
                paramStruct  = paramStruct2;
                resultStruct = resultStruct2;
            else
                acceptedProps(MH_ctr) = false;
            end
            MH_ctr = MH_ctr + 1;
        end
    end
end

acceptedProps     = acceptedProps(1:(MH_ctr-1));
log_p_accept = log_p_accept(1:(MH_ctr-1));