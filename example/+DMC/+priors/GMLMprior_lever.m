function [results_group] = GMLMprior_lever(params, results, groupNum, levPrior_setup)
if(isempty(params))
    results_group = 1;
    return;
end
params_group = params.Groups(groupNum);
results_group = results.Groups(groupNum);

if(numel(params_group.H) ~= 1)
    error('Incorrect number of hyperparameters')
end

results_group.log_prior_VT = -1/2*sum(double(params_group.V.^2), 'all');
if(~isempty(results_group.dV))
    results_group.dV = results_group.dV-params_group.V;
end

if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
    results_group.dH(:) = 0;
end

%% derivatives and prior for any non-normalized components
dim_S = numel(params_group.T);
for ss = 1:dim_S
    
    if(strcmpi(params_group.dim_names(ss), "timing"))
        %% i.i.d. normal with H as the log sigma term for the temporal kernels
        
        H  = double(params_group.H);
        wl = double(params_group.T{ss});
        
        %% log prior
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            [lp, dlp_w, dlp_h] = DMC.priors.simpleGaussianPrior(wl, H);
        elseif(~isempty(results_group.dT{ss}))
            [lp, dlp_w] = DMC.priors.simpleGaussianPrior(wl, H);
        else
            lp = DMC.priors.simpleGaussianPrior(wl, H);
        end

        %% log hyperprior
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            [lp_H, dlp_H_hyper] = DMC.priors.halfTPrior(H, levPrior_setup.hyperprior.nu);
        else
            lp_H = DMC.priors.halfTPrior(H, levPrior_setup.hyperprior.nu);
        end

        %% sum the two together
        if(~isempty(results_group.dT{ss}))
            results_group.dT{ss} = results_group.dT{ss} + dlp_w;
        end
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            results_group.dH(:) = dlp_h + dlp_H_hyper;
        end
        results_group.log_prior_VT = results_group.log_prior_VT + lp + lp_H;
        
    else
        %% standard i.i.d. normal prior for any unknown coefficients
        results_group.log_prior_VT = results_group.log_prior_VT - 1/2*sum(double(params_group.T{ss}).^2, 'all');

        if(~isempty(results_group.dT{ss}))
            results_group.dT{ss} = results_group.dT{ss} - params_group.T{ss};
        end
    end
end