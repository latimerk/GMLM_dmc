function [results_group] = GMLMprior_stim(params, results, groupNum, stimPrior_setup)
if(isempty(params))
    results_group = 1;
    return;
end
params_group = params.Groups(groupNum);
results_group = results.Groups(groupNum);

%modified from computeGLMPrior
%% i.i.d. unit normal prior on loading matrix (no hyperparams)
results_group.log_prior_VT = -1/2*sum(params_group.V.^2, 'all');
if(~isempty(results_group.dV))
    results_group.dV = results_group.dV - params_group.V;
end

if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
    results_group.dH(:) = 0;
end

%% for each stim dimension
dim_S = numel(params_group.T);
for ss = 1:dim_S
    %% main prior for the stimulus info dimension in the task: direction and category loadings
    if(endsWith(params_group.dim_names(ss), "stim"))
        %%
        H = double(params_group.H(:));
        ws = double(params_group.T{ss});
        
        %% get log prior
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            [lp, dlp_w, dlp_h] = DMC.priors.stimulusGaussianPrior(ws, H, stimPrior_setup, false);
        elseif(~isempty(results_group.dT{ss}))
            [lp, dlp_w] = DMC.priors.stimulusGaussianPrior(ws, H, stimPrior_setup, false);
        else
            lp = DMC.priors.stimulusGaussianPrior(ws, H, stimPrior_setup, false);
        end
        
        %% get log hyperprior
        
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            [lp_H, dlp_H_hyper] = DMC.priors.halfTPrior(H, stimPrior_setup.hyperprior.nu);
        else
            lp_H = DMC.priors.halfTPrior(H, stimPrior_setup.hyperprior.nu);
        end

        %% sum the two together
        if(~isempty(results_group.dT{ss}))
            results_group.dT{ss} = results_group.dT{ss} + dlp_w;
        end
        if(isfield(results_group, 'dH') && ~isempty(results_group.dH))
            results_group.dH(:) = dlp_h + dlp_H_hyper;
        end
        results_group.log_prior_VT = results_group.log_prior_VT + lp + sum(lp_H);
        
    elseif(~startsWith(params_group.dim_names(ss), "np_"))
        %% i.i.d. standard normal prior on any remaining dimensions (no hyperparams)
        results_group.log_prior_VT = results_group.log_prior_VT - 1/2 * sum(params_group.T{ss}.^2, 'all');
        
        if(~isempty(results_group.dT{ss}))
            results_group.dT{ss} = results_group.dT{ss} - params_group.T{ss};
        end
    end
end
