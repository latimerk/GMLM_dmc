% gets the number of hyperparams for the horseshoe prior
function [dim_H] = getDimH(rank, prior_setup)

if( prior_setup.V.N_scales > 0)
    NS = prior_setup.V.N_scales;
else
    NS = 0;
end
if(~isfield(prior_setup.V,'N_hyperparams'))
    NH = 0;
elseif(isa(prior_setup.V.N_hyperparams, 'function_handle'))
    NH = prior_setup.V.N_hyperparams(rank);
else
    NH = prior_setup.V.N_hyperparams;
end

for jj = 1:numel(prior_setup.T)
    if( prior_setup.T(jj).N_scales > 0)
        NS = NS + prior_setup.T(jj).N_scales;
    else
        NS = NS + 0;
    end
    if(~isfield(prior_setup.T(jj),'N_hyperparams'))
        NH = NH + 0;
    elseif(isa(prior_setup.T(jj).N_hyperparams, 'function_handle'))
        NH = NH + prior_setup.T(jj).N_hyperparams(rank);
    else
        NH = NS + prior_setup.T(jj).N_hyperparams;
    end
end
if(rank == 1)
    dim_H = NS + 1 + NH;
else
    dim_H = rank*(NS) + NS*prior_setup.crossrankwise_psi + prior_setup.regularized + rank*prior_setup.rankwise_tau + ~prior_setup.rankwise_tau + NH;
end
