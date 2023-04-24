% gets the number of hyperparams for the horseshoe prior
function [dim_H] = getDimH(R, prior_setup)
dim_H = R + 2;

D = numel(prior_setup.T) + 1; % order
for ss = 1:D
    if(ss == 1) 
        prior_setup_ss = prior_setup.V;
    else %T{ss-1}
        prior_setup_ss = prior_setup.T(ss-1);
    end

    if(prior_setup_ss.on)
        if(isempty(prior_setup_ss.grps))
            NS = prior_setup_ss.dim_T;
        else
            NS = numel(prior_setup_ss.grps);
        end
        dim_H = dim_H + R*NS;
    end
end

