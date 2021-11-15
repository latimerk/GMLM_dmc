function [prior_setup] = getNscales(prior_setup, dim_T, dim_P)

D = numel(prior_setup.T) + 1;
for ss = 1:D
    if(ss == 1) % V
        prior_setup_ss = prior_setup.V.setup;
        dim_U = dim_P;
    else %T{ss-1}
        prior_setup_ss = prior_setup.T(ss-1).setup;
        dim_U = dim_T(ss-1);
    end

    NS = 0;
    NC = numel(prior_setup_ss.parts);
    for cc = 1:NC
        type = prior_setup_ss.parts(cc).type;
        if(strcmpi(type, "all_group"))
            NS = NS + 1;
        elseif(strcmpi(type, "all_independent"))
            NS = NS + dim_U;
        elseif(strcmpi(type, "group"))
            NS = NS + 1;
        elseif(strcmpi(type, "std"))
            NS = NS + 0;
        else
            error("Invalid covariance type");
        end
    end

    if(ss == 1) % V}
        prior_setup.V.N_scales = NS;
    else %T{ss-1}
        prior_setup.T(ss-1).N_scales = NS;
    end
end