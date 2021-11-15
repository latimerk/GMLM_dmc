function [params_group] = getPriorVariance(params_group, prior_setup)

dim_S = numel(params_group.T);
params_group.V(:) = 1;
for ss = 1:dim_S
    if(strcmpi(params_group.dim_names(ss), prior_setup.groupName))
        H  = double(params_group.H);
        params_group.T{ss}(:) = exp(2*H);
    else
        params_group.T{ss}(:) = 1;
    end
end