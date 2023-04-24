function [log_rate, xx, R] = computeLogRate_host_v2(obj, params)

obj.setupComputeStructuresHost();
LL_info = obj.LL_info;
X_groups = obj.X_groups;
X_lin = LL_info.X_lin;
Y = LL_info.Y;

J = numel(X_groups);
D = [X_groups(:).dim_D];
A = [X_groups(:).dim_A];

T = size(Y,1);
P = size(Y,2);
isPop = P > 1;

%% linear term
if(isempty(X_lin))
    log_rate = zeros(T, P);
else
    if(~iscell(X_lin) && isPop)
        log_rate = repmat(X_lin * params.B, [1 1 P]);
    elseif(isPop)
        log_rate = zeros(T, P);
        for pp = 1:P
            log_rate(:, pp) = X_lin{pp} * params.B(:,pp);
        end
    else
        log_rate = zeros(T, P);
        for pp = 1:P
            tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
            log_rate(tt,:) = X_lin{tt} * params.B(:,pp);
        end
    end
end

%% add constants
if(isPop)
    log_rate = log_rate + params.W(:)';
else
    for pp = 1:P
        tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
        log_rate(tt) = log_rate(tt) + params.W(pp);
    end
end


%% add group terms
R = zeros(J,1);
xx = struct("c", cell(J,1), "a", [], "m", []);
for jj = 1:J
    R(jj) = size(params.Groups(jj).V,2);

    xx(jj).c = zeros(T*A(jj), R(jj), D(jj));
    for dd = 1:D(jj)
        F = getF(dd, X_groups(jj).factorIdxs , X_groups(jj).dim_F, params.Groups(jj));
        if(X_groups(jj).isShared(dd))
            XF = X_groups(jj).X_local{dd}  * F;
            xx(jj).c(X_groups(jj).iX_shared(dd).cols, :, dd) = XF(X_groups(jj).iX_shared(dd).rows,:);
            %xx(jj).c(:, :, dd) = X_groups(jj).iX_shared(dd).iX'*XF;
        elseif(size(X_groups(jj).X_local{dd},1) == T && A(jj) > 1)
            xx(jj).c(:, :, dd) = repmat(X_groups(jj).X_local{dd} * F, [A 1]);
        else
            xx(jj).c(:, :, dd) = X_groups(jj).X_local{dd} * F;
        end
    end

    xx(jj).m = prod(xx(jj).c,3);
    xx(jj).a = squeeze(sum(reshape(xx(jj).m, [T, A(jj), R(jj)]),2));

    if(isPop)
        log_rate = log_rate + xx(jj).a * params.Groups(jj).V';
    else
        for pp = 1:P
            tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
            log_rate(tt) = log_rate(tt) + xx(jj).a(tt,:)*params.Groups(jj).V(pp,:)';
        end
    end
end
end


function [F] = getF(dd, factor_idx, dim_F, params_group)
    dim_R = size(params_group.V,2);
    fis = sort(find(factor_idx == dd));
    F   = nan(dim_F(dd), dim_R);
    
    for rr = 1:dim_R
        F_r = params_group.T{fis(1)}(:,rr);
        for ss = 2:numel(fis)
            F_r = kron(params_group.T{fis(ss)}(:,rr), F_r);
        end
        F(:, rr) = F_r;
    end
end