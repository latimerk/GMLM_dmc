function [results] = computeLogLikelihood_host_v2(obj, params, opts, results)
obj.setupComputeStructuresHost();
if(nargin < 4 || isempty(results))
    results = obj.getEmptyResultsStruct(opts);
end
    

LL_info = obj.LL_info;
X_groups = obj.X_groups;
X_lin = LL_info.X_lin;
Y = LL_info.Y;
dt = LL_info.bin_size;
dim_N_ranges = LL_info.dim_N_ranges;

J = numel(X_groups);
D = [X_groups(:).dim_D];
A = [X_groups(:).dim_A];
M = numel(dim_N_ranges) - 1;

T = size(Y,1);
P = size(Y,2);
isPop = P > 1;

%% get rates
[log_rate, xx, R] = obj.computeLogRate_host_v2(params);

compute_dll = opts.dB || opts.dW;
for jj = 1:J
    compute_dll = compute_dll | opts.Groups(jj).dV | any(opts.Groups(jj).dT);
end

%% compute term-wise log-rate

if(compute_dll)
    [ll,dll] = LL_info.logLikeFun(log_rate, Y, dt);
else
    ll = LL_info.logLikeFun(log_rate, Y, dt);
end

if(~isempty(opts.trial_weights) && compute_dll)
    for mm = 1:M
        dll(dim_N_ranges(mm):(dim_N_ranges(mm+1)-1),:) = dll(dim_N_ranges(mm):(dim_N_ranges(mm+1)-1),:) .* opts.trial_weights(mm,:);
    end
end

if(opts.trialLL)
    trialLL = nan(M, P);
    for mm = 1:M
        trialLL(mm, :) = sum(ll(dim_N_ranges(mm):(dim_N_ranges(mm+1)-1),:),1)  + LL_info.Y_const(mm,:);

        if(~isempty(opts.trial_weights))
            trialLL(mm, :)  = trialLL(mm, :)  .* opts.trial_weights(mm,:);
        end
    end
end

results.trialLL(1:size(trialLL,1),:) = trialLL;

%% do any requested derivatives
if(opts.dW)
    if(isPop)
        results.dW(:) = sum(dll,1)';
    else
        for pp = 1:P
            tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
            results.dW(pp) = sum(ll(tt));
        end
    end
end
if(opts.dB && ~isempty(X_lin))
    if(~iscell(X_lin) && isPop)
        results.dB(:,:) = X_lin'*dLL;
    elseif(isPop)
        for pp = 1:P
            results.dB(:, pp) = X_lin{pp}' * dll(:, pp);
        end
    else
        for pp = 1:P
            tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
            results.dB(:, pp) = X_lin{pp}' * dll(tt, pp);
        end
    end
end
for jj = 1:J
    if(opts.Groups(jj).dV)
        if(isPop)
            results.Groups(jj).dV(:,:) = dll' * xx(jj).a;
        else
            results.Groups(jj).dV(:,:) = zeros(P,R(jj));
            for pp = 1:P
                tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
                results.Groups(jj).dV(pp, :) = dll(tt)' * xx(jj).a(tt,:);
            end
        end
    end

    %% for each D
    dll_v = [];
    for dd = 1:D(jj)
        dTs = (opts.Groups(jj).dT & X_groups(jj).factorIdxs == dd);
        dTs = dTs(:)';
        if(any(dTs))
            if(isempty(dll_v))
                if(isPop)
                    dll_v = repmat(dll * params.Groups(jj).V,  A(jj), 1);
                else
                    dll_v = repmat(dll, [1 R(jj)]);
                    for pp = 1:P
                        tt = LL_info.dim_N_neuron_ranges(pp):(LL_info.dim_N_neuron_ranges(pp+1)-1);
                        dll_v(tt,:) = dll_v(tt,:) .* params.Groups(jj).V(pp,:);
                    end
                    dll_v = repmat(dll_v, A(jj), 1);
                end
            end

            dxx = prod(xx(jj).c(:, :, ~ismember(1:D(jj), dd)), 3) .* dll_v;
            if(X_groups(jj).isShared(dd))
                dF = X_groups(jj).X_local{dd}' * (X_groups(jj).iX_shared(dd).iX * dxx);
            elseif(size(X_groups(jj).X_local{dd},1) == T && A(jj) > 1)
                dF = 0;
                for aa = 1:A(jj)
                    tts = (1:T) + (aa-1)*T;
                    dF = dF + X_groups(jj).X_local{dd}' * dxx(tts,:);
                end
            else 
                dF = X_groups(jj).X_local{dd}' * dxx;
            end
        
            %% each T in D
            if(sum(X_groups(jj).factorIdxs == dd) == 1)
                results.Groups(jj).dT{X_groups(jj).factorIdxs == dd}(:,:) = dF;
            else
                for ss = find(dTs)
                    dFdT = getdFdT(ss, dd,  X_groups(jj).factorIdxs , X_groups(jj).dim_F, params.Groups(jj));
                    for rr = 1:R(jj)
                        results.Groups(jj).dT{ss}(:,rr) = dFdT(:,:,rr)'*dF(:,rr);
                    end
                end
            end
        end
    end
end

end



function [dFdT] = getdFdT(ss, dd, factor_idx, dim_F, params_group)
    dim_R = size(params_group.V,2);
    fis = sort(find(factor_idx == dd));
    T = size(params_group.T{ss},1);
    dFdT   = nan(dim_F(dd), T, dim_R);
    
    for rr = 1:dim_R
        for tt = 1:T
            params_group.T{ss}(:,rr) = 0;
            params_group.T{ss}(tt,rr) = 1;
            F_r = params_group.T{fis(1)}(:,rr);
            for ss2 = 2:numel(fis)
                F_r = kron(params_group.T{fis(ss2)}(:,rr), F_r);
            end
            dFdT(:, tt, rr) = F_r;
        end
    end
end