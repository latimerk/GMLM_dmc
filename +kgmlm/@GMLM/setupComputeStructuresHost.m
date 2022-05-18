function [] = setupComputeStructuresHost(obj, reset, order)
if(nargin < 2)
    reset = false;
end
if(~isempty(obj.LL_info) &&  ~reset)
    return;
end

if(nargin < 3 || isempty(order))
    if(obj.isSimultaneousPopulation)
        order = 1:numel(obj.trials);
    else
        [neuronIdx,order] = sort(obj.neuronIdx);
        neuronIdx_u = unique(neuronIdx);
    end
end

trs = obj.trials(order);

obj.LL_info = struct();
if(obj.isSimultaneousPopulation)
    X_lin_0 = cell2mat({trs(:).X_lin}');
    if(size(X_lin_0,3) > 1)
        X_lin = cell(size(X_lin_0,3),1);
        for ii = 1:size(X_lin_0,3)
            X_lin{ii} = X_lin_0(:,:,ii);
        end
    else
        X_lin = X_lin_0;
    end
else
    X_lin = cell(numel(neuronIdx_u),1);

    for pp = 1:numel(neuronIdx_u)
        X_lin{pp} = cell2mat({trs(neuronIdx == neuronIdx_u(pp)).X_lin}');
    end
end

obj.LL_info.X_lin = X_lin;

obj.LL_info.Y = cell2mat({trs(:).Y}');
obj.LL_info.bin_size = obj.bin_size;


llType = obj.logLikeType;
obj.LL_info.logLikeType = llType;
if(strcmpi(llType, "poissExp") || strcmpi(llType, "poissSoftRec"))
    obj.LL_info.Y_const = zeros(numel(trs), size(obj.LL_info.Y,2));
    for mm = 1:numel(trs)
        obj.LL_info.Y_const(mm,:) = -sum(gammaln(trs(mm).Y+1),1);
    end
else
    obj.LL_info.Y_const = zeros(numel(trs), size(obj.LL_info.Y,2));
end

if(strcmpi(llType, "poissExp"))
    obj.LL_info.logLikeFun = @kgmlm.utils.poissExpLL;
elseif(strcmpi(llType, "poissSoftRec"))
    obj.LL_info.logLikeFun = @kgmlm.utils.poissSoftRecLL;
elseif(strcmpi(llType, "truncatedPoissExp"))
    obj.LL_info.logLikeFun = @kgmlm.utils.truncatedPoissExpLL;
elseif(strcmpi(llType, "sqErr"))
    obj.LL_info.logLikeFun = @kgmlm.utils.sqErrLL;
else
    error("Invalid log likelihood");
end



N = obj.dim_N();
N = N(order);
obj.LL_info.dim_N = N;
obj.LL_info.dim_N_ranges = cumsum([1;N]);

if(~obj.isSimultaneousPopulation)
    P = obj.dim_P;
    dim_N_neuron = zeros(P,1);
    for pp = 1:P
        dim_N_neuron(pp) = sum(N(neuronIdx == neuronIdx_u(pp)));
    end
    obj.LL_info.dim_N_neuron = N_neuron;
    obj.LL_info.dim_N_neuron_ranges = cumsum([1;N_neuron]);
end


for jj = 1:obj.dim_J
    A =  obj.dim_A(jj);
    obj.X_groups(jj).factorIdxs = obj.getFactorIdxs(jj);
    obj.X_groups(jj).dim_A = A;
    obj.X_groups(jj).dim_D = obj.dim_D(jj);
    obj.X_groups(jj).dim_F = zeros(obj.X_groups(jj).dim_D, 1);
    obj.X_groups(jj).isShared = obj.isSharedRegressor(jj);
    obj.X_groups(jj).X_local   = cell(obj.dim_D(jj), 1); 
    obj.X_groups(jj).iX_sparse = struct("nonzeros", cell(obj.dim_D(jj), 1), "rows", [], "cols", [],  "elements", [], "iX", [], "idx", [],  "dims", []); 
    for dd = 1:obj.X_groups(jj).dim_D 
        if(obj.X_groups(jj).isShared(dd))
            X_s = obj.GMLMstructure.Groups(jj).X_shared{dd};
            idx = cell2mat(arrayfun(@(aa) aa.Groups(jj).iX_shared{dd}, trs, 'UniformOutput', false));
            if(size(idx,2) == 1 && A > 1)
                idx = repmat(idx,[1 A]);
            end
            idx = idx(:);
            dims = [size(X_s,1) numel(idx)];
            vv = idx > 0 & idx <= dims(1);
            idx(~vv) = 0;

            nz = sum(vv);
            rows = idx(vv);
            cols = find(vv);

            
            obj.X_groups(jj).iX_shared(dd).nonzeros = nz;
            obj.X_groups(jj).iX_shared(dd).rows = rows;
            obj.X_groups(jj).iX_shared(dd).cols = cols;
            obj.X_groups(jj).X_local{dd}  = X_s;
            obj.X_groups(jj).iX_shared(dd).dims = dims;
            obj.X_groups(jj).iX_shared(dd).idx = reshape(idx, [], A);
            obj.X_groups(jj).iX_shared(dd).iX = sparse(rows, cols, ones(nz,1), dims(1), dims(2));
            obj.X_groups(jj).iX_shared(dd).elements = find(obj.X_groups(jj).iX_shared(dd).iX);
        else
            Xl = cell2mat(arrayfun(@(aa) aa.Groups(jj).X_local{dd}, trs, 'UniformOutput', false));
            if(size(Xl,3) > 1)
                Xl_0 = Xl;
                Xl = zeros(size(Xl_0,1) * size(Xl_0,3), size(Xl_0,2));
                for ii = 1:size(Xl_0,3)
                    Xl((1:size(Xl_0,1)) + (ii-1)*size(Xl_0,1), :) = Xl_0(:, :, ii);
                end
            end
            obj.X_groups(jj).X_local{dd} = Xl;
        end
        obj.X_groups(jj).dim_F(dd) = size(obj.X_groups(jj).X_local{dd}, 2);
    end
end

end