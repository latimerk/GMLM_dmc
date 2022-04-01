function [] = setupComputeStructuresHost(obj, reset)
if(nargin < 2)
    reset = false;
end
if(~isempty(obj.LL_info) &&  ~reset)
    return;
end
trs = obj.trials;

obj.LL_info = struct();

obj.LL_info.X = cell2mat(reshape([obj.trials(:).X], [], numel(obj.trials))');
obj.LL_info.Y = cell2mat({trs(:).Y}');
obj.LL_info.bin_size = obj.bin_size;

N = obj.dim_N();

llType = obj.logLikeType;
obj.LL_info.logLikeType = llType;
if(strcmpi(llType, "poissExp") || strcmpi(llType, "poissSoftRec"))
    obj.LL_info.Y_const = zeros(numel(trs), 1);
    for mm = 1:numel(trs)
        obj.LL_info.Y_const(mm,:) = -sum(gammaln(trs(mm).Y+1),1);
    end
else
    obj.LL_info.Y_const = zeros(numel(trs), 1);
end


obj.LL_info.dim_N = N;
obj.LL_info.dim_N_ranges = cumsum([1;N]);


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