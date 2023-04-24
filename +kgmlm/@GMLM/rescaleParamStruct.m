function [paramStruct_rescaled] = rescaleParamStruct(obj, paramStruct, scaled_WB, scaled_VT)

paramStruct_rescaled = paramStruct;
J = numel(paramStruct.Groups);
    
for jj = 1:J
    S = numel(paramStruct.Groups(jj).T);
    if(scaled_VT(jj))
        params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));
        paramStruct_rescaled.Groups(jj).V(:) = params_0.V(:);

        for ss = 1:S
            paramStruct_rescaled.Groups(jj).T{ss}(:) = params_0.T{ss}(:);
        end
    end
end
if(scaled_WB)
    params_0 = obj.GMLMstructure.scaleParams(paramStruct_rescaled);
    paramStruct_rescaled.W(:) = params_0.W(:);
    paramStruct_rescaled.B(:) = params_0.B(:);
end