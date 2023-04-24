function [paramStruct2] = saveSampleToFile(obj, samples_file, paramStruct, sample_idx, scaled_WB, scaled_VT, save_H, saveUnscaled)

paramStruct2 = paramStruct;
J = numel(paramStruct.Groups);
    
for jj = 1:J
    if(save_H.Groups(jj).H)
        samples_file.Data.(sprintf("G%d_H", jj))(:,sample_idx) = paramStruct.Groups(jj).H;
    end
    if(save_H.Groups(jj).H_gibbs)
        samples_file.Data.(sprintf("G%d_H_gibbs", jj))(:,sample_idx) = paramStruct.Groups(jj).H_gibbs;
    end
    S = numel(paramStruct.Groups(jj).T);
    if(scaled_VT(jj))
        params_0 = obj.GMLMstructure.Groups(jj).scaleParams(paramStruct.Groups(jj));

        samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx) = params_0.V;
        paramStruct2.Groups(jj).V(:) = params_0.V(:);

        if(saveUnscaled)
            samples_file.Data.(sprintf("G%d_V_scaled", jj))(:,:,sample_idx) = paramStruct.Groups(jj).V;
        end
        for ss = 1:S
            samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx) = params_0.T{ss};
            paramStruct2.Groups(jj).T{ss}(:) = params_0.T{ss}(:);
            if(saveUnscaled)
                samples_file.Data.(sprintf("G%d_T_%d_scaled", jj, ss))(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
            end
        end
    else
        samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx) = paramStruct.Groups(jj).V;
        for ss = 1:S
            samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx) = paramStruct.Groups(jj).T{ss};
        end
    end
end
if(scaled_WB)
    params_0 = obj.GMLMstructure.scaleParams(paramStruct2);

    samples_file.Data.W(:,  sample_idx) = params_0.W(:);
    if(save_H.B)
        samples_file.Data.B(:,:,sample_idx) = params_0.B(:,:);
    end
    paramStruct2.W(:) = params_0.W(:);
    paramStruct2.B(:) = params_0.B(:);

    if(saveUnscaled)
        samples_file.Data.W_scaled(:,  sample_idx) = paramStruct.W(:);
        if(save_H.B)
            samples_file.Data.B_scaled(:,:,sample_idx) = paramStruct.B(:,:);
        end
    end
else
    samples_file.Data.W(:,  sample_idx) = paramStruct.W(:);
    if(save_H.B)
        samples_file.Data.B(:,:,sample_idx) = paramStruct.B(:,:);
    end
end
if(save_H.H)
    samples_file.Data.H(:,sample_idx)   = paramStruct.H(:);
end
if(save_H.H_gibbs)
    samples_file.Data.H_gibbs(:,sample_idx)   = paramStruct.H_gibbs(:);
end