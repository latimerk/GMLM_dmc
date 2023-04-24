function [] = saveSampleToFile(obj, samples_file, paramStruct, paramStruct_rescaled, sample_idx, scaled_WB, scaled_VT, save_H, saveUnscaled) %#ok<INUSL> 


J = numel(paramStruct(1).Groups);
    
for jj = 1:J
    if(save_H.Groups(jj).H)
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.(sprintf("G%d_H", jj))(:,sample_idx_c) = paramStruct(sample_c).Groups(jj).H;
        end
    end
    if(save_H.Groups(jj).H_gibbs)
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.(sprintf("G%d_H_gibbs", jj))(:,sample_idx_c) = paramStruct(sample_c).Groups(jj).H_gibbs;
        end
    end
    S = numel(paramStruct(sample_c).Groups(jj).T);
    if(scaled_VT(jj))
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx_c) = paramStruct_rescaled(sample_c).Groups(jj).V;
        end

        if(saveUnscaled)
            for sample_c = 1:numel(paramStruct)
                sample_idx_c = sample_idx(sample_c);
                samples_file.Data.(sprintf("G%d_V_scaled", jj))(:,:,sample_idx_c) = paramStruct(sample_c).Groups(jj).V;
            end
        end
        for ss = 1:S
            for sample_c = 1:numel(paramStruct)
                sample_idx_c = sample_idx(sample_c);
                samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx_c) = paramStruct_rescaled(sample_c).Groups(jj).T{ss};
            end
            if(saveUnscaled)
                for sample_c = 1:numel(paramStruct)
                    sample_idx_c = sample_idx(sample_c);
                    samples_file.Data.(sprintf("G%d_T_%d_scaled", jj, ss))(:,:,sample_idx_c) = paramStruct(sample_c).Groups(jj).T{ss};
                end
            end
        end
    else
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.(sprintf("G%d_V", jj))(:,:,sample_idx_c) = paramStruct(sample_c).Groups(jj).V;
        end
        for ss = 1:S
            for sample_c = 1:numel(paramStruct)
                sample_idx_c = sample_idx(sample_c);
                samples_file.Data.(sprintf("G%d_T_%d", jj, ss))(:,:,sample_idx_c) = paramStruct(sample_c).Groups(jj).T{ss};
            end
        end
    end
end
if(scaled_WB)
    for sample_c = 1:numel(paramStruct)
        sample_idx_c = sample_idx(sample_c);
        samples_file.Data.W(:,  sample_idx_c) = paramStruct_rescaled(sample_c).W(:);
    end
    if(save_H.B)
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.B(:,:,sample_idx_c) = paramStruct_rescaled(sample_c).B(:,:);
        end
    end

    if(saveUnscaled)
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.W_scaled(:,  sample_idx_c) = paramStruct(sample_c).W(:);
        end
        if(save_H.B)
            for sample_c = 1:numel(paramStruct)
                sample_idx_c = sample_idx(sample_c);
                samples_file.Data.B_scaled(:,:,sample_idx_c) = paramStruct(sample_c).B(:,:);
            end
        end
    end
else
    for sample_c = 1:numel(paramStruct)
        sample_idx_c = sample_idx(sample_c);
        samples_file.Data.W(:,  sample_idx_c) = paramStruct(sample_c).W(:);
    end
    if(save_H.B)
        for sample_c = 1:numel(paramStruct)
            sample_idx_c = sample_idx(sample_c);
            samples_file.Data.B(:,:,sample_idx_c) = paramStruct(sample_c).B(:,:);
        end
    end
end
if(save_H.H)
    for sample_c = 1:numel(paramStruct)
        sample_idx_c = sample_idx(sample_c);
        samples_file.Data.H(:,sample_idx_c)   = paramStruct(sample_c).H(:);
    end
end
if(save_H.H_gibbs)
    for sample_c = 1:numel(paramStruct)
        sample_idx_c = sample_idx(sample_c);
        samples_file.Data.H_gibbs(:,sample_idx_c)   = paramStruct(sample_c).H_gibbs(:);
    end
end