function [samples_file_format, totalParams] = getSampleFileFormat(obj, TotalSamples, dataType_samples, paramStruct, scaled_WB, scaled_VT, saveUnscaled)

samples_file_format = cell(0, 3);
ctr = 1;
totalParams = 0;

samples_file_format{ctr, 1} = dataType_samples;
samples_file_format{ctr, 2} = [numel(paramStruct.W) TotalSamples];
samples_file_format{ctr, 3} = "W"; totalParams = totalParams + numel(paramStruct.W);
ctr = ctr + 1;
samples_file_format{ctr, 1} = dataType_samples;
samples_file_format{ctr, 2} = [size(paramStruct.B) TotalSamples];
samples_file_format{ctr, 3} = "B"; totalParams = totalParams + numel(paramStruct.B);
ctr = ctr + 1;

if(saveUnscaled && scaled_WB)
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.W) TotalSamples];
    samples_file_format{ctr, 3} = "W_scaled"; totalParams = totalParams + numel(paramStruct.W);
    ctr = ctr + 1;
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [size(paramStruct.B) TotalSamples];
    samples_file_format{ctr, 3} = "B_scaled"; totalParams = totalParams + numel(paramStruct.B);
    ctr = ctr + 1;
end
samples_file_format{ctr, 1} = dataType_samples;
samples_file_format{ctr, 2} = [numel(paramStruct.H) TotalSamples];
samples_file_format{ctr, 3} = "H"; totalParams = totalParams + numel(paramStruct.H);
ctr = ctr + 1;
samples_file_format{ctr, 1} = dataType_samples;
samples_file_format{ctr, 2} = [numel(paramStruct.H_gibbs) TotalSamples];
samples_file_format{ctr, 3} = "H_gibbs"; totalParams = totalParams + numel(paramStruct.H_gibbs);
ctr = ctr + 1;

for jj = 1:numel(paramStruct.Groups)
    S = numel(paramStruct.Groups(jj).T) ;

    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).V) TotalSamples];
    samples_file_format{ctr, 3} = sprintf("G%d_V", jj);
    ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).V);

    for ss = 1:S 
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).T{ss}) TotalSamples];
        samples_file_format{ctr, 3} = sprintf("G%d_T_%d", jj, ss);
        ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).T{ss});
    end

    if(saveUnscaled && scaled_VT(jj))
        samples_file_format{ctr, 1} = dataType_samples;
        samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).V) TotalSamples];
        samples_file_format{ctr, 3} = sprintf("G%d_V_scaled", jj);
        ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).V);
    
        for ss = 1:S
            samples_file_format{ctr, 1} = dataType_samples;
            samples_file_format{ctr, 2} = [size(paramStruct.Groups(jj).T{ss}) TotalSamples];
            samples_file_format{ctr, 3} = sprintf("G%d_T_%d_scaled", jj, ss);
            ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).T{ss});
        end
    end

    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.Groups(jj).H) TotalSamples];
    samples_file_format{ctr, 3} = sprintf("G%d_H", jj);
    ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).H);
    samples_file_format{ctr, 1} = dataType_samples;
    samples_file_format{ctr, 2} = [numel(paramStruct.Groups(jj).H_gibbs) TotalSamples];
    samples_file_format{ctr, 3} = sprintf("G%d_H_gibbs", jj);
    ctr = ctr + 1; totalParams = totalParams + numel(paramStruct.Groups(jj).H_gibbs);
end
