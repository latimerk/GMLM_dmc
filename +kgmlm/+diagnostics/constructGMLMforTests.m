function [GMLMstructure, trials] = constructGMLMforTests( factor_idx_0, isShared, isSharedEye, dim_A)
dim_R_max = 4;

if(nargin < 4)
    dim_A = 2;
end

[~,~,factor_idx] = unique(factor_idx_0(:));
dim_S = numel(factor_idx);
dim_T = randi(20,[dim_S 1]);

dim_D = max(factor_idx);
dim_F = nan(dim_D,1);
for ff = 1:dim_D
    dim_F(ff) = prod(dim_T(factor_idx == ff));
end

if(numel(isShared) ~= dim_D)
    error('isShared not setup');
end
if(numel(isSharedEye) ~= dim_D)
    error('isSharedEye not setup');
end

dim_X_shared = randi(100, [dim_D, 1]) + 50;
dim_X_shared(isSharedEye) = dim_T(isSharedEye);
dim_B = 6;

dim_M = 100;
dim_P = 10;

dim_N_min = 50;
dim_N_max = 100;

GMLMstructure.dim_B = dim_B;
GMLMstructure.Groups = struct('X_shared', [], 'dim_R_max', [], 'dim_A', [], 'name', [], 'dim_names', [], 'dim_T', [], 'factor_idx', [], 'prior', []);


GMLMstructure.Groups(1).X_shared = cell(1, dim_D);
for ff = 1:dim_D
    if(isShared(ff))
        if(isSharedEye(ff))
            GMLMstructure.Groups(1).X_shared{ff} = eye(dim_F(ff));
        else
            GMLMstructure.Groups(1).X_shared{ff} = randn(dim_X_shared(ff), dim_F(ff))./sqrt(dim_F(ff));
        end
    end
end
GMLMstructure.Groups(1).dim_R_max = dim_R_max;
GMLMstructure.Groups(1).dim_A = dim_A;
GMLMstructure.Groups(1).dim_T = dim_T;
GMLMstructure.Groups(1).name = "testTensor";
GMLMstructure.Groups(1).factor_idx = factor_idx;

GMLMstructure.Groups(1).dim_names = sprintf("a%d", 1);
for ss = 2:dim_S
    GMLMstructure.Groups(1).dim_names(ss) = sprintf("a%d", ss);
end

trials = struct('Y', cell(dim_M,1), 'X_lin', [], 'neuron', [], 'Groups', []);
for mm = 1:dim_M
    dim_N = randi([dim_N_min, dim_N_max], 1);
    
    trials(mm).Y = randi([0,2], [dim_N 1]);
    trials(mm).X_lin = randn([dim_N dim_B])./sqrt(dim_B);
    trials(mm).neuron = randi(dim_P);
    
    trials(mm).Groups = struct('X_local', [], 'iX_shared', []);
    trials(mm).Groups(1).X_local = cell(1, dim_D);
    trials(mm).Groups(1).iX_shared = cell(1, dim_D);

    for ff = 1:dim_D
        if(isShared(ff))
            trials(mm).Groups(1).X_local{ff} = [];
            trials(mm).Groups(1).iX_shared{ff} = randi([-5 dim_X_shared(ff)+20], [dim_N dim_A]);
        else
            trials(mm).Groups(1).X_local{ff} = randn([dim_N dim_F(ff) dim_A])./sqrt(dim_F(ff));
            %trials(mm).Groups(1).X_local{ff}(:) = 1;
            trials(mm).Groups(1).iX_shared{ff} = [];
        end
    end
end
