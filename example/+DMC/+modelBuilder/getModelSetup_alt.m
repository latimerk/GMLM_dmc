%% getModelSetup_at
% gets the different types of models. By default, uses the cosine tuning model (same dir tuning for sample/test, but different cat tuning)
%
% Directions that are only used in the sample or test stimulus can be specified using the optional fields TaskInfo.Direction_sample and TaskInfo.Directions_test
%   which are logical vectors stating if the directions are used or not.
%
% optional key/value pairs are:
%  dirTuningType can be 'cosine', 'full', or 'none'
%  dirSameSampleTest = true/false (same dir tuning for sample/test, doesn't matter if dirTuningType = 'none')
%  includeCategory   = true/false (false only includes sample/test signal)
%  includeSubCategory   = true/false (false only includes sample/test signal)
%  includeSampleTest = true/false (allow cat / mean differences across sample/test stim)
%
% With 'full' dir tuning and dirSameSampleTest and ~includeCategory, there can obviously still be cat tuning, but it won't change across sample/test.
%
% The prior may not setup correctly for every combination of settings (it should still run, it just might include too much info).
% It will work for all the combinations I care about.
%
%  Package GMLM_dmc for dimensionality reduction of neural data.
%   
%  References
%   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
%   decisions in parietal cortex reflects long-term training history.
%   bioRxiv
%
%  Copyright (c) 2021 Kenneth Latimer
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
function [stimConfig, stimPrior_setup] = getModelSetup_alt(TaskInfo, varargin)
p = inputParser;
p.CaseSensitive = false;

addRequired( p, 'TaskInfo'     ,     @isstruct);
addParameter(p, 'dirTuningType'    , 'cosine', @ischar);
addParameter(p, 'dirSameSampleTest',  true,    @islogical);
addParameter(p, 'includeCategory'  ,  true,    @islogical);
addParameter(p, 'includeSubCategory'  ,  false,    @islogical);
addParameter(p, 'includeSubCategorySampleTest'  ,  false,    @islogical);
addParameter(p, 'includeCatSampleTest'  ,  true,    @islogical);
addParameter(p, 'includeOnset'  ,  true,    @islogical);
addParameter(p, 'includeOnsetSampleTest'  ,  true,    @islogical);
addParameter(p, 'bases'  ,  DMC.modelBuilder.setupBasis_spline(),    @isstruct);


parse(p, TaskInfo, varargin{:});
% then set/get all the inputs out of this structure
dirTuningType       = p.Results.dirTuningType;
dirSameSampleTest   = p.Results.dirSameSampleTest;
includeCategory     = p.Results.includeCategory;
includeSubCategory  = p.Results.includeSubCategory;
includeSubCatSampleTest  = p.Results.includeSubCategorySampleTest;
includeCatSampleTest   = p.Results.includeCatSampleTest;
includeOnset   = p.Results.includeOnset;
includeOnsetSampleTest   = p.Results.includeOnsetSampleTest;
bases   = p.Results.bases;

%%
thetas = TaskInfo.Directions(:)';
cats   = TaskInfo.Categories(:)';
if(numel(thetas) ~= numel(cats))
    error("Number of directions must match number of categories.\n");
end


cats_unique = unique(cats);
cats_unique = cats_unique(:)';

NU = numel(cats_unique);
ND = numel(thetas);

NU_sub = zeros(numel(cats),1);
if(isfield(TaskInfo, "Subcategories"))
    subcats   = TaskInfo.Subcategories(:)';
    
    if(numel(subcats) ~= numel(cats))
        error("Invalid subcategory setup.\n");
    end
    
    subcats_unique = cell(NU,1);
    for cc = 1:NU
        subcats_unique{cc} = unique(subcats(cats == cats_unique(cc)));
        subcats_unique{cc} = subcats_unique(:)';
        NU_sub = numel(subcats_unique{cc});
    end
else
    subcats = [];
    subcats_unique = cell(NU,1);
end

if((includeCategory && NU ~= 2) || (includeSubCategory && ~all(NU_sub == 2,'all')))
    error("This model builder can only handle two categories/subcategories.\n");
end

if(~isfield(TaskInfo, 'Directions_sample'))
    TaskInfo.Directions_sample = true(ND,1);
end
if(~isfield(TaskInfo, 'Directions_test'))
    TaskInfo.Directions_test = true(ND,1);
end

ND_sample = numel(TaskInfo.Directions(TaskInfo.Directions_sample));
ND_test   = numel(TaskInfo.Directions(TaskInfo.Directions_test));

Directions_used = ismember(thetas(:), unique([TaskInfo.Directions(TaskInfo.Directions_sample);TaskInfo.Directions(TaskInfo.Directions_test)]));
ND_used = sum(Directions_used);

cosine_cols    = [];
cosine_cols_st = []; 
sine_cols      = [];
sine_cols_st   = [];

cat_cols       = [];
cat_cols_st    = [];

subcat_cols       = [];
subcat_cols_st    = [];

onset_cols     = [];
onset_cols_st  = [];

dir_cols1        = [];
dir_cols1_st     = [];
dir_cols1_theta  = [];
dir_cols2        = [];
dir_cols2_st     = [];
dir_cols2_theta  = [];

NF = 0;


%%

%figures out model type
if(strcmpi(dirTuningType, 'cosine') || strcmpi(dirTuningType, 'none'))
    if(includeOnset)
        onset_cols = NF + 1;
        NF = NF + 1;
    end
    if(includeOnsetSampleTest)
        onset_cols_st = NF + 1;
        NF = NF + 1;
    end
    
    if(includeCategory)
        cat_cols      = NF + 1;
        NF = NF + 1;
    end
    if(includeCatSampleTest)
        cat_cols_st   = NF + 1;
        NF = NF + 1;
    end
    if(includeSubCategory)
        subcat_cols      = NF + 1;
        NF = NF + 1;
    end
    if(includeSubCatSampleTest)
        subcat_cols_st      = NF + 1;
        NF = NF + 1;
    end
    
    if(strcmpi(dirTuningType, 'cosine'))
        if(dirSameSampleTest)
            sine_cols   = NF + 1;
            cosine_cols = NF + 2;
            sine_cols_st   = [];
            cosine_cols_st = [];
            NF = NF + 2;
        else
            sine_cols   = NF + 1;
            cosine_cols = NF + 2;
            sine_cols_st   = NF + 3;
            cosine_cols_st = NF + 4;
            NF = NF + 4;
        end
    end
elseif(strcmpi(dirTuningType, 'full'))
    if(dirSameSampleTest)
        if(includeOnsetSampleTest)
            onset_cols_st = NF + 1;
            NF = NF + 1;
        end
        if(includeCatSampleTest)
            cat_cols_st   = NF + 1;
            NF = NF + 1;
        end
        if(includeSubCatSampleTest)
            subcat_cols_st   = NF + 1;
            NF = NF + 1;
        end
    
        dir_cols1       = NF + (1:ND_used);
        dir_cols1_st    = ones(1,ND_used)*-1;
        dir_cols1_theta = thetas(Directions_used);
        NF = NF + ND_used;
    else
        dir_cols1       = NF + (1:ND_sample);
        dir_cols1_st    = ones(1,ND_sample) ;
        dir_cols1_theta = thetas(TaskInfo.Directions_sample);
        NF = NF + ND_sample;
        
        dir_cols2       = NF + (1:ND_test);
        dir_cols2_st    = ones(1,ND_test)*2;
        dir_cols2_theta = thetas(TaskInfo.Directions_test);
        NF = NF + ND_test;
    end
else
    error('invalid option for model type');
end


%% sets up the configuration
stimConfig = zeros(ND, NF, 2);
stimConfig = addCols(stimConfig, onset_cols, 'all');
stimConfig = addCols(stimConfig, onset_cols_st, 'sample/test');

if(includeCatSampleTest)
    stimConfig = addCols(stimConfig, cat_cols,    'sample', -1 + 2*(cats(:) == cats_unique(1)));
    stimConfig = addCols(stimConfig, cat_cols_st, 'test',   -1 + 2*(cats(:) == cats_unique(1)));
elseif(includeCategory)
    stimConfig = addCols(stimConfig, cat_cols, 'all',       -1 + 2*(cats(:) == cats_unique(1)));
end

if(includeSubCatSampleTest)
    for cc = 1:NU
        sc = -1 + 2*(subcats(:) == subcats_unique{cc});
        stimConfig = addCols(stimConfig, subcat_cols(cc),    'sample', sc.*(cats(:) == cats_unique(cc)));
        stimConfig = addCols(stimConfig, subcat_cols_st(cc), 'test',   sc.*(cats(:) == cats_unique(cc)));
    end
elseif(includeSubCategory)
    for cc = 1:NU
        sc = -1 + 2*(subcats(:) == subcats_unique{cc});
        stimConfig = addCols(stimConfig, subcat_cols(cc),    'all', sc.*(cats(:) == cats_unique(cc)));
    end
end

if(dirSameSampleTest)
    stimConfig = addCols(stimConfig, sine_cols,   'all',   sind(TaskInfo.Directions(:)));
    stimConfig = addCols(stimConfig, cosine_cols, 'all', cosd(TaskInfo.Directions(:)));
else
    stimConfig = addCols(stimConfig, sine_cols,   'sample',   sind(TaskInfo.Directions(:)));
    stimConfig = addCols(stimConfig, cosine_cols, 'sample', cosd(TaskInfo.Directions(:)));
    stimConfig = addCols(stimConfig, sine_cols_st,   'test',   sind(TaskInfo.Directions(:)));
    stimConfig = addCols(stimConfig, cosine_cols_st, 'test', cosd(TaskInfo.Directions(:)));
end

if(~isempty(dir_cols1))
    stimConfig = addCols(stimConfig, dir_cols1, dir_cols1_st, thetas' == dir_cols1_theta);
end
if(~isempty(dir_cols2))
    stimConfig = addCols(stimConfig, dir_cols2, dir_cols2_st, thetas' == dir_cols2_theta);
end



%% construct the prior distribution


coefs_ind_groups = {cat_cols,       "cat";
                    cat_cols_st,    "cat_test";
                    subcat_cols,    "subcat";
                    subcat_cols_st, "subcat_test";
                    onset_cols,     "onset";
                    onset_cols_st,  "onset-sample/test"};
coefs_grp_groups = {[sine_cols cosine_cols],      "cosine_dir";
                    [sine_cols_st cosine_cols_st],   "cosine_dir_test"};

coefs_ind_groups = coefs_ind_groups(~cellfun(@isempty, coefs_ind_groups(:,1)), :);
coefs_grp_groups = coefs_grp_groups(~cellfun(@isempty, coefs_grp_groups(:,1)), :);

NG_i         = sum(cellfun(@numel, coefs_ind_groups(:,1)));
NG_g         = size(coefs_grp_groups, 1);
NG = NG_i + NG_g + ~isempty(dir_cols1) + ~isempty(dir_cols2);

timing_idx = 1;
stim_idx   = 2;
            
stimPrior_setup.T(stim_idx).setup.parts = struct("type", cell(NG,1), "idx_hyperparams", [], "idx_params", [], "var", []);
ctr = 1;
for gg = 1:NG_i
    c_idx = coefs_ind_groups{gg, 1};
    if(~all(c_idx > 0))
        warning('invalid idx');
    end
    for ii = 1:numel(c_idx)
        stimPrior_setup.T(stim_idx).setup.parts(ctr).type = "group"; 
        stimPrior_setup.T(stim_idx).setup.parts(ctr).name = coefs_ind_groups{gg, 2};
        stimPrior_setup.T(stim_idx).setup.parts(ctr).var = 1;
        stimPrior_setup.T(stim_idx).setup.parts(ctr).idx_hyperparams = [];
        stimPrior_setup.T(stim_idx).setup.parts(ctr).idx_params = c_idx(ii);
        ctr = ctr + 1;
    end
end
for gg = 1:NG_g
    c_idx = coefs_grp_groups{gg, 1};
    if(~all(c_idx > 0))
        warning('invalid idx');
    end
    stimPrior_setup.T(stim_idx).setup.parts(ctr).type = "group"; 
    stimPrior_setup.T(stim_idx).setup.parts(ctr).name = coefs_grp_groups{gg, 2};
    stimPrior_setup.T(stim_idx).setup.parts(ctr).var = 1;
    stimPrior_setup.T(stim_idx).setup.parts(ctr).idx_hyperparams = [];
    stimPrior_setup.T(stim_idx).setup.parts(ctr).idx_params = c_idx(:);
    ctr = ctr + 1;
end
stimPrior_setup.T(stim_idx).N_scales = NG_g + NG_i;
stimPrior_setup.T(stim_idx).N_hyperparams = @(rank) 0;
stimPrior_setup.T(stim_idx).mu = 0;


stimPrior_setup.T(timing_idx).mu = 0;
stimPrior_setup.T(timing_idx).N_scales = 0;%size(bases.stim.B,2);%-1;
stimPrior_setup.T(timing_idx).N_hyperparams = @(rank) 0;
stimPrior_setup.T(timing_idx).setup.parts.type = "std";%independent";
stimPrior_setup.T(timing_idx).setup.parts.var = 1;

stimPrior_setup.V.mu = 0;
stimPrior_setup.V.N_scales = 0;
stimPrior_setup.V.N_hyperparams = @(rank) 0;
stimPrior_setup.V.setup.parts.type = "std";
stimPrior_setup.V.setup.parts.idx_hyperparams = 0;
stimPrior_setup.V.setup.parts.var = 1;


stimPrior_setup.hyperparams.c.a = 2;
stimPrior_setup.hyperparams.c.b = 8;
stimPrior_setup.hyperparams.df_global = 3;
stimPrior_setup.hyperparams.df_local  = 3;
stimPrior_setup.hyperparams.H.nu = 3;
end

function [stimConfig] = addCols(stimConfig, cols, col_types, weights)
    if(nargin < 4)
        weights = 1;
    end
    
    if(size(weights,2) == 1)
        weights = repmat(weights, [1 numel(cols)]);
    end
    
    for ii = 1:numel(cols)
        if(strcmpi(col_types, "all"))
            st = 1:2;
            mm = [1 1];
        elseif(strcmpi(col_types, "sample"))
            st = 1;
            mm = 1;
        elseif(strcmpi(col_types, "sample/test"))
            st = 1:2;
            mm = [1 -1];
        elseif(strcmpi(col_types, "test"))
            st = 2;
            mm = 1;
        else
            error("invalid coef. type");
        end
        
        for ss_idx = 1:numel(st)
            ss = st(ss_idx);
            stimConfig(:, cols(ii), ss) = weights(:, ii)*mm(ss_idx);
        end
    end
end


