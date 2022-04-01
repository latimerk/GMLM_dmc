%% getModelSetup
% gets the different types of models. By default, uses the cosine tuning model (same dir tuning for sample/test, but different cat tuning)
%
% Directions that are only used in the sample or test stimulus can be specified using the optional fields TaskInfo.Direction_sample and TaskInfo.Directions_test
%   which are logical vectors stating if the directions are used or not.
%
% optional key/value pairs are:
%  dirTuningType can be 'cosine', 'full', or 'none'
%  dirSameSampleTest = true/false (same dir tuning for sample/test, doesn't matter if dirTuningType = 'none')
%  includeCategory   = true/false (only includes sample/test signal)
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
function [stimConfig, stimPrior_setup] = getModelSetup(TaskInfo, varargin)
p = inputParser;
p.CaseSensitive = false;

addRequired( p, 'TaskInfo'     ,     @isstruct);
addParameter(p, 'dirTuningType'    , 'cosine', @(aa)(ischar(aa) || isstring(aa)));
addParameter(p, 'dirSameSampleTest',  true,    @islogical);
addParameter(p, 'includeCategory'  ,  true,    @islogical);
addParameter(p, 'includeSampleTest'  ,  true,    @islogical);


parse(p, TaskInfo, varargin{:});
% then set/get all the inputs out of this structure
dirTuningType     = p.Results.dirTuningType;
dirSameSampleTest = p.Results.dirSameSampleTest;
includeCategory   = p.Results.includeCategory;
includeSampleTest = p.Results.includeSampleTest;

%%
thetas = TaskInfo.Directions(:)';
cats   = TaskInfo.Categories(:)';
cats_unique = unique(cats);
cats_unique = cats_unique(:)';
NU = numel(cats_unique);
ND = numel(thetas);

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
cat_cols_cats  = [];


cat_test_cols      = [];
cat_test_cols_st   = [];
cat_test_cols_cats = [];

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
    if(includeSampleTest && includeCategory)
        cat_cols      = NF + (1:(NU*2));
        cat_cols_st   = [ones(1,NU) ones(1,NU)*2];
        cat_cols_cats = [cats_unique cats_unique];
        
        NF = NF + 4;
    elseif(~includeSampleTest && includeCategory)
        cat_cols      = NF + (1:NU);
        cat_cols_st   = ones(1,NU)*-1;
        cat_cols_cats = cats_unique;
        NF = NF + 2;
    elseif(includeSampleTest && ~includeCategory)
        onset_cols = NF + [1 2];
        onset_cols_st = [1 2];
        NF = NF + 2;
    elseif(~includeSampleTest && ~includeCategory)
        onset_cols =NF+ 1;
        onset_cols_st = -1;
        NF = NF + 1;
    end
    
    if(strcmpi(dirTuningType, 'cosine'))
        if(dirSameSampleTest)
            sine_cols   = NF + 1;
            cosine_cols = NF + 2;
            sine_cols_st   = -1;
            cosine_cols_st = -1;
            NF = NF + 2;
        else
            sine_cols   = NF + [1 3];
            cosine_cols = NF + [2 4];
            sine_cols_st   = [1 2];
            cosine_cols_st = [1 2];
            NF = NF + 2;
        end
    end
elseif(strcmpi(dirTuningType, 'full'))
    if(dirSameSampleTest)
        if(includeSampleTest && includeCategory)
            cat_test_cols      = NF + (1:NU);
            cat_test_cols_st   = ones(1,2)*NU;
            cat_test_cols_cats = cats_unique;
            NF = NF + 2;
        elseif(includeSampleTest && ~includeCategory)
            onset_cols = NF + 1;
            onset_cols_st = 2;
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
stimConfig = addCols(stimConfig, onset_cols, onset_cols_st);
if(~isempty(cat_cols))
    stimConfig = addCols(stimConfig, cat_cols,   cat_cols_st, cats' == cat_cols_cats);
end
if(~isempty(cat_test_cols))
    stimConfig = addCols(stimConfig, cat_test_cols,   cat_test_cols_st, cats' == cat_test_cols_cats);
end
stimConfig = addCols(stimConfig, sine_cols,   sine_cols_st, sind(TaskInfo.Directions(:)));
stimConfig = addCols(stimConfig, cosine_cols, cosine_cols_st, cosd(TaskInfo.Directions(:)));

if(~isempty(dir_cols1))
    stimConfig = addCols(stimConfig, dir_cols1, dir_cols1_st, thetas' == dir_cols1_theta);
end
if(~isempty(dir_cols2))
    stimConfig = addCols(stimConfig, dir_cols2, dir_cols2_st, thetas' == dir_cols2_theta);
end



%% construct the prior distribution
stimPrior_setup.NH = 1; %number of hyperparameters
stimPrior_setup.numFilters = NF;
stimPrior_setup.M = zeros(NF,1);
stimPrior_setup.S = 1;
stimPrior_setup.covTypes = ["independent"]; %#ok<*NBRAK>
stimPrior_setup.names    = ["contrast"];
stimPrior_setup.hyperparams_idx = {1};
stimPrior_setup.hyperprior.nu       = 4;

if(NF == 1)
    % if only 1 filter, do simple regularization
    stimPrior_setup.M = 1;
else
    %% add each filter
    ctr = 1;
    
    %% each offset filter (one hyperparam for the offset)
    NC = numel(onset_cols);
    if(NC > 0)
        stimPrior_setup.NH = stimPrior_setup.NH + 1;
        stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH;
        stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
        stimPrior_setup.names    = [stimPrior_setup.names ;"offsets"];
        stimPrior_setup.hyperprior.nu       = [stimPrior_setup.hyperprior.nu;4];
        
        
        S_new = nan(size(stimPrior_setup.S,1) + NC, size(stimPrior_setup.S,2) + NC, size(stimPrior_setup.S,3) + 1);
        S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
        S_new(ctr + (1:NC), ctr + (1:NC), end) = eye(NC);
        
        stimPrior_setup.S  = S_new;
        
        M_new = [stimPrior_setup.M zeros(NF, NC)]; 
        M_new(onset_cols, 1) = 1;
        M_new(onset_cols, ctr + (1:NC)) = eye(NC);
        stimPrior_setup.M = M_new;
        
        
        
        ctr = ctr + NC;
    end
    
    %% each category filter (one hyperparam for the category)
    NC = numel(cat_cols);
    if(NC > 0)
        stimPrior_setup.NH = stimPrior_setup.NH + 1;
        stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH;
        stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
        stimPrior_setup.names    = [stimPrior_setup.names ;"categories"];
        stimPrior_setup.hyperprior.nu       = [stimPrior_setup.hyperprior.nu;4];
        
        
        S_new = nan(size(stimPrior_setup.S,1) + NC, size(stimPrior_setup.S,2) + NC, size(stimPrior_setup.S,3) + 1);
        S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
        S_new(ctr + (1:NC), ctr + (1:NC), end) = eye(NC);
        
        stimPrior_setup.S  = S_new;
        
        M_new = [stimPrior_setup.M zeros(NF, NC)]; 
        M_new(cat_cols, 1) = 1;
        M_new(cat_cols, ctr + (1:NC)) = eye(NC);
        stimPrior_setup.M = M_new;
        
        ctr = ctr + NC;
    end
    
    
    %% each cosine/sine filter
    sine_cosine_cols = [sine_cols cosine_cols];
    NC = numel(sine_cosine_cols);
    if(NC > 0)
        stimPrior_setup.NH = stimPrior_setup.NH + 1;
        stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH;
        stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
        stimPrior_setup.names    = [stimPrior_setup.names ;"sine/cosine"];
        stimPrior_setup.hyperprior.nu       = [stimPrior_setup.hyperprior.nu;4];
        
        
        S_new = nan(size(stimPrior_setup.S,1) + NC, size(stimPrior_setup.S,2) + NC, size(stimPrior_setup.S,3) + 1);
        S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
        S_new(ctr + (1:NC), ctr + (1:NC), end) = eye(NC);
        
        stimPrior_setup.S  = S_new;
        
        M_new = [stimPrior_setup.M zeros(NF, NC)]; 
        M_new(sine_cosine_cols, ctr + (1:NC)) = eye(NC);
        stimPrior_setup.M = M_new;
        
        ctr = ctr + NC;
    end
    
    %% each dir1 filter
    GPdirAdded = false;
    
    block1_cat_S_idx = [];
    block1_cat_H_idx = [];
    
    for cc = 1:2
        switch cc
            case 1
                dir_cols = dir_cols1;
                dir_cols_theta = dir_cols1_theta;
            case 2
                dir_cols = dir_cols2;
                dir_cols_theta = dir_cols2_theta;
        end
        
        
        
        NC = numel(dir_cols);
        if(NC > 0)
            if(~GPdirAdded)
                StoAdd = 3;
                stimPrior_setup.NH = stimPrior_setup.NH + 4;
                stimPrior_setup.hyperprior.nu       = [stimPrior_setup.hyperprior.nu;ones(4,1)*4];
            else
                StoAdd = 0;
            end


            %% add category variance
            stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
            stimPrior_setup.names    = [stimPrior_setup.names ;sprintf("fulldir_cat_block%d",cc)];
                stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH + (-3);
            
            NP = NU;
            S_new = nan(size(stimPrior_setup.S,1) + NP, size(stimPrior_setup.S,2) + NP, size(stimPrior_setup.S,3) + StoAdd);
            S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
            S_new(ctr + (1:NP), ctr + (1:NP), end-2) = eye(NP);
            stimPrior_setup.S  = S_new;
            
            
            M_new = [stimPrior_setup.M zeros(NF, NP)]; 
            M_new(dir_cols, 1) = 1;
            for ii = 1:NP
                thetas_c = thetas(cats == cats_unique(ii));
                M_new(dir_cols(ismember(dir_cols_theta, thetas_c)), ctr + ii) = 1;
            end
            stimPrior_setup.M = M_new;


            if(cc == 1)
                block1_cat_S_idx = ctr + (1:NP);
                block1_cat_H_idx = size(S_new,3) - 2;
            end
            
            ctr = ctr + NP;
            %% add cosine/sine variance
            stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
            stimPrior_setup.names    = [stimPrior_setup.names ;sprintf("fulldir_sine_block%d",cc)];
                stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH + (-2);
            
            
            NP = 2;
            S_new = nan(size(stimPrior_setup.S,1) + NP, size(stimPrior_setup.S,2) + NP, size(stimPrior_setup.S,3));
            S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
            S_new(ctr + (1:NP), ctr + (1:NP), end-1) = eye(NP);
            stimPrior_setup.S  = S_new;
            
            M_new = [stimPrior_setup.M zeros(NF, NP)]; 
            M_new(dir_cols, ctr + (1:NP)) = [sind(dir_cols_theta(:)) cosd(dir_cols_theta(:))];
            stimPrior_setup.M = M_new;
            
            
            ctr = ctr + NP;


            %% add GP variance
            stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "angular_gp"];
            stimPrior_setup.names    = [stimPrior_setup.names ;sprintf("fulldir_gp_block%d",cc)];
            stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH + (-1:0);
            
            NP = NC;
            S_new = nan(size(stimPrior_setup.S,1) + NP, size(stimPrior_setup.S,2) + NP, size(stimPrior_setup.S,3));
            S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
            
            S_new(ctr + (1:NP), ctr + (1:NP), end) =  acos(cosd(dir_cols_theta(:) - dir_cols_theta(:)'));
            stimPrior_setup.S  = S_new;
            
            M_new = [stimPrior_setup.M zeros(NF, NP)]; 
            M_new(dir_cols, ctr + (1:NP)) = eye(NP);
            stimPrior_setup.M = M_new;
            ctr = ctr + NP;


            GPdirAdded = true;
        end
    end
    
    
    
    %% each test category filter which subtracts out sample cat (one hyperparam for the category)
    NC = numel(cat_test_cols);
    if(NC > 0)
        if(isempty(block1_cat_S_idx))
            warning('cat_test_cols may not be setup correctly: was expecting elements in dir_cols1 to subtract out category');
            stimPrior_setup.NH = stimPrior_setup.NH + 1;
            stimPrior_setup.hyperparams_idx{end+1} = stimPrior_setup.NH;
            stimPrior_setup.covTypes = [stimPrior_setup.covTypes; "independent"];
            stimPrior_setup.names    = [stimPrior_setup.names ;"categories_test"];


            S_new = nan(size(stimPrior_setup.S,1) + NC, size(stimPrior_setup.S,2) + NC, size(stimPrior_setup.S,3) + 1);
            S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
            S_new(ctr + (1:NC), ctr + (1:NC), end) = eye(NC);

            stimPrior_setup.S  = S_new;

            M_new = [stimPrior_setup.M zeros(NF, NC)]; 
            M_new(cat_test_cols, 1) = 1;
            M_new(cat_test_cols, ctr + (1:NC)) = eye(NC);
            stimPrior_setup.M = M_new;
        else
            %% uses cat variance defined in the GP prior
            S_new = nan(size(stimPrior_setup.S,1) + NC, size(stimPrior_setup.S,2) + NC, size(stimPrior_setup.S,3));
            S_new(1:size(stimPrior_setup.S,1), 1:size(stimPrior_setup.S,2), 1:size(stimPrior_setup.S,3)) = stimPrior_setup.S;
            S_new(ctr + (1:NC), ctr + (1:NC), block1_cat_H_idx) = eye(NC);
            stimPrior_setup.S  = S_new;
            
            M_new = [stimPrior_setup.M zeros(NF, NC)]; 
            M_new(cat_test_cols, ctr + (1:NC)) = eye(NC);
            for ii = 1:numel(cat_test_cols)
                M_new(cat_test_cols(ii), block1_cat_S_idx(cats_unique == cat_test_cols_cats(ii))) = -1;
            end
            stimPrior_setup.M = M_new;
        end
        ctr = ctr + NC; %#ok<NASGU>
    end
end



end

function [stimConfig] = addCols(stimConfig, cols, col_types, weights)
    if(nargin < 4)
        weights = 1;
    end
    if(size(weights,2) == 1)
        weights = repmat(weights, [1 numel(cols)]);
    end
    
    for ii = 1:numel(cols)
        if(col_types(ii) < 1)
            st = 1:2;
        else
            st = col_types(ii);
        end
        
        for ss = st
            stimConfig(:, cols(ii), ss) = weights(:, ii);
        end
    end
end


