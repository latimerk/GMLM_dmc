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
function [prior_function_stim, prior_function_lever, prior_function_spkHist, prior_function_glmComplete, levPrior_setup, spkHistPrior_setup, prior_function_dynSpkHist] = setupPriorFunctions(stimPrior_setup, bases, dynHspkPrior_setup)
if(nargin < 3)
    useSharedHspk = false;
end

prior_function_stim    = @(params, results, groupNum) DMC.priors.GMLMprior_stim(params, results, groupNum, stimPrior_setup); 
    
%prior over lever filters (1 hyperparameter H_lev)
%  Prior assumes each component is indepedent:
%  P(params.Groups(lever).T{1}(:,jj)) ~ N(0, I*exp(H(1)*2))
levPrior_setup.hyperprior.nu       = 4; %for a half-t distribution over the standard deviations (exp(H(1))
levPrior_setup.NH       = 1; %number of hyperparameters
levPrior_setup.numBases = size(bases.leverBasis, 2);
prior_function_lever    = @(params, results, groupNum) DMC.priors.GMLMprior_lever(params, results, groupNum,  levPrior_setup); 

%prior over spike history filters (1 hyperparameter H_spk)
%  Prior assumes each neuron's spike history filter is indepedent:
%  P(params.B(:,jj)) ~ N(0, I*exp(H_spk(1)))

spkHistPrior_setup.hyperprior.nu       = 4;
spkHistPrior_setup.NH       = 1; %number of hyperparameters

spkHistPrior_setup.numBases = size(bases.spkHistBasis, 2);
prior_function_spkHist      = @(params, results) DMC.priors.GMLMprior_spkHist(params, results, spkHistPrior_setup);

if(nargin > 2)
    prior_function_dynSpkHist    = @(params, results, groupNum) DMC.priors.GMLMprior_stim(params, results, groupNum, dynHspkPrior_setup); 
end

%note: the example does not include a prior over the baseline rate parameters (assumes an improper uniform prior over params.W)
prior_function_glmComplete = @(params, results) DMC.priors.GLMprior_allFilters(params, results, stimPrior_setup, levPrior_setup, spkHistPrior_setup);