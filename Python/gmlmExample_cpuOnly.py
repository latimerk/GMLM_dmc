import numpy as np;
import scipy.optimize as spo; 
from pyGMLM import pyGMLMcuda;
from pyGMLM.pyGMLMhelper import GMLMHelperCPU;

## This example shows how to build a GMLM and access the values, run basic optimization, etc...
# this is almost identical to gmlmExample.py : it just doesn't call the toGPU function and uses the gmlmHelper_cpu class.

## define the structure of the GMLM and data+regressors

# number of linear terms
num_neurons = 16; # number of neurons
num_linear_covariates = 8; # number of full linear covariates (GLM portion of the GMLM)
ll_type = pyGMLMcuda.ll_poiss_exp; # likelihood type: here is standard Poisson
bin_size_sec = 0.010; # bin size in seconds
is_simultaneous_recording = True; # if the cells are simultaneously recorded or not

model_structure = pyGMLMcuda.ModelStructure(num_neurons, num_linear_covariates, ll_type,  bin_size_sec, is_simultaneous_recording);

# Add a set of tensor parameters to the model
num_events_grp0 = 2; # Number of "events" for the multilinear filter
max_rank_grp0 = 8; # Max rank for the group (for memory allocation purposes) : rank is initalized to this max value
mode_dimensions_grp0 = [12, 8]; # dimensions of each mode
mode_parts_grp0 = [0, 1]; # for defining tensor structure of regressors. If modeParts[0] = modeParts[1] (assuming modeParts is length 2), then the regressors will be of length modeDimensions[0] * modeDimensions[1]  but have the tensorized shape/CP decomposition
#                                              This is useful for things like spike coupling filters: neuron X coupling filter length. 
#                                              If modeParts[0] != modeParts[1], then we have two regressors: one of length modeDimensions[0] and a second of modeDimensions[1]. This allows for smaller/sparser representations
#                   # The order and number in modeParts matters: must contain all ints 0 to max(modeParts).

model_structure_grp0 = pyGMLMcuda.ModelStructureGroup(num_events_grp0, max_rank_grp0, mode_dimensions_grp0, mode_parts_grp0);

# set dimension 1 to be a "global or shared regressor": this is a sparse structure that I exploited for increased speed
#   Also useful for trial-wise components (a la TCA): the global/shared regressors can be the identity matrix and we can specify trial numbers as indices.
X_global_grp0_mode1 = np.random.randn(32, mode_dimensions_grp0[1]);
model_structure_grp0.set_shared_regressors(1, X_global_grp0_mode1);

model_structure.add_group(model_structure_grp0);

# Add a second set of tensor parameters
num_events_grp1 = 1;
max_rank_grp1 = 6;
mode_dimensions_grp1 = [4, 8];
mode_parts_grp1 = [0,0];

model_structure_grp1 = pyGMLMcuda.ModelStructureGroup(num_events_grp1, max_rank_grp1, mode_dimensions_grp1, mode_parts_grp1);
model_structure.add_group(model_structure_grp1);


# Add a third set of tensor parameters
num_events_grp2 = 1;
max_rank_grp2 = 6;
mode_dimensions_grp2 = [4, 8];
mode_parts_grp2 = [0,1];

model_structure_grp2 = pyGMLMcuda.ModelStructureGroup(num_events_grp2, max_rank_grp2, mode_dimensions_grp2, mode_parts_grp2);
model_structure.add_group(model_structure_grp2);

# Making some trials now: these will be purely random, just to test if the functions can be called without any seg faults or other errors
# Trials are divided into blocks. Each block is on a single GPU. (Multiple blocks can be given on the same GPU if you want)
GPUs_for_blocks = [0, 0]; # the device numbers of each block
num_blocks = len(GPUs_for_blocks);
num_trials_per_block = 10; # NOTE: blocks can have different numbers of trials if you want
trial_ctr = 0; # need to specify the absolute trial number (for combining blocks and such)

blocks = list();
for bb in range(num_blocks):
    # create block
    # GPUGMLM_trialBlock_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_, unsigned int devNum)   
    block = pyGMLMcuda.TrialBlock(model_structure, GPUs_for_blocks[bb]);
    for tt in range(num_trials_per_block):
        #create trial
        trialLength = np.random.randint(50)+50; # in bins
        # the linear terms: shaped time x num_linear_covariates x neuron
        X_lin = np.random.randn(trialLength, num_linear_covariates, num_neurons); # note: I've always used this term for spike history
                                                                               # If the linear term is the same for all neurons in a simultaneous population recording, then X_lin can be trialLength X num_linear_covariates X 1
        # the observations: shaped time x neuron (for simultaneous population model)
        Y = np.random.poisson(1,(trialLength,num_neurons));

        trial = pyGMLMcuda.Trial(Y, X_lin, trial_ctr);
        trial_ctr += 1;

        # create regressors for group 0
        grp0 = pyGMLMcuda.TrialGroup();
        # "local" regressors (dense matrix) for mode 0: trialLength x mode_dimensions_grp0[0] x num_events_grp0
        #           If num_events_grp0 > 1 and X is supposed to be the same for all events (but other mode's terms might differ): X can be trialLength x mode_dimensions_grp0[0] x 1
        X = np.random.randn(trialLength, mode_dimensions_grp0[0], num_events_grp0);
        modeNum_0 = grp0.add_local_factor(X); # regressors for each mode are added in order (order defined by modeParts)

        # "global" regressor indices for mode 1: integers of trialLength x num_events_grp0
        #  The regressors then become cat(3, X_global_grp0_mode1[iX[:,0],: ], X_global_grp0_mode1[iX[:,1],: ], ...)
        #  Regressors indices that are out of bounds (<0 or >X_global_grp0_mode1.shape[0]) are treated as 0's
        iX = np.random.randint(-5, X_global_grp0_mode1.shape[0] + 5, size=(trialLength, num_events_grp0));
        # iX = np.random.randint(0, 3, size=(trialLength, num_events_grp0));
        modeNum_1 = grp0.add_shared_idx_factor(iX);

        # create regressors for group 1
        grp1= pyGMLMcuda.TrialGroup();
        # the big set of local regressors:  trialLength x (mode_dimensions_grp1[0] * mode_dimensions_grp1[1]) X num_events_grp1
        X = np.random.randn(trialLength, mode_dimensions_grp1[0] * mode_dimensions_grp1[1], num_events_grp1);
        grp1.add_local_factor(X);

        
        # create regressors for group 2
        grp2= pyGMLMcuda.TrialGroup();
        X = np.random.randn(trialLength, mode_dimensions_grp2[0] , num_events_grp2);
        grp2.add_local_factor(X);
        X = np.random.randn(trialLength, mode_dimensions_grp2[1] , num_events_grp2);
        grp2.add_local_factor(X);
        
        # add groups to trial (in order)
        trial.add_group(grp0);
        trial.add_group(grp1);
        trial.add_group(grp2);
        # add trial to block
        block.add_trial(trial);
    blocks.append(block)

# create GMLM object
gmlm = pyGMLMcuda.kcGMLM(model_structure);

# add trial blocks to GMLM
for bb in blocks:
    gmlm.add_block(bb);

# put the gmlm in the helper class
gmlm_h = GMLMHelperCPU(gmlm);

# now randomize parameters 
gmlm_h.randomize_parameters(0.1);


# optimize
total_LL = gmlm_h.compute_log_likelihood();

# call likelihood
x0 = gmlm_h.vectorize_parameters();
nll_fun     = lambda x : gmlm_h.compute_negative_log_likelihood_vectorized(x);
nll_grad_fun = lambda x : gmlm_h.compute_gradient_negative_log_likelihood_vectorized(x);
nll_fun2 = lambda x : gmlm_h.compute_gradient_and_negative_log_likelihood_vectorized(x);

from timeit import default_timer as timer
start = timer()
N = 100;
for ii in range(N):
    nll_fun2(x0);
end = timer()
print("Time per gradient evaluation: " + str((end - start) / N * 1000) + " ms")

# optimization_results = spo.minimize(nll_fun, x0, jac=nll_grad_fun, options = {"disp" : True, "maxiter" : 500});
optimization_results = spo.minimize(nll_fun2, x0, jac=True, options = {"disp" : True, "maxiter" : 100});

gmlm_h.devectorize_parameters(optimization_results.x);

total_LL_2 = gmlm_h.compute_log_likelihood();

print("log likelihood init: " + str(total_LL))
print("log likelihood final: " + str(total_LL_2))


