import numpy as np;
import scipy.optimize as spo; 
from pyGMLM import pyGMLMcuda;
from pyGMLM.pyGMLMhelper import GMLMHelper;

## This example shows how to build a GMLM and access the values, run basic optimization, etc...

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
mode_parts_grp0 = [0, 1]; # for defining tensor structure of regressors. If mode_parts[0] = mode_parts[1] (assuming mode_parts is length 2), then the regressors will be of length modeDimensions[0] * modeDimensions[1]  but have the tensorized shape/CP decomposition
#                                              This is useful for things like spike coupling filters: neuron X coupling filter length. 
#                                              If mode_parts[0] != mode_parts[1], then we have two regressors: one of length modeDimensions[0] and a second of modeDimensions[1]. This allows for smaller/sparser representations
#                   # The order and number in mode_parts matters: must contain all ints 0 to max(mode_parts).

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
        trial_length = np.random.randint(50)+50; # in bins
        # the linear terms: shaped time x num_linear_covariates x neuron
        X_lin = np.random.randn(trial_length, num_linear_covariates, num_neurons); # note: I've always used this term for spike history
                                                                               # If the linear term is the same for all neurons in a simultaneous population recording, then X_lin can be trial_length X num_linear_covariates X 1
        # the observations: shaped time x neuron (for simultaneous population model)
        Y = np.random.poisson(1,(trial_length,num_neurons));

        trial = pyGMLMcuda.Trial(Y, X_lin, trial_ctr);
        trial_ctr += 1;

        # create regressors for group 0
        grp0 = pyGMLMcuda.TrialGroup();
        # "local" regressors (dense matrix) for mode 0: trial_length x mode_dimensions_grp0[0] x num_events_grp0
        #           If num_events_grp0 > 1 and X is supposed to be the same for all events (but other mode's terms might differ): X can be trial_length x mode_dimensions_grp0[0] x 1
        X = np.random.randn(trial_length, mode_dimensions_grp0[0], num_events_grp0);
        mode_num_0 = grp0.add_local_factor(X); # regressors for each mode are added in order (order defined by mode_parts)

        # "global" regressor indices for mode 1: integers of trial_length x num_events_grp0
        #  The regressors then become cat(3, X_global_grp0_mode1[iX[:,0],: ], X_global_grp0_mode1[iX[:,1],: ], ...)
        #  Regressors indices that are out of bounds (<0 or >X_global_grp0_mode1.shape[0]) are treated as 0's
        iX = np.random.randint(-5, X_global_grp0_mode1.shape[1] + 5, size=(trial_length, num_events_grp0));
        mode_num_1 = grp0.add_shared_idx_factor(iX);

        # create regressors for group 1
        grp1= pyGMLMcuda.TrialGroup();
        # the big set of local regressors:  trial_length x (mode_dimensions_grp1[0] * mode_dimensions_grp1[1]) X num_events_grp1
        X = np.random.randn(trial_length, mode_dimensions_grp1[0] * mode_dimensions_grp1[1], num_events_grp1);
        grp1.add_local_factor(X);
        
        # add groups to trial (in order)
        trial.add_group(grp0);
        trial.add_group(grp1);
        # add trial to block
        block.add_trial(trial);
    blocks.append(block)

# create GMLM object
gmlm = pyGMLMcuda.kcGMLM(model_structure);

# add trial blocks to GMLM
for bb in blocks:
    gmlm.add_block(bb);

# put GMLM on GPUs
gmlm.toGPU();

# get parameter object. Here's how to access and set the values
params = gmlm.get_params();
B = params.get_B(); # get linear term, num_linear_covariates x num_neurons
B[:,:] = np.random.randn(B.shape[0], B.shape[1]) * 0.1 # edit B
W = params.get_W(); # get baseline rate term,  num_neurons X 1
W[:] = np.random.randn(W.shape[0]) # edit B
params.set_W(W);
params.set_B(B);

params_grp0 = params.get_group_params(0);
params_grp0.set_rank(3); # change the rank of this group: when the parameters are sent to the GMLM with a likelihood call, the results with this rank will come back
V_grp0 = params_grp0.get_V(); # num_neurons x rank
T_grp0_0 = params_grp0.get_T(0); # mode_dimensions_grp0[0] x rank
T_grp0_1 = params_grp0.get_T(1); # mode_dimensions_grp0[1] x rank

V_grp0[:,:] = np.random.randn(V_grp0.shape[0], V_grp0.shape[1]) * 0.1;
T_grp0_0[:,:] = np.random.randn(T_grp0_0.shape[0], T_grp0_0.shape[1]);
T_grp0_1[:,:] = np.random.randn(T_grp0_1.shape[0], T_grp0_1.shape[1]);

params_grp0.set_V(V_grp0);
params_grp0.set_T(0, T_grp0_0);
params_grp0.set_T(1, T_grp0_1);

params_grp1 = params.get_group_params(1);
V_grp1 = params_grp1.get_V(); # num_neurons x rank
T_grp1_0 = params_grp1.get_T(0); # mode_dimensions_grp1[0] x rank
T_grp1_1 = params_grp1.get_T(1); # mode_dimensions_grp1[1] x rank

V_grp1[:,:] = np.random.randn(V_grp1.shape[0], V_grp1.shape[1]) * 0.1;
T_grp1_0[:,:] = np.random.randn(T_grp1_0.shape[0], T_grp1_0.shape[1]);
T_grp1_1[:,:] = np.random.randn(T_grp1_1.shape[0], T_grp1_1.shape[1]);
 
params_grp1.set_V(V_grp1);
params_grp1.set_T(0, T_grp1_0);
params_grp1.set_T(1, T_grp1_1);

# call likelihood : results will be divided up into parts
gmlm.set_compute_gradient(True); # compute function & gradient (if False, only results.get_trial_LL() will have valid values)

results = gmlm.compute_log_likelihood(params);

trial_LL = results.get_trial_LL(); # the computed log likelihood: is a matrix trials x neuron (or trials x 1 for a non-simultaneous setup)
total_LL = results.get_trial_LL().sum();

# methods to get gradients of each piece follow same pattern as the parameters
results.get_DW()
results.get_DB()
results.get_group_results(0).get_DV()
results.get_group_results(0).get_DT(0)
results.get_group_results(0).get_DT(1)
results.get_group_results(1).get_DV()
results.get_group_results(1).get_DT(0)
results.get_group_results(1).get_DT(1)

# Instead of doing this all directly, we can put the gmlm in the helper class
gmlm_h = GMLMHelper(gmlm);

# now randomize parameters and optimize
gmlm_h.randomize_parameters(0.1);

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



# free GMLM from GPUs
gmlm_h.freeGPU();