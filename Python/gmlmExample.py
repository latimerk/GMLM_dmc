import numpy as np;
import scipy.optimize as spo; 
from pyGMLM import pyGMLMcuda;
from pyGMLM.pyGMLMhelper import gmlmHelper;

## This example shows how to build a GMLM and access the values, run basic optimization, etc...

## define the structure of the GMLM and data+regressors

# number of linear terms
numNeurons = 16; # number of neurons
numLinearCovariates = 8; # number of full linear covariates (GLM portion of the GMLM)
llType = pyGMLMcuda.ll_poissExp; # likelihood type: here is standard Poisson
binSize_sec = 0.010; # bin size in seconds
isSimultaneousRecording = True; # if the cells are simultaneously recorded or not

modelStructure = pyGMLMcuda.kcGMLM_modelStructure(numNeurons, numLinearCovariates, llType,  binSize_sec, isSimultaneousRecording);

# Add a set of tensor parameters to the model
numEvents_grp0 = 2; # Number of "events" for the multilinear filter
maxRank_grp0 = 8; # Max rank for the group (for memory allocation purposes) : rank is initalized to this max value
modeDimensions_grp0 = [12, 8]; # dimensions of each mode
modeParts_grp0 = [0, 1]; # for defining tensor structure of regressors. If modeParts[0] = modeParts[1] (assuming modeParts is length 2), then the regressors will be of length modeDimensions[0] * modeDimensions[1]  but have the tensorized shape/CP decomposition
#                                              This is useful for things like spike coupling filters: neuron X coupling filter length. 
#                                              If modeParts[0] != modeParts[1], then we have two regressors: one of length modeDimensions[0] and a second of modeDimensions[1]. This allows for smaller/sparser representations
#                   # The order and number in modeParts matters: must contain all ints 0 to max(modeParts).

modelStructure_grp0 = pyGMLMcuda.kcGMLM_modelStructure_tensorGroup(numEvents_grp0, maxRank_grp0, modeDimensions_grp0, modeParts_grp0);

# set dimension 1 to be a "global or shared regressor": this is a sparse structure that I exploited for increased speed
#   Also useful for trial-wise components (a la TCA): the global/shared regressors can be the identity matrix and we can specify trial numbers as indices.
X_global_grp0_mode1 = np.random.randn(32, modeDimensions_grp0[1]);
modelStructure_grp0.setSharedRegressors(1, X_global_grp0_mode1);

modelStructure.addGroup(modelStructure_grp0);

# Add a second set of tensor parameters
numEvents_grp1 = 1;
maxRank_grp1 = 6;
modeDimensions_grp1 = [4, 8];
modeParts_grp1 = [0,0];

modelStructure_grp1 = pyGMLMcuda.kcGMLM_modelStructure_tensorGroup(numEvents_grp1, maxRank_grp1, modeDimensions_grp1, modeParts_grp1);
modelStructure.addGroup(modelStructure_grp1);

# Making some trials now: these will be purely random, just to test if the functions can be called without any seg faults or other errors
# Trials are divided into blocks. Each block is on a single GPU. (Multiple blocks can be given on the same GPU if you want)
GPUsForBlocks = [0, 0]; # the device numbers of each block
numBlocks = len(GPUsForBlocks);
numTrialsPerBlock = 10; # NOTE: blocks can have different numbers of trials if you want
trialNum = 0; # need to specify the absolute trial number (for combining blocks and such)

blocks = list();
for bb in range(numBlocks):
    # create block
    # GPUGMLM_trialBlock_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_, unsigned int devNum)   
    block = pyGMLMcuda.kcGMLM_trialBlock(modelStructure, GPUsForBlocks[bb]);
    for tt in range(numTrialsPerBlock):
        #create trial
        trialLength = np.random.randint(50)+50; # in bins
        # the linear terms: shaped time x numLinearCovariates x neuron
        X_lin = np.random.randn(trialLength, numLinearCovariates, numNeurons); # note: I've always used this term for spike history
                                                                               # If the linear term is the same for all neurons in a simultaneous population recording, then X_lin can be trialLength X numLinearCovariates X 1
        # the observations: shaped time x neuron (for simultaneous population model)
        Y = np.random.poisson(1,(trialLength,numNeurons));

        trial = pyGMLMcuda.kcGMLM_trial(Y, X_lin, trialNum);
        trialNum += 1;

        # create regressors for group 0
        grp0 = pyGMLMcuda.kcGMLM_trial_tensorGroup();
        # "local" regressors (dense matrix) for mode 0: trialLength x modeDimensions_grp0[0] x numEvents_grp0
        #           If numEvents_grp0 > 1 and X is supposed to be the same for all events (but other mode's terms might differ): X can be trialLength x modeDimensions_grp0[0] x 1
        X = np.random.randn(trialLength, modeDimensions_grp0[0], numEvents_grp0);
        modeNum_0 = grp0.addLocalFactor(X); # regressors for each mode are added in order (order defined by modeParts)

        # "global" regressor indices for mode 1: integers of trialLength x numEvents_grp0
        #  The regressors then become cat(3, X_global_grp0_mode1[iX[:,0],: ], X_global_grp0_mode1[iX[:,1],: ], ...)
        #  Regressors indices that are out of bounds (<0 or >X_global_grp0_mode1.shape[0]) are treated as 0's
        iX = np.random.randint(-5, X_global_grp0_mode1.shape[1] + 5, size=(trialLength, numEvents_grp0));
        modeNum_1 = grp0.addSharedIdxFactor(iX);

        # create regressors for group 1
        grp1= pyGMLMcuda.kcGMLM_trial_tensorGroup();
        # the big set of local regressors:  trialLength x (modeDimensions_grp1[0] * modeDimensions_grp1[1]) X numEvents_grp1
        X = np.random.randn(trialLength, modeDimensions_grp1[0] * modeDimensions_grp1[1], numEvents_grp1);
        grp1.addLocalFactor(X);
        
        # add groups to trial (in order)
        trial.addGroup(grp0);
        trial.addGroup(grp1);
        # add trial to block
        block.addTrial(trial);
    blocks.append(block)

# create GMLM object
gmlm = pyGMLMcuda.kcGMLM(modelStructure);

# add trial blocks to GMLM
for bb in blocks:
    gmlm.addBlock(bb);

# put GMLM on GPUs
gmlm.toGPU();

# get parameter object. Here's how to access and set the values
params = gmlm.getParams();
B = params.getB(); # get linear term, numLinearCovariates x numNeurons
B[:,:] = np.random.randn(B.shape[0], B.shape[1]) * 0.1 # edit B
W = params.getW(); # get baseline rate term,  numNeurons X 1
W[:] = np.random.randn(W.shape[0]) # edit B
params.setLinearParams(W, B); # I don't know why this is one function while the getters are split

params_grp0 = params.getGroupParams(0);
params_grp0.setRank(3); # change the rank of this group: when the parameters are sent to the GMLM with a likelihood call, the results with this rank will come back
V_grp0 = params_grp0.getV(); # numNeurons x rank
T_grp0_0 = params_grp0.getT(0); # modeDimensions_grp0[0] x rank
T_grp0_1 = params_grp0.getT(1); # modeDimensions_grp0[1] x rank

V_grp0[:,:] = np.random.randn(V_grp0.shape[0], V_grp0.shape[1]) * 0.1;
T_grp0_0[:,:] = np.random.randn(T_grp0_0.shape[0], T_grp0_0.shape[1]);
T_grp0_1[:,:] = np.random.randn(T_grp0_1.shape[0], T_grp0_1.shape[1]);

params_grp0.setV(V_grp0);
params_grp0.setT(0, T_grp0_0);
params_grp0.setT(1, T_grp0_1);

params_grp1 = params.getGroupParams(1);
V_grp1 = params_grp1.getV(); # numNeurons x rank
T_grp1_0 = params_grp1.getT(0); # modeDimensions_grp1[0] x rank
T_grp1_1 = params_grp1.getT(1); # modeDimensions_grp1[1] x rank

V_grp1[:,:] = np.random.randn(V_grp1.shape[0], V_grp1.shape[1]) * 0.1;
T_grp1_0[:,:] = np.random.randn(T_grp1_0.shape[0], T_grp1_0.shape[1]);
T_grp1_1[:,:] = np.random.randn(T_grp1_1.shape[0], T_grp1_1.shape[1]);
 
params_grp1.setV(V_grp1);
params_grp1.setT(0, T_grp1_0);
params_grp1.setT(1, T_grp1_1);

# call likelihood : results will be divided up into parts
gmlm.setComputeGradient(True); # compute function & gradient (if False, only results.getTrialLL() will have valid values)

results = gmlm.computeLogLikelihood(params);

trialLL = results.getTrialLL(); # the computed log likelihood: is a matrix trials x neuron (or trials x 1 for a non-simultaneous setup)
totalLL = results.getTrialLL().sum();

# methods to get gradients of each piece follow same pattern as the parameters
results.getDW()
results.getDB()
results.getGroupResults(0).getDV()
results.getGroupResults(0).getDT(0)
results.getGroupResults(0).getDT(1)
results.getGroupResults(1).getDV()
results.getGroupResults(1).getDT(0)
results.getGroupResults(1).getDT(1)

# Instead of doing this all directly, we can put the gmlm in the helper class
gmlm_h = gmlmHelper(gmlm);

# now randomize parameters and optimize
gmlm_h.randomizeParameters(0.1);

x0 = gmlm_h.vectorizeParameters();
nllFun     = lambda x : gmlm_h.computeNegativeLogLikelihood_vectorized(x);
nllGradFun = lambda x : gmlm_h.computeGradientNegativeLogLikelihood_vectorized(x);
nllFun2 = lambda x : gmlm_h.computeGradientAndNegativeLogLikelihood_vectorized(x);

from timeit import default_timer as timer
start = timer()
N = 100;
for ii in range(N):
    nllFun2(x0);
end = timer()
print("Time per gradient evaluation: " + str((end - start) / N * 1000) + " ms")

# optimizationResults = spo.minimize(nllFun, x0, jac=nllGradFun, options = {"disp" : True, "maxiter" : 500});
optimizationResults = spo.minimize(nllFun2, x0, jac=True, options = {"disp" : True, "maxiter" : 100});

gmlm_h.devectorizeParameters(optimizationResults.x);

totalLL_2 = gmlm_h.computeLogLikelihood();

print("log likelihood init: " + str(totalLL))
print("log likelihood final: " + str(totalLL_2))



# free GMLM from GPUs
gmlm.freeGPU();