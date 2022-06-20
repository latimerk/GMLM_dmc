"""
This submodule contains convenience wrappers for performing operations like maximum likelihood estimation.
"""
import numpy as np
import scipy.sparse as sps
from scipy.linalg import khatri_rao
from scipy.special import gammaln
from pyGMLM import pyGMLMcuda

'''
Log likelihood functions

x = the rate parameter
y = the spike counts
dt = bin size (in seconds)
'''


def boundedExp(x):
    MINEXP = -90;
    MAXEXP =  90;
    return np.exp(np.maximum(np.minimum(MAXEXP,x), MINEXP));

def softplus(x):
    return x * (x >= 0) + np.log1p(np.exp(-np.abs(x)));  
def softplusD(x):
    nex = np.exp(-np.abs(x));
    f = x * (x >= 0) + np.log1p(nex);
    g = 1/(1+nex);
    return (f,g);  

def poissExpLL(x, y, dt):
    x += np.log(dt);
    rate = boundedExp(x);
    f = -rate + y * x;
    return f;
def poissExpDLL(x, y, dt):
    x += np.log(dt);
    rate = boundedExp(x);
    f = -rate + y * x;
    g = -rate + y;
    return (f,g);

def poissExpLL_nc(y):
    return -gammaln(y+1);

def poissSoftRecLL(x, y, dt):
    rate = softplus(x);
    log_rate = np.log(rate);
    f = -rate*dt + y*(log_rate + np.log(dt));
    return f;
def poissSoftRecDLL(x, y, dt):
    (rate, drate) = softplusD(x);
    log_rate = np.log(rate);
    f = -rate*dt + y*(log_rate + np.log(dt));
    g = -drate*dt + y*(drate/rate);
    return (f,g);

def poissSoftRecLL_nc(y):
    return -gammaln(y+1);

def truncatedPoissExpLL(x, y, dt):
    x += np.log(dt);
    rate     = boundedExp(x);
    expNrate = boundedExp(-rate(spks));
    
    if(len(y) > 1):
        spks = y > 0;
        f       = -rate;
        f[spks] = np.log(1.0 - expNrate);
    elif(y > 0):
        f = np.log(1.0 - expNrate);
    else:
        f       = -rate;

    return f;

def truncatedPoissExpDLL(x, y, dt):
    x += np.log(dt);
    rate     = boundedExp(x);
    expNrate = boundedExp(-rate(spks));
    exm1 = boundedExp(rate(spks)) - 1;
    
    if(len(y) > 1):
        spks = y > 0;
        f       = -rate;
        f[spks] = np.log(1.0 - expNrate);
        g = -rate;
        g[spks] = rate(spks)/exm1;
    elif(y > 0):
        f = np.log(1.0 - expNrate);
        g = rate/exm1;
    else:
        g = -rate;
        f       = -rate;

    return (f,g);

    
def truncatedPoissExpLL_nc(y):
    return np.zeros(y.shape);

def sqErrLL(x, y, dt):
    d = x - y;
    f = -d ** 2;
    return f;
def sqErrDLL(x, y, dt):
    d = x - y;
    f = -d ** 2;
    g = -2 * d;
    return (f,g);

def sqErrLL_nc(y):
    return np.zeros(y.shape);


'''
Classes to store and access the gmlm functionality easier.
'''


class gmlmHelper:
    """
    Wrapper class for the kcGMLM object (built in pybind11).
    Helps access the C++ functions of that class for operations like optimization.

    This is for the GMLM likelihood only: no hyperparameters are used or considered by the functions in this class.
    """

    def __init__(self, gmlm : pyGMLMcuda.kcGMLM):
        """
        Args:
          gmlm: The constructed built kcGMLM object.
        """
        self._gmlm = gmlm;
        self._gmlmStructure = gmlm.getGMLMStructure();
        self._params  = self._gmlm.getParams();
        self._results = self._gmlm.getResultsStruct();

    @property
    def dim_P(self) -> int:
        """The number of neurons in the GMLM."""
        return self._gmlmStructure.getNumNeurons()

    @property
    def gmlmStructure(self) -> pyGMLMcuda.kcGMLM_modelStructure:
        """The  GMLM structure."""
        return self._gmlmStructure;

    @property
    def params(self) -> pyGMLMcuda.kcGMLM_parameters:
        """The current parameter object."""
        return self._params;
        
    @property
    def results(self) -> pyGMLMcuda.kcGMLM_results:
        """The current results object."""
        return self._results;

    @property
    def isSimultaneousRecording(self) -> bool:
        """Whether the all neurons are recorded on each trial or whether a trial only has one neuron."""
        return self._gmlmStructure.isSimultaneousRecording()

    def toGPU(self) -> None:
        self._gmlm.toGPU();

    def vectorizeParameters(self) -> np.ndarray:
        """
        Builds a vectorized form of the parameters. 

        Returns:
          Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...

        """
        ps = [self._params.getW().flatten(order='F'), self._params.getB().flatten(order='F')];

        numGroups = self._gmlmStructure.getNumGroups()
        for jj in range(numGroups):
            grp_params = self._params.getGroupParams(jj);
            grp_structure = self._gmlmStructure.getGroup(jj);

            dim_R = grp_params.getRank();
            if(dim_R == 0):
                continue;

            ps.append(grp_params.getV().flatten(order='F'));
            numModes = grp_structure.getNumModes();
            for ss in range(numModes):
                ps.append(grp_params.getT(ss).flatten(order='F'));
                
        return np.concatenate(ps);

    def vectorizeGradient(self) -> np.ndarray:
        """
        Builds a vectorized form of the gradient of the current results (of log likelihood, not negative log likelihood). 

        Returns:
          Flattened gradient in order: dW, dB, Group[0].dV, Group[0].dT[0], ..., Group[0].dT[S_0-1], Group[1].dV, Group[1].dT[0], ...
        """
        grad = [self._results.getDW().flatten(order='F'), self._results.getDB().flatten(order='F')];

        numGroups = self._gmlmStructure.getNumGroups()
        for jj in range(numGroups):
            grp_params = self._params.getGroupParams(jj);
            grp_structure = self._gmlmStructure.getGroup(jj);

            dim_R = grp_params.getRank();
            if(dim_R == 0):
                continue;

            grp_results = self._results.getGroupResults(jj);
            grad.append(grp_results.getDV().flatten(order='F'));
            numModes = grp_structure.getNumModes();
            for ss in range(numModes):
                grad.append(grp_results.getDT(ss).flatten(order='F'));

        return np.concatenate(grad);

    def devectorizeParameters(self, paramVector : np.ndarray):
        """
        Sets the parameters from a vector. Stored in self.params.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
        """
        ctr = 0;
        W = (paramVector[ctr:(ctr + self.dim_P)]).reshape((self.dim_P,1), order='F');
        ctr += W.size;
        B = (paramVector[ctr:(ctr + self.dim_P*self._gmlmStructure.getNumLinearTerms() )]).reshape((self._gmlmStructure.getNumLinearTerms(), self.dim_P), order='F');
        ctr += B.size;
        self.params.setLinearParams(W, B);

        numGroups = self.gmlmStructure.getNumGroups()
        for jj in range(numGroups):
            grp_params = self.params.getGroupParams(jj);
            grp_structure = self.gmlmStructure.getGroup(jj);

            dim_R = grp_params.getRank();
            if(dim_R == 0):
                continue;

            V = (paramVector[ctr:(ctr + self.dim_P*dim_R )]).reshape((self.dim_P, dim_R), order='F');
            grp_params.setV(V);
            ctr += V.size;

            numModes = grp_structure.getNumModes();
            for ss in range(numModes):
                dim_T = grp_structure.getModeDim(ss);
                T = (paramVector[ctr:(ctr + dim_T*dim_R )]).reshape((dim_T, dim_R), order='F');
                grp_params.setT(ss, T);
                ctr += T.size;

    def computeLogLikelihood(self) -> float:
        """
        Computes the log likelihood for the current parameters.
        
        Returns:
          Log likelihood.
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        if(self._gmlm.isOnGPU()):
            self._gmlm.setComputeGradient(False);
            self._gmlm.computeLogLikelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return self._results.getTrialLL().sum();

    def computeGradientNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the negative log likelihood for a set of vectorized parameters.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorizeParameters(paramVector);
        if(self._gmlm.isOnGPU()):
            self._gmlm.setComputeGradient(True);
            self._gmlm.computeLogLikelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return  -self.vectorizeGradient();

    def computeGradientAndNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the negative log likelihood and gradient for a set of vectorized parameters.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          A tuple (f,g)
           
            f (float): Log likelihood.
          
            g (np.ndarray): Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorizeParameters(paramVector);
        if(self._gmlm.isOnGPU()):
            self._gmlm.setComputeGradient(True);
            self._gmlm.computeLogLikelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return (-self._results.getTrialLL().sum(), -self.vectorizeGradient());

    def computeNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> float:
        """
        Computes the negative log likelihood for a set of vectorized parameters.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Negative log likelihood.
          
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorizeParameters(paramVector);

        if(self._gmlm.isOnGPU()):
            self._gmlm.setComputeGradient(False);
            self._gmlm.computeLogLikelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return -self._results.getTrialLL().sum();

    def randomizeParameters(self, std : float = 1.0):
        """
        Sets all the parameters to be i.i.d. normal with mean 0 and the given standard deviation.

        Args:
          std (float): the standard deviation (defaults to 1 for standard normal).
        """
        W = np.random.randn(self.dim_P) * std;
        B = np.random.randn(self.gmlmStructure.getNumLinearTerms() , self.dim_P) * std;
        self._params.setLinearParams(W, B); 

        for jj in range(self.gmlmStructure.getNumGroups()):
            grp = self.params.getGroupParams(jj);
            V = np.random.randn(self.dim_P, grp.getRank()) * std;
            grp.setV(V);
            for ss in range(grp.getDimS()):
                T = np.random.randn(grp.getDimT(ss), grp.getRank()) * std;
                grp.setT(ss, T);

    def _getF(self, factor : int, factorIdxs : list[int], dim_F : int, paramsGroup : pyGMLMcuda.kcGMLM_parameters_tensorGroup) -> np.ndarray:
        """
        Computes the full coefficient for a factor using the Khatri-Rao product
        
        Args:
          factor: whether or not to compute the derivatives too.
          factorIdxs: the factor numbers of each mode
          dim_F: the total size expected for this factor
          paramsGroup: the tensor group's parameters object 

        Returns:
          The full factor coefficients.
        
        """

        fis = np.where(np.equal(factorIdxs, factor))[0];
        F = paramsGroup.getT(fis[0]);

        for ss in range(1,len(fis)):
            T = paramsGroup.getT(fis[ss]);
            F = khatri_rao(T, F);
            # F = np.vstack([np.kron(T[:, k], F[:, k]) for k in range(F.shape[1])]).T;
        return F;


class gmlmHelper_cpu(gmlmHelper):
    def __init__(self, gmlm  : pyGMLMcuda.kcGMLM):
        """
        Puts all the trials together into one compact structure for computing the log likelihood
        and derivative using numpy computations.

        This still uses the GMLM data in the C++ objects. This choice I made is annoying, but it keeps all the trial checking & organizing
        code the same for both the CPU and GPU versions. Currently, I don't plan to make a Python version of the class holding all the GMLM data,
        despite the fact that it'd be easier to work with. So for now, the C++ module must still be compiled even to run this numpy version
        (with cmake option '-DWITH_GPU=Off' to not require the CUDA library if desired).
        Sorry for the inconvenience.

        Args:
          gmlmHelper: The GMLM object

        Raises:
          ValueError: if the trial setup doesn't work
        """
        super().__init__(gmlm);


        # gets the block and trial number for each trial
        NT = 0;
        for bb in range(self._gmlm.getNumBlocks()):
            NT += self._gmlm.getBlock(bb).getNumTrials();

        self.trIdx = [(-1,-1,-1,-1)] * NT;
        for bb in range(self._gmlm.getNumBlocks()): # for each block
            block = self._gmlm.getBlock(bb);
            # for each trial
            for tt in range(block.getNumTrials()):
                # for the trial
                trial = block.getTrial(tt);

                trNum = trial.getTrialNum();
                neuronNum = trial.getNeuronNum();
                if(trNum > NT or self.trIdx[trNum] != (-1,-1,-1,-1)):
                    raise ValueError("Trial indices are not valid: needs to be 0,1,2,...,numTrials-1");
                else:
                    self.trIdx[trNum] = (bb, tt, trial.getDimN(), neuronNum);

        trialLengths = [ii[2] for ii in self.trIdx];
        if(not all(np.greater(trialLengths, 0))):
            raise ValueError("Complete set of trials not found!");

        self.TT = np.sum(trialLengths);
        self.dim_M = len(trialLengths);
        self.trialRanges = np.concatenate(([0], np.cumsum(trialLengths)));

        if(not self.isSimultaneousRecording):    
            neuronNums = [ii[3] for ii in self.trIdx];
            if(not all(np.greater_equal(neuronNums[1:], neuronNums[0:-1]))):
                raise ValueError("Trials must be ordered by neuron number for easier computation!");
            
            neuronLengths = np.zeros(self.dim_P);
            for pp in range(self.dim_P):
                neuronLengths = np.sum(trialLengths[neuronNums == pp]);
            if(not all(np.greater(neuronLengths, 0))):
                raise ValueError("Complete set of neuron indices not found!");
            self.neuronRanges = np.concatenate(([0], np.cumsum(neuronLengths)));

        # sets up the observations
        if(self.isSimultaneousRecording):
            self.Y = np.zeros((self.TT, self.dim_P));
            self.Y_normConstants = np.zeros((self.dim_M, self.dim_P));
        else:
            self.Y = np.zeros((self.TT, 1));
            self.Y_normConstants = np.zeros((self.dim_M, 1));


        log_like_type = self._gmlmStructure.getLogLikeType();
        for tt, idx in enumerate(self.trIdx):
            Y_c = self._gmlm.getBlock(idx[0]).getTrial(idx[1]).getObservations();
            self.Y[self.trialRanges[tt]:self.trialRanges[tt+1],:] = Y_c;
        
            # precomputes the normalizing constant for the trial
            if(log_like_type == pyGMLMcuda.logLikeType.ll_poissExp):
                nc = poissExpLL_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncatedPoissExp):
                nc = truncatedPoissExpLL_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_poissSoftRec):
                nc = poissSoftRecLL_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_sqErr):
                nc = sqErrLL_nc(Y_c);
            else:
                raise ValueError("Invalid log likelihood type.");
            self.Y_normConstants[tt,:] = nc.sum(axis=0);
        

        # sets ups the linear terms
        X_lin_size = 1;
        self.dim_B = self._gmlmStructure.getNumLinearTerms();
        if(self.dim_B > 0):
            if(self.isSimultaneousRecording):
                for tt, idx in enumerate(self.trIdx):
                    xx = self._gmlm.getBlock(idx[0]).getTrial(idx[1]).getLinearCoefficients();
                    if(xx.ndim > 2 and xx.shape[2] > 1):
                        X_lin_size = self.dim_P;
                        break;
            if(X_lin_size == 1):
                self.X_lin = np.zeros((self.TT, self.dim_B));
            else:
                self.X_lin = np.zeros((self.TT, self.dim_B, X_lin_size));

            for tt, idx in enumerate(self.trIdx):
                xx = self._gmlm.getBlock(idx[0]).getTrial(idx[1]).getLinearCoefficients();
                if(X_lin_size == 1):
                    if(xx.ndim == 2):
                        self.X_lin[self.trialRanges[tt]:self.trialRanges[tt+1],:] = xx;
                    elif(xx.ndim > 2 and xx.shape[2] == 1):
                        self.X_lin[self.trialRanges[tt]:self.trialRanges[tt+1],:] = xx[:,:,0];
                    else:
                        raise ValueError("Linear coefficient shape is bad.")
                else:
                    if(xx.ndim > 2 and xx.shape[2] == X_lin_size):
                        self.X_lin[self.trialRanges[tt]:self.trialRanges[tt+1],:,:] = xx;
                    elif(xx.ndim > 2 and xx.shape[2] == 1):
                        self.X_lin[self.trialRanges[tt]:self.trialRanges[tt+1],:,:] = np.tile(xx,(1,1,X_lin_size));
                    elif(xx.ndim == 2):
                        self.X_lin[self.trialRanges[tt]:self.trialRanges[tt+1],:,:] = np.tile(np.expand_dims(xx,axis=2),(1,1,X_lin_size));
                    else:
                        raise ValueError("Linear coefficient shape is bad.")
        else:
            self.X_lin = np.empty(0);

        # for each group
        self.dim_J = self._gmlmStructure.getNumGroups();
        self.X_shared = [[]] * self.dim_J;
        self.X_local = [[]] * self.dim_J;

        self.dim_D = [0] * self.dim_J;
        self.dim_S = [0] * self.dim_J;

        for jj in range(self.dim_J):
            group = self._gmlmStructure.getGroup(jj);

            self.dim_D[jj] = group.getNumFactors();
            self.dim_S[jj] = group.getNumModes();

            self.X_shared[jj] = [[]] * self.dim_D[jj];
            self.X_local[jj]  = [[]] * self.dim_D[jj];

            # for each factor
            for dd in range(self.dim_D[jj]):
              # if shared
              if(group.isSharedRegressor(dd)):
                  xx = group.getSharedRegressor(dd);

                  for tt, idx in enumerate(self.trIdx):
                      iX_c = self._gmlm.getBlock(idx[0]).getTrial(idx[1]).getGroup(jj).getSharedIdxFactor(dd);
                      if(tt == 0):
                          if(iX_c.ndim == 1):
                              aa = 1;
                          else:
                              aa = iX_c.shape[1];

                          iX = np.zeros((self.TT, aa), dtype=int);

                      iX[self.trialRanges[tt]:self.trialRanges[tt+1],:] = iX_c;

                  iX = iX.flatten(order="F");
                  dims = (xx.shape[0], iX.size);
                  vv   = np.logical_and(np.greater_equal(iX, 0), np.less(iX, dims[0]));
                  iX[np.logical_not(vv)] = -1;

                  nz = np.sum(vv);
                  rows = iX[vv];
                  cols = np.argwhere(vv).flatten();

                  iX_sparse = sps.coo_matrix((np.ones((nz)), (rows, cols)), shape=dims);

                  self.X_shared[jj][dd] = {"X" : xx, "nonzeros" : nz, "rows" : rows, "cols" : cols, "dims" : dims, "idx" : iX, "iX" : iX_sparse};
              # if local
              else:
                  for tt, idx in enumerate(self.trIdx):
                      X_c = self._gmlm.getBlock(idx[0]).getTrial(idx[1]).getGroup(jj).getLocalFactor(dd);
                      if(X_c.ndim == 1):
                          X_c = X_c.reshape((X_c.size,1));

                      if(X_c.ndim == 2):
                          aa = 1;
                      else:
                          aa = X_c.shape[2];

                      if(tt == 0):
                          X = np.zeros((self.TT * aa, X_c.shape[1]));

                      if(X_c.ndim == 2):
                          X[self.trialRanges[tt]:self.trialRanges[tt+1],:] = X_c;
                      else:
                          for daa in range(aa):
                              X[(self.trialRanges[tt] + daa*self.TT):(self.trialRanges[tt+1] + daa*self.TT),:] = X_c[:,:,daa];

                      self.X_local[jj][dd] = X;

        

    def toGPU(self) -> None:
        """
        Overwrites the toGPU function for the helper to raise an error - this type of object will not send to GPU.
        """
        raise TypeError("This wrapper is for CPU-only computations.");

        
    def computeLogLikelihood_cpu(self, computeDerivatives : bool = False) -> float:
        """
        Computes the log likelihood for the current parameters using the CPU.
        
        Args:
          computeDerivatives: whether or not to compute the derivatives too.

        Returns:
          Log likelihood.
        
        """

        # number of operations per bin
        if(self.isSimultaneousRecording):
            P = self.dim_P;
        else:
            P = 1;

        # initializes the rate for each bin with the linear terms
        if(self.dim_B > 0):
            if(self.isSimultaneousRecording and self.X_lin.ndim == 2):
                # same X_lin for each neuron in simultanesouly recorded pop
                log_rate = self.X_lin @ self._params.getB();

            elif(self.isSimultaneousRecording):
                # different X_lin for each neuron in simultanesouly recorded pop
                log_rate = np.zeros((self.TT, P), order="F");
                for pp in range(P):
                    # np.add.at(log_rate, np.s_[:,pp], self.X_lin[:, :, pp] @ self._params.getB()[:,pp])
                    log_rate[:,pp] = self.X_lin[:, :, pp] @ self._params.getB()[:,pp];

            else: 
                # X_lin for each neuron in independently recorded pop
                log_rate = np.zeros((self.TT, P), order="F");
                for pp in range(P):
                    #np.add.at(log_rate, np.s_[self.neuronRanges[pp]:self.neuronRanges[pp+1],0], self.X_lin[self.neuronRanges[pp]:self.neuronRanges[pp+1], :] @ self._params.getB()[:,pp])
                    log_rate[self.neuronRanges[pp]:self.neuronRanges[pp+1],0] = self.X_lin[self.neuronRanges[pp]:self.neuronRanges[pp+1], :] @ self._params.getB()[:,pp];
        else:
            log_rate = np.zeros((self.TT, P), order="F");

        # add constants
        if(self.isSimultaneousRecording):
            log_rate += self._params.getW().reshape((1, self.dim_P));
        else:
            for pp in range(P):
                # np.add.at(log_rate, np.s_[self.neuronRanges[pp]:self.neuronRanges[pp+1],0],  self._params.getW()[pp])
                log_rate[self.neuronRanges[pp]:self.neuronRanges[pp+1],0] += self._params.getW()[pp];

        # add group terms
        groupCalculations = [[]] * self.dim_J;
        for jj in range(self.dim_J):
            groupCalculations[jj] = {"c" : [], "a" : [], "m" : []};

            aa = self.gmlmStructure.getGroup(jj).getDimA();
            paramsGroup = self._params.getGroupParams(jj);
            structureGroup = self._gmlmStructure.getGroup(jj);
            factorIdxs = structureGroup.getModeParts();

            R = paramsGroup.getRank();
            self._results.getGroupResults(jj).setRank(R);

            if(R > 0):
                groupCalculations[jj]["c"] = np.zeros((self.TT * aa, R, self.dim_D[jj]), order="F");

                # compute regressors * coef for each tensor part/mode
                for dd in range(self.dim_D[jj]):
                    F = self._getF(dd, factorIdxs, structureGroup.getFactorDim(dd), paramsGroup);

                    isShared = self.X_local[jj][dd] == [];

                    if(isShared):
                        XF = self.X_shared[jj][dd]["X"] @ F;
                        groupCalculations[jj]["c"][self.X_shared[jj][dd]["cols"],:,dd]  = XF[self.X_shared[jj][dd]["rows"], :];
                    elif(self.X_local[jj][dd].shape[0] == self.TT and aa > 1):
                        groupCalculations[jj]["c"][:,:,dd] = np.tile(self.X_local[jj][dd] @ F, (aa,1));
                    else:
                        groupCalculations[jj]["c"][:,:,dd] = self.X_local[jj][dd] @ F;
            
                # combine all the modes
                groupCalculations[jj]["m"] = groupCalculations[jj]["c"].prod(axis=2);
                groupCalculations[jj]["a"] = groupCalculations[jj]["m"].reshape((self.TT,aa,R), order="F").sum(axis=1).squeeze();

                # multiply in neuron coefficients and add to rate
                if(self.isSimultaneousRecording):
                    log_rate += groupCalculations[jj]["a"] @ paramsGroup.getV().T;
                else:
                    for pp in range(P):
                        log_rate[self.neuronRanges[pp]:self.neuronRanges[pp+1],0] += groupCalculations[jj]["a"] @ paramsGroup.getV()[:,pp];
            else:
                groupCalculations[jj]["c"] = 0;
                groupCalculations[jj]["a"] = 0;
                groupCalculations[jj]["m"] = 0;

        # compute the log likelihood for each trial
        if(computeDerivatives):
            (ll, dll) = self.logLikeFunD(log_rate, self.Y);
        else:
            ll = self.logLikeFun(log_rate, self.Y);

        trialLL = self._results.getTrialLL();
        trialLL[:,:] = np.add.reduceat(ll, self.trialRanges[0:-1],axis=0).reshape(trialLL.shape, order="F") + self.Y_normConstants;

        # compute the derivatives
        if(computeDerivatives):
            #dW
            dW = self._results.getDW();
            if(self.isSimultaneousRecording):
                dW[:] = dll.sum(axis=0);
            else:
                dW[:] = np.add.reduceat(dll, self.neuronRanges[0:-1],axis=0).reshape(dW.shape, order="F");

            #dB
            if(self._gmlmStructure.getNumLinearTerms() > 0):
                dB = self._results.getDB();
                if(self.isSimultaneousRecording):
                    if(self.X_lin.ndim == 2):
                        dB[:,:] = self.X_lin.T @ dll;
                    else:
                        for pp in range(self.dim_P):
                            dB[:,pp] = self.X_lin[:,:,pp].T @ dll[:,pp];
                else:
                    for pp in range(self.dim_P):
                        dB[:,pp] = self.X_lin[self.neuronRanges[pp]:self.neuronRanges[pp+1],:].T @ dll[self.neuronRanges[pp]:self.neuronRanges[pp+1],0];

            #for each group
            for jj in range(self.dim_J):
                resultsGroup = self._results.getGroupResults(jj);
                aa = self.gmlmStructure.getGroup(jj).getDimA();
                paramsGroup = self._params.getGroupParams(jj);
                structureGroup = self._gmlmStructure.getGroup(jj);
                factorIdxs = structureGroup.getModeParts();
                R = paramsGroup.getRank();

                #dV
                dV = resultsGroup.getDV();
                if(self.isSimultaneousRecording):
                    dV[:,:] = dll.T @ groupCalculations[jj]["a"];
                else:
                    for pp in range(self.dim_P):
                        dV[pp,:] = groupCalculations[jj]["a"][self.neuronRanges[pp]:self.neuronRanges[pp+1],:] @ dll[self.neuronRanges[pp]:self.neuronRanges[pp+1],0];

                # for each factor
                assigned_dllv = False;
                for dd in range(self.dim_D[jj]):
                    modeIdxs = np.where(np.equal(factorIdxs,dd))[0];

                    if(not assigned_dllv):
                        # dll_v is a summary that can be used for each mode: computing in loop in case I add options to only compute specific derivatives
                        if(self.isSimultaneousRecording):
                            dll_v = np.tile(dll @ paramsGroup.getV(), (aa,1));
                        else:
                            dll_v = np.tile(dll, (1, R));
                            for pp in range(self.dim_P):
                                dll_v[self.neuronRanges[pp]:self.neuronRanges[pp+1],:] = dll_v[self.neuronRanges[pp]:self.neuronRanges[pp+1],:] * paramsGroup.getV()[pp,:];
                            if(aa > 1):
                                dll_v = np.tile(dll_v, (aa,1));
                        assigned_dllv = True;

                    isShared = self.X_local[jj][dd] == [];
                    dds = np.arange(self.dim_D[jj]);
                    dxx = np.prod(groupCalculations[jj]["c"][:,:,dds != dd], axis=2) * dll_v;

                    if(isShared):
                        dF = self.X_shared[jj][dd]["X"].T @ (self.X_shared[jj][dd]["iX"] @ dxx)
                    elif(self.X_local[jj][dd].shape[0] == self.TT and aa > 1):
                        dF = 0;
                        for aa_i in range(aa):
                            dF = dF + self.X_local[jj][dd].T @ dxx[range(aa_i*self.TT, (aa_i+1)*self.TT),:];
                    else:
                        dF = self.X_local[jj][dd].T @ dxx;

                    # for each mode in the factor
                    if(len(modeIdxs) == 1):
                        resultsGroup.getDT(modeIdxs[0])[:,:] = dF;
                    else:
                        for ss in modeIdxs:
                            dT = resultsGroup.getDT(ss);
                            dFdT = self._getDfDT(ss, modeIdxs, structureGroup.getFactorDim(dd), paramsGroup);
                            for rr in range(R):
                                dT[:,rr] = dFdT[:,:,rr].T @ dF[:,rr];

        # return the complete log likelihood
        return trialLL.sum();

    def _getDfDT(self, mode,  modeIdxs, dim_F, paramsGroup):
        """
        Helper function for building a structure for derivative computation.
        
        """

        dim_R = paramsGroup.getRank();
        dim_T = paramsGroup.getDimT(mode);
        T = paramsGroup.getT(mode).copy();

        dFdT = np.zeros((dim_F, dim_T, dim_R), order="F");
        for rr in range(dim_R):
            for tt in range(dim_T):
                T[:,rr]  = 0;
                T[tt,rr] = 1;

                if(mode == modeIdxs[0]):
                    F_r = T[:,rr];
                else:
                    F_r = paramsGroup.getT(modeIdxs[0])[:,rr];

                for ss in range(1,len(modeIdxs)):
                    if(mode == modeIdxs[ss]):
                        T_r = T[:,rr];
                    else:
                        T_r = paramsGroup.getT(modeIdxs[ss])[:,rr];
                    F_r = khatri_rao(T_r.reshape((len(T_r),1)), F_r.reshape((len(F_r),1)));

                dFdT[:,tt,rr] = F_r.squeeze();
        return dFdT;

    def logLikeFun(self, log_rate : np.ndarray, Y : np.ndarray) -> np.ndarray:
        """
        Computes the log likelihood for each bin given the linear term and observed spike counts.
        
        Args:
          log_rate: linear term in each bin. For Poisson with exponential nonlinearity, this is the log spike rate.
          Y: spike count in each bin. Should be the same size as log_rate

        Returns:
          Log likelihood of each bin. Same size as log_rate and Y
        
        """
        assert log_rate.shape == Y.shape or Y.size == 1, "log rate and spike counts do not match"

        log_like_type = self._gmlmStructure.getLogLikeType();
        dt = self._gmlmStructure.getBinSize();

        if(log_like_type == pyGMLMcuda.logLikeType.ll_poissExp):
            log_like = poissExpLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncatedPoissExp):
            log_like = truncatedPoissExpLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_poissSoftRec):
            log_like = poissSoftRecLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_sqErr):
            log_like = sqErrLL(log_rate, Y, dt);
        else:
            raise ValueError("Invalid log likelihood type.");

        return log_like;

    def logLikeFunD(self, log_rate : np.ndarray, Y : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the log likelihood and derivative  for each bin given the linear term and observed spike counts.
        
        Args:
          log_rate: linear term in each bin. For Poisson with exponential nonlinearity, this is the log spike rate.
          Y: spike count in each bin. Should be the same size as log_rate

        Returns:
          (log likelihood of each bin. Same size as log_rate and Y, derivative of the log likelihood of each bin. Same size as log_rate and Y)
        
        """
        assert log_rate.shape == Y.shape or Y.size == 1, "log rate and spike counts do not match"

        log_like_type = self._gmlmStructure.getLogLikeType();
        dt = self._gmlmStructure.getBinSize();

        if(log_like_type == pyGMLMcuda.logLikeType.ll_poissExp):
            log_like = poissExpDLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncatedPoissExp):
            log_like = truncatedPoissExpDLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_poissSoftRec):
            log_like = poissSoftRecDLL(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_sqErr):
            log_like = sqErrDLL(log_rate, Y, dt);
        else:
            raise ValueError("Invalid log likelihood type.");

        return log_like;

    def computeLogLikelihood(self) -> float:
        """
        Computes the log likelihood for the current parameters.
        Computations done on CPU;
        
        Returns:
          Log likelihood.
        """
        return self.computeLogLikelihood_cpu(False);

    def computeGradientNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the negative log likelihood for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        """
        self.devectorizeParameters(paramVector);
        self.computeLogLikelihood_cpu(True);
        return  -self.vectorizeGradient();

    def computeGradientAndNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the negative log likelihood and gradient for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          A tuple (f,g)
           
            f (float): Log likelihood.
          
            g (np.ndarray): Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        """
        self.devectorizeParameters(paramVector);
        self.computeLogLikelihood_cpu(True);
        return (-self._results.getTrialLL().sum(), -self.vectorizeGradient());

    def computeNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> float:
        """
        Computes the negative log likelihood for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Negative log likelihood.
        """
        self.devectorizeParameters(paramVector);
        self.computeLogLikelihood_cpu(False);
        return -self._results.getTrialLL().sum();