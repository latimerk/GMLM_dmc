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

def LL_poiss_exp(x, y, dt):
    x += np.log(dt);
    rate = boundedExp(x);
    f = -rate + y * x;
    return f;
def DLL_poiss_exp(x, y, dt):
    x += np.log(dt);
    rate = boundedExp(x);
    f = -rate + y * x;
    g = -rate + y;
    return (f,g);

def LL_poiss_exp_nc(y):
    return -gammaln(y+1);

def LL_poiss_softplus(x, y, dt):
    rate = softplus(x);
    log_rate = np.log(rate);
    f = -rate*dt + y*(log_rate + np.log(dt));
    return f;
def DLL_poiss_softplus(x, y, dt):
    (rate, drate) = softplusD(x);
    log_rate = np.log(rate);
    f = -rate*dt + y*(log_rate + np.log(dt));
    g = -drate*dt + y*(drate/rate);
    return (f,g);

def LL_poiss_softplus_nc(y):
    return -gammaln(y+1);

def LL_truncpoiss_exp(x, y, dt):
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

def DLL_truncpoiss_exp(x, y, dt):
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

    
def LL_truncpoiss_exp_nc(y):
    return np.zeros(y.shape);

def LL_squared_error(x, y, dt):
    d = x - y;
    f = -d ** 2;
    return f;
def DLL_squared_error(x, y, dt):
    d = x - y;
    f = -d ** 2;
    g = -2 * d;
    return (f,g);

def LL_squared_error_nc(y):
    return np.zeros(y.shape);


'''
Classes to store and access the gmlm functionality easier.
'''


class GMLMHelper:
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
        self._gmlm_structure = self._gmlm.get_GMLM_structure();
        self._params  = self._gmlm.get_params();
        self._results = self._gmlm.get_results_struct();

    @property
    def dim_P(self) -> int:
        """The number of neurons in the GMLM."""
        return self._gmlm_structure.get_num_neurons()

    @property
    def GMLM_structure(self) -> pyGMLMcuda.ModelStructure:
        """The  GMLM structure."""
        return self._gmlm_structure;

    @property
    def params(self) -> pyGMLMcuda.Parameters:
        """The current parameter object."""
        return self._params;
        
    @property
    def results(self) -> pyGMLMcuda.Results:
        """The current results object."""
        return self._results;

    @property
    def is_simultaneous_recording(self) -> bool:
        """Whether the all neurons are recorded on each trial or whether a trial only has one neuron."""
        return self._gmlm_structure.is_simultaneous_recording()

    def toGPU(self) -> None:
        self._gmlm.toGPU();
    def freeGPU(self) -> None:
        self._gmlm.freeGPU();

    def vectorize_parameters(self) -> np.ndarray:
        """
        Builds a vectorized form of the parameters. 

        Returns:
          Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...

        """
        ps = [self._params.get_W().flatten(order='F'), self._params.get_B().flatten(order='F')];

        num_groups = self._gmlm_structure.get_num_groups()
        for jj in range(num_groups):
            grp_params = self._params.get_group_params(jj);
            grp_structure = self._gmlm_structure.get_group(jj);

            dim_R = grp_params.get_rank();
            if(dim_R == 0):
                continue;

            ps.append(grp_params.get_V().flatten(order='F'));
            num_modes = grp_structure.get_num_modes();
            for ss in range(num_modes):
                ps.append(grp_params.get_T(ss).flatten(order='F'));
                
        return np.concatenate(ps);

    def vectorize_gradient(self) -> np.ndarray:
        """
        Builds a vectorized form of the gradient of the current results (of log likelihood, not negative log likelihood). 

        Returns:
          Flattened gradient in order: dW, dB, Group[0].dV, Group[0].dT[0], ..., Group[0].dT[S_0-1], Group[1].dV, Group[1].dT[0], ...
        """
        grad = [self._results.get_DW().flatten(order='F'), self._results.get_DB().flatten(order='F')];

        num_groups = self._gmlm_structure.get_num_groups()
        for jj in range(num_groups):
            grp_params = self._params.get_group_params(jj);
            grp_structure = self._gmlm_structure.get_group(jj);

            dim_R = grp_params.get_rank();
            if(dim_R == 0):
                continue;

            grp_results = self._results.get_group_results(jj);
            grad.append(grp_results.get_DV().flatten(order='F'));
            num_modes = grp_structure.get_num_modes();
            for ss in range(num_modes):
                grad.append(grp_results.get_DT(ss).flatten(order='F'));

        return np.concatenate(grad);

    def devectorize_parameters(self, param_vector : np.ndarray):
        """
        Sets the parameters from a vector. Stored in self.params.

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
        """
        ctr = 0;
        W = (param_vector[ctr:(ctr + self.dim_P)]).reshape((self.dim_P,1), order='F');
        ctr += W.size;
        B = (param_vector[ctr:(ctr + self.dim_P*self._gmlm_structure.get_num_linear_terms() )]).reshape((self._gmlm_structure.get_num_linear_terms(), self.dim_P), order='F');
        ctr += B.size;
        self.params.set_linear_params(W, B);

        num_groups = self.GMLM_structure.get_num_groups()
        for jj in range(num_groups):
            grp_params = self.params.get_group_params(jj);
            grp_structure = self.GMLM_structure.get_group(jj);

            dim_R = grp_params.get_rank();
            if(dim_R == 0):
                continue;

            V = (param_vector[ctr:(ctr + self.dim_P*dim_R )]).reshape((self.dim_P, dim_R), order='F');
            grp_params.set_V(V);
            ctr += V.size;

            num_modes = grp_structure.get_num_modes();
            for ss in range(num_modes):
                dim_T = grp_structure.get_mode_dim(ss);
                T = (param_vector[ctr:(ctr + dim_T*dim_R )]).reshape((dim_T, dim_R), order='F');
                grp_params.set_T(ss, T);
                ctr += T.size;

    def compute_log_likelihood(self) -> float:
        """
        Computes the log likelihood for the current parameters.
        
        Returns:
          Log likelihood.
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        if(self._gmlm.is_on_GPU()):
            self._gmlm.set_compute_gradient(False);
            self._gmlm.compute_log_likelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return self._results.get_trial_LL().sum();

    def compute_gradient_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the negative log likelihood for a set of vectorized parameters.

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorize_parameters(param_vector);
        if(self._gmlm.is_on_GPU()):
            self._gmlm.set_compute_gradient(True);
            self._gmlm.compute_log_likelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return  -self.vectorize_gradient();

    def compute_gradient_and_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the negative log likelihood and gradient for a set of vectorized parameters.

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          A tuple (f,g)
           
            f (float): Log likelihood.
          
            g (np.ndarray): Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorize_parameters(param_vector);
        if(self._gmlm.is_on_GPU()):
            self._gmlm.set_compute_gradient(True);
            self._gmlm.compute_log_likelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return (-self._results.get_trial_LL().sum(), -self.vectorize_gradient());

    def compute_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> float:
        """
        Computes the negative log likelihood for a set of vectorized parameters.

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Negative log likelihood.
          
        Raises:
          RuntimeError: if the GMLM has not been sent to the GPU
        """
        self.devectorize_parameters(param_vector);

        if(self._gmlm.is_on_GPU()):
            self._gmlm.set_compute_gradient(False);
            self._gmlm.compute_log_likelihood(self._params);
        else:
            raise RuntimeError("GMLM GPU code called before GMLM sent to GPU.");
        return -self._results.get_trial_LL().sum();

    def randomize_parameters(self, std : float = 1.0):
        """
        Sets all the parameters to be i.i.d. normal with mean 0 and the given standard deviation.

        Args:
          std (float): the standard deviation (defaults to 1 for standard normal).
        """
        W = np.random.randn(self.dim_P) * std;
        B = np.random.randn(self.GMLM_structure.get_num_linear_terms() , self.dim_P) * std;
        self._params.set_linear_params(W, B); 

        for jj in range(self.GMLM_structure.get_num_groups()):
            grp = self.params.get_group_params(jj);
            V = np.random.randn(self.dim_P, grp.get_rank()) * std;
            grp.set_V(V);
            for ss in range(grp.get_dim_S()):
                T = np.random.randn(grp.get_dim_T(ss), grp.get_rank()) * std;
                grp.set_T(ss, T);

    def _get_F(self, factor : int, factor_idxs : list[int], dim_F : int, params_group : pyGMLMcuda.ParametersGroup) -> np.ndarray:
        """
        Computes the full coefficient for a factor using the Khatri-Rao product
        
        Args:
          factor: whether or not to compute the derivatives too.
          factor_idxs: the factor numbers of each mode
          dim_F: the total size expected for this factor
          params_group: the tensor group's parameters object 

        Returns:
          The full factor coefficients.
        
        """

        fis = np.where(np.equal(factor_idxs, factor))[0];
        F = params_group.get_T(fis[0]);

        for ss in range(1,len(fis)):
            T = params_group.get_T(fis[ss]);
            F = khatri_rao(T, F);
            # F = np.vstack([np.kron(T[:, k], F[:, k]) for k in range(F.shape[1])]).T;
        return F;


class GMLMHelperCPU(GMLMHelper):
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
          GMLMHelper: The GMLM object

        Raises:
          ValueError: if the trial setup doesn't work
        """
        super().__init__(gmlm);


        # gets the block and trial number for each trial
        NT = 0;
        for bb in range(self._gmlm.get_num_blocks()):
            NT += self._gmlm.get_block(bb).get_num_trials();

        self.tr_idx = [(-1,-1,-1,-1)] * NT;
        for bb in range(self._gmlm.get_num_blocks()): # for each block
            block = self._gmlm.get_block(bb);
            # for each trial
            for tt in range(block.get_num_trials()):
                # for the trial
                trial = block.get_trial(tt);

                tr_num = trial.get_trial_num();
                neuron_num = trial.get_neuron_num();
                if(tr_num > NT or self.tr_idx[tr_num] != (-1,-1,-1,-1)):
                    raise ValueError("Trial indices are not valid: needs to be 0,1,2,...,numTrials-1");
                else:
                    self.tr_idx[tr_num] = (bb, tt, trial.get_dim_N(), neuron_num);

        trial_lengths = [ii[2] for ii in self.tr_idx];
        if(not all(np.greater(trial_lengths, 0))):
            raise ValueError("Complete set of trials not found!");

        self.TT = np.sum(trial_lengths);
        self.dim_M = len(trial_lengths);
        self.trial_ranges = np.concatenate(([0], np.cumsum(trial_lengths)));

        if(not self.is_simultaneous_recording):    
            neuron_nums = [ii[3] for ii in self.tr_idx];
            if(not all(np.greater_equal(neuron_nums[1:], neuron_nums[0:-1]))):
                raise ValueError("Trials must be ordered by neuron number for easier computation!");
            
            neuronLengths = np.zeros(self.dim_P);
            for pp in range(self.dim_P):
                neuronLengths = np.sum(trial_lengths[neuron_nums == pp]);
            if(not all(np.greater(neuronLengths, 0))):
                raise ValueError("Complete set of neuron indices not found!");
            self.neuron_ranges = np.concatenate(([0], np.cumsum(neuronLengths)));

        # sets up the observations
        if(self.is_simultaneous_recording):
            self.Y = np.zeros((self.TT, self.dim_P));
            self.Y_norm_constants = np.zeros((self.dim_M, self.dim_P));
        else:
            self.Y = np.zeros((self.TT, 1));
            self.Y_norm_constants = np.zeros((self.dim_M, 1));


        log_like_type = self._gmlm_structure.get_log_like_type();
        for tt, idx in enumerate(self.tr_idx):
            Y_c = self._gmlm.get_block(idx[0]).get_trial(idx[1]).get_observations();
            self.Y[self.trial_ranges[tt]:self.trial_ranges[tt+1],:] = Y_c;
        
            # precomputes the normalizing constant for the trial
            if(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_exp):
                nc = LL_poiss_exp_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncpoiss_exp):
                nc = LL_truncpoiss_exp_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_softplus):
                nc = LL_poiss_softplus_nc(Y_c);
            elif(log_like_type == pyGMLMcuda.logLikeType.ll_squared_error):
                nc = LL_squared_error_nc(Y_c);
            else:
                raise ValueError("Invalid log likelihood type.");
            self.Y_norm_constants[tt,:] = nc.sum(axis=0);
        

        # sets ups the linear terms
        X_lin_size = 1;
        self.dim_B = self._gmlm_structure.get_num_linear_terms();
        if(self.dim_B > 0):
            if(self.is_simultaneous_recording):
                for tt, idx in enumerate(self.tr_idx):
                    xx = self._gmlm.get_block(idx[0]).get_trial(idx[1]).get_linear_coefficients();
                    if(xx.ndim > 2 and xx.shape[2] > 1):
                        X_lin_size = self.dim_P;
                        break;
            if(X_lin_size == 1):
                self.X_lin = np.zeros((self.TT, self.dim_B));
            else:
                self.X_lin = np.zeros((self.TT, self.dim_B, X_lin_size));

            for tt, idx in enumerate(self.tr_idx):
                xx = self._gmlm.get_block(idx[0]).get_trial(idx[1]).get_linear_coefficients();
                if(X_lin_size == 1):
                    if(xx.ndim == 2):
                        self.X_lin[self.trial_ranges[tt]:self.trial_ranges[tt+1],:] = xx;
                    elif(xx.ndim > 2 and xx.shape[2] == 1):
                        self.X_lin[self.trial_ranges[tt]:self.trial_ranges[tt+1],:] = xx[:,:,0];
                    else:
                        raise ValueError("Linear coefficient shape is bad.")
                else:
                    if(xx.ndim > 2 and xx.shape[2] == X_lin_size):
                        self.X_lin[self.trial_ranges[tt]:self.trial_ranges[tt+1],:,:] = xx;
                    elif(xx.ndim > 2 and xx.shape[2] == 1):
                        self.X_lin[self.trial_ranges[tt]:self.trial_ranges[tt+1],:,:] = np.tile(xx,(1,1,X_lin_size));
                    elif(xx.ndim == 2):
                        self.X_lin[self.trial_ranges[tt]:self.trial_ranges[tt+1],:,:] = np.tile(np.expand_dims(xx,axis=2),(1,1,X_lin_size));
                    else:
                        raise ValueError("Linear coefficient shape is bad.")
        else:
            self.X_lin = np.empty(0);

        # for each group
        self.dim_J = self._gmlm_structure.get_num_groups();
        self.X_shared = [[]] * self.dim_J;
        self.X_local = [[]] * self.dim_J;

        self.dim_D = [0] * self.dim_J;
        self.dim_S = [0] * self.dim_J;

        for jj in range(self.dim_J):
            group = self._gmlm_structure.get_group(jj);

            self.dim_D[jj] = group.get_num_factors();
            self.dim_S[jj] = group.get_num_modes();

            self.X_shared[jj] = [[]] * self.dim_D[jj];
            self.X_local[jj]  = [[]] * self.dim_D[jj];

            # for each factor
            for dd in range(self.dim_D[jj]):
              # if shared
              if(group.is_shared_regressor(dd)):
                  xx = group.get_shared_regressor(dd);

                  for tt, idx in enumerate(self.tr_idx):
                      iX_c = self._gmlm.get_block(idx[0]).get_trial(idx[1]).get_group(jj).get_shared_idx_factor(dd);
                      if(tt == 0):
                          if(iX_c.ndim == 1):
                              aa = 1;
                          else:
                              aa = iX_c.shape[1];

                          iX = np.zeros((self.TT, aa), dtype=int);

                      iX[self.trial_ranges[tt]:self.trial_ranges[tt+1],:] = iX_c;

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
                  for tt, idx in enumerate(self.tr_idx):
                      X_c = self._gmlm.get_block(idx[0]).get_trial(idx[1]).get_group(jj).get_local_factor(dd);
                      if(X_c.ndim == 1):
                          X_c = X_c.reshape((X_c.size,1));

                      if(X_c.ndim == 2):
                          aa = 1;
                      else:
                          aa = X_c.shape[2];

                      if(tt == 0):
                          X = np.zeros((self.TT * aa, X_c.shape[1]));

                      if(X_c.ndim == 2):
                          X[self.trial_ranges[tt]:self.trial_ranges[tt+1],:] = X_c;
                      else:
                          for daa in range(aa):
                              X[(self.trial_ranges[tt] + daa*self.TT):(self.trial_ranges[tt+1] + daa*self.TT),:] = X_c[:,:,daa];

                      self.X_local[jj][dd] = X;

        

    def toGPU(self) -> None:
        """
        Overwrites the toGPU function for the helper to raise an error - this type of object will not send to GPU.
        """
        raise TypeError("This wrapper is for CPU-only computations.");

        
    def compute_log_likelihood_cpu(self, computeDerivatives : bool = False) -> float:
        """
        Computes the log likelihood for the current parameters using the CPU.
        
        Args:
          computeDerivatives: whether or not to compute the derivatives too.

        Returns:
          Log likelihood.
        
        """

        # number of operations per bin
        if(self.is_simultaneous_recording):
            P = self.dim_P;
        else:
            P = 1;

        # initializes the rate for each bin with the linear terms
        if(self.dim_B > 0):
            if(self.is_simultaneous_recording and self.X_lin.ndim == 2):
                # same X_lin for each neuron in simultanesouly recorded pop
                log_rate = self.X_lin @ self._params.get_B();

            elif(self.is_simultaneous_recording):
                # different X_lin for each neuron in simultanesouly recorded pop
                log_rate = np.zeros((self.TT, P), order="F");
                for pp in range(P):
                    # np.add.at(log_rate, np.s_[:,pp], self.X_lin[:, :, pp] @ self._params.get_B()[:,pp])
                    log_rate[:,pp] = self.X_lin[:, :, pp] @ self._params.get_B()[:,pp];

            else: 
                # X_lin for each neuron in independently recorded pop
                log_rate = np.zeros((self.TT, P), order="F");
                for pp in range(P):
                    #np.add.at(log_rate, np.s_[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0], self.X_lin[self.neuron_ranges[pp]:self.neuron_ranges[pp+1], :] @ self._params.get_B()[:,pp])
                    log_rate[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0] = self.X_lin[self.neuron_ranges[pp]:self.neuron_ranges[pp+1], :] @ self._params.get_B()[:,pp];
        else:
            log_rate = np.zeros((self.TT, P), order="F");

        # add constants
        if(self.is_simultaneous_recording):
            log_rate += self._params.get_W().reshape((1, self.dim_P));
        else:
            for pp in range(P):
                # np.add.at(log_rate, np.s_[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0],  self._params.get_W()[pp])
                log_rate[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0] += self._params.get_W()[pp];

        # add group terms
        group_calculations = [[]] * self.dim_J;
        for jj in range(self.dim_J):
            group_calculations[jj] = {"c" : [], "a" : [], "m" : []};

            aa = self.GMLM_structure.get_group(jj).get_dim_A();
            params_group = self._params.get_group_params(jj);
            structure_group = self._gmlm_structure.get_group(jj);
            factor_idxs = structure_group.get_mode_parts();

            R = params_group.get_rank();
            self._results.get_group_results(jj).set_rank(R);

            if(R > 0):
                group_calculations[jj]["c"] = np.zeros((self.TT * aa, R, self.dim_D[jj]), order="F");

                # compute regressors * coef for each tensor part/mode
                for dd in range(self.dim_D[jj]):
                    F = self._get_F(dd, factor_idxs, structure_group.get_factor_dim(dd), params_group);

                    isShared = self.X_local[jj][dd] == [];

                    if(isShared):
                        XF = self.X_shared[jj][dd]["X"] @ F;
                        group_calculations[jj]["c"][self.X_shared[jj][dd]["cols"],:,dd]  = XF[self.X_shared[jj][dd]["rows"], :];
                    elif(self.X_local[jj][dd].shape[0] == self.TT and aa > 1):
                        group_calculations[jj]["c"][:,:,dd] = np.tile(self.X_local[jj][dd] @ F, (aa,1));
                    else:
                        group_calculations[jj]["c"][:,:,dd] = self.X_local[jj][dd] @ F;
            
                # combine all the modes
                group_calculations[jj]["m"] = group_calculations[jj]["c"].prod(axis=2);
                group_calculations[jj]["a"] = group_calculations[jj]["m"].reshape((self.TT,aa,R), order="F").sum(axis=1).squeeze();

                # multiply in neuron coefficients and add to rate
                if(self.is_simultaneous_recording):
                    log_rate += group_calculations[jj]["a"] @ params_group.get_V().T;
                else:
                    for pp in range(P):
                        log_rate[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0] += group_calculations[jj]["a"] @ params_group.get_V()[:,pp];
            else:
                group_calculations[jj]["c"] = 0;
                group_calculations[jj]["a"] = 0;
                group_calculations[jj]["m"] = 0;

        # compute the log likelihood for each trial
        if(computeDerivatives):
            (ll, dll) = self.derivative_log_like_function(log_rate, self.Y);
        else:
            ll = self.log_like_function(log_rate, self.Y);

        trial_log_like = self._results.get_trial_LL();
        trial_log_like[:,:] = np.add.reduceat(ll, self.trial_ranges[0:-1],axis=0).reshape(trial_log_like.shape, order="F") + self.Y_norm_constants;

        # compute the derivatives
        if(computeDerivatives):
            #dW
            dW = self._results.get_DW();
            if(self.is_simultaneous_recording):
                dW[:] = dll.sum(axis=0);
            else:
                dW[:] = np.add.reduceat(dll, self.neuron_ranges[0:-1],axis=0).reshape(dW.shape, order="F");

            #dB
            if(self._gmlm_structure.get_num_linear_terms() > 0):
                dB = self._results.get_DB();
                if(self.is_simultaneous_recording):
                    if(self.X_lin.ndim == 2):
                        dB[:,:] = self.X_lin.T @ dll;
                    else:
                        for pp in range(self.dim_P):
                            dB[:,pp] = self.X_lin[:,:,pp].T @ dll[:,pp];
                else:
                    for pp in range(self.dim_P):
                        dB[:,pp] = self.X_lin[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],:].T @ dll[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0];

            #for each group
            for jj in range(self.dim_J):
                results_group = self._results.get_group_results(jj);
                aa = self.GMLM_structure.get_group(jj).get_dim_A();
                params_group = self._params.get_group_params(jj);
                structure_group = self._gmlm_structure.get_group(jj);
                factor_idxs = structure_group.get_mode_parts();
                R = params_group.get_rank();

                #dV
                dV = results_group.get_DV();
                if(self.is_simultaneous_recording):
                    dV[:,:] = dll.T @ group_calculations[jj]["a"];
                else:
                    for pp in range(self.dim_P):
                        dV[pp,:] = group_calculations[jj]["a"][self.neuron_ranges[pp]:self.neuron_ranges[pp+1],:] @ dll[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],0];

                # for each factor
                assigned_dllv = False;
                for dd in range(self.dim_D[jj]):
                    mode_idxs = np.where(np.equal(factor_idxs,dd))[0];

                    if(not assigned_dllv):
                        # dll_v is a summary that can be used for each mode: computing in loop in case I add options to only compute specific derivatives
                        if(self.is_simultaneous_recording):
                            dll_v = np.tile(dll @ params_group.get_V(), (aa,1));
                        else:
                            dll_v = np.tile(dll, (1, R));
                            for pp in range(self.dim_P):
                                dll_v[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],:] = dll_v[self.neuron_ranges[pp]:self.neuron_ranges[pp+1],:] * params_group.get_V()[pp,:];
                            if(aa > 1):
                                dll_v = np.tile(dll_v, (aa,1));
                        assigned_dllv = True;

                    isShared = self.X_local[jj][dd] == [];
                    dds = np.arange(self.dim_D[jj]);
                    dxx = np.prod(group_calculations[jj]["c"][:,:,dds != dd], axis=2) * dll_v;

                    if(isShared):
                        dF = self.X_shared[jj][dd]["X"].T @ (self.X_shared[jj][dd]["iX"] @ dxx)
                    elif(self.X_local[jj][dd].shape[0] == self.TT and aa > 1):
                        dF = 0;
                        for aa_i in range(aa):
                            dF = dF + self.X_local[jj][dd].T @ dxx[range(aa_i*self.TT, (aa_i+1)*self.TT),:];
                    else:
                        dF = self.X_local[jj][dd].T @ dxx;

                    # for each mode in the factor
                    if(len(mode_idxs) == 1):
                        results_group.get_DT(mode_idxs[0])[:,:] = dF;
                    else:
                        for ss in mode_idxs:
                            dT = results_group.get_DT(ss);
                            dFdT = self._get_DfDT(ss, mode_idxs, structure_group.get_factor_dim(dd), params_group);
                            for rr in range(R):
                                dT[:,rr] = dFdT[:,:,rr].T @ dF[:,rr];

        # return the complete log likelihood
        return trial_log_like.sum();

    def _get_DfDT(self, mode,  mode_idxs, dim_F, params_group):
        """
        Helper function for building a structure for derivative computation.
        
        """

        dim_R = params_group.get_rank();
        dim_T = params_group.get_dim_T(mode);
        T = params_group.get_T(mode).copy();

        dFdT = np.zeros((dim_F, dim_T, dim_R), order="F");
        for rr in range(dim_R):
            for tt in range(dim_T):
                T[:,rr]  = 0;
                T[tt,rr] = 1;

                if(mode == mode_idxs[0]):
                    F_r = T[:,rr];
                else:
                    F_r = params_group.get_T(mode_idxs[0])[:,rr];

                for ss in range(1,len(mode_idxs)):
                    if(mode == mode_idxs[ss]):
                        T_r = T[:,rr];
                    else:
                        T_r = params_group.get_T(mode_idxs[ss])[:,rr];
                    F_r = khatri_rao(T_r.reshape((len(T_r),1)), F_r.reshape((len(F_r),1)));

                dFdT[:,tt,rr] = F_r.squeeze();
        return dFdT;

    def log_like_function(self, log_rate : np.ndarray, Y : np.ndarray) -> np.ndarray:
        """
        Computes the log likelihood for each bin given the linear term and observed spike counts.
        
        Args:
          log_rate: linear term in each bin. For Poisson with exponential nonlinearity, this is the log spike rate.
          Y: spike count in each bin. Should be the same size as log_rate

        Returns:
          Log likelihood of each bin. Same size as log_rate and Y
        
        """
        assert log_rate.shape == Y.shape or Y.size == 1, "log rate and spike counts do not match"

        log_like_type = self._gmlm_structure.get_log_like_type();
        dt = self._gmlm_structure.get_bin_size_sec();

        if(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_exp):
            log_like = LL_poiss_exp(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncpoiss_exp):
            log_like = LL_truncpoiss_exp(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_softplus):
            log_like = LL_poiss_softplus(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_squared_error):
            log_like = LL_squared_error(log_rate, Y, dt);
        else:
            raise ValueError("Invalid log likelihood type.");

        return log_like;

    def derivative_log_like_function(self, log_rate : np.ndarray, Y : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the log likelihood and derivative  for each bin given the linear term and observed spike counts.
        
        Args:
          log_rate: linear term in each bin. For Poisson with exponential nonlinearity, this is the log spike rate.
          Y: spike count in each bin. Should be the same size as log_rate

        Returns:
          (log likelihood of each bin. Same size as log_rate and Y, derivative of the log likelihood of each bin. Same size as log_rate and Y)
        
        """
        assert log_rate.shape == Y.shape or Y.size == 1, "log rate and spike counts do not match"

        log_like_type = self._gmlm_structure.get_log_like_type();
        dt = self._gmlm_structure.get_bin_size_sec();

        if(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_exp):
            log_like = DLL_poiss_exp(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_truncpoiss_exp):
            log_like = DLL_truncpoiss_exp(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_poiss_softplus):
            log_like = DLL_poiss_softplus(log_rate, Y, dt);
        elif(log_like_type == pyGMLMcuda.logLikeType.ll_squared_error):
            log_like = DLL_squared_error(log_rate, Y, dt);
        else:
            raise ValueError("Invalid log likelihood type.");

        return log_like;

    def compute_log_likelihood(self) -> float:
        """
        Computes the log likelihood for the current parameters.
        Computations done on CPU;
        
        Returns:
          Log likelihood.
        """
        return self.compute_log_likelihood_cpu(False);

    def compute_gradient_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the negative log likelihood for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        
        """
        self.devectorize_parameters(param_vector);
        self.compute_log_likelihood_cpu(True);
        return  -self.vectorize_gradient();

    def compute_gradient_and_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the negative log likelihood and gradient for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          A tuple (f,g)
           
            f (float): Log likelihood.
          
            g (np.ndarray): Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        """
        self.devectorize_parameters(param_vector);
        self.compute_log_likelihood_cpu(True);
        return (-self._results.get_trial_LL().sum(), -self.vectorize_gradient());

    def compute_negative_log_likelihood_vectorized(self, param_vector : np.ndarray) -> float:
        """
        Computes the negative log likelihood for a set of vectorized parameters.
        Computations done on CPU;

        Args:
          param_vector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Negative log likelihood.
        """
        self.devectorize_parameters(param_vector);
        self.compute_log_likelihood_cpu(False);
        return -self._results.get_trial_LL().sum();