"""
This submodule contains convenience wrappers for performing operations like maximum likelihood estimation.
"""
import numpy as np
from pyGMLM import pyGMLMcuda

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
        self._results = self._gmlm.getResults();

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

    def computeLogLikelihood(self)-> float:
        """
        Computes the log likelihood for the current parameters.
        
        Returns:
          Log likelihood.
        """
        self._gmlm.setComputeGradient(False);
        self._gmlm.computeLogLikelihood(self._params);
        return self._results.getTrialLL().sum();

    def computeGradientNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the negative log likelihood for a set of vectorized parameters.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Flattened gradient in order: -dW, -dB, -Group[0].dV, -Group[0].dT[0], ..., -Group[0].dT[S_0-1], -Group[1].dV, -Group[1].dT[0], ...
        """
        self.devectorizeParameters(paramVector);
        self._gmlm.setComputeGradient(True);
        self._gmlm.computeLogLikelihood(self._params);
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
        """
        self.devectorizeParameters(paramVector);
        self._gmlm.setComputeGradient(True);
        self._gmlm.computeLogLikelihood(self._params);
        return (-self._results.getTrialLL().sum(), -self.vectorizeGradient());

    def computeNegativeLogLikelihood_vectorized(self, paramVector : np.ndarray) -> float:
        """
        Computes the negative log likelihood for a set of vectorized parameters.

        Args:
          paramVector: Flattened parameters in order: W, B, Group[0].V, Group[0].T[0], ..., Group[0].T[S_0-1], Group[1].V, Group[1].T[0], ...
          
        Returns:
          Negative log likelihood.
        """
        self.devectorizeParameters(paramVector);
        self._gmlm.setComputeGradient(False);
        self._gmlm.computeLogLikelihood(self._params);
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
