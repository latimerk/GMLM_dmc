# I have GLM code for the GPUs here with a trial structure system that aligns with the GMLM.
# I've written down some code to show how to setup and run the GLM, but for typical applications,
# standard GLM libraries would be a better choice.
import numpy as np;
import pyGMLM.pyGMLMcuda.kcGLM as glm;
from pyGMLM.pyGMLMcuda import ll_poiss_exp;
import scipy.special as sp; 

X = np.asfortranarray(np.random.randn(100,5));
Y = np.asfortranarray(np.random.poisson(1.0, size=(100))).astype('double');
K = np.asfortranarray(np.random.randn(5));

tr = glm.Trial(0, X, Y);
bl = glm.TrialBlock(0);
bl.add_trial(tr);

kg = glm.GLM(5, ll_poiss_exp, 1);
kg.add_block(bl);
kg.toGPU();

eXK = np.exp(X @ K);

ff = kg.compute_log_likelihood(K);
aa_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa = np.sum(aa_0);
print("LL difference between Python and GPU: " + str(ff-aa))

gg = kg.compute_log_likelihood_grad(K);
bb = X.T @ (-eXK + Y);
print("dLL difference between Python and GPU: " + str(gg-bb))

hh = kg.compute_log_likelihood_hess(K);
cc = X.T @ (-eXK[:,np.newaxis] * X); 
print("d2LL difference between Python and GPU: " + str(hh-cc))

kg.freeGPU();

# running some checks about the likelihood functionality

X[:,0] = np.random.randn(100);
kg.toGPU();
ff2 = kg.compute_log_likelihood(K);
eXK = np.exp(X @ K);
aa2_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa2 = np.sum(aa2_0);
print("o2")
print(ff2-aa2)
print(ff2-aa)
print(" ")
kg.freeGPU();

X = np.asfortranarray(np.random.randn(100,5));
kg.toGPU();
ff2 = kg.compute_log_likelihood(K);
eXK = np.exp(X @ K);
aa2_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa2 = np.sum(aa2_0);
print("o1")
print(ff2-aa2)
print(ff2-aa)
print(" ")
kg.freeGPU();
