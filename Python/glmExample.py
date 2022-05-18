import numpy as np;
from pyGMLM import pyGMLMcuda;
import scipy.special as sp; 

X = np.asfortranarray(np.random.randn(100,5));
Y = np.asfortranarray(np.random.poisson(1.0, size=(100))).astype('double');
K = np.asfortranarray(np.random.randn(5));

tr = pyGMLMcuda.kcGLM_trial(0, X, Y);
bl = pyGMLMcuda.kcGLM_trialBlock(0);
bl.addTrial(tr);

kg = pyGMLMcuda.kcGLM(5, pyGMLMcuda.ll_poissExp, 1);
kg.addBlock(bl);
kg.toGPU();

eXK = np.exp(X @ K);

ff = kg.computeLogLikelihood(K);
aa_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa = np.sum(aa_0);
print("LL difference between Python and GPU: " + str(ff-aa))

gg = kg.computeLogLikelihood_grad(K);
bb = X.T @ (-eXK + Y);
print("dLL difference between Python and GPU: " + str(gg-bb))

hh = kg.computeLogLikelihood_hess(K);
cc = X.T @ (-eXK[:,np.newaxis] * X); 
print("d2LL difference between Python and GPU: " + str(hh-cc))

kg.freeGPU();

# running some checks about the likelihood functionality

X[:,0] = np.random.randn(100);
kg.toGPU();
ff2 = kg.computeLogLikelihood(K);
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
ff2 = kg.computeLogLikelihood(K);
eXK = np.exp(X @ K);
aa2_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa2 = np.sum(aa2_0);
print("o1")
print(ff2-aa2)
print(ff2-aa)
print(" ")
kg.freeGPU();
