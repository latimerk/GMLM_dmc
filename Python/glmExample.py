import numpy as np;
import pyGMLM;
import scipy.special as sp; 

X = np.asfortranarray(np.random.randn(100,5));
Y = np.asfortranarray(np.random.poisson(1.0, size=(100))).astype('double');
K = np.asfortranarray(np.random.randn(5));

tr = pyGMLM.kcGLM_trial(0, X, Y);
bl = pyGMLM.kcGLM_trialBlock(0);
bl.addTrial(tr);

kg = pyGMLM.kcGLM(5, pyGMLM.ll_poissExp, 1);
kg.addBlock(bl);
kg.toGPU();

eXK = np.exp(X @ K);

ff = kg.computeLogLikelihood(K);
aa_0 = -eXK + (X @ K) * Y - sp.loggamma(Y+1);
aa = np.sum(aa_0);
print(ff-aa)

gg = kg.computeLogLikelihood_grad(K);
bb = X.T @ (-eXK + Y);
print(gg-bb)

hh = kg.computeLogLikelihood_hess(K);
cc = X.T @ (-eXK[:,np.newaxis] * X); 
print(hh-cc)

kg.freeGPU();



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
