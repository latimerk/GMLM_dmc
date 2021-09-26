# GMLM_dmc
Generalized multilinear model for dimensionality reduction of neural population spike trains.
While this code is intended for modeling responses during decision-making tasks, it could be used more broadly for tensor regression.
Currently, the code primarily supports Poisson spike count observations, but squared errors are also included.
The main GMLM class can work with simultaneously or independently recorded neurons. A future update will include optimizations specifically for population recordings.
MODEL FITTING REQUIRES NVIDIA GPUS.

The core of the code is a C++/CUDA library with optimized log likelihood and derivative computations.
This code requires a CUDA capable GPU (toolkit v11.3).
Currently, only a MATLAB interface is given, but it's possible I'll get around to adding a Python interface.
While I could get the model working in PyTorch or Tensorflow, running inference like HMC needed all the performance I could get. That's why it's in CUDA with a bunch of initial data structure setup.

MATLAB visualizations in the example GMLM use Tensor Toolbox (http://tensortoolbox.org/).
Tensor toolbox should be added to the MATLAB path.

An example dataset and code is given for a macaque performing delayed-match-to-category (DMC) task.

Code tested using:
CUDA 11.3
MATLAB R2020a
TensorToolbox 3.1

## Citation (preprint)
```
Latimer, K. W., & Freedman, D. J. (2021). Low-dimensional encoding of decisions in parietal cortex reflects long-term training history. bioRxiv.
```
