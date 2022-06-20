# GMLM_dmc
Generalized multilinear model for dimensionality reduction of neural population spike trains.
While this code is intended for modeling responses during decision-making tasks, it could be used more broadly for tensor regression.
Currently, the code primarily supports Poisson spike count observations, but squared errors are also included.
The main GMLM class can work with simultaneously or independently recorded neurons. The example scripts currently only show the setup for the individual recording configuration.
Model fitting runs best on GPUs, but an okay set of host code is now included.

The core of the code is a C++/CUDA library with optimized log likelihood and derivative computations.
This code requires a CUDA capable GPU (toolkit v11.3+).
Currently, only a MATLAB interface is given, but it's possible I'll get around to adding a Python interface.
While I could get the model working in PyTorch or Tensorflow, running inference like HMC needed all the performance I could get. That's why it's in CUDA with a bunch of initial data structure setup.

MATLAB visualizations in the example GMLM use Tensor Toolbox (http://tensortoolbox.org/).
Tensor toolbox should be added to the MATLAB path.

An example dataset and code is given for a macaque performing delayed-match-to-category (DMC) task.

Code tested using:
CUDA 11.3
MATLAB R2020a
TensorToolbox 3.1

# Python bindings (work in progress)

A basic library can be compiled using the cmake function and there are a couple Python scripts that setup the GMLM for GPUs.
There is a bit of demo code for building a GMLM and fitting it in **`gmlmExample.py`**.

The API requires **[pybind11](https://github.com/pybind/pybind11)**

To compile the library and run the example:
```console
user@DESKTOP:~/PROJECTHOME$ mkdir build
user@DESKTOP:~/PROJECTHOME/build$ cd build
user@DESKTOP:~/PROJECTHOME/build$ cmake ..
user@DESKTOP:~/PROJECTHOME/build$ make
user@DESKTOP:~/PROJECTHOME/Python$ cd ../Python
user@DESKTOP:~/PROJECTHOME/Python$ python gmlmExample.py
```

The Python code includes a CPU version with Numpy.
However, this still requires compiling the C++ library to make sure that the same code handles organizing the GMLM and trial structures - Sorry, this is an annoying and less flexible design choice on my part to not make everything in Python.
However, the library can be compiled without needing any cuda by passing in an option to cmake: <code>cmake -DWITH_GPU=Off ..</code>.

The library can be compiled to use single-precision data (double is default) using the <code>-DWITH_DOUBLE_PRECISION=Off</code> cmake option.
Given pybind11's limitations with templating, it was way easier to just require recompiling that supporting both simultaneously.
If there's any real demand to include both, I could add better support for both precisions.

## Citation (preprint)
```
Latimer, K. W., & Freedman, D. J. (2021). Low-dimensional encoding of decisions in parietal cortex reflects long-term training history. bioRxiv.
```
https://www.biorxiv.org/content/10.1101/2021.10.07.463576v1
