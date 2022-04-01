#include "kcGMLMPython_glm.hpp"

namespace kCUDA { 
template <class FPTYPE>
kcGLM_python<FPTYPE>::kcGLM_python(unsigned int numCovariates, logLikeType ll_type, FPTYPE binSize) {
    structure = new GPUGLM_structure_args<FPTYPE>();
    structure->dim_K = numCovariates;
    structure->logLikeSettings = ll_type;
    structure->binSize = binSize;
    kcglm = NULL;
    msg = std::make_shared<GPUGL_msg_python>();

    params  = new GPUGLM_params_python<FPTYPE>(numCovariates);
    results = new GPUGLM_results_python<FPTYPE>(numCovariates);
    opts    = new GPUGLM_computeOptions_python<FPTYPE>();
}
template <class FPTYPE>
kcGLM_python<FPTYPE>::~kcGLM_python() {
    //py::print("kcGLM_python destructor");
    freeGPU();
    delete structure;
    delete results;
    delete opts;
    delete params;
    blocks_shared.clear(); // shouldn't be needed, but whatever
}

template <class FPTYPE>
bool kcGLM_python<FPTYPE>::isOnGPU() {
    return kcglm != NULL;
}
template <class FPTYPE>
void kcGLM_python<FPTYPE>::freeGPU() {
    if(isOnGPU()) {
        delete kcglm;
        kcglm = NULL;
    }
}
template <class FPTYPE>
int kcGLM_python<FPTYPE>::addBlock(std::shared_ptr<kcGLM_trialBlock<FPTYPE>> block) {
    freeGPU();
    blocks_shared.push_back(block);
    blocks.push_back(block.get()); // gross to store both the shared & raw pointer, but my simple API needed the raw pointers and I don't want to change that 
    return blocks.size()-1;
}

template <class FPTYPE>
void kcGLM_python<FPTYPE>::toGPU() {
    if(blocks.empty()) {
        throw std::length_error("No trial blocks given!");
    }
    for(auto bb : blocks) {
        if(bb->trials.empty()) {
            throw std::length_error("Empty trial block found!");
        }
    }

    freeGPU();
    kcglm = new GPUGLM<FPTYPE>(structure, blocks, msg);

    //setup result
    results->setupResults(kcglm->dim_M());
}

// compute log likelihood: returns trial-wise LL
template <class FPTYPE>
py::array_t<FPTYPE, py::array::f_style> kcGLM_python<FPTYPE>::computeLogLikelihood(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K) {
    //setup compute options
    opts->reset();
    opts->compute_trialLL = true;
    
    runComputation(K);

    //return the numpy array
    return results->getTrialLL();
}

// compute log likelihood gradient
template <class FPTYPE>
py::array_t<FPTYPE, py::array::f_style> kcGLM_python<FPTYPE>::computeLogLikelihood_grad(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K) {
    //setup compute options
    opts->reset();
    opts->compute_dK      = true;

    runComputation(K);

    //return the numpy array
    return results->getDK();
}

// compute log likelihood hessian
template <class FPTYPE>
py::array_t<FPTYPE, py::array::f_style> kcGLM_python<FPTYPE>::computeLogLikelihood_hess(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K) {
    //setup compute options
    opts->reset();
    opts->compute_d2K     = true;

    runComputation(K);

    //return the numpy array
    return results->getD2K();
}
        



template <class FPTYPE>
void kcGLM_python<FPTYPE>::runComputation(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K) {
    //assume options are already setup
    if(!isOnGPU()) {
        throw py::buffer_error("kcGLM not on GPU!");
    }

    //setup parameters
    params->setK(K);

    //call log likelihood function
    kcglm->computeLogLikelihood(params, opts, results);
}


//explicitly create classes for single and double precision floating point for library
template class kcGLM_python<float>;
template class kcGLM_python<double>;
}; //namespace
