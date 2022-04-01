/*
 * kcGLM.hpp
 * Main class for holding a GLM (across multiple GPUs).
 * The classes which the user provides to communicate with the GLM are provided here.
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2021 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#ifndef GLM_GLM_CLASS_H
#define GLM_GLM_CLASS_H

#include "kcShared.hpp"

namespace kCUDA { 
    
// GPUGLM_params
//  Holds the model parameters
template <class FPTYPE>
class GPUGLM_params {
public:
    GLData<FPTYPE> * K; // length dim_K (to make sure everything matches), should be a COLUMN vector
    
    virtual ~GPUGLM_params() {};
    inline size_t dim_K(std::shared_ptr<GPUGL_msg> msg) { // number of parameters
        size_t dim_K_c = K->getSize(0);
    }
};


// GPUGLM_results
//  Holds the model results
template <class FPTYPE>
class GPUGLM_results {
public:
    GLData<FPTYPE> * dK  = NULL;  // length dim_K, should be a COLUMN vector
    GLData<FPTYPE> * d2K = NULL; // size dim_K x dim_K
    
    GLData<FPTYPE> * trialLL = NULL; // length dim_M, column vector
    
    inline size_t dim_K(std::shared_ptr<GPUGL_msg> msg) { // number of parameters
        if(dK == NULL) {
            std::ostringstream output_stream;
            output_stream << "GPUGLM_results errors: no results initialized!\n";
            msg->callErrMsgTxt(output_stream);
        }

        size_t dim_K_c = dK->getSize(0);
        if(d2K != NULL && !(d2K->empty()) && (d2K->getSize(0) != dim_K_c || d2K->getSize(1) != dim_K_c)) {
            std::ostringstream output_stream;
            output_stream << "GPUGLM_results errors: inconsistent size of derivative and Hessian!\n";
            output_stream << "\tdK = (" << dK->getSize(0) << ", "   << dK->getSize(1)  << ")  " << dK->getData()  << "\t" << dK  << "\n";
            output_stream << "\td2K = (" << d2K->getSize(0) << ", " << d2K->getSize(1) << ")  " << d2K->getData() << "\t" << d2K << "\n";
            msg->callErrMsgTxt(output_stream);
        }
        return dim_K_c;
    }
    inline size_t dim_M() {
        if(trialLL == NULL) {
            return 0;
        }
        return trialLL->getSize(0);
    }
    
    virtual ~GPUGLM_results() {};
};

// options for what to compute in a log likelihood run
template <class FPTYPE>
class GPUGLM_computeOptions {
public:
    bool compute_dK;
    bool compute_d2K;
    bool compute_trialLL; // if you don't want to copy all the individual trial likelihoods back to host
    
    std::vector<FPTYPE> trial_weights; // length max_trials or EMPTY. Weighting for each trial (if 0, doesn't compute trial at all). Can be used for SGD
    
    virtual ~GPUGLM_computeOptions() {};
};

template <class FPTYPE>
class GPUGLM_trial_args {
    public:
        unsigned int trial_idx;  // a trial number index
        
        GLData<FPTYPE> * Y; // length dim_N, the spike counts per bin. Column vector
        GLData<FPTYPE> * X; // size dim_N x dim_K, the linear term of the neuron
        
        inline size_t dim_N(std::shared_ptr<GPUGL_msg> msg) {
            size_t dim_N_c = Y->getSize(0);
            if(X->getSize(0) != dim_N_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGLM_trial_args errors: inconsistent size of regressors and observations!";
                msg->callErrMsgTxt(output_stream);
            }
            return dim_N_c;
        }

        virtual ~GPUGLM_trial_args() {};
};

template <class FPTYPE>
class GPUGLM_GPU_block_args {
    public:
        int dev_num;                   //which GPU to use
        int max_trials_for_sparse_run; //max trials that can be run in a minibatch (for pre-allocating space. Can always run the full set)

        std::vector<GPUGLM_trial_args <FPTYPE> *> trials; // the trials for this block

        virtual ~GPUGLM_GPU_block_args() {};
};

template <class FPTYPE>
class GPUGLM_structure_args {
    public:
        size_t dim_K; // size of the linear term in the model
        
        logLikeType logLikeSettings = ll_poissExp;
        std::vector<FPTYPE> logLikeParams;

        FPTYPE binSize;    // bin width (in seconds) if you want the baseline rate parameters to be in terms of sp/s (totally redundant with the exp nonlinear, but fine)

        virtual ~GPUGLM_structure_args() {};
};

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* class GPUGLM <FPTYPE = double, float>
*    Holds a complete GLM
*       The GLM can be split across multiple GPUs as a set of GPUGLM_block objs
*
*    Variables: Observations
*
*        Blocks [dim_B x 1] (GPUGLM_block<FPTYPE>*)
*            Each GLM block (each should be on a different GPU). 
*/  

template <class FPTYPE> class GPUGLM_computeBlock;

template <class FPTYPE> 
class GPUGLM {
    private: 
        std::ostringstream output_stream;
        std::shared_ptr<GPUGL_msg> msg;
        
        std::vector<GPUGLM_computeBlock<FPTYPE> *> gpu_blocks;
        
        void syncStreams();
        unsigned int max_trials;
        size_t dim_K_; 

    public:
        /* Primary constructor.
         * Takes in the set of args defined by GPUGLM_regressor_args and loads everything to the GPU(s).
         *
         * The full GLM structure is defined in GLMstructure, and the device usage and trial data are in the vector of blocks
         */
        GPUGLM(const GPUGLM_structure_args <FPTYPE> * GLMstructure, const std::vector<GPUGLM_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_);
        
        /* Deconstructor
         * Clears all GPU and host memory.
         */
        ~GPUGLM();
        
        /* METHOD computeLogLikelihood
         *  Runs the complete log likelihood computation for a set of parameters.
         *  Does not copy the answer.
         *
         *  inputs:
         *      params (GPUGLM_params<FPTYPE> *): the complete set of params for the model
         *      opts   (GPUGLM_computeOptions *): the options for what to compute (gradient & hessian)
         *
         *  outputs:
         *      results (GPUGLM_results<FPTYPE>*)
         *          Any values in the results obj not selected by opts may be 0's or NaN's.
         */
        void computeLogLikelihood(const GPUGLM_params<FPTYPE> * params, const GPUGLM_computeOptions<FPTYPE> * opts, GPUGLM_results<FPTYPE>* results);
                /* GPUGLM calls each GPU
                 *     GPUGLM_GPUportion holds a set of trials, and set of compute blocks 
                 *         The GPUGLM_computeBlocks iterate through the whole set of GPUGLM_trials
                 *         Then the results from the GPUGLM_computeBlocks are summed into the  GPUGLM_GPUportion's results
                 *         
                 *         GPUGLM_GPUportion's are summed on host and returned
                 *
                 */

        size_t dim_M() const {// number of trials
            return max_trials;
        }
        size_t dim_K() { // number of parameters
            return dim_K_;
        }
        
};

}; // end namespace
#endif