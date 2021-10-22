/*
 * kcGMLMPop.hpp
 * Main class for holding a GMLMPop (across multiple GPUs).
 * The classes which the user provides to communicate with the GMLMPop are provided here.
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   Population GMLM and indidivual-cell-recordings GMLM are in separate CUDA files.
 *   I decided to do this in order to make the different optimization requirements a little more clean.
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
#ifndef GMLMPop_GMLMPop_CLASS_H
#define GMLMPop_GMLMPop_CLASS_H

#include <vector>
#include <sstream>
#include <memory> // for std::shared_ptr on linux (compiles fine on Windows without this)
#include "kcShared.hpp"

namespace kCUDA { 

 
/* CLASSES GPUGMLMPop_params, GPUGMLMPop_group_params
 *  The argument structures for the GMLMPop parameters
 *
 *  GPUGMLMPop_params
 *      W [dim_P x 1] (FPTYPE): baseline rate terms
 *      Groups [dim_J] (GPUGMLMPop_group_params): each group's parameters
 *  
 *  GPUGMLMPop_group_params
 *      dim_R (scalar) [size_t]: rank of the tensor group
 *      dim_T [dim_S x 1]: sizes of each component
 *
 *      V [dim_P x dim_R] (FPTYPE): the neuron weighting components
 *      T [dim_S x 1]
 *          T[ss] [dim_T[ss] x dim_R]: the components for dim ss
 */
template <class FPTYPE>
class GPUGMLMPop_group_params {
public:
    std::vector<GLData<FPTYPE> *> T; //length dim_S: each T[ss] is dim_T[ss] x dim_R
    GLData<FPTYPE> * V; // dim_P x dim_R
    
    virtual ~GPUGMLMPop_group_params() {};
    
    inline size_t dim_R(std::shared_ptr<GPUGL_msg> msg) const {
        size_t dim_R_c = V->getSize(1);
        for(int ss = 0; ss < T.size(); ss++) {
            if(T[ss]->getSize(1) != dim_R_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLMPop_group_params errors: inconsistent ranks!";
                msg->callErrMsgTxt(output_stream);
            }
        }
        return dim_R_c;
    }
    inline size_t dim_S() const {
        return T.size();
    }
    inline size_t dim_T(int dim, std::shared_ptr<GPUGL_msg> msg) const  {
        if(dim < 0 || dim > dim_S()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_group_params errors: invalid dim!";
            msg->callErrMsgTxt(output_stream);
        }
        return T[dim]->getSize(0);
    }
    inline size_t dim_P() const {
        return V->getSize(0);
    }
};

template <class FPTYPE>
class GPUGMLMPop_params {
public:
    GLData<FPTYPE> * W; // length dim_P
    GLData<FPTYPE> * B; // dim_B x dim_P (lead dimension is ld_B)
    
    std::vector<GPUGMLMPop_group_params<FPTYPE> *> Groups; //length dim_J
    
    virtual ~GPUGMLMPop_params() {};
    
    inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const {
        if(W->size() != B->getSize(1) && B->size() > 0) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_params errors: inconsistent dim_P between W and B!";
            msg->callErrMsgTxt(output_stream);
        }
        return W->size();
    }
    inline size_t dim_B() const {
        return B->getSize(0);
    }
};

/* CLASSES GPUGMLMPop_computeOptions, GPUGMLMPop_group_computeOptions
 *  The option structures for what to compute.
 *  Note that if the derivative computation is false, the corresponding second derivative will never be computed.
 *
 *  GPUGMLMPop_params
 *      compute_dW [scalar]: options to compute gradient of W
 *      compute_trialLL [scalar]: option to compute the trial-specific log likelihoods
 *      Groups [dim_J] (GPUGMLMPop_group_computeOptions): each group's options
 *  
 *  GPUGMLMPop_group_computeOptions
 *      compute_dV [scalar]: options to compute gradient
 *      compute_dT [dim_S x 1]:  options to compute gradient of each T
 */
class GPUGMLMPop_group_computeOptions {
public:
    bool compute_dV;
    std::vector<bool> compute_dT;  //length dim_S
    
    virtual ~GPUGMLMPop_group_computeOptions() {};
};

template <class FPTYPE>
class GPUGMLMPop_computeOptions {
public:
    bool compute_dB;
    bool compute_dW;
    bool compute_trialLL; // if you don't want to copy all the individual trial likelihoods back to host
    
    GLData<FPTYPE> * trial_weights; // EMPTY or (max_trials x 1) or (max_trials x dim_P). Weighting for each trial (if 0, doesn't compute trial at all). Can be used for SGD
                                    // if EMPTY: all trial weights 1
                                    // if (max_trials x 1): weights for each trial
                                    // if (max_trials x dim_P): weights for each neuron in each trial
    
    std::vector<GPUGMLMPop_group_computeOptions *> Groups; //length dim_J
    
    virtual ~GPUGMLMPop_computeOptions() {};
};


/* CLASSES GPUGMLMPop_results, GPUGMLMPop_group_results
 *  The results structures for the GMLMPop (not similarity to the arg struct).
 *  WARNING: if any field is not specified by the 'opts' argument to the log likelihood call,
 *           the results may not be valid!
 *
 *  GPUGMLMPop_params
 *      trialLL [dim_M x dim_P] (FPTYPE): trial-wise log likelihood per neuron
 *      dW      [dim_P x 1] (FPTYPE): baseline rate terms
 *      dB      [dim_B x dim_P] (FPTYPE): linear terms (neuron specific)
 *      Groups  [dim_J] (GPUGMLMPop_group_params): each group's parameters
 *  
 *  GPUGMLMPop_group_params
 *      dim_R (scalar) [size_t]: rank of the tensor group
 *      dim_T [dim_S x 1]: sizes of each component
 *
 *      dV  [dim_P       x dim_R] (FPTYPE): the neuron weighting components
 *      dT [dim_S x 1]
 *          dT[ss]  [dim_T[ss] x dim_R]: the components for dim ss
 */
template <class FPTYPE>
class GPUGMLMPop_group_results {
public:
    std::vector<GLData<FPTYPE> *>  dT;  //length dim_S: each T[ss] is dim_T[ss] x dim_R 
    GLData<FPTYPE>  *  dV; //size dim_P x dim_R
    
    virtual ~GPUGMLMPop_group_results() {};
    
    inline size_t dim_R(std::shared_ptr<GPUGL_msg> msg) const {
        size_t dim_R = dV->getSize(1);
        for(int ss = 0; ss < dT.size(); ss++) {
            if(dim_R == 0) {
                dim_R = dT[ss]->getSize(1);
            }
            if(!(dT[ss]->empty()) || dT[ss]->getSize(1) != dim_R) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLMPop_group_results errors: inconsistent ranks!";
                msg->callErrMsgTxt(output_stream);
            }
        }
        return dim_R;
    }
    inline size_t dim_S() const {
        return dT.size();
    }
    inline size_t dim_T(int dim, std::shared_ptr<GPUGL_msg> msg) const {
        if(dim < 0 || dim > dim_S()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_group_results errors: invalid dim!";
            msg->callErrMsgTxt(output_stream);
        }
        return dT[dim]->getSize(0);
    }
    inline size_t dim_P() const {
        return dV->getSize(0);
    }
};


template <class FPTYPE>
class GPUGMLMPop_results {
public:
    GLData<FPTYPE> * dW; //length dim_P
    GLData<FPTYPE> * dB; //size dim_P x dim_K (lead dim is ld_K)
    
    GLData<FPTYPE> * trialLL; //size dim_M x dim_P
    
    std::vector<GPUGMLMPop_group_results<FPTYPE> *> Groups; //length dim_J
    virtual ~GPUGMLMPop_results() {};
    
    inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const  {
        size_t dim_P_c = 0;
        if(!trialLL->empty()) {
            dim_P_c = trialLL->getSize(1);
        }
        else if(!dW->empty()) {
            dim_P_c = dW->getNumElements();
        }
        else if(!dB->empty()) {
            dim_P_c = dB->getSize(1);
        }
        
        if(dW->getNumElements() != dim_P_c && !dW->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for dW!";
            msg->callErrMsgTxt(output_stream);
        }
        if(trialLL->getSize(1) != dim_P_c && !trialLL->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for trialLL!";
            msg->callErrMsgTxt(output_stream);
        }
        if(dB->getSize(1) != dim_P_c && !dB->empty()) {
            std::ostringstream output_stream;
            output_stream << "GPUGMLMPop_results errors: inconsistent dim_P for dB!";
            msg->callErrMsgTxt(output_stream);
        }
        return dim_P_c;
    }
    inline size_t dim_B() const {
        return dB->getSize(0);
    }
    inline size_t dim_M() const {
        return trialLL->getSize(0);
    }
};



/* CLASSES GPUGMLMPop_structure_args, GPUGMLMPop_structure_Group_args, GPUGMLMPop_GPU_block_args
 *  complete specification of inputs to create a GMLMPop: contains all dimension information and regressors
 *  
 *  Trials are placed into a vector of GPUGMLMPop_GPU_block_args for each GPU.
 *  A single GPUGMLMPop_structure_args is created which described how each trial should look.
 *
 *  Callers to this GMLMPop should construct one of these can make sure the dimensions and entries make sense
 *  GPUGMLMPop should do a deep copy of everything (all the big elements are copied to the GPUs).
 *
 *  Note: we require some exposed pointers just to make sure that everything plays nicely with CUDA.
 *        If the pointers and dimensions aren't correct, this can lead to seg faults
 */
template <class FPTYPE>
class GPUGMLMPop_trial_Group_args {
    public:
        //all vectors are length dim_S
        std::vector<GLData<FPTYPE> *> X; //each X is size dim_N x dim_F[dd] x (dim_A or 1) (or empty if the universal regressors are used)
        
        std::vector<GLData<int> *> iX; // for each non-empty X, iX is a dim_N x dim_A matrix of indices into X_shared (the universal regressors)

        virtual ~GPUGMLMPop_trial_Group_args() {};
};

template <class FPTYPE>
class GPUGMLMPop_trial_args {
    public:
        std::vector<GPUGMLMPop_trial_Group_args<FPTYPE> *> Groups; // the regressor tensor groups

        unsigned int trial_idx;  // a trial number index
        
        GLData<FPTYPE> * Y; // size dim_N x dim_P, the spike counts per bin for each neuron
        
        GLData<FPTYPE> * X_lin; // size dim_N x dim_B x dim_P, the linear term of the neuron

        
        virtual ~GPUGMLMPop_trial_args() {};
        
        inline size_t dim_N() const {
            return Y->getSize(0);
        }
        inline size_t dim_P(std::shared_ptr<GPUGL_msg> msg) const {
            size_t dim_P_c = Y->getSize(1);
            if(X_lin->getSize(2) != 1 && X_lin->getSize(2) != dim_P_c) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLMPop_trial_args errors: inconsistent dim_P!";
                msg->callErrMsgTxt(output_stream);
            }
            return dim_P_c;
        }
        
};


template <class FPTYPE>
class GPUGMLMPop_GPU_block_args {
    public:
        int dev_num;            //which GPU to use
        int max_trials_for_sparse_run; //max trials that can be run in a minibatch (for pre-allocating space. Can always run the full set)

        std::vector<GPUGMLMPop_trial_args <FPTYPE> *> trials; // the trials for this block

        virtual ~GPUGMLMPop_GPU_block_args() {};
};

template <class FPTYPE>
class GPUGMLMPop_structure_Group_args {
    private:
    
        inline size_t dim_D() const {
            size_t max_factor = X_shared.size();
            for(int dd = 0; dd < max_factor; dd++) {
                bool found = false;
                for(int ss = 0; ss < factor_idx.size() && !found; ss++) {
                    if(factor_idx[ss] == dd) {
                        found = true;
                    }
                    else if(factor_idx[ss] >= max_factor) {
                        return 0;
                    }
                }
                if(!found) {
                    return 0;
                }
            }
            
            return max_factor;
        }
    public:
        //all vectors are length dim_S
        std::vector<size_t  > dim_T;             // outer tensor dimension for each component
        std::vector<GLData<FPTYPE> *> X_shared; // dim_D, any univerisal regressors (size dim_X_shared[dd] x dim_T[ss]): empty if no shared indices used

        std::vector<unsigned int> factor_idx; // size of dim_T, factors for each dim (how the coeffs are divided into partial CP decompositions)
                                     //(MUST BE 0 to max factor - 1, where max factor gives dim_D!)
        
        size_t dim_A;                 // number of events for this tensor Group
        size_t dim_R_max;             // max rank of group (for pre-allocation)

        virtual ~GPUGMLMPop_structure_Group_args() {};
        
        inline size_t dim_S() const {
            return dim_T.size();
        }
        inline size_t dim_D(std::shared_ptr<GPUGL_msg> msg) const {
            size_t max_factor = X_shared.size();
            if(dim_T.size() != factor_idx.size()) {
                std::ostringstream output_stream;
                output_stream << "GPUGMLMPop_structure_Group_args errors: inconsistent dim_S!";
                msg->callErrMsgTxt(output_stream);
            }
            //search for each factor
            for(int dd = 0; dd < max_factor; dd++) {
                bool found = false;
                for(int ss = 0; ss < factor_idx.size() && !found; ss++) {
                    if(factor_idx[ss] == dd) {
                        found = true;
                    }
                    else if(factor_idx[ss] >= max_factor) {
                        std::ostringstream output_stream;
                        output_stream << "GPUGMLMPop_structure_Group_args errors: invalid CP decomp factorization!";
                        msg->callErrMsgTxt(output_stream);
                    }
                }
                if(!found) {
                    std::ostringstream output_stream;
                    output_stream << "GPUGMLMPop_structure_Group_args errors: invalid CP decomp factorization!";
                    msg->callErrMsgTxt(output_stream);
                }
            }
            
            return max_factor;
        }
        
        //validates some of the basic GMLMPop structure of the regressors for a trial
        bool validateTrialStructure(const GPUGMLMPop_trial_Group_args <FPTYPE> * trial) const {
            
            //checks if local vectors are all correct length
            if(dim_D() == 0) {
                return false;
            }
            
            //checks if trial vectors are all correct length
            if(trial->X.size() != dim_D()) {
                return false;
            }
            if(trial->iX.size() != dim_D()) {
                return false;
            }
            
            //checks if universal regressor indices match
            for(int ff = 0; ff < dim_D(); ff++) {
                if(X_shared[ff]->empty()) {
                    if(trial->X[ff]->empty() || !(trial->iX[ff]->empty())) {
                        return false; //expected a trial-wise set of regressors, not iX
                    }
                }
                else {
                    if(!(trial->X[ff]->empty()) || trial->iX[ff]->empty()) {
                        return false; //expected iX to hold a set of indices into X_shared
                    }
                }
            }
            
            return true; //passed basic tests (seg faults may still exist if the pointers are bad!)
        }
};

template <class FPTYPE>
class GPUGMLMPop_structure_args {
    public:
        std::vector<GPUGMLMPop_structure_Group_args<FPTYPE> *> Groups; // the coefficient tensor groups, length dim_J

        size_t dim_B; // size of the linear term in the model
        size_t dim_P; // number of neurons to expect
        
        logLikeType logLikeSettings = ll_poissExp;
        std::vector<FPTYPE> logLikeParams;

        FPTYPE binSize;    // bin width (in seconds) if you want the baseline rate parameters to be in terms of sp/s (totally redundant with the exp nonlinear, but fine)
        
        virtual ~GPUGMLMPop_structure_args() {};
    
        //validates some of the basic GMLMPop structure of the regressors for a trial
        // but not thoroughly here
        bool validateTrialStructure(const GPUGMLMPop_trial_args <FPTYPE> * trial) const { 
            
            if(trial->Groups.size() != Groups.size()) {
                return false; //invalid number of tensor groups
            }
            
            for(int jj = 0; jj < Groups.size(); jj++) {
                if(!Groups[jj]->validateTrialStructure(trial->Groups[jj])) {
                    return false; // group structure was invalid 
                }
            }
            
            return true; //passed basic tests (seg faults may still exist if the pointers are bad!)
        }
        bool validateTrialStructure(const GPUGMLMPop_GPU_block_args <FPTYPE> * block) const {
            for(int mm = 0; mm < block->trials.size(); mm++) {
                //checks each trial on a block
                if(!validateTrialStructure(block->trials[mm])) {
                    return false;
                }
            }
            return true;
        }                           
        bool validateTrialStructure(const std::vector<GPUGMLMPop_GPU_block_args <FPTYPE> *> blocks) const {
            for(int bb = 0; bb < blocks.size(); bb++) {
                //checks each block
                if(!validateTrialStructure(blocks[bb])) {
                    return false;
                }
            }
            return true;
        }
};



/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* class GPUGMLMPop <FPTYPE = double, float>
*    Holds a complete GMLMPop
*       The GMLMPop can be split across multiple GPUs as a set of GPUGMLMPop_block objs
*
*    Variables: Observations
*
*        Blocks [dim_B x 1] (GPUGMLMPop_block<FPTYPE>*)
*            Each GMLMPop block (each should be on a different GPU). 
*/  

template <class FPTYPE> class GPUGMLMPop_computeBlock;

template <class FPTYPE> 
class GPUGMLMPop {
    private: 
        std::ostringstream output_stream;
        std::shared_ptr<GPUGL_msg> msg;
        
        std::vector<GPUGMLMPop_computeBlock<FPTYPE> *> gpu_blocks;
        
        void syncStreams();
        
    public:
        /* Primary constructor.
         * Takes in the set of args defined by GPUGMLMPop_regressor_args and loads everything to the GPU(s).
         *
         * The full GMLMPop structure is defined in GMLMPopstructure, and the device usage and trial data are in the vector of blocks
         */
        GPUGMLMPop(const GPUGMLMPop_structure_args <FPTYPE> * GMLMPopstructure, const std::vector<GPUGMLMPop_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_);
        
        /* Deconstructor
         * Clears all GPU and host memory.
         */
        ~GPUGMLMPop();
        
        /* METHOD computeLogLikelihood
         *  Runs the complete log likelihood computation for a set of parameters.
         *  Does not copy the answer.
         *
         *  inputs:
         *      params (GPUGMLMPop_params<FPTYPE> *): the complete set of params for the model
         *      opts   (GPUGMLMPop_computeOptions *): the options for what to compute (i.e., which derivatives)
         *
         *  outputs:
         *      results (GPUGMLMPop_results<FPTYPE>*)
         *          Any values in the results obj not selected by opts may be 0's or NaN's.
         */
        void computeLogLikelihood(const GPUGMLMPop_params<FPTYPE> * params, const GPUGMLMPop_computeOptions<FPTYPE> * opts, GPUGMLMPop_results<FPTYPE>* results);
                /* GPUGMLMPop calls each GPU
                 *     GPUGMLMPop_GPUportion holds a set of trials, and set of compute blocks 
                 *         The GPUGMLMPop_computeBlocks iterate through the whole set of GPUGMLMPop_trials
                 *         Then the results from the GPUGMLMPop_computeBlocks are summed into the  GPUGMLMPop_GPUportion's results
                 *         
                 *         GPUGMLMPop_GPUportion's are summed on host and returned
                 *
                 */
        
};


}; //namespace
#endif