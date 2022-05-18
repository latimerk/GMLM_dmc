/*
 * kcGLM_dataStructures.hpp
 * Holds all the data - parameters, results, regressors, computation space
 * for a GLM (on one GPU).
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
#ifndef GLM_GLM_DATASTRUCTURES_H
#define GLM_GLM_DATASTRUCTURES_H

#include "kcBase.hpp"
#include "kcGLM.hpp"

namespace kCUDA {
    
/*The main datastructures for the GLM on the GPU
 *      parameters
 *      results
 *      dataset (all data + pre-allocated compute space)
 *    
 *      I make overly generous use of friend classes to streamline some computation code.
 */
template <class FPTYPE> class GPUGLM_parameters_GPU;
template <class FPTYPE> class GPUGLM_results_GPU;
template <class FPTYPE> class GPUGLM_dataset_GPU;

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * PARAMETERS on GPU 
 */

template <class FPTYPE>
class GPUGLM_parameters_GPU : GPUGL_base {
private:
    friend class GPUGLM_computeBlock<FPTYPE>; 
    friend class GPUGLM_results_GPU<FPTYPE>;
    
    // parameters
    GPUData<FPTYPE> * K; //vector size dim_K
    
    GPUData<FPTYPE> * trial_weights;                  // can be NULL or trial_weights_temp (NULL means all weights == 1)
    GPUData<FPTYPE> * trial_weights_0;                // is (dim_M x 0) 
    GPUData<FPTYPE> * trial_weights_temp;             // vector size dim_M
    
    GPUData<unsigned int> * trial_included;           // can be NULL or trial_included_temp (NULL means not doing a sparse run)
    GPUData<unsigned int> * trial_included_temp;      // vector size trial_weights_nonzero_cnt (0 to max_trials)
                                                      // list of which trials are currently being computed (non-zero weights) - page-locked host + GPU memory 
    GPUData<unsigned int> * trial_included_0;         // is (dim_M x 0) 
    
    logLikeType logLikeSettings;
    GPUData<FPTYPE> * logLikeParams;
    
public:
    
    //Primary constructor takes in the GLM setup
    GPUGLM_parameters_GPU(const GPUGLM_structure_args<FPTYPE> * GLMstructure, const size_t dim_M_, const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGLM_parameters_GPU();
    
    //sends user-provided parameters to the GPU loading with the given streams
        //opts gives the user-provided trial weights. if opts is NULL, assumes all trial_weights are 1
    void copyToGPU(const GPUGLM_params<FPTYPE> * glm_params, GPUGLM_dataset_GPU<FPTYPE> * dataset,  const cudaStream_t stream, const GPUGLM_computeOptions<FPTYPE> * opts = NULL);


    //dimensions
    inline size_t dim_K() const { // number of covariates
        return K->getSize(0);
    }
    inline size_t dim_M() const { // max trials on this GPU
        return trial_weights_temp->getSize_max(0);
    }
    inline size_t getNumberOfNonzeroWeights() const {
        return trial_included->getSize(0);
    }
};

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * RESULTS on GPU 
 * 
 */
template <class FPTYPE>
class GPUGLM_results_GPU : GPUGL_base {
private:
    friend class GPUGLM_computeBlock<FPTYPE>; 
    
    //each trials's log like
    GPUData<FPTYPE> * trialLL;
    
    //derivative 
    GPUData<FPTYPE> * dK;
    
    //hessian
    GPUData<FPTYPE> * d2K;
    
public:
    
    //Primary constructor takes in the GLM setup
    GPUGLM_results_GPU(const GPUGLM_structure_args <FPTYPE> * GLMstructure, const size_t max_trials_, const int dev_, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGLM_results_GPU();
    
    //brings results from GPU to page-locked host memory
    void gatherResults(const GPUGLM_parameters_GPU<FPTYPE> * params, const GPUGLM_computeOptions<FPTYPE> * opts, const cudaStream_t stream, const cudaStream_t stream2);
    //adds results in page-locked host memory to user-supplied object for returning
    //  If reset is false, adds. If true, sets to 0 before adding.
    void addToHost(const GPUGLM_parameters_GPU<FPTYPE> * params, GPUGLM_results<FPTYPE>* results_dest, const GPUGLM_computeOptions<FPTYPE> * opts, const GPUGLM_dataset_GPU<FPTYPE> * dataset, const bool reset = false);

    //dimensions
    inline size_t dim_K() const { // number of covariates
        return dK->getSize(0);
    }
    inline size_t max_trials() const { // max trials
        return trialLL->getSize_max(0);
    }
};

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * DATASET on GPU 
 * 
 */
template <class FPTYPE>
class GPUGLM_dataset_GPU : GPUGL_base {
private:
    friend class GPUGLM_results_GPU<FPTYPE>; 
    friend class GPUGLM_parameters_GPU<FPTYPE>; 
    friend class GPUGLM_computeBlock<FPTYPE>; 

    FPTYPE log_dt; // log of bin size
    FPTYPE dt;
    
    size_t max_trial_length; // max trial length for pre-allocation
    
    size_t max_trials_for_sparse_run;
    
    std::vector<bool> isInDataset_trial;  // size max_trials - if each trial is in this block
    
    GPUData<size_t> * dim_N = NULL;  // trial lengths (size dim_M x 1)
    
    size_t dim_N_temp; // number of bins in this block for current sparse run (running only some trials)
    
    //spike counts (all trials)
    GPUData<FPTYPE> * Y = NULL; // size   dim_N_total x 1
    
    GPUData<FPTYPE> * normalizingConstants_trial = NULL; //normalizing constant for each trial
    
    //linear terms (split by neuron)
    GPUData<FPTYPE> * X = NULL;  // X is dim_N_total x dim_K
    
    GPUData<FPTYPE> * X_temp = NULL;  
    
    //indices
    GPUData<unsigned int> * ridx_t_all = NULL;    // idx of each trial into complete data arrays (like Y)
    
    GPUData<unsigned int> * ridx_st_sall = NULL;       // idx of each trial (indexed by current run on non-zero weighted trials) into sparse index
    GPUData<unsigned int> * ridx_sa_all = NULL;        // idx of sparse run indices into complete data arrays
    GPUData<unsigned int> * ridx_a_all = NULL;      // empty index to indicate not a sparse run
    GPUData<unsigned int> * ridx_a_all_c = NULL;        // current ridx_sa_all (is ridx_a_all or ridx_sa_all)
    
    GPUData<unsigned int> * ridx_t_all_c = NULL;
    
    GPUData<unsigned int> * id_t_trial = NULL;  // total trial number of each trial (total is across all GPUs, indexes into trial_weights and trialLL)   size is dim_M x 1
    
    GPUData<unsigned int> * id_a_trialM = NULL;  // local trial index of each observation  size is dim_N_total x 1
                        // the M says that these numbers are 0:(dim_M-1). This is a reminder that it's different numbers than id_t_trial (without the M)
    
    //compute space
    GPUData<FPTYPE> * lambda = NULL; // dim_N_total x dim_K
    
    GPUData<FPTYPE> *   LL = NULL; // size   dim_N_total x 1
    GPUData<FPTYPE> *  dLL = NULL; // size   dim_N_total x 1
    GPUData<FPTYPE> * d2LL = NULL; // size   dim_N_total x 1
    
public:
    
    //Primary constructor takes in all the block data and GLM setup
    GPUGLM_dataset_GPU(const GPUGLM_structure_args<FPTYPE> * GLMstructure, const GPUGLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_,  const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg_);
        
    //destructor
    ~GPUGLM_dataset_GPU();
    
    inline bool isSparseRun(const GPUGLM_parameters_GPU<FPTYPE> * params) {
        return params->getNumberOfNonzeroWeights() <= max_trials_for_sparse_run;
    }
    
    //dimensions
    inline size_t dim_K() const { // number of covariates
        return X->getSize(1);
    }
    inline size_t dim_M() const { // max trials
        return dim_N->getSize(0);
    }
    inline size_t dim_N_total() const { // total number of observatiobs
        return Y->getSize(0);
    }
    inline size_t max_trials() const {
        return isInDataset_trial.size();
    }
};

}; //namespace
#endif