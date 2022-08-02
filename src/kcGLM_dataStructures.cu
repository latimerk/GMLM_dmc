/*
 * kcGLM_dataStructures.cu
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
#include "kcGLM_dataStructures.hpp"
//#include <tgmath.h>

namespace kCUDA {

//============================================================================================================================
//Parameter class
        //constructor
template <class FPTYPE>
GPUGLM_parameters_GPU<FPTYPE>::GPUGLM_parameters_GPU(const GPUGLM_structure_args <FPTYPE> * GLMstructure, const size_t dim_M_, const int dev_, std::shared_ptr<GPUGL_msg> msg_) {
    dev = dev_;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0; // default stream

    //setup any log like settings
    logLikeSettings = GLMstructure->logLikeSettings;
    if(GLMstructure->logLikeParams.size() > 0) {
        logLikeParams = new GPUData<FPTYPE>(ce, GPUData_HOST_STANDARD, stream, GLMstructure->logLikeParams.size());
        checkCudaErrors(ce,  "GPUGLM_parameters_GPU errors: could not allocate space for logLikeParams!" );
        for(unsigned int ii = 0; ii < GLMstructure->logLikeParams.size(); ii++) {
            (*logLikeParams)[ii] = GLMstructure->logLikeParams[ii];
        }
        ce = logLikeParams->copyHostToGPU(stream);
        checkCudaErrors(ce,  "GPUGLM_parameters_GPU errors: could not copy logLikeParams to GPU!" );
    }
    else {
        logLikeParams = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, 0);
        checkCudaErrors(ce,  "GPUGLM_parameters_GPU errors: could not allocate space for logLikeParams!" );
    }

    //allocate GPU space for trial weights

    trial_weights_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_);
    checkCudaErrors(ce, "GPUGLM_parameters_GPU errors: could not allocate space for trial_weights_temp!" );
    trial_included_temp = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_);
    checkCudaErrors(ce, "GPUGLM_parameters_GPU errors: could not allocate space for trial_included_temp!" );


    trial_weights_0 = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGLM_parameters_GPU errors: could not allocate space for trial_weights_0!" );
    trial_weights = trial_weights_0;

    trial_included_0 = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGLM_parameters_GPU errors: could not allocate space for trial_included_0!" );
    trial_included = trial_included_0;

    //allocate GPU space
    K = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_parameters_GPU errors: could not allocate space for K!" );
}

//destructor
template <class FPTYPE>
GPUGLM_parameters_GPU<FPTYPE>::~GPUGLM_parameters_GPU() {
    switchToDevice();
    cudaSafeFree(    trial_weights_temp      , "GPUGLM_parameters_GPU errors: could not free trial_weights_temp");
    cudaSafeFree(    trial_included_temp     , "GPUGLM_parameters_GPU errors: could not free trial_included_temp");
    cudaSafeFree(    trial_weights_0      , "GPUGLM_parameters_GPU errors: could not free trial_weights_0");
    cudaSafeFree(    trial_included_0     , "GPUGLM_parameters_GPU errors: could not free trial_included_0");

    cudaSafeFree(logLikeParams, "GPUGLM_parameters_GPU errors: could not free logLikeParams." );

    cudaSafeFree(K, "GPUGLM_parameters_GPU errors: could not free K");
}

/* kernel for setting up sparse run indices
*   One thread per trial being run. Sets up a map between the current indices (0:dim_N_temp-1) to the full indices (0:dim_N-1)
* 
*/
__global__ void kernel_ParamsSparseRunSetup_GLM(GPUData_kernel<unsigned int> ridx_sa_all,
                                 const GPUData_kernel<unsigned int> trial_included, 
                                 const GPUData_kernel<unsigned int> ridx_st_sall, 
                                 const GPUData_kernel<unsigned int> ridx_t_all,
                                 const GPUData_kernel<size_t> dim_N) {
    unsigned int tr = blockIdx.x * blockDim.x + threadIdx.x;
    if(tr < trial_included.x) {
        unsigned int mm = trial_included[tr];
        unsigned int start_all = ridx_t_all[mm];
        unsigned int start_sp  = ridx_st_sall[tr];
        for(unsigned int nn = 0; nn < dim_N[mm]; nn++) {
            ridx_sa_all[nn + start_sp] = start_all + nn;
        }
    }
}

//copy all parameters to GPU
template <class FPTYPE>
void GPUGLM_parameters_GPU<FPTYPE>::copyToGPU(const GPUGLM_params<FPTYPE> * glm_params, GPUGLM_dataset_GPU<FPTYPE> * dataset, const cudaStream_t stream, const GPUGLM_computeOptions<FPTYPE> * opts) {
    switchToDevice();

    //copies trial weights if given
    if(opts != NULL && !opts->trial_weights.empty() && opts->trial_weights.size() != dataset->max_trials()) {
        output_stream << "GPUGLM_parameters_GPU errors: input does not have correct number of trial weights" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    if(opts != NULL && !opts->trial_weights.empty()) {
        size_t trial_weights_nonzero_cnt_c = 0;
        dataset->dim_N_temp = 0;

        //gets weights for each trial on this GPU block
        for(unsigned int mm = 0; mm < dim_M(); mm++) {
            (*trial_weights_temp)[mm] = opts->trial_weights[(*(dataset->id_t_trial))[mm]];
    
            //if trial is included
            if((*trial_weights_temp)[mm] != 0) {
                (*trial_included_temp)[trial_weights_nonzero_cnt_c] = mm;
                (*dataset->ridx_st_sall)[trial_weights_nonzero_cnt_c] = dataset->dim_N_temp;
                dataset->dim_N_temp += (*(dataset->dim_N))[mm];
                trial_weights_nonzero_cnt_c++;
            }
        }

        checkCudaErrors(trial_included_temp->resize(stream, trial_weights_nonzero_cnt_c), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
        trial_included = trial_included_temp;

        if(trial_weights_nonzero_cnt_c != 0) {
            // copies weights to GPU
            checkCudaErrors(trial_weights_temp->copyHostToGPU(stream), "GPUGLM_parameters_GPU errors: could not copy trial_weights_temp to device!");
            trial_weights = trial_weights_temp;
        }
        else {
            // if there are no trials, might as well not copy anything more
            return;
        }


        //copy list of trials with nonzero weights to host only if the number is small enough for a sparse run
        if(trial_weights_nonzero_cnt_c <= dataset->max_trials_for_sparse_run) {
            //sets some sizes
            checkCudaErrors(dataset->d2LL->resize(stream, dataset->dim_N_temp), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->dLL->resize(stream, dataset->dim_N_temp), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->LL->resize(stream, dataset->dim_N_temp), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_temp, -1), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->resize(stream, trial_weights_nonzero_cnt_c), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_st_sall->resize(stream, trial_weights_nonzero_cnt_c), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_sa_all->resize(stream, dataset->dim_N_temp), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->copyHostToGPU(stream), "GPUGLM_parameters_GPU errors: could not copy trial_included_temp to device!");
            checkCudaErrors(dataset->ridx_st_sall->copyHostToGPU(stream), "GPUGLM_parameters_GPU errors: could not copy ridx_st_sall to device!");
            
            trial_included = trial_included_temp;
            dataset->ridx_a_all_c = dataset->ridx_sa_all;
            dataset->ridx_t_all_c = dataset->ridx_st_sall;

            //setup a special index variable
            dim3 block_size;
            block_size.x = min(static_cast<size_t>(1024), trial_weights_nonzero_cnt_c);
            dim3 grid_size;
            grid_size.x = trial_weights_nonzero_cnt_c / block_size.x + ((trial_weights_nonzero_cnt_c % block_size.x == 0)? 0:1);
            kernel_ParamsSparseRunSetup_GLM<<<grid_size, block_size,  0, stream>>>(dataset->ridx_sa_all->device(),
                                                        trial_included->device(), 
                                                        dataset->ridx_st_sall->device(), 
                                                        dataset->ridx_t_all->device(),
                                                        dataset->dim_N->device());
        }
        else {
            trial_included = trial_included_0;
            dataset->ridx_a_all_c = dataset->ridx_a_all;
            dataset->ridx_t_all_c = dataset->ridx_t_all;

            //sets some sizes
            checkCudaErrors(dataset->d2LL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for normal weighted run!");
            checkCudaErrors(dataset->dLL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for normal weighted run!");
            checkCudaErrors(dataset->LL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for normal weighted run!");
            checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_total(), -1), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for normal weighted run!");
        }
    }
    else {
        // this says all trial weights are 1 (normal log likelihood computation)
        trial_weights  = trial_weights_0;
        trial_included = trial_included_0;
        dataset->ridx_a_all_c = dataset->ridx_a_all;
        dataset->ridx_t_all_c = dataset->ridx_t_all;

        //sets some sizes
        checkCudaErrors(dataset->d2LL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->dLL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->LL->resize(stream, dataset->dim_N_total()), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_total(), -1), "GPUGLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
    }
    
    if(glm_params != NULL) { //this null check is so I could use this function only to change the weights if I wanted (I probably won't)
        msg->printMsgTxt(output_stream);
            
        checkCudaErrors(K->copyTo(stream, glm_params->K, false), "GPUGLM_parameters_GPU errors: could not copy K to device!");
    }
}

//============================================================================================================================
//Results class
        //constructor
template <class FPTYPE>
GPUGLM_results_GPU<FPTYPE>::GPUGLM_results_GPU(const GPUGLM_structure_args <FPTYPE> * GLMstructure, const size_t max_trials_, const int dev_, std::shared_ptr<GPUGL_msg> msg_) {
    dev = dev_;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0; //default stream

    //allocate GPU space for trial log likelihoods
    trialLL = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, max_trials_);
    checkCudaErrors(ce, "GPUGLM_results_GPU errors: could not allocate space for trialLL!" );

    //allocate GPU space for derivative + Hess results
    dK  = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_results_GPU errors: could not allocate space for dK!" );

    d2K = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GLMstructure->dim_K, GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_results_GPU errors: could not allocate space for d2K!" );
}


//destructor
template <class FPTYPE>
GPUGLM_results_GPU<FPTYPE>::~GPUGLM_results_GPU() {
    switchToDevice();
    cudaSafeFree(    trialLL     , "GPUGLM_results_GPU errors: could not free trial_weights_temp");

    cudaSafeFree( dK, "GPUGLM_results_GPU errors: could not free dK");
    cudaSafeFree(d2K, "GPUGLM_results_GPU errors: could not free d2K");
}

//copy back to host memory (into the object's own page locked memory)
template <class FPTYPE>
void GPUGLM_results_GPU<FPTYPE>::gatherResults(const GPUGLM_parameters_GPU<FPTYPE> * params, const GPUGLM_computeOptions<FPTYPE> * opts, const cudaStream_t stream, const cudaStream_t stream2) {
    switchToDevice();

    //copy the trial-wise log-likelihood
    if(opts->compute_trialLL) {
        checkCudaErrors(trialLL->copyGPUToHost(stream), "GPUGLM_results_GPU::copyResultsToHost errors: could not copy trialLL to host!");  
    }

    //copy dK
    if(opts->compute_dK) {
        checkCudaErrors(dK->copyGPUToHost(stream), "GPUGLM_results_GPU::copyResultsToHost errors: could not copy dK to host!"); 
    }

    //copy d2K
    if(opts->compute_d2K) {
        checkCudaErrors(d2K->copyGPUToHost(stream2), "GPUGLM_results_GPU::copyResultsToHost errors: could not copy d2K to host!"); 
    }
}

//adds results in page-locked host memory to user-supplied object for returning
template <class FPTYPE>
void GPUGLM_results_GPU<FPTYPE>::addToHost(const GPUGLM_parameters_GPU<FPTYPE> * params, GPUGLM_results<FPTYPE>* results_dest, const GPUGLM_computeOptions<FPTYPE> * opts, const GPUGLM_dataset_GPU<FPTYPE> * dataset, const bool reset) {

    //check the dims of the destination to see if they hold up
    if(opts->compute_trialLL && results_dest->dim_M() != max_trials()) {
        output_stream << "GPUGLM_results_GPU::addResults errors: results.dim_M = " << results_dest->dim_M() << " is the incorrect size! (expected dim_M = " << max_trials() << ")";
        msg->callErrMsgTxt(output_stream);
    }
    if((opts->compute_dK || opts->compute_d2K)  && results_dest->dim_K(msg) != dim_K()) {
        output_stream << "GPUGLM_results_GPU::addResults errors: results.dim_K = " << results_dest->dim_K(msg) << " is the incorrect size! (expected dim_B = " << dim_K() << ")";
        msg->callErrMsgTxt(output_stream);
    }
    
    //if reset, set destination memory to all 0's
    if(reset) {
        if(opts->compute_trialLL) {
            results_dest->trialLL->assign(0);
        }
        if(opts->compute_dK) {
            results_dest->dK->assign(0);
        }
        if(opts->compute_d2K) {
            results_dest->d2K->assign(0);
        }
    }

    //adds local results to dest
    if(opts->compute_trialLL) {
        for(unsigned int mm = 0; mm < max_trials(); mm++) {
            if(dataset->isInDataset_trial[mm] && (opts->trial_weights.empty() || opts->trial_weights[mm] != 0)) {
                (*(results_dest->trialLL))[mm] += (*trialLL)[mm];
            }
        }
    }

    if(opts->compute_dK) {
        for(unsigned int kk = 0; kk < dim_K(); kk++) {
            (*(results_dest->dK))[kk] += (*dK)[kk];
        }
    }
    if(opts->compute_d2K) {
        for(unsigned int kk = 0; kk < dim_K(); kk++) {
            for(unsigned int bb = kk; bb < dim_K(); bb++) {
                (*(results_dest->d2K))(kk, bb) += (*d2K)(kk, bb);
                //for symmetric matrices
                if(bb != kk) {
                    (*(results_dest->d2K))(bb, kk)  = (*(results_dest->d2K))(kk, bb);
                }
            }
        }
    }
}

//============================================================================================================================
//Dataset class
        
//Constructor takes in all the group data and GLM setup
template <class FPTYPE>
GPUGLM_dataset_GPU<FPTYPE>::GPUGLM_dataset_GPU(const GPUGLM_structure_args<FPTYPE> * GLMstructure, const GPUGLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_,  const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg_) {
    dev = block->dev_num;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;

    dt = GLMstructure->binSize;
    log_dt = log(GLMstructure->binSize);
    dim_N = new GPUData<size_t>(ce, GPUData_HOST_STANDARD, stream, block->trials.size()); //sets up dim_M()
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate dim_N on device!");

    // number of trials
    if(dim_M() == 0) {
        output_stream << "GPUGLM_dataset_GPU errors: no trials given to GPU block!";
        msg->callErrMsgTxt(output_stream);
    }
    max_trials_for_sparse_run = min(dim_M()/2, static_cast<size_t>(block->max_trials_for_sparse_run));

    if(GLMstructure->dim_K == 0) {
        output_stream << "GPUGLM_dataset_GPU errors: no covariates given!";
        msg->callErrMsgTxt(output_stream);
    }

    // setup up the order that trials go to the GPU
    //   in blocks ordered by neurons     
    isInDataset_trial.assign( max_trials_, false); //if each trial is in this block

    size_t dim_N_total_c = 0;
    dim_N_temp = 0;
    max_trial_length = 1;

    ridx_t_all = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate ridx_t_all on device!");
    id_t_trial = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate id_t_trial on device!");
    normalizingConstants_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate normalizingConstants_trial on device!");

    size_t dim_N_c = 0;
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        //save trial indices
        (*ridx_t_all)[mm] = dim_N_total_c;

        // get trial length
        (*dim_N)[mm] = block->trials[mm]->dim_N(msg);
        if((*dim_N)[mm] == 0) {
            output_stream << "GPUGLM_dataset_GPU errors: trials cannot be empty!";
            msg->callErrMsgTxt(output_stream);
        }

        dim_N_c       += (*dim_N)[mm] ; // add length to current neuron's total
        dim_N_total_c += (*dim_N)[mm] ; // add length to total 

        max_trial_length = max(max_trial_length, (*dim_N)[mm] ); //update max trial length

        //save trial and neuron number
        (*id_t_trial)[mm] = block->trials[mm]->trial_idx;
        if(isInDataset_trial[block->trials[mm]->trial_idx]) { //trial index already found
            output_stream << "GPUGLM_dataset_GPU errors: trial indices must be unique!";
            msg->callErrMsgTxt(output_stream);
        }
        isInDataset_trial[block->trials[mm]->trial_idx] = true;

        FPTYPE nc = 0; // normalizing constant
        if(GLMstructure->logLikeSettings == ll_poissExp || GLMstructure->logLikeSettings == ll_poissSoftRec) {
            for(unsigned int nn = 0; nn < (*dim_N)[mm]; nn++) {
                FPTYPE Y_c = (*(block->trials[mm]->Y))[nn];
                nc += (Y_c >= 0) ? -lgamma(floor(Y_c) + 1.0) : 0;
            }
        }
        (*normalizingConstants_trial)[mm] = nc;
    }

   id_a_trialM = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_N_total_c);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate id_a_trialM on device!");

    size_t N_total_ctr = 0;
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        for(unsigned int nn = 0; nn < (*dim_N)[mm]; nn++) {
            (*id_a_trialM)[N_total_ctr + nn] = mm;
        }
        N_total_ctr += (*dim_N)[mm];
    }

    //allocate space on GPU for data and copy any local values to GPU
        //spike counts
    Y = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for Y!" );
        //linear term
    X = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c, GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for X!" );
        

        //copy each trial to GPU
    
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        // spike counts
        cudaPos copyOffset = make_cudaPos((*ridx_t_all)[mm], 0, 0);

        checkCudaErrors(Y->copyTo(stream, block->trials[mm]->Y, true, copyOffset), "GPUGLM_dataset_GPU errors: could not copy Y to device!");

        //coefficients
        checkCudaErrors(X->copyTo(stream, block->trials[mm]->X, true, copyOffset), "GPUGLM_dataset_GPU errors: could not copy X to device!");
        
    } 

    //upload vectors to GPU
    checkCudaErrors(normalizingConstants_trial->copyHostToGPU(stream), "GPUGLM_dataset_GPU errors: could not copy normalizingConstants_trial to device!");
   
    checkCudaErrors(ridx_t_all->copyHostToGPU(stream), "GPUGLM_dataset_GPU errors: could not copy ridx_t_all to device!");
    checkCudaErrors(id_t_trial->copyHostToGPU(stream), "GPUGLM_dataset_GPU errors: could not copy id_t_trial to device!");
    checkCudaErrors(id_a_trialM->copyHostToGPU(stream), "GPUGLM_dataset_GPU errors: could not copy id_a_trialM to device!");
     
    checkCudaErrors(dim_N->copyHostToGPU(stream), "GPUGLM_dataset_GPU errors: could not copy dim_N to device!");
    
    //setup compute space
    LL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for LL!" );
    dLL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for dLL!" );
    d2LL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for d2LL!" );

    ridx_sa_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate ridx_sa_all on device!");
    ridx_st_sall = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M());
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate ridx_st_sall on device!");

    ridx_a_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total(), 0);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate ridx_a_all on device!");
    ridx_a_all_c = ridx_a_all;
    
    lambda = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total(), GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for lambda!" );

    size_t max_X_temp = max_trial_length * max_trials_for_sparse_run;
    if(max_X_temp > dim_N_total()) {
        output_stream << "GPUGLM_dataset_GPU errors: error creating compute space for sparse runs - required matrices too big" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    X_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, max_X_temp, GLMstructure->dim_K);
    checkCudaErrors(ce, "GPUGLM_dataset_GPU errors: could not allocate space for X_temp!" );
    
}

// destructor
template <class FPTYPE>
GPUGLM_dataset_GPU<FPTYPE>::~GPUGLM_dataset_GPU() {
    cudaSafeFree(Y, "GPUGLM_dataset_GPU errors: could not free Y");
    
    cudaSafeFree(X, "GPUGLM_dataset_GPU errors: could not free X");
    cudaSafeFree(X_temp, "GPUGLM_dataset_GPU errors: could not free X_temp");
    
    cudaSafeFree(normalizingConstants_trial, "GPUGLM_dataset_GPU errors: could not free normalizingConstants_trial");
    
    cudaSafeFree(ridx_t_all   , "GPUGLM_dataset_GPU errors: could not free ridx_t_all");
    cudaSafeFree(ridx_sa_all  , "GPUGLM_dataset_GPU errors: could not free ridx_sa_all");
    cudaSafeFree(ridx_st_sall , "GPUGLM_dataset_GPU errors: could not free ridx_st_sall");

    cudaSafeFree(ridx_a_all, "GPUGLM_dataset_GPU errors: could not free ridx_a_all");
    
    cudaSafeFree(id_t_trial , "GPUGLM_dataset_GPU errors: could not free id_t_trial");
    cudaSafeFree(id_a_trialM, "GPUGLM_dataset_GPU errors: could not free id_a_trialM");
    
    cudaSafeFree(dim_N, "GPUGLM_dataset_GPU errors: could not free dim_N");
    
    cudaSafeFree(  LL, "GPUGLM_dataset_GPU errors: could not free   LL");
    cudaSafeFree( dLL, "GPUGLM_dataset_GPU errors: could not free  dLL");
    cudaSafeFree(d2LL, "GPUGLM_dataset_GPU errors: could not free d2LL");
    cudaSafeFree(lambda, "GPUGLM_dataset_GPU errors: could not free lambda");
}

//explicitly create classes for single and double precision floating point for library
template class GPUGLM_parameters_GPU<float>;
template class GPUGLM_parameters_GPU<double>;

template class GPUGLM_results_GPU<float>;
template class GPUGLM_results_GPU<double>;
        
template class GPUGLM_dataset_GPU<float>;
template class GPUGLM_dataset_GPU<double>;

};//end namespace