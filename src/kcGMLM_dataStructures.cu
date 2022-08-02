/*
 * kcGMLM_dataStructures.cu
 * Holds all the data - parameters, results, regressors, computation space
 * for a GMLM (on one GPU).
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
#include "kcGMLM_dataStructures.hpp"
//#include <tgmath.h>

namespace kCUDA {
   

//============================================================================================================================
//Parameter class
        //constructor
template <class FPTYPE>
GPUGMLM_parameters_GPU<FPTYPE>::GPUGMLM_parameters_GPU(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const size_t dim_M_,  const int dev_, std::shared_ptr<GPUGL_msg> msg_)  : isSimultaneousPopulation(GMLMstructure->isSimultaneousPopulation) {
    dev = dev_;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0;

    //setup any log like settings
    logLikeSettings = GMLMstructure->logLikeSettings;
    if(GMLMstructure->logLikeParams.size() > 0) {
        logLikeParams = new GPUData<FPTYPE>(ce, GPUData_HOST_STANDARD, stream, GMLMstructure->logLikeParams.size());
        checkCudaErrors(ce,  "GPUGMLM_parameters_GPU errors: could not allocate space for logLikeParams!" );
        for(unsigned int ii = 0; ii < GMLMstructure->logLikeParams.size(); ii++) {
            (*logLikeParams)[ii] = GMLMstructure->logLikeParams[ii];
        }
        ce = logLikeParams->copyHostToGPU(stream);
        checkCudaErrors(ce,  "GPUGMLM_parameters_GPU errors: could not copy logLikeParams to GPU!" );
    }
    else {
        logLikeParams = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, 0);
        checkCudaErrors(ce,  "GPUGMLM_parameters_GPU errors: could not allocate space for logLikeParams!" );
    }

    //allocate GPU space for trial weights
    size_t cols_for_weights = isSimultaneousPopulation ? GMLMstructure->dim_P : 1;
    trial_weights_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_, cols_for_weights);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_weights_temp!" );
    trial_included_temp = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_, cols_for_weights);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_included_temp!" );

    trial_weights_0 = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_weights_0!" );
    trial_weights = trial_weights_0;

    trial_included_0 = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_included_0!" );
    trial_included = trial_included_0;
    
    //allocate GPU space
    W = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_P);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for W!" );
    B = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_B, GMLMstructure->dim_P);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for B!" );

    checkCudaErrors(cudaEventCreate(&paramsLoaded_event), "GPUGMLM_parameters_GPU errors: could not create event!");

    //setup each group
    Groups.resize(GMLMstructure->Groups.size());
    for(unsigned int jj = 0; jj < GMLMstructure->Groups.size(); jj++) {
        Groups[jj] = new GPUGMLM_parameters_Group_GPU<FPTYPE>(GMLMstructure->Groups[jj], this);
    }
}

template <class FPTYPE>
GPUGMLM_parameters_Group_GPU<FPTYPE>::GPUGMLM_parameters_Group_GPU(const GPUGMLM_structure_Group_args<FPTYPE> * GMLMGroupStructure, const GPUGMLM_parameters_GPU<FPTYPE> * parent_) : parent(parent_) {
    msg = parent->msg;
    dev = parent->dev;
    switchToDevice();
    cudaError_t ce;

    cudaStream_t stream = 0;

    //allocate GPU space
    V = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_P(), GMLMGroupStructure->dim_R_max);
    checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for V!" );
    T.assign(GMLMGroupStructure->dim_S(), NULL);
    F.assign(GMLMGroupStructure->dim_D(msg), NULL);
    dF_dT.assign(GMLMGroupStructure->dim_S(), NULL);

    compute_dT = new GPUData<bool>(ce, GPUData_HOST_PAGELOCKED, stream, dim_S());
    checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for compute_dT!" );
    compute_dF = new GPUData<bool>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMGroupStructure->dim_D(msg));
    checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for compute_dF!" );
    
    factor_idx = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_S()); 
    checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate factor_idx!");
    N_per_factor = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, GMLMGroupStructure->dim_D(msg)); 
    checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate N_per_factor!");

    N_per_factor->assign(0);
    
    std::vector<size_t> dim_F_c;
    dim_F_c.assign(dim_D(), 1);
    if(dim_D() > MAX_DIM_D) {
        output_stream << "GPUGMLM_parameters_Group_GPU errors: dim_D greater than max allowed (" << MAX_DIM_D << "!";
        msg->callErrMsgTxt(output_stream);
    }

    dim_F_max   = 0;
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        if(GMLMGroupStructure->factor_idx[ss] >= dim_D()) {
            output_stream << "GPUGMLM_parameters_Group_GPU errors: invalid factor index!";
            msg->callErrMsgTxt(output_stream);
        }
        (*factor_idx)[ss] = GMLMGroupStructure->factor_idx[ss];

        dim_F_c[GMLMGroupStructure->factor_idx[ss]] *= GMLMGroupStructure->dim_T[ss];
        (*N_per_factor)[GMLMGroupStructure->factor_idx[ss]] += 1;
        T[ss] = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMGroupStructure->dim_T[ss], GMLMGroupStructure->dim_R_max);
        checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for T[ss]!" );
    }
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        int dd = (*factor_idx)[ss];
        dF_dT[ss] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, GMLMGroupStructure->dim_T[ss], dim_F_c[dd], GMLMGroupStructure->dim_R_max);
        checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for dF_dT[ss]!" );
    }
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        if((*N_per_factor)[dd] > 1) {
            F[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_F_c[dd], GMLMGroupStructure->dim_R_max);
            checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for F[dd]!" );
        }
        else if((*N_per_factor)[dd] == 1) {
            //find the T for this factor if is unique
            for(unsigned int ss = 0; ss < dim_S(); ss++) {
                if((*factor_idx)[ss] == dd) {
                    F[dd] = T[ss];
                    break;
                }
            }
        }
        else {
            output_stream << "GPUGMLM_parameters_Group_GPU errors: tensor factor has no components!";
            msg->callErrMsgTxt(output_stream);
        }
        dim_F_max = max(dim_F_max, dim_F_c[dd]);           
    }

    if(GMLMGroupStructure->dim_S() == 0 || dim_F_max == 0) {
        output_stream << "GPUGMLM_parameters_Group_GPU errors: tensor has no components!";
        msg->callErrMsgTxt(output_stream);
    }

    checkCudaErrors(factor_idx->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy factor_idx to device!");
    checkCudaErrors(N_per_factor->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy N_per_factor to device!");
}

//destructor
template <class FPTYPE>
GPUGMLM_parameters_GPU<FPTYPE>::~GPUGMLM_parameters_GPU() {
    switchToDevice();
    checkCudaErrors("Error in start of GPUGMLM_parameters_GPU destructor!");
    

    checkCudaErrors(cudaEventDestroy(paramsLoaded_event), "GPUGMLM_parameters_GPU errors: could not free event!");

    cudaSafeFree(trial_weights_temp , "GPUGMLM_parameters_GPU errors: could not free trial_weights_temp");
    cudaSafeFree(trial_included_temp, "GPUGMLM_parameters_GPU errors: could not free trial_included_temp");
    cudaSafeFree(trial_weights_0    , "GPUGMLM_parameters_GPU errors: could not free trial_weights_0");
    cudaSafeFree(trial_included_0   , "GPUGMLM_parameters_GPU errors: could not free trial_included_0");

    cudaSafeFree(logLikeParams, "GPUGMLM_parameters_GPU errors: could not free logLikeParams." );

    cudaSafeFree(W, "GPUGMLM_parameters_GPU errors: could not free W");
    cudaSafeFree(B, "GPUGMLM_parameters_GPU errors: could not free B");
    for(unsigned int jj = 0; jj < Groups.size(); jj++) {
        delete Groups[jj];
    }
}
template <class FPTYPE>
GPUGMLM_parameters_Group_GPU<FPTYPE>::~GPUGMLM_parameters_Group_GPU() {
    switchToDevice();
    for(unsigned int dd = 0; dd < N_per_factor->size(); dd++) {
        if((*N_per_factor)[dd] > 1) {
            cudaSafeFree(F[dd], "GPUGMLM_parameters_Group_GPU errors: could not free F");
        }
    }
    cudaSafeFreeVector(T, "GPUGMLM_parameters_Group_GPU errors: could not free T");
    cudaSafeFreeVector(dF_dT, "GPUGMLM_parameters_Group_GPU errors: could not free dF_dT");
    cudaSafeFree(      V, "GPUGMLM_parameters_Group_GPU errors: could not free V");
    cudaSafeFree(compute_dT, "GPUGMLM_parameters_Group_GPU errors: could not free compute_dT");
    cudaSafeFree(compute_dF, "GPUGMLM_parameters_Group_GPU errors: could not free compute_dF");

    cudaSafeFree(factor_idx, "GPUGMLM_parameters_Group_GPU errors: could not free factor_idx");
    cudaSafeFree(N_per_factor, "GPUGMLM_parameters_Group_GPU errors: could not free N_per_factor");

}

/* kernel for setting up sparse run indices
*   One thread per trial being run. Sets up a map between the current indices (0:dim_N_temp-1) to the full indices (0:dim_N-1)
* 
*/
__global__ void kernel_ParamsSparseRunSetup(GPUData_kernel<unsigned int> ridx_sa_all,
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
void GPUGMLM_parameters_GPU<FPTYPE>::copyToGPU(const GPUGMLM_params<FPTYPE> * gmlm_params, GPUGMLM_dataset_GPU<FPTYPE> * dataset, const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLM_computeOptions<FPTYPE> * opts) {
    if(isSimultaneousPopulation) {
        output_stream << "GPUGMLM_parameters_GPU errors: copyToGPU called for population GMLM setup when individual neuron GMLM was expected." << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    switchToDevice();

    //copies trial weights if given
    if(opts != NULL && !opts->trial_weights->empty() && (opts->trial_weights->getSize(0) != dataset->max_trials() || opts->trial_weights->getSize(1) != 1)) {
        output_stream << "GPUGMLM_parameters_GPU errors: input does not have correct number of trial weights" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    bool reset_sizes = false;
    if(opts != NULL && opts->update_weights && !opts->trial_weights->empty()) {
        size_t trial_weights_nonzero_cnt_c = 0;
        dataset->dim_N_temp = 0;
        dataset->dim_N_neuron_temp.assign(dim_P(), 0);

        //gets weights for each trial on this GPU block
        for(unsigned int mm = 0; mm < dim_M(); mm++) {
            (*trial_weights_temp)[mm] = (*(opts->trial_weights))[(*(dataset->id_t_trial))[mm]];
    
            //if trial is included
            if((*trial_weights_temp)[mm] != 0) {
                (*trial_included_temp)[trial_weights_nonzero_cnt_c] = mm;
                (*(dataset->ridx_st_sall))[trial_weights_nonzero_cnt_c] = dataset->dim_N_temp;

                dataset->dim_N_temp += (*(dataset->dim_N))[mm];

                dataset->dim_N_neuron_temp[dataset->id_t_neuron[mm]] += (*(dataset->dim_N))[mm];

                trial_weights_nonzero_cnt_c++;
            }
        }

        checkCudaErrors(trial_included_temp->resize(stream, trial_weights_nonzero_cnt_c), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
        trial_included = trial_included_temp;

        if(trial_weights_nonzero_cnt_c != 0) {
            // copies weights to GPU
            checkCudaErrors(trial_weights_temp->copyHostToGPU(stream), "GPUGMLM_parameters_GPU errors: could not copy trial_weights_temp to device!");  
            trial_weights = trial_weights_temp;
        }
        else {
            // if there are no trials, might as well not copy anything more
            return;
        }

        
        //copy list of trials with nonzero weights to host only if the number is small enough for a sparse run
        if(trial_weights_nonzero_cnt_c <= dataset->max_trials_for_sparse_run) {
            
                // neuron index: this assumes I loaded the data correctly so it's sorted by neuron
            (*(dataset->ridx_sn_sall))[0] = 0;
            for(unsigned int pp = 1; pp < dim_P() + 1; pp++) {
                (*(dataset->ridx_sn_sall))[pp] = dataset->dim_N_neuron_temp[pp-1] + (*(dataset->ridx_sn_sall))[pp-1];
            }

            //sets some sizes
            checkCudaErrors(dataset->dLL->resize(   stream, dataset->dim_N_temp), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->LL->resize(    stream, dataset->dim_N_temp), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_temp, -1), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->resize(  stream, trial_weights_nonzero_cnt_c), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_st_sall->resize(stream, trial_weights_nonzero_cnt_c), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_sa_all->resize( stream, dataset->dim_N_temp), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->copyHostToGPU(  stream), "GPUGMLM_parameters_GPU errors: could not copy trial_included_temp to device!");
            checkCudaErrors(dataset->ridx_st_sall->copyHostToGPU(stream), "GPUGMLM_parameters_GPU errors: could not copy ridx_st_sall to device!");
            checkCudaErrors(dataset->ridx_sn_sall->copyHostToGPU(stream), "GPUGMLM_parameters_GPU errors: could not copy ridx_sn_sall to device!");
            
            trial_included = trial_included_temp;
            dataset->ridx_a_all_c = dataset->ridx_sa_all;
            dataset->ridx_t_all_c = dataset->ridx_st_sall;

            //setup a special index variable
            dim3 block_size;
            block_size.x = min(static_cast<size_t>(1024), trial_weights_nonzero_cnt_c);
            dim3 grid_size;
            grid_size.x = trial_weights_nonzero_cnt_c / block_size.x + ((trial_weights_nonzero_cnt_c % block_size.x == 0)? 0:1);
            kernel_ParamsSparseRunSetup<<<grid_size, block_size,  0, stream>>>(dataset->ridx_sa_all->device(),
                                                                               trial_included->device(), 
                                                                               dataset->ridx_st_sall->device(), 
                                                                               dataset->ridx_t_all->device(),
                                                                               dataset->dim_N->device());
        }
        else {
            reset_sizes = true;
        }
        //checkCudaErrors( cudaStreamSynchronize(stream), "GPUGMLM_parameters_GPU::copyToGPU errors: could not synchronize stream for sparse run!");
        checkCudaErrors(cudaEventRecord(paramsLoaded_event, stream), "GPUGMLM_parameters_GPU::copyToGPU errors: could not add event to stream for sparse run!");
    }
    else if(opts->update_weights) {
        // this says all trial weights are 1 (normal log likelihood computation)
        trial_weights  = trial_weights_0;
        reset_sizes = true;
        checkCudaErrors(cudaEventRecord(paramsLoaded_event), "GPUGMLM_parameters_GPU::copyToGPU errors: could not record event!");
    }
    else {
        reset_sizes = false;
        checkCudaErrors(cudaEventRecord(paramsLoaded_event), "GPUGMLM_parameters_GPU::copyToGPU errors: could not record event!");
    }

    if(reset_sizes) {
        trial_included = trial_included_0;
        dataset->ridx_a_all_c = dataset->ridx_a_all;
        dataset->ridx_t_all_c = dataset->ridx_t_all;

        for(unsigned int pp = 0; pp < dim_P(); pp++) {
            dataset->dim_N_neuron_temp[pp] = dataset->dim_N_neuron[pp];
        }

         //sets some sizes
        checkCudaErrors(dataset->dLL->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->LL->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
    }
    
    if(gmlm_params != NULL) { //this null check is so I could use this function only to change the weights if I wanted (I probably won't)
        //checks some dimensions
        if(gmlm_params->dim_B() != dim_B()) {
            output_stream << "GPUGMLM_parameters_GPU errors: input does not have correct number of linear coefficients (received " << gmlm_params->dim_B() << ", expected " << dim_B() << ")" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }
        if(gmlm_params->dim_P(msg) != dim_P()) {
            output_stream << "GPUGMLM_parameters_GPU errors: input does not have correct number of neurons (received " << gmlm_params->dim_P(msg) << ", expected " << dim_P() << ")" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }
        if(opts->Groups.size() != dim_J()) {
            output_stream << "GPUGMLM_parameters_GPU errors: input options does not have correct number of groups!" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }

        //copy parameters to GPU
        checkCudaErrors(W->copyTo(stream, gmlm_params->W, false), "GPUGMLM_parameters_GPU errors: could not copy W to device!");
        checkCudaErrors(B->copyTo(stream, gmlm_params->B, false), "GPUGMLM_parameters_GPU errors: could not copy B to device!");
        
        //for each group
        for(unsigned int jj = 0; jj < dim_J(); jj++) {
            Groups[jj]->copyToGPU(gmlm_params->Groups[jj], stream_Groups[jj], opts->Groups[jj]);
        }
    }
}


//copy all parameters to GPU
template <class FPTYPE>
void GPUGMLM_parameters_GPU<FPTYPE>::copyToGPU(const GPUGMLM_params<FPTYPE> * gmlm_params, GPUGMLMPop_dataset_GPU<FPTYPE> * dataset, const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLM_computeOptions<FPTYPE> * opts) {
    if(!isSimultaneousPopulation) {
        output_stream << "GPUGMLM_parameters_GPU errors: copyToGPU called for individual neuron GMLM setup when population GMLM was expected." << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    switchToDevice();

    //copies trial weights if given
    if(opts != NULL && !opts->trial_weights->empty() && opts->trial_weights->getSize(0) != dataset->max_trials() && (opts->trial_weights->getSize(1) != 1 || opts->trial_weights->getSize(1) != dim_P())) {
        output_stream << "GPUGMLMPop_parameters_GPU errors: input does not have correct number of trial weights" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    bool reset_sizes = false;
    if(opts != NULL && opts->update_weights && opts->trial_weights->size() != 0) {
        size_t trial_weights_nonzero_cnt_c = 0;
        dataset->dim_N_temp = 0;
        checkCudaErrors(trial_weights_temp->resize(stream, -1, opts->trial_weights->getSize(1)), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
        
        //gets weights for each trial on this GPU block
        for(unsigned int mm = 0; mm < dim_M(); mm++) {
            bool included = false;
            if(opts->trial_weights->getSize(1) == 1) {
                (*trial_weights_temp)[mm] = (*(opts->trial_weights))[(*(dataset->id_t_trial))[mm]];
                included = (*trial_weights_temp)[mm] != 0;
            }
            else {
                for(unsigned int pp = 0; pp < dim_P(); pp++) {
                    (*trial_weights_temp)(mm, pp) = (*(opts->trial_weights))((*(dataset->id_t_trial))[mm], pp);
                    included = included || (*trial_weights_temp)(mm,pp) != 0;
                }
            }
    
            //if trial is included
            if(included) {
                (*trial_included_temp)[trial_weights_nonzero_cnt_c] = mm;
                (*(dataset->ridx_st_sall))[trial_weights_nonzero_cnt_c] = dataset->dim_N_temp;
                dataset->dim_N_temp += (*(dataset->dim_N))[mm];
                trial_weights_nonzero_cnt_c++;
            }
        }

        checkCudaErrors(trial_included_temp->resize(stream, trial_weights_nonzero_cnt_c), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
        trial_included = trial_included_temp;

        if(trial_weights_nonzero_cnt_c != 0) {
            // copies weights to GPU
            checkCudaErrors(trial_weights_temp->copyHostToGPU(stream), "GPUGMLMPop_parameters_GPU errors: could not copy trial_weights_temp to device!");  
            trial_weights = trial_weights_temp;
        }
        else {
            // if there are no trials, might as well not copy anything more
            return;
        }

        //copy list of trials with nonzero weights to host only if the number is small enough for a sparse run
        if(trial_weights_nonzero_cnt_c <= dataset->max_trials_for_sparse_run) {
            //sets some sizes
            checkCudaErrors(dataset->dLL->resize(   stream, dataset->dim_N_temp), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->LL->resize(    stream, dataset->dim_N_temp), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_temp), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->resize(  stream, trial_weights_nonzero_cnt_c), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_st_sall->resize(stream, trial_weights_nonzero_cnt_c), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");
            checkCudaErrors(dataset->ridx_sa_all->resize( stream, dataset->dim_N_temp), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not set sizes for sparse run!");

            checkCudaErrors(trial_included_temp->copyHostToGPU(  stream), "GPUGMLMPop_parameters_GPU errors: could not copy trial_included_temp to device!");
            checkCudaErrors(dataset->ridx_st_sall->copyHostToGPU(stream), "GPUGMLMPop_parameters_GPU errors: could not copy ridx_st_sall to device!");
            
            trial_included = trial_included_temp;
            dataset->ridx_a_all_c = dataset->ridx_sa_all;
            dataset->ridx_t_all_c = dataset->ridx_st_sall;

            //setup a special index variable
            dim3 block_size;
            block_size.x = min(static_cast<size_t>(1024), trial_weights_nonzero_cnt_c);
            dim3 grid_size;
            grid_size.x = trial_weights_nonzero_cnt_c / block_size.x + ((trial_weights_nonzero_cnt_c % block_size.x == 0)? 0:1);
            kernel_ParamsSparseRunSetup<<<grid_size, block_size,  0, stream>>>(dataset->ridx_sa_all->device(),
                                                                               trial_included->device(), 
                                                                               dataset->ridx_st_sall->device(), 
                                                                               dataset->ridx_t_all->device(),
                                                                               dataset->dim_N->device());
        }
        else {
            reset_sizes = true;
        }
        //checkCudaErrors( cudaStreamSynchronize(stream), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not synchronize stream for sparse run!");
        checkCudaErrors(cudaEventRecord(paramsLoaded_event, stream), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not add event to stream for sparse run!");
    }    
    else if(opts->update_weights) {
        // this says all trial weights are 1 (normal log likelihood computation)
        trial_weights  = trial_weights_0;
        reset_sizes = true;
        checkCudaErrors(cudaEventRecord(paramsLoaded_event), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not record event!");
    }
    else {
        reset_sizes = false;
        checkCudaErrors(cudaEventRecord(paramsLoaded_event), "GPUGMLMPop_parameters_GPU::copyToGPU errors: could not record event!");
    }

    if(reset_sizes) {
        trial_included = trial_included_0;
        dataset->ridx_a_all_c = dataset->ridx_a_all;
        dataset->ridx_t_all_c = dataset->ridx_t_all;

         //sets some sizes
        checkCudaErrors(dataset->dLL->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->LL->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
        checkCudaErrors(dataset->lambda->resize(stream, dataset->dim_N_total()), "GPUGMLM_parameters_GPU::copyToGPU errors: could not set sizes for full run!");
    }
    
    if(gmlm_params != NULL) { //this null check is so I could use this function only to change the weights if I wanted (I probably won't)
        //checks some dimensions
        if(gmlm_params->dim_B() != dim_B()) {
            output_stream << "GPUGMLMPop_parameters_GPU errors: input does not have correct number of linear coefficients (received " << gmlm_params->dim_B() << ", expected " << dim_B() << ")" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }
        if(gmlm_params->dim_P(msg) != dim_P()) {
            output_stream << "GPUGMLMPop_parameters_GPU errors: input does not have correct number of neurons (received " << gmlm_params->dim_P(msg) << ", expected " << dim_P() << ")" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }
        if(opts->Groups.size() != dim_J()) {
            output_stream << "GPUGMLMPop_parameters_GPU errors: input options does not have correct number of groups!" << std::endl;
            msg->callErrMsgTxt(output_stream);
        }

        //copy parameters to GPU
        checkCudaErrors(W->copyTo(stream, gmlm_params->W, false), "GPUGMLMPop_parameters_GPU errors: could not copy W to device!");
        checkCudaErrors(B->copyTo(stream, gmlm_params->B, false), "GPUGMLMPop_parameters_GPU errors: could not copy B to device!");
        
        //for each group
        for(unsigned int jj = 0; jj < dim_J(); jj++) {
            Groups[jj]->copyToGPU(gmlm_params->Groups[jj], stream_Groups[jj], opts->Groups[jj]);
        }
    }
}
 
        
//copy to GPU
template <class FPTYPE>
void GPUGMLM_parameters_Group_GPU<FPTYPE>::copyToGPU(const GPUGMLM_group_params<FPTYPE> * gmlm_group_params, const cudaStream_t stream, const GPUGMLM_group_computeOptions * opts) {
    switchToDevice();
    //set current rank
    size_t dim_R_results = gmlm_group_params->dim_R(msg);
    checkCudaErrors(set_dim_R(dim_R_results, stream), "GPUGMLM_parameters_Group_GPU errors: could not set new dim_R");

    //check dimensions
    if(dim_S() != gmlm_group_params->dim_S()) {
        output_stream << "GPUGMLM_parameters_Group_GPU errors: Invalid tensor coefficient group order. received dim_S = " << gmlm_group_params->dim_S() << ", expected dim_S = " << dim_S() << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        if(gmlm_group_params->dim_T(ss, msg) != dim_T(ss)) {
            output_stream << "GPUGMLM_parameters_Group_GPU errors: Invalid tensor coefficient size. Received dim_T = " << gmlm_group_params->dim_T(ss, msg) << ", expected dim_T = " << dim_T(ss) << std::endl;
            msg->callErrMsgTxt(output_stream);
        }
    }

    //load compute_T to GPU
    if(opts->compute_dT.size() != dim_S()) {
        output_stream << "GPUGMLM_parameters_Group_GPU errors: Invalid compute_dt" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    compute_dF->assign(false);
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        (*compute_dT)[ss] = opts->compute_dT[ss];
        (*compute_dF)[(*factor_idx)[ss]] = (*compute_dF)[(*factor_idx)[ss]] || opts->compute_dT[ss];
    }
    checkCudaErrors(compute_dT->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy compute_dT to device!");
    checkCudaErrors(compute_dF->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy compute_dF to device!");
        
    //copy  to GPU
    checkCudaErrors(V->copyTo(stream, gmlm_group_params->V, false), "GPUGMLM_parameters_Group_GPU errors: could not copy V to device!");

    //copy each T
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        checkCudaErrors(T[ss]->copyTo(stream, gmlm_group_params->T[ss], false), "GPUGMLM_parameters_Group_GPU errors: could not copy T to device!");
    }

    assembleF(stream);
}

/* kernel for setting up full regressor matrix
* 
*/
        
template <class FPTYPE>
__global__ void kernel_assembleFactorFilter(GPUData_array_kernel<FPTYPE, MAX_DIM_D> F, GPUData_array_kernel<FPTYPE, MAX_DIM_D> dF_dT,
        const GPUData_array_kernel<FPTYPE, MAX_DIM_D> T,
        const GPUData_kernel<unsigned int> factor_idx,
        const GPUData_kernel<unsigned int> N_per_factor, const GPUData_kernel<bool> compute_dT) {
     
    size_t row    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t factor = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(factor < F.N && N_per_factor[factor] > 1) {
        const size_t dim_S = T.N;
        const size_t dim_F = F[factor].x;
        
        if(row < dim_F) {
            for(unsigned int rr = 0; rr < F[factor].y; rr++) {
                for(unsigned int ss_c = 0; ss_c < dim_S; ss_c++) {
                    if(factor_idx[ss_c] == factor && compute_dT[ss_c]) {
                        for(unsigned int tt = 0; tt < dF_dT[ss_c].x; tt++) {
                            dF_dT[ss_c](tt, row, rr) = 0;
                        }
                    }
                }

                size_t T_ctr = 1;
                for(unsigned int ss = 0; ss < dim_S; ss++) {
                    if(factor_idx[ss] == factor) {
                        size_t tt = (row/T_ctr) % T[ss].x;
                        if(T_ctr == 1) {
                            F[factor](row, rr)  = T[ss](tt, rr);
                        }
                        else {
                            F[factor](row, rr) *= T[ss](tt, rr);
                        }
                        T_ctr *= T[ss].x;

                        if(compute_dT[ss]) {
                        	dF_dT[ss](tt, row, rr) = 1;
                        }
                    }
                }
                
                for(unsigned int ss_c = 0; ss_c < dim_S; ss_c++) {
                    if(factor_idx[ss_c] == factor && compute_dT[ss_c]) {

                        size_t T_ctr = 1;
                        for(unsigned int ss = 0; ss < dim_S; ss++) {
                            if(factor_idx[ss] == factor) {
                                size_t tt = (row/T_ctr) % T[ss].x;
                                if(ss != ss_c) {
                                    for(unsigned int tt_0 = 0; tt_0 < dF_dT[ss_c].x; tt_0++) {
                                        dF_dT[ss_c](tt_0, row, rr) *= T[ss](tt, rr);
                                    }
                                }
                                T_ctr *= T[ss].x;
                            }
                        }
                    }
                }
            }
        }
    }
}

//assembles the complete regressor matrix (without neuron weights)
//if dim_S>derivative_dim>=0, replaces T[derivative_dim] with ones. if dd < 0, does all factors
template <class FPTYPE>
void GPUGMLM_parameters_Group_GPU<FPTYPE>::assembleF(const cudaStream_t stream) {
    if(dim_S() > 1) {
        dim3 block_size;
        block_size.x = min(static_cast<size_t>(256), dim_F_max);
        block_size.y = min(static_cast<size_t>(4)  , dim_D());
        dim3 grid_size;
        grid_size.x = dim_F_max / block_size.x + ((dim_F_max % block_size.x == 0)? 0:1);
        grid_size.y = dim_D()   / block_size.y + ((dim_D()   % block_size.y == 0)? 0:1);

        kernel_assembleFactorFilter<<<grid_size, block_size, 0, stream>>>( GPUData<FPTYPE>::assembleKernels(F), GPUData<FPTYPE>::assembleKernels(dF_dT),  GPUData<FPTYPE>::assembleKernels(T), factor_idx->device(), N_per_factor->device(), compute_dT->device());
    }
}

//============================================================================================================================
//Results class
        //constructor
template <class FPTYPE>
GPUGMLM_results_GPU<FPTYPE>::GPUGMLM_results_GPU(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const size_t max_trials_, const int dev_, std::shared_ptr<GPUGL_msg> msg_)  : isSimultaneousPopulation(GMLMstructure->isSimultaneousPopulation)  {
    dev = dev_;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0;

    //allocate GPU space for trial weights
    if(isSimultaneousPopulation) {
        trialLL = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, max_trials_, GMLMstructure->dim_P);
    }
    else {
        trialLL = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, max_trials_);
    }
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for trialLL!" );

    //allocate GPU space
    dW = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_P);
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for dW!" );
    
    dB = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_B, GMLMstructure->dim_P);
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for dB!" );

    //setup each group
    Groups.resize(GMLMstructure->Groups.size());
    for(unsigned int jj = 0; jj < dim_J(); jj++) {
        Groups[jj] = new GPUGMLM_results_Group_GPU<FPTYPE>(GMLMstructure->Groups[jj], this);
    }
}

template <class FPTYPE>
GPUGMLM_results_Group_GPU<FPTYPE>::GPUGMLM_results_Group_GPU(const GPUGMLM_structure_Group_args<FPTYPE> * GMLMGroupStructure, const GPUGMLM_results_GPU<FPTYPE> * parent_) : parent(parent_) {
    msg = parent->msg;
    dev = parent->dev;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0;

    //allocate GPU space
    dV = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_P(), GMLMGroupStructure->dim_R_max);
    checkCudaErrors(ce, "GPUGMLM_results_Group_GPU errors: could not allocate space for dV!" );

    dT.resize(GMLMGroupStructure->dim_T.size());
    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        dT[ss] = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMGroupStructure->dim_T[ss], GMLMGroupStructure->dim_R_max);
        checkCudaErrors(ce, "GPUGMLM_results_Group_GPU errors: could not allocate space for T[ss]!" );
    }

    
    dF.assign(GMLMGroupStructure->dim_D(msg), NULL);
    dF_assigned.assign(dim_D(), false);

    std::vector<size_t> dim_F_c;
    std::vector<size_t> NF;
    dim_F_c.assign(dim_D(), 1);
    NF.assign(dim_D(), 0);

    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        NF[GMLMGroupStructure->factor_idx[ss]]++;
        dim_F_c[GMLMGroupStructure->factor_idx[ss]] *= GMLMGroupStructure->dim_T[ss];
    }
    
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        if(NF[dd] > 1) {
            dF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_F_c[dd], GMLMGroupStructure->dim_R_max);
            checkCudaErrors(ce, "GPUGMLM_results_Group_GPU errors: could not allocate space for dF[dd]!" );
            dF_assigned[dd] = true;
        }
        else {
            dF_assigned[dd] = false;
            for(unsigned int ss = 0; ss < dim_S(); ss++) {
                if(GMLMGroupStructure->factor_idx[ss] == dd) {
                    dF[dd] = dT[ss];
                    break;
                }
            }
        }
    }
}

//destructor
template <class FPTYPE>
GPUGMLM_results_GPU<FPTYPE>::~GPUGMLM_results_GPU() {
    switchToDevice();
    cudaSafeFree(    trialLL     , "GPUGMLM_results_GPU errors: could not free trialLL");

    cudaSafeFree(dW, "GPUGMLM_results_GPU errors: could not free W");
    cudaSafeFree(dB, "GPUGMLM_results_GPU errors: could not free B");
    for(unsigned int jj = 0; jj < Groups.size(); jj++) {
        delete Groups[jj];
    }
}
template <class FPTYPE>
GPUGMLM_results_Group_GPU<FPTYPE>::~GPUGMLM_results_Group_GPU() {
    switchToDevice();
    cudaSafeFreeVector(dT, "GPUGMLM_results_Group_GPU errors: could not free dT");
    for(unsigned int dd = 0; dd < dF.size(); dd++) {
        if(dF_assigned[dd]) {
            cudaSafeFree(dF[dd], "GPUGMLM_results_Group_GPU errors: could not free dF");
        }
    }
    cudaSafeFree(      dV, "GPUGMLM_results_Group_GPU errors: could not free dV");
}

//copy back to host memory (into the object's own page locked memory)
template <class FPTYPE>
void GPUGMLM_results_GPU<FPTYPE>::gatherResults(const GPUGMLM_parameters_GPU<FPTYPE> * params, const GPUGMLM_computeOptions<FPTYPE> * opts, const cudaStream_t stream_main, const std::vector<cudaStream_t> stream_Groups) {
    switchToDevice();

    //copy the trial-wise log-likelihood
    if(opts->compute_trialLL) {
        //trialLL->printInfo(output_stream, "trialLL");
        //msg->printMsgTxt(output_stream);
        checkCudaErrors(trialLL->copyGPUToHost(stream_main), "GPUGMLM_results_GPU::copyResultsToHost errors: could not copy trialLL to host!");  
    }

    //copy dW
    if(opts->compute_dW) {
        checkCudaErrors(dW->copyGPUToHost(stream_main), "GPUGMLM_results_GPU::copyResultsToHost errors: could not copy dW to host!"); 
    }

    //copy dB
    if(opts->compute_dB) {
        checkCudaErrors(dB->copyGPUToHost(stream_main),"GPUGMLM_results_GPU::gatherResults errors: could not copy dB to host!"); 
    }

    //copy each group
    if(opts->Groups.size() != Groups.size()) {
        output_stream << "GPUGMLM_Group_GPU::gatherResults errors: invalid options!";
        msg->callErrMsgTxt(output_stream);
    }
    for(unsigned int jj = 0; jj < Groups.size(); jj++) {
        Groups[jj]->gatherResults(params->Groups[jj], opts->Groups[jj], stream_Groups[jj]);
    }
}

template <class FPTYPE>
void GPUGMLM_results_Group_GPU<FPTYPE>::gatherResults(const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream) {
    switchToDevice();
    //check dims
    if(opts->compute_dT.size() != dT.size()) {
        output_stream << "GPUGMLM_results_Group_GPU::gatherResults errors: invalid options!";
        msg->callErrMsgTxt(output_stream);
    }

    //copy dV
    if(opts->compute_dV) {
        checkCudaErrors(dV->copyGPUToHost(stream),"GPUGMLM_results_Group_GPU::gatherResults errors: could not copy dV to host!"); 
    }

    //copy dT
    for(unsigned int ss = 0; ss < dT.size(); ss++) {
        if(opts->compute_dT[ss]) {
            checkCudaErrors(dT[ss]->copyGPUToHost(stream),"GPUGMLM_results_Group_GPU::gatherResults errors: could not copy dT to host!"); 
        }
    }
}

//adds results in page-locked host memory to user-supplied object for returning
template <class FPTYPE>
void GPUGMLM_results_GPU<FPTYPE>::addToHost(const GPUGMLM_parameters_GPU<FPTYPE> * params, GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const std::vector<bool> * isInDataset_trial, const std::vector<size_t> * dim_N_neuron_temp, const bool reset) {

    //check the dims of the destination to see if they hold up
    if(opts->compute_trialLL && results_dest->dim_M() != max_trials()) {
        output_stream << "GPUGMLM_results_GPU::addResults errors: results.dim_M = " << results_dest->dim_M() << " is the incorrect size! (expected dim_M = " << max_trials() << ")";
        msg->callErrMsgTxt(output_stream);
    }
    if(opts->compute_dB && results_dest->dim_B() != dim_B()) {
        output_stream << "GPUGMLM_results_GPU::addResults errors: results.dim_B = " << results_dest->dim_B() << " is the incorrect size! (expected dim_B = " << dim_B() << ")";
        msg->callErrMsgTxt(output_stream);
    }
    if(dim_J() != results_dest->Groups.size()) {
        output_stream << "GPUGMLM_results_GPU::addResults errors: results.dim_J is the incorrect size!";
        msg->callErrMsgTxt(output_stream);
    }
    
    //if reset, set destination memory to all 0's
    if(reset) {
        if(opts->compute_trialLL) {
            results_dest->trialLL->assign(0);
        }
        if(opts->compute_dW) {
            if(!(dW->isEqualSize(results_dest->dW))) {
                output_stream << "GPUGMLM_results_GPU::addResults errors: results.dim_P = " << results_dest->dim_P(msg) << " is the incorrect size! (expected dim_P = " << dim_P() << ")";
                msg->callErrMsgTxt(output_stream);
            }
            results_dest->dW->assign(0);
        }
        if(opts->compute_dB && dim_B() > 0) {
            if(!(dB->isEqualSize(results_dest->dB))) {
                output_stream << "GPUGMLM_results_GPU::addResults errors: results.dim_P = " << results_dest->dim_P(msg) << " is the incorrect size! (expected dim_P = " << dim_P() << ")";
                msg->callErrMsgTxt(output_stream);
            }
            results_dest->dB->assign(0);
        }
    }

    //adds local results to dest

    if(opts->compute_dW) {
        for(unsigned int pp = 0; pp < dim_P(); pp++) {
            if(isSimultaneousPopulation || (*dim_N_neuron_temp)[pp] > 0) {
                (*(results_dest->dW))[pp] += (*dW)[pp];
            }
        }
    }
    if(opts->compute_dB && dim_B() > 0) {
        for(unsigned int pp = 0; pp < dim_P(); pp++) {
            if(isSimultaneousPopulation || (*dim_N_neuron_temp)[pp] > 0) {
                for(unsigned int bb = 0; bb < dim_B(); bb++) {
                    (*(results_dest->dB))(bb, pp) += (*dB)(bb, pp);
                }
            }
        }
    }

    //adds local results to dest
    if(opts->compute_trialLL) {
        for(unsigned int mm = 0; mm < max_trials(); mm++) {
            if((*isInDataset_trial)[mm]) {
                int cols = isSimultaneousPopulation ? dim_P() : 1;
                for(unsigned int pp = 0; pp < cols; pp++) {
                    FPTYPE weight = 1;
                    if(!opts->trial_weights->empty()) {
                        if(!isSimultaneousPopulation || opts->trial_weights->getSize(1) == 1) {
                            weight = (*(opts->trial_weights))[mm];
                        }
                        else {
                            weight = (*(opts->trial_weights))(mm, pp);
                        }
                    }
                    if(weight != 0) {
                        (*(results_dest->trialLL))(mm, pp) += (*trialLL)(mm, pp);
                    }
                }
            }
        }
    }

    for(unsigned int jj = 0; jj < dim_J(); jj++) {
        Groups[jj]->addToHost(params->Groups[jj], results_dest->Groups[jj], opts->Groups[jj], dim_N_neuron_temp, isSimultaneousPopulation, reset);
    }
}

template <class FPTYPE>
void GPUGMLM_results_Group_GPU<FPTYPE>::addToHost(const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, GPUGMLM_group_results<FPTYPE>* results_dest, const GPUGMLM_group_computeOptions * opts, const std::vector<size_t> * dim_N_neuron_temp, bool isSimultaneousPopulation,  const bool reset) {
    //check the dims of the destination to see if they hold up

    
    //if reset, set destination memory to all 0's
    if(reset) {
        if(opts->compute_dV) {
            if(!(dV->isEqualSize(results_dest->dV))) {
                output_stream << "GPUGMLM_results_Group_GPU::addResults errors: results struct is the incorrect size!";
                msg->callErrMsgTxt(output_stream);
            }
            results_dest->dV->assign(0);
        }
        if(results_dest->dim_S() != dim_S()) {
            output_stream << "GPUGMLM_results_Group_GPU::addResults errors: results struct is the incorrect size!";
            msg->callErrMsgTxt(output_stream);
        }
        for(unsigned int ss = 0; ss < dim_S(); ss++) {
            if(opts->compute_dT[ss]) {
                if(!(dT[ss]->isEqualSize(results_dest->dT[ss]))) {
                    output_stream << "GPUGMLM_results_Group_GPU::addResults errors: results struct is the incorrect size!";
                    msg->callErrMsgTxt(output_stream);
                }
                results_dest->dT[ss]->assign(0);
            }
        }
    }

    //adds on results
    //individual model
    if(opts->compute_dV) {
        for(unsigned int pp = 0; pp < parent->dim_P(); pp++) {
            if(isSimultaneousPopulation || (*dim_N_neuron_temp)[pp] > 0) { //if there is anything to add for this neuron
            	for(unsigned int rr = 0; rr < dim_R(); rr++) {
                    (*(results_dest->dV))(pp, rr) += (*dV)(pp, rr);
                }
            }
        }
    }

    for(unsigned int ss = 0; ss < dim_S(); ss++) {
        if(opts->compute_dT[ss]) {
            for(unsigned int tt = 0; tt < dim_T(ss); tt++) {
                for(unsigned int rr = 0; rr < dim_R(); rr++) {
                    (*(results_dest->dT[ss]))(tt, rr) += (*(dT[ss]))(tt, rr);
                }
            }
        }
    }
}



//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================

//explicitly create classes for single and double precision floating point for library
template class GPUGMLM_parameters_Group_GPU<float>;
template class GPUGMLM_parameters_Group_GPU<double>;
template class GPUGMLM_parameters_GPU<float>;
template class GPUGMLM_parameters_GPU<double>;

template class GPUGMLM_results_Group_GPU<float>;
template class GPUGMLM_results_Group_GPU<double>;
template class GPUGMLM_results_GPU<float>;
template class GPUGMLM_results_GPU<double>;
        


};//end namespace