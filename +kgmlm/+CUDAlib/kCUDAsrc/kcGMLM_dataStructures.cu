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
   
//for templating the sparse matrix ops
template <class FPTYPE> cudaDataType_t getCudaType();
template <> cudaDataType_t getCudaType<float>() {
return CUDA_R_32F;
}
template <> cudaDataType_t getCudaType<double>() {
    return CUDA_R_64F;
}  

//============================================================================================================================
//Parameter class
        //constructor
template <class FPTYPE>
GPUGMLM_parameters_GPU<FPTYPE>::GPUGMLM_parameters_GPU(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const size_t dim_M_, const size_t dim_P_, const int dev_, std::shared_ptr<GPUGL_msg> msg_) {
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
        for(int ii = 0; ii < GMLMstructure->logLikeParams.size(); ii++) {
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
    trial_weights_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_, 1);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_weights_temp!" );
    trial_included_temp = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M_, 1);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_included_temp!" );

    trial_weights_0 = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_weights_0!" );
    trial_weights = trial_weights_0;

    trial_included_0 = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_M_, 0);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for trial_included_0!" );
    trial_included = trial_included_0;
    
    //allocate GPU space
    W = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_P_);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for W!" );
    B = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_B, dim_P_);
    checkCudaErrors(ce, "GPUGMLM_parameters_GPU errors: could not allocate space for B!" );

    checkCudaErrors(cudaEventCreate(&paramsLoaded_event), "GPUGMLM_parameters_GPU errors: could not create event!");

    //setup each group
    Groups.resize(GMLMstructure->Groups.size());
    for(int jj = 0; jj < GMLMstructure->Groups.size(); jj++) {
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
    for(int ss = 0; ss < dim_S(); ss++) {
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
    for(int ss = 0; ss < dim_S(); ss++) {
        int dd = (*factor_idx)[ss];
        dF_dT[ss] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, GMLMGroupStructure->dim_T[ss], dim_F_c[dd], GMLMGroupStructure->dim_R_max);
        checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for dF_dT[ss]!" );
    }
    for(int dd = 0; dd < dim_D(); dd++) {
        if((*N_per_factor)[dd] > 1) {
            F[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_F_c[dd], GMLMGroupStructure->dim_R_max);
            checkCudaErrors(ce, "GPUGMLM_parameters_Group_GPU errors: could not allocate space for F[dd]!" );
        }
        else if((*N_per_factor)[dd] == 1) {
            //find the T for this factor if is unique
            for(int ss = 0; ss < dim_S(); ss++) {
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
    for(int jj = 0; jj < Groups.size(); jj++) {
        delete Groups[jj];
    }
}
template <class FPTYPE>
GPUGMLM_parameters_Group_GPU<FPTYPE>::~GPUGMLM_parameters_Group_GPU() {
    switchToDevice();
    for(int dd = 0; dd < N_per_factor->size(); dd++) {
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
        for(int nn = 0; nn < dim_N[mm]; nn++) {
            ridx_sa_all[nn + start_sp] = start_all + nn;
        }
    }
}


//copy all parameters to GPU
template <class FPTYPE>
void GPUGMLM_parameters_GPU<FPTYPE>::copyToGPU(const GPUGMLM_params<FPTYPE> * gmlm_params, GPUGMLM_dataset_GPU<FPTYPE> * dataset, const cudaStream_t stream, const std::vector<cudaStream_t> stream_Groups, const GPUGMLM_computeOptions<FPTYPE> * opts) {
    switchToDevice();

    //copies trial weights if given
    if(opts != NULL && !opts->trial_weights.empty() && opts->trial_weights.size() != dataset->max_trials()) {
        output_stream << "GPUGMLM_parameters_GPU errors: input does not have correct number of trial weights" << std::endl;
        msg->callErrMsgTxt(output_stream);
    }
    bool reset_sizes = false;
    if(opts != NULL && opts->trial_weights.size() != 0) {
        size_t trial_weights_nonzero_cnt_c = 0;
        dataset->dim_N_temp = 0;
        dataset->dim_N_neuron_temp.assign(dim_P(), 0);

        //gets weights for each trial on this GPU block
        for(unsigned int mm = 0; mm < dim_M(); mm++) {
            (*trial_weights_temp)[mm] = opts->trial_weights[(*(dataset->id_t_trial))[mm]];
    
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
    else {
        // this says all trial weights are 1 (normal log likelihood computation)
        trial_weights  = trial_weights_0;
        reset_sizes = true;
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
        for(int jj = 0; jj < dim_J(); jj++) {
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
    for(int ss = 0; ss < dim_S(); ss++) {
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
    for(int ss = 0; ss < dim_S(); ss++) {
        (*compute_dT)[ss] = opts->compute_dT[ss];
        (*compute_dF)[(*factor_idx)[ss]] = (*compute_dF)[(*factor_idx)[ss]] || opts->compute_dT[ss];
    }
    checkCudaErrors(compute_dT->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy compute_dT to device!");
    checkCudaErrors(compute_dF->copyHostToGPU(stream), "GPUGMLM_parameters_Group_GPU errors: could not copy compute_dF to device!");
        
    //copy  to GPU
    checkCudaErrors(V->copyTo(stream, gmlm_group_params->V, false), "GPUGMLM_parameters_Group_GPU errors: could not copy V to device!");

    //copy each T
    for(int ss = 0; ss < dim_S(); ss++) {
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
            for(int rr = 0; rr < F[factor].y; rr++) {
                for(int ss_c = 0; ss_c < dim_S; ss_c++) {
                    if(factor_idx[ss_c] == factor && compute_dT[ss_c]) {
                        for(int tt = 0; tt < dF_dT[ss_c].x; tt++) {
                            dF_dT[ss_c](tt, row, rr) = 0;
                        }
                    }
                }

                size_t T_ctr = 1;
                for(int ss = 0; ss < dim_S; ss++) {
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
                
                for(int ss_c = 0; ss_c < dim_S; ss_c++) {
                    if(factor_idx[ss_c] == factor && compute_dT[ss_c]) {

                        size_t T_ctr = 1;
                        for(int ss = 0; ss < dim_S; ss++) {
                            if(factor_idx[ss] == factor) {
                                size_t tt = (row/T_ctr) % T[ss].x;
                                if(ss != ss_c) {
                                    for(int tt_0 = 0; tt_0 < dF_dT[ss_c].x; tt_0++) {
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
GPUGMLM_results_GPU<FPTYPE>::GPUGMLM_results_GPU(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const size_t max_trials_, const size_t dim_P_, const int dev_, std::shared_ptr<GPUGL_msg> msg_) {
    dev = dev_;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;
    cudaStream_t stream = 0;

    //allocate GPU space for trial weights
    trialLL = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, max_trials_);
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for trialLL!" );

    //allocate GPU space
    dW = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, dim_P_);
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for dW!" );
    
    dB = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMstructure->dim_B, dim_P_);
    checkCudaErrors(ce, "GPUGMLM_results_GPU errors: could not allocate space for dB!" );

    //setup each group
    Groups.resize(GMLMstructure->Groups.size());
    for(int jj = 0; jj < dim_J(); jj++) {
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
    for(int ss = 0; ss < dim_S(); ss++) {
        dT[ss] = new GPUData<FPTYPE>(ce, GPUData_HOST_PAGELOCKED, stream, GMLMGroupStructure->dim_T[ss], GMLMGroupStructure->dim_R_max);
        checkCudaErrors(ce, "GPUGMLM_results_Group_GPU errors: could not allocate space for T[ss]!" );
    }

    
    dF.assign(GMLMGroupStructure->dim_D(msg), NULL);
    dF_assigned.assign(dim_D(), false);

    std::vector<size_t> dim_F_c;
    std::vector<size_t> NF;
    dim_F_c.assign(dim_D(), 1);
    NF.assign(dim_D(), 0);

    for(int ss = 0; ss < dim_S(); ss++) {
        NF[GMLMGroupStructure->factor_idx[ss]]++;
        dim_F_c[GMLMGroupStructure->factor_idx[ss]] *= GMLMGroupStructure->dim_T[ss];
    }
    
    for(int dd = 0; dd < dim_D(); dd++) {
        if(NF[dd] > 1) {
            dF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_F_c[dd], GMLMGroupStructure->dim_R_max);
            checkCudaErrors(ce, "GPUGMLM_results_Group_GPU errors: could not allocate space for dF[dd]!" );
            dF_assigned[dd] = true;
        }
        else {
            dF_assigned[dd] = false;
            for(int ss = 0; ss < dim_S(); ss++) {
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
    for(int jj = 0; jj < Groups.size(); jj++) {
        delete Groups[jj];
    }
}
template <class FPTYPE>
GPUGMLM_results_Group_GPU<FPTYPE>::~GPUGMLM_results_Group_GPU() {
    switchToDevice();
    cudaSafeFreeVector(dT, "GPUGMLM_results_Group_GPU errors: could not free dT");
    for(int dd = 0; dd < dF.size(); dd++) {
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
    for(int jj = 0; jj < Groups.size(); jj++) {
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
    for(int ss = 0; ss < dT.size(); ss++) {
        if(opts->compute_dT[ss]) {
            checkCudaErrors(dT[ss]->copyGPUToHost(stream),"GPUGMLM_results_Group_GPU::gatherResults errors: could not copy dT to host!"); 
        }
    }
}

//adds results in page-locked host memory to user-supplied object for returning
template <class FPTYPE>
void GPUGMLM_results_GPU<FPTYPE>::addToHost(const GPUGMLM_parameters_GPU<FPTYPE> * params, GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const GPUGMLM_dataset_GPU<FPTYPE> * dataset, const bool reset) {

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
    if(opts->compute_trialLL) {
        for(int mm = 0; mm < max_trials(); mm++) {
            if(dataset->isInDataset_trial[mm] && (opts->trial_weights.empty() || opts->trial_weights[mm] != 0)) {
                (*(results_dest->trialLL))[mm] += (*trialLL)[mm];
            }
        }
    }

    if(opts->compute_dW) {
        for(int pp = 0; pp < dim_P(); pp++) {
            if(dataset->dim_N_neuron_temp[pp] > 0) {
                (*(results_dest->dW))[pp] += (*dW)[pp];
            }
        }
    }
    if(opts->compute_dB && dim_B() > 0) {
        for(int pp = 0; pp < dim_P(); pp++) {
            if(dataset->dim_N_neuron_temp[pp] > 0) {
                for(int bb = 0; bb < dim_B(); bb++) {
                    (*(results_dest->dB))(bb, pp) += (*dB)(bb, pp);
                }
            }
        }
    }

    for(int jj = 0; jj < dim_J(); jj++) {
        Groups[jj]->addToHost(params->Groups[jj], results_dest->Groups[jj], opts->Groups[jj], dataset, reset);
    }
}

template <class FPTYPE>
void GPUGMLM_results_Group_GPU<FPTYPE>::addToHost(const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, GPUGMLM_group_results<FPTYPE>* results_dest, const GPUGMLM_group_computeOptions * opts, const GPUGMLM_dataset_GPU<FPTYPE> * dataset, const bool reset) {
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
        for(int ss = 0; ss < dim_S(); ss++) {
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
    if(opts->compute_dV) {
        for(int pp = 0; pp < parent->dim_P(); pp++) {
            if(dataset->dim_N_neuron_temp[pp] > 0) { //if there is anything to add for this neuron
            	for(int rr = 0; rr < dim_R(); rr++) {
                    (*(results_dest->dV))(pp, rr) += (*dV)(pp, rr);
                }
            }
        }
    }

    for(int ss = 0; ss < dim_S(); ss++) {
        if(opts->compute_dT[ss]) {
            for(int tt = 0; tt < dim_T(ss); tt++) {
                for(int rr = 0; rr < dim_R(); rr++) {
                    (*(results_dest->dT[ss]))(tt, rr) += (*(dT[ss]))(tt, rr);
                }
            }
        }
    }
}


//============================================================================================================================
//Dataset class
        
//Constructor takes in all the group data and GMLM setup
template <class FPTYPE>
GPUGMLM_dataset_GPU<FPTYPE>::GPUGMLM_dataset_GPU(const GPUGMLM_structure_args<FPTYPE> * GMLMstructure, const GPUGMLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_, const size_t dim_P_, const cudaStream_t stream, const std::vector<cusparseHandle_t> & cusparseHandle_Groups, std::shared_ptr<GPUGL_msg> msg_) {
    dev = block->dev_num;
    msg = msg_;
    switchToDevice();
    cudaError_t ce;

    log_dt = log(GMLMstructure->binSize);
    Groups.assign(GMLMstructure->Groups.size(), NULL); //sets up dim_J()
    dim_N = new GPUData<size_t>(ce, GPUData_HOST_STANDARD, stream, block->trials.size()); //sets up dim_M()
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate dim_N on device!");
    isInDataset_neuron.assign(dim_P_, false);      // if each neuron has at least one trial in this block, sets up dim_P()
            
    // number of trials
    isInDataset_trial.assign( max_trials_, false); //if each trial is in this block
    if(dim_M() == 0) {
        output_stream << "GPUGMLM_dataset_GPU errors: no trials given to GPU block!";
        msg->callErrMsgTxt(output_stream);
    }

    max_trials_for_sparse_run = min(dim_M()/2, static_cast<size_t>(block->max_trials_for_sparse_run));

    // setup up the order that trials go to the GPU
    //   in blocks ordered by neurons     

    size_t dim_N_total_c = 0;
    dim_N_temp = 0;
    max_trial_length = 1;
    unsigned int tr_ctr = 0; 
    std::vector<int> trial_load_order(dim_M(), 0);

    ridx_n_all = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_P()); 
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_n_all on device!");

    ridx_t_all = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_t_all on device!");
    ridx_n_tr  = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_P());
    checkCudaErrors(ce, "unsigned int errors: could not allocate ridx_n_tr on device!");
    id_t_trial = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate id_t_trial on device!");
    normalizingConstants_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate normalizingConstants_trial on device!");
    id_t_neuron.assign(dim_M(), 0);

    for(int pp = 0; pp < dim_P(); pp++) {
        //find each trial for neuron pp
        size_t dim_N_c = 0;
        (*ridx_n_all)[pp] = dim_N_total_c;
        (*ridx_n_tr)[pp] = tr_ctr;
        for(int mm = 0; mm < dim_M(); mm++) {
            if(block->trials[mm]->neuron == pp) { // is neuron
                //save trial indices
                (*ridx_t_all)[tr_ctr] = dim_N_total_c;
                        
                // get trial length
                (*dim_N)[tr_ctr] = block->trials[mm]->dim_N();
                if((*dim_N)[tr_ctr] == 0) {
                    output_stream << "GPUGMLM_dataset_GPU errors: trials cannot be empty!";
                    msg->callErrMsgTxt(output_stream);
                }
                        
                dim_N_c       += (*dim_N)[tr_ctr]; // add length to current neuron's total
                dim_N_total_c += (*dim_N)[tr_ctr]; // add length to total 
                        
                max_trial_length = max(max_trial_length, (*dim_N)[tr_ctr]); //update max trial length
                
                //save trial and neuron number
                (*id_t_trial)[tr_ctr] = block->trials[mm]->trial_idx;
                id_t_neuron[tr_ctr] = block->trials[mm]->neuron;
                if(isInDataset_trial[block->trials[mm]->trial_idx]) { //trial index already found
                    output_stream << "GPUGMLM_dataset_GPU errors: trial indices must be unique!";
                    msg->callErrMsgTxt(output_stream);
                }
                
                isInDataset_trial[block->trials[mm]->trial_idx] = true;
                isInDataset_neuron[block->trials[mm]->neuron] = true;

                FPTYPE nc = 0; // normalizing constant
                for(int nn = 0; nn < (*dim_N)[tr_ctr]; nn++) {
                    if(GMLMstructure->logLikeSettings == ll_poissExp) {
                        FPTYPE Y_c = (*(block->trials[mm]->Y))[nn];
                        nc += (Y_c >= 0) ? -lgamma(floor(Y_c) + 1.0) : 0;
                    }
                }
                (*normalizingConstants_trial)[tr_ctr] = nc;

                trial_load_order[tr_ctr] = mm;
                tr_ctr++;
            }
        }
        dim_N_neuron.push_back(dim_N_c); // save total bins for the neuron
    }

    dim_N_neuron_temp = std::vector<size_t>(dim_N_neuron);
    id_a_trialM = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_N_total_c);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate id_a_trialM on device!");
    id_a_neuron = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_N_total_c);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate id_a_neuron on device!");

    size_t N_total_ctr = 0;
    for(int mm = 0; mm < dim_M(); mm++) {
        for(int nn = 0; nn < (*dim_N)[mm]; nn++) {
            (*id_a_trialM)[N_total_ctr + nn] = mm;
            (*id_a_neuron)[N_total_ctr + nn] = id_t_neuron[mm];
        }
        N_total_ctr += (*dim_N)[mm];
    }

    //allocate space on GPU for data and copy any local values to GPU
        //spike counts
    Y = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate Y on device!");
    
        //linear term (divded up into per-neuron blocks)
    X_lin = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c, GMLMstructure->dim_B);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate X_lin on device!");
        
        //copy each trial to GPU
    for(int mm = 0; mm < dim_M(); mm++) {
        int mm_0 = trial_load_order[mm];

        // spike counts
        cudaPos copyOffset = make_cudaPos((*ridx_t_all)[mm], 0, 0);
        checkCudaErrors(Y->copyTo(stream, block->trials[mm_0]->Y, true, copyOffset), "GPUGMLM_dataset_GPU errors: could not copy Y to device!");
                
        // linear term
        if(dim_B() > 0) { //don't call if no linear term
            checkCudaErrors(X_lin->copyTo(stream, block->trials[mm_0]->X_lin, true, copyOffset), "GPUGMLM_dataset_GPU errors: could not copy X_lin to device!");
        }
    } 

    //upload vectors to GPU
    checkCudaErrors(normalizingConstants_trial->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy normalizingConstants_trial to device!");
   
    checkCudaErrors(ridx_n_all->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy ridx_n_all to device!");
    checkCudaErrors(ridx_t_all->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy ridx_t_all to device!");
    checkCudaErrors(id_t_trial->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy id_t_trial to device!");
    checkCudaErrors(id_a_trialM->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy id_a_trialM to device!");
    checkCudaErrors(id_a_neuron->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy id_a_neuron to device!");
     
    checkCudaErrors(dim_N->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy dim_N to device!");
    
    checkCudaErrors(ridx_n_tr->copyHostToGPU(stream), "GPUGMLM_dataset_GPU errors: could not copy ridx_n_tr to device!");

    //setup compute space
     LL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate LL on device!");
    dLL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate dLL on device!");

    ridx_sa_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_sa_all on device!");
    ridx_a_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total(), 0);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_a_all on device!");
    ridx_st_sall = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_st_sall on device!");
    ridx_sn_sall = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_P()+1);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate ridx_sn_sall on device!");
    
    lambda = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total(), dim_J());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate lambda on device!");

    X_lin_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, max_trial_length * max_trials_for_sparse_run, dim_B());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate X_lin_temp on device!");
    dW_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_M(), 1);
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate dW_trial on device!");
    dB_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_B(), dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_GPU errors: could not allocate dB_trial on device!");

    //setup the groups
    for(int jj = 0; jj < dim_J(); jj++) {
        Groups[jj] = new GPUGMLM_dataset_Group_GPU<FPTYPE>(jj, GMLMstructure->Groups[jj], block->trials, trial_load_order, this, stream, cusparseHandle_Groups[jj]);
    }
}


template <class FPTYPE>
GPUGMLM_dataset_Group_GPU<FPTYPE>::GPUGMLM_dataset_Group_GPU(const int groupNum_, const GPUGMLM_structure_Group_args<FPTYPE> * GMLMGroupStructure, const std::vector<GPUGMLM_trial_args <FPTYPE> *> trials, const std::vector<int> trial_load_order, const GPUGMLM_dataset_GPU<FPTYPE> * parent_, const cudaStream_t stream, const cusparseHandle_t & cusparseHandle) : parent(parent_), groupNum(groupNum_) {
    dev = parent->dev;
    msg = parent->msg;
    switchToDevice();
    cudaError_t ce;
    
    //sets up dimensions
    X.resize( GMLMGroupStructure->dim_D(msg));
    XF.resize(dim_D());
    iX.resize(dim_D());
    X_temp.resize( dim_D());
    lambda_d.resize( dim_D());

    isShared = new GPUData<bool>(ce, GPUData_HOST_STANDARD, stream, dim_D()); 
    checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate isShared!");
    isSharedIdentity = new GPUData<bool>(ce, GPUData_HOST_STANDARD, stream, dim_D()); 
    checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate isSharedIdentity!");

    dim_A = GMLMGroupStructure->dim_A;

    size_t dim_T_total = 1;
    std::vector<size_t> dim_F_c;
    dim_F_c.assign(dim_D(), 1);
    size_t max_dim_F = 0;
    for(int ss = 0; ss < GMLMGroupStructure->dim_S(); ss++) {
        dim_T_total *= GMLMGroupStructure->dim_T[ss];
        dim_F_c[GMLMGroupStructure->factor_idx[ss]] *= GMLMGroupStructure->dim_T[ss];
        
        max_dim_F = max(max_dim_F,  dim_F_c[GMLMGroupStructure->factor_idx[ss]]);
    }

    if(GMLMGroupStructure->dim_S() == 0 || dim_T_total == 0) {
        output_stream << "GPUGMLM_dataset_Group_GPU errors: tensor has no components!";
        msg->callErrMsgTxt(output_stream);
    }
    if(GMLMGroupStructure->dim_A == 0) {
        output_stream << "GPUGMLM_dataset_Group_GPU errors: tensor has no events/data!";
        msg->callErrMsgTxt(output_stream);
    }
    
    //allocated space for regressors and copy to GPU
    size_t max_dim_X_shared = 0;

    for(int dd = 0; dd < dim_D(); dd++) {
        (*isShared)[dd] = !(GMLMGroupStructure->X_shared[dd]->empty());

        if((*isShared)[dd]) {
            //if shared
            max_dim_X_shared = max(max_dim_X_shared, GMLMGroupStructure->X_shared[dd]->getSize(0));

            //gets depth
            size_t depth = GMLMGroupStructure->X_shared[dd]->getSize(2);
            if(depth != 1) {
                output_stream << "GPUGMLM_dataset_Group_GPU errors: X_shared depth must be 1!";
                msg->callErrMsgTxt(output_stream);
            }

            //allocate space
            X[dd]  = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, GMLMGroupStructure->X_shared[dd]->getSize(0), dim_F_c[dd], depth);
            iX[dd] = new GPUData<int   >(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_A);

            //copy to GPU
            checkCudaErrors(X[dd]->copyTo(stream, GMLMGroupStructure->X_shared[dd], false), "GPUGMLM_dataset_Group_GPU errors: could not copy X[dd] shared to device!");
       
            // copy each trial's data to GPU
            for(int mm = 0; mm < trials.size(); mm++) {
                int mm_0 = trial_load_order[mm];
                
                cudaPos copyOffset = make_cudaPos((*(parent->ridx_t_all))[mm], 0, 0); //get row for current trial
                checkCudaErrors(iX[dd]->copyTo(stream, trials[mm_0]->Groups[groupNum]->iX[dd], true, copyOffset), "GPUGMLM_dataset_Group_GPU errors: could not copy iX[dd] shared to device!");
            }

            //check if X_shared is the identity matrix
            if(X[dd]->getSize(0) == X[dd]->getSize(1)) {
                (*isSharedIdentity)[dd] = true;
                for(int ii = 0; ii < X[dd]->getSize(0) && (*isSharedIdentity)[dd]; ii++) {
                    for(int jj = 0; jj < X[dd]->getSize(1) && (*isSharedIdentity)[dd]; jj++) {
                        if(ii == jj) {
                            (*isSharedIdentity)[dd] = 1 == (*(GMLMGroupStructure->X_shared[dd]))(ii,jj);
                        }
                        else {
                            (*isSharedIdentity)[dd] = 0 == (*(GMLMGroupStructure->X_shared[dd]))(ii,jj);
                        }
                    }
                }
            }
            else {
                (*isSharedIdentity)[dd] = false;
            }

            if(!((*isSharedIdentity)[dd])) {
                //XF comp space
                XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_X(dd), GMLMGroupStructure->dim_R_max);
                checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for XF[dd] shared!" );
            }
            else {
                XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_X(dd), GMLMGroupStructure->dim_R_max, 0); // is empty, but has correct dimensions
                checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for XF[dd] shared+identity!" );
            }

            //X space for sparse runs
            X_temp[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->max_trial_length * parent->max_trials_for_sparse_run, dim_F_c[dd], dim_A, true);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for X_temp[dd]!" );
        }
        else {
            //if local
            (*isSharedIdentity)[dd] = false;

            //gets depth
            size_t depth = trials[0]->Groups[groupNum]->X[dd]->getSize(2);
            if(depth != 1 && depth != dim_A) {
                output_stream << "GPUGMLM_dataset_Group_GPU errors: X_local depth must be dim_A or 1!";
                msg->callErrMsgTxt(output_stream);
            }

            //allocate space
            X[dd]  = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_F_c[dd], depth, true);
            iX[dd] = new GPUData<int   >(ce, GPUData_HOST_NONE, stream, 0, GMLMGroupStructure->dim_A);

            // copy each trial's data
            for(int mm = 0; mm < trials.size(); mm++) {
                int mm_0 = trial_load_order[mm];

                cudaPos copyOffset = make_cudaPos((*(parent->ridx_t_all))[mm], 0, 0); //get row for current trial
                checkCudaErrors(X[dd]->copyTo(stream, trials[mm_0]->Groups[groupNum]->X[dd], true, copyOffset), "GPUGMLM_dataset_Group_GPU errors: could not copy X[dd] local to device!");
            }

            //XF comp space
            XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), GMLMGroupStructure->dim_R_max, depth, true);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for XF[dd] shared!" );

            //X space for sparse runs
            X_temp[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->max_trial_length * parent->max_trials_for_sparse_run, dim_F_c[dd], depth, true);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for X_temp[dd]!" );
        }

    }
    

    checkCudaErrors(isShared->copyHostToGPU(stream), "GPUGMLM_dataset_Group_GPU errors: could not copy isShared to device!");
    checkCudaErrors(isSharedIdentity->copyHostToGPU(stream), "GPUGMLM_dataset_Group_GPU errors: could not copy isSharedIdentity to device!");
             
    //upload host ptrs and dims to GPU
    
    //setup compute space
    lambda_v = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_R_max());
    checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for lambda_v!" );

    // pitched memory for lambda_d: note arrangement is (dim_N_total*dim_A) x dim_R
    //                                this stacks the events to line up with X or S
    lambda_d.assign(dim_D(), NULL);
    for(int dd = 0; dd < dim_D(); dd++) {
        size_t depth = dim_A;
        if(!((*isShared)[dd]) && X[dd]->getSize(2) == 1) {
            depth = 1;
        }
        lambda_d[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_R_max(), depth, true);
        checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for lambda_d!" );
    }

    phi_d =  new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, max_dim_X_shared, dim_R_max());
    checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for phi_d!" );

    dV_trial =  new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_R_max(), parent->dim_M());
    checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for dV_trial!" );

    //setup sparse matrices for dT
    spi_rows.assign(dim_D(), NULL);
    spi_cols.assign(dim_D(), NULL);
    spi_data.assign(dim_D(), NULL);

    spi_S.assign(dim_D(), NULL);
    spi_phi_d.assign(dim_D(), NULL);
    spi_lambda_d.assign(dim_D(), NULL);

    spi_buffer.assign(dim_D(), NULL);
    spi_buffer_size.assign(dim_D(), 0);

    for(int dd = 0; dd < dim_D(); dd++) {
        if((*isShared)[dd]) {
            //gets the rows and cols of the spm in the correct order
                //shorter algorithm is too slow for my level of patience, so we do this in a couple steps
                //first, get valid entries and number of entries per row of spi_S
            size_t ctr = 0;
            std::vector<int> row_ctr;
            row_ctr.resize(dim_X(dd));
            for(int mm = 0; mm < parent->dim_M(); mm++) { //for each trial
                unsigned int mm_0 = trial_load_order[mm];
                for(int aa = 0; aa < dim_A; aa++) { //for each event
                    for(int nn = 0; nn < trials[mm_0]->dim_N(); nn++) { //for each observation
                        //gets the entry in the input data
                        int row = (*(trials[mm_0]->Groups[groupNum]->iX[dd]))(nn, aa);
                        if(row >= 0 && row < dim_X(dd)) { //if valid row (invalid indices are 0's)
                            row_ctr[row]++;
                            ctr++;
                        }
                    }
                }
            }

                //gets to cumulative sum of the rows
            std::vector<int> row_idx;
            row_idx.resize(dim_X(dd));
            row_idx[0] = 0;
            for(int xx = 1; xx < dim_X(dd); xx++) {
                row_idx[xx] = row_ctr[xx-1] + row_idx[xx-1]; 
            }
                //goes back through the indices and adds them on
            spi_rows[dd] = new GPUData<int>(ce, GPUData_HOST_STANDARD, stream, ctr, 1, 1);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for spi_rows[dd]!");
            spi_cols[dd] = new GPUData<int>(ce, GPUData_HOST_STANDARD, stream, ctr, 1, 1);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for spi_cols[dd]!");

            row_ctr.assign(dim_X(dd), 0); //reset row counter
            for(int mm = 0; mm < parent->dim_M(); mm++) { //for each trial
                unsigned int mm_0 = trial_load_order[mm];
                for(int aa = 0; aa < dim_A; aa++) { //for each event
                    for(int nn = 0; nn < trials[mm_0]->dim_N(); nn++) { //for each observation
                        //gets the entry in the input data
                        int row = (*(trials[mm_0]->Groups[groupNum]->iX[dd]))(nn, aa);
                        if(row >= 0 && row < dim_X(dd)) { //if valid row
                            //inserts element
                            size_t entry_num = row_idx[row] + row_ctr[row];
                            (*(spi_cols[dd]))[entry_num] = (*(parent->ridx_t_all))[mm] + nn + aa * parent->dim_N_total();
                            (*(spi_rows[dd]))[entry_num] = row;

                            row_ctr[row]++;
                        }
                    }
                }
            }

            //copy indices to device
            checkCudaErrors(spi_rows[dd]->copyHostToGPU(stream), "GPUGMLM_dataset_Group_GPU errors: could not copy spi_rows[dd] to device!");
            checkCudaErrors(spi_cols[dd]->copyHostToGPU(stream), "GPUGMLM_dataset_Group_GPU errors: could not copy spi_cols[dd] to device!");
            
            spi_data[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, spi_rows[dd]->size(), 1, 1);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for spi_data[dd]!");

            //setup sparse matrix handle
            cusparseStatus_t cusparse_stat;
            spi_S[dd] =  new cusparseSpMatDescr_t;
            cusparse_stat = cusparseCreateCoo(spi_S[dd],
                        dim_X(dd), lambda_d[dd]->getSize(0) * lambda_d[dd]->getSize(2), //num rows, cols
                        spi_rows[dd]->size(), //number of non-zeros
                        spi_rows[dd]->getData_gpu(), //row offsets
                        spi_cols[dd]->getData_gpu(), //col offsets
                        spi_data[dd]->getData_gpu(), //the entries
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        getCudaType<FPTYPE>());
            checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: creating sparse mat spi_S for dT failed.");

            //setup dense handle for lambda_d
            spi_lambda_d[dd] = new cusparseDnVecDescr_t;
            cusparse_stat = cusparseCreateDnVec(spi_lambda_d[dd],
                                                lambda_d[dd]->getSize(0) * lambda_d[dd]->getSize(2),  //size
                                                lambda_d[dd]->getData_gpu(),
                                                getCudaType<FPTYPE>());
            checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: creating dense vec cusparse handle spi_lambda_d failed.");

            //setup dense handle for phi_d
            spi_phi_d[dd] = new cusparseDnVecDescr_t;
            cusparse_stat = cusparseCreateDnVec(spi_phi_d[dd],
                                                dim_X(dd), //size
                                                phi_d->getData_gpu(), //values
                                                getCudaType<FPTYPE>()); //valueType
            checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: creating dense vec cusparse handle spi_phi_d failed.");

            //checks buffer for spi
            size_t buffer;
            FPTYPE alpha = 1;
            FPTYPE beta  = 0;
            cusparse_stat = cusparseSpMV_bufferSize(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    *(spi_S[dd]),
                    *(spi_lambda_d[dd]),
                    &beta,
                    *( spi_phi_d[dd] ),
                    getCudaType<FPTYPE>(),
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    &(buffer));
            checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: getting buffer size for SpMV failed.");

            spi_buffer[dd] = new GPUData<char>(ce, GPUData_HOST_NONE, stream, buffer, 1, 1);
            checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU errors: could not allocate space for spi_buffer[dd]!" );
            spi_buffer_size[dd] = buffer; 
        }
    }


    checkCudaErrors(cudaEventCreate(&group_LL_event), "GPUGMLM_dataset_Group_GPU errors: could not create LL event!");
}

// destructor
template <class FPTYPE>
GPUGMLM_dataset_GPU<FPTYPE>::~GPUGMLM_dataset_GPU() {
    cudaSafeFree(Y, "GPUGMLM_dataset_GPU errors: could not free Y");

    
    cudaSafeFree(X_lin, "GPUGMLM_dataset_GPU errors: could not free X_lin");
    cudaSafeFree(X_lin_temp, "GPUGMLM_dataset_GPU errors: could not free X_lin_temp");
    
    cudaSafeFree(normalizingConstants_trial, "GPUGMLM_dataset_GPU errors: could not free normalizingConstants_trial");
    
    cudaSafeFree(ridx_t_all   , "GPUGMLM_dataset_GPU errors: could not free ridx_t_all");
    cudaSafeFree(ridx_n_all   , "GPUGMLM_dataset_GPU errors: could not free ridx_n_all");
    cudaSafeFree(ridx_st_sall , "GPUGMLM_dataset_GPU errors: could not free ridx_st_sall");
    cudaSafeFree(ridx_sn_sall , "GPUGMLM_dataset_GPU errors: could not free ridx_sn_sall");
    cudaSafeFree(ridx_sa_all  , "GPUGMLM_dataset_GPU errors: could not free ridx_sa_all");
    cudaSafeFree(ridx_a_all  , "GPUGMLM_dataset_GPU errors: could not free ridx_a_all");
    cudaSafeFree(ridx_n_tr, "GPUGMLM_dataset_GPU errors: could not free ridx_n_tr");
    
    cudaSafeFree(id_t_trial , "GPUGMLM_dataset_GPU errors: could not free id_t_trial");
    cudaSafeFree(id_a_trialM, "GPUGMLM_dataset_GPU errors: could not free id_a_trialM");
    cudaSafeFree(id_a_neuron, "GPUGMLM_dataset_GPU errors: could not free id_a_neuron");
    
    cudaSafeFree(dim_N, "GPUGMLM_dataset_GPU errors: could not free dim_N");

    
    cudaSafeFree( LL, "GPUGMLM_dataset_GPU errors: could not free  LL");
    cudaSafeFree(dLL, "GPUGMLM_dataset_GPU errors: could not free dLL");
    cudaSafeFree(lambda, "GPUGMLM_dataset_GPU errors: could not free lambda");
    cudaSafeFree(dW_trial, "GPUGMLM_dataset_GPU errors: could not free dW_trial");
    cudaSafeFree(dB_trial, "GPUGMLM_dataset_GPU errors: could not free dB_trial");

    //clear the groups
    for(auto gg : Groups) {
        delete gg;
    }
}

template <class FPTYPE>
GPUGMLM_dataset_Group_GPU<FPTYPE>::~GPUGMLM_dataset_Group_GPU() {
    checkCudaErrors(cudaEventDestroy(group_LL_event), "GPUGMLM_dataset_Group_GPU errors: could not clear LL event!");
    cudaSafeFreeVector(X, "GPUGMLM_dataset_Group_GPU errors: could not free X[dd]");
    cudaSafeFreeVector(XF, "GPUGMLM_dataset_Group_GPU errors: could not free iX[dd]");
    cudaSafeFreeVector(iX, "GPUGMLM_dataset_Group_GPU errors: could not free iX[dd]");
    cudaSafeFreeVector(X_temp   , "GPUGMLM_dataset_Group_GPU errors: could not free X_temp[dd]");
    
    cudaSafeFree(isShared, "GPUGMLM_dataset_Group_GPU errors: could not free isShared");
    cudaSafeFree(isSharedIdentity, "GPUGMLM_dataset_Group_GPU errors: could not free isSharedIdentity");;

    cudaSafeFree(lambda_v, "GPUGMLM_dataset_Group_GPU errors: could not free lambda_v");
    cudaSafeFreeVector(lambda_d, "GPUGMLM_dataset_Group_GPU errors: could not free lambda_d[dd]");
    cudaSafeFree(   phi_d, "GPUGMLM_dataset_Group_GPU errors: could not free phi_d");
    cudaSafeFree(dV_trial, "GPUGMLM_dataset_Group_GPU errors: could not free dV_trial");

    cudaSafeFreeVector(spi_rows, "GPUGMLM_dataset_Group_GPU errors: could not free spi_rows");
    cudaSafeFreeVector(spi_cols, "GPUGMLM_dataset_Group_GPU errors: could not free spi_cols");
    cudaSafeFreeVector(spi_data, "GPUGMLM_dataset_Group_GPU errors: could not free spi_data");
    cudaSafeFreeVector(spi_buffer, "GPUGMLM_dataset_Group_GPU errors: could not free spi_buffer");
    //destroy any cusparse handles
    for(int dd = 0; dd < spi_S.size(); dd++) {
        if(spi_S[dd] != NULL) {
            checkCudaErrors(cusparseDestroySpMat(*spi_S[dd]), "GPUGMLM_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_S descr.");
            delete spi_S[dd];
        }
        if(spi_phi_d[dd] != NULL) {
            checkCudaErrors(cusparseDestroyDnVec(*spi_phi_d[dd]), "GPUGMLM_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_phi_d descr.");
        	delete spi_phi_d[dd];
        }
        if(spi_lambda_d[dd] != NULL) {
            checkCudaErrors(cusparseDestroyDnVec(*spi_lambda_d[dd]), "GPUGMLM_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_lambda_d descr.");
            delete spi_lambda_d[dd];
        }
    }
}

  

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================
/*Kernel for each observation in a sparse run, for a group
 * Builds the dense regressor matrix with local regressors
*  ridx_sa_all must be assigned
*/
template <class FPTYPE>
__global__ void kernel_getGroupX_local_full(GPUData_kernel<FPTYPE> X_temp, const GPUData_kernel<FPTYPE> X,
                                    const GPUData_kernel<unsigned int> ridx_sa_all) {
    //get current observation number
    unsigned int tt_start = blockIdx.y * blockDim.y + threadIdx.y;
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < ridx_sa_all.x) {
        size_t iX_row;
        iX_row = ridx_sa_all[row];

        //for each event 
        for(unsigned int aa = 0; aa < X_temp.z; aa++) {
            //for each regressor (on this thread)
            for(unsigned int tt = tt_start; tt < X_temp.y; tt += blockDim.y * gridDim.y) {
                X_temp(row, tt, aa) = X(iX_row, tt, aa);
            }
        }
    }
}

//functions to multiply the tensor coefficients by the current parameters
template <class FPTYPE>
void GPUGMLM_dataset_Group_GPU<FPTYPE>::multiplyCoefficients(const bool isSparseRun, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const cudaStream_t stream, const cublasHandle_t cublasHandle, cudaEvent_t & paramsLoaded) {
    checkCudaErrors(set_dim_R(params->dim_R(), stream), "GPUGMLM_dataset_Group_GPU errors: could not set dim_R!");
    if(params->dim_R() == 0) {
        return;
    }
    if(params->dim_R() > dim_R_max()) {
        output_stream << "GPUGMLM_dataset_Group_GPU errors: dim_R too large for pre-allocated space!";
        msg->callErrMsgTxt(output_stream);
    }
    checkCudaErrors(cudaStreamWaitEvent(stream, paramsLoaded), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not wait for event.");
    
    if(isSparseRun) {
        checkCudaErrors(lambda_v->resize(stream, parent->dim_N_temp, -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
    }
    else {
        checkCudaErrors(lambda_v->resize(stream, parent->lambda->getSize_max(0), -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
    }

    for(int dd = 0; dd < dim_D(); dd++) {
        
        GPUData<FPTYPE> * X_c = X[dd];
        if(isSparseRun) {
            checkCudaErrors(X_temp[dd]->resize(  stream, parent->dim_N_temp, -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
            checkCudaErrors(lambda_d[dd]->resize(stream, parent->dim_N_temp, -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
        }
        else {
            checkCudaErrors(lambda_d[dd]->resize(stream, lambda_d[dd]->getSize_max(0),-1,-1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for full runs.");
        }
        if((*isSharedIdentity)[dd]) {
            continue;
        }

        if(isSparseRun && !(*isShared)[dd]) {
            // if sparse run and local regressors, build matrix then multiply
            dim3 block_size;
            if(dim_F(dd) > 8) { 
                block_size.y = 8;
            }
            else if(dim_F(dd) >= 4) { 
                block_size.y = 4;
            }
                block_size.y = 1;
            block_size.x = 1024 / block_size.y;
            dim3 grid_size;
            grid_size.x = parent->dim_N_temp / block_size.x + ((parent->dim_N_temp  % block_size.x == 0)? 0:1);
            grid_size.y = 1;
            kernel_getGroupX_local_full<<<grid_size, block_size, 0, stream>>>(X_temp[dd]->device(), X[dd]->device(), 
                                        parent->ridx_sa_all->device());
            checkCudaErrors("GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors:  kernel_getGroupX_local_full launch failed");

            X_c = X_temp[dd];
        }

        checkCudaErrors(XF[dd]->resize(stream, X_c->getSize(0)), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set matrix size for run.");
            
        checkCudaErrors(X_c->GEMM(XF[dd], params->F[dd], cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors:  X*F -> XF failed");


    }
}

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================

        /*Kernel for each observation, for a group
 * For each component (rr = 0:(dim_R-1)) takes the product of the XT terms into lambda_v, then lambda_v'*V -> lambda 
 * Returns the observation-wise constribution to the rate from this group (lambda) and sets up the dV computation
 *
 * If computing any dT values AND dim_S > 1, needs some dynamic shared memory to make this work on both 1080 and 2080 cards well. Memory size in bytes is dim_S * blockDim.x * sizeof(FPTYPE)
 */
template <class FPTYPE>
__global__ void kernel_getGroupRate(GPUData_kernel<FPTYPE> lambda, 
        GPUData_kernel<FPTYPE> lambda_v, 
        GPUData_array_kernel<FPTYPE, MAX_DIM_D> lambda_d,
        const int col,
        const GPUData_array_kernel<FPTYPE, MAX_DIM_D> XF,
        const GPUData_array_kernel<FPTYPE, MAX_DIM_D> F,
        const GPUData_array_kernel<int, MAX_DIM_D> iX,
        const GPUData_kernel<bool> isShared,
        const GPUData_kernel<bool> isSharedIdentity,
        const GPUData_kernel<FPTYPE> V, 
        const GPUData_kernel<unsigned int> id_a_neuron,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights, 
        const bool compute_dV, const GPUData_kernel<bool> compute_dF, const bool compute_dT_any,
        const GPUData_kernel<unsigned int> ridx_sa_all, const size_t dim_A) {
    //get current observation number
    extern __shared__ int t_array_0[];
    FPTYPE * t_array = (FPTYPE*)t_array_0; // shared memory for derivative setup

    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < lambda.x) {
        size_t iX_row = row; //if full run
        if(ridx_sa_all.y > 0) {
            //if sparse run
            iX_row = ridx_sa_all[row];
        }

        if(trial_weights.y == 0 || trial_weights[id_a_trialM[iX_row]] != 0) { //if trial not censored
            unsigned int neuron_num = id_a_neuron[iX_row];

            //for each rank
            FPTYPE ll = 0;
            for(int rr = 0; rr < V.y; rr++) { //dim_R = V->Y
                //for each event 
                FPTYPE lv = 0;
                const FPTYPE vv = V(neuron_num, rr);

                if(compute_dT_any) {
                    for(unsigned int dd = 0 ; dd < XF.N; dd++) {
                        if(compute_dF[dd]) {
                            for(unsigned int aa = 0; aa < lambda_d[dd].z; aa++) {
                                lambda_d[dd](row, rr, aa) = 0;
                            }
                        }
                    }
                }
                for(unsigned int aa = 0; aa < dim_A; aa++) { //over dim_A
                    FPTYPE lv_aa = 1;
                    //for each factor
                    for(unsigned int dd = 0; dd < XF.N; dd++) { //dim_D = XF->N, dim_S = T->N
                        FPTYPE tc = 0;
                        if(isShared[dd]) { //shared regressors
                            int idx_0 = iX[dd](iX_row, aa);
                            if(idx_0 >= 0) {
                                if(isSharedIdentity[dd]) {
                                    if(idx_0 < F[dd].x) {
                                        tc = F[dd](idx_0, rr);
                                    }
                                }
                                else {
                                    if(idx_0 < XF[dd].x) {
                                        tc = XF[dd](idx_0, rr);
                                    }
                                }
                            }
                        }
                        else  { //local regressors
                            tc = XF[dd](row, rr, aa);
                        }

                        lv_aa *= tc;
                        if(compute_dT_any && XF.N > 1) {
                            t_array[dd + threadIdx.x*XF.N] = tc;
                        }
                        else if(tc == 0 ) {
                            break;
                        }
                        
                    } // dd
                    lv += lv_aa;

                    //sets up any dT matrices (doing this here eliminates the need to go back through the XT matrices in a different kernel)
                    //  I do this outside the previous loop because otherwise everything was super slow on the 1080 cards
                    if(compute_dT_any) {
                        for(unsigned int dd = 0 ; dd < XF.N; dd++) {
                            if(compute_dF[dd]) {
                                FPTYPE tt = vv;
                                for(unsigned int dd2 = 0; dd2 < XF.N; dd2++) {
                                    if(dd2 != dd) {
                                        tt *= t_array[dd2 + threadIdx.x*XF.N];
                                    }
                                }
                                lambda_d[dd](row, rr, aa) += tt;
                            }
                        } //dd
                    }
                } // aa
                if(compute_dV) {
                    lambda_v(row, rr) = lv;
                }
                ll += lv * vv; 
            }
            lambda(row, col) = ll;
        }
    }
}
template <class FPTYPE>
void GPUGMLM_dataset_Group_GPU<FPTYPE>::getGroupRate(const bool isSparseRun, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream) { 

    if(params->dim_R() == 0) {
        // set lambda to 0
        FPTYPE * col = parent->lambda->getData_gpu() + groupNum * parent->lambda->getLD_gpu();
        checkCudaErrors(cudaMemsetAsync(col, 0, parent->lambda->getSize(0)*sizeof(FPTYPE), stream), "GPUGMLM_dataset_Group_GPU::getGroupRate errors: errors setting rate to 0 for dim_R=0 group");
    }
    else {
        dim3 block_size;
        dim3 grid_size;

        block_size.x = 1024; 
        grid_size.x  = parent->lambda->getSize(0) / block_size.x + ((parent->lambda->getSize(0) % block_size.x == 0)? 0:1);

        bool compute_dT_any = false;
        for(int ss = 0; ss < params->dim_S(); ss++) {
            if(opts->compute_dT[ss]) {
                compute_dT_any = true;
                break;
            }
        }

        size_t size_shared = (compute_dT_any && params->dim_D() > 1) ? (sizeof(FPTYPE) * params->dim_D() * block_size.x) : 0;
        kernel_getGroupRate<<<grid_size, block_size, size_shared, stream>>>(parent->lambda->device(), lambda_v->device(), GPUData<FPTYPE>::assembleKernels(lambda_d), groupNum,
                                                                            GPUData<FPTYPE>::assembleKernels(XF), GPUData<FPTYPE>::assembleKernels(params->F), GPUData<int>::assembleKernels(iX),
                                                                            isShared->device(), isSharedIdentity->device(),
                                                                            params->V->device(), 
                                                                            parent->id_a_neuron->device(),
                                                                            parent->id_a_trialM->device(),
                                                                            params->getTrialWeights()->device(),
                                                                            opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                            parent->ridx_a_all_c->device(), dim_A);
        checkCudaErrors("GPUGMLM_dataset_Group_GPU::getGroupRate errors:  kernel_getGroupRate launch failed");
    }
    checkCudaErrors(cudaEventRecord(group_LL_event, stream), "GPUGMLM_dataset_Group_GPU::getGroupRate errors: could not add LL event to stream!");
}

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================

/* Kernel for each entry of the sparse matrix for S*lambda_t -> phi_t (shared regressor compression)
*  sets up the elements of S to be dLL
*/
template <class FPTYPE>
__global__ void kernel_set_spi_S( GPUData_kernel<FPTYPE> S,  const GPUData_kernel<FPTYPE> dLL,
                               const GPUData_kernel<int> S_idx) {
    size_t nn = blockIdx.x * blockDim.x + threadIdx.x;
    if(nn < S.x) {
        S[nn] = dLL[S_idx[nn] % dLL.x];
    }
}

/*Kernel for each observation in a sparse run, for a group
 * Builds the dense regressor matrix times the dLL terms for one dimension with a shared regressor.
*  ridx_sa_all must be assigned
*/
template <class FPTYPE>
__global__ void kernel_getGroupX_shared_full(GPUData_kernel<FPTYPE> X_temp, const GPUData_kernel<FPTYPE> X, const GPUData_kernel<FPTYPE> dLL,
                                    GPUData_kernel<int>  iX,      
                                    GPUData_kernel<unsigned int>  ridx_sa_all,
                                    const bool isIdentity)   {
    //get current observation number
    unsigned int tt_start = blockIdx.y * blockDim.y + threadIdx.y;
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < X_temp.x) {
        size_t iX_row;
        iX_row = ridx_sa_all[row];
        FPTYPE dL_c = dLL[row];

        //for each regressor (on this thread)
        for(unsigned int tt = tt_start; tt < X.y; tt += blockDim.y * gridDim.y) {
            //for each event 
            for(unsigned int aa = 0; aa < iX.y; aa++) {
                int idx_0 = iX(iX_row, aa);
                if(idx_0 < 0 || idx_0 >= X.x) {
                    X_temp(row, tt, aa) = 0;
                }
                else {
                    if(isIdentity) {
                        X_temp(row, tt, aa) = (idx_0 == tt) ?  dL_c : 0;
                    }
                    else {
                        X_temp(row, tt, aa) = X(idx_0, tt) * dL_c;
                    }
                }
            }
        }
    }
}


/* Kernel for each observation that sets up lambda_t with dLL
*/
template <class FPTYPE>
__global__ void kernel_dLL_mult(GPUData_kernel<FPTYPE> lambda_d, const GPUData_kernel<FPTYPE> dLL) {
    //get current observation number
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < dLL.x) {
        //for each regressor (on this thread)
        for(int rr = 0; rr < lambda_d.y; rr++) {
            //for each event 
            for(int aa = 0; aa < lambda_d.z; aa++) {
                lambda_d(row, rr, aa) *= dLL[row]; 
            }
        }
    }                             
}

template <class FPTYPE>
void GPUGMLM_dataset_Group_GPU<FPTYPE>::computeDerivatives(GPUGMLM_results_Group_GPU<FPTYPE> * results, const bool isSparseRun, GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle, const cusparseHandle_t cusparseHandle, cudaEvent_t & main_LL_event) {
    if(params->dim_R() == 0) {
        return; //nothing to compute
    }

    checkCudaErrors(cudaStreamWaitEvent(stream, main_LL_event, 0), "GPUGMLM_dataset_Group_GPU::computeDerivatives errors: could not wait for stream");

    if(opts->compute_dV) {
        //for each neuron
        FPTYPE alpha = 1;
        FPTYPE beta  = 0;
        for(int pp = 0; pp < parent->dim_P(); pp++) {// if there are any trials for this neuron in current GPU block
            if(parent->dim_N_neuron_temp[pp] > 0) {
                unsigned int neuron_start_c = (*(parent->ridx_n_all))[pp];
                if(isSparseRun) {
                    // sparse run
                    neuron_start_c = (*(parent->ridx_sn_sall))[pp];     
                }

                cublasStatus_t ce = cublasGEMV(cublasHandle,
                                               CUBLAS_OP_T,
                                               parent->dim_N_neuron_temp[pp], params->dim_R(),
                                               &alpha,
                                               lambda_v->getData_gpu()   + neuron_start_c, lambda_v->getLD_gpu(),
                                               parent->dLL->getData_gpu() + neuron_start_c, 1,
                                               &beta,
                                               results->dV->getData_gpu() + pp, results->dV->getLD_gpu());
                checkCudaErrors(ce, "GPUGMLM_dataset_Group_GPU::computeDerivatives errors:  lambda_v'*dLL -> dV failed");
            }
        }
    }

    //check if computing any derivatives first
    std::vector<bool> compute_dF;
    compute_dF.assign(dim_D(), false);
    for(int ss = 0; ss < params->dim_S(); ss++) {
        unsigned int dd = (*(params->factor_idx))[ss];
        compute_dF[dd] = compute_dF[dd] || opts->compute_dT[ss];
    }

    //for each factor
    for(int dd = 0; dd < dim_D(); dd++) {
        if(compute_dF[dd]) {
            // lambda_t setup in the kernel call in computeRateParts 
            //matrix mult of X'*(dLL .* lambda_t)
            GPUData<FPTYPE> * phi_c;
            GPUData<FPTYPE> * X_c;
            if((*isShared)[dd] && isSparseRun) { 
                //  if doing sparse run with shared regressor, builds temporary X matrix
                dim3 block_size;
                block_size.y = 1;
                block_size.x = 1024 / block_size.y;
                dim3 grid_size;
                grid_size.x = parent->dim_N_temp  / block_size.x + ((parent->dim_N_temp  % block_size.x == 0)? 0:1);
                grid_size.y = 1;
                

                kernel_getGroupX_shared_full<<<grid_size, block_size, 0, stream>>>(X_temp[dd]->device(), X[dd]->device(), parent->dLL->device(),
                                            	iX[dd]->device(), 
                                                parent->ridx_sa_all->device(),
                                                (*isSharedIdentity)[dd]);
                checkCudaErrors("GPUGMLM_dataset_Group_GPU::computeDerivatives errors:  kernel_getGroupX_shared_full launch failed");

                X_c   = X_temp[dd];
                phi_c = lambda_d[dd];
            }
            else if(!((*isShared)[dd]) && isSparseRun) { 
                // if local regressors and sparse run
                //          multiply diag(dLL) x lambda_t for each event
                dim3 block_size;
                block_size.x = 1024;
                dim3 grid_size;
                grid_size.x = parent->dim_N_temp / block_size.x + ((parent->dim_N_temp % block_size.x == 0)? 0:1);

                kernel_dLL_mult<<<grid_size, block_size, 0, stream>>>(lambda_d[dd]->device(), parent->dLL->device());
                checkCudaErrors("GPUGMLM_dataset_Group_GPU::computeDerivatives errors:  diag(dLL)*lambda_d -> lambda_d failed");

                X_c     = X_temp[dd];
                phi_c = lambda_d[dd];
            }
            else if((*isShared)[dd]) { // only do this if doing full run
                //this step is faster with sparse matrices for shared regressors

                //call kernel to setup entries spi to dLL 
                dim3 block_size;
                block_size.x = 1024;
                dim3 grid_size;
                grid_size.x = spi_rows[dd]->size()/ block_size.x + ((spi_rows[dd]->size() % block_size.x == 0)? 0:1);

                kernel_set_spi_S<<<grid_size, block_size, 0, stream>>>(spi_data[dd]->device(), parent->dLL->device(),
                                                     spi_cols[dd]->device());
                checkCudaErrors("GPUGMLM_dataset_Group_GPU::computeDerivatives errors:  kernel_set_spi_S launch failed");

                FPTYPE alpha = 1;
                FPTYPE beta  = 0;
                for(int rr = 0; rr < params->dim_R(); rr++) {
                    //I found - on a 1080ti at least - doing this series of SpMV ops was typically faster than a single SpMM (annoyingly)
                    cusparseStatus_t cusparse_stat;
                    cusparse_stat = cusparseDnVecSetValues(*(spi_lambda_d[dd]), lambda_d[dd]->getData_gpu() + rr*lambda_d[dd]->getLD_gpu());
                    checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: cusparseDnVecSetValues failed for lambda_t.");
                    if((*isSharedIdentity)[dd]) {
                        cusparse_stat = cusparseDnVecSetValues(*(spi_phi_d[dd]), results->dF[dd]->getData_gpu() + rr*results->dF[dd]->getLD_gpu());
                    }
                    else {
                        cusparse_stat = cusparseDnVecSetValues(*(spi_phi_d[dd]), phi_d->getData_gpu() + rr*phi_d->getLD_gpu());
                    }
                    checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: cusparseDnVecSetValues failed for phi_d.");
                       
                    cusparse_stat = cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 *(spi_S[dd]),
                                 *(spi_lambda_d[dd]),
                                 &beta,
                                 *(spi_phi_d[dd]),
                                 getCudaType<FPTYPE>(),
                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                 spi_buffer[dd]->getData_gpu());
                    checkCudaErrors(cusparse_stat, "GPUGMLM_dataset_Group_GPU errors: S*lambda->phi_t SpMV failed.");
                
                  /* OLD VERSION: used SpMM: but as with cublasGEMM, these matrix ops with tall, skinny matrices are often faster as a group of MV ops 
                                  That's why this loop is here.*/
                }

                X_c   = X[dd];
                phi_c = phi_d;
            }
            else {
                //     if local regressors and full run
                //          multiply diag(dLL) x lambda_t for each event
                dim3 block_size;
                block_size.x = 1024;
                dim3 grid_size;
                grid_size.x = parent->dim_N_total() / block_size.x + ((parent->dim_N_total() % block_size.x == 0)? 0:1);

                kernel_dLL_mult<<<grid_size, block_size, 0, stream>>>(lambda_d[dd]->device(), parent->dLL->device());
                checkCudaErrors("GPUGMLM_dataset_Group_GPU::computeDerivatives errors:  diag(dLL)*lambda_d -> lambda_d failed");

                X_c   = X[dd];
                phi_c = lambda_d[dd];
            }

            checkCudaErrors(phi_c->resize(stream, X_c->getSize(0), results->dF[dd]->getSize(1), X_c->getSize(2)), "GPUGMLM_dataset_Group_GPU::computeDerivatives errors: setting size of phi_c failed");

            /*output_stream << " dd = " << dd << "  ";
            phi_c->printInfo(output_stream, "phi_c");
            X_c->printInfo(output_stream, "X_c");
            results->dF[dd]->printInfo(output_stream, "results->dF[dd]");
            msg->printMsgTxt(output_stream);*/

            // matrix mult to get dF (local and shared)
            if((*isShared)[dd] && !isSparseRun && (*isSharedIdentity)[dd]) {
                //nothing needed
            }
            else {
                checkCudaErrors(X_c->GEMM(results->dF[dd], phi_c, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N), "GPUGMLM_dataset_Group_GPU::computeDerivatives errors:   X'*phi -> dF");
                //output_stream << " GEMM " << num << "\n";
                //msg->printMsgTxt(output_stream);
            }
            
            // matrix mults to get dT
            if((*(params->N_per_factor))[dd] > 1) {
                for(int ss = 0; ss < params->dim_S(); ss++) {
                    if((*(params->factor_idx))[ss] == dd && opts->compute_dT[ss]) {
                        checkCudaErrors(params->dF_dT[ss]->GEMVs(results->dT[ss], results->dF[dd], cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N), "GPUGMLM_dataset_Group_GPU::computeDerivatives errors: dF_dT'*dF -> dT");
                    }
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
        
template class GPUGMLM_dataset_Group_GPU<float>;
template class GPUGMLM_dataset_Group_GPU<double>;
template class GPUGMLM_dataset_GPU<float>;
template class GPUGMLM_dataset_GPU<double>;


};//end namespace