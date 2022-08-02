/*
 * kcGMLMPop_computeBlock.cu
 * Computations for a GMLMPop+derivatives (on one GPU).
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
#include "kcGMLMPop_computeBlock.hpp"

namespace kCUDA {
    
template <class FPTYPE>
GPUGMLMPop_computeBlock<FPTYPE>::GPUGMLMPop_computeBlock(const GPUGMLM_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLM_GPU_block_args<FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_) {
    this->msg = msg_;
    this->dev  = block->dev_num;
    this->switchToDevice();
    this->checkDeviceComputeCapability();
    dim_J = GMLMPopstructure->Groups.size();

    size_t dim_M = block->trials.size();
    if(dim_M == 0) {
        this->output_stream << "GPUGMLMPop_computeBlock errors: no trials in block!";
        this->msg->callErrMsgTxt(this->output_stream);
    }   

    //setup the streams
    this->checkCudaErrors(cudaStreamCreate(&(stream)), "GPUGMLMPop_computeBlock errors: failed initializing stream!");
    stream_Groups.resize(dim_J);
    for(unsigned int jj = 0; jj < dim_J; jj++) {
        this->checkCudaErrors(cudaStreamCreate(&(stream_Groups[jj])), "GPUGMLMPop_computeBlock errors: failed initializing group streams!");
    }

    //setup cublas handles
    cublasMath_t mathMode = CUBLAS_DEFAULT_MATH;
    #if __CUDA_ARCH__ >= 700
        mathMode = CUBLAS_TF32_TENSOR_OP_MATH;
    #endif
    
    this->checkCudaErrors(cublasCreate(&(cublasHandle)), "GPUGMLMPop_computeBlock errors: CUBLAS initialization failed.");
    this->checkCudaErrors(cublasSetStream(cublasHandle, stream), "GPUGMLMPop_computeBlock errors: set cublas stream failed.");
    this->checkCudaErrors(cublasSetMathMode(cublasHandle, mathMode), "GPUGMLMPop_computeBlock errors: set cublas math mode failed.");
    this->checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cublas pointer mode failed.");

    cublasWorkspace = NULL;
    /*size_t cublasWorkspace_size_0 = 1024 * 1024 * 0;	// if greater than 0, sets special workspace size (doesn't seem to help the current computations)	
    if(cublasWorkspace_size_0 > 0) {
        this->checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&(cublasWorkspace)), &cublasWorkspace_size, cublasWorkspace_size_0, 1), "GPUGMLMPop_computeBlock errors: allocating cublas workspace failed.");
        this->checkCudaErrors(cublasSetWorkspace(cublasHandle, cublasWorkspace, cublasWorkspace_size), "GPUGMLMPop_computeBlock errors: setting CUBLAS workspace failed.");
    }*/


    cublasHandle_Groups.resize(dim_J);
    cublasWorkspaces.assign(dim_J, NULL);
    cublasWorkspaces_size.assign(dim_J, cublasWorkspace_size);
    for(unsigned int jj = 0; jj < dim_J; jj++) {
        this->checkCudaErrors(cublasCreate(&(cublasHandle_Groups[jj])), "GPUGMLMPop_computeBlock errors: CUBLAS groups initialization failed.");
        this->checkCudaErrors(cublasSetMathMode(cublasHandle_Groups[jj], mathMode), "GPUGMLMPop_computeBlock errors: set cublas group math mode failed.");
        this->checkCudaErrors(cublasSetPointerMode(cublasHandle_Groups[jj], CUBLAS_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cublas groups pointer mode failed.");
        this->checkCudaErrors(cublasSetStream(cublasHandle_Groups[jj], stream_Groups[jj]), "GPUGMLMPop_computeBlock errors: set cublas groups stream failed.");

     /*   if(cublasWorkspaces_size[jj] > 0) {
            this->checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&(cublasWorkspaces[jj])), &cublasWorkspaces_size[jj], cublasWorkspace_size_0, 1), "GPUGMLMPop_computeBlock errors: allocating group cublas workspace failed.");
            this->checkCudaErrors(cublasSetWorkspace(cublasHandle_Groups[jj], cublasWorkspaces[jj], cublasWorkspaces_size[jj]), "GPUGMLMPop_computeBlock errors: setting group CUBLAS workspace failed.");
        }*/
    }

    //setup cusparse handle
    cusparseHandle_Groups.resize(dim_J);
    for(unsigned int jj = 0; jj < dim_J; jj++) {
        this->checkCudaErrors(cusparseCreate(       &(cusparseHandle_Groups[jj])), "GPUGMLMPop_computeBlock errors: cusparse groups initialization failed.");
        this->checkCudaErrors(cusparseSetPointerMode(cusparseHandle_Groups[jj], CUSPARSE_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cusparse groups pointer mode failed.");
        this->checkCudaErrors(cusparseSetStream(      cusparseHandle_Groups[jj], stream_Groups[jj]), "GPUGMLMPop_computeBlock errors: set cusparse groups stream failed.");
    }

    //setup the parameter structure
    params = new GPUGMLM_parameters_GPU<FPTYPE>(GMLMPopstructure, dim_M, this->dev, this->msg);
    //params = NULL;

    //setup the results structure
    results = new GPUGMLM_results_GPU<FPTYPE>(GMLMPopstructure, max_trials_, this->dev, this->msg);
    //results = NULL;
            
    //setup the dataset structure
    dataset = new GPUGMLMPop_dataset_GPU<FPTYPE>(GMLMPopstructure, block, max_trials_, stream, cusparseHandle_Groups, this->msg);
    //dataset = NULL;
    this->checkCudaErrors(cudaEventCreate(&LL_event), "GPUGMLMPop_computeBlock errors: could not create LL event!");
}

template <class FPTYPE>
GPUGMLMPop_computeBlock<FPTYPE>::~GPUGMLMPop_computeBlock() {
    this->switchToDevice();
    
    delete results;
    delete params;
    delete dataset;

    this->checkCudaErrors(cudaEventDestroy(LL_event), "GPUGMLMPop_computeBlock errors: could not clear LL event!");

    //destroy cublas handles
    this->checkCudaErrors(cublasDestroy(cublasHandle), "GPUGMLMPop_computeBlock errors: failed to destroy cublas handle." );
    for(auto jj : cublasHandle_Groups) {
        this->checkCudaErrors(cublasDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group cublas handles." );
    }
    for(auto jj : cusparseHandle_Groups) {
        this->checkCudaErrors(cusparseDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group cusparse handles." );
    }
       
    this->cudaSafeFreePtr(cublasWorkspace, "GPUGMLMPop_computeBlock errors: failed to destroy cublas workspace." );
    this->cudaSafeFreePtrVector(cublasWorkspaces, "GPUGMLMPop_computeBlock errors: failed to destroy cublas group workspaces." );
    //destroy streams
    this->checkCudaErrors(cudaStreamDestroy(stream), "GPUGMLMPop_computeBlock errors: failed destroying stream!");
    for(auto jj : stream_Groups) {
        this->checkCudaErrors(cudaStreamDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group streams." );
    }  
}

template <class FPTYPE>
bool GPUGMLMPop_computeBlock<FPTYPE>::loadParams(const GPUGMLM_params<FPTYPE> * params_host, const GPUGMLM_computeOptions<FPTYPE> * opts) { 
    this->switchToDevice();
    params->copyToGPU(params_host, dataset, stream, stream_Groups, opts);
    for(unsigned int jj = 0; jj < params->dim_J(); jj++) {
        this->checkCudaErrors(results->set_dim_R(jj, params->dim_R(jj), stream), "GPUGMLMPop_computeBlock::loadParams errors: could not set results dim_R");
    }
    bool isSparseRun = dataset->isSparseRun(params);
    if(params->getNumberOfNonzeroWeights() > 0) { //make sure there's something to compute
        results_set = true;

        //for each group, multiply coefficients by X*T -> XT
        for(unsigned int jj = 0; jj < dim_J && jj < dataset->dim_J(); jj++) {
            dataset->Groups[jj]->multiplyCoefficients(isSparseRun, opts->update_weights, params->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj], params->paramsLoaded_event);
        }
    }
    else {
        results_set = false;
    }
    return isSparseRun;
}
        
template <class FPTYPE>
void GPUGMLMPop_computeBlock<FPTYPE>::computeRateParts(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    this->switchToDevice();

    //for each group
    for(unsigned int jj = 0; jj < dataset->dim_J(); jj++ ) {
        dataset->Groups[jj]->getGroupRate(isSparseRun,  params->Groups[jj], opts->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj]);
    }
}


/*Kernel for each observation
 *  for sparse runs with compute_dB, saves out the partial X_lin into X_lin_temp
 */        
template <class FPTYPE>
__global__ void kernel_setup_X_lin_temp(
        const GPUData_kernel<FPTYPE> X_lin ,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights,
        const GPUData_kernel<unsigned int> ridx_sa_all,
        GPUData_kernel<FPTYPE> X_lin_temp) {
    //current observation index

    for(size_t row_0 = blockIdx.x * blockDim.x; row_0 < X_lin_temp.x; row_0 += blockDim.x * gridDim.x) {
        size_t row = row_0 + threadIdx.x;
        size_t Xlin_row = row; //if full run
        if(ridx_sa_all.y > 0 && row < ridx_sa_all.x) {
            //if sparse run
            Xlin_row = ridx_sa_all[row];
        }
        FPTYPE tw_c = 1;
        if(row < X_lin_temp.x && trial_weights.y == 1) {
            unsigned int tr_idx = id_a_trialM[Xlin_row];
            if(trial_weights.x > tr_idx) {
                tw_c = trial_weights[tr_idx];
            }
        }
        __syncthreads();
        for(unsigned int pp_0 = blockIdx.y * blockDim.y; pp_0 < X_lin_temp.z; pp_0 += blockDim.y * gridDim.y) {
            unsigned int pp = pp_0 + threadIdx.y;

            if(X_lin_temp.z > 1 && row < X_lin_temp.x && trial_weights.y > 1 && pp < trial_weights.y) {
                unsigned int tr_idx = id_a_trialM[Xlin_row];
                if(trial_weights.x > tr_idx) {
                    tw_c = trial_weights(tr_idx, pp);
                }
            }

            bool elementIncluded = row < X_lin_temp.x && pp < X_lin_temp.z && tw_c != 0;

            __syncwarp();

            for(int bb = 0; bb < X_lin.y; bb++) {
                if(elementIncluded) {
                    if(ridx_sa_all.y > 0 && (pp == 0 || X_lin_temp.z > 1)) { // for dB when doing sparse run
                        X_lin_temp(row, bb, pp) = X_lin(Xlin_row, bb, pp);
                    }
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
}

/*Kernel for each observation
 * Summarizes the contributions from each tensor group (lambda), linear term (X_lin,B), baseline rate (w,log_dt)
 * Returns the  observation-wise log like (LL - no normalizing constant) and it's derivative portion (dLL)
 *
 */        
template <class FPTYPE>
__global__ void kernel_getObs_LL_pop(GPUData_kernel<FPTYPE> LL, GPUData_kernel<FPTYPE> dLL,
        const GPUData_kernel<FPTYPE> Y,
        const GPUData_kernel<FPTYPE> lambda,
        bool addW,
        const GPUData_kernel<FPTYPE> W, 
        const FPTYPE log_dt, const FPTYPE dt,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights,
        const GPUData_kernel<unsigned int> ridx_sa_all,
        const logLikeType logLikeSettings, const GPUData_kernel<FPTYPE> logLikeParams) {
    //current observation index

    for(size_t row_0 = blockIdx.x * blockDim.x; row_0 < LL.x; row_0 += blockDim.x * gridDim.x) {
        size_t row = row_0 + threadIdx.x;
        size_t Xlin_row = row; //if full run
        if(ridx_sa_all.y > 0 && row < ridx_sa_all.x) {
            //if sparse run
            Xlin_row = ridx_sa_all[row];
        }
        FPTYPE tw_c = 1;
        if(row < LL.x && trial_weights.y == 1) {
            unsigned int tr_idx = id_a_trialM[Xlin_row];
            if(trial_weights.x > tr_idx) {
                tw_c = trial_weights[tr_idx];
            }
        }
        __syncthreads();
        for(unsigned int pp_0 = blockIdx.y * blockDim.y; pp_0 < LL.y; pp_0 += blockDim.y * gridDim.y) {
            unsigned int pp = pp_0 + threadIdx.y;
            if(row < LL.x && trial_weights.y > 1 && pp < trial_weights.y) {
                unsigned int tr_idx = id_a_trialM[Xlin_row];
                if(trial_weights.x > tr_idx) {
                    tw_c = trial_weights(tr_idx, pp);
                }
            }

            bool elementIncluded = row < LL.x && pp < LL.y && tw_c != 0;
        
            FPTYPE  LL_c = 0;  
            FPTYPE dLL_c = 0;
            FPTYPE log_rate = 0; 
            FPTYPE Y_c  = 0;   

            if(elementIncluded) {
                Y_c = Y(Xlin_row, pp);
                if(!addW) {
                    log_rate = W[pp];
                }
                else {
                    log_rate = W[pp] + LL(row, pp);
                }
            }
            __syncwarp();

            
            for(int jj = 0; jj < lambda.z; jj++) {
                if(elementIncluded) {
                    log_rate += lambda(row, pp, jj);
                }
                __syncwarp();
            }
            __syncthreads();


            if(elementIncluded) {
                if(logLikeSettings == ll_poissExp) {
                    if(Y_c >= 0) { // negatives get censored by Poisson LL
                        Y_c = floor(Y_c);
                        log_rate += log_dt;
                        FPTYPE rate = safeExp(log_rate);
                        LL_c = (-rate + Y_c * log_rate);
                        dLL_c = (-rate + Y_c);
                    }
                }
                else if(logLikeSettings == ll_poissSoftRec) {
                    if(Y_c >= 0) { // negatives get censored by Poisson LL
                        Y_c = floor(Y_c);
                        FPTYPE rate;
                        FPTYPE drate;
                        FPTYPE drate_rate;
                        if(log_rate > 30) {
                            rate  = log_rate ; // in this model, log_dt is actually just dt
                            drate = 1;
                            drate_rate = 1.0/log_rate;
                        }
                        else {
                            log_rate = log_rate < -30 ? -30 : log_rate; // to be safe with the log
                            rate  = log1p(safeExp(log_rate));
                            drate = (1.0 + safeExp(-log_rate));
                            drate_rate = 1.0/(drate * rate);
                            drate = 1.0/drate;
                        }
                        LL_c  = (-rate*dt + Y_c *(log(rate) + log_dt));
                        dLL_c = (-drate*dt + Y_c * drate_rate);
                    }
                }
                else if(logLikeSettings == ll_sqErr) {
                    FPTYPE eY_c = log_rate - Y_c;
                    LL_c = -0.5*(eY_c*eY_c);
                    dLL_c = -eY_c;
                }
                else if(logLikeSettings == ll_truncatedPoissExp) {
                    if(Y_c >= 1) { 
                        log_rate += log_dt;
                        if(log_rate > -30) {
                            FPTYPE rate = safeExp(log_rate);
                            LL_c = log(1.0 - safeExp(-rate));
                            dLL_c = rate/safeExpm1(rate);
                        }
                        else { // more numerically save approximation in an extreme case
                            LL_c = log_rate;
                            dLL_c = 1;
                        }
                    }
                    else if(Y_c == 0) {
                        FPTYPE rate = safeExp(log_rate + log_dt);
                        LL_c = -rate;
                        dLL_c = -rate;
                    }
                    // negatives get censored by Poisson LL
                }
                else if(logLikeSettings == ll_poissExpRefractory) {
                    // ll_poissExpRefractory uses the correction from Citi, L., Ba, D., Brown, E. N., & Barbieri, R. (2014). Likelihood methods for point processes with refractoriness. Neural computation, 26(2), 237-263.
                    if(Y_c >= 0) { // negatives get censored by Poisson LL
                        Y_c = floor(Y_c);
                        log_rate += log_dt;
                        FPTYPE rate = safeExp(log_rate);
                        LL_c = (-(1-Y_c/2)*rate + Y_c * log_rate);
                        dLL_c = (-(1-Y_c/2)*rate + Y_c);
                    }
                }

                LL(row, pp) =  LL_c*tw_c;
                dLL(row, pp) = dLL_c*tw_c;
            }
            else if(row < LL.x && pp < LL.y) {
                LL(row, pp) = 0;
                dLL(row, pp) = 0;
            }
            __syncthreads();
        }
    }
}

/* Kernel for each trial
*  Sums up the trial log likelihoods (results->trialLL)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)*/
template <class FPTYPE>
__global__ void kernel_sum_trialLL_pop(GPUData_kernel<FPTYPE> trialLL, GPUData_kernel<FPTYPE> dW_trial, 
                                 const GPUData_kernel<unsigned int> trial_included, 
                                 const GPUData_kernel<FPTYPE> LL, const GPUData_kernel<FPTYPE> dLL, 
                                 const bool compute_trialLL, const bool compute_dW, 
                                 const GPUData_kernel<size_t> dim_N,
                                 const GPUData_kernel<unsigned int> ridx_t_all,
                                 const GPUData_kernel<unsigned int> id_t_trial,
                                 const GPUData_kernel<FPTYPE> trial_weights,
                                 const GPUData_kernel<FPTYPE> normalizingConstants) {
    size_t tr = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pp = blockIdx.y * blockDim.y + threadIdx.y;
    size_t mm = dim_N.x; //default is invalid value - will just skip
    if(trial_included.y > 0) { //if is sparse run
        if(tr < trial_included.x) {
            mm = trial_included[tr];
        }
    }
    else {
        mm = tr;
    }

    if(mm < dim_N.x && pp < trialLL.y) { // if valid trial
        FPTYPE tw_c = 1;
        if(trial_weights.y == 1 && trial_weights.x > mm) {
            tw_c = trial_weights[mm];
        }
        else if(trial_weights.y > 1 && trial_weights.x > mm) {
            tw_c = trial_weights(mm, pp);
        }

        if(tw_c != 0) {  
            unsigned int row = ridx_t_all[tr];  // this uses 'tr' so that it works for sparse runs

            //sum up LL
            FPTYPE ll_total = normalizingConstants(mm, pp) * tw_c;
            FPTYPE dll_total = 0;
            for(int tt = 0; tt < dim_N[mm]; tt++) {
                if(compute_trialLL) {
                    ll_total  += LL(row + tt, pp);
                }
                if(compute_dW) {
                    dll_total  += dLL(row + tt, pp);
                }
            }
            if(compute_trialLL) {
                trialLL(id_t_trial[mm], pp) = ll_total;
            }
            if(compute_dW) {
                dW_trial(mm, pp) = dll_total;
            }
        }
        //no need to compute sum; set results to 0
        else {
            if(compute_trialLL) {
                trialLL(id_t_trial[mm], pp) = 0;
            }
            if(compute_dW) {
                dW_trial(mm, pp) = 0;
            }
        }
    }
}

template <class FPTYPE>
void GPUGMLMPop_computeBlock<FPTYPE>::computeLogLike(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }

    this->switchToDevice();
         //launch kernel to sum lambda, X_lin*B -> LL, dLL (launch over all observations)
         //launch kernel to sum lambda for each trial

         
    GPUData<FPTYPE> * X_lin_c = isSparseRun ?  dataset->X_lin_temp : dataset->X_lin;

    if(isSparseRun && opts->update_weights) {
        dataset->X_lin_temp->resize(stream, dataset->LL->getSize(0));

        dim3 block_size;
        block_size.x = 1024/block_size.y;
        dim3 grid_size;
        size_t max_blocks_needed  = dataset->LL->getSize(0) / block_size.x + ( (dataset->LL->getSize(0) % block_size.x == 0) ? 0 : 1);
        size_t blocks_to_use = 1024;
        grid_size.x  = min(max_blocks_needed, blocks_to_use);
        grid_size.y = dataset->X_lin_temp->getSize(2) / block_size.y + ( (dataset->X_lin_temp->getSize(2) % block_size.y == 0) ? 0 : 1);
        kernel_setup_X_lin_temp<<<grid_size, block_size, 0, stream>>>( dataset->X_lin->device(),
                dataset->id_a_trialM->device(),
                params->trial_weights->device(),
                dataset->ridx_a_all_c->device(),
                dataset->X_lin_temp->device());
    }

    //X_lin*B -> LL
    if(dataset->dim_B() > 0) {
        cublasStatus_t ce;
        if(X_lin_c->getInc_gpu() == 0) {
            ce = X_lin_c->GEMM(dataset->LL, params->B, cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N);
        }       
        else {
            ce = X_lin_c->GEMVs(dataset->LL, params->B, cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N);
        }
        this->checkCudaErrors(ce, "GPUGMLMPop_computeBlock::computeLogLike errors:  X_lin * B launch failed");
    }
    
    //LL + W + log_dt + sum(lambda) -> LL for each neuron
    this->checkCudaErrors(dataset->waitForGroups_LL(stream), "GPUGMLM_computeBlock::computeLogLike errors:  waitForGroups_LL failed");

    dim3 block_size;
    if(params->dim_P() > 8) {
    	block_size.y = 4;
    }
    else if(params->dim_P() > 4) {
    	block_size.y = 2;
    }
    else {
    	block_size.y = 1;
    }
    
        
    block_size.x = 1024/block_size.y;
    dim3 grid_size;
    size_t max_blocks_needed  = dataset->LL->getSize(0) / block_size.x + ( (dataset->LL->getSize(0) % block_size.x == 0) ? 0 : 1);
    size_t blocks_to_use = 1024;
    grid_size.x  = min(max_blocks_needed, blocks_to_use);
    grid_size.y = dataset->dim_P() / block_size.y + ( (dataset->dim_P() % block_size.y == 0) ? 0 : 1);

    /*this->output_stream << " grid_size.y  = " << grid_size.y  << ", " << " block_size.y  = " << block_size.y << "\n";
    this->output_stream << " grid_size.x  = " << grid_size.x  << ", " << " block_size.x  = " << block_size.x << "\n";
    this->output_stream << " dim_P  = " << dataset->dim_P() << ", " << " dataset->LL->getSize(0) = " << dataset->LL->getSize(0) << "\n";
    this->msg->printMsgTxt(this->output_stream);
    this->checkCudaErrors("PRE GPUGMLMPop_computeBlock::computeLogLike errors:  kernel_getObs_LL launch failed");*/

    kernel_getObs_LL_pop<<<grid_size, block_size, 0, stream>>>(dataset->LL->device(), dataset->dLL->device(),
                  dataset->Y->device(),
                  dataset->lambda->device(),
                  dataset->dim_B() > 0,
                   params->W->device(), dataset->log_dt, dataset->dt,
                  dataset->id_a_trialM->device(),
                  params->trial_weights->device(),
                  dataset->ridx_a_all_c->device(),
                   params->logLikeSettings, params->logLikeParams->device());
                   
    this->checkCudaErrors("GPUGMLMPop_computeBlock::computeLogLike errors:  kernel_getObs_LL launch failed");
    this->checkCudaErrors(cudaEventRecord(LL_event, stream), "GPUGMLMPop_computeBlock::computeLogLike errors: could not add LL event to stream!");

    //sum up the LL for each trial (and dLL to setup for dW, dB)
    if(opts->compute_trialLL || opts->compute_dW) {
        //same block size

        size_t dim_M_c = params->getNumberOfNonzeroWeights();
        grid_size.x = params->getNumberOfNonzeroWeights()  / block_size.x + ( (params->getNumberOfNonzeroWeights()  % block_size.x == 0) ? 0 : 1);
        grid_size.y = dataset->dim_P() / block_size.y + ( (dataset->dim_P() % block_size.y == 0) ? 0 : 1);

        //this->output_stream << " grid_size.y  = " << grid_size.y  << ", " << " block_size.y  = " << block_size.y << "\n";
        //this->msg->printMsgTxt(output_stream);
        kernel_sum_trialLL_pop<<<grid_size, block_size, 0, stream>>>(results->trialLL->device(), dataset->dW_trial->device(),
                                                                 params->trial_included->device(), 
                                                                 dataset->LL->device(), dataset->dLL->device(), 
                                                                 opts->compute_trialLL, opts->compute_dW, 
                                                                 dataset->dim_N->device(),
                                                                 dataset->ridx_t_all_c->device(),
                                                                 dataset->id_t_trial->device(),
                                                                 params->trial_weights->device(),
                                                                 dataset->normalizingConstants_trial->device());
        this->checkCudaErrors("GPUGMLMPop_computeBlock::computeLogLike errors:  kernel_sum_trialLL launch failed");
    }
}

/* Kernel for each neuron
*  Sums up the trial dW (results->dW)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)
*/
template <class FPTYPE>
__global__ void kernel_sum_dW_pop( GPUData_kernel<FPTYPE> dW,  const GPUData_kernel<FPTYPE> dW_trial, const GPUData_kernel<FPTYPE> trial_weights) {
    size_t pp = blockIdx.x * blockDim.x + threadIdx.x;
    if(pp < dW.x) {
        FPTYPE dW_sum = 0;
        for(int tr = 0; tr < dW_trial.x; tr++) {
            if(trial_weights.y == 0 || (trial_weights.y == 1 && trial_weights[tr] != 0) || (trial_weights.y > 1 && trial_weights(tr,pp) != 0)) {
                dW_sum += dW_trial(tr,pp);
            }
        }
        dW[pp] = dW_sum;
    }
}


template <class FPTYPE>
void GPUGMLMPop_computeBlock<FPTYPE>::computeDerivatives(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    this->switchToDevice();
         //launch kernel to sum dLL -> dW, dB for each trial?
         //         or kernel to sum up dLL->dW and GEMV for dB?
         
    //for each Group
    for(unsigned int jj = 0; jj < dim_J; jj++) {
        dataset->Groups[jj]->computeDerivatives(results->Groups[jj], isSparseRun, opts->update_weights, params->Groups[jj], opts->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj], cusparseHandle_Groups[jj], LL_event);
    }   
    
    if(opts->compute_dW) {
        dim3 block_size;
        block_size.x = min(dataset->dim_P(), static_cast<size_t>(1024));
        dim3 grid_size;
        grid_size.x = dataset->dim_P() / block_size.x + ((dataset->dim_P() % block_size.x == 0)? 0:1);

      /*  this->output_stream << results->dW->getDevice() << "  " << dataset->dW_trial->getDevice() << "  " << params->trial_weights->getDevice() << "\n";
        this->msg->printMsgTxt(output_stream);*/


        kernel_sum_dW_pop<<<grid_size, block_size, 0, stream>>>(results->dW->device(), dataset->dW_trial->device(), params->trial_weights->device());
        this->checkCudaErrors("GPUGMLMPop_computeBlock::computeDerivatives errors:  kernel_sum_dW launch failed");
    }

    if(opts->compute_dB && dataset->dim_B() > 0) {
        GPUData<FPTYPE> * X_lin_c = isSparseRun ?  dataset->X_lin_temp : dataset->X_lin;


        cublasStatus_t ce;
        if(X_lin_c->getSize(2) == 1) { //if one shared X_lin term
            ce = X_lin_c->GEMM(results->dB,  dataset->dLL, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N);
        }
        else { // X_lin terms for each neuron
            ce = X_lin_c->GEMVs(results->dB, dataset->dLL, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N);
        }
        this->checkCudaErrors(ce, "GPUGMLMPop_computeBlock::computeDerivatives errors:  X_lin'*dLL -> dB failed");
    }      
}


//============================================================================================================================
//Dataset class
        
//Constructor takes in all the group data and GMLMPop setup
template <class FPTYPE>
GPUGMLMPop_dataset_GPU<FPTYPE>::GPUGMLMPop_dataset_GPU(const GPUGMLM_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_, const cudaStream_t stream, const std::vector<cusparseHandle_t> & cusparseHandle_Groups, std::shared_ptr<GPUGL_msg> msg_) {
    this->dev  = block->dev_num;
    this->msg = msg_;
    this->switchToDevice();
    cudaError_t ce;

    dt = GMLMPopstructure->binSize;
    log_dt = log(GMLMPopstructure->binSize);

    Groups.assign(GMLMPopstructure->Groups.size(), NULL); //sets up dim_J()
    dim_N = new GPUData<size_t>(ce, GPUData_HOST_STANDARD, stream, block->trials.size()); //sets up dim_M()
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate dim_N on device!");
            
    // number of trials
    isInDataset_trial.assign( max_trials_, false); //if each trial is in this block
    if(dim_M() == 0) {
        this->output_stream << "GPUGMLMPop_dataset_GPU errors: no trials given to GPU block!";
        this->msg->callErrMsgTxt(output_stream);
    }

    max_trials_for_sparse_run = min(dim_M()/2, static_cast<size_t>(block->max_trials_for_sparse_run));

    // setup up the order that trials go to the GPU
    //   in blocks ordered by neurons     

    size_t dim_N_total_c = 0;
    dim_N_temp = 0;
    max_trial_length = 1;

    ridx_t_all = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate ridx_t_all on device!");
    id_t_trial = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_M());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate id_t_trial on device!");
    normalizingConstants_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_STANDARD, stream, dim_M(), GMLMPopstructure->dim_P);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate normalizingConstants_trial on device!");

    size_t X_lin_depth;
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        if(mm == 0) {
            X_lin_depth = block->trials[mm]->X_lin->getSize(2);
        }

        if(block->trials[mm]->X_lin->getSize(2) != X_lin_depth || (X_lin_depth > 1 && X_lin_depth != GMLMPopstructure->dim_P)) {
            this->output_stream << "GPUGMLMPop_dataset_GPU errors: invalid size of X_lin: depth must be 0-1 or dim_P!";
            this->msg->callErrMsgTxt(output_stream);
        }
        
        //save trial indices
        (*ridx_t_all)[mm] = dim_N_total_c;

        // get trial length
        (*dim_N)[mm] = block->trials[mm]->dim_N(msg);
        if((*dim_N)[mm] == 0) {
            this->output_stream << "GPUGMLMPop_dataset_GPU errors: trials cannot be empty!";
            this->msg->callErrMsgTxt(output_stream);
        }
        dim_N_total_c += (*dim_N)[mm]; // add length to total 

        max_trial_length = max(max_trial_length, (*dim_N)[mm]); //update max trial length

        //save trial and neuron number
        (*id_t_trial)[mm] = block->trials[mm]->trial_idx;
        if(isInDataset_trial[block->trials[mm]->trial_idx]) { //trial index already found
            this->output_stream << "GPUGMLMPop_dataset_GPU errors: trial indices must be unique!";
            this->msg->callErrMsgTxt(output_stream);
        }

        isInDataset_trial[block->trials[mm]->trial_idx] = true;

        for(unsigned int pp = 0; pp < GMLMPopstructure->dim_P; pp++) {
            FPTYPE nc = 0; // normalizing constant
            for(unsigned int nn = 0; nn < (*dim_N)[mm]; nn++) {
                if(GMLMPopstructure->logLikeSettings == ll_poissExp || GMLMPopstructure->logLikeSettings == ll_poissSoftRec) {
                    FPTYPE Y_c = (*(block->trials[mm]->Y))(nn, pp);
                    nc += (Y_c >= 0) ? -lgamma(floor(Y_c) + 1.0) : 0;
                }
            }
            (*normalizingConstants_trial)(mm, pp) = nc;
        }
    }
    id_a_trialM = new GPUData<unsigned int>(ce, GPUData_HOST_STANDARD, stream, dim_N_total_c);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate id_a_trialM on device!");

    size_t N_total_ctr = 0;
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        for(unsigned int nn = 0; nn < (*dim_N)[mm]; nn++) {
            (*id_a_trialM)[N_total_ctr + nn] = mm;
        }
        N_total_ctr += (*dim_N)[mm];
    }

    //allocate space on GPU for data and copy any local values to GPU
        //spike counts
    Y = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c, GMLMPopstructure->dim_P);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate Y on device!");
    
        //linear term (divded up into per-neuron blocks)
    X_lin = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total_c, GMLMPopstructure->dim_B, X_lin_depth);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate X_lin on device!");
        
        //copy each trial to GPU
    for(unsigned int mm = 0; mm < dim_M(); mm++) {
        // spike counts
        cudaPos copyOffset = make_cudaPos((*ridx_t_all)[mm], 0, 0);
        this->checkCudaErrors(Y->copyTo(stream, block->trials[mm]->Y, true, copyOffset), "GPUGMLMPop_dataset_GPU errors: could not copy Y to device!");
                
        // linear term
        if(!X_lin->empty()) { //don't call if no linear term
            this->checkCudaErrors(X_lin->copyTo(stream, block->trials[mm]->X_lin, true, copyOffset), "GPUGMLMPop_dataset_GPU errors: could not copy X_lin to device!");
        }
    } 

    //upload vectors to GPU
    this->checkCudaErrors(normalizingConstants_trial->copyHostToGPU(stream), "GPUGMLMPop_dataset_GPU errors: could not copy normalizingConstants_trial to device!");
   
    this->checkCudaErrors(ridx_t_all->copyHostToGPU(stream), "GPUGMLMPop_dataset_GPU errors: could not copy ridx_t_all to device!");
    this->checkCudaErrors(id_t_trial->copyHostToGPU(stream), "GPUGMLMPop_dataset_GPU errors: could not copy id_t_trial to device!");
    this->checkCudaErrors(id_a_trialM->copyHostToGPU(stream), "GPUGMLMPop_dataset_GPU errors: could not copy id_a_trialM to device!");
     
    this->checkCudaErrors(dim_N->copyHostToGPU(stream), "GPUGMLMPop_dataset_GPU errors: could not copy dim_N to device!");

    //setup compute space
     LL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total(), dim_P());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate LL on device!");
    dLL = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total(), dim_P());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate dLL on device!");

    ridx_sa_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate ridx_sa_all on device!");
    ridx_a_all = new GPUData<unsigned int>(ce, GPUData_HOST_NONE, stream, dim_N_total(), 0);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate ridx_a_all on device!");
    ridx_st_sall = new GPUData<unsigned int>(ce, GPUData_HOST_PAGELOCKED, stream, dim_M());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate ridx_st_sall on device!");
    
    lambda = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_N_total(), dim_P(), dim_J());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate lambda on device!");

    X_lin_temp = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, max_trial_length * max_trials_for_sparse_run, dim_B(), X_lin_depth);
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate X_lin_temp on device!");

    dW_trial = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_M(), dim_P());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_GPU errors: could not allocate dW_trial on device!");

    //setup the groups
    for(unsigned int jj = 0; jj < dim_J(); jj++) {
        Groups[jj] = new GPUGMLMPop_dataset_Group_GPU<FPTYPE>(jj, GMLMPopstructure->Groups[jj], block->trials, this, stream, cusparseHandle_Groups[jj]);
    }
}

template <class FPTYPE>
GPUGMLMPop_dataset_Group_GPU<FPTYPE>::GPUGMLMPop_dataset_Group_GPU(const int groupNum_, const GPUGMLM_structure_Group_args<FPTYPE> * GMLMPopGroupStructure, const std::vector<GPUGMLM_trial_args <FPTYPE> *> trials, const GPUGMLMPop_dataset_GPU<FPTYPE> * parent_, const cudaStream_t stream, const cusparseHandle_t & cusparseHandle) : parent(parent_), groupNum(groupNum_) {
    this->dev  = parent->dev;
    this->msg = parent->msg;
    this->switchToDevice();
    cudaError_t ce;
    
    //sets up dimensions
    X.resize( GMLMPopGroupStructure->dim_D(msg));
    XF.resize(dim_D());
    iX.resize(dim_D());
    X_temp.resize( dim_D());
    lambda_d.resize( dim_D());

    isShared = new GPUData<bool>(ce, GPUData_HOST_STANDARD, stream, dim_D()); 
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate isShared!");
    isSharedIdentity = new GPUData<bool>(ce, GPUData_HOST_STANDARD, stream, dim_D()); 
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate isSharedIdentity!");

    dim_A = GMLMPopGroupStructure->dim_A;

    size_t dim_T_total = 1;
    std::vector<size_t> dim_F_c;
    dim_F_c.assign(dim_D(), 1);
    size_t max_dim_F_dim_P = parent->dim_P();
    for(unsigned int ss = 0; ss < GMLMPopGroupStructure->dim_S(); ss++) {
        dim_T_total *= GMLMPopGroupStructure->dim_T[ss];
        dim_F_c[GMLMPopGroupStructure->factor_idx[ss]] *= GMLMPopGroupStructure->dim_T[ss];

        max_dim_F_dim_P = max(max_dim_F_dim_P,  dim_F_c[GMLMPopGroupStructure->factor_idx[ss]]);
    }

    if(GMLMPopGroupStructure->dim_S() == 0 || dim_T_total == 0) {
        this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: tensor has no components!";
        this->msg->callErrMsgTxt(output_stream);
    }
    if(GMLMPopGroupStructure->dim_A == 0) {
        this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: tensor has no events/data!";
        this->msg->callErrMsgTxt(output_stream);
    }
    if(GMLMPopGroupStructure->dim_R_max < 1) {
        this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: tensor max rank must be at least 1!";
        this->msg->callErrMsgTxt(output_stream);
    }
    
    //allocated space for regressors and copy to GPU
    size_t max_dim_X_shared = parent->dim_N_total();

    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        (*isShared)[dd] = !(GMLMPopGroupStructure->X_shared[dd]->empty());

        if((*isShared)[dd]) {
            //if shared
            max_dim_X_shared = max(max_dim_X_shared, GMLMPopGroupStructure->X_shared[dd]->getSize(0));

            //gets depth
            size_t depth = GMLMPopGroupStructure->X_shared[dd]->getSize(2);
            if(depth != 1) {
                this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: X_shared depth must be 1!";
                this->msg->callErrMsgTxt(output_stream);
            }

            //allocate space
            X[dd]  = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, GMLMPopGroupStructure->X_shared[dd]->getSize(0), dim_F_c[dd], depth);
            iX[dd] = new GPUData<int   >(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_A);

            //copy to GPU
            this->checkCudaErrors(X[dd]->copyTo(stream, GMLMPopGroupStructure->X_shared[dd], false), "GPUGMLMPop_dataset_Group_GPU errors: could not copy X[dd] shared to device!");
       
            // copy each trial's data to GPU
            for(unsigned int mm = 0; mm < trials.size(); mm++) {
                cudaPos copyOffset = make_cudaPos((*(parent->ridx_t_all))[mm], 0, 0); //get row for current trial
                this->checkCudaErrors(iX[dd]->copyTo(stream, trials[mm]->Groups[groupNum]->iX[dd], true, copyOffset), "GPUGMLMPop_dataset_Group_GPU errors: could not copy iX[dd] shared to device!");
            }

            //check if X_shared is the identity matrix
            if(X[dd]->getSize(0) == X[dd]->getSize(1)) {
                (*isSharedIdentity)[dd] = true;
                for(unsigned int ii = 0; ii < X[dd]->getSize(0) && (*isSharedIdentity)[dd]; ii++) {
                    for(unsigned int jj = 0; jj < X[dd]->getSize(1) && (*isSharedIdentity)[dd]; jj++) {
                        if(ii == jj) {
                            (*isSharedIdentity)[dd] = 1 == (*(GMLMPopGroupStructure->X_shared[dd]))(ii,jj);
                        }
                        else {
                            (*isSharedIdentity)[dd] = 0 == (*(GMLMPopGroupStructure->X_shared[dd]))(ii,jj);
                        }
                    }
                }
            }
            else {
                (*isSharedIdentity)[dd] = false;
            }

            if(!((*isSharedIdentity)[dd])) {
                //XF comp space
                XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_X(dd), GMLMPopGroupStructure->dim_R_max);
                this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for XF[dd] shared!" );
            }
            else {
                XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, dim_X(dd), GMLMPopGroupStructure->dim_R_max, 0); // is empty, but has correct dimensions
                this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for XF[dd] shared+identity!" );
            }

            //X space for sparse runs
            X_temp[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->max_trial_length * parent->max_trials_for_sparse_run, dim_F_c[dd], dim_A, true);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for X_temp[dd]!" );
        }
        else {
            //if local
            (*isSharedIdentity)[dd] = false;

            //gets depth
            size_t depth = trials[0]->Groups[groupNum]->X[dd]->getSize(2);
            if(depth != 1 && depth != dim_A) {
                this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: X_local depth must be dim_A or 1!";
                this->msg->callErrMsgTxt(output_stream);
            }

            //allocate space
            X[dd]  = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_F_c[dd], depth, true);
            iX[dd] = new GPUData<int   >(ce, GPUData_HOST_NONE, stream, 0, GMLMPopGroupStructure->dim_A);

            // copy each trial's data
            for(unsigned int mm = 0; mm < trials.size(); mm++) {
                cudaPos copyOffset = make_cudaPos((*(parent->ridx_t_all))[mm], 0, 0); //get row for current trial
                this->checkCudaErrors(X[dd]->copyTo(stream, trials[mm]->Groups[groupNum]->X[dd], true, copyOffset), "GPUGMLMPop_dataset_Group_GPU errors: could not copy X[dd] local to device!");
            }

            //XF comp space
            XF[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), GMLMPopGroupStructure->dim_R_max, depth, true);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for XF[dd] shared!" );

            //X space for sparse runs
            X_temp[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->max_trial_length * parent->max_trials_for_sparse_run, dim_F_c[dd], depth, true);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for X_temp[dd]!" );
        }

    }

    this->checkCudaErrors(isShared->copyHostToGPU(stream), "GPUGMLMPop_dataset_Group_GPU errors: could not copy isShared to device!");
    this->checkCudaErrors(isSharedIdentity->copyHostToGPU(stream), "GPUGMLMPop_dataset_Group_GPU errors: could not copy isSharedIdentity to device!");
    
    //setup compute space
    lambda_v = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_R_max());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for lambda_v!" );

    // pitched memory for lambda_d: note arrangement is (dim_N_total*dim_A) x dim_R
    //                                this stacks the events to line up with X or S
    lambda_d.assign(dim_D(), NULL);
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        size_t depth = dim_A;
        if(!((*isShared)[dd]) && X[dd]->getSize(2) == 1) {
            depth = 1;
        }
        lambda_d[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, parent->dim_N_total(), dim_R_max(), depth, true);
        this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for lambda_d!" );
    }

    phi_d =  new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, max_dim_X_shared, dim_R_max());
    this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for phi_d!" );

    //setup sparse matrices for dT
    spi_rows.assign(dim_D(), NULL);
    spi_cols.assign(dim_D(), NULL);
    spi_data.assign(dim_D(), NULL);

    spi_S.assign(dim_D(), NULL);
    spi_phi_d.assign(dim_D(), NULL);
    spi_lambda_d.assign(dim_D(), NULL);

    spi_buffer.assign(dim_D(), NULL);
    spi_buffer_size.assign(dim_D(), 0);

    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        if((*isShared)[dd]) {
            //gets the rows and cols of the spm in the correct order
                //shorter algorithm is too slow for my level of patience, so we do this in a couple steps
                //first, get valid entries and number of entries per row of spi_S
            size_t ctr = 0;
            std::vector<int> row_ctr;
            row_ctr.resize(dim_X(dd));
            for(unsigned int mm = 0; mm < parent->dim_M(); mm++) { //for each trial
                for(unsigned int aa = 0; aa < dim_A; aa++) { //for each event
                    for(unsigned int nn = 0; nn < trials[mm]->dim_N(msg); nn++) { //for each observation
                        //gets the entry in the input data
                        int row = (*(trials[mm]->Groups[groupNum]->iX[dd]))(nn, aa);
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
            for(unsigned int xx = 1; xx < dim_X(dd); xx++) {
                row_idx[xx] = row_ctr[xx-1] + row_idx[xx-1]; 
            }
                //goes back through the indices and adds them on
            spi_rows[dd] = new GPUData<int>(ce, GPUData_HOST_STANDARD, stream, ctr, 1, 1);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for spi_rows[dd]!");
            spi_cols[dd] = new GPUData<int>(ce, GPUData_HOST_STANDARD, stream, ctr, 1, 1);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for spi_cols[dd]!");

            row_ctr.assign(dim_X(dd), 0); //reset row counter
            for(unsigned int mm = 0; mm < parent->dim_M(); mm++) { //for each trial
                for(unsigned int aa = 0; aa < dim_A; aa++) { //for each event
                    for(unsigned int nn = 0; nn < trials[mm]->dim_N(msg); nn++) { //for each observation
                        //gets the entry in the input data
                        int row = (*(trials[mm]->Groups[groupNum]->iX[dd]))(nn, aa);
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
            this->checkCudaErrors(spi_rows[dd]->copyHostToGPU(stream), "GPUGMLMPop_dataset_Group_GPU errors: could not copy spi_rows[dd] to device!");
            this->checkCudaErrors(spi_cols[dd]->copyHostToGPU(stream), "GPUGMLMPop_dataset_Group_GPU errors: could not copy spi_cols[dd] to device!");
            
            spi_data[dd] = new GPUData<FPTYPE>(ce, GPUData_HOST_NONE, stream, spi_rows[dd]->size(), 1, 1);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for spi_data[dd]!");

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
            this->checkCudaErrors(cusparse_stat, "GPUGMLMPop_dataset_Group_GPU errors: creating sparse mat spi_S for dT failed.");

            //setup dense handle for lambda_d
            spi_lambda_d[dd] = new cusparseDnVecDescr_t;
            cusparse_stat = cusparseCreateDnVec(spi_lambda_d[dd],
                                                lambda_d[dd]->getSize(0) * lambda_d[dd]->getSize(2),  //size
                                                lambda_d[dd]->getData_gpu(),
                                                getCudaType<FPTYPE>());
            output_stream << "GPUGMLMPop_dataset_Group_GPU errors: creating dense vec cusparse handle spi_lambda_d failed.\n";
            lambda_d[dd]->printInfo(output_stream);
            output_stream << " group " << groupNum << "\n";
            output_stream << " factor " << dd << "\n";
            this->checkCudaErrors(cusparse_stat, output_stream.str().c_str());
            output_stream.str("");
            output_stream.clear();

            //setup dense handle for phi_d
            spi_phi_d[dd] = new cusparseDnVecDescr_t;
            cusparse_stat = cusparseCreateDnVec(spi_phi_d[dd],
                                                dim_X(dd), //size
                                                phi_d->getData_gpu(), //values
                                                getCudaType<FPTYPE>()); //valueType
            this->checkCudaErrors(cusparse_stat, "GPUGMLMPop_dataset_Group_GPU errors: creating dense vec cusparse handle spi_phi_d failed.");

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
                    CUSPARSE_SPMV_COO_ALG1,//CUSPARSE_SPMV_ALG_DEFAULT,
                    &(buffer));
            this->checkCudaErrors(cusparse_stat, "GPUGMLMPop_dataset_Group_GPU errors: getting buffer size for SpMV failed.");

            spi_buffer[dd] = new GPUData<char>(ce, GPUData_HOST_NONE, stream, buffer, 1, 1);
            this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU errors: could not allocate space for spi_buffer[dd]!" );
            spi_buffer_size[dd] = buffer; 
        }
    }

    this->checkCudaErrors(cudaEventCreate(&group_LL_event), "GPUGMLMPop_dataset_Group_GPU errors: could not create LL event!");
}

// destructor
template <class FPTYPE>
GPUGMLMPop_dataset_GPU<FPTYPE>::~GPUGMLMPop_dataset_GPU() {
    cudaSafeFree(Y, "GPUGMLMPop_dataset_GPU errors: could not free Y");
    
    cudaSafeFree(X_lin, "GPUGMLMPop_dataset_GPU errors: could not free X_lin");
    cudaSafeFree(X_lin_temp, "GPUGMLMPop_dataset_GPU errors: could not free X_lin_temp");
    
    cudaSafeFree(normalizingConstants_trial, "GPUGMLMPop_dataset_GPU errors: could not free normalizingConstants_trial");
    
    cudaSafeFree(ridx_t_all   , "GPUGMLMPop_dataset_GPU errors: could not free ridx_t_all");
    cudaSafeFree(ridx_st_sall , "GPUGMLMPop_dataset_GPU errors: could not free ridx_st_sall");
    cudaSafeFree(ridx_sa_all  , "GPUGMLMPop_dataset_GPU errors: could not free ridx_sa_all");
    cudaSafeFree(ridx_a_all  , "GPUGMLMPop_dataset_GPU errors: could not free ridx_a_all");
    
    cudaSafeFree(id_t_trial , "GPUGMLMPop_dataset_GPU errors: could not free id_t_trial");
    cudaSafeFree(id_a_trialM, "GPUGMLMPop_dataset_GPU errors: could not free id_a_trialM");
    
    cudaSafeFree(dim_N, "GPUGMLMPop_dataset_GPU errors: could not free dim_N");
    
    cudaSafeFree( LL, "GPUGMLMPop_dataset_GPU errors: could not free  LL");
    cudaSafeFree(dLL, "GPUGMLMPop_dataset_GPU errors: could not free dLL");
    cudaSafeFree(lambda, "GPUGMLMPop_dataset_GPU errors: could not free lambda");
    cudaSafeFree(dW_trial, "GPUGMLMPop_dataset_GPU errors: could not free dW_trial");

    //clear the groups
    for(auto gg : Groups) {
        delete gg;
    }
}

template <class FPTYPE>
GPUGMLMPop_dataset_Group_GPU<FPTYPE>::~GPUGMLMPop_dataset_Group_GPU() {
    this->checkCudaErrors(cudaEventDestroy(group_LL_event), "GPUGMLMPop_dataset_Group_GPU errors: could not clear LL event!");
    cudaSafeFreeVector(X, "GPUGMLMPop_dataset_Group_GPU errors: could not free X[dd]");
    cudaSafeFreeVector(XF, "GPUGMLMPop_dataset_Group_GPU errors: could not free iX[dd]");
    cudaSafeFreeVector(iX, "GPUGMLMPop_dataset_Group_GPU errors: could not free iX[dd]");
    cudaSafeFreeVector(X_temp   , "GPUGMLMPop_dataset_Group_GPU errors: could not free X_temp[dd]");
    
    cudaSafeFree(isShared, "GPUGMLMPop_dataset_Group_GPU errors: could not free isShared");
    cudaSafeFree(isSharedIdentity, "GPUGMLMPop_dataset_Group_GPU errors: could not free isSharedIdentity");

    cudaSafeFree(lambda_v, "GPUGMLMPop_dataset_Group_GPU errors: could not free lambda_v");
    cudaSafeFreeVector(lambda_d, "GPUGMLMPop_dataset_Group_GPU errors: could not free lambda_d[dd]");
    cudaSafeFree(   phi_d, "GPUGMLMPop_dataset_Group_GPU errors: could not free phi_d");

    cudaSafeFreeVector(spi_rows, "GPUGMLMPop_dataset_Group_GPU errors: could not free spi_rows");
    cudaSafeFreeVector(spi_cols, "GPUGMLMPop_dataset_Group_GPU errors: could not free spi_cols");
    cudaSafeFreeVector(spi_data, "GPUGMLMPop_dataset_Group_GPU errors: could not free spi_data");
    cudaSafeFreeVector(spi_buffer, "GPUGMLMPop_dataset_Group_GPU errors: could not free spi_buffer");
    //destroy any cusparse handles
    for(unsigned int dd = 0; dd < spi_S.size(); dd++) {
        if(spi_S[dd] != NULL) {
            this->checkCudaErrors(cusparseDestroySpMat(*spi_S[dd]), "GPUGMLMPop_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_S descr.");
            delete spi_S[dd];
        }
        if(spi_phi_d[dd] != NULL) {
            this->checkCudaErrors(cusparseDestroyDnVec(*spi_phi_d[dd]), "GPUGMLMPop_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_phi_d descr.");
        	delete spi_phi_d[dd];
        }
        if(spi_lambda_d[dd] != NULL) {
            this->checkCudaErrors(cusparseDestroyDnVec(*spi_lambda_d[dd]), "GPUGMLMPop_dataset_Group_GPU errors: CUSPARSE failed to destroy spi_lambda_d descr.");
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
__global__ void kernel_getGroupX_local_full_pop(GPUData_kernel<FPTYPE> X_temp, const GPUData_kernel<FPTYPE> X,
                                    const GPUData_kernel<unsigned int> ridx_sa_all) {
    //get current observation number
    unsigned int tt_start = blockIdx.y * blockDim.y;
    unsigned int aa_start = blockIdx.z * blockDim.z;
    size_t row_start = blockIdx.x * blockDim.x;
    for(size_t row_0 = row_start; row_0 < ridx_sa_all.x; row_0 += blockDim.x * gridDim.x) {
        size_t row = row_0 + threadIdx.x;
        size_t iX_row;
        if(row < ridx_sa_all.x  ) {
            iX_row = ridx_sa_all[row];
        }
        __syncwarp();
        for(size_t aa_0 = aa_start; aa_0 < X_temp.z; aa_0 += blockDim.z * gridDim.z) {
            size_t aa = aa_0 + threadIdx.z;

            for(size_t tt_0 = tt_start; tt_0 < X_temp.y; tt_0 += blockDim.y * gridDim.y) {
                size_t tt = tt_0 + threadIdx.y;

                //for each event 
                if(row < ridx_sa_all.x && tt < X_temp.y ) {
                    X_temp(row, tt, aa) = X(iX_row, tt, aa);
                }
                __syncwarp();
            }
        }
    }
}

//functions to multiply the tensor coefficients by the current parameters
template <class FPTYPE>
void GPUGMLMPop_dataset_Group_GPU<FPTYPE>::multiplyCoefficients(const bool isSparseRun, const bool update_weights, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const cudaStream_t stream, const cublasHandle_t cublasHandle, cudaEvent_t & paramsLoaded) {
    this->checkCudaErrors(set_dim_R(params->dim_R(), stream), "GPUGMLMPop_dataset_Group_GPU errors: could not set dim_R!");
    if(params->dim_R() == 0) {
        return;
    }
    if(params->dim_R() > dim_R_max()) {
        this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: dim_R too large for pre-allocated space!";
        this->msg->callErrMsgTxt(output_stream);
    }
    this->checkCudaErrors(cudaStreamWaitEvent(stream, paramsLoaded, 0), "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors: could not wait for event.");

    if(isSparseRun && update_weights) {
        this->checkCudaErrors(lambda_v->resize(stream, parent->dim_N_temp, -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
    }
    else if(update_weights) {
        this->checkCudaErrors(lambda_v->resize(stream, parent->lambda->getSize_max(0), -1, -1), "GPUGMLM_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
    }
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        GPUData<FPTYPE> * X_c = X[dd];
        if(isSparseRun && update_weights) {
            this->checkCudaErrors(X_temp[dd]->resize(  stream, parent->dim_N_temp, -1, -1), "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
            this->checkCudaErrors(lambda_d[dd]->resize(stream, parent->dim_N_temp, -1, -1), "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors: could not set size for sparse runs.");
        }
        else if(update_weights) {
            this->checkCudaErrors(lambda_d[dd]->resize(stream, lambda_d[dd]->getSize_max(0),-1,-1), "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors: could not set size for full runs.");
        }
        if((*isSharedIdentity)[dd]) {
            continue;
        }

        if(isSparseRun && !(*isShared)[dd]) {
            // if sparse run and local regressors, build matrix then multiply
            if(update_weights) {
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
                kernel_getGroupX_local_full_pop<<<grid_size, block_size, 0, stream>>>(X_temp[dd]->device(), X[dd]->device(), 
                                            parent->ridx_sa_all->device());
                this->checkCudaErrors("GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors:  kernel_getGroupX_local_full launch failed");
            }
            X_c = X_temp[dd];
        }

        this->checkCudaErrors(XF[dd]->resize(stream, X_c->getSize(0)), "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors: could not set matrix size for run.");    
        cublasStatus_t cse = X_c->GEMM(XF[dd], params->F[dd], cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N);
        if(cse != CUBLAS_STATUS_SUCCESS) {
            this->output_stream << " dd " << dd << "\n";
            X_c->printInfo(output_stream, "X_c");
            XF[dd]->printInfo(output_stream, "XF[dd]");
            params->F[dd]->printInfo(output_stream, "params->F[dd]");
            this->msg->printMsgTxt(output_stream);
        }       
        this->checkCudaErrors(cse, "GPUGMLMPop_dataset_Group_GPU::multiplyCoefficients errors:  X*F -> XF failed");
    }
}

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================

        /*Kernel for each observation, for a group
 * For each component (rr = 0:(dim_R-1)) takes the product of the XT terms into lambda_v, then lambda_v'*V -> lambda 
 * Returns the observation-wise constribution to the rate from this group (lambda) and sets up the dV computation
 *
 * If computing any dT values AND dim_S > 1, needs some dynamic  memory to make this work on both 1080 and 2080 cards well. Memory size in bytes is dim_S * blockDim.x * sizeof(FPTYPE)
 */
template <class FPTYPE, unsigned int max_order>
__global__ void kernel_getGroupRate_pop(GPUData_kernel<FPTYPE> lambda_v, 
        GPUData_array_kernel<FPTYPE,MAX_DIM_D> lambda_d,
        const GPUData_array_kernel<FPTYPE,MAX_DIM_D> XF,
        const GPUData_array_kernel<FPTYPE,MAX_DIM_D> F,
        const GPUData_array_kernel<int,MAX_DIM_D> iX,
        const GPUData_kernel<bool> isShared,
        const GPUData_kernel<bool> isSharedIdentity,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights, 
        const bool compute_dV, const GPUData_kernel<bool> compute_dF, const bool compute_dT_any,
        const GPUData_kernel<unsigned int> ridx_sa_all, const size_t dim_A) {
    __shared__ bool isShared_local[max_order];
    __shared__ bool isSharedIdentity_local[max_order];
    FPTYPE t_array[max_order];

    if(threadIdx.x < XF.N) {
        isShared_local[threadIdx.x] = isShared[threadIdx.x];
        isSharedIdentity_local[threadIdx.x] = isSharedIdentity[threadIdx.x];
    }
    int idx_0 = -1;
    __syncthreads();

    for(size_t row_0 = blockIdx.x * blockDim.x; row_0 < lambda_v.x; row_0 += blockDim.x * gridDim.x) {
        size_t row = row_0 + threadIdx.x;
        size_t iX_row = row; //if full run
        if(ridx_sa_all.y > 0 && row < ridx_sa_all.x) {
            //if sparse run
            iX_row = ridx_sa_all[row];
        }
        bool rowIncluded = true;
        if( row >= lambda_v.x || !(trial_weights.y == 0 || trial_weights.y > 1 || (trial_weights.y == 1 && trial_weights[id_a_trialM[iX_row]] != 0))) {
            rowIncluded = false;
        }
        __syncthreads();

        //for each rank
        for(unsigned int rr_0 = blockIdx.y * blockDim.y; rr_0 < lambda_v.y; rr_0 += blockDim.y * gridDim.y) { //dim_R = V->Y
            unsigned int rr = rr_0 + threadIdx.y;
            bool elementIncluded = rowIncluded && rr < lambda_v.y;

            FPTYPE lv = 0;
            for(unsigned int aa = 0; aa < dim_A; aa++) { //over dim_A
                //for each event 
                FPTYPE lv_aa = 1;
                //for each factor
                for(unsigned int dd = 0; dd < XF.N; dd++) { //dim_D = XF->N, dim_S = T->N

                    if(elementIncluded && isShared_local[dd]) { //if trial not censored
                        idx_0 = iX[dd](iX_row, aa); //get index for shared regressors
                                                    // this is pulled up too many times (is same over all rr): haven't found a good way to parallelize
                    }
                    __syncwarp(); 
                    if(elementIncluded && isSharedIdentity_local[dd]) { 
                        if(idx_0 >= 0 && idx_0 < F[dd].x) {
                            t_array[dd]  = F[dd](idx_0, rr);
                        }
                        else {
                            t_array[dd] = 0;
                        }
                    }
                    __syncwarp(); 
                    if(elementIncluded && (!isShared_local[dd] || !isSharedIdentity_local[dd])) { 
                        if(!isShared_local[dd]) {
                            t_array[dd]  = XF[dd](row, rr, aa);
                        }
                        else if(idx_0 >= 0 && idx_0 < XF[dd].x) {
                            t_array[dd]  = XF[dd](idx_0, rr);
                        }
                        else {
                            t_array[dd] = 0;
                        }
                    }
                    lv_aa *= t_array[dd] ;
                    __syncwarp(); 
                } // dd
                lv += lv_aa;

                //sets up any dT matrices (doing this here eliminates the need to go back through the XT matrices in a different kernel)
                //  I do this outside the previous loop because otherwise everything was super slow on the 1080 cards
                if(compute_dT_any) {
                    for(unsigned int dd = 0 ; dd < XF.N; dd++) {
                        if(elementIncluded && compute_dF[dd]) {
                            FPTYPE tt = 1;
                            for(unsigned int dd2 = 0; dd2 < XF.N; dd2++) {
                                if(dd2 != dd) {
                                    tt *= t_array[dd2];
                                }
                            }
                            if(aa < lambda_d[dd].z) {
                                lambda_d[dd](row, rr, aa) = tt;
                            }
                            else {
                                lambda_d[dd](row, rr, aa) += tt;
                            }
                        }
                        __syncwarp();
                    } // dd
                }
                __syncthreads();
            }// aa
            if(elementIncluded) {
                lambda_v(row, rr) = lv;
            }
            __syncthreads();
        } // rr
    } // row
}

template <class FPTYPE>
void GPUGMLMPop_dataset_Group_GPU<FPTYPE>::getGroupRate(const bool isSparseRun, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle) { 
    if(params->dim_R() == 0) {
        // set lambda to 0
        FPTYPE * col = parent->lambda->getData_gpu() + groupNum * parent->lambda->getLD_gpu();
        this->checkCudaErrors(cudaMemsetAsync(col, 0, parent->lambda->getSize(0)*sizeof(FPTYPE), stream), "GPUGMLMPop_dataset_Group_GPU::getGroupRate errors: errors setting rate to 0 for dim_R=0 group");
    }
    else {
        dim3 block_size;
        dim3 grid_size;

        if(dim_R() > 8) {
            block_size.y = 4;
        }
        else  if(dim_R() > 4) {
            block_size.y = 2;
        }
        else {
            block_size.y = 1;
        }
        block_size.x = 1024 / block_size.y;
        size_t max_blocks_needed  = parent->lambda->getSize(0) / block_size.x + ((parent->lambda->getSize(0) % block_size.x == 0)? 0:1);
        size_t blocks_to_use = (parent->dim_J() == 1) ? 1024 : 512;
        grid_size.x  = min(max_blocks_needed, blocks_to_use);
        grid_size.y  = dim_R() / block_size.y + ((dim_R() % block_size.x == 0)? 0:1);

        bool compute_dT_any = false;
        for(unsigned int ss = 0; ss < params->dim_S(); ss++) {
            if(opts->compute_dT[ss]) {
                compute_dT_any = true;
                break;
            }
        }
        
        switch( params->dim_S()) {
                case 1:
                    kernel_getGroupRate_pop<FPTYPE,1><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 2:
                    kernel_getGroupRate_pop<FPTYPE,2><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 3:
                    kernel_getGroupRate_pop<FPTYPE,3><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 4:
                    kernel_getGroupRate_pop<FPTYPE,4><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 5:
                    kernel_getGroupRate_pop<FPTYPE,5><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 6:
                    kernel_getGroupRate_pop<FPTYPE,6><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                case 7:
                    kernel_getGroupRate_pop<FPTYPE,7><<<grid_size, block_size, 0, stream>>>( lambda_v->device(),  GPUData<FPTYPE>::assembleKernels(lambda_d), 
                                                                                        GPUData<FPTYPE>::assembleKernels(XF),  GPUData<FPTYPE>::assembleKernels(params->F),  GPUData<int>::assembleKernels(iX),
                                                                                        isShared->device(), isSharedIdentity->device(),
                                                                                        parent->id_a_trialM->device(),
                                                                                        params->getTrialWeights()->device(),
                                                                                        opts->compute_dV, params->compute_dF->device(), compute_dT_any,
                                                                                        parent->ridx_a_all_c->device(), dim_A);
                    break;
                default:
                this->checkCudaErrors(cudaErrorInvalidConfiguration, "GPUGMLMPop_dataset_Group_GPU::getGroupRate_pop errors:  kernel_getGroupRate_pop launch failed - invalid tensor order");
        }
        this->checkCudaErrors("GPUGMLMPop_dataset_Group_GPU::getGroupRate_pop errors:  kernel_getGroupRate_pop launch failed");

        // multiply lambda_v * V' -> lambda(:, :, groupNum)
        FPTYPE alpha = 1;
        FPTYPE beta  = 0;
        cublasStatus_t ce =  cublasGEMM(cublasHandle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_T,
                              lambda_v->getSize(0), dim_P(), dim_R(),
                              &alpha,
                              lambda_v->getData_gpu(), lambda_v->getLD_gpu(),
                              params->V->getData_gpu(), params->V->getLD_gpu(),
                              &beta,
                              parent->lambda->getData_gpu() + groupNum*parent->lambda->getInc_gpu(), parent->lambda->getLD_gpu());
        this->checkCudaErrors(ce, "GPUGMLMPop_dataset_Group_GPU::getGroupRate errors:  lambda_v * V' -> lambda(:, :, groupNum) failed");
    }
    this->checkCudaErrors(cudaEventRecord(group_LL_event, stream), "GPUGMLMPop_dataset_Group_GPU::getGroupRate errors: could not add LL event to stream!");
}

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================

/* Kernel for each entry of the sparse matrix for S*lambda_t -> phi_t (shared regressor compression)
*  sets up the elements of S to be a column of lambda_d
*/
template <class FPTYPE>
__global__ void kernel_set_spi_S_pop( GPUData_kernel<FPTYPE> S,  const GPUData_kernel<FPTYPE> lambda_v,
                               const GPUData_kernel<int> S_idx, const unsigned int col) {
    for(size_t nn = blockIdx.x * blockDim.x + threadIdx.x; nn < S.x; nn += gridDim.x * blockDim.x) {
        S[nn] = lambda_v(S_idx[nn] % lambda_v.x, col);
    }
}

template <class FPTYPE>
__global__ void kernel_PointWiseMultiply_derivativeSetup_pop( GPUData_kernel<FPTYPE> lambda_d,  const GPUData_kernel<FPTYPE> lambda_v) {
    size_t row_start = blockIdx.x * blockDim.x;
    size_t col_start = blockIdx.y * blockDim.y;
    size_t sec_start = blockIdx.z * blockDim.z;
    for(size_t row_0 = row_start; row_0 < lambda_d.x; row_0 += blockDim.x * gridDim.x) {
        size_t row = row_0 + threadIdx.x;
        for(size_t col_0 = col_start; col_0 < lambda_d.y; col_0 += blockDim.y * gridDim.y) {
            size_t col = col_0 + threadIdx.y;
            for(size_t sec_0 = sec_start; sec_0 < lambda_d.z; sec_0 += blockDim.z * gridDim.z) {
                size_t sec = sec_0 + threadIdx.z;
                if(row < lambda_d.x && col < lambda_d.y && sec < lambda_d.z) {
                    lambda_d(row, col, sec) *= lambda_v(row, col);
                }
                __syncwarp();
            }
        }
    }
}

/*Kernel for each observation in a sparse run, for a group
*  ridx_sa_all must be assigned
*/
template <class FPTYPE>
__global__ void kernel_getGroupX_shared_full_pop(GPUData_kernel<FPTYPE> X_temp, const GPUData_kernel<FPTYPE> X,
                                    GPUData_kernel<int> iX,      
                                    GPUData_kernel<unsigned int> ridx_sa_all,
                                    const bool isIdentity)   {
    //get current observation number
    unsigned int tt_start = blockIdx.y * blockDim.y;
    size_t row_start = blockIdx.x * blockDim.x;
    for(size_t row_0 = row_start; row_0 < X_temp.x; row_0 += blockDim.x * gridDim.x) {
        //for each regressor (on this thread)
        for(unsigned int tt_0 = tt_start; tt_0 < X.y; tt_0 += blockDim.y * gridDim.y) {
            size_t row = row_0 + threadIdx.x;
            size_t tt  = tt_0 + threadIdx.y;

            if(row < X_temp.x && tt < X.y) {
                size_t iX_row;
                iX_row = ridx_sa_all[row];
                //for each event 
                for(unsigned int aa = 0; aa < iX.y; aa++) {
                    int idx_0 = iX(iX_row, aa);
                    if(idx_0 < 0 || idx_0 >= X.x) {
                        X_temp(row, tt, aa) = 0;
                    }
                    else {
                        if(isIdentity) {
                            X_temp(row, tt, aa) = (idx_0 == tt) ?  1 : 0;
                        }
                        else {
                            X_temp(row, tt, aa) = X(idx_0, tt);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

template <class FPTYPE>
void GPUGMLMPop_dataset_Group_GPU<FPTYPE>::computeDerivatives(GPUGMLM_results_Group_GPU<FPTYPE> * results, const bool isSparseRun, const bool update_weights, GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle, const cusparseHandle_t cusparseHandle, cudaEvent_t & main_LL_event) {
    if(params->dim_R() == 0) {
        return; //nothing to compute
    }
    this->checkCudaErrors(cudaStreamWaitEvent(stream, main_LL_event, 0), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors: could not wait for stream");
    
    if(opts->compute_dV) {
        //for each neuron
        this->checkCudaErrors(parent->dLL->GEMM(results->dV, lambda_v, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:  dLL'*lambda_v -> dV failed");
    }

    //check if computing any derivatives first
    std::vector<bool> compute_dF;
    compute_dF.assign(dim_D(), false);
    for(unsigned int ss = 0; ss < params->dim_S(); ss++) {
        unsigned int dd = (*(params->factor_idx))[ss];
        compute_dF[dd] = compute_dF[dd] || opts->compute_dT[ss];
    }

    // compute lambda_v = dLL * V
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        if(compute_dF[dd]) {
            this->checkCudaErrors(parent->dLL->GEMM(lambda_v, params->V, cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:  dLL*V -> lambda_v failed");
            break;
        }
    }

    //for each factor
    for(unsigned int dd = 0; dd < dim_D(); dd++) {
        if(compute_dF[dd]) {
            // lambda_d init setup in the kernel call in computeRateParts 
            // two steps
            //  lambda_d = lambda_d .* lambda_v
            //  matrix mult of X'*(lambda_d)
                    
            GPUData<FPTYPE> * phi_c;
            GPUData<FPTYPE> * X_c;
            if((*isShared)[dd] && !isSparseRun) { // only do this if doing full run
                //this step is faster with sparse matrices for shared regressors

                //call kernel to setup entries spi to dLL 
                dim3 block_size;
                block_size.x = 1024;
                dim3 grid_size;
                size_t max_blocks_needed  = spi_rows[dd]->size()/ block_size.x + ((spi_rows[dd]->size() % block_size.x == 0)? 0:1);
                size_t blocks_to_use = (parent->dim_J() == 1) ? 1024 : 512;
                grid_size.x  = min(max_blocks_needed, blocks_to_use);

                FPTYPE alpha = 1;
                FPTYPE beta  = 0;
                for(unsigned int rr = 0; rr < params->dim_R(); rr++) {
                    kernel_set_spi_S_pop<<<grid_size, block_size, 0, stream>>>(spi_data[dd]->device(), lambda_v->device(),
                                                         spi_cols[dd]->device(), rr);
                    this->checkCudaErrors("GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:  kernel_set_spi_S launch failed");

                    //I found - on a 1080ti at least - doing this series of SpMV ops was typically faster than a single SpMM (annoyingly)
                    cusparseStatus_t cusparse_stat;
                    cusparse_stat = cusparseDnVecSetValues(*(spi_lambda_d[dd]), lambda_d[dd]->getData_gpu() + rr*lambda_d[dd]->getLD_gpu());
                    this->checkCudaErrors(cusparse_stat, "GPUGMLMPop_dataset_Group_GPU errors: cusparseDnVecSetValues failed for lambda_t.");
                    if((*isSharedIdentity)[dd]) {
                        cusparse_stat = cusparseDnVecSetValues(*(spi_phi_d[dd]), results->dF[dd]->getData_gpu() + rr*results->dF[dd]->getLD_gpu());
                    }
                    else {
                        cusparse_stat = cusparseDnVecSetValues(*(spi_phi_d[dd]), phi_d->getData_gpu() + rr*phi_d->getLD_gpu());
                    }
                    this->checkCudaErrors(cusparse_stat, "GPUGMLMPop_dataset_Group_GPU errors: cusparseDnVecSetValues failed for phi_d.");
                       
                    cusparse_stat = cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 *(spi_S[dd]),
                                 *(spi_lambda_d[dd]),
                                 &beta,
                                 *(spi_phi_d[dd]),
                                 getCudaType<FPTYPE>(),
                                 CUSPARSE_SPMV_COO_ALG1, //CUSPARSE_SPMV_ALG_DEFAULT,
                                 spi_buffer[dd]->getData_gpu()); //spi_buffer[dd]->getData_gpu()

                    this->output_stream << "GPUGMLMPop_dataset_Group_GPU errors: S*lambda->phi_t SpMV failed.\n";
                    this->output_stream << " rr = " << rr << ", dim_R = " << params->dim_R() << ", dd = " << dd << "\n";
                    this->output_stream << "buffer size " << spi_buffer_size[dd]  << "\n";
                    const char * cc = spi_buffer[dd]->getData_gpu();
                    long long cc2 = reinterpret_cast<long long>(cc);
                    this->output_stream << "buffer address " << cc2   << " (% 128 = " <<  cc2 % 128 << ")" << "\n";

                    this->checkCudaErrors(cusparse_stat, output_stream.str().c_str());

                    this->output_stream.str("");
                    this->output_stream.clear();
                }

                X_c   = X[dd];
                phi_c = phi_d;
            }
            else { 
                if((*isShared)[dd] && update_weights) { 
                    //  if doing sparse run with shared regressor, builds temporary X matrix (local regressors)
                    dim3 block_size;
                    block_size.y = 1;
                    block_size.x = 1024 / block_size.y;
                    dim3 grid_size;
                    size_t max_blocks_needed  = X_temp[dd]->getSize(0)  / block_size.x + ((X_temp[dd]->getSize(0)  % block_size.x == 0)? 0:1);
                    size_t blocks_to_use = (parent->dim_J() == 1) ? 1024 : 512;
                    grid_size.x  = min(max_blocks_needed, blocks_to_use);
                    grid_size.y = 1;

                    kernel_getGroupX_shared_full_pop<<<grid_size, block_size, 0, stream>>>(X_temp[dd]->device(), X[dd]->device(), 
                                                    iX[dd]->device(), 
                                                    parent->ridx_sa_all->device(),
                                                    (*isSharedIdentity)[dd]);
                    this->checkCudaErrors("GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:  kernel_getGroupX_shared_full launch failed");
                }
                
                // if local regressors
                dim3 block_size;
                if(dim_R() > 8) {
                    block_size.y = 4;
                }
                else  if(dim_R() > 4) {
                    block_size.y = 2;
                }
                else {
                    block_size.y = 1;
                }
                block_size.x = 1024 / block_size.y;
                dim3 grid_size;
                size_t max_blocks_needed = lambda_v->getSize(0) / block_size.x + ((lambda_v->getSize(0) % block_size.x == 0)? 0:1);
                size_t blocks_to_use = (parent->dim_J() == 1) ? 1024 : 512;
                grid_size.x  = min(max_blocks_needed, blocks_to_use);
                
                grid_size.y = dim_R()  / block_size.y + ((dim_R()  % block_size.y == 0)? 0:1);
                        
                kernel_PointWiseMultiply_derivativeSetup_pop<<<grid_size, block_size, 0, stream>>>(lambda_d[dd]->device(), lambda_v->device());
                this->checkCudaErrors("GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:  kernel_PointWiseMultiply_derivativeSetup launch failed");

                if(isSparseRun) {
                    // if sparse run
                    X_c     = X_temp[dd];
                    phi_c = lambda_d[dd];
                }
                else {
                    // if local regressors and full run
                    X_c   = X[dd];
                    phi_c = lambda_d[dd];
                }
            }

            this->checkCudaErrors(phi_c->resize(stream, X_c->getSize(0), results->dF[dd]->getSize(1), X_c->getSize(2)), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors: setting size of phi_c failed");

            // matrix mult to get dF (local and shared)
            if((*isShared)[dd] && !isSparseRun && (*isSharedIdentity)[dd]) {
                //nothing needed
            }
            else {
                //int mt;
                //cublasStatus_t cse = X_c->GEMM(results->dF[dd], phi_c, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 0, NULL, &mt);
                //output_stream << "DT multiplication type " << mt << "\n";
                //this->msg->printMsgTxt(output_stream);
                cublasStatus_t cse = X_c->GEMM(results->dF[dd], phi_c, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N);
                this->checkCudaErrors(static_cast<cudaError_t>(cse), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors:   X'*phi -> dF");
            }
            
            // matrix mults to get dT
            if((*(params->N_per_factor))[dd] > 1) {
                for(unsigned int ss = 0; ss < params->dim_S(); ss++) {
                    if((*(params->factor_idx))[ss] == dd && opts->compute_dT[ss]) {
                        this->checkCudaErrors(params->dF_dT[ss]->GEMVs(results->dT[ss], results->dF[dd], cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N), "GPUGMLMPop_dataset_Group_GPU::computeDerivatives errors: dF_dT'*dF -> dT");
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
template class GPUGMLMPop_computeBlock<float>;
template class GPUGMLMPop_computeBlock<double>;

template class GPUGMLMPop_dataset_Group_GPU<float>;
template class GPUGMLMPop_dataset_Group_GPU<double>;
template class GPUGMLMPop_dataset_GPU<float>;
template class GPUGMLMPop_dataset_GPU<double>;
};//namespace