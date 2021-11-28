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
#include "kcGMLMPop_dataStructures.hpp"
#include "kcGMLMPop_computeBlock.hpp"

namespace kCUDA {
    
template <class FPTYPE>
GPUGMLMPop_computeBlock<FPTYPE>::GPUGMLMPop_computeBlock(const GPUGMLMPop_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLMPop_GPU_block_args<FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_) {
    msg = msg_;
    dev = block->dev_num;
    switchToDevice();
    dim_J = GMLMPopstructure->Groups.size();

    size_t dim_M = block->trials.size();
    if(dim_M == 0) {
        output_stream << "GPUGMLMPop_computeBlock errors: no trials in block!";
        msg->callErrMsgTxt(output_stream);
    }   

    //setup the streams
    checkCudaErrors(cudaStreamCreate(&(stream)), "GPUGMLMPop_computeBlock errors: failed initializing stream!");
    stream_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cudaStreamCreate(&(stream_Groups[jj])), "GPUGMLMPop_computeBlock errors: failed initializing group streams!");
    }

    //setup cublas handles
    checkCudaErrors(cublasCreate(&(cublasHandle)), "GPUGMLMPop_computeBlock errors: CUBLAS initialization failed.");
    checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cublas pointer mode failed.");
    checkCudaErrors(cublasSetStream(cublasHandle, stream), "GPUGMLMPop_computeBlock errors: set cublas stream failed.");
    cublasHandle_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cublasCreate(&(cublasHandle_Groups[jj])), "GPUGMLMPop_computeBlock errors: CUBLAS groups initialization failed.");
        checkCudaErrors(cublasSetPointerMode(cublasHandle_Groups[jj], CUBLAS_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cublas groups pointer mode failed.");
        checkCudaErrors(cublasSetStream(cublasHandle_Groups[jj], stream_Groups[jj]), "GPUGMLMPop_computeBlock errors: set cublas groups stream failed.");
    }

    //setup cusparse handle
    cusparseHandle_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cusparseCreate(       &(cusparseHandle_Groups[jj])), "GPUGMLMPop_computeBlock errors: cusparse groups initialization failed.");
        checkCudaErrors(cusparseSetPointerMode(cusparseHandle_Groups[jj], CUSPARSE_POINTER_MODE_HOST), "GPUGMLMPop_computeBlock errors: set cusparse groups pointer mode failed.");
        checkCudaErrors(cusparseSetStream(      cusparseHandle_Groups[jj], stream_Groups[jj]), "GPUGMLMPop_computeBlock errors: set cusparse groups stream failed.");
    }

    //setup the parameter structure
    params = new GPUGMLMPop_parameters_GPU<FPTYPE>(GMLMPopstructure, dim_M, dev, msg);
    //params = NULL;

    //setup the results structure
    results = new GPUGMLMPop_results_GPU<FPTYPE>(GMLMPopstructure, max_trials_, dev, msg);
    //results = NULL;
            
    //setup the dataset structure
    dataset = new GPUGMLMPop_dataset_GPU<FPTYPE>(GMLMPopstructure, block, max_trials_, stream, cusparseHandle_Groups, msg);
    //dataset = NULL;
    checkCudaErrors(cudaEventCreate(&LL_event), "GPUGMLMPop_computeBlock errors: could not create LL event!");
}

template <class FPTYPE>
GPUGMLMPop_computeBlock<FPTYPE>::~GPUGMLMPop_computeBlock() {
    switchToDevice();
    
    delete results;
    delete params;
    delete dataset;

    checkCudaErrors(cudaEventDestroy(LL_event), "GPUGMLMPop_computeBlock errors: could not clear LL event!");


    //destroy cublas handles
    checkCudaErrors(cublasDestroy(cublasHandle), "GPUGMLMPop_computeBlock errors: failed to destroy cublas handle." );
    for(auto jj : cublasHandle_Groups) {
        checkCudaErrors(cublasDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group cublas handles." );
    }
    for(auto jj : cusparseHandle_Groups) {
        checkCudaErrors(cusparseDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group cusparse handles." );
    }
       
    //destroy streams
    checkCudaErrors(cudaStreamDestroy(stream), "GPUGMLMPop_computeBlock errors: failed destroying stream!");
    for(auto jj : stream_Groups) {
        checkCudaErrors(cudaStreamDestroy(jj), "GPUGMLMPop_computeBlock errors: failed to destroy group streams." );
    }  
}

template <class FPTYPE>
bool GPUGMLMPop_computeBlock<FPTYPE>::loadParams(const GPUGMLMPop_params<FPTYPE> * params_host, const GPUGMLMPop_computeOptions<FPTYPE> * opts) { 
	switchToDevice();
    params->copyToGPU(params_host, dataset, stream, stream_Groups, opts);
    for(int jj = 0; jj < params->dim_J(); jj++) {
        checkCudaErrors(results->set_dim_R(jj, params->dim_R(jj), stream), "GPUGMLMPop_computeBlock::loadParams errors: could not set results dim_R");
    }
    bool isSparseRun = dataset->isSparseRun(params);
    if(params->getNumberOfNonzeroWeights() > 0) { //make sure there's something to compute
        results_set = true;

        //for each group, multiply coefficients by X*T -> XT
        for(int jj = 0; jj < dim_J && jj < dataset->dim_J(); jj++) {
            dataset->Groups[jj]->multiplyCoefficients(isSparseRun, params->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj], params->paramsLoaded_event);
        }
    }
    else {
        results_set = false;
    }
    return isSparseRun;
}
        
template <class FPTYPE>
void GPUGMLMPop_computeBlock<FPTYPE>::computeRateParts(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    switchToDevice();

    //for each group
    for(int jj = 0; jj < dataset->dim_J(); jj++ ) {
        dataset->Groups[jj]->getGroupRate(isSparseRun,  params->Groups[jj], opts->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj]);
    }
}

/*Kernel for each observation
 * Summarizes the contributions from each tensor group (lambda), linear term (X_lin,B), baseline rate (w,log_dt)
 * Returns the  observation-wise log like (LL - no normalizing constant) and it's derivative portion (dLL)
 *
 *  for sparse runs with compute_dB, saves out the partial X_lin into X_lin_temp
 */        
template <class FPTYPE>
__global__ void kernel_getObs_LL(GPUData_kernel<FPTYPE> LL, GPUData_kernel<FPTYPE> dLL,
        const GPUData_kernel<FPTYPE> Y,
        const GPUData_kernel<FPTYPE> lambda,
        const GPUData_kernel<FPTYPE> X_lin ,
        const GPUData_kernel<FPTYPE> B     , 
        const GPUData_kernel<FPTYPE> W, 
        const FPTYPE log_dt,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights,
        const GPUData_kernel<unsigned int> ridx_sa_all,
        GPUData_kernel<FPTYPE> X_lin_temp, const bool compute_dB,
        const logLikeType logLikeSettings, const GPUData_kernel<FPTYPE> logLikeParams) {
    //current observation index
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pp  = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < LL.x && pp < LL.y) {
        size_t Xlin_row;
        if(ridx_sa_all.y < 1) {
            //if full run
            Xlin_row = row;
        }
        else {
            //if sparse run
            Xlin_row = ridx_sa_all[row];
        }
        FPTYPE tw_c = 1;
        if(trial_weights.y == 1) {
            tw_c = trial_weights[id_a_trialM[Xlin_row]];
        }
        else if(trial_weights.y > 1) {
            tw_c = trial_weights(id_a_trialM[Xlin_row], pp);
        }

        FPTYPE Y_c = Y(Xlin_row, pp);

        FPTYPE  LL_c = 0;  
        FPTYPE dLL_c = 0;    
        if(tw_c != 0) { //if trial not censored
            FPTYPE log_rate = W[pp];
            for(int bb = 0; bb < X_lin.y; bb++) {
                log_rate += X_lin(Xlin_row, bb, pp) * B(bb, pp);
                if(ridx_sa_all.y > 0 && compute_dB && (pp == 0 || X_lin_temp.z > 1)) { // for dB when doing sparse run
                    X_lin_temp(row, bb, pp) = X_lin(Xlin_row, bb, pp);
                }
            }
            for(int jj = 0; jj < lambda.z; jj++) {
                log_rate += lambda(row, pp, jj);
            }

            if(logLikeSettings == ll_poissExp) {
                int Y_ci = floor(Y_c);
                if(Y_ci >= 0) { // negatives get censored by Poisson LL
                    log_rate += log_dt;
                    FPTYPE rate = safeExp(log_rate);
                     LL_c = (-rate + Y_ci * log_rate);
                    dLL_c = (-rate + Y_ci);
                }
            }
            else if(logLikeSettings == ll_sqErr) {
                FPTYPE eY_c = log_rate - Y_c;
                 LL_c = -(eY_c*eY_c);
                dLL_c = -2*eY_c;
            }
            else if(logLikeSettings == ll_truncatedPoissExp) {
                int Y_ci = floor(Y_c);
                if(Y_ci > 0) { 
                    log_rate += log_dt;
                    if(log_rate > -30) {
                        FPTYPE rate = safeExp(log_rate);
                         LL_c = log(1 - safeExp(-rate));
                        dLL_c = rate/safeExpm1(rate);
                    }
                    else { // more numerically save approximation in an extreme case
                         LL_c = log_rate;
                        dLL_c = 1;
                    }
                }
                else if(Y_ci == 0) {
                    FPTYPE rate = safeExp(log_rate + log_dt);
                     LL_c = -rate;
                    dLL_c = -rate;
                }
                // negatives get censored by Poisson LL
            }
            else if(logLikeSettings == ll_poissExpRefractory) {
                // ll_poissExpRefractory uses the correction from Citi, L., Ba, D., Brown, E. N., & Barbieri, R. (2014). Likelihood methods for point processes with refractoriness. Neural computation, 26(2), 237-263.
                int Y_ci = floor(Y_c);
                if(Y_ci >= 0) { // negatives get censored by Poisson LL
                    log_rate += log_dt;
                    FPTYPE rate = safeExp(log_rate);
                     LL_c = (-(1-Y_ci/2)*rate + Y_ci * log_rate);
                    dLL_c = (-(1-Y_ci/2)*rate + Y_ci);
                }
            }
        }
         LL(row, pp) =  LL_c*tw_c;
        dLL(row, pp) = dLL_c*tw_c;
    }
}

/* Kernel for each trial
*  Sums up the trial log likelihoods (results->trialLL)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)*/
template <class FPTYPE>
__global__ void kernel_sum_trialLL(GPUData_kernel<FPTYPE> trialLL, GPUData_kernel<FPTYPE> dW_trial, 
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
        if(trial_weights.y == 1) {
            tw_c = trial_weights[mm];
        }
        else if(trial_weights.y > 1) {
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
void GPUGMLMPop_computeBlock<FPTYPE>::computeLogLike(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }

    switchToDevice();
         //launch kernel to sum lambda, X_lin*B -> LL, dLL (launch over all observations)
         //launch kernel to sum lambda for each trial
    
    //X_lin*B + W + log_dt + sum(lambda) -> LL for each neuron
    checkCudaErrors(dataset->waitForGroups_LL(stream), "GPUGMLM_computeBlock::computeLogLike errors:  waitForGroups_LL failed");

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

    if(opts->compute_dB && isSparseRun) {
        dataset->X_lin_temp->resize(stream, dataset->LL->getSize(0));
    }
        
    block_size.x = 1024/block_size.y;
    dim3 grid_size;
    grid_size.x = dataset->LL->getSize(0) / block_size.x + ( (dataset->LL->getSize(0) % block_size.x == 0) ? 0 : 1);
    grid_size.y = dataset->dim_P() / block_size.y + ( (dataset->dim_P() % block_size.y == 0) ? 0 : 1);
    kernel_getObs_LL<<<grid_size, block_size, 0, stream>>>(dataset->LL->device(), dataset->dLL->device(),
                  dataset->Y->device(),
                  dataset->lambda->device(),
                  dataset->X_lin->device(),
                   params->B->device(),
                   params->W->device(), dataset->log_dt,
                  dataset->id_a_trialM->device(),
                  params->trial_weights->device(),
                  dataset->ridx_a_all_c->device(),
                  dataset->X_lin_temp->device(), opts->compute_dB, 
                   params->logLikeSettings, params->logLikeParams->device());
    checkCudaErrors("GPUGMLMPop_computeBlock::computeLogLike errors:  kernel_getObs_LL launch failed");
    checkCudaErrors(cudaEventRecord(LL_event, stream), "GPUGMLMPop_computeBlock::computeLogLike errors: could not add LL event to stream!");

    //sum up the LL for each trial (and dLL to setup for dW, dB)
    if(opts->compute_trialLL || opts->compute_dW) {
        //same block size

        size_t dim_M_c = params->getNumberOfNonzeroWeights();
        grid_size.x = params->getNumberOfNonzeroWeights()  / block_size.x + ( (params->getNumberOfNonzeroWeights()  % block_size.x == 0) ? 0 : 1);
        grid_size.y = dataset->dim_P() / block_size.y + ( (dataset->dim_P() % block_size.y == 0) ? 0 : 1);

        kernel_sum_trialLL<<<grid_size, block_size, 0, stream>>>(results->trialLL->device(), dataset->dW_trial->device(),
                                                                 params->trial_included->device(), 
                                                                 dataset->LL->device(), dataset->dLL->device(), 
                                                                 opts->compute_trialLL, opts->compute_dW, 
                                                                 dataset->dim_N->device(),
                                                                 dataset->ridx_t_all_c->device(),
                                                                 dataset->id_t_trial->device(),
                                                                 params->trial_weights->device(),
                                                                 dataset->normalizingConstants_trial->device());
        checkCudaErrors("GPUGMLMPop_computeBlock::computeLogLike errors:  kernel_sum_trialLL launch failed");
    }
}

/* Kernel for each neuron
*  Sums up the trial dW (results->dW)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)
*/
template <class FPTYPE>
__global__ void kernel_sum_dW( GPUData_kernel<FPTYPE> dW,  const GPUData_kernel<FPTYPE> dW_trial, const GPUData_kernel<FPTYPE> trial_weights) {
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
void GPUGMLMPop_computeBlock<FPTYPE>::computeDerivatives(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    switchToDevice();
         //launch kernel to sum dLL -> dW, dB for each trial?
         //         or kernel to sum up dLL->dW and GEMV for dB?
         
    //for each Group
    for(int jj = 0; jj < dim_J; jj++) {
        dataset->Groups[jj]->computeDerivatives(results->Groups[jj], isSparseRun, params->Groups[jj], opts->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj], cusparseHandle_Groups[jj], LL_event);
    }   
    
    if(opts->compute_dW) {
        dim3 block_size;
        block_size.x = min(dataset->dim_P(), static_cast<size_t>(1024));
        dim3 grid_size;
        grid_size.x = dataset->dim_P() / block_size.x + ((dataset->dim_P() % block_size.x == 0)? 0:1);

      /*  output_stream << results->dW->getDevice() << "  " << dataset->dW_trial->getDevice() << "  " << params->trial_weights->getDevice() << "\n";
        msg->printMsgTxt(output_stream);*/


        kernel_sum_dW<<<grid_size, block_size, 0, stream>>>(results->dW->device(), dataset->dW_trial->device(), params->trial_weights->device());
        checkCudaErrors("GPUGMLMPop_computeBlock::computeDerivatives errors:  kernel_sum_dW launch failed");
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
        checkCudaErrors(ce, "GPUGMLMPop_computeBlock::computeDerivatives errors:  X_lin'*dLL -> dB failed");
    }      
}
       
//explicitly create classes for single and double precision floating point for library
template class GPUGMLMPop_computeBlock<float>;
template class GPUGMLMPop_computeBlock<double>;

};//namespace