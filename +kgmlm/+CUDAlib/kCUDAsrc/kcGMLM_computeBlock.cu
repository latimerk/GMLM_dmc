/*
 * kcGMLM_computeBlock.cu
 * Computations for a GMLM+derivatives (on one GPU).
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
#include "kcGMLM_computeBlock.hpp"

namespace kCUDA {
    
template <class FPTYPE>
GPUGMLM_computeBlock<FPTYPE>::GPUGMLM_computeBlock(const GPUGMLM_structure_args<FPTYPE> * GMLMstructure, const GPUGMLM_GPU_block_args<FPTYPE> * block, const size_t max_trials_, const size_t dim_P_, std::shared_ptr<GPUGL_msg> msg_) {
    msg = msg_;
    dev = block->dev_num;
    switchToDevice();
    dim_J = GMLMstructure->Groups.size();

    size_t dim_M = block->trials.size();
    if(dim_M == 0) {
        output_stream << "GPUGMLM_computeBlock errors: no trials in block!";
        msg->callErrMsgTxt(output_stream);
    }   

    //setup the streams
    checkCudaErrors(cudaStreamCreate(&(stream)), "GPUGMLM_computeBlock errors: failed initializing stream!");
    stream_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cudaStreamCreate(&(stream_Groups[jj])), "GPUGMLM_computeBlock errors: failed initializing group streams!");
    }

    //setup cublas handles
    checkCudaErrors(cublasCreate(&(cublasHandle)), "GPUGMLM_computeBlock errors: CUBLAS initialization failed.");
    checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "GPUGMLM_computeBlock errors: set cublas pointer mode failed.");
    checkCudaErrors(cublasSetStream(cublasHandle, stream), "GPUGMLM_computeBlock errors: set cublas stream failed.");
    cublasHandle_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cublasCreate(&(cublasHandle_Groups[jj])), "GPUGMLM_computeBlock errors: CUBLAS groups initialization failed.");
        checkCudaErrors(cublasSetPointerMode(cublasHandle_Groups[jj], CUBLAS_POINTER_MODE_HOST), "GPUGMLM_computeBlock errors: set cublas groups pointer mode failed.");
        checkCudaErrors(cublasSetStream(cublasHandle_Groups[jj], stream_Groups[jj]), "GPUGMLM_computeBlock errors: set cublas groups stream failed.");
    }

    //setup cusparse handle
    cusparseHandle_Groups.resize(dim_J);
    for(int jj = 0; jj < dim_J; jj++) {
        checkCudaErrors(cusparseCreate(       &(cusparseHandle_Groups[jj])), "GPUGMLM_computeBlock errors: cusparse groups initialization failed.");
        checkCudaErrors(cusparseSetPointerMode(cusparseHandle_Groups[jj], CUSPARSE_POINTER_MODE_HOST), "GPUGMLM_computeBlock errors: set cusparse groups pointer mode failed.");
        checkCudaErrors(cusparseSetStream(      cusparseHandle_Groups[jj], stream_Groups[jj]), "GPUGMLM_computeBlock errors: set cusparse groups stream failed.");
    }

    //setup the parameter structure
    params = new GPUGMLM_parameters_GPU<FPTYPE>(GMLMstructure, dim_M, dim_P_, dev, msg);
    //params = NULL;

    //setup the results structure
    results = new GPUGMLM_results_GPU<FPTYPE>(GMLMstructure, max_trials_, dim_P_, dev, msg);
    //results = NULL;
            
    //setup the dataset structure
    dataset = new GPUGMLM_dataset_GPU<FPTYPE>(GMLMstructure, block, max_trials_, dim_P_, stream, cusparseHandle_Groups, msg);
    //dataset = NULL;

    checkCudaErrors(cudaEventCreate(&LL_event), "GPUGMLM_computeBlock errors: could not create LL event!");
}

template <class FPTYPE>
GPUGMLM_computeBlock<FPTYPE>::~GPUGMLM_computeBlock() {
    switchToDevice();
    
    delete results;
    delete params;
    delete dataset;

    checkCudaErrors(cudaEventDestroy(LL_event), "GPUGMLM_computeBlock errors: could not clear LL event!");

    //destroy cublas handles
    checkCudaErrors(cublasDestroy(cublasHandle), "GPUGMLM_computeBlock errors: failed to destroy cublas handle." );
    for(auto jj : cublasHandle_Groups) {
        checkCudaErrors(cublasDestroy(jj), "GPUGMLM_computeBlock errors: failed to destroy group cublas handles." );
    }
    for(auto jj : cusparseHandle_Groups) {
        checkCudaErrors(cusparseDestroy(jj), "GPUGMLM_computeBlock errors: failed to destroy group cusparse handles." );
    }
       
    //destroy streams
    checkCudaErrors(cudaStreamDestroy(stream), "GPUGMLM_computeBlock errors: failed destroying stream!");
    for(auto jj : stream_Groups) {
        checkCudaErrors(cudaStreamDestroy(jj), "GPUGMLM_computeBlock errors: failed to destroy group streams." );
    }  
}


template <class FPTYPE>
bool GPUGMLM_computeBlock<FPTYPE>::loadParams(const GPUGMLM_params<FPTYPE> * params_host, const GPUGMLM_computeOptions<FPTYPE> * opts) { 
	switchToDevice();
    params->copyToGPU(params_host, dataset, stream, stream_Groups, opts);
    for(int jj = 0; jj < params->dim_J(); jj++) {
        checkCudaErrors(results->set_dim_R(jj, params->dim_R(jj), stream), "GPUGMLM_computeBlock::loadParams errors: could not set results dim_R");
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
void GPUGMLM_computeBlock<FPTYPE>::computeRateParts(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    switchToDevice();

    //for each group
    for(int jj = 0; jj < dataset->dim_J(); jj++ ) {
        dataset->Groups[jj]->getGroupRate(isSparseRun,  params->Groups[jj], opts->Groups[jj], stream_Groups[jj]);
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
        const GPUData_kernel<FPTYPE> W, const FPTYPE log_dt,
        const GPUData_kernel<unsigned int> id_a_neuron,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights,
        const GPUData_kernel<unsigned int> ridx_sa_all,
        GPUData_kernel<FPTYPE> X_lin_temp, const bool compute_dB,
        const logLikeType logLikeSettings, const GPUData_kernel<FPTYPE> logLikeParams) {
    //current observation index
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < LL.x) {
        size_t Xlin_row;
        if(ridx_sa_all.y < 1) {
            //if full run
            Xlin_row = row;
        }
        else {
            //if sparse run
            Xlin_row = ridx_sa_all[row];
        }
        size_t neuron_num = id_a_neuron[Xlin_row];
        FPTYPE tw_c = (trial_weights.y < 1) ? 1 : trial_weights[id_a_trialM[Xlin_row]];

        FPTYPE Y_c = Y[Xlin_row];

        FPTYPE  LL_c = 0;  
        FPTYPE dLL_c = 0;    
        if(tw_c != 0) { //if trial not censored
            FPTYPE log_rate = W[neuron_num];
            for(int bb = 0; bb < X_lin.y; bb++) {
                log_rate += X_lin(Xlin_row, bb) * B(bb, neuron_num);
                if(ridx_sa_all.y > 0 && compute_dB) { // for dB when doing sparse run
                    X_lin_temp(row, bb) = X_lin(Xlin_row, bb);
                }
            }
            for(int jj = 0; jj < lambda.y; jj++) {
                log_rate += lambda(row, jj);
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
                        FPTYPE expNrate = safeExp(-rate);
                         LL_c = log(1 - expNrate);
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
        LL[row]  =  LL_c*tw_c;
        dLL[row] = dLL_c*tw_c;
    }
}
/* Kernel for each trial
*  Sums up the trial log likelihoods (results->trialLL)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)*/
template <class FPTYPE>
__global__ void kernel_sum_trialLL(GPUData_kernel<FPTYPE> trialLL, GPUData_kernel<FPTYPE> dW_trial,
                                 const GPUData_kernel<unsigned int>  trial_included, 
                                 const GPUData_kernel<FPTYPE> LL, const GPUData_kernel<FPTYPE> dLL, 
                                 const bool compute_trialLL, const bool compute_dW, 
                                 const GPUData_kernel<size_t> dim_N,
                                 const GPUData_kernel<unsigned int> ridx_t_all,
                                 const GPUData_kernel<unsigned int> id_t_trial,
                                 const GPUData_kernel<FPTYPE> trial_weights,
                                 const GPUData_kernel<FPTYPE> normalizingConstants) {
    size_t tr = blockIdx.x * blockDim.x + threadIdx.x;
    size_t mm = dim_N.x; //default is invalid value - will just skip
    if(trial_included.y > 0) { //if is sparse run
        if(tr < trial_included.x) {
            mm = trial_included[tr];
        }
    }
    else {
        mm = tr;
    }

    if(mm < dim_N.x) { // if valid trial
        FPTYPE tw_c = (trial_weights.y < 1) ? 1 : trial_weights[mm];

        if(tw_c != 0) {  
            unsigned int row = ridx_t_all[tr];  // this uses 'tr' so that it works for sparse runs

            //sum up LL
            FPTYPE ll_total = normalizingConstants[mm] * tw_c;
            FPTYPE dLL_total = 0;
            for(int tt = 0; tt < dim_N[mm]; tt++) {
                if(compute_trialLL) {
                    ll_total  += LL[row + tt];
                }
                if(compute_dW) {
                    dLL_total += dLL[row + tt];
                }
            }
            if(compute_trialLL) {
                trialLL[id_t_trial[mm]] = ll_total;
            }
            if(compute_dW) {
                dW_trial[mm] = dLL_total;
            }
        }
        //no need to compute sum; set results to 0
        else {
            if(compute_trialLL) {
                trialLL[id_t_trial[mm]] = 0;
            }
            if(compute_dW) {
                dW_trial[mm] = 0;
            }
        }
    }
}

template <class FPTYPE>
void GPUGMLM_computeBlock<FPTYPE>::computeLogLike(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }

    switchToDevice();
         //launch kernel to sum lambda, X_lin*B -> LL, dLL (launch over all observations)
         //launch kernel to sum lambda for each trial
    
    //X_lin*B + W + log_dt + sum(lambda) -> LL for each neuron
            
    checkCudaErrors(dataset->waitForGroups_LL(stream), "GPUGMLM_computeBlock::computeLogLike errors:  waitForGroups_LL failed");
    if(opts->compute_dB && isSparseRun) {
        dataset->X_lin_temp->resize(stream, dataset->LL->getSize(0));
    }

    dim3 block_size;
    block_size.x = 1024;
    dim3 grid_size;
    grid_size.x = dataset->LL->getSize(0) / block_size.x + ( (dataset->LL->getSize(0) % block_size.x == 0) ? 0 : 1);
    kernel_getObs_LL<<<grid_size, block_size, 0, stream>>>(dataset->LL->device(), dataset->dLL->device(),
                  dataset->Y->device(),
                  dataset->lambda->device(),
                  dataset->X_lin->device(),
                   params->B->device(),
                   params->W->device(), dataset->log_dt,
                  dataset->id_a_neuron->device(),
                  dataset->id_a_trialM->device(),
                  params->trial_weights->device(),
                  dataset->ridx_a_all_c->device(),
                  dataset->X_lin_temp->device(), opts->compute_dB, 
                   params->logLikeSettings, params->logLikeParams->device());
    checkCudaErrors("GPUGMLM_computeBlock::computeLogLike errors:  kernel_getObs_LL launch failed");

    //sum up the LL for each trial (and dLL to setup for dW, dB)
    if(opts->compute_trialLL || opts->compute_dW) {
        block_size.x = 1024;
        block_size.y = 1;

        size_t dim_M_c = params->getNumberOfNonzeroWeights();
        grid_size.x = params->getNumberOfNonzeroWeights()  / block_size.x + ( (params->getNumberOfNonzeroWeights()  % block_size.x == 0) ? 0 : 1);
        grid_size.y = 1;

        kernel_sum_trialLL<<<grid_size, block_size, 0, stream>>>(results->trialLL->device(), dataset->dW_trial->device(),
                                                                 params->trial_included->device(), 
                                                                 dataset->LL->device(), dataset->dLL->device(), 
                                                                 opts->compute_trialLL, opts->compute_dW, 
                                                                 dataset->dim_N->device(),
                                                                 dataset->ridx_t_all_c->device(),
                                                                 dataset->id_t_trial->device(),
                                                                 params->trial_weights->device(),
                                                                 dataset->normalizingConstants_trial->device());
        checkCudaErrors("GPUGMLM_computeBlock::computeLogLike errors:  kernel_sum_trialLL launch failed");
    }
    checkCudaErrors(cudaEventRecord(LL_event, stream), "GPUGMLM_computeBlock::computeLogLike errors: could not add LL event to stream!");
}


/* Kernel for each neuron
*  Sums up the trial dW (results->dW)
*   also sets up some derivative computations (dataset->dW_trial, dataset->dB_trial)
*/
template <class FPTYPE>
__global__ void kernel_sum_dW( GPUData_kernel<FPTYPE> dW,  const GPUData_kernel<FPTYPE> dW_trial, const GPUData_kernel<FPTYPE> trial_weights,
                                 const GPUData_kernel<unsigned int> ridx_n_tr) {
    size_t pp = blockIdx.x * blockDim.x + threadIdx.x;
    if(pp < dW.x) {
        unsigned int t_start = ridx_n_tr[pp];
        unsigned int t_end   = (pp == dW.x-1) ? dW_trial.x : ridx_n_tr[pp+1];

        FPTYPE dW_sum = 0;
        for(int tr = t_start; tr < t_end; tr++) {
            if(trial_weights.y == 0 || trial_weights[tr] != 0) {
                dW_sum += dW_trial[tr];
            }
        }
        dW[pp] = dW_sum;
    }
}

template <class FPTYPE>
void GPUGMLM_computeBlock<FPTYPE>::computeDerivatives(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    switchToDevice();
         //launch kernel to sum dLL -> dW, dB for each trial?
         //         or kernel to sum up dLL->dW and GEMV for dB?
    if(opts->compute_dW) {
        dim3 block_size;
        block_size.x = min(dataset->dim_P(), static_cast<size_t>(1024));
        dim3 grid_size;
        grid_size.x = dataset->dim_P() / block_size.x + ((dataset->dim_P() % block_size.x == 0)? 0:1);

        kernel_sum_dW<<<grid_size, block_size, 0, stream>>>(results->dW->device(), dataset->dW_trial->device(), params->trial_weights->device(),
                                                            dataset->ridx_n_tr->device());
        checkCudaErrors("GPUGMLM_computeBlock::computeDerivatives errors:  kernel_sum_dW launch failed");
    }

    if(opts->compute_dB && dataset->dim_B() > 0) {
        FPTYPE alpha = 1;
        FPTYPE beta  = 0;
        GPUData<FPTYPE> * X_lin_c = isSparseRun ?  dataset->X_lin_temp : dataset->X_lin;

        for(int pp = 0; pp < dataset->dim_P(); pp++) {
            if(dataset->dim_N_neuron_temp[pp] > 0) {// if there are any trials for this neuron in current GPU block
                unsigned int neuron_start_c = isSparseRun ? (*(dataset->ridx_sn_sall))[pp] : (*(dataset->ridx_n_all))[pp];
                    
                cublasStatus_t ce = cublasGEMV(cublasHandle,
                                               CUBLAS_OP_T,
                                               dataset->dim_N_neuron_temp[pp], dataset->dim_B(),
                                               &alpha,
                                               X_lin_c->getData_gpu()      + neuron_start_c, X_lin_c->getLD_gpu(),
                                               dataset->dLL->getData_gpu() + neuron_start_c, 1,
                                               &beta,
                                               results->dB->getData_gpu() + pp * results->dB->getLD_gpu(), 1);
                checkCudaErrors(ce, "GPUGMLM_computeBlock::computeDerivatives errors:  X_lin'*dLL -> dB failed");
            }
        }
    }
    //for each Group
    for(int jj = 0; jj < dim_J; jj++) {
        dataset->Groups[jj]->computeDerivatives(results->Groups[jj], isSparseRun, params->Groups[jj], opts->Groups[jj], stream_Groups[jj], cublasHandle_Groups[jj], cusparseHandle_Groups[jj], LL_event);
    }         
}
        

//explicitly create classes for single and double precision floating point for library


template class GPUGMLM_computeBlock<float>;
template class GPUGMLM_computeBlock<double>;

};//namespace