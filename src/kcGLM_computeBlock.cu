/*
 * kcGLM_computeBlock.cu
 * Computations for a GLM+derivatives (on one GPU).
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
#include "kcGLM_computeBlock.hpp"

namespace kCUDA {

template <class FPTYPE>
GPUGLM_computeBlock<FPTYPE>::GPUGLM_computeBlock(const GPUGLM_structure_args<FPTYPE> * GLMstructure, const GPUGLM_GPU_block_args<FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_) {
    msg = msg_;
    dev = block->dev_num;
    switchToDevice();

    size_t dim_M = block->trials.size();
    if(dim_M == 0) {
        output_stream << "GPUGLM_computeBlock errors: no trials in block!";
        msg->callErrMsgTxt(output_stream);
    }   

    //setup the streams
    checkCudaErrors(cudaStreamCreate(&(stream )), "GPUGLM_computeBlock errors: failed initializing stream!");
    checkCudaErrors(cudaStreamCreate(&(stream2)), "GPUGLM_computeBlock errors: failed initializing stream2!");
    
    //setup cublas handles
    checkCudaErrors(cublasCreate(&(cublasHandle)), "GPUGLM_computeBlock errors: CUBLAS initialization failed.");
    checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "GPUGLM_computeBlock errors: set cublas pointer mode failed.");
    checkCudaErrors(cublasSetStream(cublasHandle, stream), "GPUGLM_computeBlock errors: set cublas stream failed.");
    checkCudaErrors(cublasCreate(&(cublasHandle2)), "GPUGLM_computeBlock errors: CUBLAS initialization failed 2.");
    checkCudaErrors(cublasSetPointerMode(cublasHandle2, CUBLAS_POINTER_MODE_HOST), "GPUGLM_computeBlock errors: set cublas pointer mode failed 2.");
    checkCudaErrors(cublasSetStream(cublasHandle2, stream2), "GPUGLM_computeBlock errors: set cublas stream2 failed.");

    //setup the parameter structure
    params = new GPUGLM_parameters_GPU<FPTYPE>(GLMstructure, dim_M, dev, msg);

    //setup the results structure
    results = new GPUGLM_results_GPU<FPTYPE>(GLMstructure, max_trials_, dev, msg);

    //setup the dataset structure
    dataset = new GPUGLM_dataset_GPU<FPTYPE>(GLMstructure, block, max_trials_, stream, msg);
}

template <class FPTYPE>
GPUGLM_computeBlock<FPTYPE>::~GPUGLM_computeBlock() {
    switchToDevice();
    
    delete results;
    delete params;
    delete dataset;

    //destroy cublas handles
    checkCudaErrors(cublasDestroy(cublasHandle ), "GPUGLM_computeBlock errors: failed to destroy cublas handle." );
    checkCudaErrors(cublasDestroy(cublasHandle2), "GPUGLM_computeBlock errors: failed to destroy cublas handle." );

    //destroy streams
    checkCudaErrors(cudaStreamDestroy(stream ), "GPUGLM_computeBlock errors: failed destroying stream!");
    checkCudaErrors(cudaStreamDestroy(stream2), "GPUGLM_computeBlock errors: failed destroying stream!");
}

/*Kernel for each observation in a sparse run, for a group
 * Builds the dense regressor matrix with local regressors
*  ridx_sa_all must be assigned
*/
template <class FPTYPE>
__global__ void kernel_getX_temp(GPUData_kernel<FPTYPE> X_temp, const GPUData_kernel<FPTYPE> X, 
                                    const GPUData_kernel<unsigned int> ridx_sa_all)   {
    //get current observation number
    int tt_start = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < X_temp.x) {
        unsigned int iX_row = ridx_sa_all[row];

        //for each regressor (on this thread)
        for(int tt = tt_start; tt < X_temp.y; tt += blockDim.y * gridDim.y) {
            //for each event 
            X_temp(row, tt) = X(iX_row, tt);
        }
    }
}


template <class FPTYPE>
bool GPUGLM_computeBlock<FPTYPE>::loadParams(const GPUGLM_params<FPTYPE> * params_host, const GPUGLM_computeOptions<FPTYPE> * opts) { 
	switchToDevice();
    params->copyToGPU(params_host, dataset, stream, opts);
    bool isSparseRun = dataset->isSparseRun(params);

    if(params->getNumberOfNonzeroWeights() > 0) { //make sure there's something to compute
        results_set = true;
        
        GPUData<FPTYPE> * X_c;
 
        if(isSparseRun) {
            // builds temp X matrix
            checkCudaErrors(dataset->X_temp->resize(stream, dataset->dim_N_temp, -1), "GPUGLM_computeBlock::loadParams errors: could not resize X_temp");
                    
            dim3 block_size;
            if(params->dim_K() > 8) { 
                block_size.y = 8;
            }
            else if(params->dim_K() >= 4) { 
                block_size.y = 4;
            }
            block_size.x = 1024 / block_size.y;
            dim3 grid_size;
            grid_size.x = dataset->dim_N_temp / block_size.x + ((dataset->dim_N_temp  % block_size.x == 0)? 0:1);
            grid_size.y = params->dim_K() / 128 + ((params->dim_K()  % 128 == 0)? 0:1);
            kernel_getX_temp<<<grid_size, block_size, 0, stream>>>(dataset->X_temp->device(), dataset->X->device(), 
                                        dataset->ridx_sa_all->device());
            checkCudaErrors("GPUGLM_computeBlock::loadParams errors:  kernel_getX_temp launch failed");

            X_c     = dataset->X_temp;
        }
        else {
            X_c     = dataset->X;
        }

        //multiply coefficients and regressors (X_c * K -> LL)
        //X_c->printInfo(output_stream, "X_c");
        //params->K->printInfo(output_stream, "K");
        //dataset->LL->printInfo(output_stream, "dataset->LL");
        //msg->printMsgTxt(output_stream);
        checkCudaErrors(X_c->GEMM(dataset->LL, params->K, cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N), "GPUGLM_computeBlock::loadParams errors:  X*K -> LL");
    }
    else {
        results_set = false;
    }
    return isSparseRun;
}

/*Kernel for each observation
 *Returns the  observation-wise log like (LL - no normalizing constant), it's derivative portion (dLL),
 * and the SQUARE ROOT of the absolute value of the second derivative
 */
template <class FPTYPE>
__global__ void kernel_getObs_LL_GLM(GPUData_kernel<FPTYPE> LL, GPUData_kernel<FPTYPE> dLL, GPUData_kernel<FPTYPE> d2LL,
        const GPUData_kernel<FPTYPE> Y,
        const FPTYPE log_dt, const FPTYPE dt,
        const GPUData_kernel<unsigned int> id_a_trialM,
        const GPUData_kernel<FPTYPE> trial_weights,
        const GPUData_kernel<unsigned int> ridx_sa_all,
        const logLikeType logLikeSettings, GPUData_kernel<FPTYPE> logLikeParams,
        const bool compute_dK, const bool compute_d2K) {
    //current observation index
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < LL.x) {
        unsigned int X_row;
        if(ridx_sa_all.y == 0) {
            //if full run
            X_row = row;
        }
        else {
            //if sparse run
            X_row = ridx_sa_all[row];
        }
        FPTYPE tw_c = (trial_weights.y == 0) ? 1 : trial_weights[id_a_trialM[X_row]];
        FPTYPE Y_c = Y[X_row];

        FPTYPE   LL_c = 0;  
        FPTYPE  dLL_c = 0;   
        FPTYPE d2LL_c = 0;    
        if(tw_c != 0) { //if trial not censored
            FPTYPE log_rate = LL[row];

            if(logLikeSettings == ll_poissExp) {
                int Y_ci = floor(Y_c);
                if(Y_ci >= 0) { // negatives get censored by Poisson LL
                    log_rate += log_dt;
                    FPTYPE rate = safeExp(log_rate);
                     LL_c = (-rate + Y_ci * log_rate);
                    dLL_c = (-rate + Y_ci) ;
                    if(compute_d2K) {
                        d2LL_c = safeExp(static_cast<FPTYPE>(0.5)*log_rate);
                    }
                }
            }
            else if(logLikeSettings == ll_sqErr) {
                FPTYPE eY_c = log_rate - Y_c;
                 LL_c = -(eY_c*eY_c);
                dLL_c = -2*eY_c;
                if(compute_d2K) {
                    d2LL_c = sqrt(static_cast<FPTYPE>(2.0));
                }
            }

            else if(logLikeSettings == ll_truncatedPoissExp) {
                int Y_ci = floor(Y_c);
                if(Y_ci > 0) { 
                    log_rate += log_dt;
                    if(log_rate > -30) {
                        FPTYPE rate = safeExp(log_rate);
                        FPTYPE expNrate = safeExp(-rate);
                         LL_c = log(1.0 - expNrate);
                        FPTYPE exm1 = safeExpm1(rate);
                        dLL_c = rate/exm1;
                        if(compute_d2K) {
                            FPTYPE enxm1 = safeExpm1(-rate);
                            d2LL_c = sqrt(-rate*(1.0-rate - expNrate)/(exm1 + enxm1)); // THIS NEEDS TO BE CHECKED
                        }
                    }
                    else { // more numerically save approximation in an extreme case
                         LL_c = log_rate;
                        dLL_c = 1;
                        d2LL_c = 0;
                    }
                }
                else if(Y_ci == 0) {
                    FPTYPE rate = safeExp(log_rate + log_dt);
                     LL_c = -rate;
                    dLL_c = -rate;
                    if(compute_d2K) {
                        d2LL_c = safeExp(static_cast<FPTYPE>(0.5)*(log_rate + log_dt));
                    }
                }
                // negatives get censored by Poisson LL
            }
            else if(logLikeSettings == ll_poissSoftRec) {
                int Y_ci = floor(Y_c);
                if(Y_ci >= 0) { // negatives get censored by Poisson LL
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
                    LL_c  = (-rate*dt + Y_ci *(log(rate) + log_dt));
                    dLL_c = (-drate*dt + Y_ci * drate_rate);

                    if(compute_d2K) {
                        FPTYPE d2rate = drate*(1.0-drate);
                        d2LL_c = -d2rate*dt + Y_ci * ((d2rate*rate - drate*drate)/(rate*rate));
                        d2LL_c = (d2LL_c < 0) ? sqrt(-d2LL_c) : 0;
                    }
                }
            }
            else if(logLikeSettings == ll_poissExpRefractory) {
                // ll_poissExpRefractory uses the correction from Citi, L., Ba, D., Brown, E. N., & Barbieri, R. (2014). Likelihood methods for point processes with refractoriness. Neural computation, 26(2), 237-263.
                int Y_ci = floor(Y_c);
                if(Y_ci >= 0) { // negatives get censored by Poisson LL
                    log_rate += log_dt;
                    FPTYPE rate = safeExp(log_rate);
                     LL_c = (-(1.0-Y_ci/2.0)*rate + Y_ci * log_rate);
                    dLL_c = (-(1.0-Y_ci/2.0)*rate + Y_ci);
                    if(compute_d2K) {
                        d2LL_c = sqrt(1.0-Y_ci/2.0)*safeExp(static_cast<FPTYPE>(0.5)*log_rate);
                    }
                }
            }
        }
        LL[row]  =  LL_c*tw_c;
        if(compute_dK) {
            dLL[row] = dLL_c*tw_c;
        }
        if(compute_d2K) {
            d2LL[row] = d2LL_c*sqrt(tw_c);
        }
    }
}

/* Kernel for each trial
*  Sums up the trial log likelihoods (results->trialLL)
*/     
template <class FPTYPE>
__global__ void kernel_sum_trialLL_GLM(GPUData_kernel<FPTYPE> trialLL,
                                 const GPUData_kernel<unsigned int> trial_included, 
                                 const GPUData_kernel<FPTYPE> LL, 
                                 const GPUData_kernel<size_t> dim_N,
                                 const GPUData_kernel<unsigned int> ridx_t_all,
                                 const GPUData_kernel<unsigned int>  id_t_trial,
                                 const GPUData_kernel<FPTYPE> trial_weights,
                                 const GPUData_kernel<FPTYPE> normalizingConstants) {
     
    unsigned int tr = blockIdx.x * blockDim.x + threadIdx.x;
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
        FPTYPE tw_c = (trial_weights.y == 0) ? 1 : trial_weights[mm];
        if(tw_c != 0) {  
            unsigned int row = ridx_t_all[tr];  // this uses 'tr' so that it works for sparse runs

            //sum up LL
            FPTYPE ll_total = normalizingConstants[mm] * tw_c;
            for(int tt = 0; tt < dim_N[mm]; tt++) {
                ll_total  += LL[row + tt];
            }
            trialLL[id_t_trial[mm]] = ll_total;
        }
        //no need to compute sum; set results to 0
        else {
            trialLL[id_t_trial[mm]] = 0;
        }
    }
}

template <class FPTYPE>
void GPUGLM_computeBlock<FPTYPE>::computeLogLike(const GPUGLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }

    switchToDevice();
         //launch kernel to sum lambda, X_lin*B -> LL, dLL (launch over all observations)
         //launch kernel to sum lambda for each trial
    
    //X_lin*B + W + log_dt + sum(lambda) -> LL for each neuron

    dim3 block_size;
    block_size.x = 1024;
    dim3 grid_size;
    grid_size.x = dataset->LL->getSize(0) / block_size.x + ( (dataset->LL->getSize(0) % block_size.x == 0) ? 0 : 1);
    kernel_getObs_LL_GLM<<<grid_size, block_size, 0, stream>>>(dataset->LL->device(), dataset->dLL->device(), dataset->d2LL->device(),
                  dataset->Y->device(),
                  dataset->log_dt, dataset->dt,
                  dataset->id_a_trialM->device(), params->trial_weights->device(),
                  dataset->ridx_a_all_c->device(),
                  params->logLikeSettings, params->logLikeParams->device(), 
                  opts->compute_dK, opts->compute_d2K);
    checkCudaErrors("GPUGLM_computeBlock::computeLogLike errors:  kernel_getObs_LL_GLM launch failed");

    //sum up the LL for each trial (and dLL to setup for dW, dB)
    if(opts->compute_trialLL) {

        block_size.x = 1024;
        block_size.y = 1;

        grid_size.x = dataset->ridx_t_all_c->getSize(0) / block_size.x + ( (dataset->ridx_t_all_c->getSize(0) % block_size.x == 0) ? 0 : 1);
        grid_size.y = 1;

        kernel_sum_trialLL_GLM<<<grid_size, block_size, 0, stream>>>(results->trialLL->device(),
                                                                 params->trial_included->device(), 
                                                                 dataset->LL->device(), 
                                                                 dataset->dim_N->device(),
                                                                 dataset->ridx_t_all_c->device(),
                                                                 dataset->id_t_trial->device(),
                                                                 params->trial_weights->device(), 
                                                                 dataset->normalizingConstants_trial->device());
        checkCudaErrors("GPUGLM_computeBlock::computeLogLike errors:  kernel_sum_trialLL_GLM launch failed");
    }
}


template <class FPTYPE>
void GPUGLM_computeBlock<FPTYPE>::computeDerivatives(const GPUGLM_computeOptions<FPTYPE> * opts, const bool isSparseRun) {
    if(params->getNumberOfNonzeroWeights() == 0) { //nothing to compute
        return;
    }
    switchToDevice();
    GPUData<FPTYPE> * X_c;
    if(isSparseRun) {
        //if sparse run
        X_c = dataset->X_temp;
    }
    else {
        //if full run
        X_c = dataset->X;
    }

         //launch kernel to sum dLL -> dW, dB for each trial?
         //         or kernel to sum up dLL->dW and GEMV for dB?
    if(opts->compute_dK) {
        checkCudaErrors(X_c->GEMM(results->dK, dataset->dLL, cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N), "GPUGLM_computeBlock::computeDerivatives errors:  X'*dLL -> dK");
    }
    
    // Hessian
    if(opts->compute_d2K) {
        //setup the temp space
        cublasStatus_t ce = cublasDGMM(cublasHandle2, CUBLAS_SIDE_LEFT,  //CUBLAS_SIDE_RIGHT: A x diag(X), CUBLAS_SIDE_LEFT: diag(X) x A
                                      X_c->getSize(0), X_c->getSize(1),
                                      X_c->getData_gpu(), X_c->getLD_gpu(),
                                      dataset->d2LL->getData_gpu(), 1,
                                      dataset->lambda->getData_gpu(), dataset->lambda->getLD_gpu());
        checkCudaErrors(ce, "GPUGLM_computeBlock::computeDerivatives errors:  X.*d2LL -> lambda");
    
        FPTYPE alpha = -1; //The negative is important here because of this symmetric op!
        FPTYPE beta  =  0; 
        ce = dataset->lambda->GEMM(results->d2K, dataset->lambda, cublasHandle2, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta);
        checkCudaErrors(ce, "GPUGLM_computeBlock::computeDerivatives errors:  lambda'*lambda' -> d2K");
        // NOTE: cublasSYRK is usually way slower than GEMM even though it's specialized*/
    }
}
        

//explicitly create classes for single and double precision floating point for library
template class GPUGLM_computeBlock<float>;
template class GPUGLM_computeBlock<double>;

};//namespace