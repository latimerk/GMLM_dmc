/*
 * kcGLM_computeBlock.hpp
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
#ifndef GLM_GLM_COMPUTEBLOCK_H
#define GLM_GLM_COMPUTEBLOCK_H

#include "kcGLM.hpp"
#include "kcGLM_dataStructures.hpp"

namespace kCUDA {

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* CLASS GPUGLM_computeBlock <FPTYPE = double, float>
*        This is for data on a single GPU. Holds all the trials and compute space.
*/
template <class FPTYPE> 
class GPUGLM_computeBlock : GPUGL_base {
    private:
        // parameter space on GPU (plus page-locked memory copy)
        GPUGLM_parameters_GPU<FPTYPE> * params;
        
        // dataset
        GPUGLM_dataset_GPU<FPTYPE> * dataset;
        
        // results
        GPUGLM_results_GPU<FPTYPE> * results;
        
        // streams and handles
        cudaStream_t stream; // main stream for block
        cublasHandle_t cublasHandle; // main handle for block
        cudaStream_t stream2; // secondary stream for block
        cublasHandle_t cublasHandle2; // secondary handle for block
        
        bool results_set = false;
        
    public:
        //Primary constructor takes in the GLM setup (as the full GPUGLM class does), but only the specific block data
        GPUGLM_computeBlock(const GPUGLM_structure_args<FPTYPE> * GLMstructure, const GPUGLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_);
        
        //destructor
        ~GPUGLM_computeBlock();
        
        //sync all streams on the block
        inline void syncStreams() {
            switchToDevice();
            checkCudaErrors( cudaStreamSynchronize(stream ), "GPUGLM_computeBlock errors: could not synchronize streams!");
            checkCudaErrors( cudaStreamSynchronize(stream2), "GPUGLM_computeBlock errors: could not synchronize streams!");
        }
        
        bool loadParams(const GPUGLM_params<FPTYPE> * params_host, const GPUGLM_computeOptions<FPTYPE> * opts = NULL);
        
        void computeLogLike(const GPUGLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeDerivatives(const GPUGLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);
        
        inline void gatherResults(const GPUGLM_computeOptions<FPTYPE> * opts) {
            if(results_set) {
                results->gatherResults(params, opts, stream, stream2);
            }
        }
        inline bool addResultsToHost(GPUGLM_results<FPTYPE>* results_dest, const GPUGLM_computeOptions<FPTYPE> * opts, const bool reset = false) {
            if(results_set) {
                results->addToHost(params, results_dest, opts, dataset, reset);
            }
            return results_set;
        }
};


}; //namespace
#endif