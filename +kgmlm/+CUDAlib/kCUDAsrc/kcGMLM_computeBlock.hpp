/*
 * kcGMLM_computeBlock.hpp
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
#ifndef GMLM_GMLM_COMPUTEBLOCK_H
#define GMLM_GMLM_COMPUTEBLOCK_H

#include "kcGMLM.hpp"
#include "kcGMLM_dataStructures.hpp"

namespace kCUDA {

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* CLASS GPUGMLM_computeBlock <FPTYPE = double, float>
*        This is for data on a single GPU. Holds all the trials and compute space.
*/
template <class FPTYPE> 
class GPUGMLM_computeBlock : GPUGL_base {
    private:
        // parameter space on GPU (plus page-locked memory copy)
        GPUGMLM_parameters_GPU<FPTYPE> * params;
        
        // dataset
        GPUGMLM_dataset_GPU<FPTYPE> * dataset;
        
        // results
        GPUGMLM_results_GPU<FPTYPE> * results;
        
        // streams and handles
        cudaStream_t stream; // main stream for block
        std::vector<cudaStream_t> stream_Groups;
        cublasHandle_t cublasHandle; // main handle for block
        std::vector<cublasHandle_t> cublasHandle_Groups;
        std::vector<cusparseHandle_t> cusparseHandle_Groups;
        
        size_t dim_J;
        
        bool results_set = false;

        cudaEvent_t LL_event;
        
    public:
        //Primary constructor takes in the GMLM setup (as the full GPUGMLM class does), but only the specific block data
        GPUGMLM_computeBlock(const GPUGMLM_structure_args<FPTYPE> * GMLMstructure, const GPUGMLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_, const size_t dim_P_, std::shared_ptr<GPUGL_msg> msg_);
        
        //destructor
        ~GPUGMLM_computeBlock();
        
        //sync all streams on the block
        inline void syncStreams() {
            switchToDevice();
            checkCudaErrors( cudaStreamSynchronize(stream), "GPUGMLM_computeBlock errors: could not synchronize streams!");
            for(auto jj : stream_Groups) {
                checkCudaErrors( cudaStreamSynchronize(jj), "GPUGMLM_computeBlock errors: could not synchronize group streams!");
            }
        }
        
        bool loadParams(const GPUGMLM_params<FPTYPE> * params_host, const GPUGMLM_computeOptions<FPTYPE> * opts = NULL);
        
        void computeRateParts(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeLogLike(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeDerivatives(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);
        
        inline void gatherResults(const GPUGMLM_computeOptions<FPTYPE> * opts) {
            if(results_set) {
                results->gatherResults(params, opts, stream, stream_Groups);
            }
        }
        inline bool addResultsToHost(GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const bool reset = false) {
            if(results_set) {
                results->addToHost(params, results_dest, opts, dataset, reset);
            }
            
            return results_set;
        }
};


}; //namespace
#endif