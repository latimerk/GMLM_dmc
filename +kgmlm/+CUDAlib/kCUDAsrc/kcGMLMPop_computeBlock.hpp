/*
 * kcGMLMPop_computeBlock.hpp
 * Computations for a GMLMPop+derivatives (on one GPU).
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   Population GMLM and indidivual-cell-recordings GMLM are in separate CUDA files.
 *   I decided to do this in order to make the different optimization requirements a little more clean.
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
#ifndef GMLMPop_GMLMPop_COMPUTEBLOCK_H
#define GMLMPop_GMLMPop_COMPUTEBLOCK_H

#include "kcGMLMPop.hpp"
#include "kcGMLMPop_dataStructures.hpp"

namespace kCUDA {

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* CLASS GPUGMLMPop_computeBlock <FPTYPE = double, float>
*        This is for data on a single GPU. Holds all the trials and compute space.
*/
template <class FPTYPE> 
class GPUGMLMPop_computeBlock : GPUGL_base {
    private:
        // parameter space on GPU (plus page-locked memory copy)
        GPUGMLMPop_parameters_GPU<FPTYPE> * params;
        
        // dataset
        GPUGMLMPop_dataset_GPU<FPTYPE> * dataset;
        
        // results
        GPUGMLMPop_results_GPU<FPTYPE> * results;
        
        // streams and handles
        cudaStream_t stream; // main stream for block
        std::vector<cudaStream_t> stream_Groups;
        cublasHandle_t cublasHandle; // main handle for block
        std::vector<cublasHandle_t> cublasHandle_Groups;
        std::vector<cusparseHandle_t> cusparseHandle_Groups;
        
        size_t dim_J;
        
        bool results_set = false;
        
    public:
        //Primary constructor takes in the GMLMPop setup (as the full GPUGMLMPop class does), but only the specific block data
        GPUGMLMPop_computeBlock(const GPUGMLMPop_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLMPop_GPU_block_args <FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_);
        
        //destructor
        ~GPUGMLMPop_computeBlock();
        
        //sync all streams on the block
        inline void syncStreams() {
            switchToDevice();
            checkCudaErrors( cudaStreamSynchronize(stream), "GPUGMLMPop_computeBlock errors: could not synchronize streams!");
            for(auto jj : stream_Groups) {
                checkCudaErrors( cudaStreamSynchronize(jj), "GPUGMLMPop_computeBlock errors: could not synchronize group streams!");
            }
        }
        
        bool loadParams(const GPUGMLMPop_params<FPTYPE> * params_host, const GPUGMLMPop_computeOptions<FPTYPE> * opts = NULL);
        
        void computeRateParts(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeLogLike(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeDerivatives(const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool isSparseRun);
        
        inline void gatherResults(const GPUGMLMPop_computeOptions<FPTYPE> * opts) {
            if(results_set) {
                results->gatherResults(params, opts, stream, stream_Groups);
            }
        }
        inline bool addResultsToHost(GPUGMLMPop_results<FPTYPE>* results_dest, const GPUGMLMPop_computeOptions<FPTYPE> * opts, const bool reset = false) {
            if(results_set) {
                results->addToHost(params, results_dest, opts, dataset, reset);
            }
            
            return results_set;
        }
};


}; //namespace
#endif