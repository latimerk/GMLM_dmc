/*
 * kcGLM.cu
 * Main class for holding a GLM (across multiple GPUs).
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
#include "kcGLM.hpp"
#include "kcGLM_computeBlock.hpp"
        
namespace kCUDA {
    
template <class FPTYPE>
GPUGLM<FPTYPE>::GPUGLM(const GPUGLM_structure_args <FPTYPE> * GLMstructure, const std::vector<GPUGLM_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_) {
    msg = msg_;
    // check to see if data is provided
    if(blocks.empty()) {
        output_stream << "GPUGLM errors: no data blocks given!";
        msg->callErrMsgTxt(output_stream);
    }      

    //get the max trial and neuron index
    max_trials = 1;
    for(auto bb : blocks) {
        for(auto mm : bb->trials) {
            max_trials = max(max_trials, mm->trial_idx + 1);
        }
    }
            
    //build each block
    dim_K_ = GLMstructure->dim_K;
    gpu_blocks.resize(blocks.size());
    for(unsigned int bb = 0; bb < blocks.size(); bb++) {
        GPUGLM_computeBlock<FPTYPE> * block = new GPUGLM_computeBlock<FPTYPE>(GLMstructure, blocks[bb], max_trials, msg);
        gpu_blocks[bb] = block;
    }
}

template <class FPTYPE>
GPUGLM<FPTYPE>::~GPUGLM() {   
    //output_stream << "MAIN DESTRUCTOR: CALLED!\n"; msg->printMsgTxt(output_stream);
    // delete each block
    for(auto bb : gpu_blocks) {
        delete bb;
    }
}
 
template <class FPTYPE>
void GPUGLM<FPTYPE>::computeLogLikelihood(const GPUGLM_params<FPTYPE> * params, const GPUGLM_computeOptions<FPTYPE> * opts, GPUGLM_results<FPTYPE> * results) {
    std::vector<bool> isSparse;
    //load params to each block
    for(auto bb: gpu_blocks) {
        isSparse.push_back(bb->loadParams(params, opts));
    }
            
    //call bits of LL computation
    for(unsigned int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->syncStreams();
        gpu_blocks[bb]->computeLogLike(opts, isSparse[bb]);
    }
    for(unsigned int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->syncStreams();
        gpu_blocks[bb]->computeDerivatives(opts,  isSparse[bb]);
    }
          
    //gather to host
    for(auto bb: gpu_blocks) {
        bb->gatherResults(opts);
    }

    //sync everything
    syncStreams();
            
    //put results in user-supplied struct
    bool reset_needed = true;
    for(auto bb: gpu_blocks) {
        bool results_added = bb->addResultsToHost(results, opts, reset_needed);
        if(reset_needed && results_added) {
            reset_needed = false;
        }
    }
}  

template <class FPTYPE>
void GPUGLM<FPTYPE>::syncStreams() {
    for(auto bb : gpu_blocks) {
        bb->syncStreams(); 
    }
}   

//explicitly create classes for single and double precision floating point
template class GPUGLM<float>;
template class GPUGLM<double>;

}; //end namespace