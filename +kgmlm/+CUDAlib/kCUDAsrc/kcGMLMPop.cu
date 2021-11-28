/*
 * kcGMLMPop.cu
 * Main class for holding a GMLMPop (across multiple GPUs).
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
#include "kcGMLMPop.hpp"
#include "kcGMLMPop_computeBlock.hpp"
        
namespace kCUDA {
    
template <class FPTYPE>
GPUGMLMPop<FPTYPE>::GPUGMLMPop(const GPUGMLMPop_structure_args <FPTYPE> * GMLMPopstructure, const std::vector<GPUGMLMPop_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_) {
    msg = msg_;
    // check to see if data is provided
    if(blocks.empty()) {
        output_stream << "GPUGMLMPop errors: no data blocks given!";
        msg->callErrMsgTxt(output_stream);
    }      

    // check for dimension consistency in the args
    if(!GMLMPopstructure->validateTrialStructure(blocks)) {
        output_stream << "GPUGMLMPop errors: inconsistent GMLMPop setup!";
        msg->callErrMsgTxt(output_stream);
    }   

    //get the max trial and neuron index
    unsigned int max_trials = 1;
    for(auto bb : blocks) {
        for(auto mm : bb->trials) {
            max_trials = max(max_trials, mm->trial_idx + 1);
        }
    }
            
    //build each block
    gpu_blocks.resize(blocks.size());
    for(int bb = 0; bb < blocks.size(); bb++) {
        GPUGMLMPop_computeBlock<FPTYPE> * block = new GPUGMLMPop_computeBlock<FPTYPE>(GMLMPopstructure, blocks[bb], max_trials,  msg);
        gpu_blocks[bb] = block;
    }
}

template <class FPTYPE>
GPUGMLMPop<FPTYPE>::~GPUGMLMPop() {   
    //output_stream << "MAIN DESTRUCTOR: CALLED!\n"; msg->printMsgTxt(output_stream);
    // delete each block
    for(auto bb : gpu_blocks) {
        delete bb;
    }
}
 
 template <class FPTYPE>
void GPUGMLMPop<FPTYPE>::computeLogLikelihood_async(std::shared_ptr<GPUGMLMPop_params<FPTYPE>> params_, std::shared_ptr<GPUGMLMPop_computeOptions<FPTYPE>> opts_) {
    opts = opts_;
    params = params_;

    std::vector<bool> isSparse;
    //load params to each block
    for(auto bb: gpu_blocks) {
        isSparse.push_back(bb->loadParams(params.get(), opts.get()));
    }

    //call bits of LL computation
    for(int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeRateParts(opts.get(), isSparse[bb]);
    }
    for(int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeLogLike(opts.get(), isSparse[bb]);
    }
    for(int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeDerivatives(opts.get(),  isSparse[bb]);
    }
          
    //gather to host
    for(auto bb: gpu_blocks) {
        bb->gatherResults(opts.get());
    }
}
template <class FPTYPE>
void GPUGMLMPop<FPTYPE>::computeLogLikelihood_gather( GPUGMLMPop_results<FPTYPE> * results, const bool reset_needed_0) {
    //sync everything
    syncStreams();
            
    //put results in user-supplied struct
    bool reset_needed = reset_needed_0;
    for(auto bb: gpu_blocks) {
        bool results_added = bb->addResultsToHost(results, opts.get(), reset_needed);
        if(reset_needed && results_added) {
            reset_needed = false;
        }
    }
}
template <class FPTYPE>
void GPUGMLMPop<FPTYPE>::computeLogLikelihood(std::shared_ptr<GPUGMLMPop_params<FPTYPE>> params_, std::shared_ptr<GPUGMLMPop_computeOptions<FPTYPE>> opts_, GPUGMLMPop_results<FPTYPE> * results) {
    computeLogLikelihood_async(params_, opts_);
    computeLogLikelihood_gather(results);
}  

template <class FPTYPE>
void GPUGMLMPop<FPTYPE>::syncStreams() {
    for(auto bb : gpu_blocks) {
        bb->syncStreams(); 
    }
}   

//explicitly create classes for single and double precision floating point
template class GPUGMLMPop<float>;
template class GPUGMLMPop<double>;

}; //end namespace