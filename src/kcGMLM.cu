/*
 * kcGMLM.cu
 * Main class for holding a GMLM (across multiple GPUs).
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
#include "kcGMLM.hpp"
#include "kcGMLM_dataStructures.hpp"
        
namespace kCUDA {
    
int getValidGPU() {

    #ifdef USE_GPU
        int nd = 0;
        cudaError_t ce = cudaGetDeviceCount(&nd);
        if(ce == cudaSuccess) {
            for(int device = 0; device < nd; device++) {
                cudaDeviceProp deviceProp;
                ce = cudaGetDeviceProperties(&deviceProp, device);
                if(ce == cudaSuccess) {
                    if(610 <= deviceProp.major*100 + deviceProp.minor*10) {
                        return device;
                    }
                }
            }
        }
        return -1;
    #else
        return -2;
    #endif
}
bool gpuAvailable() {
    return getValidGPU() >= 0;
}

template <class FPTYPE>
GPUGMLM<FPTYPE>::GPUGMLM(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const std::vector<GPUGMLM_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_) : isSimultaneousPopulation_(GMLMstructure->isSimultaneousPopulation) {
    msg = msg_;
    // check to see if data is provided
    if(blocks.empty()) {
        output_stream << "GPUGMLM errors: no data blocks given!";
        msg->callErrMsgTxt(output_stream);
    }      

    // check for dimension consistency in the args
    if(!GMLMstructure->validateTrialStructure(blocks)) {
        output_stream << "GPUGMLM errors: inconsistent GMLM setup!";
        msg->callErrMsgTxt(output_stream);
    }   

    //get the max trial and neuron index
    max_trials = 1;
    for(auto bb : blocks) {
        for(auto mm : bb->trials) {
            max_trials = max(max_trials, mm->trial_idx + 1);
            if(!isSimultaneousPopulation() && mm->neuron  >= GMLMstructure->dim_P) {
                output_stream << "GPUGMLM errors: neuron number cannot exceed expected number of neurons (dim_P)!";
                msg->callErrMsgTxt(output_stream);
            }
        }
    }
            
    //build each block
    gpu_blocks.resize(blocks.size());
    for(unsigned int bb = 0; bb < blocks.size(); bb++) {
        if(isSimultaneousPopulation()) {
            gpu_blocks[bb] = new GPUGMLMPop_computeBlock<FPTYPE>(GMLMstructure, blocks[bb], max_trials, msg);
        }
        else {
            gpu_blocks[bb] = new GPUGMLM_computeBlock<FPTYPE>(GMLMstructure, blocks[bb], max_trials, msg);
        }
    }
}

template <class FPTYPE>
GPUGMLM<FPTYPE>::~GPUGMLM() {   
    //output_stream << "MAIN DESTRUCTOR: CALLED!\n"; msg->printMsgTxt(output_stream);
    // delete each block
    for(auto bb : gpu_blocks) {
        delete bb;
    }
}

template <class FPTYPE>
void GPUGMLM<FPTYPE>::computeLogLikelihood_async(std::shared_ptr<GPUGMLM_params<FPTYPE>> params_, std::shared_ptr<GPUGMLM_computeOptions<FPTYPE>> opts_) {
    opts = opts_;
    params = params_;

    std::vector<bool> isSparse;
    //load params to each block
    for(auto bb: gpu_blocks) {
        isSparse.push_back(bb->loadParams(params.get(), opts.get()));
    }

    //call bits of LL computation
    for(unsigned int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeRateParts(opts.get(), isSparse[bb]);
    }
    for(unsigned int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeLogLike(opts.get(), isSparse[bb]);
    }
    for(unsigned int bb = 0; bb < gpu_blocks.size(); bb++) {
        gpu_blocks[bb]->computeDerivatives(opts.get(),  isSparse[bb]);
    }
          
    //gather to host
    for(auto bb: gpu_blocks) {
        bb->gatherResults(opts.get());
    }
}
template <class FPTYPE>
void GPUGMLM<FPTYPE>::computeLogLikelihood_gather( GPUGMLM_results<FPTYPE> * results, const bool reset_needed_0) {
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
void GPUGMLM<FPTYPE>::computeLogLikelihood(std::shared_ptr< GPUGMLM_params<FPTYPE>> params_, std::shared_ptr<GPUGMLM_computeOptions<FPTYPE>> opts_, GPUGMLM_results<FPTYPE> * results) {
    computeLogLikelihood_async(params_, opts_);
    computeLogLikelihood_gather(results);
}  

template <class FPTYPE>
void GPUGMLM<FPTYPE>::syncStreams() {
    for(auto bb : gpu_blocks) {
        bb->syncStreams(); 
    }
}   

//explicitly create classes for single and double precision floating point
template class GPUGMLM<float>;
template class GPUGMLM<double>;

}; //end namespace