/*
 * kcGMLM.hpp
 * Main class for holding a GMLM (across multiple GPUs).
 * The classes which the user provides to communicate with the GMLM are provided here.
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
#ifndef GMLM_GMLM_CLASS_H
#define GMLM_GMLM_CLASS_H

#include <vector>
#include <sstream>
#include <memory> // for std::shared_ptr on linux (compiles fine on Windows without this)
#include "kcGMLM_common.hpp"

namespace kCUDA { 
    
int getValidGPU();
bool gpuAvailable();

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* class GPUGMLM <FPTYPE = double, float>
*    Holds a complete GMLM
*       The GMLM can be split across multiple GPUs as a set of GPUGMLM_block objs
*
*    Variables: Observations
*
*        Blocks [dim_B x 1] (GPUGMLM_block<FPTYPE>*)
*            Each GMLM block (each should be on a different GPU). 
*/  
template <class FPTYPE> class GPUGMLM_computeBlock_base;

template <class FPTYPE> 
class GPUGMLM {
    private: 
        std::ostringstream output_stream;
        std::shared_ptr<GPUGL_msg> msg;
        std::shared_ptr<GPUGMLM_computeOptions<FPTYPE>> opts;
        std::shared_ptr< GPUGMLM_params<FPTYPE>> params;
        
        std::vector<GPUGMLM_computeBlock_base<FPTYPE> *> gpu_blocks;
        
        void syncStreams();

        const bool isSimultaneousPopulation_;
        unsigned int max_trials;
        
    public:
        /* Primary constructor.
         * Takes in the set of args defined by GPUGMLM_regressor_args and loads everything to the GPU(s).
         *
         * The full GMLM structure is defined in GMLMstructure, and the device usage and trial data are in the vector of blocks
         */
        GPUGMLM(const GPUGMLM_structure_args <FPTYPE> * GMLMstructure, const std::vector<GPUGMLM_GPU_block_args<FPTYPE> *> blocks, std::shared_ptr<GPUGL_msg> msg_);
        
        /* Deconstructor
         * Clears all GPU and host memory.
         */
        ~GPUGMLM();
        
        /* METHOD computeLogLikelihood
         *  Runs the complete log likelihood computation for a set of parameters.
         *  Does not copy the answer.
         *
         *  inputs:
         *      params (GPUGMLM_params<FPTYPE> *): the complete set of params for the model
         *      opts   (GPUGMLM_computeOptions *): the options for what to compute (i.e., which derivatives)
         *
         *  outputs:
         *      results (GPUGMLM_results<FPTYPE>*)
         *          Any values in the results obj not selected by opts may be 0's or NaN's.
         */
        void computeLogLikelihood(std::shared_ptr< GPUGMLM_params<FPTYPE>> params_, std::shared_ptr<GPUGMLM_computeOptions<FPTYPE>> opts_, GPUGMLM_results<FPTYPE>* results);
                /* GPUGMLM calls each GPU
                 *     GPUGMLM_GPUportion holds a set of trials, and set of compute blocks 
                 *         The GPUGMLM_computeBlocks iterate through the whole set of GPUGMLM_trials
                 *         Then the results from the GPUGMLM_computeBlocks are summed into the  GPUGMLM_GPUportion's results
                 *         
                 *         GPUGMLM_GPUportion's are summed on host and returned
                 *
                 */
        void computeLogLikelihood_async(std::shared_ptr< GPUGMLM_params<FPTYPE>> params_, std::shared_ptr<GPUGMLM_computeOptions<FPTYPE>> opts_);
        void computeLogLikelihood_gather( GPUGMLM_results<FPTYPE> * results, const bool reset_needed_0 = true);

        bool isSimultaneousPopulation() {
            return isSimultaneousPopulation_;
        }
        unsigned int numTrials() {
            return max_trials;
        }
        
};


}; //namespace
#endif