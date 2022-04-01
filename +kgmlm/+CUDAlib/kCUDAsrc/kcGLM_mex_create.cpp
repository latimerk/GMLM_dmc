/*
 * kcGLM_mex_create.cu
 * Mex function to put GLM data on GPUs and setup computation space.
 *  Takes 3 arguments:   GLMstructure struct (from kcGLM class)
 *                       trials struct (from kcGLM class)
 *                       boolean (is double) for if the object is double precision (or single)
 *
 *  Requires 1 output:   ptr (in long int form) to GLM object
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
#include "kcGMLM_mex_shared.hpp"
#include "kcGLM.hpp"
#include <cmath>
#include <string>

class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
    // Create an output stream
    std::ostringstream stream;
    
    template <class FPTYPE>
    kCUDA::GPUGLM<FPTYPE> * setupGPUGLM(const matlab::data::StructArray GLMstructure_mat, const matlab::data::StructArray trialBlocks_mat) { 
        
        //setup object for GPUGLM to send messages and errors to MATLAB
        std::shared_ptr<kCUDA::GPUGL_msg> msgObj = std::make_shared<GPUGL_msg_mex>(matlabPtr);//new GPUGK_msg_mex(matlabPtr);
        
        // start with the structure
        kCUDA::GPUGLM_structure_args<FPTYPE> * GLMstructure = new kCUDA::GPUGLM_structure_args<FPTYPE>;
        
        const matlab::data::TypedArray<const FPTYPE> binSize = GLMstructure_mat[0]["binSize"];
        GLMstructure->binSize = binSize[0];
        
        const matlab::data::TypedArray<const uint64_t> dim_K = GLMstructure_mat[0]["dim_K"];
        GLMstructure->dim_K  = dim_K[0];
        if(GLMstructure->dim_K < 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("No covariates given!") }));
        }
        
        const matlab::data::TypedArray<const int> logLikeSettings = GLMstructure_mat[0]["logLikeSettings"];
        int logLikeSettings_temp = logLikeSettings[0];
        switch(logLikeSettings_temp) {
            case kCUDA::ll_poissExp:
                GLMstructure->logLikeSettings = kCUDA::ll_poissExp;
                break;
            case kCUDA::ll_poissSoftRec:
                GLMstructure->logLikeSettings = kCUDA::ll_poissSoftRec;
                break;
            case kCUDA::ll_sqErr:
                GLMstructure->logLikeSettings = kCUDA::ll_sqErr;
                break;
            case kCUDA::ll_truncatedPoissExp:
                GLMstructure->logLikeSettings = kCUDA::ll_truncatedPoissExp;
                break;
            default:
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("Invalid log likelihood type") }));
        }
         
        const matlab::data::TypedArray<const FPTYPE> logLikeParams = GLMstructure_mat[0]["logLikeParams"];
        size_t np = logLikeParams.getNumberOfElements();
        GLMstructure->logLikeParams.resize(np);
        for(int ii = 0; ii < np; ii++) {
            GLMstructure->logLikeParams[ii] = logLikeParams[ii];
        }
        
        
        // sets up the trial blocks
        size_t numBlocks = trialBlocks_mat.getNumberOfElements();
        if(numBlocks == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("No trial blocks given: nothing to load to GPU!") }));
        }
        
        std::vector<kCUDA::GPUGLM_GPU_block_args<FPTYPE> *> gpuBlocks;
        gpuBlocks.resize(numBlocks);
        for(int bb = 0; bb < numBlocks; bb++) {
            gpuBlocks[bb] = new kCUDA::GPUGLM_GPU_block_args<FPTYPE>;
            
            //gets device and compute block info
            const matlab::data::TypedArray<const int32_t> dev_num = trialBlocks_mat[bb]["GPU"];
            gpuBlocks[bb]->dev_num = dev_num[0];
            const matlab::data::TypedArray<const int32_t> max_trials_for_sparse_run = trialBlocks_mat[bb]["max_trials_for_sparse_run"];
            gpuBlocks[bb]->max_trials_for_sparse_run = max_trials_for_sparse_run[0];
            
            //sets up trials
            const matlab::data::StructArray trials_mat = trialBlocks_mat[bb]["trials"];
            
            size_t dim_M_c = trials_mat.getNumberOfElements();
            if(dim_M_c == 0) {
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("No trials given in block!") }));
            }
            
            //for each trial
            gpuBlocks[bb]->trials.resize(dim_M_c);
            for(int mm = 0; mm < dim_M_c; mm++) {
                gpuBlocks[bb]->trials[mm] = new kCUDA::GPUGLM_trial_args<FPTYPE>;
                
                //trial id
                const matlab::data::TypedArray<const uint32_t> trial_idx = trials_mat[mm]["trial_idx"];
                gpuBlocks[bb]->trials[mm]->trial_idx = trial_idx[0];
                
                //gets spike counts 
                const matlab::data::TypedArray<const FPTYPE> Y = trials_mat[mm]["Y"];
                gpuBlocks[bb]->trials[mm]->Y = new GLData_matlab<FPTYPE>(Y);
                
                //linear term
                const matlab::data::TypedArray<const FPTYPE> X = trials_mat[mm]["X"];
                gpuBlocks[bb]->trials[mm]->X = new GLData_matlab<FPTYPE>(X);
                
            } // end trials
        } // end trial blocks
        
        // calls the glm constructor
        kCUDA::GPUGLM<FPTYPE> * glm = new kCUDA::GPUGLM<FPTYPE>(GLMstructure, gpuBlocks, msgObj);
        
        // cleans up the local objects
        delete GLMstructure;
        
        for(int bb = 0; bb < gpuBlocks.size(); bb++) {
            for(int mm = 0; mm < gpuBlocks[bb]->trials.size(); mm++) {
                delete gpuBlocks[bb]->trials[mm]->X;
                delete gpuBlocks[bb]->trials[mm]->Y;
                delete gpuBlocks[bb]->trials[mm];
            }
            delete gpuBlocks[bb];
        }
        
        return glm;
    }
    
public:
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // check args: arg 0 is jcGLM object, args 1 is the block start indices, args 2 is the gpu numbers
        // needs 1 output for MATLAB's weird copy on write behavior
        
        if(outputs.size() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("One outputs required to kcGLM_mex_create - otherwise memory leaks could ensue!") }));
        }
        
        if(inputs.size() != 3) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("3 inputs required to kcGLM_mex_create!") }));
        }
        
        if(inputs[0].getType() != matlab::data::ArrayType::STRUCT ) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 1 must be a struct array!") }));
        }
        if(inputs[1].getType() != matlab::data::ArrayType::STRUCT) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 2 must be a struct array!") }));
        }
        if(inputs[2].getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 3 must be a scalar!") }));
        }
        
        const matlab::data::StructArray  GLMstructure = inputs[0];
        const matlab::data::StructArray  trialBlocks   = inputs[1];
        const matlab::data::TypedArray<bool> isDouble  = inputs[2]; //floating point type
        
        //call SETUP func for correct data type with inputs
        uint64_t glmPtr = 10;
        if(isDouble[0]) {
            kCUDA::GPUGLM<double> * glm = setupGPUGLM<double>(GLMstructure, trialBlocks);
            //store the pointer as an int to return to matlab
            glmPtr = reinterpret_cast<uint64_t>(glm);
        }
        else {
            kCUDA::GPUGLM<float > * glm = setupGPUGLM<float >(GLMstructure, trialBlocks);
            //store the pointer as an int to return to matlab
            glmPtr = reinterpret_cast<uint64_t>(glm);
        }
        
        //return ptr to object
        outputs[0] = factory.createScalar(glmPtr);
    };
};
