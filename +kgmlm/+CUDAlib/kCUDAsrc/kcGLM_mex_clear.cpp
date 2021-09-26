/*
 * kcGLM_mex_clear.cu
 * Mex function to clear a GLM object from the GPU.
 *  Takes two arguments: ptr (in long int form) to GLM object
 *                       boolean (is double) for if the object is double precision (or single)
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

class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // check args: arg 0 is GLM object
        if(inputs.size() != 2) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required to kcGLM_mex_clear!") }));
        }
        
        //checks that arg 0 is a pointer value to the GLM C++ object
        const matlab::data::TypedArray<uint64_t> gpu_ptr_a = inputs[0];
        if(gpu_ptr_a.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("pointer argument must only contain 1 element!") }));
        }
        
        //get datatype field from GLM object
        const matlab::data::TypedArray<bool> isDouble = inputs[1];
        if(isDouble.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("cannot determine gpu data type!") }));
        }
        
        
        if(gpu_ptr_a.getNumberOfElements() < 1 || gpu_ptr_a[0] == 0) {
            //no glm object to clear
            matlabPtr->feval(u"warning", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized: not clearing data!") }));
        }
        else {
            
            uint64_t gpu_ptr = gpu_ptr_a[0];
            //converts pointer to kCUDA::GPUGLM and deletes
            if(isDouble[0]) {
                kCUDA::GPUGLM<double> * glm = reinterpret_cast<kCUDA::GPUGLM<double>*>(gpu_ptr);
                delete glm;
            }
            else {
                kCUDA::GPUGLM<float > * glm = reinterpret_cast<kCUDA::GPUGLM<float >*>(gpu_ptr);
                delete glm;
            }
        }
    }
};