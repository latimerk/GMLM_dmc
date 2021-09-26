/*
 * kcResetDevices.cu
 * Calls cuda reset GPU (eg., if memory wasn't cleared properly or something).
 *  Arguments are the GPU devies numbers to reset.
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
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.

class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // check args: arg 0 is GMLM object
        if(inputs.size() < 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("One input required!") }));
        }
        int max_dev;
        auto ce = cudaGetDeviceCount(&(max_dev));
        if(ce != cudaSuccess) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("CUDA errors: could not get device count!") }));
        }  
        
        for (const matlab::data::TypedArray<double>& elem : inputs) {
            if(elem.getNumberOfElements() > 128) {
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("Too many device numbers given!") }));
            }
            if(elem.getType() == matlab::data::ArrayType::DOUBLE && elem.getType() == matlab::data::ArrayType::DOUBLE && elem.getType() == matlab::data::ArrayType::INT32) {
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("Input must be a double or int32 array!") }));
            }
            for(int ii = 0; ii < elem.getNumberOfElements(); ii++) {
                int dev = static_cast<int>(elem[ii]);
                
                //resets all the requested devices - helpful in debuging
                //check gpu device number
                if(dev >= max_dev || dev < 0) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("CUDA errors: device number invalid!") }));
                }

                //set gpu
                ce = cudaSetDevice(dev);
                if(ce != cudaSuccess) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("CUDA errors: could not set device") }));
                } 

                //resets device!
                ce = cudaDeviceReset();
                if(ce != cudaSuccess) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("CUDA errors: could not reset device!") }));
                }
            }
        }
    }
};