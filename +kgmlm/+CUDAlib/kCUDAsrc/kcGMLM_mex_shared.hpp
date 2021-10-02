/*
 * kcGMLM_mex_shared.cu
 * Some shared classes for the mex files in this GMLM package.
 *  GPUGL_msg_mex handles message class to print error txt from GLM/GMLMs in MATLAB.
 *  GLData_matlab turns a matlab matrix into a GLData object to transfor to the GLM/GMLM c++ class.
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
#ifndef GMLM_MEX_SHARED_H
#define GMLM_MEX_SHARED_H

#include "kcShared.hpp"
#include <iostream>
#include <sstream>
#include "mex.hpp"
#include "mexAdapter.hpp"


class GPUGL_msg_mex : public kCUDA::GPUGL_msg {
private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    matlab::data::ArrayFactory factory;
    
public:
    GPUGL_msg_mex(std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr) {
        this->matlabPtr = matlabPtr;
    }
    
    void callErrMsgTxt(std::ostringstream & output_stream) {
        std::vector<matlab::data::Array> msgArray({ factory.createScalar(output_stream.str()) });
        output_stream.str("");
        (matlabPtr)->feval(u"error", 0, msgArray);
        
    }
    void printMsgTxt(  std::ostringstream & output_stream) {
        (matlabPtr)->feval(u"fprintf", 0, std::vector<matlab::data::Array>({ factory.createScalar(output_stream.str()) }));
        // Clear stream buffer
        output_stream.str("");
    }
    ~GPUGL_msg_mex() {
    };
};

template <class FPTYPE>
class GLData_matlab : public kCUDA::GLData<FPTYPE> {
public:
    //constructor
    GLData_matlab(const matlab::data::TypedArray<const FPTYPE> matlab_data ) {
        //gets the input dimensions
        FPTYPE * data = const_cast<FPTYPE*>((matlab_data.begin()).operator->());
        
        const matlab::data::ArrayDimensions dims = matlab_data.getDimensions();
        size_t x = dims[0];
        size_t y = 1;
        size_t z = 1;
        if(dims.size() > 1) {
            y = dims[1];
        }
        if(dims.size() > 2) {
            z = dims[2];
        }
        
        this->assign(data, x, y, z);
    }
    GLData_matlab(const matlab::data::TypedArray<FPTYPE> matlab_data ) {
        //gets the input dimensions
        FPTYPE * data = const_cast<FPTYPE*>((matlab_data.begin()).operator->());
        
        const matlab::data::ArrayDimensions dims = matlab_data.getDimensions();
        size_t x = dims[0];
        size_t y = 1;
        size_t z = 1;
        if(dims.size() > 1) {
            y = dims[1];
        }
        if(dims.size() > 2) {
            z = dims[2];
        }
        
        this->assign(data, x, y, z);
    }
    GLData_matlab(const matlab::data::TypedArray<const FPTYPE> * matlab_data_ptr) {
        //gets the input dimensions
        FPTYPE * data = NULL;
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        if(matlab_data_ptr != NULL) {
            const matlab::data::TypedArray<const FPTYPE> matlab_data = *matlab_data_ptr;
            data = const_cast<FPTYPE*>((matlab_data.begin()).operator->());

            const matlab::data::ArrayDimensions dims = matlab_data.getDimensions();
            x = dims[0];
            y = 1;
            z = 1;
            if(dims.size() > 1) {
                y = dims[1];
            }
            if(dims.size() > 2) {
                z = dims[2];
            }
        }

        this->assign(data, x, y, z);
    }
    GLData_matlab(const matlab::data::TypedArray<FPTYPE> * matlab_data_ptr) {
        //gets the input dimensions
        FPTYPE * data = NULL;
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        if(matlab_data_ptr != NULL) {
            const matlab::data::TypedArray<FPTYPE> matlab_data = *matlab_data_ptr;
            data = const_cast<FPTYPE*>((matlab_data.begin()).operator->());

            const matlab::data::ArrayDimensions dims = matlab_data.getDimensions();
            x = dims[0];
            y = 1;
            z = 1;
            if(dims.size() > 1) {
                y = dims[1];
            }
            if(dims.size() > 2) {
                z = dims[2];
            }
        }

        this->assign(data, x, y, z);
    }
    GLData_matlab() {
        this->assign(NULL, 0, 0, 0);
    }
    //destructor
    ~GLData_matlab() {
    }
};


#endif
