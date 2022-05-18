/*
 * ksSharedPython.hpp
 * Structures for linking my C++/CUDA GMLM code to Python via pybind11.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2022 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#ifndef GMLM_PYTHON_SHARED_H
#define GMLM_PYTHON_SHARED_H
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "kcShared.hpp"
#include <iostream>
#include <sstream>



class GPUGL_msg_python : public kCUDA::GPUGL_msg {
    private:
    
    public:
        GPUGL_msg_python() {
        }
        
        void callErrMsgTxt(std::ostringstream & output_stream) {
            auto err = std::runtime_error(output_stream.str());
            output_stream.str("");
            throw err;
        }
        void printMsgTxt(  std::ostringstream & output_stream) {
            py::print(output_stream.str());
            // Clear stream buffer
            output_stream.str("");
        }
        ~GPUGL_msg_python() {
        };
};


template <class FPTYPE>
class GLData_numpy : public kCUDA::GLData<FPTYPE> {
    public:
        //constructor
        GLData_numpy(py::array_t<FPTYPE,  py::array::f_style | py::array::forcecast> python_data ) {
            py::buffer_info python_data_buf = python_data.request();
            if(python_data_buf.shape.size() == 0 || (python_data_buf.shape.size() > 1 && python_data_buf.shape[1] == 0)) {
                this->assign(NULL, 0, 0, 0);
            }
            else {

                //gets the input dimensions
                size_t x = python_data_buf.shape[0];
                size_t y = 1;
                size_t z = 1;
                size_t ld = 0;
                size_t inc = 0;
                if(python_data_buf.ndim > 1) {
                    y = python_data_buf.shape[1];
                    ld = python_data_buf.strides[1] / sizeof(FPTYPE);
                    if(ld == 1 && y == 1) {
                        ld = x; // confusing thing about strides in numpy with size (x,1) arrays
                    }
                }
                if(python_data_buf.ndim > 2) {
                    z = python_data_buf.shape[2];
                    inc = python_data_buf.strides[2] / sizeof(FPTYPE);
                    if(inc == 1 && y == 1) {
                        inc = 0; // confusing thing about strides in numpy with size (x,1,1) arrays
                    }
                }
                FPTYPE * ptr = static_cast<FPTYPE*>(python_data_buf.ptr);

                this->assign(ptr, x, y, z, ld, inc);

            /* std::ostringstream output_stream;
                output_stream << " GLData_numpy [" << python_data_buf.shape.size() << "] ";
                for(int ii = 0; ii < python_data_buf.shape.size(); ii++) {
                    output_stream << python_data_buf.shape[ii] << "\t";
                }
                output_stream << "\n\tstrides ";
                for(int ii = 0; ii < python_data_buf.strides.size(); ii++) {
                    output_stream << python_data_buf.strides[ii] << "\t";
                }
                output_stream << "\t" << x << "," << y << ", " << z << "; " << ld << ", " << inc << "\n";
                py::print(output_stream.str());
                output_stream.str("");*/
            }
        }

        GLData_numpy() {
            this->assign(NULL, 0, 0, 0);
        }
        //destructor
        ~GLData_numpy() {
            /*std::ostringstream output_stream;
            py::print("GLData_numpy destructor ");
            output_stream << this << "\n";
            py::print(output_stream.str());*/
        }
};

#endif