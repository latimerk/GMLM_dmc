/*
 * kcShared.hpp
 * Holds classes for communicating to the GLM/GMLM classes.
 * GPUGL_msg defines the necessary methods for the GLM/GMLM to send messsages
 * back to the caller (MATLAB, etc.)
 * The GLData class stores a pointer to data + dimensions (row-major format).
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
#ifndef GMLM_GLSHARED_H
#define GMLM_GLSHARED_H

#include <vector>
#include <sstream>
#include <memory> 

namespace kCUDA { 
    
enum logLikeType {ll_poissExp = 0, ll_sqErr = 1, ll_truncatedPoissExp = 2, ll_poissSoftRec = 3, ll_poissExpRefractory = 4};

// ll_poissExpRefractory uses the correction from Citi, L., Ba, D., Brown, E. N., & Barbieri, R. (2014). Likelihood methods for point processes with refractoriness. Neural computation, 26(2), 237-263.
    
//virtual class for comminicating error messages: to be instantiated by an interface to the GLM routines (i.e., MATLAB , Python, Julia, R, ...)
class GPUGL_msg {
public:
    virtual void callErrMsgTxt(std::ostringstream & output_stream) {}; //throw error with message: abstract function is here for eventual compatability with both Matlab & Python (matlab C calls mexErrMsgTxt)
    virtual void printMsgTxt(  std::ostringstream & output_stream) {}; //print message: abstract function is here for eventual compatability with both Matlab & Python (matlab C calls mexPrintf)
    virtual ~GPUGL_msg() {};
};

template <class FPTYPE> 
class GLData {
    private:
        FPTYPE * data = NULL; //on host
        
        //dimensions
        size_t x; //rows
        size_t y; //cols
        size_t z; //depth
        
        //increment over z dim
        size_t inc;
        
        //increment over y dim
        size_t ld;
    protected:
        void assign(FPTYPE * data_, const size_t x_, const size_t y_, const size_t z_, const size_t ld_, const size_t inc_) {
            data = data_;
            x = x_;
            y = y_;
            z = z_;
            ld = ld_;
            inc = inc_;
        }
        void assign(FPTYPE * data_, const size_t x_, const size_t y_= 1, const size_t z_= 1) {
            assign(data_, x_, y_, z_, x_ , (z_ > 1) ? x_ * y_ : 0);
        }
    public:
        // constructor sets vars
        GLData() {
            data = NULL;
            x = 0;
            y = 0;
            z = 0;
            inc = 0;
            ld = 0;
        }
        GLData(FPTYPE * data_, const size_t x_, const size_t y_ = 1, const size_t z_ = 1) : data(data_), x(x_), y(y_), z(z_) {
            ld =  x ;
            inc = (z > 1) ? x*y : 0;
        }
        GLData(FPTYPE * data_, const size_t x_, const size_t y_, const size_t z_, const size_t ld_, const size_t inc_) : data(data_), x(x_), y(y_), z(z_), ld(ld_), inc(inc_) {
        }

        
        // destructor does not destroy memory
        virtual ~GLData() {
        }
        
        //pointers to the data
        inline const FPTYPE * getData() const {
            return data;
        }
        
        //lead dimensions of matrix (for pitched memory), across the height dimension (or columns because of BLAS/MATLAB style memory)
        inline size_t getLD() const {
            return ld;
        }
        
        //incremement (in number of elements, not bytes) across the depth dimension
        inline size_t getInc() const {
            return inc;
        }
        
        //total size info
        inline bool empty() const {
            return getNumElements() == 0;
        }
        inline size_t getNumElements() const {
            return getSize(0) * getSize(1) * getSize(2);
        }
        inline size_t size() const {
            return getNumElements();
        }
        
        //size of each dimension
        inline size_t getSize(const int dim) const {
            switch(dim) {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                default:
                    return 0;
            }  
        }
        //assigns a single value to all elements 
        inline void assign(const FPTYPE val) {
            for(int xx = 0; xx < getSize(0); xx++) {
                for(int yy = 0; yy < getSize(1); yy++) {
                    for(int zz = 0; zz < getSize(2); zz++) {
                        data[xx + yy*ld + zz*inc] = val;
                    }
                }
            }
        }
        
        //access values
        const FPTYPE & operator[](const size_t index) const {
            return data[index];
        }
        FPTYPE & operator[](const size_t index)  {
            return data[index];
        }
        const FPTYPE & operator()(const size_t xx, const size_t yy, const size_t zz = 0) const {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        FPTYPE & operator()(const size_t xx, const size_t yy, const size_t zz = 0) {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        
        const FPTYPE & get(const size_t xx, const size_t yy, const size_t zz = 0) {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        void set(const FPTYPE & val, const size_t xx, const size_t yy, const size_t zz = 0) {
            size_t index = xx + yy * ld + zz * inc;
            data[index] = val;
        }
};
}
#endif