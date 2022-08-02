/*
 * kcBase.hpp
 * Defines a base class for the GMLM and GLM classes. Holds common error handling
 * methods.
 * Defines classes for storing matrices (including stacks of matrices)
 * keeps track of all the GPU stuff.
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
#ifndef GMLM_GLBASE_H
#define GMLM_GLBASE_H

#include "kcDefs.hpp"
#include "kcShared.hpp"

namespace kCUDA { 

    
    
const unsigned int MAX_DIM_D = 6;
//for templating the sparse matrix ops
template <class FPTYPE> cudaDataType_t getCudaType();
    
enum GPUData_HOST_ALLOCATION {GPUData_HOST_NONE = 0, GPUData_HOST_PAGELOCKED = 1, GPUData_HOST_STANDARD = 2}; 

    //classes for storing matrices (including stacks of matrices)
// base classes for CUDA arrays
//  holds data in up to a 3D array, while managing the pitch sizes and such
template <typename FPTYPE> 
struct GPUData_kernel { // holds data on GPU for more compact/clean kernel calls
    unsigned int x, y, z, ld, inc;
    FPTYPE * data; 
    
    __device__ __host__ GPUData_kernel() : x(0), y(0), z(0), ld(0), inc(0), data(NULL) {
    }

    __device__ __host__
    const FPTYPE & operator()(const size_t xx, const size_t yy = 0, const size_t zz = 0) const {
        size_t index = xx + yy * ld + zz * inc;
        return data[index];
    }
    __device__ __host__
    FPTYPE & operator()(const size_t xx, const size_t yy = 0, const size_t zz = 0) {
        size_t index = xx + yy * ld + zz * inc;
        return data[index];
    }
    __device__ __host__
    const FPTYPE & operator[](const size_t index) const {
        return data[index];
    }
    __device__ __host__
    FPTYPE & operator[](const size_t index) {
        return data[index];
    }
    __device__ __host__
    const FPTYPE & get(const size_t xx, const size_t yy = 0, const size_t zz = 0) const {
        size_t index = xx + yy * ld + zz * inc;
        return data[index];
    }
    __device__ __host__
    void set(const FPTYPE & val, const size_t xx, const size_t yy = 0, const size_t zz = 0) {
        size_t index = xx + yy * ld + zz * inc;
        data[index] = val;
    }
};
    

template <typename FPTYPE, unsigned int N_max> 
struct GPUData_array_kernel {
    GPUData_kernel<FPTYPE> kernel[N_max];
    unsigned int N = 0;
    
    __device__ __host__
    const GPUData_kernel<FPTYPE> & operator[](const size_t index) const {
        return kernel[index];
    }
    __device__ __host__
    GPUData_kernel<FPTYPE> & operator[](const size_t index) {
        return kernel[index];
    }
};

template <class FPTYPE> 
class GPUData {
    private:
        cudaPitchedPtr data_gpu;
        
        GPUData_kernel<FPTYPE> data_kernel;
        GPUData_kernel<FPTYPE> data_host;
        
        dim3 data_size; //for GPU
        
        int devNum;
        bool allocated_host  = false;
        bool allocated_gpu   = false;
        bool page_locked = false;
        bool is_stacked_gpu = false;
        
    public:
        // constructor sets vars to unallocated settings
        GPUData();
        
            // constructor with GPU allocation, returns a CUDA status in first argument
            //  include_host 0 == none, 1 = page locked, 2 = host
        GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t x, size_t y = 1, size_t z = 1, bool stacked_depth = false);
        GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, const dim3 size, bool stacked_depth = false);

        // allocate memory on GPU (and page-locked host if requested), returns cuda status
        cudaError_t allocate_gpu(GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t x, size_t y = 1, size_t z = 1, bool stacked_depth = false);
        
        // allocate memory on host (is page-locked host if requested, otherwise normal memory), returns cuda status (will only be set if page-locked memory requested)
        cudaError_t allocate_host(bool page_locked_memory, size_t x, size_t y = 1, size_t z = 1);
        
        // deletes all memory currently allocated, returns cuda status
        cudaError_t deallocate();
        
        // destructor destroys memory - does not check for CUDA errors! Recommend calling deallocate manually
        ~GPUData();
        
        //gets the struct to send matrix to a cuda kernel
        inline const GPUData_kernel<FPTYPE> device() const {
            return data_kernel;
        }
        inline GPUData_kernel<FPTYPE> device() {
            return data_kernel;
        }
        
        //pointers to the data
        inline const FPTYPE * getData_gpu() const {
            return data_kernel.data;
        }
        inline const FPTYPE * getData_gpu_alt() const {
            return reinterpret_cast<FPTYPE *>(data_gpu.ptr);
        }
        inline FPTYPE * getData_gpu() {
            return data_kernel.data;
        }
        inline FPTYPE * getData_host() {
            return data_host.data;
        }
        
        //info about allocated
        inline bool isPageLocked() const {
            return page_locked && isOnHost();
        }
        inline bool isOnHost() const {
            return allocated_host;
        }
        inline bool isOnGPU() const {
            return allocated_gpu;
        }
        inline int getDevice() const {
            return devNum;
        }
        inline bool isAllocated() const {
            return allocated_gpu || allocated_host;
        }
        
        //lead dimensions of matrix (for pitched memory), across the y dimension (or columns because of BLAS/MATLAB style memory)
        inline size_t getLD_gpu() const {
            return data_kernel.ld;
        }
        inline size_t getLD_gpu_bytes() const {
            return getLD_gpu() * sizeof(FPTYPE);
        }
        inline size_t getLD_host_bytes() const {
            return getLD_host() * sizeof(FPTYPE);
        }
        inline size_t getLD_host() const {
            return data_host.ld;
        }
        
        //incremement (in number of elements, not bytes) across the z dimension
        inline size_t getInc_gpu() const {
            return data_kernel.inc;
        }
        inline size_t getInc_host() const {
            return data_host.inc;
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
        
        inline bool isEqualSize(const GPUData<FPTYPE> * A) const {
            return A->getSize(0) == getSize(0) && A->getSize(1) == getSize(1) && A->getSize(2) == getSize(2);
        }
        inline bool isEqualSize(const GLData<FPTYPE> * A) const {
            return A->getSize(0) == getSize(0) && A->getSize(1) == getSize(1) && A->getSize(2) == getSize(2);
        }
        
        //size of each dimension
        inline size_t getSize(int dim) const {
            switch(dim) {
                case 0:
                    return data_kernel.x;
                case 1:
                    return data_kernel.y;
                case 2:
                    return data_kernel.z;
                default:
                    return 0;
            }  
        }
        inline dim3 getSize() const {
            return make_dim3(data_kernel.x, data_kernel.y, data_kernel.z);
        }
        
        inline size_t getSize_max(int dim) const {
            switch(dim) {
                case 0:
                    return data_size.x;
                case 1:
                    return data_size.y;
                case 2:
                    return data_size.z;
                default:
                    return 0;
            }  
        }

        inline void printInfo(std::ostringstream & output_stream, const char * name = "GPUData") {
            output_stream << "Info: " << name << "\n";
            output_stream << "\t" << "size = (" << getSize(0) << ", " << getSize(1) << ", " << getSize(2) << ")\n";
            output_stream << "\t" << "ld_gpu = " << getLD_gpu() << ", inc_gpu = " << getInc_gpu() << "\n";
            output_stream << "\t" << "is_stacked_gpu = " << is_stacked_gpu << ", " << " page locked = " << isPageLocked() << "\n";
        }
        
        //assigns a single value to all elements (HOST MEMORY ONLY!)
        inline void assign(FPTYPE val) {
            if(isOnHost()) {
                for(int xx = 0; xx < getSize(0); xx++) {
                    for(int yy = 0; yy < getSize(1); yy++) {
                        for(int zz = 0; zz < getSize(2); zz++) {
                            data_host(xx, yy, zz) = val;
                        }
                    }
                }
            }
        }
        
        //op_A(A) * op_B(B) -> C,  A = this
        cublasStatus_t GEMM(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha = 1, const FPTYPE beta = 0, GPUData<FPTYPE> * BUFFER = NULL, int * multType = NULL);
        cublasStatus_t GEMVs(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha = 1, const FPTYPE beta = 0);
        
        //resizes current data (within pre-allocated bounds - doesn't change memory size, just dims for computations)
        cudaError_t resize(const cudaStream_t stream, int x = -1, int y = -1, int z = -1);
        
        //copy a GPU array into current GPU array -> must be same size!
        // if allowSmaller, source can be smaller than the current GPU array, otherwise must be the same size to not return error
        cudaError_t copyTo(const cudaStream_t stream, const GPUData<FPTYPE> * source, bool allowSmaller = false, const cudaPos copyPos_dest = make_cudaPos(0, 0, 0));
        cudaError_t copyTo(const cudaStream_t stream, const GLData<FPTYPE>  * source, bool allowSmaller = false, const cudaPos copyPos_dest = make_cudaPos(0, 0, 0));
        
        //copy page-locked memory to GPU (asynchronous!)
        cudaError_t copyHostToGPU(const cudaStream_t stream);
        
        //copy GPU memory to page-locked host (asynchronous!)
        cudaError_t copyGPUToHost(const cudaStream_t stream);
        
        //assign values to HOST indices
        const FPTYPE & operator[](size_t index) const {
            if(isOnHost()) {
                return data_host[index];
            }
            else {
                return data_host[0];
            }
        }
        FPTYPE & operator[](size_t index)  {
            if(isOnHost()) {
                return data_host[index];
            }
            else {
                return data_host[0];
            }
        }
        
        const FPTYPE & operator()(size_t xx, size_t yy, size_t zz = 0) const {
            if(isOnHost()) {
                return data_host(xx, yy, zz);
            }
            else {
                return data_host[0];
            }
        }
        FPTYPE & operator()(size_t xx, size_t yy, size_t zz = 0) {
            if(isOnHost()) {
                return data_host(xx, yy, zz);
            }
            else {
                return data_host[0];
            }
        }
        
        // assembles an array_kernel
        static GPUData_array_kernel<FPTYPE, MAX_DIM_D> assembleKernels(const std::vector<GPUData<FPTYPE> *> data) {
            GPUData_array_kernel<FPTYPE, MAX_DIM_D> a;
            a.N = data.size();
            for(unsigned int ii = 0; ii < a.N; ii++) {
                a[ii] = data[ii]->device();
            }
            return a;
        }
        
};

 

 
//base class with some convenient functions    
class GPUGL_base  {
    protected:
        std::ostringstream output_stream;
        std::shared_ptr<GPUGL_msg> msg;
        
        unsigned int dev; //the device number to use
        
        //checks error messages for CUDA rt errors, CUBLAS, and CUSPARSE
        // returns true if no error
        inline bool checkCudaErrors(const cudaError_t ce, const char * msg_str = "CUDA error", const bool printOnly = false) {
            if(ce == cudaSuccess) {
                return true;
            }
            output_stream << msg_str << " - " << cudaGetErrorString(ce) << std::endl;
            if(printOnly) {
                msg->printMsgTxt(output_stream);
            }
            else {
                msg->callErrMsgTxt(output_stream);
            }
            return false;
        }
        
        inline bool checkCudaErrors(const char * msg_str = "CUDA error", const bool printOnly = false) {
            return checkCudaErrors(cudaGetLastError(), msg_str, printOnly);
        }

        inline bool checkCudaErrors(const cublasStatus_t ce, const char * msg_str = "CUBLAS error", const bool printOnly = false) {
            if(ce == CUBLAS_STATUS_SUCCESS) {
                return true;
            }
            output_stream << msg_str << " - cublasError " << cublasGetErrorString(ce) << std::endl;
            if(printOnly) {
                msg->printMsgTxt(output_stream);
            }
            else {
                msg->callErrMsgTxt(output_stream);
            }
            return false;
        }
        inline bool checkCudaErrors(const cusparseStatus_t ce, const char * msg_str = "CUSPARSE error", const bool printOnly = false) {
            if(ce == CUSPARSE_STATUS_SUCCESS) {
                return true;
            }
            output_stream << msg_str << " - cusparseError " << cusparseGetErrorString(ce) << std::endl;
            if(printOnly) {
                msg->printMsgTxt(output_stream);
            }
            else {
                msg->callErrMsgTxt(output_stream);
            }
            return false;
        }
        
        //Frees CUDA allocated pointers and checks for errors
        template <typename T>
        inline bool cudaSafeFree(GPUData<T> * & a, const char * msg_str = "cudaFree error") {
            if(a) {
                bool result = checkCudaErrors(a->deallocate(), msg_str, true);
                delete a;
                a = NULL;
                return result;
            }
            else {
                return true;
            }
        }
        template <typename T>
        inline bool cudaSafeFreeVector(std::vector<GPUData<T> *> & arr, const char * msg_str = "cudaFree vector error") {
            bool allFreed = true;
            for (auto& it : arr) { 
                allFreed = allFreed && cudaSafeFree(it, msg_str);
            }
            return allFreed;
        }
        
        //Frees CUDA allocated pointers and checks for errors
        template <typename T>
        inline bool cudaSafeFreePtr(T * & a, const char * msg_str = "cudaFree error") {
            if(a) {
                bool result = checkCudaErrors(cudaFree(a), msg_str, true);
                a = NULL;
                return result;
            }
            else {
                return true;
            }
        }

        //frees all CUDA pointers in a vector
        template <typename T>
        inline bool cudaSafeFreePtrVector(std::vector<T *> & arr, const char * msg_str = "cudaFree vector error") {
            bool allFreed = true;
            for (auto& it : arr) { 
                allFreed = allFreed && cudaSafeFreePtr(it, msg_str);
            }
            return allFreed;
        }
        
        inline bool switchToDevice(const bool printOnly = false) {
            return checkCudaErrors(cudaSetDevice(dev), "error switching to device", printOnly);
        }
        bool checkDeviceComputeCapability(const bool printOnly = false) {
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev), "Error getting CUDA device properties (checkDeviceComputeCapability)", printOnly);
            if(610 <= deviceProp.major*100 + deviceProp.minor*10) {
                return true;
            }
            else {
                checkCudaErrors(cudaErrorInvalidDevice, "CUDA compute capability error (requires 6.1 or greater)", printOnly);
                return false;
            }
        }
        
        
    public:
        inline int getDev() const {
            return dev;
        }
};



}; // end namespace
#endif