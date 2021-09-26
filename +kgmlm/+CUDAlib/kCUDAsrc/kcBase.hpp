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
enum GPUData_HOST_ALLOCATION {GPUData_HOST_NONE = 0, GPUData_HOST_PAGELOCKED = 1, GPUData_HOST_STANDARD = 2}; 

    //classes for storing matrices (including stacks of matrices)
// base classes for CUDA arrays
//  holds data in up to a 3D array, while managing the pitch sizes and such
template <typename FPTYPE> 
class GPUData_kernel { // holds data on GPU for more compact/clean kernel calls
    public:
        size_t x;
        size_t y;
        size_t z;

        size_t x_s;
        size_t z_s;

        size_t ld;
        size_t inc;

        FPTYPE * data; 
        
        __device__
        const FPTYPE & operator()(const size_t xx, const size_t yy = 0, const size_t zz = 0) const {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        __device__
        FPTYPE & operator()(const size_t xx, const size_t yy = 0, const size_t zz = 0) {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        __device__
        const FPTYPE & operator[](const size_t index) const {
            return data[index];
        }
        __device__
        FPTYPE & operator[](const size_t index) {
            return data[index];
        }
        __device__
        const FPTYPE & get(const size_t xx, const size_t yy = 0, const size_t zz = 0) const {
            size_t index = xx + yy * ld + zz * inc;
            return data[index];
        }
        __device__
        void set(const FPTYPE & val, const size_t xx, const size_t yy = 0, const size_t zz = 0) {
            size_t index = xx + yy * ld + zz * inc;
            data[index] = val;
        }
};
    
template <class FPTYPE> 
class GPUData {
    private:
        cudaPitchedPtr data_gpu;
        
        GPUData_kernel<FPTYPE> * data_kernel = NULL;
        
        FPTYPE * data_host = NULL;
        
        cudaExtent data_size; //for GPU
        cudaExtent data_size_bytes; // data_size_bytes.depth = data_size.depth * sizeof(FPTYPE), convenient for some cuda api calls
        
        cudaExtent data_size_c; // current size (in elements, not bytes) is <= data_size
        
        size_t inc_gpu;
        size_t inc_host;
        
        size_t inc_gpu_bytes;
        size_t inc_host_bytes;
        
        size_t ld_gpu;
        size_t ld_host;
        
        size_t ld_gpu_bytes;
        size_t ld_host_bytes;
        
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
        GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t width, size_t height = 1, size_t depth = 1, bool stacked_depth = false);
        GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, cudaExtent size, bool stacked_depth = false);

        // allocate memory on GPU (and page-locked host if requested), returns cuda status
        cudaError_t allocate_gpu(GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t width, size_t height = 1, size_t depth = 1, bool stacked_depth = false);
        
        // allocate memory on host (is page-locked host if requested, otherwise normal memory), returns cuda status (will only be set if page-locked memory requested)
        cudaError_t allocate_host(bool page_locked_memory, size_t width, size_t height = 1, size_t depth = 1);
        
        // deletes all memory currently allocated, returns cuda status
        cudaError_t deallocate();
        
        // destructor destroys memory - does not check for CUDA errors! Recommend calling deallocate manually
        ~GPUData();
        
        //gets the struct to send matrix to a cuda kernel
        inline const GPUData_kernel<FPTYPE> * device() const {
            return data_kernel;
        }
        inline GPUData_kernel<FPTYPE> * device() {
            return data_kernel;
        }
        
        //pointers to the data
        inline const FPTYPE * getData_gpu() const {
            return reinterpret_cast<FPTYPE*>(data_gpu.ptr) ;
        }
        inline FPTYPE * getData_gpu() {
            return reinterpret_cast<FPTYPE*>(data_gpu.ptr) ;
        }
        inline FPTYPE * getData_host() {
            return data_host;
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
        
        //lead dimensions of matrix (for pitched memory), across the height dimension (or columns because of BLAS/MATLAB style memory)
        inline size_t getLD_gpu() const {
            return ld_gpu;
        }
        inline size_t getLD_gpu_bytes() const {
            return ld_gpu_bytes;
        }
        inline size_t getLD_host_bytes() const {
            return ld_host_bytes;
        }
        inline size_t getLD_host() const {
            return ld_host;
        }
        
        //incremement (in number of elements, not bytes) across the depth dimension
        inline size_t getInc_gpu() const {
            return inc_gpu;
        }
        inline size_t getInc_host() const {
            return inc_host;
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
                    return data_size_c.width;
                case 1:
                    return data_size_c.height;
                case 2:
                    return data_size_c.depth;
                default:
                    return 0;
            }  
        }
        inline cudaExtent getSize() const {
            return data_size_c;
        }
        
        inline size_t getSize_max(int dim) const {
            switch(dim) {
                case 0:
                    return data_size.width;
                case 1:
                    return data_size.height;
                case 2:
                    return data_size.depth;
                default:
                    return 0;
            }  
        }
        
        //assigns a single value to all elements (HOST MEMORY ONLY!)
        inline void assign(FPTYPE val) {
            if(isOnHost()) {
                for(int xx = 0; xx < getSize(0); xx++) {
                    for(int yy = 0; yy < getSize(1); yy++) {
                        for(int zz = 0; zz < getSize(2); zz++) {
                            data_host[xx + yy*ld_host + zz*inc_host] = val;
                        }
                    }
                }
            }
        }
        
        //op_A(A) * op_B(B) -> C,  A = this
        cublasStatus_t GEMM(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha = 1, const FPTYPE beta = 0);
        cublasStatus_t GEMVs(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha = 1, const FPTYPE beta = 0);
        
        //resizes current data (within pre-allocated bounds - doesn't change memory size, just dims for computations)
        cudaError_t resize(const cudaStream_t stream, int width = -1, int height = -1, int depth = -1);
        
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
                size_t index = xx + yy * ld_host + zz * inc_host;
                return data_host[index];
            }
            else {
                return data_host[0];
            }
        }
        FPTYPE & operator()(size_t xx, size_t yy, size_t zz = 0) {
            if(isOnHost()) {
                size_t index = xx + yy * ld_host + zz * inc_host;
                return data_host[index];
            }
            else {
                return data_host[0];
            }
        }
        
};

 //classes for storing matrices (including stacks of matrices)
template <typename FPTYPE> 
class GPUData_array_kernel { // holds data on GPU for more compact/clean kernel calls
    public:
        GPUData_kernel<FPTYPE> ** data = NULL;   
        size_t N;

        __device__ const GPUData_kernel<FPTYPE> & operator[](size_t index) const {
            return (*data[index]);
        }
        __device__  GPUData_kernel<FPTYPE> & operator[](size_t index) {
            return (*data[index]);
        }
        
}; 

//points to a bunch of matrices on the GPU
template <class FPTYPE> 
class GPUData_array {
    private:
        //ptr array on GPU
        GPUData_kernel<FPTYPE> ** data_gpu = NULL;                
        size_t N_elements;
        std::vector<GPUData_kernel<FPTYPE> *> data_host;

        GPUData_array_kernel<FPTYPE> * data_kernel = NULL;
        
        int devNum;
    public:
        // allocate from vector of GPUData
        GPUData_array();
        
        GPUData_array(cudaError_t & ce, std::vector<GPUData<FPTYPE> *> & data, const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg);
        
        // destructor destroys memory - does not check for CUDA errors! Recommend calling deallocate manually
        ~GPUData_array();
        
        //check if setup
        inline bool isAllocated() {
            return data_gpu != NULL;
        }
    
        //get data for device
        inline GPUData_array_kernel<FPTYPE> * device() {
            return data_kernel;
        }
        
        // deallocate everything
        cudaError_t deallocate();
        
        cudaError_t allocate(std::vector<GPUData<FPTYPE> *> & data, const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg);
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
            output_stream << msg_str << " - " << cublasGetErrorString(ce) << std::endl;
            if(printOnly) {
                msg->printMsgTxt(output_stream);
            }
            else {
                msg->callErrMsgTxt(output_stream);
            }
            return true;
        }
        inline bool checkCudaErrors(const cusparseStatus_t ce, const char * msg_str = "CUSPARSE error", const bool printOnly = false) {
            if(ce == CUSPARSE_STATUS_SUCCESS) {
                return false;
            }
            output_stream << msg_str << " - " << cusparseGetErrorString(ce) << std::endl;
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
        inline bool cudaSafeFree(GPUData_array<T> * & a, const char * msg_str = "cudaFree error") {
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
     /*   template <typename T>
        inline bool cudaSafeFree(T * & a, const char * msg_str = "cudaFree error") {
            if(a) {
                bool result = checkCudaErrors(cudaFree(a), msg_str, true);
                a = NULL;
                return result;
            }
            else {
                return true;
            }
        }
        //Frees CUDA allocated host pointers and checks for errors
        template <typename T>
        inline bool cudaSafeFreeHost(T * & a, const char * msg_str = "cudaFreeHost error") {
            if(a) {
                bool result = checkCudaErrors(cudaFreeHost(a), msg_str, true);
                a = NULL;
                return result;
            }
            else {
                return true;
            }
        }
        //frees all CUDA pointers in a vector
        template <typename T>
        inline bool cudaSafeFreeVector(std::vector<T *> & arr, const char * msg_str = "cudaFree vector error") {
            bool allFreed = true;
            for (auto& it : arr) { 
                allFreed = allFreed && cudaSafeFree(it, msg_str);
            }
            return allFreed;
        }
        template <typename T>
        inline bool cudaSafeFreeHostVector(std::vector<T *> & arr, const char * msg_str = "cudaFreeHost vector error") {
            bool allFreed = true;
            for (auto& it : arr) { 
                allFreed = allFreed && cudaSafeFreeHost(it, msg_str);
            }
            return allFreed;
        }*/
        
        inline bool switchToDevice(const bool printOnly = false) {
            return checkCudaErrors(cudaSetDevice(dev), "error switching to device", printOnly);
        }
        
  /*      template <class vecType>
        inline void copyVectorToGPU(vecType * & dest, const std::vector<vecType> host, const cudaStream_t stream = 0, const char * msg_str = "copyVectorToGPU vector error") {
            checkCudaErrors(cudaMemcpyAsync(dest, host.data(), host.size() * sizeof(vecType), cudaMemcpyHostToDevice, stream), msg_str);
        }
        
        template <class vecType>
        inline void allocateAndCopyVectorToGPU(vecType * & dest, const std::vector<vecType> host, const cudaStream_t stream = 0, const char * msg_str = "allocateAndCopyVectorToGPU vector error") {
            checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&(dest)), host.size() * sizeof(vecType)), msg_str);
            copyVectorToGPU(dest, host, stream, msg_str);
        }*/
        
    public:
        inline int getDev() const {
            return dev;
        }
};




        



}; // end namespace
#endif