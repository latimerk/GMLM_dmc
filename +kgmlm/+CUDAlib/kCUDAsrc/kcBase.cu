/*
 * kcBase.cu
 * classes for storing matrices (including stacks of matrices)
 * keeps track of all the GPU stuff
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
#include "kcBase.hpp"

namespace kCUDA {

    
bool GPU_USE_PAGELOCKED_HOST_STORAGE = true; // can enable/disable use of page-locked memory


template <> cudaDataType_t getCudaType<float>() {
return CUDA_R_32F;
}
template <> cudaDataType_t getCudaType<double>() {
    return CUDA_R_64F;
}  

//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================
// here we define the methods for the GPUData class
        

// constructor - sets everything to blank
template <class FPTYPE>
GPUData<FPTYPE>::GPUData() {
    data_size = make_dim3(0, 0, 0);

    data_kernel = GPUData_kernel<FPTYPE>();
    data_host = GPUData_kernel<FPTYPE>();


    allocated_gpu  = false;
    allocated_host = false;
    is_stacked_gpu = false;

    data_gpu.ptr = NULL;
    devNum = -1;
}

    // constructor - allocates GPU memory with given size, returns a CUDA status
template <class FPTYPE>
GPUData<FPTYPE>::GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t x, size_t y, size_t z, bool stacked_depth) : GPUData<FPTYPE>() {
    ce = allocate_gpu(include_host, stream, x, y, z, stacked_depth);
}
template <class FPTYPE>
GPUData<FPTYPE>::GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, const dim3 size, bool stacked_depth) {
    GPUData(ce, include_host, stream, size.x, size.y, size.z, stacked_depth);
}


template <class FPTYPE>
GPUData<FPTYPE>::~GPUData() {
    deallocate();
}

//allocates GPU memory (and page-locked host memory if requested)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::allocate_gpu(GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t x, size_t y, size_t z, bool stacked_depth) {
    cudaError_t ce = cudaSuccess;
    // deallocate any existing memory
    if(isAllocated()) {
        ce = deallocate();
        if(ce != cudaSuccess) {
            return ce;
        }
    }

    //setup dimensions
    data_size       = make_dim3(x, y, z);

    data_gpu.ptr = NULL;
    data_kernel = GPUData_kernel<FPTYPE>();
    data_host   = GPUData_kernel<FPTYPE>();
    allocated_host = false;
    allocated_gpu  = true;
    page_locked    = false;

    data_kernel.x = x;
    data_kernel.y = y;
    data_kernel.z = z;
    data_host.x = x;
    data_host.y = y;
    data_host.z = z;

    //allocate GPU memory
    ce = cudaGetDevice(&devNum);
    
    stacked_depth = stacked_depth || z <= 1;

    if(!stacked_depth) {
        // do not stack depth dimension into a matrix
        data_host.ld = getSize(0);
        data_host.inc = (z <= 1) ? 0 : getLD_host() * y;
        data_kernel.ld  = 0;
        data_kernel.inc = 0;

        size_t total_size = data_host.ld * getSize(1)  * getSize(2);
        if(ce == cudaSuccess) {
            if(size() > 0) {
                cudaExtent data_size_bytes = make_cudaExtent(getSize(0)*sizeof(FPTYPE), getSize(1), getSize(2));
                ce = cudaMalloc3D(&(data_gpu), data_size_bytes);
            }
            else {
                data_gpu = make_cudaPitchedPtr(NULL, 0, 0, y) ;
            }
            data_kernel.data = reinterpret_cast<FPTYPE*>(data_gpu.ptr);
            data_kernel.ld   = data_gpu.pitch / sizeof(FPTYPE);
            data_kernel.inc  = (z <= 1 || size() == 0) ? 0 : getLD_gpu() * y;
        }
        is_stacked_gpu = false;

    }
    else {
        // stack depth dimension into a matrix
        data_host.ld  = getSize(0) * getSize(2);
        data_host.inc = (z <= 1) ? 0 : getSize(0);
        data_kernel.inc = data_host.inc;

        if(ce == cudaSuccess) {
            if(size() > 0) {
                ce = cudaMallocPitch(&(data_gpu.ptr), &(data_gpu.pitch), getSize(0) * getSize(2) * sizeof(FPTYPE), getSize(1)); 
            }
            else {
                data_gpu = make_cudaPitchedPtr(NULL, 0, 0, y) ;
            }
            data_kernel.data = reinterpret_cast<FPTYPE*>(data_gpu.ptr);
            data_kernel.ld   = data_gpu.pitch / sizeof(FPTYPE);
        }
        is_stacked_gpu = true;
    }
        
    //page locked memory if requested
    if(GPU_USE_PAGELOCKED_HOST_STORAGE && include_host == GPUData_HOST_PAGELOCKED && ce == cudaSuccess) {
        if(size() > 0) {
            ce = cudaMallocHost(reinterpret_cast<void**>(&(data_host.data)), size() * sizeof(FPTYPE));
        }   
        else {
            data_host.data = NULL;
        }
        page_locked = true;
        allocated_host = true;
    }
    else if((include_host == GPUData_HOST_PAGELOCKED || include_host == GPUData_HOST_STANDARD) && ce == cudaSuccess) {
        if(size() > 0) {
            data_host.data = new FPTYPE[size()];
        }   
        else {
            data_host.data = NULL;
        }
        page_locked   = false;
        allocated_host = true;
    }
    data_kernel.data = data_kernel.data;
    return ce;
}

//resizes current data (within pre-allocated bounds)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::resize(const cudaStream_t stream, int x, int y, int z) {
    cudaError_t ce = cudaSuccess;
    
    //default values
    x = (x < 0) ? data_kernel.x : x;
    y = (y < 0) ? data_kernel.y : y;
    z = (z < 0) ? data_kernel.z : z;
    
    //if invalid
    if(x > data_size.x || y > data_size.y || z > data_size.z) {
        ce = cudaErrorInvalidValue;
    }

    if(data_kernel.x != x || data_kernel.y != y || data_kernel.z != z) { 
        if(z <= 1) {
            data_host.inc = 0; 
            data_kernel.inc = 0; 
        }
        else if(!is_stacked_gpu) {
            data_kernel.inc = y * data_kernel.ld;
            data_host.inc   = y * data_host.ld;   
        }
        else if(is_stacked_gpu) {
            data_kernel.inc = x;
            data_host.inc   = x;    
        }
        data_kernel.x  = x;
        data_kernel.y  = y;
        data_kernel.z  = z;
        data_host.x  = x;
        data_host.y  = y;
        data_host.z  = z;
    }

    return ce;
}

//allocates GPU memory (is page-locked host memory if requested, otherwise regular host memory)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::allocate_host(bool page_locked_memory, size_t x, size_t y, size_t z) {
    cudaError_t ce;
    // deallocate any existing memory
    if(isAllocated()) {
        ce = deallocate();
        if(ce != cudaSuccess) {
            return ce;
        }
    }

    //setup dimensions
    data_size       = make_dim3(x, y, z);

    //allocate GPU memory
    data_gpu.ptr  = NULL;
    data_kernel = GPUData_kernel<FPTYPE>();
    data_host   = GPUData_kernel<FPTYPE>();
    allocated_host = false;
    allocated_gpu  = false;
    page_locked    = false;
    is_stacked_gpu = false;

    data_kernel.x = x;
    data_kernel.y = y;
    data_kernel.z = z;
    data_host.x = x;
    data_host.y = y;
    data_host.z = z;


    data_kernel.ld  = x;
    data_host.ld    = x;
    if(z > 1) {
        data_kernel.inc = x * y;
        data_host.inc   = x * y;
    }
    else {
        data_kernel.inc = 0;
        data_host.inc   = 0;
    }

    //allocate memory
    if(size() > 0) {
        if(GPU_USE_PAGELOCKED_HOST_STORAGE && page_locked_memory) {
            ce = cudaMallocHost(reinterpret_cast<void**>(&(data_host.data)), size() * sizeof(FPTYPE));
            if(ce == cudaSuccess) {
                ce = cudaGetDevice(&devNum); 
            }
            page_locked = true;
        }
        else {
            ce = cudaSuccess;
            data_host.data = new FPTYPE[size()];
            devNum = -1;
            page_locked = false;
        }
    }
    else {
        if(GPU_USE_PAGELOCKED_HOST_STORAGE && page_locked_memory) {
            data_host.data = NULL;
            page_locked = true;
            ce = cudaGetDevice(&devNum); 
        }
        else {
            data_host.data = NULL;
            devNum = -1;
        }
    }
    allocated_host = true;
    allocated_gpu = false;
    return ce;
}

//deallocates CUDA memory
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::deallocate() {
    cudaError_t ce = cudaSuccess;
    if(isAllocated()) {
        ce = cudaSetDevice(devNum);
        if(data_host.data != NULL && !page_locked && allocated_host) {
            delete[] data_host.data;
        }
        if(ce == cudaSuccess && data_host.data != NULL && page_locked && allocated_host) {
            ce = cudaFreeHost(data_host.data);
        }
        if(ce == cudaSuccess && data_gpu.ptr != NULL && allocated_gpu) {
            ce = cudaFree(data_gpu.ptr);
        }
        data_gpu.ptr = NULL;
        data_kernel = GPUData_kernel<FPTYPE>();
        data_host   = GPUData_kernel<FPTYPE>();
        allocated_gpu = false;
        allocated_host = false;
        page_locked = false;
    }
    return ce;
}

template <typename FPTYPE> 
__global__ void GPUData_3DCopy(GPUData_kernel<FPTYPE> dest, const GPUData_kernel<FPTYPE> source, const cudaPos copyPos_dest) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x < source.x && y < source.y && z < source.z) {
        dest(x + copyPos_dest.x, y + copyPos_dest.y, z + copyPos_dest.z) = source(x, y, z);
    }
}


//copy a GPU array into current GPU array -> must be same size!
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyTo(const cudaStream_t stream, const GPUData<FPTYPE> * source, bool allowSmaller, const cudaPos copyPos_dest) {
    //if both valid
    if(source == NULL || !(source->isAllocated()) || !isAllocated()) {
        return cudaErrorInvalidValue;
    }

    //check matching sizes!
    if((!allowSmaller && getSize(0) != source->getSize(0) + copyPos_dest.x) || (allowSmaller && getSize(0) < source->getSize(0) + copyPos_dest.x)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(1) != source->getSize(1) + copyPos_dest.y) || (allowSmaller && getSize(1) < source->getSize(1) + copyPos_dest.y)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(2) != source->getSize(2) + copyPos_dest.z) || (allowSmaller && getSize(2) < source->getSize(2) + copyPos_dest.z)) {
        return cudaErrorInvalidValue;
    }

    //if source host allocated
    cudaError_t ce = cudaSuccess;
    if(source->isOnHost())  {

        //if dest host allocated
        if(isOnHost()) {
            for(int xx_c = 0; xx_c < source->getSize(0); xx_c++) {
                int xx = xx_c + copyPos_dest.x;
                for(int yy_c = 0; yy_c < source->getSize(1); yy_c++) {
                    int yy = yy_c + copyPos_dest.y;
                    for(int zz_c = 0; zz_c < source->getSize(2); zz_c++) {
                        int zz = zz_c + copyPos_dest.z;
                        data_host(xx, yy, zz) = (*source)(xx_c, yy_c, zz_c);
                    }
                }
            }
            //if dest GPU memory
            if(isOnGPU())  {
                return copyHostToGPU(stream);
            }
        }

        //if dest GPU memory
        else if(isOnGPU()) {
            //set current GPU
            if(size() > 0) {
                ce = cudaSetDevice(devNum);
                if(ce == cudaSuccess) {
                    for(int zz = 0; zz < source->getSize(2) && ce == cudaSuccess; zz++) {
                        ce = cudaMemcpy2DAsync(getData_gpu() + copyPos_dest.x + copyPos_dest.y * getLD_gpu() + ( copyPos_dest.z + zz) * getInc_gpu(),
                             getLD_gpu_bytes(),
                             source->data_host.data + zz * source->getInc_host(),
                             source->getLD_host_bytes(),
                             source->getSize(0) * sizeof(FPTYPE),
                             source->getSize(1),
                             cudaMemcpyHostToDevice, stream);
                    }
                }
            }   
        }
    }
    // else if source is on GPU
    else if(source->isOnGPU()) {
        if(source->devNum == devNum) {
            //set current GPU
            if(size() > 0) {
                ce = cudaSetDevice(devNum);
                if(ce == cudaSuccess) {
                    dim3 block_size;
                    block_size.x = min(static_cast<size_t>(1024), source->getSize(0));
                    block_size.y = 1;
                    block_size.z = 1;
                    dim3 grid_size;
                    grid_size.x = source->getSize(0) / block_size.x + ((source->getSize(0) % block_size.x == 0)? 0:1);
                    grid_size.y = source->getSize(1) / block_size.y + ((source->getSize(1) % block_size.y == 0)? 0:1);
                    grid_size.z = source->getSize(2) / block_size.z + ((source->getSize(2) % block_size.z == 0)? 0:1);
                    GPUData_3DCopy<<<grid_size, block_size,  0, stream>>>(device(), source->device(), copyPos_dest);
                    ce = cudaGetLastError();
                }
            }   
        }
        else {
            return cudaErrorInvalidValue;
            // NOT SUPPORTED YET!
        }
    }
    return ce;
}



//copy a GPU array into current GPU array -> must be same size!
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyTo(const cudaStream_t stream, const GLData<FPTYPE> * source, bool allowSmaller, const cudaPos copyPos_dest) {
    //if both valid
    if(source == NULL || !isAllocated()) {
        return cudaErrorInvalidValue;
    }

    //check matching sizes!
    if((!allowSmaller && getSize(0) != source->getSize(0) + copyPos_dest.x) || (allowSmaller && getSize(0) < source->getSize(0) + copyPos_dest.x)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(1) != source->getSize(1) + copyPos_dest.y) || (allowSmaller && getSize(1) < source->getSize(1) + copyPos_dest.y)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(2) != source->getSize(2) + copyPos_dest.z) || (allowSmaller && getSize(2) < source->getSize(2) + copyPos_dest.z)) {
        return cudaErrorInvalidValue;
    }

    //if source host allocated
    cudaError_t ce = cudaSuccess;

    //if dest host allocated
    if(isOnHost()) {
        for(int xx_c = 0; xx_c < source->getSize(0); xx_c++) {
            int xx = xx_c + copyPos_dest.x;
            for(int yy_c = 0; yy_c < source->getSize(1); yy_c++) {
                int yy = yy_c + copyPos_dest.y;
                for(int zz_c = 0; zz_c < source->getSize(2); zz_c++) {
                    int zz = zz_c + copyPos_dest.z;
                    data_host(xx, yy, zz) = (*source)(xx_c, yy_c, zz_c);
                }
            }
        }
        //if dest GPU memory
        if(isOnGPU())  {
            return copyHostToGPU(stream);
        }
    }

    //if dest GPU memory
    else if(isOnGPU()) {
        //set current GPU
        if(size() > 0) {
            ce = cudaSetDevice(devNum);
            if(ce == cudaSuccess) {
                for(int zz = 0; zz < source->getSize(2) && ce == cudaSuccess; zz++) {
                    ce = cudaMemcpy2DAsync(getData_gpu() + (copyPos_dest.z + zz) * getInc_gpu() + copyPos_dest.y * getLD_gpu() + copyPos_dest.x,
                         getLD_gpu_bytes(),
                         source->getData() + zz * source->getInc(),
                         source->getLD() * sizeof(FPTYPE),
                         source->getSize(0) * sizeof(FPTYPE), 
                         source->getSize(1),
                         cudaMemcpyHostToDevice, stream);
                }
            }
        }   
    }

    return ce;
}

//copies data in GPU memory to host (if both as assigned)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyGPUToHost(const cudaStream_t stream) {
    if(allocated_host && allocated_gpu) {
        if(size() > 0) {
            //set current GPU
            cudaError_t ce = cudaSetDevice(devNum);
            //for each z (this loop is easier to me than setting up that cursed memcpy3d call)
            for(unsigned int zz = 0; zz < getSize(2) && ce == cudaSuccess; zz++) {
                ce = cudaMemcpy2DAsync(getData_host() + zz * getInc_host(),
                     getLD_host_bytes(),
                     getData_gpu() + zz * getInc_gpu(),
                     getLD_gpu_bytes(),
                     getSize(0) * sizeof(FPTYPE), 
                     getSize(1),
                     cudaMemcpyDeviceToHost, stream);
            }
            return ce;
        }
        else {
            return cudaSuccess;
        }
    }
    else {
        return cudaErrorInvalidValue;
    }
}

//copies data in host memory to GPU (if both as assigned)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyHostToGPU(const cudaStream_t stream) {
    if(isOnHost() && isOnGPU()) {
        //set current GPU
        cudaError_t ce = cudaSetDevice(devNum); 
        //for each z
        for(unsigned int zz = 0; zz < getSize(2) && ce == cudaSuccess; zz++) {
            ce = cudaMemcpy2DAsync(getData_gpu() + zz * getInc_gpu(),
                 getLD_gpu_bytes(),
                 getData_host() + zz * getInc_host(),
                 getLD_host_bytes(),
                 getSize(0) * sizeof(FPTYPE), 
                 getSize(1), 
                 cudaMemcpyHostToDevice, stream);
        }
        return ce;
    }
    else {
        return cudaErrorInvalidValue;
    }
}

// Covers a set of different useful GEMM arrangements for GMLMs
template <class FPTYPE>
cublasStatus_t GPUData<FPTYPE>::GEMM(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha, const FPTYPE beta,  GPUData<FPTYPE> * BUFFER, int * multType) {
    cublasStatus_t ce = CUBLAS_STATUS_SUCCESS;

    size_t depth_A = getSize(2);
    size_t depth_B = B->getSize(2);
    size_t depth_C = C->getSize(2);
    size_t rows_A = getSize(0);
    size_t rows_B = B->getSize(0);
    size_t rows_C = C->getSize(0);
    size_t cols_A = getSize(1);
    size_t cols_B = B->getSize(1);
    size_t cols_C = C->getSize(1);

    if(is_stacked_gpu && C->is_stacked_gpu) {
        rows_A = getSize(0) * getSize(2);
        rows_C = C->getSize(0) * C->getSize(2);
        depth_A = 1;
        depth_C = 1;
        if(B->is_stacked_gpu) {
            rows_B = B->getSize(0) * B->getSize(2);
            depth_B = 1;
        }
    }
    size_t depth = max(max(depth_A, depth_B), depth_C);

    size_t rows_op_A = (op_A == CUBLAS_OP_N) ? rows_A : cols_A; 
    size_t cols_op_A = (op_A == CUBLAS_OP_N) ? cols_A : rows_A; 
    size_t rows_op_B = (op_B == CUBLAS_OP_N) ? rows_B : cols_B; 
    size_t cols_op_B = (op_B == CUBLAS_OP_N) ? cols_B : rows_B; 

    if(multType != NULL) {multType[0] = -1;};

    if(cols_C != cols_op_B || rows_C != rows_op_A || cols_op_A != rows_op_B
            || (depth_A > 1 && depth_A != depth)
            || (depth_B > 1 && depth_B != depth)
            || (depth_C > 1 && depth_C != depth)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const size_t MAX_COLS = 16;
    const size_t MAX_COLS_ALGO0 = MAX_COLS * 8;
    const size_t MAX_DEFAULT = 2048;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO0;
    if((cols_op_A <= MAX_DEFAULT && rows_op_A <= MAX_DEFAULT && cols_op_B <= MAX_DEFAULT && rows_op_B <= MAX_DEFAULT) || cols_op_B > MAX_COLS_ALGO0) {
        algo = CUBLAS_GEMM_DEFAULT;
    }
    
    #if __CUDA_ARCH__ >= 700
        if(sizeof(FPTYPE) <= 4) {
            algo = CUBLAS_GEMM_ALGO0_TENSOR_OP; 
            if((cols_op_A <= MAX_DEFAULT && rows_op_A <= MAX_DEFAULT && cols_op_B <= MAX_DEFAULT && rows_op_B <= MAX_DEFAULT) || cols_op_B > MAX_COLS_ALGO0) {
                algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            }
        }
    #endif
    
    //CUBLAS_GEMM_DEFAULT
    //CUBLAS_GEMM_ALGO0 // fastest for many of the typical multiplications I've done in double precision (DEFAULT can be super slow for the big tall-skinny matrices!)
    //CUBLAS_GEMM_DEFAULT_TENSOR_OP
    //CUBLAS_GEMM_ALGO0_TENSOR_OP 

    if(cols_op_B == 1) {
        if(multType != NULL) {multType[0] = 1;};
       //GEMV is sometimes way faster then GEMM (even on the same problem size) - call it if it's all that's needed
        size_t op_B_stride = (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->getLD_gpu();
        for(int dd = 0; dd < depth && ce == CUBLAS_STATUS_SUCCESS; dd++) {
            ce = cublasGEMV(handle, op_A,
                          rows_A, cols_A,
                          &alpha,
                          getData_gpu() + dd*getInc_gpu(), getLD_gpu(),
                          B->getData_gpu() + dd*B->getInc_gpu(), op_B_stride,
                          &beta,
                          C->getData_gpu() + dd*C->getInc_gpu(), 1);
        }
    }
    else if(depth > 1) {
        if(multType != NULL) {multType[0] = 2;}; //{multType[0] = algo;};
        //for largish size of C, call GEMM (single - run below)
        ce = cublasGEMMEXStridedBatched(handle,
                                  op_A,
                                  op_B,
                                  rows_op_A, cols_op_B, cols_op_A,
                                  &alpha,
                                  getData_gpu(), getLD_gpu(),
                                  getInc_gpu(),
                                  B->getData_gpu(), B->getLD_gpu(),
                                  B->getInc_gpu(),
                                  &beta,
                                  C->getData_gpu(), C->getLD_gpu(),
                                  C->getInc_gpu(),
                                  depth, algo);
    }
    else if(depth == 1 && (algo != CUBLAS_GEMM_DEFAULT_TENSOR_OP && algo != CUBLAS_GEMM_DEFAULT) && cols_op_B > MAX_COLS) {
        if(multType != NULL) {multType[0] = 3;}; //{multType[0] = algo;};
        //for largish size of C, call GEMM (single - run below)
        
        for(size_t cc = 0; cc < cols_op_B && ce == CUBLAS_STATUS_SUCCESS; cc += MAX_COLS) {
            size_t cols_op_B_c = (cols_op_B - cc > MAX_COLS) ? MAX_COLS : (cols_op_B - cc);
            ce = cublasGEMMEX(handle,
                            op_A,
                            op_B,
                            rows_op_A, cols_op_B_c, cols_op_A,
                            &alpha,
                            getData_gpu(), getLD_gpu(),
                            B->getData_gpu() + (B->getLD_gpu()) * cc, B->getLD_gpu(),
                            &beta,
                            C->getData_gpu() + (C->getLD_gpu()) * cc, C->getLD_gpu(), algo);
        }
    }
    else {
        if(multType != NULL) {multType[0] = 4;};// {multType[0] = algo;};
        ce = cublasGEMMEX(handle,
                          op_A,
                          op_B,
                          rows_op_A, cols_op_B, cols_op_A,
                          &alpha,
                          getData_gpu(), getLD_gpu(),
                          B->getData_gpu(), B->getLD_gpu(),
                          &beta,
                          C->getData_gpu(), C->getLD_gpu(), algo);
    }
    return ce;
}


template <class FPTYPE>
cublasStatus_t GPUData<FPTYPE>::GEMVs(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha, const FPTYPE beta) {
    
    cublasStatus_t ce = CUBLAS_STATUS_SUCCESS;

    size_t depth = getSize(2);
    size_t rows_A = getSize(0);
    size_t rows_B = B->getSize(0);
    size_t cols_A = getSize(1);
    size_t cols_B = B->getSize(1);

    size_t rows_op_A = (op_A == CUBLAS_OP_N) ? rows_A : cols_A; 
    size_t cols_op_A = (op_A == CUBLAS_OP_N) ? cols_A : rows_A; 
    size_t rows_op_B = (op_B == CUBLAS_OP_N) ? rows_B : cols_B; 
    size_t cols_op_B = (op_B == CUBLAS_OP_N) ? cols_B : rows_B;

    size_t op_B_ld     = (op_B == CUBLAS_OP_N) ? B->getLD_gpu() : static_cast<size_t>(1);
    size_t op_B_stride = (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->getLD_gpu();

    if(C->getSize(1) != cols_op_B || C->getSize(0) != rows_op_A || cols_op_A != rows_op_B || (depth != 1 && cols_op_B != depth)) {
        ce = CUBLAS_STATUS_INVALID_VALUE;
    }

    for(int rr = 0; rr < cols_op_B && ce == CUBLAS_STATUS_SUCCESS; rr++) {
        ce = cublasGEMV(handle, op_A,
                      rows_A, cols_A,
                      &alpha,
                      getData_gpu()    + rr*getInc_gpu(), getLD_gpu(),
                      B->getData_gpu() + rr*op_B_ld, op_B_stride,
                      &beta,
                      C->getData_gpu() + rr*C->getLD_gpu(), 1);
    }
    return ce;
}

//explicitly create classes for single and double precision floating point for library
template class GPUData<float>;
template class GPUData<double>;
template class GPUData<int>;
template class GPUData<char>;
template class GPUData<bool>;
template class GPUData<unsigned int>;
template class GPUData<size_t>;

};//end namespace