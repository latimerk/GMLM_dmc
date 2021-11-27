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
    if(include_host == GPUData_HOST_PAGELOCKED && ce == cudaSuccess) {
        if(size() > 0) {
            ce = cudaMallocHost(reinterpret_cast<void**>(&(data_host.data)), size() * sizeof(FPTYPE));
        }   
        else {
            data_host.data = NULL;
        }
        page_locked = true;
        allocated_host = true;
    }
    else if(include_host == GPUData_HOST_STANDARD && ce == cudaSuccess) {
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
        if(page_locked_memory) {
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
        if(page_locked_memory) {
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

/* kernel for a quick MM operation X*B where X is tall * skinny and B is small
*     Trying to get a speedup from CUBLAS in regions where its slow.
*       TODO: replace with TSM2L lib?
*/
template <class FPTYPE>
__global__ void kernel_MM_quick(GPUData_kernel<FPTYPE> XF, const GPUData_kernel<FPTYPE> X, const GPUData_kernel<FPTYPE> F, const FPTYPE alpha, const FPTYPE beta, const cublasOperation_t op_A, const cublasOperation_t op_B)   {
    int rr_start = blockIdx.y * blockDim.y + threadIdx.y;
    size_t row   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t A     = blockIdx.z * blockDim.z + threadIdx.z;
    if(row < XF.x && A < X.z) {
        FPTYPE ll;
        FPTYPE t, y, c;
        for(int rr = rr_start; rr < XF.y; rr+= blockDim.y * gridDim.y) {
            ll = 0;
            c = 0;
            if(op_A == CUBLAS_OP_N && op_B == CUBLAS_OP_N) {
                for(int tt = 0; tt < F.x; tt++) {
                   // ll += X(row, tt, A) * F(tt, rr, A);
                    y  = X(row, tt, A) * F(tt, rr, A) - c;
                    t = ll + y;
                    c = (t - ll) - y;
                    ll = t;
                }
            }
            else if(op_A == CUBLAS_OP_N) {
                for(int tt = 0; tt < F.y; tt++) {
                    //ll += X(row, tt, A) * F(rr, tt, A);
                    y = X(row, tt, A) * F(rr, tt, A) - c;
                    t = ll + y;
                    c = (t - ll) - y;
                    ll = t;
                }
            }
            else if(op_B == CUBLAS_OP_N) {
                for(int tt = 0; tt < F.x; tt++) {
                    //ll += X(tt, row, A) * F(tt, rr, A);
                    y  = X(tt, row, A) * F(tt, rr, A) - c;
                    t = ll + y;
                    c = (t - ll) - y;
                    ll = t;
                }
            }
            else {
                for(int tt = 0; tt < F.y; tt++) {
                    //ll += X(tt, row, A) * F(rr, tt, A);
                    y  = X(tt, row, A) * F(rr, tt, A) - c;
                    t = ll + y;
                    c = (t - ll) - y;
                    ll = t;
                }
            }
            if(beta == 0) {
                XF(row, rr, A) = alpha*ll;
            }
            else {
                XF(row, rr, A) = alpha*ll + beta*XF(row, rr, A);
            }
        }
    }
}



template <class FPTYPE>
cublasStatus_t GPUData<FPTYPE>::GEMM(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha, const FPTYPE beta,  GPUData<FPTYPE> * BUFFER) {
    
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

    if(cols_C != cols_op_B || rows_C != rows_op_A || cols_op_A != rows_op_B
            || (depth_A > 1 && depth_A != depth)
            || (depth_B > 1 && depth_B != depth)
            || (depth_C > 1 && depth_C != depth)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    //for tall skinny op_A
    if(rows_op_A / cols_op_A >= 4 && cols_op_A <= 2048 && cols_op_B > 1 && cols_op_B <= 256) {
        if( op_A == CUBLAS_OP_N && op_B == CUBLAS_OP_N) {
            cudaStream_t stream;
            ce = cublasGetStream(handle, &stream);

            if(ce == CUBLAS_STATUS_SUCCESS) {
                ce = launchKernelTsm2<FPTYPE>(stream, this, B,  C, alpha, beta);
            }
        }
        else {
            //for smaller dim_T_c, call my own makeshift GEMM that's somehow faster for the typical sized problem
            // I should probably use a more efficient algorithm for this? (I don't think this case is important anymore - above TSM2 method is called in typical GMLM problems)
            dim3 block_size;
            dim3 grid_size;
            if(cols_op_B > 8) { 
                block_size.y = 8;
            }
            else if(cols_op_B >= 4) { 
                block_size.y = 4;
            }
            block_size.x = 1024/block_size.y;
            grid_size.x  = getSize(0) / block_size.x + ( (getSize(0) % block_size.x == 0) ? 0 : 1);
            grid_size.y  = 1;
            block_size.z = 1;
            grid_size.z  = getSize(2);

            cudaStream_t stream;
            ce = cublasGetStream(handle, &stream);

            if(ce == CUBLAS_STATUS_SUCCESS) {
                kernel_MM_quick<<<grid_size, block_size, 0, stream>>>(C->device(), device(), B->device(), alpha, beta, op_A, op_B);
                if(cudaSuccess != cudaGetLastError()) {
                    ce = CUBLAS_STATUS_INVALID_VALUE;
                }
                else {
                    ce = CUBLAS_STATUS_SUCCESS;
                }
            }
        }
    }
    else if(cols_op_B == 1) {
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
    else if(cols_op_A / rows_op_A >= 4 && cols_op_A >= 2048  && cols_op_B <= 256) {
        if(op_A == CUBLAS_OP_T && op_B == CUBLAS_OP_N && BUFFER != NULL) {
            // special case using a (somewhat) optimized kernel - much faster than just running cublasGEMM (and sometimes faster than the GEMVs below)
            cudaStream_t stream;
            ce = cublasGetStream(handle, &stream);

            if(ce == CUBLAS_STATUS_SUCCESS) {
                ce = launchKerneltsTmts<FPTYPE>(stream, this, B,  C, BUFFER, alpha, beta, depth);
            }
        }
        else {
            //A'*B for tall, skinny A&B is slow with GEMM, somehow faster with multiple GEMV calls - perverse but it's a bit speedup
            // - I haven't found available code to implement better GEMMs for these irregular matrices (although a couple papers exist). Is it worth trying to implement those kernels myself?
            size_t op_B_ld     = (op_B == CUBLAS_OP_N) ? B->getLD_gpu() : static_cast<size_t>(1);
            size_t op_B_stride = (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->getLD_gpu();
                    
            for(int dd = 0; dd < depth && ce == CUBLAS_STATUS_SUCCESS; dd++) {
                for(int rr = 0; rr < C->getSize(1) && ce == CUBLAS_STATUS_SUCCESS; rr++) {
                    ce = cublasGEMV(handle, op_A,
                                rows_A, cols_A,
                                &alpha,
                                getData_gpu() + dd*getInc_gpu(), getLD_gpu(),
                                B->getData_gpu() + rr*op_B_ld + dd*B->getInc_gpu(), op_B_stride,
                                &beta,
                                C->getData_gpu() + rr*C->getLD_gpu() + dd*C->getInc_gpu(), 1);
                }
            }
        }
    }
    else if(depth > 1) {
        //for largish size of C, call GEMM (single - run below)
        ce = cublasGEMMStridedBatched(handle,
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
                                  depth);
    }
    else {
        ce = cublasGEMM(handle,
                          op_A,
                          op_B,
                          rows_op_A, cols_op_B, cols_op_A,
                          &alpha,
                          getData_gpu(), getLD_gpu(),
                          B->getData_gpu(), B->getLD_gpu(),
                          &beta,
                          C->getData_gpu(), C->getLD_gpu());
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



/* 
 * Tall-and-skinny x small regularish matrix multiplication
 * C = A * B;
 *  TSM2 algorithm. Refs:
 *     [1] Cody Rivera, Jieyang Chen, Nan Xiong, Shuaiwen Leon Song, and Dingwen Tao. 
 *         "TSM2X: High-Performance Tall-and-Skinny Matrix-MatrixMultiplication on GPUs." 2020. arXiv:2002.03258 [cs.DC].
 *     [2] Jieyang Chen, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming Ouyang, 
 *         Kai Zhao, Nathan DeBardeleben, Qiang Guan, and Zizhong Chen.
 *        "TSM2: optimizing tall-and-skinny matrix-matrix multiplication on GPUs."
 *        In Proceedings of the ACM International Conference on Supercomputing (ICS), pp. 106-116. ACM, 2019. https://doi.org/10.1145/3330345.3330355
 * 
 * modified from https://github.com/codyjrivera/tsm2x-imp/ (retrieved Nov 26, 2021 under MIT license)
 *     files: src/kernel_tsm2.cuh
 *            src/v100/kernels_select.cuh
 */
template <typename FPTYPE, int blk_NR, int blk_NC_B, int blk_NC_A>
__global__ void kernelTsm2(const GPUData_kernel<FPTYPE> A, const GPUData_kernel<FPTYPE> B,  GPUData_kernel<FPTYPE> C, const FPTYPE alpha, const FPTYPE beta) {
    // Names mostly follow the paper's
    __shared__ FPTYPE currB[blk_NR * blk_NC_B];

    FPTYPE currA[blk_NC_A];
    FPTYPE nextA[blk_NC_A];
    FPTYPE nextB[blk_NC_B];
    FPTYPE currC[blk_NC_B];

    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread;

    unsigned int depth = blockIdx.y;
    if(depth < A.z) {

        // This implementation can respond to arbitrary input

        // We cannot rule out a thread's participation based on
        // whether it corresponds to a row in Matrix A, so we
        // introduce threadBase.
        for (; threadBase < A.x; threadBase += blockDim.x * gridDim.x) {
            thread = threadBase + tid;
            for (int p = 0; p < B.y; p += blk_NC_B) {
                // Load loops have extra conditionals to ensure
                // they do not make bad memory accesses

                // Loads first tile of output registers and A
                if (thread < A.x) {
                    #pragma unroll
                    for (int i = 0; i < blk_NC_B; ++i) {
                        if (p + i < B.y) {
                            currC[i] = (beta == 0) ? 0 : beta * C(thread, p + i, depth);
                        }
                    }
                    // Loads currA
                    #pragma unroll
                    for (int i = 0; i < blk_NC_A; ++i) {
                        if (i < A.y) {
                            currA[i] = A(thread, i , depth);
                        }
                    }
                }
                // Loads tile of B
                if (tid < A.y) {
                    #pragma unroll
                    for (int i = 0; i < blk_NC_B; ++i) {
                        if (p + i < B.y) {
                            currB[tid + (i * blk_NR)] = alpha * B(tid, p + i, depth);
                        }
                    }
                }

                // Outer product loop
                for (int j = 0; j < A.y; j += blk_NR) {
                    __syncthreads();
                    // Loads next tile of B
                    if (j + blk_NR + tid < A.y) {
                        #pragma unroll
                        for (int i = 0; i < blk_NC_B; ++i) {
                            if (p + i < B.y) {
                                nextB[i] = alpha * B(j + blk_NR + tid, p + i, depth);
                            }
                        }
                    }

                    const int t3mod = blk_NR % blk_NC_A;

                    // Loop over A's columns
                    for (int l = j; l < j + (blk_NR - t3mod) && l < A.y; l += blk_NC_A) {
                        // Loads next A
                        #pragma unroll
                        for (int i = 0; i < blk_NC_A; ++i) {
                            if (l + blk_NC_A + i < A.y && thread < A.x) {
                                nextA[i] = A(thread, l + blk_NC_A + i, depth);
                            }
                        }

                        // Floating Point Operations (lines 32-34)
                        // Each thread does t2 * t3 mults

                        // Either dispatch guarded or unguarded instructions based
                        // on position in matrix A
                        if (l + blk_NC_A <= A.y) {
                            // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]
                            // exist
                            #pragma unroll
                            for (int a = 0; a < blk_NC_B; ++a) {
                                #pragma unroll
                                for (int b = 0; b < blk_NC_A; ++b) {
                                    currC[a] += currA[b] * currB[(l - j) + b + (a * blk_NR)];
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int a = 0; a < blk_NC_B; ++a) {
                                #pragma unroll
                                for (int b = 0; b < blk_NC_A; ++b) {
                                    if (l + b < A.y) {
                                        currC[a] += currA[b] *
                                                    currB[(l - j) + b + (a * blk_NR)];
                                    }
                                }
                            }
                        }

                        // Stores next A in curr A
                        #pragma unroll
                        for (int i = 0; i < blk_NC_A; ++i) {
                            currA[i] = nextA[i];
                        }
                    }
                    // Accommodates t3 that do not divide t1.
                    #pragma unroll
                    for (int a = 0; a < blk_NC_B; ++a) {
                        #pragma unroll
                        for (int b = 0; b < t3mod; ++b) {
                            if (j + blk_NR - t3mod + b < A.y) {
                                currC[a] +=
                                    currA[b] * currB[(blk_NR - t3mod + b) + (a * blk_NR)];
                            }
                        }
                    }

                    __syncthreads();

                    // Loads currB from each thread's nextB
                    #pragma unroll
                    for (int i = 0; i < blk_NC_B; ++i) {
                        currB[tid + (i * blk_NR)] = nextB[i];
                    }

                    // Loads next currA
                    if (t3mod != 0) {
                        #pragma unroll
                        for (int i = 0; i < blk_NC_A; ++i) {
                            if (j + blk_NR + i < A.y && thread < A.x) {
                                currA[i] = A(thread, j + blk_NR + i, depth);
                            }
                        }
                    }
                }
                // Stores C
                if (thread < A.x) {
                    #pragma unroll
                    for (int i = 0; i < blk_NC_B; ++i) {
                        if (p + i < B.y) {
                            C(thread, p + i, depth) = currC[i];
                        }
                    }
                }
            }
        }
    }
}


template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<int> * A, const GPUData<int> * B,  GPUData<int> * C, const int alpha, const int beta) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<char> * A, const GPUData<char> * B,  GPUData<char> * C, const char alpha, const char beta) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<bool> * A, const GPUData<bool> * B,  GPUData<bool> * C, const bool alpha, const bool beta) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<unsigned int> * A, const GPUData<unsigned int> * B,  GPUData<unsigned int> * C, const unsigned int alpha, const unsigned int beta) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<size_t> * A, const GPUData<size_t> * B,  GPUData<size_t> * C, const size_t alpha, const size_t beta) {
    return CUBLAS_STATUS_INVALID_VALUE;
}

template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<float> * A, const GPUData<float> * B,  GPUData<float> * C, const float alpha, const float beta) {
    const size_t m = A->getSize(0);
    const size_t k = A->getSize(1);
    const size_t n = B->getSize(1);

    int blocks = (m / FLOAT_T1) + 1;
    if(blocks > 65536) {
        blocks = 65536;
        //return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    dim3 grid_size;
    dim3 block_size;
    block_size.x = FLOAT_T1;
    block_size.y  = 1;
    grid_size.x = blocks;
    grid_size.y = A->getSize(2);

    if (n <= 2) {
        kernelTsm2<float, FLOAT_T1, 2, 32>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    else if (n <= 4) {
        kernelTsm2<float, FLOAT_T1, 4, 32>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    else if (n <= 6) {
        kernelTsm2<float, FLOAT_T1, 6, 32>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    else if (n <= 8) {
        kernelTsm2<float, FLOAT_T1, 8, 32>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    else {
        kernelTsm2<float, FLOAT_T1, 16, 32>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    if(cudaSuccess != cudaGetLastError()) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    else {
        return CUBLAS_STATUS_SUCCESS;
    }
}

template <>
cublasStatus_t launchKernelTsm2(cudaStream_t stream, const GPUData<double> * A, const GPUData<double> * B,  GPUData<double> * C, const double alpha, const double beta) {
    const size_t m = A->getSize(0);
    const size_t k = A->getSize(1);
    const size_t n = B->getSize(1);
    int blocks = (m / DOUBLE_T1) + 1;
    if(blocks > 65536) {
        blocks = 65536;
        //return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    dim3 grid_size;
    dim3 block_size;
    block_size.x = DOUBLE_T1;
    block_size.y  = 1;
    grid_size.x = blocks;
    grid_size.y = A->getSize(2);
    
    if (n <= 2) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 2, 16>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
        else {
            kernelTsm2<double, DOUBLE_T1, 2, 12>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
    }
    else if (n <= 4) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 4, 16>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
        else {
            kernelTsm2<double, DOUBLE_T1, 4, 12>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
    } else if (n <= 6) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 6, 16>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
        else {
            kernelTsm2<double, DOUBLE_T1, 6, 12>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
    }
    else if (n <= 8) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 8, 16>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
        else {
            kernelTsm2<double, DOUBLE_T1, 8, 12>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
    }
    else if (n <= 16) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 16, 16>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
        else {
            kernelTsm2<double, DOUBLE_T1, 16, 12>
                <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
        }
    }
    else {
        kernelTsm2<double, DOUBLE_T1, 32, 12>
            <<<grid_size, block_size, 0, stream>>>(A->device(), B->device(), C->device(), alpha, beta);
    }
    if(cudaSuccess != cudaGetLastError()) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    else {
        return CUBLAS_STATUS_SUCCESS;
    }
}


template <typename FPTYPE>
__global__ void kernel_reduce_tsTmts(GPUData_kernel<FPTYPE> C, const GPUData_kernel<FPTYPE> C_buf, const FPTYPE alpha, const FPTYPE beta, const unsigned int depth) {
    
    const size_t row_0 = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t col_0 = threadIdx.y + blockIdx.y * blockDim.y;
    
    FPTYPE ll, c, y, t;
    if(depth < C.z) {
        for(int row = row_0; row < C.x; row += blockDim.x * gridDim.x) {
            for(int col = col_0; col < C.y; col += blockDim.y * gridDim.y) {
                ll = 0;
                c = 0;

                for(int tt = 0; tt < C_buf.z; tt++) {
                    y  = C_buf(row, col, tt) - c;
                    t = ll + y;
                    c = (t - ll) - y;
                    ll = t;
                }
                if(beta == 0) {
                    C(row, col, depth) = alpha*ll;
                }
                else {
                    C(row, col, depth) = beta*C(row, col)  + alpha*ll;
                }
            }
        }
    }
}


template <typename FPTYPE, int blk_NR, int blk_NC_B>
__global__ void kernel_tsTmts(const GPUData_kernel<FPTYPE> A, const GPUData_kernel<FPTYPE> B,  GPUData_kernel<FPTYPE> C, const FPTYPE alpha, const FPTYPE beta, const unsigned int depth) {
    // Names mostly follow the paper's
    __shared__ FPTYPE currB[blk_NR * blk_NC_B];

    FPTYPE currA[blk_NR];

    const int tid = threadIdx.x;

    if(depth < A.z) {

        //zero out answer
        for (int pp = 0; pp < B.y; pp += blk_NC_B) {
            for(size_t aa = 0; aa < A.y; aa += blockDim.x) {
                #pragma unroll
                for (int qq = 0; qq < blk_NC_B; qq++) {
                    if(tid + aa < A.y  && qq + pp < B.y) {
                        C(tid + aa, qq + pp, blockIdx.x) = 0;
                    }
                }
            }
        }
        __syncthreads();

        for (size_t row_0 = blk_NR * (blockIdx.x); row_0 < A.x; row_0 += blk_NR * (gridDim.x) ) {

            for (int pp = 0; pp < B.y; pp += blk_NC_B) {
                  
                if(tid + pp < B.y && tid < blk_NC_B) {
                    #pragma unroll
                    for(size_t row = 0; row < blk_NR; row++) {
                        if(row + row_0 < B.x) {
                            currB[row + tid*blk_NR] = B(row + row_0, tid + pp, depth);
                        }
                    }
                }
                
                __syncthreads();

                for(size_t aa = 0; aa < A.y; aa += blockDim.x) {
                    //gets first col of A
                    if(tid + aa < A.y) {
                        #pragma unroll
                        for(size_t row = 0; row < blk_NR; row++) {
                            if(row + row_0 < A.x) {
                                currA[row] = A(row + row_0, tid + aa, depth);
                            }
                        }
                    }
                    __syncthreads();

                    if(tid + aa < A.y) {
                        for (int qq = 0; qq < blk_NC_B && qq + pp < B.y; qq++) {
                            FPTYPE currC = 0;
                            #pragma unroll
                            for(size_t row = 0; row < blk_NR; row++) {
                                if(row + row_0 < A.x) {
                                    currC += currA[row] * currB[row + qq*blk_NR];
                                }
                            }
                            C(tid + aa, qq + pp, blockIdx.x) += currC;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

//NOTE: the launch sizes in launchKerneltsTmts are not fully optimized (and are set the same for both float and double)
template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream, const GPUData<float> * A, const GPUData<float> * B,  GPUData<float> * C, GPUData<float> * buffer, const float alpha, const float beta, const unsigned int depth) {
    if(buffer == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;        
    }
    GPUData_kernel<float> C_buf_k_0 = buffer->device();

    size_t rows_A = A->getSize(0);
    size_t cols_A = A->getSize(1);
    size_t cols_B = B->getSize(1);

    size_t numBlocks = min(NRS_MAX_BLOCKS, rows_A / NRS_FLOAT + ((rows_A % NRS_FLOAT == 0) ? 0 : 1));
    size_t ld_buf    = cols_A + (cols_A % 8);

    GPUData_kernel<float> C_buf_k = buffer->device();
    C_buf_k.x    = cols_A;
    C_buf_k.y    = cols_B;
    C_buf_k.z    = numBlocks;
    C_buf_k.ld   = ld_buf;
    C_buf_k.inc  = C_buf_k.ld * cols_B;

    if(C_buf_k.inc * C_buf_k.z > C_buf_k_0.ld * C_buf_k_0.y * C_buf_k_0.z ) {
        return CUBLAS_STATUS_INVALID_VALUE;        
    }

    dim3 grid_size;
    dim3 block_size;
    block_size.x = NRS_FLOAT;
    block_size.y  = 1;
    grid_size.x = numBlocks;
    grid_size.y = 1;
    
    dim3 grid_size2;
    dim3 block_size2;

    block_size2.x = 128;
    block_size2.y = 4;
    grid_size2.x  = cols_A / block_size2.x + ((cols_A % block_size2.x  == 0) ? 0 : 1);
    grid_size2.y  = cols_B / block_size2.y + ((cols_B % block_size2.y  == 0) ? 0 : 1);

    for(unsigned int depth_c = 0; depth_c < depth; depth_c++) {
        if(B->getSize(1) >= 16) {
            kernel_tsTmts<float, NRS_FLOAT, 16><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<float><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 8) {
            kernel_tsTmts<float, NRS_FLOAT, 8><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<float><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 4) {
            kernel_tsTmts<float, NRS_FLOAT, 4><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<float><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 2) {
            kernel_tsTmts<float, NRS_FLOAT, 2><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<float><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else {
            kernel_tsTmts<float, NRS_FLOAT, 1><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<float><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
    }
    if(cudaSuccess != cudaGetLastError()) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    else {
        return CUBLAS_STATUS_SUCCESS;
    }
}

template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream, const GPUData<double> * A, const GPUData<double> * B,  GPUData<double> * C, GPUData<double> * buffer, const double alpha, const double beta, const unsigned int depth) {
    if(buffer == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;        
    }
    GPUData_kernel<double> C_buf_k_0 = buffer->device();

    size_t rows_A = A->getSize(0);
    size_t cols_A = A->getSize(1);
    size_t cols_B = B->getSize(1);

    size_t numBlocks = min(NRS_MAX_BLOCKS, rows_A / NRS_DOUBLE + ((rows_A % NRS_DOUBLE == 0) ? 0 : 1));
    size_t ld_buf    = cols_A + (cols_A % 8);

    GPUData_kernel<double> C_buf_k = buffer->device();
    C_buf_k.x    = cols_A;
    C_buf_k.y    = cols_B;
    C_buf_k.z    = numBlocks;
    C_buf_k.ld   = ld_buf;
    C_buf_k.inc  = C_buf_k.ld * cols_B;

    if(C_buf_k.inc * C_buf_k.z > C_buf_k_0.ld * C_buf_k_0.y * C_buf_k_0.z ) {
        return CUBLAS_STATUS_INVALID_VALUE;        
    }

    dim3 grid_size;
    dim3 block_size;
    block_size.x = NRS_DOUBLE;
    block_size.y  = 1;
    grid_size.x = numBlocks;
    grid_size.y = 1;
    
    dim3 grid_size2;
    dim3 block_size2;

    block_size2.x = 128;
    block_size2.y = 4;
    grid_size2.x  = cols_A / block_size2.x + ((cols_A % block_size2.x  == 0) ? 0 : 1);
    grid_size2.y  = cols_B / block_size2.y + ((cols_B % block_size2.y  == 0) ? 0 : 1);

    for(unsigned int depth_c = 0; depth_c < depth; depth_c++) {
        
        //kernel_tsTmts<double, NRS_DOUBLE, 8><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
        //kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);

        if(B->getSize(1) >= 16) {
            kernel_tsTmts<double, NRS_DOUBLE, 16><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 8) {
            kernel_tsTmts<double, NRS_DOUBLE, 8><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 4) {
            kernel_tsTmts<double, NRS_DOUBLE, 4><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else if(B->getSize(1) >= 2) {
            kernel_tsTmts<double, NRS_DOUBLE, 2><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        else {
            kernel_tsTmts<double, NRS_DOUBLE, 1><<<grid_size, block_size, 0, stream>>>(A->device(), B->device(),  C_buf_k,  alpha,  beta, depth_c);
            kernel_reduce_tsTmts<double><<<grid_size2, block_size2, 0, stream>>>(C->device(), C_buf_k, alpha, beta, depth_c);
        }
        if(cudaSuccess != cudaGetLastError()) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream,
    const GPUData<int> * A,
    const GPUData<int> * B, 
          GPUData<int> * C,
          GPUData<int> * buffer,
            const int alpha,
            const int beta, const unsigned int depth) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream,
    const GPUData<char> * A,
    const GPUData<char> * B, 
          GPUData<char> * C,
          GPUData<char> * buffer,
            const char alpha,
            const char beta, const unsigned int depth) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream,
    const GPUData<bool> * A,
    const GPUData<bool> * B, 
          GPUData<bool> * C,
          GPUData<bool> * buffer,
            const bool alpha,
            const bool beta, const unsigned int depth) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream,
    const GPUData<unsigned int> * A,
    const GPUData<unsigned int> * B, 
          GPUData<unsigned int> * C,
          GPUData<unsigned int> * buffer,
            const unsigned int alpha,
            const unsigned int beta, const unsigned int depth) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
template <>
cublasStatus_t launchKerneltsTmts(cudaStream_t stream,
    const GPUData<size_t> * A,
    const GPUData<size_t> * B, 
          GPUData<size_t> * C,
          GPUData<size_t> * buffer,
            const size_t alpha,
            const size_t beta, const unsigned int depth) {
    return CUBLAS_STATUS_INVALID_VALUE;
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